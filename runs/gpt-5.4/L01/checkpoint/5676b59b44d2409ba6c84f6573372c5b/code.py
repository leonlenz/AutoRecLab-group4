import os
import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.base import Preprocessor
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


DATASETS = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}
SEEDS = [7, 19, 42, 77, 123]
KS = [1, 5, 10]
ALGO_LABELS = {
    "LensKit.ImplicitMFScorer": "ALS",
    "LensKit.ItemKNNScorer": "ItemKNN",
    "LensKit.PopScorer": "Pop",
}


class DropToImplicit(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    def _process(self, dataset: RecSysDataSet[Any]) -> RecSysDataSet[Any]:
        df = dataset._data.df.copy()
        keep_cols = [c for c in ["user", "item"] if c in df.columns]
        if keep_cols != ["user", "item"]:
            raise ValueError("DropToImplicit requires 'user' and 'item' columns.")
        dataset._data.df = df[["user", "item"]].drop_duplicates().reset_index(drop=True)
        return dataset


class UserOnlyTestHoldout(Preprocessor):
    def __init__(self, test_size: float = 0.2) -> None:
        super().__init__()
        self.test_size = test_size

    def _process(self, dataset: RecSysDataSet[Any]) -> RecSysDataSet[Any]:
        if not (0 < float(self.test_size) < 1):
            raise ValueError("test_size must be in (0, 1)")

        df = dataset._data.df.copy().reset_index(drop=True)
        if not {"user", "item"}.issubset(df.columns):
            raise ValueError("UserOnlyTestHoldout requires 'user' and 'item' columns.")

        rng = np.random.default_rng()
        train_parts = []
        test_parts = []
        for _, udf in df.groupby("user", sort=False):
            if len(udf) < 2:
                continue
            order = rng.permutation(len(udf))
            udf = udf.iloc[order].reset_index(drop=True)
            n_test = max(1, int(np.floor(len(udf) * float(self.test_size))))
            if n_test >= len(udf):
                n_test = len(udf) - 1
            test_parts.append(udf.iloc[:n_test].copy())
            train_parts.append(udf.iloc[n_test:].copy())

        if not train_parts or not test_parts:
            raise RuntimeError("Holdout produced empty train or test split.")

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        val_df = train_df.iloc[0:0].copy()
        from omnirec.data_variants import SplitData
        dataset._data = SplitData(train=train_df, val=val_df, test=test_df)
        return dataset


class PersistSplit(Preprocessor):
    def __init__(self, out_dir: os.PathLike[str] | str) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)

    def _process(self, dataset: RecSysDataSet[Any]) -> RecSysDataSet[Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        dataset._data.train.to_csv(self.out_dir / "train.csv", index=False)
        dataset._data.val.to_csv(self.out_dir / "val.csv", index=False)
        dataset._data.test.to_csv(self.out_dir / "test.csv", index=False)
        return dataset


class AddUnitRating(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    def _process(self, dataset: RecSysDataSet[Any]) -> RecSysDataSet[Any]:
        for split_name in ["train", "val", "test"]:
            split_df = getattr(dataset._data, split_name)
            if "rating" not in split_df.columns:
                split_df = split_df.copy()
                split_df["rating"] = 1
                setattr(dataset._data, split_name, split_df)
        return dataset


class TopKPredictions(Preprocessor):
    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.k = int(k)

    def _process(self, dataset: RecSysDataSet[Any]) -> RecSysDataSet[Any]:
        for split_name in ["train", "val", "test"]:
            split_df = getattr(dataset._data, split_name)
            if "rating" not in split_df.columns:
                split_df = split_df.copy()
                split_df["rating"] = 1
                setattr(dataset._data, split_name, split_df)
        return dataset


def working_paths() -> tuple[Path, Path, Path, Path, Path]:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    root = Path(working_dir)
    outputs = root / "outputs"
    jobs = outputs / "jobs"
    eval_json = outputs / "evaluator_json"
    splits = outputs / "splits"
    outputs.mkdir(parents=True, exist_ok=True)
    jobs.mkdir(parents=True, exist_ok=True)
    eval_json.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    return root, outputs, jobs, eval_json, splits


def build_dataset(dataset_name: str, seed: int, splits_root: Path) -> RecSysDataSet[Any]:
    set_random_state(seed)
    ds = RecSysDataSet.use_dataloader(DATASETS[dataset_name])
    steps = []
    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        steps.append(MakeImplicit(3))
    else:
        steps.append(DropToImplicit())
    steps.append(CorePruning(5))
    steps.append(UserOnlyTestHoldout(0.2))
    steps.append(AddUnitRating())
    steps.append(PersistSplit(splits_root / dataset_name / f"seed_{seed}"))
    pipe = Pipe(*steps)
    return pipe.process(ds)


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name="seed_sensitivity")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer, {"feedback": "implicit"})
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def collect_results(evaluator: Evaluator, dataset_name: str, seed: int) -> pd.DataFrame:
    rows = []
    results = evaluator.get_results()
    for dataset_id, df in results.items():
        if df is None or df.empty:
            continue
        for algo, g in df.groupby("algorithm"):
            out = {"dataset": dataset_name, "seed": seed, "algorithm_raw": algo}
            algo_label = algo
            for prefix, pretty in ALGO_LABELS.items():
                if str(algo).startswith(prefix):
                    algo_label = pretty
                    break
            out["algorithm"] = algo_label
            for metric_name in ["NDCG", "Precision"]:
                mg = g[g["name"] == metric_name]
                for k in KS:
                    val = mg.loc[mg["k"] == k, "value"]
                    out[f"{metric_name}@{k}"] = float(val.iloc[0]) if len(val) else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def add_precision_from_predictions(job_df: pd.DataFrame, checkpoints_root: Path, dataset_name: str, seed: int) -> pd.DataFrame:
    if job_df.empty:
        return job_df

    split_dir = Path("working") / "outputs" / "splits" / dataset_name / f"seed_{seed}"
    test_df = pd.read_csv(split_dir / "test.csv")
    rel = test_df.groupby("user")["item"].apply(set).to_dict()

    pred_files = list(Path(checkpoints_root).rglob("predictions.json"))
    if not pred_files:
        for k in KS:
            job_df[f"Precision@{k}"] = np.nan
        return job_df

    precision_map = {}
    for pred_file in pred_files:
        path_str = str(pred_file)
        if dataset_name not in path_str:
            continue
        try:
            pred_df = pd.read_json(pred_file)
        except Exception:
            continue
        if pred_df.empty or not {"user", "item", "rank"}.issubset(pred_df.columns):
            continue
        algo_key = None
        for raw_name, pretty in ALGO_LABELS.items():
            if raw_name.split(".")[-1] in path_str or raw_name in path_str:
                algo_key = pretty
                break
        if algo_key is None:
            inferred = str(pred_file.parent.name)
            for raw_name, pretty in ALGO_LABELS.items():
                if raw_name.split(".")[-1] in inferred or raw_name in inferred:
                    algo_key = pretty
                    break
        if algo_key is None:
            continue
        algo_prec = {}
        for k in KS:
            topk = pred_df[pred_df["rank"] <= k].copy()
            if topk.empty:
                algo_prec[k] = np.nan
                continue
            topk["hit"] = topk.apply(lambda r: 1 if r["user"] in rel and r["item"] in rel[r["user"]] else 0, axis=1)
            per_user = topk.groupby("user")["hit"].sum() / float(k)
            algo_prec[k] = float(per_user.mean()) if len(per_user) else np.nan
        precision_map[algo_key] = algo_prec

    for k in KS:
        job_df[f"Precision@{k}"] = job_df["algorithm"].map(lambda a: precision_map.get(a, {}).get(k, np.nan))
    return job_df


def run_job(dataset_name: str, seed: int) -> None:
    root, outputs, jobs_root, eval_json_root, splits_root = working_paths()
    job_dir = jobs_root / dataset_name
    job_dir.mkdir(parents=True, exist_ok=True)
    job_csv = job_dir / f"seed_{seed}.csv"
    if job_csv.exists():
        print(f"Job already completed: {job_csv}")
        print(pd.read_csv(job_csv).to_string(index=False))
        return

    print(f"Preparing dataset={dataset_name}, seed={seed}")
    ds = build_dataset(dataset_name, seed, splits_root)
    split_dir = splits_root / dataset_name / f"seed_{seed}"
    train_df = pd.read_csv(split_dir / "train.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    print(f"Train interactions={len(train_df)}, Test interactions={len(test_df)}, Users={train_df['user'].nunique()}, Items={train_df['item'].nunique()}")

    plan = build_plan()
    evaluator = Evaluator(NDCG(KS))

    print(f"Running OmniRec for dataset={dataset_name}, seed={seed}")
    run_omnirec(datasets=ds, plan=plan, evaluator=evaluator)

    eval_json_path = eval_json_root / f"{dataset_name}_seed_{seed}.json"
    evaluator.save_results(eval_json_path)

    results_df = collect_results(evaluator, dataset_name, seed)
    if results_df.empty:
        raise RuntimeError("Evaluator returned no rows for this job.")

    checkpoints_root = root / "checkpoints"
    results_df = add_precision_from_predictions(results_df, checkpoints_root, dataset_name, seed)
    results_df.to_csv(job_csv, index=False)
    print("Saved job metrics:")
    print(results_df.to_string(index=False))
    print(f"Artifacts: {job_csv}")


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [f"NDCG@{k}" for k in KS] + [f"Precision@{k}" for k in KS]
    rows = []
    grouped = cast(Any, df.groupby(["dataset", "algorithm"]))
    for key, g in grouped:
        dataset, algorithm = cast(tuple[str, str], key)
        row = {"dataset": dataset, "algorithm": algorithm, "n_seeds": int(g["seed"].nunique())}
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
            mean = row[f"{m}_mean"]
            std = row[f"{m}_std"]
            row[f"{m}_cv"] = float(std / mean) if pd.notna(mean) and mean != 0 else np.nan
            row[f"{m}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{m}_max"] = float(vals.max()) if len(vals) else np.nan
            row[f"{m}_range"] = float(vals.max() - vals.min()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm"]).reset_index(drop=True)


def paired_seed_analysis(results_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"NDCG@{k}" for k in KS] + [f"Precision@{k}" for k in KS]
    rows = []
    for dataset, dfg in results_df.groupby("dataset"):
        for metric in metrics:
            pivot = dfg.pivot(index="seed", columns="algorithm", values=metric)
            algos = [c for c in pivot.columns if pivot[c].notna().any()]
            for i in range(len(algos)):
                for j in range(i + 1, len(algos)):
                    a = algos[i]
                    b = algos[j]
                    diff = (pivot[a] - pivot[b]).dropna()
                    if len(diff) == 0:
                        continue
                    mean_diff = float(diff.mean())
                    std_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
                    se = float(std_diff / np.sqrt(len(diff))) if len(diff) > 0 else np.nan
                    rows.append({
                        "dataset": dataset,
                        "metric": metric,
                        "algo_a": a,
                        "algo_b": b,
                        "n_pairs": int(len(diff)),
                        "mean_diff": mean_diff,
                        "std_diff": std_diff,
                        "se_diff": se,
                        "approx_95ci_low": mean_diff - 1.96 * se if pd.notna(se) else np.nan,
                        "approx_95ci_high": mean_diff + 1.96 * se if pd.notna(se) else np.nan,
                    })
    return pd.DataFrame(rows)


def concise_summary(summary_df: pd.DataFrame) -> str:
    lines = []
    for dataset, g in summary_df.groupby("dataset"):
        lines.append(f"Dataset: {dataset}")
        if "NDCG@10_mean" in g.columns:
            tmp = g[["algorithm", "NDCG@10_mean", "NDCG@10_std", "NDCG@10_cv"]].sort_values("NDCG@10_mean", ascending=False)
            parts = []
            for _, r in tmp.iterrows():
                parts.append(f"{r['algorithm']} ({r['NDCG@10_mean']:.4f}±{r['NDCG@10_std']:.4f}, cv={r['NDCG@10_cv']:.4f})")
            lines.append("  NDCG@10: " + ", ".join(parts))
        if "Precision@10_mean" in g.columns:
            tmp = g[["algorithm", "Precision@10_mean", "Precision@10_std", "Precision@10_cv"]].sort_values("Precision@10_mean", ascending=False)
            parts = []
            for _, r in tmp.iterrows():
                val = r["Precision@10_mean"]
                std = r["Precision@10_std"]
                cv = r["Precision@10_cv"]
                if pd.isna(val):
                    parts.append(f"{r['algorithm']} (NA)")
                else:
                    parts.append(f"{r['algorithm']} ({val:.4f}±{std:.4f}, cv={cv:.4f})")
            lines.append("  Precision@10: " + ", ".join(parts))
        sens = g[["algorithm", "NDCG@10_cv", "Precision@10_cv"]].copy()
        sens["avg_cv"] = sens[["NDCG@10_cv", "Precision@10_cv"]].mean(axis=1, skipna=True)
        sens = sens.sort_values("avg_cv", ascending=False)
        parts = []
        for _, r in sens.iterrows():
            parts.append(f"{r['algorithm']} ({r['avg_cv']:.4f})")
        lines.append("  Seed sensitivity (avg CV over @10 metrics): " + ", ".join(parts))
    return "\n".join(lines)


def aggregate_jobs() -> None:
    _, outputs, jobs_root, _, _ = working_paths()
    job_files = sorted(jobs_root.rglob("seed_*.csv"))
    if not job_files:
        raise RuntimeError("No completed job CSVs found. Run dataset/seed jobs first.")

    all_rows = []
    for fp in job_files:
        df = pd.read_csv(fp)
        all_rows.append(df)
    results = pd.concat(all_rows, ignore_index=True)
    results = results.sort_values(["dataset", "algorithm", "seed"]).reset_index(drop=True)
    results.to_csv(outputs / "seed_level_results.csv", index=False)

    summary = summarize_results(results)
    summary.to_csv(outputs / "summary_by_dataset_algorithm.csv", index=False)

    paired = paired_seed_analysis(results)
    paired.to_csv(outputs / "paired_seed_differences.csv", index=False)

    text = concise_summary(summary)
    with open(outputs / "concise_summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")

    coverage = results.groupby(["dataset", "algorithm"])["seed"].nunique().reset_index(name="n_seeds")
    coverage.to_csv(outputs / "coverage.csv", index=False)

    print("=== Seed-level results ===")
    print(results.to_string(index=False))
    print("\n=== Summary by dataset and algorithm ===")
    print(summary.to_string(index=False))
    if not paired.empty:
        print("\n=== Short statistical analysis ===")
        print(paired.to_string(index=False))
    print("\n=== Concise summary ===")
    print(text)
    print(f"\nArtifacts written to: {outputs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    run_parser = sub.add_parser("run_job")
    run_parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), required=True)
    run_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)

    sub.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "run_job":
        run_job(args.dataset, args.seed)
    elif args.mode == "aggregate":
        aggregate_jobs()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
