import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import RawData, SplitData
from omnirec.preprocess.base import Preprocessor
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.filter import RatingFilter
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, HR
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


class ToImplicitUnit(Preprocessor[RawData, RawData]):
    def __init__(self) -> None:
        super().__init__()

    def _process(self, dataset: RecSysDataSet[RawData]) -> RecSysDataSet[RawData]:
        df = dataset._data.df.copy()
        keep = [c for c in ["user", "item", "timestamp"] if c in df.columns]
        if not {"user", "item"}.issubset(df.columns):
            raise ValueError("ToImplicitUnit requires user and item columns")
        out = df[keep].drop_duplicates(subset=["user", "item"], keep="last").reset_index(drop=True)
        out["rating"] = 1
        ordered_cols = [c for c in ["user", "item", "rating", "timestamp"] if c in out.columns]
        dataset._data.df = out[ordered_cols]
        return dataset


def working_paths() -> tuple[Path, Path, Path, Path]:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    root = Path(working_dir)
    outputs = root / "outputs"
    jobs = outputs / "jobs"
    outputs.mkdir(parents=True, exist_ok=True)
    jobs.mkdir(parents=True, exist_ok=True)
    return root, outputs, jobs, root / "checkpoints"


def save_split_csvs(dataset: RecSysDataSet[SplitData], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    dataset._data.train.to_csv(split_dir / "train.csv", index=False)
    dataset._data.val.to_csv(split_dir / "val.csv", index=False)
    dataset._data.test.to_csv(split_dir / "test.csv", index=False)


def build_dataset(dataset_name: str, seed: int, split_dir: Path) -> RecSysDataSet[Any]:
    set_random_state(seed)
    ds = RecSysDataSet.use_dataloader(DATASETS[dataset_name])
    steps: list[Preprocessor[Any, Any]] = []

    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        steps.append(RatingFilter(lower=3))
        steps.append(ToImplicitUnit())
    else:
        steps.append(ToImplicitUnit())

    steps.append(CorePruning(5))
    steps.append(UserHoldout(0.0, 0.2))
    pipe = Pipe(*steps)
    ds = pipe.process(ds)
    save_split_csvs(ds, split_dir)
    return ds


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name="seed_sensitivity")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(NDCG(KS), HR(KS))


def normalize_algo_name(algo: str) -> str:
    s = str(algo)
    for raw, pretty in ALGO_LABELS.items():
        if s.startswith(raw):
            return pretty
    if "ImplicitMFScorer" in s:
        return "ALS"
    if "ItemKNNScorer" in s:
        return "ItemKNN"
    if "PopScorer" in s:
        return "Pop"
    return s


def collect_results_from_evaluator(evaluator: Evaluator, dataset_name: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    results = evaluator.get_results()
    for _, df in results.items():
        if df is None or df.empty:
            continue
        for algo, g in df.groupby("algorithm"):
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "seed": seed,
                "algorithm": normalize_algo_name(str(algo)),
                "algorithm_raw": str(algo),
            }
            for metric_name in ["NDCG", "HR"]:
                mg = g[g["name"] == metric_name]
                for k in KS:
                    val = mg.loc[mg["k"] == k, "value"]
                    row[f"{metric_name}@{k}"] = float(val.iloc[0]) if len(val) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _load_test_relevance(split_dir: Path) -> dict[Any, set[Any]]:
    test_df = pd.read_csv(split_dir / "test.csv")
    return test_df.groupby("user")["item"].apply(lambda s: set(s.tolist())).to_dict()


def _find_prediction_files(checkpoints_root: Path, dataset_name: str) -> list[Path]:
    out = []
    if not checkpoints_root.exists():
        return out
    for p in checkpoints_root.rglob("*"):
        if p.is_file() and p.name.startswith("pred") and p.suffix in {".csv", ".json", ".parquet"}:
            if dataset_name in str(p):
                out.append(p)
    return out


def _load_prediction_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported prediction file: {path}")


def _standardize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    if "user_id" in df.columns:
        colmap["user_id"] = "user"
    if "item_id" in df.columns:
        colmap["item_id"] = "item"
    if "score" not in df.columns and "prediction" in df.columns:
        colmap["prediction"] = "score"
    if "rank" not in df.columns and "rnk" in df.columns:
        colmap["rnk"] = "rank"
    df = df.rename(columns=colmap).copy()
    if not {"user", "item"}.issubset(df.columns):
        return pd.DataFrame()
    if "rank" not in df.columns:
        sort_cols = [c for c in ["user", "score"] if c in df.columns]
        if sort_cols == ["user", "score"]:
            df = df.sort_values(["user", "score"], ascending=[True, False]).copy()
            df["rank"] = df.groupby("user").cumcount() + 1
        else:
            return pd.DataFrame()
    return df


def add_precision_from_checkpoints(results_df: pd.DataFrame, checkpoints_root: Path, dataset_name: str, split_dir: Path) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    rel = _load_test_relevance(split_dir)
    pred_files = _find_prediction_files(checkpoints_root, dataset_name)
    precision_map: dict[str, dict[int, float]] = {}

    for pf in pred_files:
        try:
            pdf = _standardize_prediction_df(_load_prediction_df(pf))
        except Exception:
            continue
        if pdf.empty:
            continue
        algo = None
        pstr = str(pf)
        if "ImplicitMFScorer" in pstr:
            algo = "ALS"
        elif "ItemKNNScorer" in pstr:
            algo = "ItemKNN"
        elif "PopScorer" in pstr:
            algo = "Pop"
        if algo is None:
            continue
        algo_prec: dict[int, float] = {}
        for k in KS:
            topk = pdf[pdf["rank"] <= k].copy()
            if topk.empty:
                algo_prec[k] = np.nan
                continue
            topk["hit"] = topk.apply(lambda r: 1 if r["user"] in rel and r["item"] in rel[r["user"]] else 0, axis=1)
            per_user = topk.groupby("user")["hit"].sum() / float(k)
            algo_prec[k] = float(per_user.mean()) if len(per_user) else np.nan
        precision_map[algo] = algo_prec

    for k in KS:
        results_df[f"Precision@{k}"] = results_df["algorithm"].map(lambda a: precision_map.get(a, {}).get(k, np.nan))
    return results_df


def run_job(dataset_name: str, seed: int) -> None:
    root, outputs, jobs_root, checkpoints_root = working_paths()
    job_dir = jobs_root / dataset_name
    split_dir = outputs / "splits" / dataset_name / f"seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    job_csv = job_dir / f"seed_{seed}.csv"

    print(f"Preparing dataset={dataset_name}, seed={seed}")
    dataset = build_dataset(dataset_name, seed, split_dir)

    print(f"Train={len(dataset._data.train)} Val={len(dataset._data.val)} Test={len(dataset._data.test)}")
    plan = build_plan()
    evaluator = build_evaluator()

    print(f"Running OmniRec for dataset={dataset_name}, seed={seed}")
    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results_df = collect_results_from_evaluator(evaluator, dataset_name, seed)
    if results_df.empty:
        raise RuntimeError("Evaluator returned no rows; inspect OmniRec checkpoint logs.")

    results_df = add_precision_from_checkpoints(results_df, checkpoints_root, dataset_name, split_dir)
    results_df.to_csv(job_csv, index=False)
    print(results_df.to_string(index=False))
    print(f"Saved {job_csv}")


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [f"NDCG@{k}" for k in KS] + [f"HR@{k}" for k in KS] + [f"Precision@{k}" for k in KS]
    rows = []
    grouped = df.groupby(["dataset", "algorithm"])
    for key in grouped.groups.keys():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        dataset, algorithm = cast(tuple[Any, Any], key)
        g = grouped.get_group(key)
        row: dict[str, Any] = {"dataset": dataset, "algorithm": algorithm, "n_seeds": int(g["seed"].nunique())}
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce").dropna() if m in g.columns else pd.Series(dtype=float)
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else np.nan)
            row[f"{m}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{m}_max"] = float(vals.max()) if len(vals) else np.nan
            row[f"{m}_range"] = float(vals.max() - vals.min()) if len(vals) else np.nan
            mean = row[f"{m}_mean"]
            std = row[f"{m}_std"]
            row[f"{m}_cv"] = float(std / mean) if pd.notna(mean) and mean != 0 and pd.notna(std) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm"]).reset_index(drop=True)


def paired_seed_analysis(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"NDCG@{k}" for k in KS] + [f"HR@{k}" for k in KS] + [f"Precision@{k}" for k in KS]
    rows = []
    for dataset, dfg in df.groupby("dataset"):
        for metric in metrics:
            if metric not in dfg.columns:
                continue
            pivot = dfg.pivot(index="seed", columns="algorithm", values=metric)
            algos = list(pivot.columns)
            for i in range(len(algos)):
                for j in range(i + 1, len(algos)):
                    a = algos[i]
                    b = algos[j]
                    diff = (pivot[a] - pivot[b]).dropna()
                    if len(diff) == 0:
                        continue
                    mean_diff = float(diff.mean())
                    std_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
                    se = float(std_diff / math.sqrt(len(diff))) if len(diff) else np.nan
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
            lines.append("  NDCG@10: " + ", ".join(
                f"{r.algorithm} ({r['NDCG@10_mean']:.4f}±{r['NDCG@10_std']:.4f}, cv={r['NDCG@10_cv']:.4f})"
                for _, r in tmp.iterrows()
            ))
        if "Precision@10_mean" in g.columns:
            parts = []
            tmp = g[["algorithm", "Precision@10_mean", "Precision@10_std", "Precision@10_cv"]].sort_values("Precision@10_mean", ascending=False, na_position="last")
            for _, r in tmp.iterrows():
                if pd.isna(r["Precision@10_mean"]):
                    parts.append(f"{r.algorithm} (NA)")
                else:
                    parts.append(f"{r.algorithm} ({r['Precision@10_mean']:.4f}±{r['Precision@10_std']:.4f}, cv={r['Precision@10_cv']:.4f})")
            lines.append("  Precision@10: " + ", ".join(parts))
        sens = g[[c for c in ["algorithm", "NDCG@10_cv", "Precision@10_cv"] if c in g.columns]].copy()
        if not sens.empty:
            metric_cols = [c for c in ["NDCG@10_cv", "Precision@10_cv"] if c in sens.columns]
            sens["avg_cv"] = sens[metric_cols].mean(axis=1, skipna=True)
            sens = sens.sort_values("avg_cv", ascending=False)
            lines.append("  Seed sensitivity: " + ", ".join(f"{r.algorithm} ({r.avg_cv:.4f})" for _, r in sens.iterrows()))
    return "\n".join(lines)


def aggregate_jobs() -> None:
    _, outputs, jobs_root, _ = working_paths()
    job_files = sorted(jobs_root.rglob("seed_*.csv"))
    if not job_files:
        raise RuntimeError("No job CSVs found. Run one or more run_job commands first.")
    frames = [pd.read_csv(p) for p in job_files]
    results = pd.concat(frames, ignore_index=True).sort_values(["dataset", "algorithm", "seed"]).reset_index(drop=True)
    results.to_csv(outputs / "seed_level_results.csv", index=False)

    summary = summarize_results(results)
    summary.to_csv(outputs / "summary_by_dataset_algorithm.csv", index=False)

    paired = paired_seed_analysis(results)
    paired.to_csv(outputs / "paired_seed_differences.csv", index=False)

    text = concise_summary(summary)
    with open(outputs / "concise_summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("=== Seed-level results ===")
    print(results.to_string(index=False))
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    if not paired.empty:
        print("\n=== Short statistical analysis ===")
        print(paired.to_string(index=False))
    print("\n=== Concise summary ===")
    print(text)
    print(f"\nArtifacts written to {outputs}")


def run_all() -> None:
    for dataset_name in DATASETS:
        for seed in SEEDS:
            run_job(dataset_name, seed)
    aggregate_jobs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniRec seed-sensitivity experiment")
    sub = parser.add_subparsers(dest="mode")

    run_parser = sub.add_parser("run_job", help="Run one dataset/seed job")
    run_parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), required=True)
    run_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)

    sub.add_parser("aggregate", help="Aggregate completed jobs")
    sub.add_parser("run_all", help="Run all dataset/seed jobs then aggregate")

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        parser.exit(0)
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "run_job":
        run_job(args.dataset, args.seed)
    elif args.mode == "aggregate":
        aggregate_jobs()
    elif args.mode == "run_all":
        run_all()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
