import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

SEEDS = [11, 29, 47, 83, 131]
DATASETS = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]
KS = [1, 5, 10]


def patch_omnirec_lenskit_runner() -> None:
    import pandas as pd
    from lenskit.batch import predict as lk_predict, recommend as lk_recommend
    import omnirec_runner.lenskit_runner as lk_runner_mod

    def patched_predict(self: Any) -> Dict[str, List[Any]]:
        self.test.rename(columns={"user": "user_id", "item": "item_id"}, inplace=True)

        if "rating" in self.train.columns:
            if "rating" in self.test.columns:
                self.test.drop(columns="rating", inplace=True)
            predictions = lk_predict(self.model, self.test)
            predictions_df = predictions.to_df()
            predictions_df.rename(columns={"user_id": "user", "item_id": "item"}, inplace=True)
            if "score" in predictions_df.columns and "rating" not in predictions_df.columns:
                predictions_df.rename(columns={"score": "rating"}, inplace=True)
            return cast(Dict[str, List[Any]], predictions_df.to_dict(orient="list"))

        predictions = lk_recommend(self.model, self.test)
        predictions_df = predictions.to_df()
        predictions_df.rename(columns={"user_id": "user", "item_id": "item"}, inplace=True)

        if "score" not in predictions_df.columns:
            score_candidates = [c for c in predictions_df.columns if c not in {"user", "item", "rank"}]
            if len(score_candidates) == 1:
                predictions_df.rename(columns={score_candidates[0]: "score"}, inplace=True)
            elif len(score_candidates) > 1:
                if "prediction" in score_candidates:
                    predictions_df.rename(columns={"prediction": "score"}, inplace=True)
                else:
                    predictions_df.rename(columns={score_candidates[0]: "score"}, inplace=True)
            else:
                predictions_df["score"] = 1.0

        if "rank" not in predictions_df.columns:
            sort_cols = ["user"]
            ascending = [True]
            if "score" in predictions_df.columns:
                sort_cols.append("score")
                ascending.append(False)
            predictions_df = predictions_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
            predictions_df["rank"] = predictions_df.groupby("user").cumcount() + 1

        predictions_df = predictions_df[[c for c in ["user", "item", "score", "rank"] if c in predictions_df.columns]].copy()
        predictions_df["user"] = predictions_df["user"].astype(int)
        predictions_df["item"] = predictions_df["item"].astype(int)
        predictions_df["rank"] = predictions_df["rank"].astype(int)
        predictions_df["score"] = predictions_df["score"].astype(float)
        return cast(Dict[str, List[Any]], predictions_df.to_dict(orient="list"))

    cast(Any, lk_runner_mod.Lenskit).predict = patched_predict


def build_base_dataset(base_name: str):
    if base_name == "MovieLens100K":
        dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        pipeline = Pipe(
            MakeImplicit(4),
            CorePruning(5),
        )
    elif base_name == "Amazon2014VideoGames":
        dataset = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
        pipeline = Pipe(
            MakeImplicit(4),
            CorePruning(5),
        )
    elif base_name == "HetrecLastFM":
        dataset = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
        pipeline = Pipe(
            CorePruning(5),
        )
    else:
        raise ValueError(f"Unsupported dataset: {base_name}")
    return pipeline.process(dataset)


def make_seeded_split(dataset, seed: int):
    set_random_state(seed)
    splitter = UserHoldout(validation_size=0.01, test_size=0.20)
    return splitter.process(dataset)


def make_plan() -> ExperimentPlan:
    plan = ExperimentPlan("seed_sensitivity_lenskit_via_omnirec")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def normalize_algorithm_name(algo: str) -> str:
    if algo.startswith("LensKit.ImplicitMFScorer"):
        return "ALS"
    if algo.startswith("LensKit.ItemKNNScorer"):
        return "ItemKNN"
    if algo.startswith("LensKit.PopScorer"):
        return "Pop"
    return algo.split("-")[0]


def extract_results(evaluator: Evaluator, dataset_name: str, seed: int) -> pd.DataFrame:
    results = evaluator.get_results()
    frames: List[pd.DataFrame] = []
    for dataset_id, df in results.items():
        local = df.copy()
        local["dataset_id"] = dataset_id
        local["dataset"] = dataset_name
        local["seed"] = seed
        local["algorithm_label"] = local["algorithm"].map(normalize_algorithm_name)
        frames.append(local)
    if not frames:
        return pd.DataFrame(
            {
                "dataset": pd.Series(dtype="object"),
                "seed": pd.Series(dtype="int64"),
                "algorithm_label": pd.Series(dtype="object"),
                "name": pd.Series(dtype="object"),
                "k": pd.Series(dtype="float64"),
                "value": pd.Series(dtype="float64"),
            }
        )
    out = pd.concat(frames, ignore_index=True)
    keep_cols = [c for c in ["dataset", "seed", "algorithm", "algorithm_label", "fold", "name", "k", "value", "dataset_id"] if c in out.columns]
    return out[keep_cols]


def variability_table(long_df: pd.DataFrame) -> pd.DataFrame:
    grp = long_df.groupby(["dataset", "algorithm_label", "name", "k"], as_index=False)
    stats = grp["value"].agg(["mean", "std", "min", "max"]).reset_index()
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = np.where(stats["mean"].abs() > 1e-12, stats["std"] / stats["mean"].abs(), np.nan)
    return stats


def seed_effect_analysis(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = list(long_df.groupby(["dataset", "algorithm_label", "name", "k"]))
    for key, g in grouped:
        dataset, algorithm, metric, k = cast(Tuple[str, str, str, Any], key)
        vals = g["value"].to_numpy(dtype=float)
        mean = float(np.mean(vals)) if len(vals) else np.nan
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rel_std_pct = float(100.0 * std / abs(mean)) if abs(mean) > 1e-12 else np.nan
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "metric": metric,
                "k": k,
                "n_seeds": int(len(vals)),
                "mean": mean,
                "std": std,
                "rel_std_pct": rel_std_pct,
                "range": float(np.max(vals) - np.min(vals)) if len(vals) else np.nan,
                "spread_pct_points": float(100.0 * (np.max(vals) - np.min(vals))) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def dataset_metric_seed_eta(all_seed_algo_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = list(all_seed_algo_df.groupby(["dataset", "name", "k"]))
    for key, g in grouped:
        dataset, metric, k = cast(Tuple[str, str, Any], key)
        grand = g["value"].mean()
        seed_means = g.groupby("seed")["value"].mean()
        counts = g.groupby("seed").size()
        ss_between = float(((seed_means - grand) ** 2 * counts).sum())
        ss_total = float(((g["value"] - grand) ** 2).sum())
        eta_sq = ss_between / ss_total if ss_total > 1e-12 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "k": k,
                "eta_squared_seed": eta_sq,
                "grand_mean": float(grand),
                "n_obs": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def pivot_seed_results(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.copy()
    wide["metric"] = wide["name"] + "@" + wide["k"].astype(str)
    wide = wide.pivot_table(
        index=["dataset", "algorithm_label", "seed"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def print_summary(seed_stats: pd.DataFrame, eta_df: pd.DataFrame):
    print("\n=== Seed Sensitivity Summary (algorithm-wise) ===")
    for dataset in sorted(seed_stats["dataset"].unique()):
        print(f"\nDataset: {dataset}")
        ds = seed_stats[seed_stats["dataset"] == dataset].sort_values(["algorithm", "metric", "k"])
        for _, r in ds.iterrows():
            print(
                f"  {r['algorithm']:8s} | {r['metric']}@{int(r['k'])}: "
                f"mean={r['mean']:.4f}, std={r['std']:.4f}, rel_std={r['rel_std_pct']:.2f}%, range={r['range']:.4f}"
            )

    print("\n=== Seed Effect Size Across Algorithms Within Dataset-Metric ===")
    for dataset in sorted(eta_df["dataset"].unique()):
        print(f"\nDataset: {dataset}")
        ds = eta_df[eta_df["dataset"] == dataset].sort_values(["metric", "k"])
        for _, r in ds.iterrows():
            strength = "low"
            if r["eta_squared_seed"] >= 0.14:
                strength = "high"
            elif r["eta_squared_seed"] >= 0.06:
                strength = "moderate"
            elif r["eta_squared_seed"] >= 0.01:
                strength = "small"
            print(f"  {r['metric']}@{int(r['k'])}: eta^2={r['eta_squared_seed']:.4f} ({strength})")


def short_statistical_report(seed_stats: pd.DataFrame, eta_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append("Short statistical analysis:")
    if seed_stats.empty:
        lines.append("- No results available.")
        return lines

    overall = seed_stats["rel_std_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(overall):
        lines.append(
            f"- Across all dataset/algorithm/metric cells, the median relative std across seeds is {overall.median():.2f}% and the maximum is {overall.max():.2f}%."
        )

    if not eta_df.empty:
        high = int((eta_df["eta_squared_seed"] >= 0.14).sum())
        moderate = int(((eta_df["eta_squared_seed"] >= 0.06) & (eta_df["eta_squared_seed"] < 0.14)).sum())
        small = int(((eta_df["eta_squared_seed"] >= 0.01) & (eta_df["eta_squared_seed"] < 0.06)).sum())
        negligible = int((eta_df["eta_squared_seed"] < 0.01).sum())
        lines.append(
            f"- Seed-effect eta-squared counts across dataset-metric pairs: negligible={negligible}, small={small}, moderate={moderate}, high={high}."
        )

    for dataset in sorted(seed_stats["dataset"].unique()):
        ds = seed_stats[seed_stats["dataset"] == dataset]
        if ds.empty:
            continue
        worst = ds.sort_values("rel_std_pct", ascending=False).iloc[0]
        best = ds.sort_values("rel_std_pct", ascending=True).iloc[0]
        lines.append(
            f"- {dataset}: most seed-sensitive cell is {worst['algorithm']} {worst['metric']}@{int(worst['k'])} "
            f"(rel std {worst['rel_std_pct']:.2f}%), while the most stable is {best['algorithm']} {best['metric']}@{int(best['k'])} "
            f"(rel std {best['rel_std_pct']:.2f}%)."
        )
    return lines


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    patch_omnirec_lenskit_runner()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    evaluator = Evaluator(NDCG(KS), Precision(KS))
    plan = make_plan()

    all_results: List[pd.DataFrame] = []

    for dataset_name in DATASETS:
        print(f"\nLoading and preprocessing base dataset: {dataset_name}")
        base_dataset = build_base_dataset(dataset_name)
        print(base_dataset)

        for seed in SEEDS:
            print(f"\nRunning dataset={dataset_name}, seed={seed}")
            split_dataset = make_seeded_split(base_dataset, seed)
            evaluator = Evaluator(NDCG(KS), Precision(KS))
            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
            seed_df = extract_results(evaluator, dataset_name, seed)
            if len(seed_df) == 0:
                raise RuntimeError(f"No evaluation results found for dataset={dataset_name}, seed={seed}")
            cols_to_show = [c for c in ["dataset", "seed", "algorithm_label", "name", "k", "value"] if c in seed_df.columns]
            print(seed_df[cols_to_show].to_string(index=False))
            all_results.append(seed_df)

    long_df = pd.concat(all_results, ignore_index=True)
    long_df = long_df.sort_values(["dataset", "algorithm_label", "seed", "name", "k"]).reset_index(drop=True)

    per_seed_df = long_df[["dataset", "seed", "algorithm_label", "name", "k", "value"]].copy()
    per_seed_df.rename(columns={"algorithm_label": "algorithm", "name": "metric"}, inplace=True)

    variability_df = variability_table(long_df)
    variability_df.rename(columns={"algorithm_label": "algorithm", "name": "metric"}, inplace=True)

    seed_analysis_df = seed_effect_analysis(long_df)
    eta_df = dataset_metric_seed_eta(long_df)
    wide_df = pivot_seed_results(long_df)
    wide_df.rename(columns={"algorithm_label": "algorithm"}, inplace=True)

    per_seed_path = output_dir / "per_seed_metrics_long.csv"
    wide_path = output_dir / "per_seed_metrics_wide.csv"
    variability_path = output_dir / "seed_variability_summary.csv"
    analysis_path = output_dir / "seed_effect_algorithmwise.csv"
    eta_path = output_dir / "seed_effect_eta_squared.csv"
    json_path = output_dir / "summary.json"

    per_seed_df.to_csv(per_seed_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    variability_df.to_csv(variability_path, index=False)
    seed_analysis_df.to_csv(analysis_path, index=False)
    eta_df.to_csv(eta_path, index=False)

    report_lines = short_statistical_report(seed_analysis_df, eta_df)

    summary = {
        "datasets": DATASETS,
        "seeds": SEEDS,
        "algorithms": ["ALS", "ItemKNN", "Pop"],
        "metrics": ["NDCG@1", "NDCG@5", "NDCG@10", "Precision@1", "Precision@5", "Precision@10"],
        "notes": {
            "framework": "OmniRec used exclusively; LensKit algorithms accessed via OmniRec wrappers.",
            "bug_fix": "Monkey-patched OmniRec's installed LensKit runner so implicit recommendation predictions always include user, item, score, and rank before OmniRec ranking evaluation.",
            "split_note": "OmniRec UserHoldout requires both validation_size and test_size; this script uses validation_size=0.01 and test_size=0.20 as the closest documented user-holdout configuration.",
            "implicit_conversion": "MovieLens100K and Amazon2014VideoGames use MakeImplicit(4), which keeps ratings >= 4 and matches ratings > 3."
        },
        "files": {
            "per_seed_long": str(per_seed_path),
            "per_seed_wide": str(wide_path),
            "variability": str(variability_path),
            "algorithmwise_analysis": str(analysis_path),
            "eta_squared": str(eta_path)
        },
        "short_statistical_report": report_lines,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(seed_analysis_df, eta_df)
    print("\n" + "\n".join(report_lines))

    print("\nSaved outputs:")
    print(per_seed_path)
    print(wide_path)
    print(variability_path)
    print(analysis_path)
    print(eta_path)
    print(json_path)


if __name__ == "__main__":
    main()
