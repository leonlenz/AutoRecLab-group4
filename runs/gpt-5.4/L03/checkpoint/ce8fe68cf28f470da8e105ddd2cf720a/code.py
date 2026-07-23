import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import numpy as np
import pandas as pd

from omnirec import NDCG, RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.base import Metric, MetricResult
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


class Precision(Metric):
    def __init__(self, ks: int | Iterable[int]):
        if isinstance(ks, int):
            ks = [ks]
        self.ks: List[int] = list(ks)

    def calculate(self, predictions: pd.DataFrame, test: pd.DataFrame) -> MetricResult:
        if predictions is None or len(predictions) == 0:
            return MetricResult(
                name="Precision",
                result={k: 0.0 for k in self.ks},
            )

        pred = predictions.copy()
        truth = test[["user", "item"]].drop_duplicates().copy()

        merged = pred.merge(truth.assign(relevant=1), on=["user", "item"], how="left")
        merged["relevant"] = merged["relevant"].fillna(0).astype(int)

        if "rank" not in merged.columns:
            if "score" in merged.columns:
                merged = merged.sort_values(["user", "score"], ascending=[True, False]).copy()
                merged["rank"] = merged.groupby("user").cumcount() + 1
            else:
                merged = merged.copy()
                merged["rank"] = merged.groupby("user").cumcount() + 1

        users = merged["user"].drop_duplicates()

        results: Dict[int, float] = {}
        for k in self.ks:
            topk = merged.loc[merged["rank"] <= k]
            user_hits = topk.groupby("user")["relevant"].sum()
            precision = (user_hits / k).reindex(users, fill_value=0.0).mean()
            results[k] = float(precision)
        return MetricResult(name="Precision", result=results)


def build_dataset(base_name: str):
    if base_name == "MovieLens100K":
        dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        pipeline = Pipe(MakeImplicit(3), CorePruning(5))
    elif base_name == "Amazon2014VideoGames":
        dataset = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
        pipeline = Pipe(MakeImplicit(3), CorePruning(5))
    elif base_name == "HetrecLastFM":
        dataset = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
        pipeline = Pipe(CorePruning(5))
    else:
        raise ValueError(f"Unsupported dataset: {base_name}")
    dataset = pipeline.process(dataset)
    return dataset


def make_seeded_split(dataset, seed: int):
    set_random_state(seed)
    splitter = UserHoldout(validation_size=0.0, test_size=0.2)
    return splitter.process(dataset)


def make_plan() -> ExperimentPlan:
    plan = ExperimentPlan("seed_sensitivity_baseline")
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
    out = out[["dataset", "seed", "algorithm", "algorithm_label", "fold", "name", "k", "value", "dataset_id"]]
    return out


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
        spread_pct_points = float(100.0 * (np.max(vals) - np.min(vals))) if len(vals) else np.nan
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
                "spread_pct_points": spread_pct_points,
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
        ds = eta_df[eta_df[eta_df.columns[0]] == dataset].sort_values(["metric", "k"])
        for _, r in ds.iterrows():
            strength = "low"
            if r["eta_squared_seed"] >= 0.14:
                strength = "high"
            elif r["eta_squared_seed"] >= 0.06:
                strength = "moderate"
            elif r["eta_squared_seed"] >= 0.01:
                strength = "small"
            print(f"  {r['metric']}@{int(r['k'])}: eta^2={r['eta_squared_seed']:.4f} ({strength})")


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    seeds = [11, 29, 47, 83, 131]
    datasets = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]

    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))
    plan = make_plan()

    all_results: List[pd.DataFrame] = []

    for dataset_name in datasets:
        print(f"\nLoading and preprocessing base dataset: {dataset_name}")
        base_dataset = build_dataset(dataset_name)
        print(base_dataset)

        for seed in seeds:
            print(f"\nRunning dataset={dataset_name}, seed={seed}")
            split_dataset = make_seeded_split(base_dataset, seed)
            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
            seed_df = extract_results(evaluator, dataset_name, seed)
            if len(seed_df) == 0:
                raise RuntimeError(f"No evaluation results found for dataset={dataset_name}, seed={seed}")
            print(seed_df[["dataset", "seed", "algorithm_label", "name", "k", "value"]].to_string(index=False))
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

    summary = {
        "datasets": datasets,
        "seeds": seeds,
        "algorithms": ["ALS", "ItemKNN", "Pop"],
        "metrics": ["NDCG@1", "NDCG@5", "NDCG@10", "Precision@1", "Precision@5", "Precision@10"],
        "files": {
            "per_seed_long": str(per_seed_path),
            "per_seed_wide": str(wide_path),
            "variability": str(variability_path),
            "algorithmwise_analysis": str(analysis_path),
            "eta_squared": str(eta_path),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(seed_analysis_df, eta_df)

    print("\nSaved outputs:")
    print(per_seed_path)
    print(wide_path)
    print(variability_path)
    print(analysis_path)
    print(eta_path)
    print(json_path)


if __name__ == "__main__":
    main()
