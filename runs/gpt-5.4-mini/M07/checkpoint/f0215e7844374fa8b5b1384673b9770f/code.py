import os
import json
import math
import statistics as stats
from typing import List, Tuple

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG, HR
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.util.util import set_random_state
from omnirec.util.run import run_omnirec


def summarize_seed_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    metric_col = None
    for candidate in ["metric", "metric_name", "name"]:
        if candidate in df.columns:
            metric_col = candidate
            break
    if metric_col is None:
        raise ValueError(f"Could not find metric column in results: {list(df.columns)}")

    value_col = None
    for candidate in ["value", "score", "result"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            raise ValueError(f"Could not infer numeric value column in results: {list(df.columns)}")
        value_col = numeric_cols[0]

    rows = []
    group_cols = [c for c in ["dataset", "algorithm", metric_col] if c in df.columns]
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = pd.to_numeric(g[value_col], errors="coerce").dropna().to_list()
        if not vals:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else float("nan")
        row = {"mean": mean, "std": std, "cv": cv, "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}
        for col, key in zip(group_cols, keys):
            row[col] = key
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan("Seed-Sensitivity-Study")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def preprocess_dataset(dataset_name: str, seed: int):
    set_random_state(seed)
    if dataset_name == "MovieLens100K":
        ds = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        pipeline = Pipe(MakeImplicit(3), CorePruning(5), UserHoldout(validation_size=0.2, test_size=0.2))
    elif dataset_name == "Amazon2014VideoGames":
        ds = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
        pipeline = Pipe(MakeImplicit(3), CorePruning(5), UserHoldout(validation_size=0.2, test_size=0.2))
    elif dataset_name == "HetrecLastFM":
        ds = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
        pipeline = Pipe(CorePruning(5), UserHoldout(validation_size=0.2, test_size=0.2))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return pipeline.process(ds)


def result_to_dataframe(result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if hasattr(result, "to_df"):
        return result.to_df()
    return pd.DataFrame(result)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    dataset_names = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]

    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), HR([1, 5, 10]))

    all_results = []
    metadata = []

    for dataset_name in dataset_names:
        for seed in seeds:
            print(f"\n=== Running {dataset_name} | seed={seed} ===")
            raw_ds = RecSysDataSet.use_dataloader(getattr(DataSet, dataset_name))
            meta_row = {
                "dataset": dataset_name,
                "seed": seed,
                "raw_interactions": raw_ds.num_interactions(),
                "min_rating": raw_ds.min_rating() if hasattr(raw_ds, "min_rating") else None,
                "max_rating": raw_ds.max_rating() if hasattr(raw_ds, "max_rating") else None,
                "preprocessing": "MakeImplicit(3) -> CorePruning(5) -> UserHoldout(test_size=0.2)" if dataset_name != "HetrecLastFM" else "CorePruning(5) -> UserHoldout(test_size=0.2)",
            }
            metadata.append(meta_row)

            split_ds = preprocess_dataset(dataset_name, seed)
            result = run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            res_df = result_to_dataframe(result)
            res_df["dataset"] = dataset_name
            res_df["seed"] = seed
            all_results.append(res_df)
            print(res_df)

    results = pd.concat(all_results, ignore_index=True)
    results_path = os.path.join(working_dir, "seed_sensitivity_results.csv")
    results.to_csv(results_path, index=False)

    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(working_dir, "seed_sensitivity_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    summary = summarize_seed_sensitivity(results)
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== Dataset metadata ===")
    print(metadata_df)
    print("\n=== Seed sensitivity summary ===")
    print(summary)

    print("\nBrief statistical analysis:")
    metric_col = next((c for c in ["metric", "metric_name", "name"] if c in summary.columns), None)
    for group_key, g in summary.groupby(["dataset", "algorithm"]):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        dataset, algo = group_key
        max_cv = g["cv"].replace([np.inf, -np.inf], np.nan).max()
        best_metric_rows = g.sort_values("cv").head(3)
        print(f"- {dataset} / {algo}: max seed CV = {max_cv:.4f}")
        print(f"  Lowest-variance metrics: {best_metric_rows[[metric_col, 'mean', 'std', 'cv']].to_dict('records')}")

    print(f"\nSaved results to: {results_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
