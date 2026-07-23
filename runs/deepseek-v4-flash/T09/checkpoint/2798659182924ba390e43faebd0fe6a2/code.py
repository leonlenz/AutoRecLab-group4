#!/usr/bin/env python3
"""
Experiment: Quantifying the effect of data split random seeds on recommender system accuracy.

Tests 3 algorithms (Pop, ItemKNN, ALS/ImplicitMF) on 3 datasets
(MovieLens100K, Amazon2014VideoGames, HetrecLastFM) with 5 different seeds.

Uses OmniRec exclusively.
"""

import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# OmniRec imports
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.subsample import Subsample
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    # Setup working directory
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    # Define random seeds
    seeds = [42, 123, 456, 789, 1111]

    # Define algorithm configurations (all with implicit feedback)
    plan = ExperimentPlan("Seed-Sensitivity-Study")

    plan.add_algorithm(
        LensKit.PopScorer,
        {"feedback": "implicit"},
    )

    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {"feedback": "implicit"},
    )

    plan.add_algorithm(
        LensKit.ImplicitMFScorer,
        {"feedback": "implicit"},
    )

    # Store results per (dataset_name, algorithm_name, metric, k) across seeds
    all_results = defaultdict(list)

    # Loop over seeds
    for seed_idx, seed in enumerate(seeds):
        print("\n" + "=" * 72)
        print(f"Seed {seed_idx + 1}/{len(seeds)}: seed = {seed}")
        print("=" * 72)

        # Set global random state for reproducibility
        set_random_state(seed)

        # Create a fresh evaluator for this seed (to avoid duplicate accumulation)
        evaluator = Evaluator(
            NDCG([1, 5, 10]),
            Precision([1, 5, 10]),
        )

        # ---------------------------------------------------------------
        # Preprocess MovieLens100K: MakeImplicit(4) + CorePruning(5) + UserHoldout
        # ---------------------------------------------------------------
        print("  Preprocessing MovieLens100K...")
        ml100k = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        pipeline_ml = Pipe(
            MakeImplicit(4),                         # ratings >= 4 -> implicit
            CorePruning(5),                          # 5-core filtering
            UserHoldout(validation_size=0.0, test_size=0.2),  # 80/20 user split
        )
        ml100k_split = pipeline_ml.process(ml100k)
        train_count = len(ml100k_split._data.get("train"))
        test_count = len(ml100k_split._data.get("test"))
        print(f"    Train: {train_count} interactions")
        print(f"    Test:  {test_count} interactions")

        # ---------------------------------------------------------------
        # Preprocess Amazon2014VideoGames: Subsample(0.2) + MakeImplicit(4) + CorePruning(5) + UserHoldout
        # ---------------------------------------------------------------
        print("  Preprocessing Amazon2014VideoGames...")
        amazon = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
        pipeline_amz = Pipe(
            Subsample(0.2),                          # Subsample to 20% for speed
            MakeImplicit(4),                         # ratings >= 4 -> implicit
            CorePruning(5),                          # 5-core filtering
            UserHoldout(validation_size=0.0, test_size=0.2),  # 80/20 user split
        )
        amazon_split = pipeline_amz.process(amazon)
        train_count = len(amazon_split._data.get("train"))
        test_count = len(amazon_split._data.get("test"))
        print(f"    Train: {train_count} interactions")
        print(f"    Test:  {test_count} interactions")

        # ---------------------------------------------------------------
        # Preprocess HetrecLastFM: CorePruning(5) + UserHoldout (already implicit)
        # ---------------------------------------------------------------
        print("  Preprocessing HetrecLastFM...")
        lastfm = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
        pipeline_lfm = Pipe(
            CorePruning(5),                          # 5-core filtering
            UserHoldout(validation_size=0.0, test_size=0.2),  # 80/20 user split
        )
        lastfm_split = pipeline_lfm.process(lastfm)
        train_count = len(lastfm_split._data.get("train"))
        test_count = len(lastfm_split._data.get("test"))
        print(f"    Train: {train_count} interactions")
        print(f"    Test:  {test_count} interactions")

        # ---------------------------------------------------------------
        # Run all algorithms on all three datasets for this seed
        # ---------------------------------------------------------------
        datasets_to_run = [ml100k_split, amazon_split, lastfm_split]

        print("  Running experiments...")
        run_omnirec(datasets=datasets_to_run, plan=plan, evaluator=evaluator)

        # Collect results for THIS SEED only (fresh evaluator = only this seed's results)
        results = evaluator.get_results()
        for dataset_key, df in results.items():
            for _, row in df.iterrows():
                algo = row["algorithm"]
                metric_name = row["name"]
                k_val = row["k"]
                value = row["value"]
                result_key = (dataset_key, algo, metric_name, k_val)
                all_results[result_key].append(value)

        print(f"  Collected {sum(len(v) for v in results.values())} result rows for seed {seed}")

    # ---------------------------------------------------------------
    # Aggregate results across seeds: report mean and std
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL AGGREGATED RESULTS (mean +/- std across 5 seeds)")
    print("=" * 72)

    # Summarize by dataset, algorithm, metric
    summary_rows = []
    for (dataset_key, algo, metric_name, k_val), values in sorted(all_results.items()):
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # sample std
        summary_rows.append({
            "dataset": dataset_key,
            "algorithm": algo,
            "metric": f"{metric_name}@{k_val}",
            "mean": mean_val,
            "std": std_val,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Print per-dataset tables
    for ds in summary_df["dataset"].unique():
        ds_mask = summary_df["dataset"] == ds
        print(f"\n--- Dataset: {ds} ---")
        ds_df = summary_df[ds_mask].copy()
        for algo in ds_df["algorithm"].unique():
            algo_mask = ds_df["algorithm"] == algo
            print(f"\n  Algorithm: {algo}")
            algo_df = ds_df[algo_mask].sort_values("metric")
            for _, row in algo_df.iterrows():
                print(f"    {row['metric']:>12s}: {row['mean']:.6f} +/- {row['std']:.6f}")

    # Also print a compact summary table
    print("\n\nCOMPACT SUMMARY TABLE")
    print("-" * 96)
    print(f"{'Dataset':<30s} {'Algorithm':<28s} {'Metric':<14s} {'Mean':<14s} {'Std':<14s}")
    print("-" * 96)

    for row in sorted(summary_rows, key=lambda r: (r["dataset"], r["algorithm"], r["metric"])):
        ds_short = str(row["dataset"])[:30]
        algo_short = str(row["algorithm"])[:28]
        print(f"{ds_short:<30s} {algo_short:<28s} {row['metric']:<14s} {row['mean']:<14.6f} {row['std']:<14.6f}")

    # Statistical analysis: compute coefficient of variation (CV = std/mean) to quantify seed sensitivity
    print("\n\nSTATISTICAL ANALYSIS: Coefficient of Variation (CV = std/mean)")
    print("-" * 96)
    print(f"{'Dataset':<30s} {'Algorithm':<28s} {'Metric':<14s} {'CV':<14s}")
    print("-" * 96)
    cv_rows = []
    for row in sorted(summary_rows, key=lambda r: (r["dataset"], r["algorithm"], r["metric"])):
        if row["mean"] != 0:
            cv = row["std"] / abs(row["mean"])
        else:
            cv = float("inf")
        cv_rows.append({**row, "cv": cv})
        ds_short = str(row["dataset"])[:30]
        algo_short = str(row["algorithm"])[:28]
        print(f"{ds_short:<30s} {algo_short:<28s} {row['metric']:<14s} {cv:<14.6f}")

    # Find the most and least sensitive combinations
    if cv_rows:
        cv_df = pd.DataFrame(cv_rows)
        most_sensitive = cv_df.loc[cv_df["cv"].idxmax()]
        least_sensitive = cv_df.loc[cv_df["cv"].idxmin()]
        print(f"\nMost seed-sensitive: {most_sensitive['dataset']} / {most_sensitive['algorithm']} / {most_sensitive['metric']} (CV={most_sensitive['cv']:.4f})")
        print(f"Least seed-sensitive: {least_sensitive['dataset']} / {least_sensitive['algorithm']} / {least_sensitive['metric']} (CV={least_sensitive['cv']:.4f})")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
