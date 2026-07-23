#!/usr/bin/env python3
"""
Experiment: Quantifying the Effect of Data Split Random Seeds on Recommender Accuracy.

Tests 3 algorithms (ALS/ImplicitMF, ItemKNN, Pop) x 3 datasets (MovieLens100K,
Amazon2014VideoGames, HetrecLastFM) x 5 random seeds = 45 runs.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    # Create working directory
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    # =========================================================================
    # 1. LOAD RAW DATASETS
    # =========================================================================
    print("=" * 70)
    print("Loading datasets...")
    print("=" * 70)

    ml100k_raw = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print(f"  MovieLens100K: {ml100k_raw.num_interactions()} interactions")

    amazon_vg_raw = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
    print(f"  Amazon2014VideoGames: {amazon_vg_raw.num_interactions()} interactions")

    lastfm_raw = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
    print(f"  HetrecLastFM: {lastfm_raw.num_interactions()} interactions")

    # =========================================================================
    # 2. SEEDS FOR THE EXPERIMENT
    # =========================================================================
    seeds = [42, 123, 456, 789, 1111]

    # =========================================================================
    # 3. PREPROCESS: Generate 5 splits per dataset (different random states)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Preprocessing datasets with 5 different seeds each...")
    print("=" * 70)

    # Keep track of (dataset, seed) metadata for each preprocessed dataset
    dataset_meta = []  # list of (ds_name, seed) tuples in same order as all_preprocessed_datasets
    all_preprocessed_datasets = []

    for dataset_idx, (raw_ds, ds_name, needs_implicit) in enumerate([
        (ml100k_raw, "MovieLens100K", True),
        (amazon_vg_raw, "Amazon2014VideoGames", True),
        (lastfm_raw, "HetrecLastFM", False),
    ]):
        for seed_idx, seed in enumerate(seeds):
            # Set the global random state before ANY preprocessing for this seed
            set_random_state(seed)

            # Build the pipeline
            steps = []
            if needs_implicit:
                steps.append(MakeImplicit(3))
            steps.append(CorePruning(5))
            # UserHoldout: 80% train, 10% validation, 10% test
            # NOTE: validation_size MUST be > 0.0 or UserHoldout crashes because
            # internally it computes test_size = valid_size / (1 - test_size)
            # which becomes 0.0 when valid_size = 0.0, and sklearn rejects this.
            steps.append(UserHoldout(validation_size=0.1, test_size=0.1))

            pipeline = Pipe(*steps)
            processed = pipeline.process(raw_ds)

            # Store metadata about this dataset
            dataset_meta.append((ds_name, seed))
            all_preprocessed_datasets.append(processed)

            # Print stats using SplitData's public get() method
            train_size = len(processed._data.get("train"))
            val_size = len(processed._data.get("val"))
            test_size = len(processed._data.get("test"))

            print(f"  [{ds_name}] seed={seed}: "
                  f"train={train_size}, val={val_size}, test={test_size}")

    print(f"\n  Total preprocessed datasets: {len(all_preprocessed_datasets)}")

    # =========================================================================
    # 4. CREATE EXPERIMENT PLAN with default hyperparameters and feedback=implicit
    # =========================================================================
    print("\n" + "=" * 70)
    print("Creating experiment plan...")
    print("=" * 70)

    plan = ExperimentPlan(plan_name="Seed-Analysis")

    # ALS (Implicit MF) - implicit feedback matrix factorization
    plan.add_algorithm(
        LensKit.ImplicitMFScorer,
        {"feedback": "implicit"}
    )

    # ItemKNN - item-based k-nearest neighbors
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {"feedback": "implicit"}
    )

    # Pop - popularity-based baseline
    plan.add_algorithm(
        LensKit.PopScorer,
        {"feedback": "implicit"}
    )

    print("  Algorithms added: ImplicitMFScorer (ALS), ItemKNNScorer, PopScorer")

    # =========================================================================
    # 5. SET UP EVALUATION METRICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("Setting up evaluation metrics...")
    print("=" * 70)

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10])
    )

    print("  Metrics: NDCG@[1,5,10], Precision@[1,5,10]")

    # =========================================================================
    # 6. RUN ALL EXPERIMENTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("Running experiments (3 datasets x 3 algorithms x 5 seeds = 45 runs)...")
    print("=" * 70)

    run_omnirec(
        datasets=all_preprocessed_datasets,
        plan=plan,
        evaluator=evaluator
    )

    # =========================================================================
    # 7. COLLECT AND AGGREGATE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("Collecting results...")
    print("=" * 70)

    results_dict = evaluator.get_results()

    # Build a unified DataFrame with columns:
    # dataset_name, seed, algorithm, metric, k, value
    # The results_dict keys are dataset identifier strings (name + hash).
    # The datasets are processed in the same order as all_preprocessed_datasets,
    # so we iterate in insertion order (Python 3.7+ preserves dict insertion order).
    all_rows = []

    for idx, (dataset_id, df) in enumerate(results_dict.items()):
        # dataset_meta list is in the same order as we passed datasets to run_omnirec
        ds_name, actual_seed = dataset_meta[idx]

        for _, row in df.iterrows():
            all_rows.append({
                "dataset": ds_name,
                "seed": actual_seed,
                "algorithm": row["algorithm"],
                "metric": row["name"],
                "k": row["k"],
                "value": row["value"]
            })

    results_df = pd.DataFrame(all_rows)

    # =========================================================================
    # 8. STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS & STATISTICAL ANALYSIS")
    print("=" * 70)

    # Print overall results table
    pivot_results = results_df.pivot_table(
        index=["dataset", "seed", "algorithm"],
        columns=["metric", "k"],
        values="value",
        aggfunc="first"
    ).round(5)

    print("\nFull Results (dataset x seed x algorithm):")
    print(pivot_results.to_string())
    print()

    # Compute per-algorithm, per-dataset, per-metric statistics across seeds
    print("-" * 70)
    print("VARIABILITY ANALYSIS: Mean +/- Std across 5 seeds")
    print("-" * 70)

    stats_dfs = []
    for ds in results_df["dataset"].unique():
        for algo in results_df["algorithm"].unique():
            mask = (results_df["dataset"] == ds) & (results_df["algorithm"] == algo)
            group = results_df.loc[mask]
            metrics_stats = group.groupby(["metric", "k"])["value"].agg(["mean", "std", "min", "max"])
            metrics_stats = metrics_stats.round(5)
            metrics_stats["dataset"] = ds
            metrics_stats["algorithm"] = algo
            stats_dfs.append(metrics_stats.reset_index())

    if stats_dfs:
        stats_df = pd.concat(stats_dfs, ignore_index=True)
        stats_df = stats_df.set_index(["dataset", "algorithm", "metric", "k"])
        print(stats_df.to_string())
        print()

    # Coefficient of Variation (CV = std/mean) to measure relative variability
    print("-" * 70)
    print("COEFFICIENT OF VARIATION (CV = std/mean) * 100%")
    print("Measures how much seed variation affects results (lower = more stable)")
    print("-" * 70)

    cv_data = []
    for ds in results_df["dataset"].unique():
        for algo in results_df["algorithm"].unique():
            mask = (results_df["dataset"] == ds) & (results_df["algorithm"] == algo)
            group = results_df.loc[mask]
            for metric in group["metric"].unique():
                for k in group["k"].unique():
                    mask_k = (group["metric"] == metric) & (group["k"] == k)
                    sub = group.loc[mask_k]
                    values = sub["value"].values
                    mean_v = np.mean(values)
                    std_v = np.std(values, ddof=1)  # sample std
                    cv = (std_v / mean_v * 100) if mean_v != 0.0 else 0.0
                    cv_data.append({
                        "dataset": ds,
                        "algorithm": algo,
                        "metric": f"{metric}@{k}",
                        "cv_percent": round(cv, 3)
                    })

    cv_df = pd.DataFrame(cv_data)
    cv_pivot = cv_df.pivot_table(
        index=["dataset", "algorithm"],
        columns="metric",
        values="cv_percent"
    )
    print(cv_pivot.to_string())
    print()

    # Summary statistics
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    # Average CV across all metrics per algorithm
    print("\nAverage CV% across all metrics by algorithm:")
    algo_cv = cv_df.groupby("algorithm")["cv_percent"].agg(["mean", "std"]).round(3)
    print(algo_cv.to_string())

    print("\nAverage CV% across all metrics by dataset:")
    ds_cv = cv_df.groupby("dataset")["cv_percent"].agg(["mean", "std"]).round(3)
    print(ds_cv.to_string())

    print("\nKey Findings:")
    for algo in cv_df["algorithm"].unique():
        algo_cv_mean = cv_df[cv_df["algorithm"] == algo]["cv_percent"].mean()
        print(f"  - {algo}: Average CV = {algo_cv_mean:.2f}%")
        if algo_cv_mean < 2:
            print(f"      -> Highly stable across random seeds")
        elif algo_cv_mean < 10:
            print(f"      -> Moderately sensitive to random seeds")
        else:
            print(f"      -> Highly sensitive to random seeds")

    # Save results
    results_path = os.path.join(working_dir, "seed_analysis_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
