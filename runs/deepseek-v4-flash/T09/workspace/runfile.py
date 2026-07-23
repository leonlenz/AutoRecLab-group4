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
import shutil
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
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def main():
    # Setup working directory
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    # Define random seeds
    seeds = [42, 123, 456, 789, 1111]

    # Define algorithm identifiers (use enum values)
    algo_ids = [
        LensKit.PopScorer,
        LensKit.ItemKNNScorer,
        LensKit.ImplicitMFScorer,
    ]

    # Store results per (dataset_key, algo_name, metric, k) across seeds
    # Key: (dataset_key, algo_name, metric_name, k) -> list of values (one per seed)
    all_results = defaultdict(list)

    # Loop over seeds
    for seed_idx, seed in enumerate(seeds):
        print("\n" + "=" * 72)
        print(f"Seed {seed_idx + 1}/{len(seeds)}: seed = {seed}")
        print("=" * 72)

        # Set global random state for reproducibility
        set_random_state(seed)

        # Create a fresh evaluator for this seed to prevent result accumulation
        evaluator = Evaluator(
            NDCG([1, 5, 10]),
            Precision([1, 5, 10]),
        )

        # ---------------------------------------------------------------
        # Preprocess MovieLens100K: MakeImplicit(4) + CorePruning(5) + UserHoldout
        # UserHoldout(validation_size=0.0, test_size=0.2) for 80/20 split
        # MakeImplicit(4) converts ratings > 3 to implicit feedback (ratings >= 4)
        # ---------------------------------------------------------------
        print("  Preprocessing MovieLens100K...")
        ml100k_raw = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        pipeline_ml = Pipe(
            MakeImplicit(4),                         # ratings > 3 -> implicit (>=4)
            CorePruning(5),                          # 5-core filtering
            UserHoldout(validation_size=0.0, test_size=0.2),  # 80/20 train/test split
        )
        ml100k_split = pipeline_ml.process(ml100k_raw)

        # ---------------------------------------------------------------
        # Preprocess Amazon2014VideoGames: MakeImplicit(4) + CorePruning(5) + UserHoldout
        # No Subsample - use full dataset (after core pruning)
        # ---------------------------------------------------------------
        print("  Preprocessing Amazon2014VideoGames...")
        amazon_raw = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
        pipeline_amz = Pipe(
            MakeImplicit(4),                         # ratings > 3 -> implicit (>=4)
            CorePruning(5),                          # 5-core filtering
            UserHoldout(validation_size=0.0, test_size=0.2),  # 80/20 train/test split
        )
        amazon_split = pipeline_amz.process(amazon_raw)

        # ---------------------------------------------------------------
        # Preprocess HetrecLastFM: CorePruning(5) + UserHoldout (already implicit)
        # ---------------------------------------------------------------
        print("  Preprocessing HetrecLastFM...")
        lastfm_raw = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
        pipeline_lfm = Pipe(
            CorePruning(5),                          # 5-core filtering
            UserHoldout(validation_size=0.0, test_size=0.2),  # 80/20 train/test split
        )
        lastfm_split = pipeline_lfm.process(lastfm_raw)

        # ---------------------------------------------------------------
        # Create a fresh ExperimentPlan inside the seed loop.
        # This ensures that the plan name is unique per seed to avoid
        # checkpoint collisions, AND that each seed produces a unique
        # config hash (since the random state is now different).
        # ---------------------------------------------------------------
        plan = ExperimentPlan(f"Seed-Sensitivity-Study-seed{seed}")

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

        # ---------------------------------------------------------------
        # Remove old checkpoints to prevent caching across seeds
        # (Each seed must re-run all phases to get different splits)
        # ---------------------------------------------------------------
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

        # ---------------------------------------------------------------
        # Run all algorithms on all three datasets
        # ---------------------------------------------------------------
        datasets = [ml100k_split, amazon_split, lastfm_split]

        print("  Running experiments...")
        run_omnirec(datasets=datasets, plan=plan, evaluator=evaluator)

        # Collect results for this seed
        # The algorithm column in evaluator results has the format:
        #   "<Runner.Algo>-<hash8>-<random_state>"
        # We strip the random_state suffix to group by logical algorithm name.
        results = evaluator.get_results()
        for dataset_key, df in results.items():
            for _, row in df.iterrows():
                algo_full = str(row["algorithm"])
                # Remove the trailing random-state suffix: "-<number>"
                # The format is: <AlgoName>-<hash8>-<randstate>
                # We want to group by everything except the last "-<randstate>"
                # e.g. "LensKit.PopScorer-dc44ee64-1" -> "LensKit.PopScorer-dc44ee64"
                algo_parts = algo_full.rsplit("-", 1)
                if len(algo_parts) == 2 and algo_parts[1].isdigit():
                    algo_base = algo_parts[0]
                else:
                    algo_base = algo_full
                metric_name = row["name"]
                k_val = row["k"]
                value = row["value"]
                result_key = (dataset_key, algo_base, metric_name, k_val)
                all_results[result_key].append(value)

        print(f"  Completed seed {seed}")

    # ---------------------------------------------------------------
    # Aggregate results across seeds: report mean and std
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL AGGREGATED RESULTS (mean +/- std across 5 seeds)")
    print("=" * 72)

    # Summarize by dataset, algorithm, metric
    summary_rows = []
    for (dataset_key, algo_base, metric_name, k_val), values in sorted(all_results.items()):
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # sample std
        summary_rows.append({
            "dataset": dataset_key,
            "algorithm": algo_base,
            "metric": f"{metric_name}@{k_val}",
            "mean": mean_val,
            "std": std_val,
            "n": len(values),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Print per-dataset tables
    for ds in sorted(summary_df["dataset"].unique()):
        ds_mask = summary_df["dataset"] == ds
        print(f"\n--- Dataset: {ds} ---")
        ds_df = summary_df[ds_mask].copy()
        for algo in sorted(ds_df["algorithm"].unique()):
            algo_mask = ds_df["algorithm"] == algo
            print(f"\n  Algorithm: {algo}")
            algo_df = ds_df[algo_mask].sort_values("metric")
            for _, row in algo_df.iterrows():
                print(f"    {row['metric']:>12s}: {row['mean']:.6f} +/- {row['std']:.6f}")

    # Also print a compact summary table
    print("\n\nCOMPACT SUMMARY TABLE")
    print("-" * 72)
    print(f"{'Dataset':<30s} {'Algorithm':<28s} {'Metric':<10s} {'Mean':<12s} {'Std':<12s} {'N':<6s}")
    print("-" * 72)

    summary_rows.sort(key=lambda r: (str(r["dataset"]), str(r["algorithm"]), r["metric"]))
    for row in summary_rows:
        ds_short = str(row["dataset"])[:30]
        algo_short = str(row["algorithm"])[:28]
        print(f"{ds_short:<30s} {algo_short:<28s} {row['metric']:<10s} {row['mean']:<12.6f} {row['std']:<12.6f} {row['n']:<6d}")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()