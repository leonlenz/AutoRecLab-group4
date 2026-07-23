#!/usr/bin/env python3
"""
Experiment: Quantifying the effect of data split random seeds on recommender system accuracy.

Tests 3 algorithms (Pop, ItemKNN, ImplicitMF/ALS) on 3 datasets
(MovieLens100K, Amazon2014VideoGames, HetrecLastFM) with 5 different seeds.
"""

import os
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

    # Define evaluator with ranking metrics
    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    # Load all three datasets once (raw, before preprocessing)
    print("=" * 72)
    print("Loading datasets...")
    print("=" * 72)

    ml100k_raw = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print(f"  MovieLens100K loaded: {ml100k_raw.num_interactions()} interactions")

    amazon_raw = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
    print(f"  Amazon2014VideoGames loaded: {amazon_raw.num_interactions()} interactions")

    lastfm_raw = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
    print(f"  HetrecLastFM loaded: {lastfm_raw.num_interactions()} interactions")

    # Store results per (dataset_name, algorithm_name, metric, k) across seeds
    all_results = defaultdict(list)

    # Loop over seeds
    for seed_idx, seed in enumerate(seeds):
        print("\n" + "=" * 72)
        print(f"Seed {seed_idx + 1}/{len(seeds)}: seed = {seed}")
        print("=" * 72)

        # Set global random state for reproducibility
        set_random_state(seed)

        # ---------------------------------------------------------------
        # Preprocess MovieLens100K: MakeImplicit(4) + CorePruning(5) + UserHoldout
        # ---------------------------------------------------------------
        print("  Preprocessing MovieLens100K...")
        # Re-load raw dataset for this seed to get a fresh copy
        ml100k = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        pipeline_ml = Pipe(
            MakeImplicit(4),                         # ratings > 3 -> implicit
            CorePruning(5),                          # 5-core filtering
            UserHoldout(validation_size=0.0, test_size=0.2),  # 80/20 user split
        )
        ml100k_split = pipeline_ml.process(ml100k)
        print(f"    Train: {len(ml100k_split._data.get('train'))} interactions")
        print(f"    Test:  {len(ml100k_split._data.get('test'))} interactions")

        # ---------------------------------------------------------------
        # Preprocess Amazon2014VideoGames: MakeImplicit(4) + CorePruning(5) + UserHoldout
        # ---------------------------------------------------------------
        print("  Preprocessing Amazon2014VideoGames...")
        amazon = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
        pipeline_amz = Pipe(
            MakeImplicit(4),                         # ratings > 3 -> implicit
            CorePruning(5),                          # 5-core filtering
            UserHoldout(validation_size=0.0, test_size=0.2),  # 80/20 user split
        )
        amazon_split = pipeline_amz.process(amazon)
        print(f"    Train: {len(amazon_split._data.get('train'))} interactions")
        print(f"    Test:  {len(amazon_split._data.get('test'))} interactions")

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
        print(f"    Train: {len(lastfm_split._data.get('train'))} interactions")
        print(f"    Test:  {len(lastfm_split._data.get('test'))} interactions")

        # ---------------------------------------------------------------
        # Run all algorithms on all three datasets
        # ---------------------------------------------------------------
        datasets = [ml100k_split, amazon_split, lastfm_split]

        print("  Running experiments...")
        run_omnirec(datasets=datasets, plan=plan, evaluator=evaluator)

        # Collect results for this seed
        results = evaluator.get_results()
        for dataset_key, df in results.items():
            for _, row in df.iterrows():
                algo = row["algorithm"]
                metric_name = row["name"]
                k_val = row["k"]
                value = row["value"]
                # Use a composite key
                result_key = (dataset_key, algo, metric_name, k_val)
                all_results[result_key].append(value)

    # ---------------------------------------------------------------
    # Aggregate results across seeds: report mean and std
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL AGGREGATED RESULTS (mean ± std across 5 seeds)")
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
                print(f"    {row['metric']:>12s}: {row['mean']:.6f} ± {row['std']:.6f}")

    # Also print a compact summary table
    print("\n\nCOMPACT SUMMARY TABLE")
    print("-" * 72)
    print(f"{'Dataset':<30s} {'Algorithm':<28s} {'Metric':<10s} {'Mean':<12s} {'Std':<12s}")
    print("-" * 72)

    # Sort for consistent display
    summary_rows.sort(key=lambda r: (r["dataset"], r["algorithm"], r["metric"]))
    for row in summary_rows:
        # Shorten dataset key to just the name part
        ds_short = row["dataset"][:30]
        algo_short = row["algorithm"][:28]
        print(f"{ds_short:<30s} {algo_short:<28s} {row['metric']:<10s} {row['mean']:<12.6f} {row['std']:<12.6f}")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
