#!/usr/bin/env python3
"""
Quantify the impact of data split random seeds on recommender system accuracy.
Uses OmniRec exclusively (wraps LensKit) with three algorithms, three datasets, and 5 seeds.
"""

import os
import sys
import warnings
from typing import cast

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ── OmniRec imports ──────────────────────────────────────────────────────────
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan, AlgorithmConfig
from omnirec.runner.algos import LensKit, Algorithms
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

# ── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 999]

# Dataset names mapping to DataSet enum values
# We use the built-in loaders which auto-download from the web
DATASET_CONFIGS = [
    {
        'name': 'MovieLens100K',
        'dataset_enum': DataSet.MovieLens100K,
        'make_implicit': True,   # Convert explicit ratings > 3 to implicit
    },
    {
        'name': 'Amazon2014VideoGames',
        'dataset_enum': DataSet.Amazon2014VideoGames,
        'make_implicit': True,   # Convert explicit ratings > 3 to implicit
    },
    {
        'name': 'HetrecLastFM',
        'dataset_enum': DataSet.HetrecLastFM,
        'make_implicit': False,  # Already implicit (play counts)
    },
]

# Algorithm configurations with default/standard hyperparameters
ALGORITHM_CONFIGS = [
    {
        'name': 'Pop',
        'algo': LensKit.PopScorer,
        'params': {},  # PopScorer uses default parameters
    },
    {
        'name': 'ItemKNN',
        'algo': LensKit.ItemKNNScorer,
        'params': {
            'max_nbrs': 20,
            'min_nbrs': 1,
            'feedback': 'implicit',
        },
    },
    {
        'name': 'ALS',
        'algo': LensKit.ImplicitMFScorer,
        'params': {
            'features': 20,
            'epochs': 10,
        },
    },
]

# Working directory
WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)


def main():
    # We'll collect results across all seeds, datasets, and algorithms
    all_records: list[dict] = []

    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"Running with random seed = {seed}")
        print(f"{'='*80}")

        # Set the global random state for reproducibility
        set_random_state(seed)

        for ds_cfg in DATASET_CONFIGS:
            ds_name = ds_cfg['name']
            print(f"\n  ── Dataset: {ds_name} ──")

            # 1) Load raw dataset (auto-downloads if not already present)
            try:
                print(f"     Loading {ds_name} ...")
                dataset = RecSysDataSet.use_dataloader(
                    cast(DataSet, ds_cfg['dataset_enum'])
                )
            except Exception as e:
                print(f"     ERROR: Could not load dataset {ds_name}: {e}")
                # Record NaN for all algorithm/metric/k combinations
                for algo_cfg in ALGORITHM_CONFIGS:
                    for k in [1, 5, 10]:
                        for metric_name in ['NDCG', 'Precision']:
                            all_records.append({
                                'seed': seed,
                                'dataset': ds_name,
                                'algorithm': algo_cfg['name'],
                                'metric': metric_name,
                                'k': k,
                                'value': float('nan'),
                            })
                continue

            # 2) Build preprocessing pipeline
            pipe_steps = []

            if ds_cfg['make_implicit']:
                # For MovieLens and Amazon: ratings > 3 → implicit
                # MakeImplicit(4) keeps ratings >= 4 (same as rating > 3 for ints)
                print(f"     Converting explicit ratings > 3 to implicit ...")
                pipe_steps.append(MakeImplicit(4))

            # 5-core filtering for all datasets
            print(f"     Applying 5-core filter ...")
            pipe_steps.append(CorePruning(5))

            # User-based 80/20 holdout split
            # UserHoldout(validation_size, test_size): 0% validation, 20% test
            print(f"     Splitting 80/20 per user (seed={seed}) ...")
            pipe_steps.append(UserHoldout(0.0, 0.2))

            # Apply the preprocessing pipeline
            pipeline = Pipe(*pipe_steps)
            try:
                dataset = pipeline.process(dataset)
            except Exception as e:
                print(f"     ERROR: Preprocessing failed: {e}")
                for algo_cfg in ALGORITHM_CONFIGS:
                    for k in [1, 5, 10]:
                        for metric_name in ['NDCG', 'Precision']:
                            all_records.append({
                                'seed': seed,
                                'dataset': ds_name,
                                'algorithm': algo_cfg['name'],
                                'metric': metric_name,
                                'k': k,
                                'value': float('nan'),
                            })
                continue

            # Print split statistics
            train_df = dataset._data.get('train')
            val_df = dataset._data.get('val')
            test_df = dataset._data.get('test')
            print(f"     Train: {len(train_df)} interactions, "
                  f"Val: {len(val_df)} interactions, "
                  f"Test: {len(test_df)} interactions")

            # 3) Create experiment plan with all three algorithms
            plan = ExperimentPlan(plan_name=f"SeedImpact_{ds_name}_seed{seed}")

            for algo_cfg in ALGORITHM_CONFIGS:
                print(f"     Adding algorithm: {algo_cfg['name']} ...")
                plan.add_algorithm(
                    cast(Algorithms, algo_cfg['algo']),
                    cast(AlgorithmConfig, algo_cfg['params']),
                )

            # 4) Create evaluator with NDCG@k and Precision@k for k=1,5,10
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )

            # 5) Run all experiments
            print(f"     Running experiments ...")
            try:
                run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
            except Exception as e:
                print(f"     ERROR: Experiment failed: {e}")
                for algo_cfg in ALGORITHM_CONFIGS:
                    for k in [1, 5, 10]:
                        for metric_name in ['NDCG', 'Precision']:
                            all_records.append({
                                'seed': seed,
                                'dataset': ds_name,
                                'algorithm': algo_cfg['name'],
                                'metric': metric_name,
                                'k': k,
                                'value': float('nan'),
                            })
                continue

            # 6) Collect results
            results_dict = evaluator.get_results()
            for dataset_id, results_df in results_dict.items():
                # The results DataFrame has columns:
                # algorithm, fold, name, k, value
                for _, row in results_df.iterrows():
                    algo_full = row['algorithm']
                    # Extract short algorithm name
                    algo_short = algo_full
                    for a_cfg in ALGORITHM_CONFIGS:
                        algo_enum = cast(LensKit, a_cfg['algo'])
                        if algo_enum.value in algo_full or a_cfg['name'] in algo_full:
                            algo_short = a_cfg['name']
                            break

                    all_records.append({
                        'seed': seed,
                        'dataset': ds_name,
                        'algorithm': algo_short,
                        'metric': row['name'],
                        'k': int(row['k']) if pd.notna(row['k']) else -1,
                        'value': float(row['value']),
                    })

            print(f"     \u2713 Done")

    # ── Consolidate all results ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Consolidating results ...")
    print(f"{'='*80}")

    if all_records:
        results_df = pd.DataFrame(all_records)
    else:
        print("ERROR: No results were collected!")
        sys.exit(1)

    # ── Compute aggregate statistics ────────────────────────────────────────
    print("\nComputing aggregate statistics (mean \u00b1 std across 5 seeds) ...\n")

    # Group by dataset, algorithm, metric, k
    grouped = results_df.groupby(["dataset", "algorithm", "metric", "k"])["value"]

    stats = grouped.agg(["mean", "std"]).reset_index()

    # Print summary table
    print(f"{'Dataset':<22} {'Algorithm':<10} {'Metric':<12} {'k':<4} {'Mean':<10} {'Std':<10}")
    print("-" * 70)
    for _, row in stats.iterrows():
        if pd.notna(row['mean']) and pd.notna(row['std']):
            print(f"{row['dataset']:<22} {row['algorithm']:<10} {row['metric']:<12} {row['k']:<4} {row['mean']:<10.6f} {row['std']:<10.6f}")
        else:
            print(f"{row['dataset']:<22} {row['algorithm']:<10} {row['metric']:<12} {row['k']:<4} {'N/A':<10} {'N/A':<10}")

    # ── Statistical analysis ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Statistical Analysis: Impact of Seed Variation")
    print(f"{'='*80}")

    # Coefficient of Variation (CV) = std / mean (as a measure of seed sensitivity)
    valid_stats = stats.dropna(subset=["mean", "std"])
    valid_stats = valid_stats[valid_stats["mean"] > 0].copy()
    valid_stats["cv"] = valid_stats["std"] / valid_stats["mean"]

    print("\nCoefficient of Variation (std/mean) \u2014 higher = more sensitive to seed:\n")
    print(f"{'Dataset':<22} {'Algorithm':<10} {'Metric':<12} {'k':<4} {'CV':<12}")
    print("-" * 60)
    for _, row in valid_stats.iterrows():
        cv_str = f"{row['cv']:.4f}" if pd.notna(row.get('cv')) else "N/A"
        print(f"{row['dataset']:<22} {row['algorithm']:<10} {row['metric']:<12} {row['k']:<4} {cv_str:<12}")

    # Identify most and least sensitive combinations
    if len(valid_stats) > 0:
        most_sensitive = valid_stats.loc[valid_stats["cv"].idxmax()]
        least_sensitive = valid_stats.loc[valid_stats["cv"].idxmin()]

        print(f"\nMost sensitive to seed variation:")
        print(f"  Dataset={most_sensitive['dataset']}, Algorithm={most_sensitive['algorithm']}, "
              f"Metric={most_sensitive['metric']}@k={most_sensitive['k']}, "
              f"Mean={most_sensitive['mean']:.6f}, Std={most_sensitive['std']:.6f}, CV={most_sensitive['cv']:.4f}")

        print(f"\nLeast sensitive to seed variation:")
        print(f"  Dataset={least_sensitive['dataset']}, Algorithm={least_sensitive['algorithm']}, "
              f"Metric={least_sensitive['metric']}@k={least_sensitive['k']}, "
              f"Mean={least_sensitive['mean']:.6f}, Std={least_sensitive['std']:.6f}, CV={least_sensitive['cv']:.4f}")

    # Per-algorithm average CV
    print(f"\nAverage seed sensitivity (CV) per algorithm across all datasets/metrics/k:")
    if len(valid_stats) > 0:
        algo_cv = valid_stats.groupby("algorithm")["cv"].mean().sort_values(ascending=False)
        for algo, cv in algo_cv.items():
            print(f"  {algo:<30} {cv:.4f}")

    # Per-dataset average CV
    print(f"\nAverage seed sensitivity (CV) per dataset across all algorithms/metrics/k:")
    if len(valid_stats) > 0:
        ds_cv = valid_stats.groupby("dataset")["cv"].mean().sort_values(ascending=False)
        for ds_name_display, cv in ds_cv.items():
            print(f"  {ds_name_display:<30} {cv:.4f}")

    # ── Save results ────────────────────────────────────────────────────────
    output_path = os.path.join(WORKING_DIR, "seed_variation_results.csv")
    stats.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    # Also save per-seed raw results
    raw_path = os.path.join(WORKING_DIR, "seed_variation_raw_results.csv")
    results_df.to_csv(raw_path, index=False)
    print(f"Raw per-seed results saved to: {raw_path}")

    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
