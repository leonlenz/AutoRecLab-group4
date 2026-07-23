#!/usr/bin/env python3
"""
Quantify the impact of data split random seeds on recommender system accuracy.
Uses OmniRec to test three algorithms: ALS (ImplicitMFScorer), ItemKNN, and Pop.
Three datasets: MovieLens100K, Amazon2014VideoGames, Last.FM (HetrecLastFM).
Preprocessing: 5-core filtering, implicit conversion (ratings > 3) for non-LastFM.
User-based 80/20 holdout split. 5 random seeds. Metrics: nDCG@k and Precision@k for k=1,5,10.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import cast

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ── OmniRec imports ─────────────────────────────────────────────────────────
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
from omnirec.runner.coordinator import Coordinator
from omnirec.util.util import set_random_state


# ── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 999]

DATASET_CONFIGS = [
    {
        'name': 'MovieLens100K',
        'enum': DataSet.MovieLens100K,
        'make_implicit': True,
    },
    {
        'name': 'Amazon2014VideoGames',
        'enum': DataSet.Amazon2014VideoGames,
        'make_implicit': True,
    },
    {
        'name': 'HetrecLastFM',
        'enum': DataSet.HetrecLastFM,
        'make_implicit': False,
    },
]

# Working directory
WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)


def preprocess_dataset(ds_name: str, ds_enum, make_implicit: bool) -> RecSysDataSet:
    """Load a dataset via OmniRec's built-in loader and apply preprocessing.

    For MovieLens100K and Amazon2014VideoGames:
        - MakeImplicit(4): keep ratings > 3 (i.e., ratings >= 4)
        - CorePruning(5): 5-core filtering
        - UserHoldout(0, 0.2): user-based 80/20 split

    For HetrecLastFM (already implicit):
        - CorePruning(5): 5-core filtering
        - UserHoldout(0, 0.2): user-based 80/20 split
    """
    print(f"     Loading dataset '{ds_name}' via OmniRec (auto-download)...")
    dataset = RecSysDataSet.use_dataloader(ds_enum)

    if make_implicit:
        print(f"     Converting explicit ratings to implicit (threshold >= 4)...")
        pipeline = Pipe(
            MakeImplicit(4),       # ratings > 3 means >= 4
            CorePruning(5),        # 5-core filtering
            UserHoldout(0, 0.2),   # user-based 80/20 split (no validation set)
        )
    else:
        print(f"     Dataset is already implicit, skipping MakeImplicit...")
        pipeline = Pipe(
            CorePruning(5),        # 5-core filtering
            UserHoldout(0, 0.2),   # user-based 80/20 split (no validation set)
        )

    dataset = pipeline.process(dataset)

    # Print statistics
    n_train = len(dataset._data.get('train'))
    n_test = len(dataset._data.get('test'))
    print(f"     Train interactions: {n_train}, Test interactions: {n_test}")

    return dataset


def create_experiment_plan() -> ExperimentPlan:
    """Create an experiment plan with all three algorithms using default hyperparameters.

    - PopScorer: default parameters (PopConfig uses score='quantile')
    - ItemKNNScorer: default parameters (max_nbrs=20, min_nbrs=1, feedback auto-detected)
    - ImplicitMFScorer: features=20, epochs=10
    """
    plan = ExperimentPlan("SeedSensitivityStudy")

    # PopScorer with default parameters
    plan.add_algorithm(LensKit.PopScorer, {})

    # ItemKNNScorer with default parameters
    # The LensKit runner auto-detects feedback type from data, so no need to specify it
    plan.add_algorithm(LensKit.ItemKNNScorer, {})

    # ImplicitMFScorer with features=20, epochs=10
    plan.add_algorithm(LensKit.ImplicitMFScorer, {
        'features': 20,
        'epochs': 10,
    })

    return plan


def main():
    # We'll collect results across all seeds, datasets, and algorithms
    all_records: list[dict] = []

    # Pre-create experiment plan (same algorithms across all seeds)
    plan = create_experiment_plan()

    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"Running with random seed = {seed}")
        print(f"{'='*80}")

        # Set the global random state for this seed (affects splitting)
        set_random_state(seed)

        # Create a per-seed checkpoint directory to avoid caching collisions
        seed_checkpoint_dir = os.path.join(WORKING_DIR, f'checkpoints_seed_{seed}')
        os.makedirs(seed_checkpoint_dir, exist_ok=True)

        for ds_cfg in DATASET_CONFIGS:
            # Explicit type annotations to help the type checker narrow union types
            ds_name: str = cast(str, ds_cfg['name'])
            ds_enum = ds_cfg['enum']
            make_implicit: bool = cast(bool, ds_cfg['make_implicit'])

            print(f"\n  ── Dataset: {ds_name} ──")

            # 1) Load and preprocess the dataset
            dataset = preprocess_dataset(ds_name, ds_enum, make_implicit)

            # 2) Create evaluator with metrics for k=1, 5, 10
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )

            # 3) Run experiments via Coordinator
            print(f"     Running algorithms via Coordinator...")
            coordinator = Coordinator(checkpoint_dir=seed_checkpoint_dir)
            coordinator.run(dataset, plan, evaluator)

            # 4) Extract results from evaluator
            results_dict = evaluator.get_results()
            # The results dict keys are dataset identifiers like "MovieLens100K-<hash>"
            for ds_id, results_df in results_dict.items():
                for _, row in results_df.iterrows():
                    algo_full = row['algorithm']
                    # Extract algorithm name (first part before '-')
                    algo_name = algo_full.split('-')[0].split('.')[-1]
                    metric_name = row['name']
                    k_val = row['k']
                    metric_value = row['value']

                    all_records.append({
                        'seed': seed,
                        'dataset': ds_name,
                        'algorithm': algo_name,
                        'metric': metric_name,
                        'k': int(k_val) if pd.notna(k_val) else None,
                        'value': float(metric_value),
                    })

            print(f"     ✓ Completed {ds_name}")

    # ── Consolidate all results ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Consolidating results ...")
    print(f"{'='*80}")

    if not all_records:
        print("ERROR: No results were collected!")
        sys.exit(1)

    results_df = pd.DataFrame(all_records)
    print(f"Total records: {len(results_df)}")

    # ── Compute aggregate statistics ────────────────────────────────────────
    print("\nComputing aggregate statistics (mean ± std across 5 seeds) ...\n")

    # Group by dataset, algorithm, metric, k
    grouped = results_df.groupby(["dataset", "algorithm", "metric", "k"])["value"]

    stats = grouped.agg(["mean", "std"]).reset_index()

    # Print summary table
    print(f"{'Dataset':<22} {'Algorithm':<10} {'Metric':<12} {'k':<4} {'Mean':<10} {'Std':<10}")
    print("-" * 70)
    for _, row in stats.iterrows():
        print(f"{row['dataset']:<22} {row['algorithm']:<10} {row['metric']:<12} {row['k']:<4} {row['mean']:<10.6f} {row['std']:<10.6f}")

    # ── Statistical analysis ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Statistical Analysis: Impact of Seed Variation")
    print(f"{'='*80}")

    # Coefficient of Variation (CV) = std / mean (measure of seed sensitivity)
    valid_stats = stats.dropna(subset=["mean"])
    valid_stats = valid_stats[valid_stats["mean"] > 0].copy()
    valid_stats["cv"] = valid_stats["std"] / valid_stats["mean"]

    print("\nCoefficient of Variation (std/mean) — higher = more sensitive to seed:\n")
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
    algo_cv = valid_stats.groupby("algorithm")["cv"].mean().sort_values(ascending=False)
    for algo, cv in algo_cv.items():
        print(f"  {algo:<30} {cv:.4f}")

    # Per-dataset average CV
    print(f"\nAverage seed sensitivity (CV) per dataset across all algorithms/metrics/k:")
    ds_cv = valid_stats.groupby("dataset")["cv"].mean().sort_values(ascending=False)
    for ds_name_display, cv in ds_cv.items():
        print(f"  {ds_name_display:<30} {cv:.4f}")

    # ── Save results ────────────────────────────────────────────────────────
    output_path = os.path.join(WORKING_DIR, "seed_variation_results.csv")
    stats.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    raw_path = os.path.join(WORKING_DIR, "seed_variation_raw_results.csv")
    results_df.to_csv(raw_path, index=False)
    print(f"Raw per-seed results saved to: {raw_path}")

    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
