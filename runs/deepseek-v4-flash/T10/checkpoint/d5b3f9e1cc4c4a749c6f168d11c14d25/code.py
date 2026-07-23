#!/usr/bin/env python3
"""
Quantify the impact of data split random seeds on recommender system accuracy.
Uses OmniRec (wrapping LensKit) with three algorithms, three datasets, and 5 seeds.
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
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

# ── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 999]

DATASET_CONFIGS = {
    "MovieLens100K": {
        "enum": DataSet.MovieLens100K,
        "make_implicit": True,      # explicit → implicit (ratings > 3)
    },
    "Amazon2014VideoGames": {
        "enum": DataSet.Amazon2014VideoGames,
        "make_implicit": True,      # explicit → implicit (ratings > 3)
    },
    "HetrecLastFM": {
        "enum": DataSet.HetrecLastFM,
        "make_implicit": False,     # already implicit
    },
}

ALGORITHM_CONFIGS = {
    LensKit.PopScorer: {},
    LensKit.ItemKNNScorer: {"max_nbrs": 20, "min_nbrs": 1},
    LensKit.ImplicitMFScorer: {"features": 20, "epochs": 10},
}

# Working directory
WORKING_DIR = os.path.join(os.getcwd(), "working")
os.makedirs(WORKING_DIR, exist_ok=True)
os.chdir(WORKING_DIR)


def build_preprocessing_pipeline(make_implicit: bool):
    """Build the preprocessing Pipe for a dataset."""
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))   # ratings >= 3 → implicit
    steps.append(CorePruning(5))        # 5-core filter
    steps.append(UserHoldout(0.0, 0.2)) # 80/20 user-based holdout
    return Pipe(*steps)


def main():
    # We'll collect results across all seeds, datasets, and algorithms
    all_records = []

    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"Running with random seed = {seed}")
        print(f"{'='*80}")

        # Set the global random state for this seed
        set_random_state(seed)

        for ds_name, ds_cfg in DATASET_CONFIGS.items():
            print(f"\n  ── Dataset: {ds_name} ──")

            # 1) Load raw dataset
            print(f"     Loading dataset ...")
            dataset_enum = cast(DataSet, ds_cfg["enum"])
            dataset = RecSysDataSet.use_dataloader(dataset_enum)

            # 2) Preprocess
            print(f"     Preprocessing (make_implicit={ds_cfg['make_implicit']}, 5-core, 80/20 holdout)...")
            make_implicit = cast(bool, ds_cfg["make_implicit"])
            pipeline = build_preprocessing_pipeline(make_implicit)
            dataset = pipeline.process(dataset)

            # 3) Create experiment plan with all three algorithms
            plan = ExperimentPlan(plan_name=f"{ds_name}_seed{seed}")
            for algo_enum, algo_params in ALGORITHM_CONFIGS.items():
                plan.add_algorithm(algo_enum, algo_params)

            # 4) Evaluation metrics: NDCG@[1,5,10] and Precision@[1,5,10]
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )

            # 5) Run experiments
            print(f"     Running experiments (3 algorithms)...")
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

            # 6) Collect results
            results_dict = evaluator.get_results()
            for dataset_id, df in results_dict.items():
                df = df.copy()
                df["seed"] = seed
                df["dataset"] = ds_name
                all_records.append(df)

            print(f"     \u2713 Done with {ds_name} seed={seed}")

    # ── Consolidate all results ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Consolidating results ...")
    print(f"{'='*80}")

    if all_records:
        results_df = pd.concat(all_records, ignore_index=True)
    else:
        print("ERROR: No results were collected!")
        sys.exit(1)

    # ── Compute aggregate statistics ────────────────────────────────────────
    print("\nComputing aggregate statistics (mean \u00b1 std across 5 seeds) ...\n")

    # Group by dataset, algorithm, metric, k
    grouped = results_df.groupby(["dataset", "algorithm", "name", "k"])["value"]

    stats = grouped.agg(["mean", "std"]).reset_index()
    stats.columns = ["dataset", "algorithm", "metric", "k", "mean", "std"]

    # Print summary table
    print(f"{'Dataset':<22} {'Algorithm':<22} {'Metric':<12} {'k':<4} {'Mean':<10} {'Std':<10}")
    print("-" * 80)
    for _, row in stats.iterrows():
        print(f"{row['dataset']:<22} {row['algorithm']:<22} {row['metric']:<12} {row['k']:<4} {row['mean']:<10.6f} {row['std']:<10.6f}")

    # ── Statistical analysis ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Statistical Analysis: Impact of Seed Variation")
    print(f"{'='*80}")

    # Coefficient of Variation (CV) = std / mean  (as a measure of seed sensitivity)
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)

    print("\nCoefficient of Variation (std/mean) \u2014 higher = more sensitive to seed:\n")
    print(f"{'Dataset':<22} {'Algorithm':<22} {'Metric':<12} {'k':<4} {'CV':<12}")
    print("-" * 72)
    for _, row in stats.iterrows():
        cv_str = f"{row['cv']:.4f}" if pd.notna(row['cv']) else "N/A"
        print(f"{row['dataset']:<22} {row['algorithm']:<22} {row['metric']:<12} {row['k']:<4} {cv_str:<12}")

    # Identify most and least sensitive combinations
    valid_cv = stats.dropna(subset=["cv"])
    if len(valid_cv) > 0:
        most_sensitive = valid_cv.loc[valid_cv["cv"].idxmax()]
        least_sensitive = valid_cv.loc[valid_cv["cv"].idxmin()]

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
    algo_cv = valid_cv.groupby("algorithm")["cv"].mean().sort_values(ascending=False)
    for algo, cv in algo_cv.items():
        print(f"  {algo:<30} {cv:.4f}")

    # Per-dataset average CV
    print(f"\nAverage seed sensitivity (CV) per dataset across all algorithms/metrics/k:")
    ds_cv = valid_cv.groupby("dataset")["cv"].mean().sort_values(ascending=False)
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
