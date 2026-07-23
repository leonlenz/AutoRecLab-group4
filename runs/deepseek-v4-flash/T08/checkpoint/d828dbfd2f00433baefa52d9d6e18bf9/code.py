#!/usr/bin/env python3
"""
Experiment: Quantifying the impact of data split random seeds on recommender system accuracy.

Runs ALS (ImplicitMF), ItemKNN, and Pop on MovieLens100K, Amazon2014VideoGames, 
and HetrecLastFM across 5 different random seeds for user-based 80/20 holdout splits.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ========== OmniRec imports ==========
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
from omnirec.util.util import set_random_state, get_random_state


WORKING_DIR = os.path.join(os.getcwd(), "working")
os.makedirs(WORKING_DIR, exist_ok=True)

# Change to working directory for checkpoint storage
os.chdir(WORKING_DIR)

# Seeds for reproducibility
RANDOM_SEEDS = [42, 123, 256, 789, 1024]

# Dataset definitions
DATASET_CONFIGS = {
    "MovieLens100K": {
        "dataset_enum": DataSet.MovieLens100K,
        "make_implicit": True,    # ratings > 3 -> implicit
        "implicit_threshold": 4,  # keep only rating >= 4 (rating > 3)
    },
    "Amazon2014VideoGames": {
        "dataset_enum": DataSet.Amazon2014VideoGames,
        "make_implicit": True,    # ratings > 3 -> implicit
        "implicit_threshold": 4,  # keep only rating >= 4 (rating > 3)
    },
    "HetrecLastFM": {
        "dataset_enum": DataSet.HetrecLastFM,
        "make_implicit": False,   # already implicit, no thresholding
        "implicit_threshold": None,
    },
}


def build_preprocessed_raw(ds_name: str, cfg: dict) -> RecSysDataSet:
    """Load and preprocess a dataset (without splitting).
    Returns a RawData RecSysDataSet with MakeImplicit (if needed) and CorePruning applied.
    """
    print(f"\n{'='*60}")
    print(f"Loading and preprocessing {ds_name}...")
    
    dataset = RecSysDataSet.use_dataloader(cfg["dataset_enum"])
    print(f"  Loaded: {dataset.num_interactions()} interactions")
    
    steps = []
    if cfg["make_implicit"]:
        # For rating > 3, we use threshold=4 (MakeImplicit keeps rating >= threshold)
        steps.append(MakeImplicit(cfg["implicit_threshold"]))
        print(f"  Converting to implicit (rating >= {cfg['implicit_threshold']})...")
    
    steps.append(CorePruning(5))
    print("  Applying 5-core pruning...")
    
    pipeline = Pipe(*steps)
    dataset = pipeline.process(dataset)
    print(f"  After preprocessing: {dataset.num_interactions()} interactions")
    
    return dataset


def run_seed_experiment(
    ds_name: str,
    raw_dataset: RecSysDataSet,
    seed: int,
    results_dir: Path,
):
    """For a given seed, split the data, create experiment plan, run, and collect results."""
    print(f"\n  --- Seed {seed} ---")
    
    # Set random state for reproducible split
    set_random_state(seed)
    
    # Apply user-based holdout: ~70/10/20 train/val/test split.
    # NOTE: UserHoldout requires a POSITIVE validation_size. Internally it computes
    # test_size = valid_size / (1 - test_size) for the second train_test_split call.
    # Using validation_size=0.0 leads to test_size=0.0 which sklearn rejects.
    # We use validation_size=0.1 to create a valid 3-way split. The validation set
    # is not used in evaluation -- run_omnirec evaluates on the test set only.
    splitter = UserHoldout(validation_size=0.1, test_size=0.2)
    split_dataset = splitter.process(raw_dataset)
    
    # Create the experiment plan with all three algorithms
    plan = ExperimentPlan(plan_name=f"{ds_name}_seed{seed}")
    
    # PopScorer (popularity baseline) - default hyperparams
    plan.add_algorithm(LensKit.PopScorer, {})
    
    # ItemKNNScorer with implicit feedback mode
    plan.add_algorithm(LensKit.ItemKNNScorer, {"feedback": "implicit"})
    
    # ImplicitMFScorer (ALS) - default hyperparams
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    
    # Create evaluator with ranking metrics
    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )
    
    # Run experiments
    print(f"    Running experiment for seed {seed}...")
    run_omnirec(
        datasets=split_dataset,
        plan=plan,
        evaluator=evaluator,
    )
    
    # Collect results
    results = evaluator.get_results()
    
    # Flatten results into a single DataFrame
    all_rows = []
    for dataset_id, df in results.items():
        for _, row in df.iterrows():
            all_rows.append({
                "dataset": ds_name,
                "seed": seed,
                "algorithm": row["algorithm"],
                "fold": row["fold"],
                "metric": row["name"],
                "k": row["k"],
                "value": row["value"],
            })
    
    result_df = pd.DataFrame(all_rows)
    
    # Save per-seed results
    seed_file = results_dir / f"{ds_name}_seed{seed}_results.csv"
    result_df.to_csv(seed_file, index=False)
    print(f"    Saved {len(result_df)} result rows to {seed_file}")
    
    return result_df


def main():
    print("=" * 60)
    print("EXPERIMENT: Impact of Data Split Random Seeds on RecSys Accuracy")
    print("=" * 60)
    print(f"Working directory: {WORKING_DIR}")
    print(f"Random seeds: {RANDOM_SEEDS}")
    
    results_dir = Path(WORKING_DIR) / "results"
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n{'#' * 60}")
        print(f"Processing dataset: {ds_name}")
        print(f"{'#' * 60}")
        
        # Step 1: Load and preprocess (without split) - do this once per dataset
        raw_dataset = build_preprocessed_raw(ds_name, cfg)
        
        # Step 2: For each seed, split and run
        for seed in RANDOM_SEEDS:
            result_df = run_seed_experiment(ds_name, raw_dataset, seed, results_dir)
            all_results.append(result_df)
    
    # ========== Combine all results ==========
    print(f"\n{'=' * 60}")
    print("Consolidating results...")
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(results_dir / "all_results.csv", index=False)
    print(f"Total results: {len(combined)} rows")
    
    # ========== Statistical Analysis ==========
    print(f"\n{'=' * 60}")
    print("STATISTICAL ANALYSIS: Mean and Std Dev Across Seeds")
    print(f"{'=' * 60}")
    
    # Group by dataset, algorithm, metric, k -> compute mean and std across seeds
    stats = (
        combined.groupby(["dataset", "algorithm", "metric", "k"])["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    
    stats_file = results_dir / "statistical_analysis.csv"
    stats.to_csv(stats_file, index=False)
    
    # Print formatted results
    for ds_name in DATASET_CONFIGS.keys():
        print(f"\n--- {ds_name} ---")
        ds_stats = stats[stats["dataset"] == ds_name]
        for algo in sorted(ds_stats["algorithm"].unique()):
            print(f"\n  Algorithm: {algo}")
            algo_stats = ds_stats[ds_stats["algorithm"] == algo]
            for _, row in algo_stats.iterrows():
                metric_name = f"{row['metric']}@{int(row['k'])}" if not pd.isna(row['k']) else row['metric']
                print(f"    {metric_name:15s}  mean={row['mean']:.4f}  std={row['std']:.4f}  "
                      f"min={row['min']:.4f}  max={row['max']:.4f}")
    
    # Summary: Coefficient of Variation (CV = std/mean) as a measure of seed sensitivity
    print(f"\n{'=' * 60}")
    print("SENSITIVITY ANALYSIS (Coefficient of Variation = std/mean)")
    print(f"{'=' * 60}")
    
    # Filter to exclude near-zero means to avoid division issues
    stats_safe = stats[stats["mean"] > 0.001].copy()
    stats_safe["cv"] = stats_safe["std"] / stats_safe["mean"]
    
    # Sort by CV to find most seed-sensitive configurations
    most_sensitive = stats_safe.sort_values("cv", ascending=False).head(10)
    print("\nTop 10 most seed-sensitive configurations:")
    for _, row in most_sensitive.iterrows():
        metric_name = f"{row['metric']}@{int(row['k'])}" if not pd.isna(row['k']) else row['metric']
        print(f"  {row['dataset']:25s} | {str(row['algorithm']):35s} | {metric_name:10s} | "
              f"CV={row['cv']:.4f} (mean={row['mean']:.4f}, std={row['std']:.4f})")
    
    # Per-algorithm average CV across all datasets and metrics
    print("\nAverage CV per algorithm (across all datasets & metrics):")
    algo_cv = stats_safe.groupby("algorithm")["cv"].mean().sort_values(ascending=False)
    for algo, cv in algo_cv.items():
        print(f"  {algo:35s}: avg CV = {cv:.4f}")
    
    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"All results saved to: {results_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
