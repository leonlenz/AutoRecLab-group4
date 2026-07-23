#!/usr/bin/env python3
"""
Experiment: Quantifying the Impact of Data Split Random Seeds on Recommender System Accuracy.

This script tests three algorithms (PopScorer, ItemKNNScorer, ImplicitMFScorer) on three datasets
(MovieLens100K, Amazon2014VideoGames, HetrecLastFM) across 5 different random seeds for data splitting.
Results are analyzed to measure variance caused by data split randomness.

Uses the OmniRec framework throughout.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# OmniRec imports
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

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1111]

# ---- Working directory ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- Collect all results ----
all_results = []

# ================ STEP 1: Load and preprocess each dataset (without split) ================

# Dataset 1: MovieLens 100K (explicit -> implicit with threshold 3)
print("=" * 70)
print("Loading MovieLens100K...")
ml_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
# Apply MakeImplicit(3) and CorePruning(5) - no random component
print("Preprocessing MovieLens100K (MakeImplicit + CorePruning)...")
ml_pipeline = Pipe(
    MakeImplicit(3),
    CorePruning(5)
)
ml_processed = ml_pipeline.process(ml_dataset)
print(f"MovieLens100K after preprocessing: {ml_processed.num_interactions()} interactions")

# Dataset 2: Amazon 2014 Video Games (explicit -> implicit with threshold 3)
print("Loading Amazon2014VideoGames...")
amz_dataset = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
print("Preprocessing Amazon2014VideoGames (MakeImplicit + CorePruning)...")
amz_pipeline = Pipe(
    MakeImplicit(3),
    CorePruning(5)
)
amz_processed = amz_pipeline.process(amz_dataset)
print(f"Amazon2014VideoGames after preprocessing: {amz_processed.num_interactions()} interactions")

# Dataset 3: Hetrec LastFM (already implicit, no rating thresholding)
print("Loading HetrecLastFM...")
lfm_dataset = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
print("Preprocessing HetrecLastFM (CorePruning only)...")
lfm_pipeline = Pipe(
    CorePruning(5)
)
lfm_processed = lfm_pipeline.process(lfm_dataset)
print(f"HetrecLastFM after preprocessing: {lfm_processed.num_interactions()} interactions")

# Dictionary of preprocessed datasets (without split)
processed_datasets = {
    "MovieLens100K": ml_processed,
    "Amazon2014VideoGames": amz_processed,
    "HetrecLastFM": lfm_processed,
}

# ================ STEP 2: For each seed, split and run experiments ================

for seed in SEEDS:
    print(f"\n{'=' * 70}")
    print(f"RUNNING EXPERIMENTS WITH SEED = {seed}")
    print(f"{'=' * 70}")

    for ds_name, ds_processed in processed_datasets.items():
        print(f"\n--- Processing {ds_name} with seed {seed} ---")

        # Set global random state BEFORE splitting
        set_random_state(seed)

        # Apply the user-based 80/20 holdout split (test_size=0.2, no validation set)
        split_pipeline = Pipe(
            UserHoldout(validation_size=0.0, test_size=0.2)
        )
        ds_split = split_pipeline.process(ds_processed)

        # Create experiment plan for this split
        plan = ExperimentPlan(plan_name=f"{ds_name}_seed_{seed}")

        # Add the three algorithms with standard hyperparameters
        # PopScorer: popularity baseline, default parameters
        plan.add_algorithm(LensKit.PopScorer, {})

        # ItemKNNScorer: configured for implicit feedback
        plan.add_algorithm(
            LensKit.ItemKNNScorer,
            {"feedback": "implicit"}
        )

        # ImplicitMFScorer: ALS for implicit feedback, default parameters
        plan.add_algorithm(LensKit.ImplicitMFScorer, {})

        # Set up evaluator with Precision@[1,5,10] and NDCG@[1,5,10]
        evaluator = Evaluator(
            Precision([1, 5, 10]),
            NDCG([1, 5, 10])
        )

        # Run experiments
        print(f"  Running OmniRec experiments for {ds_name} (seed={seed})...")
        try:
            run_omnirec(
                datasets=ds_split,
                plan=plan,
                evaluator=evaluator
            )
        except Exception as e:
            print(f"  ERROR running experiments for {ds_name} seed={seed}: {e}")
            # Save partial results and continue
            continue

        # Collect results from the evaluator
        results_dict = evaluator.get_results()
        print(f"  Results collected for {ds_name} (seed={seed})")

        for dataset_id, df in results_dict.items():
            # Add seed and dataset name columns for later aggregation
            df = df.copy()
            df["seed"] = seed
            df["dataset"] = ds_name
            # Ensure 'k' column is integer for proper handling
            if "k" in df.columns:
                df["k"] = df["k"].astype(int)
            all_results.append(df)

        # Save per-seed results
        seed_results_path = os.path.join(working_dir, f"results_{ds_name}_seed_{seed}.json")
        evaluator.save_results(Path(seed_results_path))
        print(f"  Saved results to {seed_results_path}")

# ================ STEP 3: Aggregate and analyze results ================

print(f"\n{'=' * 70}")
print("AGGREGATING AND ANALYZING RESULTS")
print(f"{'=' * 70}")

if not all_results:
    print("No results collected. Exiting.")
    sys.exit(1)

# Combine all results into a single DataFrame
combined_results = pd.concat(all_results, ignore_index=True)

# Save combined raw results
combined_path = os.path.join(working_dir, "combined_results.csv")
combined_results.to_csv(combined_path, index=False)
print(f"Combined raw results saved to {combined_path}")
print(f"Total result rows: {len(combined_results)}")

# Compute mean and standard deviation across seeds for each (dataset, algorithm, name, k)
# The 'name' column contains the metric name (e.g., "NDCG", "Precision")
agg_columns = ["dataset", "algorithm", "name", "k"]
agg_results = (
    combined_results
    .groupby(agg_columns)["value"]
    .agg(["mean", "std"])
    .reset_index()
)

# Rename columns for readability
agg_results.columns = agg_columns + ["mean", "std"]

# Sort for display
agg_results = agg_results.sort_values(by=["dataset", "algorithm", "name", "k"]).reset_index(drop=True)

# Save aggregated results
agg_path = os.path.join(working_dir, "aggregated_results.csv")
agg_results.to_csv(agg_path, index=False)
print(f"Aggregated results saved to {agg_path}")

# ================ STEP 4: Print summary tables ================

print("\n" + "=" * 70)
print("FINAL AGGREGATED RESULTS (Mean ± Std across 5 seeds)")
print("=" * 70)

for ds in agg_results["dataset"].unique():
    print(f"\n{'=' * 60}")
    print(f"DATASET: {ds}")
    print(f"{'=' * 60}")

    ds_df = agg_results[agg_results["dataset"] == ds]

    for algo in ds_df["algorithm"].unique():
        print(f"\n  Algorithm: {algo}")
        algo_df = ds_df[ds_df["algorithm"] == algo]

        # Format output
        for _, row in algo_df.iterrows():
            metric_name = row["name"]
            k_val = int(row["k"])
            mean_val = row["mean"]
            std_val = row["std"]
            print(f"    {metric_name:12s} @ k={k_val:2d}:  {mean_val:.6f} ± {std_val:.6f}")

    print()

# ================ STEP 5: Statistical analysis ================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS: Variance across seeds")
print("=" * 70)

# For each (dataset, algorithm, metric), report mean std as a measure of seed sensitivity
sensitivity = (
    agg_results
    .groupby(["dataset", "algorithm", "name"])["std"]
    .agg(["mean", "max"])
    .reset_index()
)
sensitivity.columns = ["dataset", "algorithm", "metric", "avg_std", "max_std"]

for ds in sensitivity["dataset"].unique():
    print(f"\nDataset: {ds}")
    ds_sens = sensitivity[sensitivity["dataset"] == ds]
    for _, row in ds_sens.iterrows():
        print(f"  {row['algorithm']:25s} | {row['metric']:12s} | Avg Std: {row['avg_std']:.6f} | Max Std: {row['max_std']:.6f}")

# Find the most sensitive combinations
print("\n\nTop 5 most sensitive (dataset, algorithm, metric) combinations:")
top_sensitive = sensitivity.sort_values("avg_std", ascending=False).head(5)
for _, row in top_sensitive.iterrows():
    print(f"  {row['dataset']:25s} | {row['algorithm']:25s} | {row['metric']:12s} | Avg Std: {row['avg_std']:.6f}")

print("\n\nExperiment complete. All results saved in working directory.")
print(f"Working directory: {working_dir}")
