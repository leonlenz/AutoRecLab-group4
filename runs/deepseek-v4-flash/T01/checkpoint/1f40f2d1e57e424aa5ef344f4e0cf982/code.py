#!/usr/bin/env python3
"""
Experiment: Quantifying the impact of data split random seeds on recommender system accuracy.

Runs 3 algorithms (ALS/ImplicitMF, ItemKNN, Popularity) on 3 datasets (MovieLens100K,
Amazon2014VideoGames, HetrecLastFM) with 5 different random seeds for user-based 80/20 holdout.
Evaluates NDCG@k and Precision@k for k=1,5,10.
"""

import os
import sys
import warnings
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# OmniRec imports
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2          # 80/20 user-based holdout
CORE_VALUE = 5           # 5-core filtering
IMPLICIT_THRESHOLD = 4   # Ratings > 3 => >= 4

BASE_WORKING_DIR = os.path.join(os.getcwd(), 'working')

# ---------------------------------------------------------------------------
# Step 1: Load raw datasets (only once)
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading datasets...")
print("=" * 70)

# MovieLens100K (explicit ratings 1-5)
print("Loading MovieLens100K...")
ml100k_raw = RecSysDataSet.use_dataloader(DataSet.MovieLens100K, force_download=False)
print(f"  MovieLens100K: {ml100k_raw.num_interactions()} interactions, "
      f"rating range [{ml100k_raw.min_rating()}, {ml100k_raw.max_rating()}]")

# Amazon2014VideoGames (explicit ratings)
print("Loading Amazon2014VideoGames...")
amazon_raw = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames, force_download=False)
print(f"  Amazon2014VideoGames: {amazon_raw.num_interactions()} interactions, "
      f"rating range [{amazon_raw.min_rating()}, {amazon_raw.max_rating()}]")

# HetrecLastFM (already implicit - all ratings = 1)
print("Loading HetrecLastFM...")
lastfm_raw = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM, force_download=False)
print(f"  HetrecLastFM: {lastfm_raw.num_interactions()} interactions, "
      f"rating range [{lastfm_raw.min_rating()}, {lastfm_raw.max_rating()}]")

# ---------------------------------------------------------------------------
# Step 2: Preprocess each dataset (without splitting)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Preprocessing datasets (implicit conversion + 5-core)...")
print("=" * 70)

# MovieLens: MakeImplicit(4) + CorePruning(5)
print("\nPreprocessing MovieLens100K...")
ml100k_pipe = Pipe(
    MakeImplicit(IMPLICIT_THRESHOLD),
    CorePruning(CORE_VALUE),
)
ml100k_processed = ml100k_pipe.process(ml100k_raw)
print(f"  After preprocessing: {ml100k_processed.num_interactions()} interactions")

# Amazon: MakeImplicit(4) + CorePruning(5)
print("\nPreprocessing Amazon2014VideoGames...")
amazon_pipe = Pipe(
    MakeImplicit(IMPLICIT_THRESHOLD),
    CorePruning(CORE_VALUE),
)
amazon_processed = amazon_pipe.process(amazon_raw)
print(f"  After preprocessing: {amazon_processed.num_interactions()} interactions")

# LastFM: CorePruning(5) only (already implicit)
print("\nPreprocessing HetrecLastFM...")
lastfm_pipe = Pipe(
    CorePruning(CORE_VALUE),
)
lastfm_processed = lastfm_pipe.process(lastfm_raw)
print(f"  After preprocessing: {lastfm_processed.num_interactions()} interactions")

processed_datasets = {
    "MovieLens100K": ml100k_processed,
    "Amazon2014VideoGames": amazon_processed,
    "HetrecLastFM": lastfm_processed,
}

# ---------------------------------------------------------------------------
# Step 3: Loop over seeds - split, run experiments, collect results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Running experiments across 5 seeds...")
print("=" * 70)

all_results = []  # list of dicts to accumulate results

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'─' * 60}")
    print(f"SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'─' * 60}")

    # Create a seed-specific working directory to avoid checkpoint collisions
    seed_working_dir = os.path.join(BASE_WORKING_DIR, f"seed_{seed}")
    os.makedirs(seed_working_dir, exist_ok=True)
    os.chdir(seed_working_dir)

    # Set global random state for reproducibility of splitting
    set_random_state(seed)

    # Apply user-based holdout split (80/20) to each dataset
    splitter = UserHoldout(validation_size=0, test_size=TEST_SIZE)

    split_datasets = []
    for ds_name, ds in processed_datasets.items():
        print(f"  Splitting {ds_name} with seed={seed}...")
        # Re-set random state before each split to be safe
        set_random_state(seed)
        split_ds = splitter.process(ds)
        split_datasets.append(split_ds)
        # Verify split dimensions
        train_df = split_ds._data.train
        test_df = split_ds._data.test
        print(f"    Train: {len(train_df)} interactions, Test: {len(test_df)} interactions")

    # Create experiment plan with all 3 algorithms (default hyperparameters)
    plan = ExperimentPlan(plan_name=f"SeedExperiment_seed{seed}")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})    # ALS with default params
    plan.add_algorithm(LensKit.ItemKNNScorer, {})        # ItemKNN with default params
    plan.add_algorithm(LensKit.PopScorer, {})             # Popularity baseline with default params

    # Configure evaluation metrics
    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    # Run all experiments for this seed
    print(f"  Running experiments with seed={seed}...")
    try:
        run_omnirec(datasets=split_datasets, plan=plan, evaluator=evaluator)
    except Exception as e:
        print(f"  WARNING: run_omnirec raised an error for seed {seed}: {e}")
        print("  Attempting to collect partial results...")

    # Collect results
    try:
        results_dict = evaluator.get_results()
    except Exception as e:
        print(f"  WARNING: Could not retrieve results for seed {seed}: {e}")
        continue

    for dataset_id, df in results_dict.items():
        # dataset_id has format like "MovieLens100K-<hash>"
        # Extract the base dataset name
        for ds_name in processed_datasets:
            if ds_name in dataset_id:
                base_name = ds_name
                break
        else:
            base_name = dataset_id

        # Add seed and dataset columns
        df = df.copy()
        df["seed"] = seed
        df["dataset"] = base_name
        all_results.append(df)

        # Print results for this seed
        print(f"\n  Results for {base_name} (seed={seed}):")
        for _, row in df.iterrows():
            print(f"    Algo={row['algorithm']:40s} | "
                  f"Metric={row['name']:10s} | k={str(row['k']):4s} | "
                  f"Value={row['value']:.6f}")

    # Change back to original directory
    os.chdir(os.getcwd())

# ---------------------------------------------------------------------------
# Step 4: Statistical analysis across seeds
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Statistical Analysis - Mean and Std across seeds")
print("=" * 70)

if not all_results:
    print("No results collected. Exiting.")
    sys.exit(1)

# Combine all results
combined_results = pd.concat(all_results, ignore_index=True)
print(f"\nTotal result rows collected: {len(combined_results)}")

# Ensure k is treated as a string for grouping
combined_results["k"] = combined_results["k"].astype(str)

# Group by dataset, algorithm, metric, k and compute mean & std
agg_results = (
    combined_results
    .groupby(["dataset", "algorithm", "name", "k"])["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

# Rename columns for clarity
agg_results.columns = ["Dataset", "Algorithm", "Metric", "k", "Mean", "Std", "Count"]
agg_results = agg_results.sort_values(["Dataset", "Algorithm", "Metric", "k"])

# Print the aggregated results
print("\n" + "=" * 70)
print("AGGREGATED RESULTS (Mean ± Std across 5 seeds)")
print("=" * 70)

for dataset_name in agg_results["Dataset"].unique():
    print(f"\n{'─' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'─' * 60}")

    ds_df = agg_results[agg_results["Dataset"] == dataset_name]
    for _, row in ds_df.iterrows():
        algo_short = row["Algorithm"].split("-")[0] if "-" in str(row["Algorithm"]) else row["Algorithm"]
        algo_short = algo_short.split(".")[-1] if "." in str(algo_short) else algo_short
        print(f"  {algo_short:20s} | {row['Metric']:10s}@{str(row['k']):2s} | "
              f"Mean={row['Mean']:.6f} ± Std={row['Std']:.6f} (n={int(row['Count'])})")

# Also compute and print a summary of variation magnitude
print("\n" + "=" * 70)
print("VARIATION SUMMARY (Coefficient of Variation = Std/Mean)")
print("=" * 70)

agg_results["CV"] = np.where(agg_results["Mean"] > 0,
                              agg_results["Std"] / agg_results["Mean"],
                              np.nan)

for dataset_name in agg_results["Dataset"].unique():
    print(f"\nDataset: {dataset_name}")
    ds_df = agg_results[agg_results["Dataset"] == dataset_name]
    for _, row in ds_df.iterrows():
        algo_short = str(row["Algorithm"]).split("-")[0].split(".")[-1]
        cv_str = f"{row['CV']:.4f}" if not np.isnan(row.get("CV", np.nan)) else "N/A"
        print(f"  {algo_short:20s} | {row['Metric']:10s}@{str(row['k']):2s} | "
              f"CV={cv_str}")

# Save results to CSV for further analysis
os.makedirs(BASE_WORKING_DIR, exist_ok=True)
results_csv_path = os.path.join(BASE_WORKING_DIR, "seed_variation_results.csv")
combined_results.to_csv(results_csv_path, index=False)
print(f"\nDetailed results saved to: {results_csv_path}")

agg_csv_path = os.path.join(BASE_WORKING_DIR, "seed_variation_aggregated.csv")
agg_results.to_csv(agg_csv_path, index=False)
print(f"Aggregated results saved to: {agg_csv_path}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
