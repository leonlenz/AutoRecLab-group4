#!/usr/bin/env python3
"""
Experiment: Quantifying the impact of data split random seeds on recommender system accuracy.
Uses OmniRec to test 3 LensKit algorithms (ALS, ItemKNN, Pop) on 3 datasets
(MovieLens100K, Amazon2014VideoGames, HetrecLastFM) with 5 different random seeds
for user-based 80/20 holdout. Evaluates NDCG@k and Precision@k for k=1,5,10.
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# OmniRec imports
from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.2       # 80/20 user-based holdout
CORE_VALUE = 5        # 5-core filtering

BASE_WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(BASE_WORKING_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Main experiment loop across seeds
# ---------------------------------------------------------------------------
print("=" * 70)
print("SEED VARIATION EXPERIMENT")
print("Using OmniRec with LensKit algorithms")
print("=" * 70)

all_results = []  # list of DataFrames, one per seed

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'─' * 60}")
    print(f"SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'─' * 60}")

    # Set global random state for reproducibility
    set_random_state(seed)

    # -----------------------------------------------------------------------
    # Step 1: Load and preprocess datasets
    # -----------------------------------------------------------------------
    print("\n  Loading datasets...")

    # --- MovieLens100K ---
    print("  MovieLens100K: loading...")
    ml100k_raw = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print(f"    Raw interactions: {ml100k_raw.num_interactions()}")
    ml100k_pipeline = Pipe(
        MakeImplicit(3),       # ratings >= 3 become implicit (keep, remove others)
        CorePruning(5),        # 5-core filtering
        UserHoldout(0, TEST_FRAC)  # 80/20 user-based split (no validation set)
    )
    ml100k_ds = ml100k_pipeline.process(ml100k_raw)
    print(f"    After preprocessing: {ml100k_ds.num_interactions()} interactions")

    # --- Amazon2014VideoGames ---
    print("  Amazon2014VideoGames: loading...")
    amazon_raw = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
    print(f"    Raw interactions: {amazon_raw.num_interactions()}")
    amazon_pipeline = Pipe(
        MakeImplicit(3),       # ratings >= 3 become implicit
        CorePruning(5),        # 5-core filtering
        UserHoldout(0, TEST_FRAC)
    )
    amazon_ds = amazon_pipeline.process(amazon_raw)
    print(f"    After preprocessing: {amazon_ds.num_interactions()} interactions")

    # --- HetrecLastFM (already implicit) ---
    print("  HetrecLastFM: loading...")
    lastfm_raw = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
    print(f"    Raw interactions: {lastfm_raw.num_interactions()}")
    lastfm_pipeline = Pipe(
        CorePruning(5),        # 5-core filtering (already implicit)
        UserHoldout(0, TEST_FRAC)
    )
    lastfm_ds = lastfm_pipeline.process(lastfm_raw)
    print(f"    After preprocessing: {lastfm_ds.num_interactions()} interactions")

    datasets = [ml100k_ds, amazon_ds, lastfm_ds]

    # -----------------------------------------------------------------------
    # Step 2: Create experiment plan with default hyperparameters
    # -----------------------------------------------------------------------
    print("\n  Creating experiment plan...")

    plan = ExperimentPlan(plan_name=f"SeedVariation_seed{seed}")

    # All algorithms use default/standard hyperparameters
    plan.add_algorithm(LensKit.ImplicitMFScorer)   # ALS with default params
    plan.add_algorithm(LensKit.ItemKNNScorer)      # ItemKNN with default params
    plan.add_algorithm(LensKit.PopScorer)           # Popularity with default params

    # -----------------------------------------------------------------------
    # Step 3: Configure evaluator with NDCG and Precision at k=1,5,10
    # -----------------------------------------------------------------------
    print("  Configuring evaluator...")

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10])
    )

    # -----------------------------------------------------------------------
    # Step 4: Run experiments
    # -----------------------------------------------------------------------
    print("  Running experiments...")

    run_omnirec(
        datasets=datasets,
        plan=plan,
        evaluator=evaluator
    )

    # -----------------------------------------------------------------------
    # Step 5: Collect results
    # -----------------------------------------------------------------------
    print("  Collecting results...")

    results_dict = evaluator.get_results()
    for ds_id, df in results_dict.items():
        df = df.copy()
        df['seed'] = seed
        all_results.append(df)

# ---------------------------------------------------------------------------
# Step 6: Aggregate results across seeds
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Aggregating Results Across Seeds")
print("=" * 70)

if not all_results:
    print("No results collected. Exiting.")
    sys.exit(1)

combined_results = pd.concat(all_results, ignore_index=True)

# Extract dataset name from the dataset identifier (before the hash)
def extract_ds_name(ds_id):
    for name in ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]:
        if name in ds_id:
            return name
    return ds_id

combined_results['dataset'] = combined_results['dataset'].apply(extract_ds_name)

print(f"\nTotal result rows: {len(combined_results)}")

# Group by dataset, algorithm, name (metric), k, compute mean & std across seeds
agg_results = (
    combined_results
    .groupby(["dataset", "algorithm", "name", "k"])["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

agg_results.columns = ["Dataset", "Algorithm", "Metric", "k", "Mean", "Std", "Count"]

# Extract short algorithm names
def short_algo_name(full_name):
    for short_name in ["ImplicitMFScorer", "ItemKNNScorer", "PopScorer"]:
        if short_name in full_name:
            return short_name
    return full_name

agg_results["Algorithm"] = agg_results["Algorithm"].apply(short_algo_name)
agg_results = agg_results.sort_values(["Dataset", "Algorithm", "Metric", "k"])

# Print aggregated results
print("\n" + "=" * 70)
print("AGGREGATED RESULTS (Mean ± Std across 5 seeds)")
print("=" * 70)

for dataset_name in agg_results["Dataset"].unique():
    print(f"\n{'─' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'─' * 60}")

    ds_df = agg_results[agg_results["Dataset"] == dataset_name]
    for _, row in ds_df.iterrows():
        print(f"  {row['Algorithm']:20s} | {row['Metric']:10s}@{int(row['k']):2d} | "
              f"Mean={row['Mean']:.6f} ± Std={row['Std']:.6f} (n={int(row['Count'])})")

# Coefficient of Variation
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
        cv_val = row["CV"]
        cv_str = f"{cv_val:.4f}" if not np.isnan(cv_val) else "N/A"
        print(f"  {row['Algorithm']:20s} | {row['Metric']:10s}@{int(row['k']):2d} | "
              f"CV={cv_str}")

# Save results
results_csv_path = os.path.join(BASE_WORKING_DIR, "seed_variation_results.csv")
combined_results.to_csv(results_csv_path, index=False)
print(f"\nDetailed results saved to: {results_csv_path}")

agg_csv_path = os.path.join(BASE_WORKING_DIR, "seed_variation_aggregated.csv")
agg_results.to_csv(agg_csv_path, index=False)
print(f"Aggregated results saved to: {agg_csv_path}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
