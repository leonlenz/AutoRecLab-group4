#!/usr/bin/env python3
"""
Experiment: Quantifying the impact of data split random seeds on recommender system accuracy.

Uses LensKit directly to test 3 algorithms (ALS, ItemKNN, Pop) on 3 datasets
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

# LensKit imports - using LensKit directly as per requirements
from lenskit.data import load_movielens, load_amazon_ratings, from_interactions_df, Dataset
from lenskit.splitting import sample_users, SampleFrac, TTSplit
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend
from lenskit.metrics.bulk import RunAnalysis
from lenskit.metrics.ranking import NDCG, Precision
from lenskit.basic import PopScorer
from lenskit.als import ImplicitMFScorer
from lenskit.knn.item import ItemKNNScorer
from lenskit.random import random_generator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.2          # 80/20 user-based holdout per user
CORE_VALUE = 5           # 5-core filtering
IMPLICIT_THRESHOLD = 4   # Ratings > 3 => >= 4 (so threshold of 4 means rating >= 4)

BASE_WORKING_DIR = os.path.join(os.getcwd(), 'working')

# ---------------------------------------------------------------------------
# Helper: 5-core filtering on a DataFrame
# ---------------------------------------------------------------------------
def filter_5core(df: pd.DataFrame) -> pd.DataFrame:
    """Apply 5-core filtering: keep only users and items with at least 5 interactions."""
    orig_len = len(df)
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        valid_users = user_counts[user_counts >= CORE_VALUE].index
        valid_items = item_counts[item_counts >= CORE_VALUE].index
        df_filtered = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        if len(df_filtered) == len(df):
            break
        df = df_filtered
    print(f"    5-core filtering: {orig_len} -> {len(df)} interactions")
    return df

# ---------------------------------------------------------------------------
# Helper: Build a LensKit Dataset from a DataFrame of interactions
# ---------------------------------------------------------------------------
def make_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a DataFrame with user_id, item_id, (optionally rating) to a Dataset."""
    return from_interactions_df(df, user_col='user_id', item_col='item_id', rating_col='rating')

# ---------------------------------------------------------------------------
# Step 1: Load and preprocess datasets
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1 & 2: Load & Preprocess Datasets")
print("=" * 70)

# --- MovieLens100K ---
print("\nProcessing MovieLens100K...")
# Load the raw DataFrame
ml100k_raw = load_movielens('data/ml-100k.zip')
ml100k_df = ml100k_raw.interaction_table(format="pandas", original_ids=True, field="all")
print(f"  Raw: {len(ml100k_df)} interactions")
# Convert to implicit: rating > 3 => keep (rating >= 4)
ml100k_implicit = ml100k_df[ml100k_df['rating'] >= IMPLICIT_THRESHOLD].copy()
ml100k_implicit['rating'] = 1.0
print(f"  After implicit (rating >= {IMPLICIT_THRESHOLD}): {len(ml100k_implicit)} interactions")
# 5-core filtering
ml100k_implicit = filter_5core(ml100k_implicit)
# Build dataset
ml100k_ds = make_dataset(ml100k_implicit)
print(f"  Dataset: {ml100k_ds.user_count} users, {ml100k_ds.item_count} items, {ml100k_ds.interaction_count} interactions")

# --- Amazon2014VideoGames ---
print("\nProcessing Amazon2014VideoGames...")
try:
    amazon_raw = load_amazon_ratings('data/amazon_videogames_2014.csv')
    amazon_df = amazon_raw.interaction_table(format="pandas", original_ids=True, field="all")
except Exception:
    # Fallback: try to build from scratch
    print("  Attempting to load Amazon from raw CSV...")
    amazon_csv_path = 'data/amazon_videogames_2014.csv'
    if os.path.exists(amazon_csv_path):
        amazon_df = pd.read_csv(amazon_csv_path, names=['user_id', 'item_id', 'rating', 'timestamp'])
        amazon_df['rating'] = amazon_df['rating'].astype(float)
    else:
        # Try alternate paths
        alt_paths = [
            '/home/prv_tristan/AutoRecLab-group4/.cache/omnirec/datasets/Amazon2014VideoGames/ratings.csv',
            '../datasets/Amazon2014VideoGames/ratings.csv',
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                print(f"  Found at: {p}")
                amazon_df = pd.read_csv(p)
                found = True
                break
        if not found:
            print("  Warning: Amazon data not found, using dummy fallback")
            amazon_df = pd.DataFrame(columns=pd.Index(['user_id', 'item_id', 'rating', 'timestamp']))

print(f"  Raw: {len(amazon_df)} interactions")
# Convert to implicit: rating > 3 => keep (rating >= 4)
amazon_implicit = amazon_df[amazon_df['rating'] >= IMPLICIT_THRESHOLD].copy()
amazon_implicit['rating'] = 1.0
print(f"  After implicit (rating >= {IMPLICIT_THRESHOLD}): {len(amazon_implicit)} interactions")
# 5-core filtering
amazon_implicit = filter_5core(amazon_implicit)
# Build dataset
amazon_ds = make_dataset(amazon_implicit)
print(f"  Dataset: {amazon_ds.user_count} users, {amazon_ds.item_count} items, {amazon_ds.interaction_count} interactions")

# --- HetrecLastFM ---
print("\nProcessing HetrecLastFM...")
lastfm_csv_path = 'data/lastfm_ratings.csv'
if os.path.exists(lastfm_csv_path):
    lastfm_df = pd.read_csv(lastfm_csv_path)
else:
    alt_paths = [
        '/home/prv_tristan/AutoRecLab-group4/.cache/omnirec/datasets/HetrecLastFM/ratings.csv',
        '../datasets/HetrecLastFM/ratings.csv',
    ]
    found = False
    for p in alt_paths:
        if os.path.exists(p):
            print(f"  Found at: {p}")
            lastfm_df = pd.read_csv(p)
            found = True
            break
    if not found:
        print("  Warning: LastFM data not found, using dummy fallback")
        lastfm_df = pd.DataFrame(columns=pd.Index(['user_id', 'item_id', 'rating']))

print(f"  Raw: {len(lastfm_df)} interactions")
# HetrecLastFM is already implicit (ratings = 1), just set rating to 1.0
if 'rating' in lastfm_df.columns:
    lastfm_implicit = lastfm_df.copy()
    lastfm_implicit['rating'] = 1.0
else:
    lastfm_implicit = lastfm_df.copy()
    lastfm_implicit['rating'] = 1.0
print(f"  Already implicit: {len(lastfm_implicit)} interactions")
# 5-core filtering
lastfm_implicit = filter_5core(lastfm_implicit)
# Build dataset
lastfm_ds = make_dataset(lastfm_implicit)
print(f"  Dataset: {lastfm_ds.user_count} users, {lastfm_ds.item_count} items, {lastfm_ds.interaction_count} interactions")

processed_datasets = {
    "MovieLens100K": ml100k_ds,
    "Amazon2014VideoGames": amazon_ds,
    "HetrecLastFM": lastfm_ds,
}

# ---------------------------------------------------------------------------
# Step 2 & 3: Loop over seeds - split, train, recommend, evaluate
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Running experiments across 5 seeds...")
print("=" * 70)

# This will hold all result records
all_results = []  # list of dicts with dataset, algorithm, seed, metric, k, value

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'─' * 60}")
    print(f"SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'─' * 60}")

    # Create seed-specific working directory
    seed_working_dir = os.path.join(BASE_WORKING_DIR, f"seed_{seed}")
    os.makedirs(seed_working_dir, exist_ok=True)

    # For each dataset, split using sample_users with SampleFrac
    for ds_name, ds in processed_datasets.items():
        print(f"\n  Dataset: {ds_name}")

        # Create the holdout method with the seed for reproducibility
        # SampleFrac(0.2) selects 20% of items per user for testing
        holdout = SampleFrac(TEST_FRAC, rng=seed)

        # Use sample_users with n_test_users = all users (use a large number to get all users)
        # We need all users to be in the test set. sample_users selects a random subset of users
        # of size `size`. To get all users, we need to pass the total user count.
        n_users = ds.user_count
        split = sample_users(ds, n_users, holdout, rng=seed)

        train_ds = split.train
        test_ilc = split.test
        print(f"    Train users: {train_ds.user_count}, Train interactions: {train_ds.interaction_count}")
        print(f"    Test users: {len(test_ilc)}, Test interactions: {split.test_size}")

        # Define algorithms to run
        algorithms = [
            ("ALS_ImplicitMF", ImplicitMFScorer(factors=50, iterations=15, reg=0.1)),
            ("ItemKNN", ItemKNNScorer(k=50, feedback='implicit')),
            ("Pop", PopScorer()),
        ]

        for algo_name, algo in algorithms:
            print(f"    Training {algo_name}...")

            try:
                # Create top-N recommendation pipeline
                pipeline = topn_pipeline(algo, n=20)
                pipeline.train(train_ds)

                # Generate recommendations for all test users
                recs = recommend(pipeline, test_ilc.keys())

                # Evaluate
                analysis = RunAnalysis()
                analysis.add_metric(NDCG(n=1))
                analysis.add_metric(NDCG(n=5))
                analysis.add_metric(NDCG(n=10))
                analysis.add_metric(Precision(n=1))
                analysis.add_metric(Precision(n=5))
                analysis.add_metric(Precision(n=10))

                result = analysis.measure(recs, split.test)
                summary = result.list_summary()

                # Extract individual metric values
                for metric_name in summary.index:
                    mean_val = summary.loc[metric_name, 'mean']
                    # Determine k from metric name
                    if '@' in metric_name:
                        metric_base = metric_name.split('@')[0]
                        k_str = metric_name.split('@')[1]
                    else:
                        metric_base = metric_name
                        k_str = ''

                    all_results.append({
                        'dataset': ds_name,
                        'algorithm': algo_name,
                        'seed': seed,
                        'metric': metric_base,
                        'k': k_str,
                        'value': mean_val,
                    })

                    print(f"      {metric_name}: {mean_val:.6f}")

            except Exception as e:
                print(f"      ERROR with {algo_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

# ---------------------------------------------------------------------------
# Step 4: Statistical analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Statistical Analysis - Mean and Std across seeds")
print("=" * 70)

if not all_results:
    print("No results collected. Exiting.")
    sys.exit(1)

combined_results = pd.DataFrame(all_results)
print(f"\nTotal result rows collected: {len(combined_results)}")

# Group by dataset, algorithm, metric, k and compute mean & std
agg_results = (
    combined_results
    .groupby(["dataset", "algorithm", "metric", "k"])["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

agg_results.columns = ["Dataset", "Algorithm", "Metric", "k", "Mean", "Std", "Count"]
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
        print(f"  {row['Algorithm']:20s} | {row['Metric']:10s}@{str(row['k']):2s} | "
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
        cv_str = f"{row['CV']:.4f}" if not np.isnan(row.get("CV", np.nan)) else "N/A"
        print(f"  {row['Algorithm']:20s} | {row['Metric']:10s}@{str(row['k']):2s} | "
              f"CV={cv_str}")

# Save results
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
