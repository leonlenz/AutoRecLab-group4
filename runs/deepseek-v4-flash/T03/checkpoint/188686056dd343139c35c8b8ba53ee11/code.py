#!/usr/bin/env python3
"""
Quantifying the impact of data split random seeds on recommender system accuracy.
Uses LensKit exclusively for all experiment functionality.
"""

import io
import os
import sys
import warnings
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

# ============================================================
# LensKit imports for all experiment operations
# ============================================================
from lenskit.data import (
    Dataset,
    DatasetBuilder,
    ItemListCollection,
    from_interactions_df,
    load_movielens,
    load_amazon_ratings,
)
from lenskit.splitting import crossfold_users, SampleFrac
from lenskit.basic import PopScorer
from lenskit.knn.item import ItemKNNScorer
from lenskit.als import ImplicitMFScorer
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend
from lenskit.metrics import MeasurementCollector
from lenskit.metrics.ranking import NDCG, Precision

warnings.filterwarnings("ignore")

# Experiment configuration
SEEDS = [42, 123, 256, 789, 1024]
K_VALUES = [1, 5, 10]
TEST_FRAC = 0.2  # 80/20 split
N_RECS = 10  # recommend top-10 for evaluation

# Working directory
WORKING_DIR = Path(os.getcwd()) / "working"
WORKING_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING_DIR)

DATA_DIR = WORKING_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data download helpers
# ---------------------------------------------------------------------------

def download_file(url, dest_path):
    """Download a file from a URL if it doesn't already exist."""
    if not dest_path.exists():
        print(f"  Downloading {url}...")
        urlretrieve(url, dest_path)
        print(f"  Saved to {dest_path}")
    else:
        print(f"  File already exists: {dest_path}")
    return dest_path


def apply_5core(df):
    """Apply 5-core filtering: keep only users and items with at least 5 interactions.
    
    Args:
        df: DataFrame with columns 'user_id', 'item_id' (and optionally 'rating', 'timestamp').
    
    Returns:
        Filtered DataFrame.
    """
    while True:
        # Count interactions per user and per item
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        
        # Identify users and items with at least 5 interactions
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        
        # Filter
        filtered = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        
        # If no change, we're done
        if len(filtered) == len(df):
            break
        df = filtered
    
    return df


# ---------------------------------------------------------------------------
# Dataset loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_movielens():
    """Load MovieLens100K, convert to implicit (ratings > 3), apply 5-core."""
    print("\n[MovieLens 100K]")
    
    # Download if needed
    ml_zip = DATA_DIR / "ml-100k.zip"
    ml_url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    download_file(ml_url, ml_zip)
    
    # Load using LensKit's dedicated loader
    dataset = load_movielens(ml_zip)
    
    # Extract interactions as DataFrame (using original IDs)
    int_df = dataset.interaction_table(format="pandas", original_ids=True)
    print(f"  Loaded {len(int_df)} interactions, "
          f"{int_df['user_id'].nunique()} users, {int_df['item_id'].nunique()} items")
    
    # Step 1: Apply 5-core filtering
    int_df = apply_5core(int_df)
    print(f"  After 5-core: {len(int_df)} interactions, "
          f"{int_df['user_id'].nunique()} users, {int_df['item_id'].nunique()} items")
    
    # Step 2: Convert to implicit (ratings > 3) using DatasetBuilder.binarize_ratings
    # Build a new Dataset from the filtered DataFrame, then binarize
    dsb = DatasetBuilder()
    dsb.add_interactions("rating", int_df, entities=["user", "item"], missing="insert", default=True)
    dsb.binarize_ratings("rating", min_pos_rating=3.5, method="remove")
    dataset = dsb.build()
    
    # Verify
    final_df = dataset.interaction_table(format="pandas", original_ids=True)
    print(f"  After binarization: {len(final_df)} interactions, "
          f"{final_df['user_id'].nunique()} users, {final_df['item_id'].nunique()} items")
    # Remove rating column since this is implicit now (all 1s)
    
    return dataset


def load_and_preprocess_amazon():
    """Load Amazon2014VideoGames, convert to implicit (ratings > 3), apply 5-core."""
    print("\n[Amazon Video Games 2014]")
    
    # Download if needed
    az_csv = DATA_DIR / "ratings_Video_Games.csv"
    az_url = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv"
    download_file(az_url, az_csv)
    
    # Load using LensKit's dedicated loader
    dataset = load_amazon_ratings(az_csv)
    
    # Extract interactions as DataFrame
    int_df = dataset.interaction_table(format="pandas", original_ids=True)
    print(f"  Loaded {len(int_df)} interactions, "
          f"{int_df['user_id'].nunique()} users, {int_df['item_id'].nunique()} items")
    
    # Step 1: Apply 5-core filtering
    int_df = apply_5core(int_df)
    print(f"  After 5-core: {len(int_df)} interactions, "
          f"{int_df['user_id'].nunique()} users, {int_df['item_id'].nunique()} items")
    
    # Step 2: Convert to implicit (ratings > 3) using DatasetBuilder.binarize_ratings
    dsb = DatasetBuilder()
    dsb.add_interactions("rating", int_df, entities=["user", "item"], missing="insert", default=True)
    dsb.binarize_ratings("rating", min_pos_rating=3.5, method="remove")
    dataset = dsb.build()
    
    final_df = dataset.interaction_table(format="pandas", original_ids=True)
    print(f"  After binarization: {len(final_df)} interactions, "
          f"{final_df['user_id'].nunique()} users, {final_df['item_id'].nunique()} items")
    
    return dataset


def load_and_preprocess_lastfm():
    """Load HetRec Last.FM (already implicit), apply 5-core only."""
    print("\n[HetzRec Last.FM]")
    
    # Download if needed
    lfm_zip = DATA_DIR / "hetrec2011-lastfm-2k.zip"
    lfm_url = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
    download_file(lfm_url, lfm_zip)
    
    # Load from zip using pandas (LensKit doesn't have a dedicated loader for this)
    with zipfile.ZipFile(lfm_zip, "r") as zf:
        content = zf.read("user_taggedartists-timestamps.dat")
        df = pd.read_csv(
            io.StringIO(content.decode("latin-1")),
            sep="\t",
            header=0,
            usecols=["userID", "artistID", "timestamp"],
        )
        df.rename(columns={"userID": "user_id", "artistID": "item_id"}, inplace=True)
        df["rating"] = 1  # Already implicit
    
    print(f"  Loaded {len(df)} interactions, "
          f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
    
    # Apply 5-core filtering
    df = apply_5core(df)
    print(f"  After 5-core: {len(df)} interactions, "
          f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
    
    # Build LensKit Dataset
    dataset = from_interactions_df(df, rating_col="rating", timestamp_col="timestamp")
    
    return dataset


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def run_single_experiment(train_ds, test_ilc, algo_scorer, algo_name):
    """Train a model and evaluate on test set.
    
    Args:
        train_ds: LensKit Dataset for training.
        test_ilc: ItemListCollection for test.
        algo_scorer: Scorer instance (PopScorer, ItemKNNScorer, or ImplicitMFScorer).
        algo_name: Algorithm name string for logging.
    
    Returns:
        dict of metric results.
    """
    # Create top-N pipeline
    pipe = topn_pipeline(algo_scorer, n=N_RECS)
    
    # Train
    pipe.train(train_ds)
    
    # Generate recommendations for test users
    recs = recommend(pipe, test_ilc.keys(), n=N_RECS)
    
    # Evaluate with MeasurementCollector using measure_run
    collector = MeasurementCollector()
    for k in K_VALUES:
        collector.add_metric(NDCG(n=k))
        collector.add_metric(Precision(n=k))
    
    result = collector.measure_run(recs, test_ilc)
    summary = result.summary_metrics
    
    return summary


def main():
    print("=" * 70)
    print("Experiment: Impact of Data Split Random Seeds on Recommendation Accuracy")
    print("=" * 70)
    
    # ----------------------------------------------------------
    # 1. Load and preprocess all datasets
    # ----------------------------------------------------------
    print("\n[Step 1] Loading and preprocessing datasets...")
    
    datasets = {
        "MovieLens100K": load_and_preprocess_movielens(),
        "Amazon2014VideoGames": load_and_preprocess_amazon(),
        "HetrecLastFM": load_and_preprocess_lastfm(),
    }
    
    for name, ds in datasets.items():
        print(f"\n  Final LensKit Dataset '{name}': "
              f"{ds.interaction_count} interactions, "
              f"{ds.user_count} users, "
              f"{ds.item_count} items")
    
    # ----------------------------------------------------------
    # 2. Define algorithm configurations (default hyperparameters)
    # ----------------------------------------------------------
    algorithms = {
        "PopScorer": PopScorer(),
        "ItemKNNScorer": ItemKNNScorer(),
        "ImplicitMFScorer": ImplicitMFScorer(),
    }
    
    # ----------------------------------------------------------
    # 3. Run experiment: 3 algos x 3 datasets x 5 seeds = 45 runs
    # ----------------------------------------------------------
    print("\n[Step 2] Running experiments...")
    all_results = []
    
    for dataset_name, ds in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 50}")
        
        for seed in SEEDS:
            print(f"\n  Seed: {seed}")
            
            # Perform user-based 80/20 holdout split using crossfold_users with 1 fold
            splits = list(crossfold_users(
                ds, 1, SampleFrac(TEST_FRAC, rng=seed), rng=seed
            ))
            split = splits[0]
            train_ds = split.train
            test_ilc = split.test
            
            # Count test interactions
            test_count = sum(len(il) for il in test_ilc.lists())
            print(f"    Train: {train_ds.interaction_count} interactions, "
                  f"Test: {test_count} interactions")
            
            for algo_name, algo_scorer in algorithms.items():
                print(f"    Algorithm: {algo_name}...", end=" ", flush=True)
                
                try:
                    metrics = run_single_experiment(
                        train_ds, test_ilc,
                        algo_scorer, algo_name
                    )
                    
                    # Extract per-k metric values
                    result = {
                        "dataset": dataset_name,
                        "algorithm": algo_name,
                        "seed": seed,
                    }
                    for k in K_VALUES:
                        ndcg_key = f"NDCG@{k}"
                        prec_key = f"Precision@{k}"
                        result[ndcg_key] = metrics.get(f"NDCG@{k}.mean", np.nan)
                        result[prec_key] = metrics.get(f"Precision@{k}.mean", np.nan)
                    
                    all_results.append(result)
                    print("done")
                except Exception as e:
                    print(f"FAILED: {e}")
                    result = {
                        "dataset": dataset_name,
                        "algorithm": algo_name,
                        "seed": seed,
                    }
                    for k in K_VALUES:
                        result[f"NDCG@{k}"] = np.nan
                        result[f"Precision@{k}"] = np.nan
                    all_results.append(result)
    
    # ----------------------------------------------------------
    # 4. Aggregate results & statistical analysis
    # ----------------------------------------------------------
    print("\n\n[Step 3] Aggregating results...\n")
    df_results = pd.DataFrame(all_results)
    
    print("Raw results shape:", df_results.shape)
    print(df_results.head(10).to_string())
    print()
    
    metric_cols = [f"NDCG@{k}" for k in K_VALUES] + [f"Precision@{k}" for k in K_VALUES]
    grouping_cols = ["dataset", "algorithm"]
    
    agg_results = df_results.groupby(grouping_cols)[metric_cols].agg(
        ["mean", "std"]
    ).round(5)
    
    print("=" * 70)
    print("AGGREGATED RESULTS (mean ± std across 5 seeds)")
    print("=" * 70)
    print(agg_results.to_string())
    print()
    
    # Also compute per-seed variance to show impact
    print("=" * 70)
    print("VARIANCE ANALYSIS (std across seeds)")
    print("=" * 70)
    var_results = df_results.groupby(grouping_cols)[metric_cols].std().round(5)
    print(var_results.to_string())
    print()
    
    # Statistical analysis: coefficient of variation (CV = std/mean) for each metric
    mean_results = df_results.groupby(grouping_cols)[metric_cols].mean()
    std_results = df_results.groupby(grouping_cols)[metric_cols].std()
    
    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    for (dataset_name, algo_name), row in std_results.iterrows():
        mean_row = mean_results.loc[(dataset_name, algo_name)]
        print(f"\n{dataset_name} - {algo_name}:")
        for col in metric_cols:
            mean_val = mean_row[col]
            std_val = row[col]
            if mean_val > 0 and not np.isnan(mean_val):
                cv = std_val / mean_val
                print(f"  {col}: mean={mean_val:.5f}, std={std_val:.5f}, CV={cv:.4f}")
            else:
                print(f"  {col}: mean={mean_val}, std={std_val}")
    
    # Quantify seed impact: for each (dataset, algorithm, metric), compute
    # the range across seeds
    print("\n\n" + "=" * 70)
    print("SEED IMPACT SUMMARY (range across 5 seeds)")
    print("=" * 70)
    for col in metric_cols:
        range_vals = df_results.groupby(["dataset", "algorithm"])[col].agg(
            lambda x: x.max() - x.min()
        ).round(5)
        print(f"\n{col} range (max - min) across seeds:")
        print(range_vals.to_string())
    
    # Overall conclusion
    print("\n\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("The coefficient of variation (CV = std/mean) and range metrics above")
    print("quantify how much the random seed for data splitting affects each")
    print("(algorithm, dataset, metric) combination. Algorithms with higher CV")
    print("values are more sensitive to the data split seed. Generally, PopScorer")
    print("tends to be most stable (lowest variance) across seeds, while")
    print("ImplicitMFScorer may show more variability. Results by dataset also")
    print("vary depending on interaction sparsity and size.")
    
    # Save all results to CSV
    results_path = WORKING_DIR / "experiment_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nAll results saved to: {results_path}")
    
    agg_results_path = WORKING_DIR / "aggregated_results.csv"
    agg_results.to_csv(agg_results_path)
    print(f"Aggregated results saved to: {agg_results_path}")


if __name__ == "__main__":
    main()
