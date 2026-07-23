#!/usr/bin/env python3
"""
Quantify the impact of data split random seeds on recommender system accuracy.
Uses LensKit with three algorithms, three datasets, and 5 seeds.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any, TypedDict

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ── LensKit imports ──────────────────────────────────────────────────────────
from lenskit.data import (
    Dataset,
    load_movielens,
    load_amazon_ratings,
    from_interactions_df,
    ItemListCollection,
)
from lenskit.splitting import crossfold_users, SampleFrac
from lenskit.basic import PopScorer
from lenskit.knn import ItemKNNScorer
from lenskit.als import ImplicitMFScorer
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend
from lenskit.metrics import MeasurementCollector
from lenskit.metrics.ranking import NDCG, Precision
from lenskit.random import set_global_rng


# ── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 999]

# Dataset file paths
DATA_DIR = os.path.join(os.getcwd(), 'data')
ML100K_PATH = os.path.join(DATA_DIR, 'ml-100k.zip')
AZ_VIDEO_PATH = os.path.join(DATA_DIR, 'ratings_Video_Games.csv')
LASTFM_PATH = os.path.join(DATA_DIR, 'user_artists.dat')

DATASET_NAMES = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']


class AlgorithmConfig(TypedDict):
    scorer_class: type[Any]
    scorer_kwargs: dict[str, Any]
    pipeline_kwargs: dict[str, Any]


ALGORITHM_CONFIGS: dict[str, AlgorithmConfig] = {
    'Pop': {
        'scorer_class': PopScorer,
        'scorer_kwargs': {},
        'pipeline_kwargs': {'n': 10},
    },
    'ItemKNN': {
        'scorer_class': ItemKNNScorer,
        'scorer_kwargs': {'max_nbrs': 20, 'min_nbrs': 1, 'feedback': 'implicit'},
        'pipeline_kwargs': {'n': 10},
    },
    'ALS': {
        'scorer_class': ImplicitMFScorer,
        'scorer_kwargs': {'features': 20, 'epochs': 10},
        'pipeline_kwargs': {'n': 10},
    },
}

# Working directory
WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)


def load_dataset(name: str) -> Dataset:
    """Load one of the three datasets using LensKit native loaders."""
    if name == 'MovieLens100K':
        print(f"     Loading MovieLens 100K from {ML100K_PATH} ...")
        ds = load_movielens(ML100K_PATH)
    elif name == 'Amazon2014VideoGames':
        print(f"     Loading Amazon Video Games from {AZ_VIDEO_PATH} ...")
        ds = load_amazon_ratings(AZ_VIDEO_PATH)
    elif name == 'HetrecLastFM':
        print(f"     Loading Last.FM from {LASTFM_PATH} ...")
        # Last.FM is not natively supported by LensKit loaders, so use pandas
        lfm_df = pd.read_csv(LASTFM_PATH, sep='\t', header=0)
        # Rename columns to match LensKit conventions
        lfm_df = lfm_df.rename(columns={
            'userID': 'user_id', 'artistID': 'item_id', 'weight': 'count'
        })
        # Last.fm is already implicit (play counts), use count as rating
        ds = from_interactions_df(lfm_df, user_col='user_id', item_col='item_id',
                                  rating_col='count')
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return ds


def make_implicit(ds: Dataset, min_rating: int = 4) -> Dataset:
    """Convert explicit ratings to implicit feedback: keep only ratings > threshold.
    
    For ratings > 3, we keep ratings >= 4 (since ratings are integers in [1,5]).
    """
    # Get the interaction data as a pandas DataFrame with original IDs
    int_df = ds.interaction_table(format='pandas', original_ids=True)
    
    if 'rating' not in int_df.columns:
        # Already implicit (no rating column), return as is
        return ds
    
    n_before = len(int_df)
    
    # Keep only ratings > 3 (i.e., rating >= 4)
    mask = int_df['rating'] > 3
    # Also keep entries without rating column
    implicit_df = int_df[mask].copy()
    # Drop the rating column (implicit feedback - no ratings needed)
    implicit_df = implicit_df.drop(columns=['rating'])
    
    n_after = len(implicit_df)
    print(f"     Making implicit: {n_before} -> {n_after} interactions (ratings > 3)")
    
    if n_after == 0:
        print("     WARNING: No interactions remain after implicit conversion!")
        # If no interactions > 3, use all (this shouldn't happen for ML100K/Amazon)
        return ds
    
    # Build a new dataset from the filtered dataframe
    return from_interactions_df(
        implicit_df,
        user_col='user_id',
        item_col='item_id'
    )


def filter_5core(ds: Dataset) -> Dataset:
    """Apply 5-core filtering: keep only users and items with at least 5 interactions."""
    # Get interaction data as DataFrame
    int_df = ds.interaction_table(format='pandas', original_ids=True)
    
    n_before = len(int_df)
    
    # Iteratively prune until stable
    while True:
        # Count interactions per user and per item
        user_counts = int_df['user_id'].value_counts()
        item_counts = int_df['item_id'].value_counts()
        
        # Find users and items with >= 5 interactions
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        
        # Filter
        filtered = int_df[
            int_df['user_id'].isin(valid_users) &
            int_df['item_id'].isin(valid_items)
        ]
        
        if len(filtered) == len(int_df):
            break
        int_df = filtered
    
    n_after = len(int_df)
    print(f"     5-core filtering: {n_before} -> {n_after} interactions")
    
    if n_after == 0:
        print("     WARNING: No interactions remain after 5-core filtering!")
        return ds
    
    # Decide which columns to keep
    keep_cols = ['user_id', 'item_id']
    if 'rating' in int_df.columns:
        keep_cols.append('rating')
    if 'timestamp' in int_df.columns:
        keep_cols.append('timestamp')
    if 'count' in int_df.columns:
        keep_cols.append('count')
    
    return from_interactions_df(
        int_df[keep_cols],
        user_col='user_id',
        item_col='item_id',
        rating_col='rating' if 'rating' in int_df.columns else None,
        timestamp_col='timestamp' if 'timestamp' in int_df.columns else None,
    )


def main():
    # We'll collect results across all seeds, datasets, and algorithms
    all_records: list[dict[str, Any]] = []

    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"Running with random seed = {seed}")
        print(f"{'='*80}")

        # Set the global random state for this seed
        set_global_rng(seed)

        for ds_name in DATASET_NAMES:
            print(f"\n  ── Dataset: {ds_name} ──")

            # 1) Load raw dataset
            dataset = load_dataset(ds_name)

            # 2) Convert explicit to implicit (for MovieLens and Amazon)
            if ds_name != 'HetrecLastFM':
                print(f"     Converting explicit ratings to implicit ...")
                dataset = make_implicit(dataset, min_rating=4)
            else:
                print(f"     Last.FM is already implicit, skipping conversion.")

            # 3) Apply 5-core filtering
            print(f"     Applying 5-core filter ...")
            dataset = filter_5core(dataset)

            # 4) Apply user-based 80/20 holdout split using LensKit's crossfold_users
            #    We take the first (and only) fold from crossfold_users with 1 fold
            print(f"     Splitting 80/20 per user ...")
            splits = list(crossfold_users(dataset, 1, SampleFrac(0.2), rng=seed))
            if not splits:
                print(f"     ERROR: No valid split produced!")
                continue
            split = splits[0]
            print(f"     Train: {split.train.interaction_count} interactions, "
                  f"Test: {split.test_size} interactions")

            # 5) For each algorithm, train, recommend, and evaluate
            for algo_name, algo_cfg in ALGORITHM_CONFIGS.items():
                print(f"       Algorithm: {algo_name} ...")

                try:
                    # Create scorer and pipeline
                    scorer_class: type[Any] = algo_cfg['scorer_class']
                    scorer_kwargs: dict[str, Any] = algo_cfg['scorer_kwargs']
                    pipeline_kwargs: dict[str, Any] = algo_cfg['pipeline_kwargs']
                    
                    scorer = scorer_class(**scorer_kwargs)
                    pipeline = topn_pipeline(scorer, **pipeline_kwargs)

                    # Train on the training set
                    pipeline.train(split.train)

                    # Generate recommendations (top-10 to cover all k values)
                    recs = recommend(pipeline, split.test.keys(), n=10)

                    # Evaluate with NDCG@k and Precision@k for k=1, 5, 10
                    for k in [1, 5, 10]:
                        # Create a fresh MeasurementCollector for each k value
                        mc = MeasurementCollector()
                        mc.add_metric(NDCG(n=k))
                        mc.add_metric(Precision(n=k))

                        # Measure - use measure_run which returns a RunMetrics named tuple
                        result = mc.measure_run(recs, split.test)
                        summary = result.summary_metrics
                        list_metrics = result.list_metrics

                        # Extract the mean values from summary
                        ndcg_mean: float | None = None
                        prec_mean: float | None = None
                        for key, val in summary.items():
                            if key.startswith('NDCG') and (f'@{k}' in key or key == 'NDCG'):
                                if k == 10 or (k == 1 and '@1' in key) or (k == 5 and '@5' in key) or (k == 10 and '@10' in key):
                                    ndcg_mean = float(val)
                            if key.startswith('Precision') and (f'@{k}' in key or key.startswith('Precision')):
                                prec_mean = float(val)

                        # Fallback: if summary keys don't match, try list_metrics
                        if ndcg_mean is None:
                            try:
                                ndcg_mean = float(list_metrics[f'NDCG@{k}'].mean())
                            except (KeyError, TypeError):
                                # Try without @k suffix
                                try:
                                    ndcg_mean = float(list_metrics['NDCG'].mean())
                                except (KeyError, TypeError):
                                    ndcg_mean = float('nan')

                        if prec_mean is None:
                            try:
                                prec_mean = float(list_metrics[f'Precision@{k}'].mean())
                            except (KeyError, TypeError):
                                try:
                                    prec_mean = float(list_metrics['Precision'].mean())
                                except (KeyError, TypeError):
                                    prec_mean = float('nan')

                        # Record
                        all_records.append({
                            'seed': seed,
                            'dataset': ds_name,
                            'algorithm': algo_name,
                            'metric': 'NDCG',
                            'k': k,
                            'value': ndcg_mean if ndcg_mean is not None else float('nan'),
                        })
                        all_records.append({
                            'seed': seed,
                            'dataset': ds_name,
                            'algorithm': algo_name,
                            'metric': 'Precision',
                            'k': k,
                            'value': prec_mean if prec_mean is not None else float('nan'),
                        })

                    print(f"         ✓ Done")

                except Exception as e:
                    print(f"         ✗ Error: {e}")
                    # Record NaN for all k values
                    for k in [1, 5, 10]:
                        all_records.append({
                            'seed': seed,
                            'dataset': ds_name,
                            'algorithm': algo_name,
                            'metric': 'NDCG',
                            'k': k,
                            'value': float('nan'),
                        })
                        all_records.append({
                            'seed': seed,
                            'dataset': ds_name,
                            'algorithm': algo_name,
                            'metric': 'Precision',
                            'k': k,
                            'value': float('nan'),
                        })

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
    print("\nComputing aggregate statistics (mean ± std across 5 seeds) ...\n")

    # Group by dataset, algorithm, metric, k
    grouped = results_df.groupby(["dataset", "algorithm", "metric", "k"])["value"]

    stats = grouped.agg(["mean", "std"]).reset_index()
    # Filter out NaN means for display
    stats_display = stats.dropna(subset=["mean"])

    # Print summary table
    print(f"{'Dataset':<22} {'Algorithm':<10} {'Metric':<12} {'k':<4} {'Mean':<10} {'Std':<10}")
    print("-" * 70)
    for _, row in stats_display.iterrows():
        print(f"{row['dataset']:<22} {row['algorithm']:<10} {row['metric']:<12} {row['k']:<4} {row['mean']:<10.6f} {row['std']:<10.6f}")

    # ── Statistical analysis ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Statistical Analysis: Impact of Seed Variation")
    print(f"{'='*80}")

    # Coefficient of Variation (CV) = std / mean  (as a measure of seed sensitivity)
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

    # Also save per-seed raw results
    raw_path = os.path.join(WORKING_DIR, "seed_variation_raw_results.csv")
    results_df.to_csv(raw_path, index=False)
    print(f"Raw per-seed results saved to: {raw_path}")

    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
