"""
Experiment: Impact of Data Split Random Seeds on Recommender System Accuracy

Runs 3 algorithms (ALS/ImplicitMF, ItemKNN, Pop) × 3 datasets (MovieLens100K,
Amazon Video Games 2014, HetRec LastFM) × 5 random seeds = 45 runs.

Uses OmniRec for data loading/preprocessing and LensKit for splitting,
training, recommendation, and evaluation.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# ── OmniRec imports ──────────────────────────────────────────────
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.pipe import Pipe

# ── LensKit imports ──────────────────────────────────────────────
from lenskit.data import from_interactions_df, ItemListCollection
from lenskit.splitting import crossfold_users, SampleFrac
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend as batch_recommend
from lenskit.als import ImplicitMFScorer
from lenskit.knn import ItemKNNScorer
from lenskit.basic import PopScorer
from lenskit.metrics import MeasurementCollector
from lenskit.metrics.ranking import NDCG, Precision


def preprocess_dataset(dataset_name: str, make_implicit: bool = True):
    """
    Load and preprocess a dataset using OmniRec.
    
    Args:
        dataset_name: One of 'ml100k', 'amazon', 'lastfm'
        make_implicit: Whether to convert explicit ratings to implicit (ratings > 3)
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with columns user_id, item_id (and timestamp if present)
    """
    print(f"\n{'='*60}")
    print(f"Loading and preprocessing: {dataset_name}")
    print(f"{'='*60}")
    
    if dataset_name == 'ml100k':
        ds = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    elif dataset_name == 'amazon':
        ds = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
    elif dataset_name == 'lastfm':
        ds = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"  Original size: {ds.num_interactions()} interactions")
    
    # Build preprocessing pipeline
    steps = []
    if make_implicit:
        # Use threshold 3 to keep ratings >= 3, then manually filter out == 3
        # to ensure we keep ratings strictly > 3 (handles non-integer ratings like 3.5)
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))  # 5-core filtering
    
    pipeline = Pipe(*steps)
    processed = pipeline.process(ds)
    
    # Extract the dataframe from the RawData variant
    # OmniRec RawData stores DataFrame in _data.df with columns: user, item, rating, timestamp
    df = processed._data.df.copy()
    
    # After MakeImplicit with threshold 3, ratings >= 3 are kept.
    # For strict "ratings > 3" requirement (rating > 3, not >= 3),
    # we need to filter out rating == 3.
    # Note: After MakeImplicit, the 'rating' column may have been dropped
    # (MakeImplicit only keeps user, item, timestamp). So we need to check.
    if make_implicit and 'rating' in df.columns:
        # If rating column still exists, filter out exactly 3
        before = len(df)
        df = df[df['rating'] > 3]
        after = len(df)
        if before > after:
            print(f"  Filtered out {before - after} interactions with rating == 3")
        # Drop rating column for implicit feedback
        df = df.drop(columns=['rating'])
    
    # Rename columns for LensKit compatibility
    # OmniRec uses 'user' and 'item', LensKit expects 'user_id' and 'item_id'
    column_map = {}
    if 'user' in df.columns:
        column_map['user'] = 'user_id'
    if 'item' in df.columns:
        column_map['item'] = 'item_id'
    if column_map:
        df = df.rename(columns=column_map)
    
    print(f"  After preprocessing: {len(df)} interactions")
    print(f"  Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
    if 'timestamp' in df.columns:
        print(f"  Timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def evaluate_run(recs, truths):
    """
    Evaluate recommendations using NDCG and Precision at k=1, 5, 10.
    
    Uses MeasurementCollector with measure_run to compute aggregate metrics.

    Args:
        recs: ItemListCollection of recommendations
        truths: ItemListCollection of test items
    
    Returns:
        dict: Summary metrics with keys like "NDCG.1.mean", "Precision.10.mean", etc.
    """
    # Create a fresh collector for this run
    collector = MeasurementCollector()
    for k in [1, 5, 10]:
        collector.add_metric(NDCG(n=k))
        collector.add_metric(Precision(n=k))
    
    # Measure the run and get summary metrics from the returned RunMetrics object
    result = collector.measure_run(recs, truths)
    return dict(result.summary_metrics)


def main():
    # Create working directory
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)
    
    print("=" * 70)
    print("EXPERIMENT: Impact of Data Split Random Seeds on RecSys Accuracy")
    print("=" * 70)
    
    # Define experiment configuration
    datasets_config = [
        ("MovieLens100K", "ml100k", True),
        ("Amazon Video Games 2014", "amazon", True),
        ("HetRec LastFM", "lastfm", False),
    ]
    
    algorithms = ['ImplicitMF', 'ItemKNN', 'Pop']
    seeds = [42, 123, 456, 789, 1024]
    topn = 10
    
    # Store results: dict with keys (algo, dataset, metric_name) -> list of 5 values
    all_results = {}
    
    # Main experiment loop
    for ds_label, ds_key, make_impl in datasets_config:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_label}")
        print(f"{'#'*70}")
        
        # Preprocess once (same preprocessing for all seeds)
        df = preprocess_dataset(ds_key, make_implicit=make_impl)
        
        print(f"\n  Running {len(seeds)} seed splits × {len(algorithms)} algorithms")
        
        for algo_name in algorithms:
            print(f"\n  --- Algorithm: {algo_name} ---")
            
            for seed_idx, seed in enumerate(seeds):
                print(f"    Seed {seed} (run {seed_idx+1}/{len(seeds)})...", end=" ", flush=True)
                
                # Create LensKit Dataset from the preprocessed DataFrame
                # from_interactions_df auto-detects user_id and item_id columns
                ds = from_interactions_df(df)
                
                # User-based 80/20 holdout using crossfold_users with 1 fold.
                # With 1 partition, all users are in the test set.
                # SampleFrac(0.2, rng=seed) randomly selects 20% of each user's items for testing.
                # The seed controls the random item selection, making each seed produce a different split.
                splits = list(crossfold_users(ds, 1, SampleFrac(0.2, rng=seed), rng=seed))
                split = splits[0]
                
                # Get train Dataset and test ItemListCollection
                train_ds = split.train
                test_ilc = split.test
                
                # Build scorer
                if algo_name == 'ImplicitMF':
                    scorer = ImplicitMFScorer()
                elif algo_name == 'ItemKNN':
                    scorer = ItemKNNScorer()
                elif algo_name == 'Pop':
                    scorer = PopScorer()
                else:
                    raise ValueError(f"Unknown algorithm: {algo_name}")
                
                # Build and train pipeline
                pipe = topn_pipeline(scorer, n=topn)
                pipe.train(train_ds)
                
                # Generate recommendations for all test users
                # split.test.keys() returns UserIDKey objects
                # batch_recommend accepts iterables of user IDs and ItemListCollection
                recs = batch_recommend(pipe, test_ilc.keys(), n=topn)
                
                # Evaluate using the collector-based approach
                summary = evaluate_run(recs, test_ilc)
                
                # Store results
                for metric_name, metric_val in summary.items():
                    key = (algo_name, ds_label, metric_name)
                    if key not in all_results:
                        all_results[key] = []
                    all_results[key].append(metric_val)
                
                # Print a quick summary metric
                ndcg10_key = "NDCG.10.mean"
                precision10_key = "Precision.10.mean"
                ndcg_val = summary.get(ndcg10_key, 'N/A')
                prec_val = summary.get(precision10_key, 'N/A')
                ndcg_str = f"{ndcg_val:.4f}" if isinstance(ndcg_val, float) else str(ndcg_val)
                prec_str = f"{prec_val:.4f}" if isinstance(prec_val, float) else str(prec_val)
                print(f"NDCG@10={ndcg_str}, Precision@10={prec_str}")
    
    # ── Aggregate Results ──────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")
    
    if not all_results:
        print("\nNo results were collected. Experiment failed to produce any metrics.")
        print("Check for errors above.")
        return
    
    # Build results table
    rows = []
    for (algo, ds_label, metric_name), values in sorted(all_results.items()):
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            rows.append({
                'Algorithm': algo,
                'Dataset': ds_label,
                'Metric': metric_name,
                'Mean': mean_val,
                'Std': std_val,
                'N_seeds': len(values),
                'Min': np.min(values),
                'Max': np.max(values),
            })
    
    results_df = pd.DataFrame(rows)
    
    if len(results_df) == 0:
        print("\nNo results to display.")
        return
    
    # Print summary table
    print(f"\n{'Algorithm':<15} {'Dataset':<25} {'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-'*100}")
    
    for _, row in results_df.iterrows():
        print(f"{row['Algorithm']:<15} {row['Dataset']:<25} {row['Metric']:<20} {row['Mean']:<10.4f} {row['Std']:<10.4f} {row['Min']:<10.4f} {row['Max']:<10.4f}")
    
    # ── Statistical Analysis ───────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("STATISTICAL ANALYSIS: Impact of Random Seed on Accuracy")
    print(f"{'='*70}")
    
    print("\nFor each (Algorithm, Dataset, Metric) combination, we computed")
    print("the mean and standard deviation across 5 different random seeds.")
    print()
    print("Key findings:")
    print()
    
    # Calculate coefficient of variation where mean > 0
    results_df['CV'] = results_df.apply(
        lambda r: r['Std'] / r['Mean'] if r['Mean'] > 0 else np.nan, axis=1
    )
    
    cv_valid = results_df['CV'].dropna()
    
    if len(cv_valid) > 0:
        # Most stable (lowest CV)
        stable = results_df.dropna(subset=['CV']).nsmallest(min(5, len(cv_valid)), 'CV')
        print("  Most stable (least affected by seed):")
        for _, r in stable.iterrows():
            print(f"    - {r['Algorithm']:12s} | {r['Dataset']:25s} | {r['Metric']:20s} | "
                  f"Mean={r['Mean']:.4f} ± {r['Std']:.4f} (CV={r['CV']:.2%})")
        
        print()
        
        # Least stable (highest CV)
        unstable = results_df.dropna(subset=['CV']).nlargest(min(5, len(cv_valid)), 'CV')
        print("  Least stable (most affected by seed):")
        for _, r in unstable.iterrows():
            print(f"    - {r['Algorithm']:12s} | {r['Dataset']:25s} | {r['Metric']:20s} | "
                  f"Mean={r['Mean']:.4f} ± {r['Std']:.4f} (CV={r['CV']:.2%})")
        
        print()
        print("  Summary statistics across all combinations:")
        print(f"    Mean CV: {cv_valid.mean():.2%}")
        print(f"    Median CV: {cv_valid.median():.2%}")
        print(f"    Std CV: {cv_valid.std():.2%}")
        print(f"    Min CV: {cv_valid.min():.2%}")
        print(f"    Max CV: {cv_valid.max():.2%}")
        
        # Per-algorithm analysis
        print(f"\n  Per-algorithm average CV:")
        for algo in algorithms:
            algo_cv = results_df[(results_df['Algorithm'] == algo) & results_df['CV'].notna()]['CV']
            if len(algo_cv) > 0:
                print(f"    {algo:15s}: mean CV = {algo_cv.mean():.2%}")
        
        # Per-dataset analysis
        print(f"\n  Per-dataset average CV:")
        for ds_label, _, _ in datasets_config:
            ds_cv = results_df[(results_df['Dataset'] == ds_label) & results_df['CV'].notna()]['CV']
            if len(ds_cv) > 0:
                print(f"    {ds_label:25s}: mean CV = {ds_cv.mean():.2%}")
    else:
        print("  No valid CV values to analyze.")
    
    # Save results
    output_path = os.path.join(working_dir, 'experiment_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    
    print("\nExperiment complete!")


if __name__ == '__main__':
    main()
