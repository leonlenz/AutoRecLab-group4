"""
Experiment: Impact of Data Split Random Seeds on Recommender System Accuracy

Runs 3 algorithms (ALS/ImplicitMF, ItemKNN, Pop) x 3 datasets (MovieLens100K,
Amazon Video Games 2014, HetRec LastFM) x 5 random seeds = 45 runs.

Uses LensKit for all operations: data loading, preprocessing, splitting,
training, recommendation, and evaluation.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# ── LensKit imports ──────────────────────────────────────────────
from lenskit.data import (
    Dataset,
    DatasetBuilder,
    from_interactions_df,
    load_movielens,
    load_amazon_ratings,
    ItemListCollection,
    UserIDKey,
)
from lenskit.splitting import crossfold_users, SampleFrac, TTSplit
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend as batch_recommend
from lenskit.als import ImplicitMFScorer
from lenskit.knn import ItemKNNScorer
from lenskit.basic import PopScorer
from lenskit.metrics import MeasurementCollector
from lenskit.metrics.ranking import NDCG, Precision


def iterative_5core_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iteratively apply 5-core filtering: keep only users and items
    with at least 5 interactions. Repeat until convergence.
    
    Args:
        df: DataFrame with columns 'user_id' and 'item_id'
        
    Returns:
        Filtered DataFrame
    """
    changed = True
    while changed:
        changed = False
        
        # Count interactions per user
        user_counts = df['user_id'].value_counts()
        users_to_keep = user_counts[user_counts >= 5].index
        before = len(df)
        df = df[df['user_id'].isin(users_to_keep)]
        if len(df) < before:
            changed = True
        
        # Count interactions per item
        item_counts = df['item_id'].value_counts()
        items_to_keep = item_counts[item_counts >= 5].index
        before = len(df)
        df = df[df['item_id'].isin(items_to_keep)]
        if len(df) < before:
            changed = True
    
    return df


def load_and_preprocess_movielens() -> Dataset:
    """
    Load MovieLens 100K, apply rating > 3 implicit conversion, and 5-core filtering.
    
    Returns:
        Preprocessed LensKit Dataset
    """
    print("\n  Loading MovieLens 100K...")
    
    # Load raw ratings as DataFrame
    df = lenskit.data.load_movielens_df("data/ml-100k.zip")
    print(f"    Original size: {len(df)} interactions")
    
    # Filter: keep only ratings > 3
    before = len(df)
    df = df[df['rating'] > 3]
    print(f"    After rating > 3 filter: {len(df)} interactions (removed {before - len(df)})")
    
    # Drop rating column (implicit feedback)
    df = df.drop(columns=['rating'])
    
    # Apply 5-core filtering
    df = iterative_5core_filter(df)
    print(f"    After 5-core filtering: {len(df)} interactions")
    print(f"    Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
    
    # Build Dataset using DatasetBuilder with binarization
    dsb = DatasetBuilder()
    dsb.add_interactions(
        "rating",
        df,
        entities=["user", "item"],
        missing="insert",
        default=True,
    )
    ds = dsb.build()
    return ds


def load_and_preprocess_amazon() -> Dataset:
    """
    Load Amazon Video Games 2014, apply rating > 3 implicit conversion, and 5-core filtering.
    
    Returns:
        Preprocessed LensKit Dataset
    """
    print("\n  Loading Amazon Video Games 2014...")
    
    # Load raw ratings as a Dataset first, then convert to DataFrame
    raw_ds = load_amazon_ratings("data/az14/ratings_Video_Games.csv")
    print(f"    Original size: {raw_ds.interaction_count} interactions")
    
    # Get interactions as DataFrame
    df = raw_ds.interactions().pandas(ids=True)
    df = df.rename(columns={"user_num": "user_id", "item_num": "item_id"})
    
    # Filter: keep only ratings > 3
    before = len(df)
    df = df[df['rating'] > 3]
    print(f"    After rating > 3 filter: {len(df)} interactions (removed {before - len(df)})")
    
    # Drop rating column (implicit feedback)
    df = df.drop(columns=['rating'])
    
    # Apply 5-core filtering
    df = iterative_5core_filter(df)
    print(f"    After 5-core filtering: {len(df)} interactions")
    print(f"    Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
    
    # Build Dataset
    ds = from_interactions_df(df)
    return ds


def load_and_preprocess_lastfm() -> Dataset:
    """
    Load HetRec LastFM (already implicit), apply 5-core filtering.
    
    Returns:
        Preprocessed LensKit Dataset
    """
    print("\n  Loading HetRec LastFM...")
    
    # Load LastFM data from CSV
    df = pd.read_csv(
        "data/lastfm/user_artists.dat",
        sep='\t',
        header=0,
        names=['user_id', 'item_id', 'weight']
    )
    print(f"    Original size: {len(df)} interactions")
    
    # Drop weight column (implicit feedback - just presence matters)
    df = df.drop(columns=['weight'])
    
    # Apply 5-core filtering
    df = iterative_5core_filter(df)
    print(f"    After 5-core filtering: {len(df)} interactions")
    print(f"    Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
    
    # Build Dataset
    ds = from_interactions_df(df)
    return ds


def evaluate_run(recs, truths):
    """
    Evaluate recommendations using NDCG and Precision at k=1, 5, 10.
    
    Uses MeasurementCollector.add_collection_measurements() and
    summary_metrics() to compute aggregate metrics across all lists.

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
    
    # Measure using add_collection_measurements + summary_metrics
    collector.add_collection_measurements(recs, truths)
    return collector.summary_metrics()


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
        ("MovieLens100K", "ml100k"),
        ("Amazon Video Games 2014", "amazon"),
        ("HetRec LastFM", "lastfm"),
    ]
    
    algorithms = ['ImplicitMF', 'ItemKNN', 'Pop']
    seeds = [42, 123, 456, 789, 1024]
    topn = 10
    
    # Store results: dict with keys (algo, dataset, metric_name) -> list of 5 values
    all_results = {}
    
    # Main experiment loop
    for ds_label, ds_key in datasets_config:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_label}")
        print(f"{'#'*70}")
        
        # Preprocess once (same preprocessing for all seeds)
        if ds_key == 'ml100k':
            ds = load_and_preprocess_movielens()
        elif ds_key == 'amazon':
            ds = load_and_preprocess_amazon()
        elif ds_key == 'lastfm':
            ds = load_and_preprocess_lastfm()
        else:
            raise ValueError(f"Unknown dataset key: {ds_key}")
        
        print(f"\n  Running {len(seeds)} seed splits x {len(algorithms)} algorithms")
        
        for algo_name in algorithms:
            print(f"\n  --- Algorithm: {algo_name} ---")
            
            for seed_idx, seed in enumerate(seeds):
                print(f"    Seed {seed} (run {seed_idx+1}/{len(seeds)})...", end=" ", flush=True)
                
                # User-based 80/20 holdout using crossfold_users with 1 fold
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
                  f"Mean={r['Mean']:.4f} +/- {r['Std']:.4f} (CV={r['CV']:.2%})")
        
        print()
        
        # Least stable (highest CV)
        unstable = results_df.dropna(subset=['CV']).nlargest(min(5, len(cv_valid)), 'CV')
        print("  Least stable (most affected by seed):")
        for _, r in unstable.iterrows():
            print(f"    - {r['Algorithm']:12s} | {r['Dataset']:25s} | {r['Metric']:20s} | "
                  f"Mean={r['Mean']:.4f} +/- {r['Std']:.4f} (CV={r['CV']:.2%})")
        
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
        for ds_label, _ in datasets_config:
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
    # Ensure lenskit is imported at module level
    import lenskit.data
    main()
