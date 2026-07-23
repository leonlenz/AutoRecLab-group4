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
from lenskit import topn_pipeline, batch
from lenskit.als import ImplicitMFScorer
from lenskit.knn import ItemKNNScorer
from lenskit.basic import PopScorer
from lenskit.metrics import MeasurementCollector, NDCG, Precision


def preprocess_dataset(dataset_name: str, make_implicit: bool = True):
    """
    Load and preprocess a dataset using OmniRec.
    
    Args:
        dataset_name: One of 'ml100k', 'amazon', 'lastfm'
        make_implicit: Whether to convert explicit ratings to implicit (ratings > 3)
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with columns user, item, rating, timestamp
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
        steps.append(MakeImplicit(3))  # ratings >= 3 -> implicit
    steps.append(CorePruning(5))       # 5-core filtering
    
    pipeline = Pipe(*steps)
    processed = pipeline.process(ds)
    
    # Extract the dataframe
    df = processed._data.df.copy()
    print(f"  After preprocessing: {len(df)} interactions")
    print(f"  Users: {df['user'].nunique()}, Items: {df['item'].nunique()}")
    
    return df


def run_single_experiment(train_df, test_df, algo_name: str, topn: int = 10):
    """
    Train a single algorithm on train data and evaluate on test data.
    
    Args:
        train_df: Training DataFrame (columns: user, item, rating)
        test_df: Testing DataFrame (columns: user, item, rating)
        algo_name: 'ImplicitMF', 'ItemKNN', or 'Pop'
        topn: Number of recommendations to generate
    
    Returns:
        tuple: (recommendations DataFrame, truths ItemListCollection)
    """
    # Convert to LensKit Dataset
    train_ds = from_interactions_df(train_df)
    
    # Build test ItemListCollection
    test_ilc = ItemListCollection.from_df(
        test_df.rename(columns={"user": "user_id", "item": "item_id"}),
        key="user_id"
    )
    
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
    recs = batch.recommend(pipe, test_ilc.keys(), n=topn)
    
    return recs, test_ilc


def evaluate_run(recs, truths):
    """
    Evaluate recommendations using NDCG and Precision at k=1, 5, 10.
    
    Returns:
        dict: Summary metrics with keys like "NDCG.1.mean", "Precision.10.mean", etc.
    """
    collector = MeasurementCollector()
    for k in [1, 5, 10]:
        collector.add_metric(NDCG(n=k))
        collector.add_metric(Precision(n=k))
    
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
    
    # Store results: dict with keys (algo, dataset, metric_name) -> list of seed values
    all_results = {}  # (algo_name, dataset_label, metric) -> list of 5 values
    
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
                
                # Create LensKit Dataset
                ds = from_interactions_df(df)
                
                # User-based 80/20 holdout using crossfold_users with 1 fold
                # SampleFrac(0.2) takes 20% of each user's items for testing
                splits = list(crossfold_users(ds, 1, SampleFrac(0.2), rng=seed))
                split = splits[0]
                
                # Get train and test dataframes
                train_ds = split.train
                test_ilc = split.test
                
                # Get test items as DataFrame for reference
                test_df = test_ilc.to_df()
                
                # Rename columns for from_interactions_df (needs user_id, item_id)
                train_df_pd = train_ds.interaction_table(
                    format="pandas", original_ids=True
                )
                
                # Filter to only test users that are in training
                test_users_set = set(test_ilc.keys())
                train_users_set = set(train_df_pd["user_id"].unique())
                common_users = test_users_set & train_users_set
                
                if len(common_users) < len(test_users_set):
                    print(f"(filtered {len(test_users_set) - len(common_users)} new users)", end=" ", flush=True)
                
                # Train and recommend
                try:
                    # Build scorer
                    if algo_name == 'ImplicitMF':
                        scorer = ImplicitMFScorer()
                    elif algo_name == 'ItemKNN':
                        scorer = ItemKNNScorer()
                    else:
                        scorer = PopScorer()
                    
                    # Build and train pipeline
                    pipe = topn_pipeline(scorer, n=topn)
                    pipe.train(train_ds)
                    
                    # Generate recommendations for test users
                    recs = batch.recommend(pipe, test_ilc.keys(), n=topn)
                    
                    # Evaluate
                    collector = MeasurementCollector()
                    for k in [1, 5, 10]:
                        collector.add_metric(NDCG(n=k))
                        collector.add_metric(Precision(n=k))
                    
                    result = collector.measure_run(recs, test_ilc)
                    summary = dict(result.summary_metrics)
                    
                    # Store results
                    for metric_name, metric_val in summary.items():
                        key = (algo_name, ds_label, metric_name)
                        if key not in all_results:
                            all_results[key] = []
                        all_results[key].append(metric_val)
                    
                    print(f"NDCG@10={summary.get('NDCG.10.mean', 'N/A'):.4f}")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
    
    # ── Aggregate Results ──────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("A GGREGATED RESULTS")
    print(f"{'='*70}")
    
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
    
    # Find the most and least stable combinations
    metric_cols = results_df[results_df['Metric'].str.contains('.mean$', regex=False) == False]
    
    # Calculate coefficient of variation where mean > 0
    results_df['CV'] = results_df.apply(
        lambda r: r['Std'] / r['Mean'] if r['Mean'] > 0 else np.nan, axis=1
    )
    
    # Most stable (lowest CV)
    stable = results_df.dropna(subset=['CV']).nsmallest(5, 'CV')
    print("  Most stable (least affected by seed):")
    for _, r in stable.iterrows():
        print(f"    - {r['Algorithm']:12s} | {r['Dataset']:25s} | {r['Metric']:20s} | "
              f"Mean={r['Mean']:.4f} ± {r['Std']:.4f} (CV={r['CV']:.2%})")
    
    print()
    
    # Least stable (highest CV)
    unstable = results_df.dropna(subset=['CV']).nlargest(5, 'CV')
    print("  Least stable (most affected by seed):")
    for _, r in unstable.iterrows():
        print(f"    - {r['Algorithm']:12s} | {r['Dataset']:25s} | {r['Metric']:20s} | "
              f"Mean={r['Mean']:.4f} ± {r['Std']:.4f} (CV={r['CV']:.2%})")
    
    print()
    print("  Summary statistics across all combinations:")
    cv_values = results_df['CV'].dropna()
    print(f"    Mean CV: {cv_values.mean():.2%}")
    print(f"    Median CV: {cv_values.median():.2%}")
    print(f"    Std CV: {cv_values.std():.2%}")
    print(f"    Min CV: {cv_values.min():.2%}")
    print(f"    Max CV: {cv_values.max():.2%}")
    
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
    
    # Save results
    output_path = os.path.join(working_dir, 'experiment_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    
    print("\nExperiment complete!")


if __name__ == '__main__':
    main()
