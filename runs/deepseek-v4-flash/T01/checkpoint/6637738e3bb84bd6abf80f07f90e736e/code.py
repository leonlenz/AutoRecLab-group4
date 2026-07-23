"""
Experiment: Quantify the effect of data split random seeds on recommender system accuracy.
Uses LensKit for data loading, preprocessing, model training, and evaluation.
"""

import os
import sys
import warnings
from typing import Any

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── LensKit Imports ─────────────────────────────────────────────────────────
from lenskit.data import Dataset, DatasetBuilder, from_interactions_df
from lenskit.data import load_movielens
from lenskit.splitting import sample_users, SampleFrac
from lenskit.basic import PopScorer
from lenskit.knn import ItemKNNScorer
from lenskit.als import ImplicitMFScorer
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend

# ─── LensKit Evaluation Imports ──────────────────────────────────────────────
from lenskit.metrics.ranking import NDCG, Precision
from lenskit.metrics.bulk import RunAnalysis

# ─── Setup Working Directory ──────────────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
ALGORITHM_BUILDERS = {
    "PopScorer": lambda: PopScorer(),
    "ItemKNNScorer": lambda: ItemKNNScorer(feedback="implicit"),
    "ImplicitMFScorer": lambda: ImplicitMFScorer(),
}

# Results tracking
results_rows: list[dict[str, Any]] = []


def download_movielens_100k() -> str:
    """Download MovieLens 100K dataset if not already present."""
    data_dir = os.path.join(working_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    ml_path = os.path.join(data_dir, "ml-100k.zip")
    if not os.path.exists(ml_path):
        import urllib.request
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        print(f"  Downloading MovieLens 100K from {url}...")
        urllib.request.urlretrieve(url, ml_path)
        print("  Download complete.")
    return ml_path


def download_amazon_video_games() -> str:
    """Download Amazon Video Games 2014 dataset if not already present."""
    data_dir = os.path.join(working_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    az_path = os.path.join(data_dir, "amazon_video_games.csv")
    if not os.path.exists(az_path):
        import urllib.request
        url = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv"
        print(f"  Downloading Amazon Video Games from {url}...")
        urllib.request.urlretrieve(url, az_path)
        print("  Download complete.")
    return az_path


def download_lastfm() -> str:
    """Download HetRec Last.FM dataset from the GroupLens mirror."""
    data_dir = os.path.join(working_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    lfm_path = os.path.join(data_dir, "lastfm_user_artists.dat")
    if not os.path.exists(lfm_path):
        import urllib.request
        # Use the original HetRec 2011 Last.FM dataset source
        url = "https://raw.githubusercontent.com/mingfeiwang/OmniRec/main/data/hetrec2011-lastfm/user_artists.dat"
        print(f"  Downloading Last.FM from {url}...")
        try:
            urllib.request.urlretrieve(url, lfm_path)
            print("  Download complete.")
        except Exception:
            # Fallback URL from GroupLens
            url2 = "http://www.cs.cornell.edu/~shuochen/lme/data/user_artists.dat"
            print(f"  Trying fallback URL: {url2}...")
            urllib.request.urlretrieve(url2, lfm_path)
            print("  Download complete.")
    return lfm_path


def load_movielens_dataset() -> Dataset:
    """Load MovieLens 100K dataset."""
    ml_path = download_movielens_100k()
    print(f"  Loading MovieLens 100K from {ml_path}")
    return load_movielens(ml_path)


def load_amazon_dataset() -> Dataset:
    """Load Amazon Video Games 2014 dataset."""
    az_path = download_amazon_video_games()
    print(f"  Loading Amazon Video Games from {az_path}")
    # Load the Amazon CSV directly
    az_df = pd.read_csv(az_path, sep=",", header=0, names=["item_id", "user_id", "rating", "timestamp"])
    return from_interactions_df(az_df)


def load_lastfm_dataset() -> Dataset:
    """Load HetRec Last.FM dataset from user_artists.dat.
    
    The file has format: userID, artistID, weight (play count).
    We treat play count > 0 as an implicit feedback signal.
    """
    lfm_path = download_lastfm()
    print(f"  Loading Last.FM from {lfm_path}")
    df = pd.read_csv(lfm_path, sep="\t", header=0, names=["user_id", "item_id", "rating"])
    # Last.FM already has implicit feedback (play counts > 0)
    # Convert to binary implicit feedback: any positive play count = 1
    df["rating"] = 1
    return from_interactions_df(df)


def convert_to_implicit(dataset: Dataset, threshold: float = 3.0) -> Dataset:
    """Convert explicit ratings to implicit by keeping only ratings > threshold
    and setting them to 1."""
    # Get interaction table as DataFrame
    df = dataset.interaction_table(format="pandas", original_ids=True)
    
    # Check if there's a rating column
    if "rating" not in df.columns:
        return dataset
    
    # Keep only ratings > threshold
    before = len(df)
    df = df[df["rating"] > threshold].copy()
    # Set to 1 for implicit feedback (do NOT drop the column)
    df["rating"] = 1
    after = len(df)
    print(f"    Implicit conversion: {before} -> {after} interactions (rating > {threshold}, set to 1)")
    
    if after == 0:
        print(f"    WARNING: No interactions remain after implicit conversion!")
        return dataset
    
    # Build new dataset
    return from_interactions_df(df)


def apply_k_core(dataset: Dataset, k: int = 5) -> Dataset:
    """Apply k-core filtering: keep only users and items with at least k interactions."""
    df = dataset.interaction_table(format="pandas", original_ids=True)
    
    print(f"    Applying {k}-core filtering...")
    print(f"    Interactions before: {len(df)}")
    
    # Iteratively prune until stable
    changed = True
    while changed:
        changed = False
        # Count interactions per user
        user_counts = df["user_id"].value_counts()
        users_to_keep = user_counts[user_counts >= k].index
        before = len(df)
        df = df[df["user_id"].isin(users_to_keep)]
        if len(df) < before:
            changed = True
        
        # Count interactions per item
        item_counts = df["item_id"].value_counts()
        items_to_keep = item_counts[item_counts >= k].index
        before = len(df)
        df = df[df["item_id"].isin(items_to_keep)]
        if len(df) < before:
            changed = True
    
    print(f"    Interactions after: {len(df)}")
    
    if len(df) == 0:
        print(f"    WARNING: No interactions remain after {k}-core filtering!")
        return dataset
    
    return from_interactions_df(df)


def preprocess_dataset(
    dataset_name: str,
    seed: int,
) -> Dataset:
    """Load and preprocess a dataset for a specific random seed.
    
    Steps:
        1. Load raw dataset
        2. For MovieLens100K and Amazon: convert ratings > 3 to implicit (set to 1)
        3. Apply 5-core filtering
    """
    # Load the raw dataset
    if dataset_name == "MovieLens100K":
        dataset = load_movielens_dataset()
    elif dataset_name == "Amazon2014VideoGames":
        dataset = load_amazon_dataset()
    elif dataset_name == "HetrecLastFM":
        dataset = load_lastfm_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"    Loaded")
    
    # MovieLens100K and Amazon have explicit ratings -> convert to implicit
    if dataset_name in ("MovieLens100K", "Amazon2014VideoGames"):
        dataset = convert_to_implicit(dataset, threshold=3.0)
    
    # 5-core filtering
    dataset = apply_k_core(dataset, k=5)
    
    df_after = dataset.interaction_table(format="pandas", original_ids=True)
    print(f"    After preprocessing: {len(df_after)} interactions, "
          f"{df_after['user_id'].nunique()} users, {df_after['item_id'].nunique()} items")
    
    return dataset


def run_experiment_for_seed(seed: int) -> None:
    """Run all algorithms on all datasets for a given random seed."""
    print(f"\n{'='*70}")
    print(f"  RUNNING EXPERIMENTS FOR SEED = {seed}")
    print(f"{'='*70}\n")
    
    datasets_info = [
        ("MovieLens100K", "MovieLens100K"),
        ("Amazon2014VideoGames", "Amazon2014VideoGames"),
        ("HetrecLastFM", "HetrecLastFM"),
    ]
    
    for dataset_label, dataset_key in datasets_info:
        print(f"\n  --- Processing {dataset_label} with seed {seed} ---")
        
        # Preprocess the dataset
        try:
            dataset = preprocess_dataset(dataset_key, seed)
        except Exception as e:
            print(f"  ERROR preprocessing {dataset_label} with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        df_all = dataset.interaction_table(format="pandas", original_ids=True)
        if len(df_all) == 0:
            print(f"  SKIP {dataset_label} with seed {seed}: empty dataset after preprocessing")
            continue
        
        n_users = df_all["user_id"].nunique()
        n_test_users = max(1, int(0.2 * n_users))
        print(f"    Splitting: {n_test_users} test users out of {n_users}")
        
        try:
            split = sample_users(
                dataset,
                n_test_users,
                SampleFrac(0.2),
                rng=seed,
            )
        except Exception as e:
            print(f"    ERROR splitting {dataset_label} with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        train = split.train
        test = split.test
        
        train_df = train.interaction_table(format="pandas", original_ids=True)
        print(f"    Train: {len(train_df)} interactions, "
              f"{train_df['user_id'].nunique()} users, {train_df['item_id'].nunique()} items")
        print(f"    Test:  {len(test)} users")
        
        # Run each algorithm
        for algo_name, algo_builder in ALGORITHM_BUILDERS.items():
            print(f"\n    --- Algorithm: {algo_name} ---")
            
            try:
                # Build the scorer and pipeline
                scorer = algo_builder()
                # For implicit feedback, we don't set predicts_ratings=True
                pipeline = topn_pipeline(scorer, n=10)
                
                # Train
                pipeline.train(train)
                
                # Generate recommendations for all test users (use test.keys() for user list)
                recs = recommend(pipeline, test, n=10)
                
                # Evaluate using LensKit RunAnalysis with NDCG and Precision
                ra = RunAnalysis()
                # Add metrics for each k value
                ra.add_metric(NDCG(n=1))
                ra.add_metric(NDCG(n=5))
                ra.add_metric(NDCG(n=10))
                ra.add_metric(Precision(n=1))
                ra.add_metric(Precision(n=5))
                ra.add_metric(Precision(n=10))
                
                metrics = ra.measure(recs, test)
                summary = metrics.list_summary()
                
                # Extract results from the summary
                for metric_name in ["NDCG", "Precision"]:
                    for k_val in [1, 5, 10]:
                        label = f"{metric_name}@{k_val}"
                        if label in summary.index:
                            val = summary.loc[label, "mean"]
                            results_rows.append({
                                "dataset": dataset_label,
                                "algorithm": algo_name,
                                "seed": seed,
                                "metric": metric_name,
                                "k": k_val,
                                "stat": "mean",
                                "value": val,
                            })
                            print(f"        {label}={val:.6f}")
                        elif metric_name in summary.index and k_val == 10:
                            # Some metrics may not have @k suffix if n=None
                            pass
                
                print(f"      ✓ Completed {algo_name}")
                
            except Exception as e:
                print(f"      ✗ Error running {algo_name} on {dataset_label} seed {seed}: {e}")
                import traceback
                traceback.print_exc()


def collect_and_analyze_results() -> None:
    """Collect results from tracked data and compute statistics."""
    print(f"\n{'='*70}")
    print(f"  COLLECTING AND ANALYZING RESULTS")
    print(f"{'='*70}\n")
    
    if not results_rows:
        print("No results collected. Something went wrong.")
        return
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(results_rows)
    
    # Filter to mean statistics
    mean_df = metrics_df[metrics_df["stat"] == "mean"].copy()
    
    print(f"Collected {len(mean_df)} metric records.\n")
    
    # Compute mean and std across seeds for each (dataset, algorithm, metric, k)
    print("=" * 90)
    print("  RESULTS: Mean ± Std across 5 random seeds")
    print("=" * 90)
    
    for dataset_name in sorted(mean_df["dataset"].unique()):
        print(f"\n{'─'*90}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'─'*90}")
        
        ds_df = mean_df[mean_df["dataset"] == dataset_name]
        
        for algo_name in sorted(ds_df["algorithm"].unique()):
            print(f"\n  Algorithm: {algo_name}")
            algo_df = ds_df[ds_df["algorithm"] == algo_name]
            
            for metric_name in sorted(algo_df["metric"].unique()):
                metric_df = algo_df[algo_df["metric"] == metric_name]
                k_values = sorted(metric_df["k"].unique())
                
                for k_val in k_values:
                    k_df = metric_df[metric_df["k"] == k_val]
                    values = k_df["value"].values
                    
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                        print(f"    {metric_name}@{k_val}: {mean_val:.6f} ± {std_val:.6f} "
                              f"(n={len(values)} seeds)")


def main():
    print("╔" + "═" * 78 + "╗")
    print("║  Experiment: Quantifying Data Split Random Seed Effects on RS Accuracy  ║")
    print("╚" + "═" * 78 + "╝")
    print(f"  Datasets: MovieLens100K, Amazon2014VideoGames, HetrecLastFM")
    print(f"  Algorithms: PopScorer, ItemKNNScorer, ImplicitMFScorer")
    print(f"  Seeds: {SEEDS}")
    print(f"  Metrics: NDCG@[1,5,10], Precision@[1,5,10]")
    print(f"  Working Dir: {working_dir}")
    
    # Run experiments for each seed
    for seed in SEEDS:
        run_experiment_for_seed(seed)
    
    # Analyze results
    collect_and_analyze_results()
    
    print(f"\n{'='*70}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
