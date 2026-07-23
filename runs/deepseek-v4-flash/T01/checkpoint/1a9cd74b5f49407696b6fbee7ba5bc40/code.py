"""
Experiment: Quantify the effect of data split random seeds on recommender system accuracy.
Uses OmniRec for evaluation (Evaluator, NDCG, Precision).
"""

import os
import sys
import warnings
from typing import Any

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── LensKit Imports (data loading / splitting / training only) ──────────────
from lenskit.data import Dataset, DatasetBuilder, from_interactions_df
from lenskit.data import load_movielens, load_amazon_ratings
from lenskit.splitting import sample_users, SampleFrac
from lenskit.basic import PopScorer
from lenskit.knn import ItemKNNScorer
from lenskit.als import ImplicitMFScorer
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend

# ─── OmniRec Evaluation Imports ─────────────────────────────────────────────
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision

# ─── Setup Working Directory ──────────────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
ALGORITHM_BUILDERS = {
    "PopScorer": lambda: PopScorer(),
    "ItemKNNScorer": lambda: ItemKNNScorer(),
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
    """Download HetRec Last.FM dataset if not already present."""
    data_dir = os.path.join(working_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    lfm_path = os.path.join(data_dir, "lastfm_user_artists.dat")
    if not os.path.exists(lfm_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/ISG-Siegen/OmniRec/main/data/hetrec2011-lastfm/user_artists.dat"
        print(f"  Downloading Last.FM from {url}...")
        urllib.request.urlretrieve(url, lfm_path)
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
    return load_amazon_ratings(az_path)


def load_lastfm_dataset() -> Dataset:
    """Load HetRec Last.FM dataset from user_artists.dat.
    
    The file has format: userID, artistID, weight (play count).
    We treat the weight as an implicit feedback signal (1 if listened, 0 otherwise).
    """
    lfm_path = download_lastfm()
    print(f"  Loading Last.FM from {lfm_path}")
    df = pd.read_csv(lfm_path, sep="\t", header=0, names=["user_id", "item_id", "rating"])
    # Last.FM already has implicit feedback (play counts > 0)
    # Convert to binary implicit feedback
    df["rating"] = 1
    return from_interactions_df(df)


def convert_to_implicit(dataset: Dataset, threshold: float = 3.0) -> Dataset:
    """Convert explicit ratings to implicit by keeping only ratings > threshold."""
    # Get interaction table as DataFrame
    df = dataset.interaction_table(format="pandas", original_ids=True)
    
    # Check if there's a rating column
    if "rating" not in df.columns:
        return dataset
    
    # Keep only ratings > threshold
    before = len(df)
    df = df[df["rating"] > threshold].copy()
    df.drop(columns=["rating"], inplace=True)
    after = len(df)
    print(f"    Implicit conversion: {before} -> {after} interactions (rating > {threshold})")
    
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
        2. For MovieLens100K and Amazon: convert ratings > 3 to implicit
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
    
    print(f"    Loaded: {dataset.interaction_count} interactions, "
          f"{dataset.user_count} users, {dataset.item_count} items")
    
    # MovieLens100K and Amazon have explicit ratings -> convert to implicit
    if dataset_name in ("MovieLens100K", "Amazon2014VideoGames"):
        dataset = convert_to_implicit(dataset, threshold=3.0)
    
    # 5-core filtering
    dataset = apply_k_core(dataset, k=5)
    
    print(f"    After preprocessing: {dataset.interaction_count} interactions, "
          f"{dataset.user_count} users, {dataset.item_count} items")
    
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
        
        if dataset.interaction_count == 0:
            print(f"  SKIP {dataset_label} with seed {seed}: empty dataset after preprocessing")
            continue
        
        # Split: user-based 80/20 holdout
        # Use 20% of users as test users, and for each test user, take 20% of their items
        n_users = dataset.user_count
        n_test_users = max(1, int(0.2 * n_users))
        print(f"    Splitting: {n_test_users} test users out of {n_users}")
        
        try:
            split = sample_users(
                dataset,
                n_test_users,
                SampleFrac(0.2, rng=seed),
                rng=seed,
            )
        except Exception as e:
            print(f"    ERROR splitting {dataset_label} with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        train = split.train
        test = split.test
        
        print(f"    Train: {train.interaction_count} interactions, "
              f"{train.user_count} users, {train.item_count} items")
        print(f"    Test:  {len(test)} users")
        
        # Convert test to DataFrame for OmniRec evaluator
        test_df = test.to_df()
        test_df = test_df.rename(columns={"user_id": "user", "item_id": "item"})
        
        # Run each algorithm
        for algo_name, algo_builder in ALGORITHM_BUILDERS.items():
            print(f"\n    --- Algorithm: {algo_name} ---")
            
            try:
                # Build the scorer and pipeline
                scorer = algo_builder()
                pipeline = topn_pipeline(scorer)
                
                # Train
                pipeline.train(train)
                
                # Generate recommendations for all test users
                recs = recommend(pipeline, test.keys(), n=10)
                
                # Convert recommendations to DataFrame for OmniRec evaluator
                recs_df = recs.to_df()
                recs_df = recs_df.rename(columns={"user_id": "user", "item_id": "item"})
                
                # Evaluate using OmniRec Evaluator with NDCG and Precision
                evaluator = Evaluator(
                    NDCG([1, 5, 10]),
                    Precision([1, 5, 10]),
                )
                evaluator.run_evaluation(dataset_label, algo_name, recs_df, test_df)
                
                # Extract results from the evaluator
                results_dict = evaluator.get_results()
                df_results = results_dict[dataset_label]
                
                # Filter for current algorithm's results
                algo_results = df_results[df_results["algorithm"] == algo_name]
                
                for _, row in algo_results.iterrows():
                    results_rows.append({
                        "dataset": dataset_label,
                        "algorithm": algo_name,
                        "seed": seed,
                        "metric": row["name"],
                        "k": row["k"],
                        "stat": "mean",
                        "value": row["value"],
                    })
                
                print(f"      ✓ Completed {algo_name}")
                
                # Print some metrics
                for k in [1, 5, 10]:
                    ndcg_row = algo_results[(algo_results["name"] == "NDCG") & (algo_results["k"] == k)]
                    prec_row = algo_results[(algo_results["name"] == "Precision") & (algo_results["k"] == k)]
                    if not ndcg_row.empty:
                        ndcg_val = ndcg_row["value"].iloc[0]
                        prec_val = prec_row["value"].iloc[0] if not prec_row.empty else 0.0
                        print(f"        NDCG@{k}={ndcg_val:.6f}, Precision@{k}={prec_val:.6f}")
                
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
