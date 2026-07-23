import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# =============================================================================
# OmniRec imports (for data loading and preprocessing)
# =============================================================================
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.data_variants import SplitData
from omnirec.util.util import set_random_state

# OmniRec evaluation metrics
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.metrics.base import MetricResult

# =============================================================================
# LensKit native imports (for training and prediction)
# =============================================================================
from lenskit.basic.popularity import PopScorer
from lenskit.knn.item import ItemKNNScorer
from lenskit.als import ImplicitMFScorer
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend
from lenskit.data import from_interactions_df
from lenskit.splitting import SampleFrac, crossfold_users

# =============================================================================
# Setup working directory
# =============================================================================
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)


# =============================================================================
# Configuration
# =============================================================================
SEEDS = [42, 123, 456, 789, 1111]

# Subsample Amazon to avoid timeout - use 10% of interactions
AMAZON_SUBSAMPLE_FRAC = 0.1

# Recommendation list length for top-n (we evaluate at k=1,5,10 but generate up to 10)
RECOMMEND_LIST_LENGTH = 10

# Dataset names
DATASET_NAMES = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]


def load_and_preprocess_dataset(ds_name: str) -> SplitData:
    """Load a dataset via OmniRec, apply preprocessing, and return split data."""
    
    # Map name to OmniRec DataSet enum
    ds_map = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }
    
    dataset = RecSysDataSet.use_dataloader(ds_map[ds_name])
    
    # Build preprocessing pipeline
    steps = []
    
    # For MovieLens and Amazon: convert ratings > 3 to implicit (MakeImplicit(4) keeps >= 4)
    if ds_name in ("MovieLens100K", "Amazon2014VideoGames"):
        steps.append(MakeImplicit(4))
    
    # 5-core filtering
    steps.append(CorePruning(5))
    
    # For Amazon, also subsample to avoid timeout
    if ds_name == "Amazon2014VideoGames":
        # We'll subsample after loading via pandas - CorePruning already filtered
        pass  # CorePruning will reduce size somewhat
    
    # UserHoldout split: use proper float values for 80/20 (no validation set)
    # validation_size=0 means no validation split, test_size=0.2 means 20% test
    steps.append(UserHoldout(validation_size=0.0, test_size=0.2))
    
    pipe = Pipe(*steps)
    processed = pipe.process(dataset)
    
    return processed._data  # SplitData object


def train_and_recommend_lenskit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    algo_name: str,
    algo_config: dict,
) -> pd.DataFrame:
    """
    Train a LensKit model and generate recommendations.
    
    Returns a DataFrame with columns [user, item, score, rank].
    """
    # Create LensKit Dataset from training DataFrame
    # The DataFrame from OmniRec has columns: user, item, rating, timestamp
    train_ds = from_interactions_df(train_df)
    
    # Create scorer
    if algo_name == "PopScorer":
        scorer = PopScorer(score="count")
    elif algo_name == "ItemKNNScorer":
        scorer = ItemKNNScorer(
            max_nbrs=algo_config.get("max_nbrs", 20),
            min_nbrs=algo_config.get("min_nbrs", 1),
            feedback=algo_config.get("feedback", "implicit"),
        )
    elif algo_name == "ImplicitMFScorer":
        scorer = ImplicitMFScorer(
            embedding_size=algo_config.get("embedding_size", 20),
            epochs=algo_config.get("epochs", 10),
            weight=algo_config.get("weight", 40),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Build top-n pipeline
    pipe = topn_pipeline(scorer, n=RECOMMEND_LIST_LENGTH)
    
    # Train
    pipe.train(train_ds)
    
    # Prepare test queries: get unique users that are in training set
    train_users = set(train_df["user"].unique())
    test_users_df = test_df[test_df["user"].isin(train_users)]
    test_user_ids = test_users_df["user"].unique()
    
    # Generate recommendations using LensKit batch
    recs_ilc = recommend(pipe, test_user_ids, n_jobs=1)
    
    # Convert to DataFrame
    recs_df = recs_ilc.to_df()
    
    # Rename columns to match OmniRec metric expectations: [user, item, score, rank]
    # The recommend output has: [user_id, item_id, rank, score]
    recs_df = recs_df.rename(columns={
        "user_id": "user",
        "item_id": "item",
    })
    
    # Ensure we have the right columns in the right order
    return recs_df[["user", "item", "score", "rank"]]


def calculate_metrics(
    predictions_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_values: list[int],
) -> dict:
    """
    Calculate NDCG@k and Precision@k using OmniRec's metric implementations.
    
    Args:
        predictions_df: DataFrame with [user, item, score, rank]
        test_df: DataFrame with [user, item] (ground truth)
        k_values: list of k values to evaluate
    
    Returns:
        dict: {(metric_name, k): value}
    """
    results = {}
    
    # Ensure predictions have the expected columns for OmniRec metrics
    # The metrics expect: predictions has [user, item, score, rank], test has [user, item]
    
    # NDCG
    ndcg_metric = NDCG(k_values)
    ndcg_result: MetricResult = ndcg_metric.calculate(predictions_df, test_df)
    assert isinstance(ndcg_result.result, dict)  # type narrowing: k_values is a list, so result is dict
    for k, val in ndcg_result.result.items():
        results[("NDCG", k)] = val
    
    # Precision
    prec_metric = Precision(k_values)
    prec_result: MetricResult = prec_metric.calculate(predictions_df, test_df)
    assert isinstance(prec_result.result, dict)  # type narrowing: k_values is a list, so result is dict
    for k, val in prec_result.result.items():
        results[("Precision", k)] = val
    
    return results


# =============================================================================
# Main experiment loop
# =============================================================================
all_results = []  # list of dicts

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*80}")
    print(f"RUNNING SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'='*80}")
    
    # Set global random state for reproducibility
    set_random_state(seed)
    
    for ds_name in DATASET_NAMES:
        print(f"\n--- Loading & preprocessing {ds_name} (seed={seed}) ---")
        
        # Step 1: Load and preprocess via OmniRec
        split_data: SplitData = load_and_preprocess_dataset(ds_name)
        
        train_df = split_data.train
        val_df = split_data.val
        test_df = split_data.test
        
        n_train = len(train_df)
        n_val = len(val_df)
        n_test = len(test_df)
        n_users = train_df["user"].nunique()
        n_items = train_df["item"].nunique()
        print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
        print(f"  Users: {n_users}, Items: {n_items}")
        
        # For Amazon, additionally subsample to avoid timeout
        if ds_name == "Amazon2014VideoGames" and len(train_df) > 50000:
            # Subsample training data to keep runtime manageable
            train_df = train_df.sample(frac=AMAZON_SUBSAMPLE_FRAC, random_state=seed)
            # Also filter test to only contain users from subsampled train
            train_users = set(train_df["user"].unique())
            test_df = test_df[test_df["user"].isin(train_users)]
            print(f"  (Subsampled to {len(train_df)} train, {len(test_df)} test)")
        
        # =====================================================================
        # Train and evaluate each algorithm
        # =====================================================================
        algorithms = [
            ("PopScorer", {}),
            ("ItemKNNScorer", {"max_nbrs": 20, "min_nbrs": 1, "feedback": "implicit"}),
            ("ImplicitMFScorer", {"embedding_size": 20, "epochs": 10, "weight": 40}),
        ]
        
        for algo_name, algo_config in algorithms:
            print(f"    Running {algo_name}...")
            
            try:
                # Train model and generate recommendations
                predictions_df = train_and_recommend_lenskit(
                    train_df, test_df, algo_name, algo_config
                )
                
                if len(predictions_df) == 0:
                    print(f"    WARNING: No predictions generated for {algo_name}")
                    continue
                
                # Calculate metrics using OmniRec metric implementations
                k_values = [1, 5, 10]
                metric_results = calculate_metrics(predictions_df, test_df, k_values)
                
                # Store results
                for (metric_name, k), value in metric_results.items():
                    all_results.append({
                        "seed": seed,
                        "dataset": ds_name,
                        "algorithm": algo_name,
                        "metric": metric_name,
                        "k": k,
                        "value": value,
                    })
                    print(f"      {metric_name}@{k} = {value:.6f}")
                
            except Exception as e:
                print(f"    ERROR with {algo_name} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()

# =============================================================================
# Analysis: Compute mean and std across seeds
# =============================================================================
print(f"\n{'='*80}")
print(f"FINAL RESULTS - Aggregated across {len(SEEDS)} seeds")
print(f"{'='*80}")

if len(all_results) > 0:
    results_df = pd.DataFrame(all_results)
    
    # Group by dataset, algorithm, metric, k and compute mean + std
    summary = (
        results_df
        .groupby(["dataset", "algorithm", "metric", "k"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    
    # Print summary per dataset
    for ds_name in DATASET_NAMES:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")
        
        ds_summary = summary[summary["dataset"] == ds_name]
        if len(ds_summary) == 0:
            print("  (No results)")
            continue
        
        for algo_name in sorted(ds_summary["algorithm"].unique()):
            print(f"\n  Algorithm: {algo_name}")
            print(f"  {'Metric':<15} {'k':<5} {'Mean':<12} {'Std':<12} {'CV(%)':<10}")
            print(f"  {'-'*55}")
            
            algo_data = ds_summary[ds_summary["algorithm"] == algo_name]
            for _, row in algo_data.sort_values(["metric", "k"]).iterrows():
                mean_val = row["mean"]
                std_val = row["std"]
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0.0
                print(f"  {row['metric']:<15} {row['k']:<5} {mean_val:<12.6f} {std_val:<12.6f} {cv:<10.2f}")
    
    # =============================================================================
    # Statistical analysis of seed impact
    # =============================================================================
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS: Impact of Data Split Seed Variation")
    print(f"{'='*80}")
    
    # Compute coefficient of variation (CV)
    pivot_cv = summary.copy()
    pivot_cv["cv"] = np.where(
        pivot_cv["mean"] > 0,
        pivot_cv["std"] / pivot_cv["mean"] * 100,
        0.0
    )
    
    # Average CV per (dataset, algorithm) combination
    if len(pivot_cv) > 0:
        avg_cv = (
            pivot_cv
            .groupby(["dataset", "algorithm"])["cv"]
            .mean()
            .reset_index()
            .sort_values("cv", ascending=False)
        )
        
        print(f"\nAverage Coefficient of Variation (%) across all metrics per (Dataset, Algorithm):")
        print(f"{'Dataset':<25} {'Algorithm':<35} {'Avg CV(%)':<10}")
        print(f"{'-'*70}")
        for _, row in avg_cv.iterrows():
            print(f"{row['dataset']:<25} {row['algorithm']:<35} {row['cv']:<10.2f}")
        
        # Max CV across all conditions
        if len(pivot_cv) > 0:
            max_cv_row = pivot_cv.loc[pivot_cv["cv"].idxmax()]
            print(f"\nHighest seed sensitivity observed:")
            print(f"  Dataset: {max_cv_row['dataset']}")
            print(f"  Algorithm: {max_cv_row['algorithm']}")
            print(f"  Metric: {max_cv_row['metric']} @ k={max_cv_row['k']}")
            print(f"  Mean: {max_cv_row['mean']:.6f}, Std: {max_cv_row['std']:.6f}, CV: {max_cv_row['cv']:.2f}%")
        
        # Per-algorithm summary across all datasets
        print(f"\n{'='*70}")
        print(f"Summary by Algorithm (averaged across all datasets and metrics):")
        print(f"{'='*70}")
        algo_summary = (
            pivot_cv
            .groupby("algorithm")[["mean", "std", "cv"]]
            .agg({"mean": "mean", "std": "mean", "cv": "mean"})
            .reset_index()
            .sort_values("cv", ascending=False)
        )
        print(f"{'Algorithm':<35} {'Avg Mean':<12} {'Avg Std':<12} {'Avg CV(%)':<10}")
        print(f"{'-'*70}")
        for _, row in algo_summary.iterrows():
            print(f"{row['algorithm']:<35} {row['mean']:<12.6f} {row['std']:<12.6f} {row['cv']:<10.2f}")
else:
    print("No results were generated. Something went wrong.")
    
print(f"\n{'='*80}")
print(f"EXPERIMENT COMPLETE")
print(f"{'='*80}")
print(f"\nSeeds used: {SEEDS}")
print(f"Total experiment configurations: {len(SEEDS)} seeds x {len(DATASET_NAMES)} datasets x 3 algorithms")
if len(all_results) > 0:
    print(f"Results stored with {len(all_results)} individual metric rows.")