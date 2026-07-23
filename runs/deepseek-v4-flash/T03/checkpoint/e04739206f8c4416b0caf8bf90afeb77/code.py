#!/usr/bin/env python3
"""
Quantifying the impact of data split random seeds on recommender system accuracy.
Uses OmniRec for data loading/preprocessing and LensKit for model training/evaluation.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# OmniRec imports for data loading and preprocessing
# ============================================================
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.util.util import set_random_state

# ============================================================
# LensKit imports for splitting, training, recommendation, evaluation
# ============================================================
from lenskit.data import from_interactions_df
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
N_RECS = 10  # recommend top-10 for evaluation at k=10

# Working directory
WORKING_DIR = Path(os.getcwd()) / "working"
WORKING_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING_DIR)


def load_and_preprocess_movielens():
    """Load MovieLens100K, convert to implicit (ratings > 3), apply 5-core."""
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipeline = Pipe(
        MakeImplicit(3.5),  # ratings > 3 => implicit (threshold 3.5 means >= 4)
        CorePruning(5),
    )
    dataset = pipeline.process(dataset)
    df = dataset._data.df.copy()
    print(f"[MovieLens100K] Loaded {len(df)} interactions, "
          f"{df['user'].nunique()} users, {df['item'].nunique()} items")
    return df


def load_and_preprocess_amazon():
    """Load Amazon2014VideoGames, convert to implicit (ratings > 3), apply 5-core."""
    dataset = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
    pipeline = Pipe(
        MakeImplicit(3.5),  # ratings > 3 => implicit
        CorePruning(5),
    )
    dataset = pipeline.process(dataset)
    df = dataset._data.df.copy()
    print(f"[Amazon2014VideoGames] Loaded {len(df)} interactions, "
          f"{df['user'].nunique()} users, {df['item'].nunique()} items")
    return df


def load_and_preprocess_lastfm():
    """Load HetrecLastFM (already implicit), apply 5-core only."""
    dataset = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
    pipeline = Pipe(
        CorePruning(5),  # No MakeImplicit — already implicit
    )
    dataset = pipeline.process(dataset)
    df = dataset._data.df.copy()
    print(f"[HetrecLastFM] Loaded {len(df)} interactions, "
          f"{df['user'].nunique()} users, {df['item'].nunique()} items")
    return df


def prepare_lenskit_dataset(df):
    """Convert OmniRec DataFrame (columns: user, item, rating, timestamp) to LensKit Dataset."""
    # Ensure we have the rating column for implicit (will be 1)
    if "rating" not in df.columns:
        df["rating"] = 1
    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        df["timestamp"] = 0
    # Build LensKit Dataset
    ds = from_interactions_df(df)
    return ds


def run_single_experiment(train_ds, test_ilc, algo_name, algo_params):
    """Train a model and evaluate on test set.

    Args:
        train_ds: LensKit Dataset for training.
        test_ilc: ItemListCollection for test.
        algo_name: 'PopScorer', 'ItemKNNScorer', or 'ImplicitMFScorer'.
        algo_params: dict of parameters.

    Returns:
        dict of metric results.
    """
    # Build the scorer
    if algo_name == "PopScorer":
        scorer = PopScorer(**algo_params)
    elif algo_name == "ItemKNNScorer":
        scorer = ItemKNNScorer(**algo_params)
    elif algo_name == "ImplicitMFScorer":
        scorer = ImplicitMFScorer(**algo_params)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Create top-N pipeline
    pipe = topn_pipeline(scorer, n=N_RECS)

    # Train
    pipe.train(train_ds)

    # Generate recommendations for test users
    recs = recommend(pipe, test_ilc.keys(), n=N_RECS)

    # Evaluate with MeasurementCollector using add_collection_measurements + summary_metrics
    collector = MeasurementCollector()
    for k in K_VALUES:
        collector.add_metric(NDCG(n=k))
        collector.add_metric(Precision(n=k))

    collector.add_collection_measurements(recs, test_ilc)
    summary = collector.summary_metrics()
    return summary


def main():
    print("=" * 70)
    print("Experiment: Impact of Data Split Random Seeds on Recommendation Accuracy")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Load and preprocess all datasets
    # ----------------------------------------------------------
    print("\n[Step 1] Loading and preprocessing datasets...\n")
    datasets_raw = {
        "MovieLens100K": load_and_preprocess_movielens(),
        "Amazon2014VideoGames": load_and_preprocess_amazon(),
        "HetrecLastFM": load_and_preprocess_lastfm(),
    }

    # Convert to LensKit Dataset objects
    datasets = {}
    for name, df in datasets_raw.items():
        datasets[name] = prepare_lenskit_dataset(df)
        print(f"  LensKit Dataset '{name}': "
              f"{datasets[name].interaction_count} interactions, "
              f"{datasets[name].user_count} users, "
              f"{datasets[name].item_count} items")

    # ----------------------------------------------------------
    # 2. Define algorithm configurations (default hyperparameters)
    # ----------------------------------------------------------
    algorithms = {
        "PopScorer": {"algo_name": "PopScorer", "params": {}},
        "ItemKNNScorer": {"algo_name": "ItemKNNScorer", "params": {"feedback": "implicit"}},
        "ImplicitMFScorer": {"algo_name": "ImplicitMFScorer", "params": {}},
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
            # We use crossfold_users with n_folds=1 and SampleFrac(0.2)
            # This gives us a single split where ~20% of each user's items go to test
            splits = list(crossfold_users(
                ds, 1, SampleFrac(TEST_FRAC, rng=seed), rng=seed
            ))
            split = splits[0]
            train_ds = split.train
            test_ilc = split.test

            print(f"    Train: {train_ds.interaction_count} interactions, "
                  f"Test: {sum(len(il) for il in test_ilc.lists())} interactions")

            for algo_name, algo_info in algorithms.items():
                print(f"    Algorithm: {algo_name}...", end=" ", flush=True)

                try:
                    metrics = run_single_experiment(
                        train_ds, test_ilc,
                        algo_info["algo_name"],
                        algo_info["params"]
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
                        # metrics is a dict, supports .get()
                        result[f"NDCG@{k}"] = metrics.get(f"NDCG@{k}.mean", np.nan)
                        result[f"Precision@{k}"] = metrics.get(f"Precision@{k}.mean", np.nan)

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

    # Group by dataset, algorithm, and compute mean/std across seeds
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
    # the range and IQR across seeds
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
