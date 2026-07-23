#!/usr/bin/env python3
"""
Quantifying the impact of data split random seeds on recommender system accuracy.
Uses OmniRec (which wraps LensKit) for all experiment functionality.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# OmniRec imports for all experiment operations
# ============================================================
from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state

warnings.filterwarnings("ignore")

# Experiment configuration
SEEDS = [42, 123, 256, 789, 1024]
K_VALUES = [1, 5, 10]
TEST_FRAC = 0.2  # 80/20 split (no validation set needed)
N_RECS = 10  # recommend top-10 for evaluation

# Working directory
WORKING_DIR = Path(os.getcwd()) / "working"
WORKING_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING_DIR)


# ---------------------------------------------------------------------------
# Dataset loading and preprocessing using OmniRec
# ---------------------------------------------------------------------------

def load_and_preprocess_dataset(dataset_enum, name, make_implicit_threshold=None):
    """Load a dataset with OmniRec and preprocess (5-core + optional implicit conversion).
    
    Args:
        dataset_enum: DataSet enum member for the dataset.
        name: Human-readable dataset name for logging.
        make_implicit_threshold: If set, convert ratings > threshold to implicit.
    
    Returns:
        RecSysDataSet[RawData] — preprocessed raw dataset ready for splitting.
    """
    print(f"\n[{name}]")
    
    # Load using OmniRec's built-in data loader
    # OmniRec automatically: downloads, removes duplicates, normalizes IDs
    dataset = RecSysDataSet.use_dataloader(dataset_enum)
    
    # Print raw stats
    raw_df = dataset._data.df
    print(f"  Loaded {len(raw_df)} interactions, "
          f"{raw_df['user'].nunique()} users, {raw_df['item'].nunique()} items")
    
    # Build preprocessing pipeline
    steps = []
    
    # Convert to implicit if needed (for MovieLens and Amazon which have explicit ratings)
    if make_implicit_threshold is not None:
        steps.append(MakeImplicit(make_implicit_threshold))
    
    # Apply 5-core filtering
    steps.append(CorePruning(5))
    
    # Apply pipeline (but stop before splitting — we'll split per seed later)
    if steps:
        pipe = Pipe(*steps)
        dataset = pipe.process(dataset)
    
    # Print final stats
    final_df = dataset._data.df
    print(f"  After preprocessing: {len(final_df)} interactions, "
          f"{final_df['user'].nunique()} users, {final_df['item'].nunique()} items")
    
    return dataset


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment: Impact of Data Split Random Seeds on Recommendation Accuracy")
    print("=" * 70)
    
    # ----------------------------------------------------------
    # 1. Load and preprocess all datasets (raw, without splitting)
    # ----------------------------------------------------------
    print("\n[Step 1] Loading and preprocessing datasets...")
    
    # MovieLens100K — explicit ratings, convert to implicit (ratings > 3)
    ml100k_raw = load_and_preprocess_dataset(
        DataSet.MovieLens100K, "MovieLens100K", make_implicit_threshold=3
    )
    
    # Amazon2014VideoGames — explicit ratings, convert to implicit (ratings > 3)
    amazon_raw = load_and_preprocess_dataset(
        DataSet.Amazon2014VideoGames, "Amazon2014VideoGames", make_implicit_threshold=3
    )
    
    # HetrecLastFM — already implicit (rating=1), no conversion needed
    lastfm_raw = load_and_preprocess_dataset(
        DataSet.HetrecLastFM, "HetrecLastFM", make_implicit_threshold=None
    )
    
    datasets = {
        "MovieLens100K": ml100k_raw,
        "Amazon2014VideoGames": amazon_raw,
        "HetrecLastFM": lastfm_raw,
    }
    
    # ----------------------------------------------------------
    # 2. Define algorithm configurations (standard hyperparameters)
    # ----------------------------------------------------------
    # Use default hyperparameters for all three LensKit algorithms
    algorithm_configs = [
        ("PopScorer", LensKit.PopScorer, {}),
        ("ItemKNNScorer", LensKit.ItemKNNScorer, {"max_nbrs": 20, "min_nbrs": 5}),
        ("ImplicitMFScorer", LensKit.ImplicitMFScorer, {"features": 50, "iterations": 20}),
    ]
    
    # ----------------------------------------------------------
    # 3. Run experiment: 3 algos x 3 datasets x 5 seeds = 45 runs
    # ----------------------------------------------------------
    print("\n[Step 2] Running experiments...")
    all_results = []
    
    for dataset_name, raw_dataset in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 50}")
        
        for seed in SEEDS:
            print(f"\n  Seed: {seed}")
            
            # Set the global random state so UserHoldout uses this seed
            set_random_state(seed)
            
            # Split the raw dataset per seed using UserHoldout
            # UserHoldout ensures each user appears in all splits
            # We want 80/20 train/test, no validation, so test_size=0.2, validation_size=0
            split_pipe = Pipe(UserHoldout(validation_size=0.0, test_size=TEST_FRAC))
            split_dataset = split_pipe.process(raw_dataset)
            
            # Get train/test sizes
            train_df = split_dataset._data.get("train")
            test_df = split_dataset._data.get("test")
            print(f"    Train: {len(train_df)} interactions, "
                  f"Test: {len(test_df)} interactions")
            
            # Create experiment plan with all 3 algorithms
            plan = ExperimentPlan(
                plan_name=f"{dataset_name}_seed{seed}"
            )
            for algo_name, algo_enum, algo_params in algorithm_configs:
                plan.add_algorithm(algo_enum, algo_params)
            
            # Set up evaluator with NDCG and Precision at k=1,5,10
            evaluator = Evaluator(
                NDCG(K_VALUES),
                Precision(K_VALUES),
            )
            
            try:
                # Run all algorithms on this split
                run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
                
                # Collect results from the evaluator
                for dataset_id, results_df in evaluator.get_results().items():
                    for _, row in results_df.iterrows():
                        result = {
                            "dataset": dataset_name,
                            "algorithm": row["algorithm"].split("-")[0],  # Remove config hash
                            "seed": seed,
                            "metric": f"{row['name']}@{row['k']}" if row['k'] is not None else row['name'],
                            "value": row["value"],
                        }
                        all_results.append(result)
                
                print("    All algorithms done")
            except Exception as e:
                print(f"    FAILED for seed {seed}: {e}")
                # Fill with NaN for this seed
                for algo_name, _, _ in algorithm_configs:
                    for k in K_VALUES:
                        all_results.append({
                            "dataset": dataset_name,
                            "algorithm": algo_name,
                            "seed": seed,
                            "metric": f"NDCG@{k}",
                            "value": np.nan,
                        })
                        all_results.append({
                            "dataset": dataset_name,
                            "algorithm": algo_name,
                            "seed": seed,
                            "metric": f"Precision@{k}",
                            "value": np.nan,
                        })
    
    # ----------------------------------------------------------
    # 4. Aggregate results & statistical analysis
    # ----------------------------------------------------------
    print("\n\n[Step 3] Aggregating results...\n")
    df_results = pd.DataFrame(all_results)
    
    # Pivot to wide format: one row per (dataset, algorithm, seed)
    df_pivot = df_results.pivot_table(
        index=["dataset", "algorithm", "seed"],
        columns="metric",
        values="value"
    ).reset_index()
    
    print("Raw results shape:", df_pivot.shape)
    print(df_pivot.head(10).to_string())
    print()
    
    metric_cols = [f"NDCG@{k}" for k in K_VALUES] + [f"Precision@{k}" for k in K_VALUES]
    grouping_cols = ["dataset", "algorithm"]
    
    # Ensure all metric columns exist
    for col in metric_cols:
        if col not in df_pivot.columns:
            df_pivot[col] = np.nan
    
    agg_results = df_pivot.groupby(grouping_cols)[metric_cols].agg(
        ["mean", "std"]
    ).round(5)
    
    print("=" * 70)
    print("AGGREGATED RESULTS (mean ± std across 5 seeds)")
    print("=" * 70)
    print(agg_results.to_string())
    print()
    
    # Variance analysis
    print("=" * 70)
    print("VARIANCE ANALYSIS (std across seeds)")
    print("=" * 70)
    var_results = df_pivot.groupby(grouping_cols)[metric_cols].std().round(5)
    print(var_results.to_string())
    print()
    
    # Statistical analysis: coefficient of variation (CV = std/mean) for each metric
    mean_results = df_pivot.groupby(grouping_cols)[metric_cols].mean()
    std_results = df_pivot.groupby(grouping_cols)[metric_cols].std()
    
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
    
    # Seed impact summary (range across seeds)
    print("\n\n" + "=" * 70)
    print("SEED IMPACT SUMMARY (range across 5 seeds)")
    print("=" * 70)
    for col in metric_cols:
        range_vals = df_pivot.groupby(["dataset", "algorithm"])[col].agg(
            lambda x: x.max() - x.min()
        ).round(5)
        print(f"\n{col} range (max - min) across seeds:")
        print(range_vals.to_string())
    
    # Conclusion
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
    df_pivot.to_csv(results_path, index=False)
    print(f"\nAll results saved to: {results_path}")
    
    agg_results_path = WORKING_DIR / "aggregated_results.csv"
    agg_results.to_csv(agg_results_path)
    print(f"Aggregated results saved to: {agg_results_path}")


if __name__ == "__main__":
    main()
