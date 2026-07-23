"""
Experiment: Impact of Data Split Random Seeds on Recommender Accuracy
========================================================================
Tests 3 algorithms (ALS/ItemKNN/Pop) × 3 datasets × 5 random seeds
Measures nDCG@k and Precision@k for k=1,5,10
"""

import os
import sys
import itertools
import pandas as pd
import numpy as np

from omnirec import RecSysDataSet, NDCG, HR, Recall
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
from omnirec.util.util import set_random_state

# Working directory
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

print("=" * 80)
print("Experiment: Impact of Data Split Random Seeds on Recommender Accuracy")
print("=" * 80)

# =============================================================================
# 1. Define datasets to load
# =============================================================================
# We'll load raw datasets first (convert to implicit + core prune per seed)
dataset_names = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}

# Algorithms to test
algorithms_config = {
    "ALS": (LensKit.ImplicitMFScorer, {"feedback": "implicit"}),
    "ItemKNN": (LensKit.ItemKNNScorer, {"feedback": "implicit"}),
    "Pop": (LensKit.PopScorer, {"feedback": "implicit"}),
}

# Random seeds
random_seeds = [42, 123, 456, 789, 1111]

# Evaluation metrics
evaluator = Evaluator(
    NDCG([1, 5, 10]),
    Precision([1, 5, 10]),
)

print(f"\nRandom seeds: {random_seeds}")
print(f"Datasets: {list(dataset_names.keys())}")
print(f"Algorithms: {list(algorithms_config.keys())}")
print(f"Metrics: NDCG@[1,5,10], Precision@[1,5,10]")
print(f"\nTotal conditions: {len(dataset_names)} datasets × {len(algorithms_config)} algorithms × {len(random_seeds)} seeds = {len(dataset_names) * len(algorithms_config) * len(random_seeds)}")

# =============================================================================
# 2. Run experiments: for each seed, preprocess and run all algorithms
# =============================================================================
all_results = []  # Collect results across seeds for analysis

for seed_idx, seed in enumerate(random_seeds):
    print(f"\n{'=' * 70}")
    print(f"Seed {seed_idx + 1}/{len(random_seeds)}: seed = {seed}")
    print(f"{'=' * 70}")
    
    # Set random state for reproducibility
    set_random_state(seed)
    
    # For each dataset, preprocess and split with this seed, then run
    for ds_name, ds_enum in dataset_names.items():
        print(f"\n  --- Dataset: {ds_name} ---")
        
        # Load raw dataset
        dataset = RecSysDataSet.use_dataloader(ds_enum)
        print(f"  Loaded: {dataset.num_interactions()} interactions (raw)")
        
        # Build preprocessing pipeline
        # Only apply MakeImplicit for MovieLens100K and Amazon2014VideoGames
        if ds_name in ("MovieLens100K", "Amazon2014VideoGames"):
            pipeline = Pipe(
                MakeImplicit(3),          # Convert ratings >= 3 to implicit
                CorePruning(5),           # 5-core pruning
                UserHoldout(0, 0.2),      # User-based 80/20 holdout (no validation)
            )
        else:
            # HetrecLastFM is already implicit
            pipeline = Pipe(
                CorePruning(5),           # 5-core pruning
                UserHoldout(0, 0.2),      # User-based 80/20 holdout (no validation)
            )
        
        # Apply preprocessing
        dataset = pipeline.process(dataset)
        print(f"  After preprocessing (seed={seed}): {dataset.num_interactions()} interactions")
        
        # Create experiment plan for this (dataset, seed) combination
        # We use a unique plan name per seed to organize checkpoints
        plan = ExperimentPlan(plan_name=f"SeedComparison_{ds_name}_seed{seed}")
        
        # Add all three algorithms with standard hyperparameters
        plan.add_algorithm(
            LensKit.ImplicitMFScorer,
            {
                "feedback": "implicit",
                "random_state": seed,  # Set algorithm's internal seed
            }
        )
        plan.add_algorithm(
            LensKit.ItemKNNScorer,
            {"feedback": "implicit"}
        )
        plan.add_algorithm(
            LensKit.PopScorer,
            {"feedback": "implicit"}
        )
        
        # Run experiments for this dataset/seed
        run_omnirec(
            datasets=dataset,
            plan=plan,
            evaluator=evaluator,
        )

# =============================================================================
# 3. Collect and process results
# =============================================================================
print(f"\n{'=' * 80}")
print("Collecting Results")
print(f"{'=' * 80}")

results_dict = evaluator.get_results()
print(f"Results from {len(results_dict)} dataset-hash combinations")

# Combine all results into a single DataFrame
all_results_dfs = []
for ds_hash_key, df in results_dict.items():
    # Extract dataset name from the hash key (format: "DatasetName-xxxxxx")
    ds_name = ds_hash_key.split("-")[0] if "-" in ds_hash_key else ds_hash_key
    df_copy = df.copy()
    df_copy["dataset"] = ds_name
    all_results_dfs.append(df_copy)

if all_results_dfs:
    combined_results = pd.concat(all_results_dfs, ignore_index=True)
else:
    print("WARNING: No results collected yet. Results may have been checkpointed.")
    print("Attempting to load from checkpoint...")
    # Check if evaluator has saved results
    combined_results = pd.DataFrame()

print(f"\nCombined results shape: {combined_results.shape}")
if len(combined_results) > 0:
    print("\nRaw results preview:")
    print(combined_results.head(20).to_string())

# =============================================================================
# 4. Statistical Analysis: Mean and Std across seeds
# =============================================================================
print(f"\n{'=' * 80}")
print("Statistical Analysis: Mean ± Std Across 5 Seeds")
print(f"{'=' * 80}")

if len(combined_results) > 0:
    # Parse algorithm name (may include hash suffix)
    # The algorithm column has format like "LensKit.ImplicitMFScorer-hash"
    combined_results["algo_short"] = combined_results["algorithm"].apply(
        lambda x: x.split(".")[-1].split("-")[0] if "." in str(x) else str(x)
    )
    
    # Map algorithm names to simpler labels
    algo_map = {
        "ImplicitMFScorer": "ALS",
        "ItemKNNScorer": "ItemKNN",
        "PopScorer": "Pop",
    }
    combined_results["algorithm_label"] = combined_results["algo_short"].map(algo_map).fillna(combined_results["algo_short"])
    
    # Group by dataset, algorithm, metric, k and compute mean/std
    stat_analysis = combined_results.groupby(
        ["dataset", "algorithm_label", "name", "k"]
    )["value"].agg(["mean", "std", "count"])
    
    stat_analysis = stat_analysis.reset_index()
    stat_analysis["mean_std"] = stat_analysis.apply(
        lambda r: f"{r['mean']:.6f} ± {r['std']:.6f}", axis=1
    )
    
    print("\n--- Summary Statistics (Mean ± Std across seeds) ---")
    print("\nSorted by dataset, algorithm, metric, k:\n")
    
    for ds in sorted(stat_analysis["dataset"].unique()):
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'─' * 60}")
        ds_data = stat_analysis[stat_analysis["dataset"] == ds]
        for algo in sorted(ds_data["algorithm_label"].unique()):
            print(f"\n  Algorithm: {algo}")
            algo_data = ds_data[ds_data["algorithm_label"] == algo]
            for _, row in algo_data.iterrows():
                k_val = int(row["k"]) if pd.notna(row["k"]) else "N/A"
                print(f"    {row['name']}@{k_val}: {row['mean']:.6f} ± {row['std']:.6f}  (n={int(row['count'])})")
            print()
    
    # Also print a compact table
    print(f"\n{'=' * 80}")
    print("Compact Result Table")
    print(f"{'=' * 80}")
    
    pivot = stat_analysis.pivot_table(
        index=["dataset", "algorithm_label"],
        columns=["name", "k"],
        values="mean_std",
        aggfunc=lambda x: x.iloc[0] if len(x) > 0 else ""
    )
    print(pivot.to_string())

else:
    print("No results available. The experiments may have checkpointed results.")
    print("Check the checkpoints/ directory for saved results.")

print(f"\n{'=' * 80}")
print("Experiment Complete")
print(f"{'=' * 80}")
print(f"\nCheckpoints saved in: {os.path.join(working_dir, 'checkpoints')}")
