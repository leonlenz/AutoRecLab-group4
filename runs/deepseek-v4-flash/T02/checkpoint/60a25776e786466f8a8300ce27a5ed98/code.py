"""
Experiment: Impact of Data Split Random Seeds on Recommender Accuracy
========================================================================
Tests 3 algorithms (ALS/ItemKNN/Pop) x 3 datasets x 5 random seeds
Measures NDCG@k and Precision@k for k=1, 5, 10
Reports mean +/- std across seeds for statistical analysis
"""

import os
import sys
import json
import itertools
import pandas as pd
import numpy as np

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.subsample import Subsample
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


# =============================================================================
# Working directory
# =============================================================================
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

print("=" * 80)
print("Experiment: Impact of Data Split Random Seeds on Recommender Accuracy")
print("=" * 80)

# =============================================================================
# 1. Define datasets and algorithms
# =============================================================================
DATASET_ENUMS = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}

# Random seeds for data splitting
RANDOM_SEEDS = [42, 123, 456, 789, 1111]

# Metrics: NDCG and Precision at k=1,5,10
METRICS = [1, 5, 10]

print(f"\nRandom seeds: {RANDOM_SEEDS}")
print(f"Datasets: {list(DATASET_ENUMS.keys())}")
print(f"Algorithms: ALS (ImplicitMFScorer), ItemKNN, Pop")
print(f"Metrics: NDCG@{METRICS}, Precision@{METRICS}")
print(f"\nTotal conditions: {len(DATASET_ENUMS)} datasets x 3 algorithms x {len(RANDOM_SEEDS)} seeds = {len(DATASET_ENUMS) * 3 * len(RANDOM_SEEDS)}")

# =============================================================================
# 2. Run experiments for each seed
# =============================================================================
# Collect results per seed
all_seed_results = []

for seed_idx, seed in enumerate(RANDOM_SEEDS):
    print(f"\n{'=' * 70}")
    print(f"Seed {seed_idx + 1}/{len(RANDOM_SEEDS)}: seed = {seed}")
    print(f"{'=' * 70}")

    # Set global random state for reproducibility of splits
    set_random_state(seed)

    # Build datasets for this seed
    seed_datasets = []
    for ds_name, ds_enum in DATASET_ENUMS.items():
        print(f"\n  --- Dataset: {ds_name} ---")

        # Load raw dataset
        dataset = RecSysDataSet.use_dataloader(ds_enum)
        n_raw_raw = dataset.num_interactions()
        print(f"  Loaded: {n_raw_raw} interactions (raw)")

        # Build preprocessing pipeline
        steps = []

        if ds_name in ("MovieLens100K", "Amazon2014VideoGames"):
            # Convert ratings > 3 to implicit feedback (ratings >= 4 -> 1, else 0)
            # Using MakeImplicit(3) means ratings > 3 become 1
            steps.append(MakeImplicit(3))

        # 5-core filtering (removes users/items with < 5 interactions)
        steps.append(CorePruning(5))

        # User holdout: ~70% train, ~10% val, 20% test
        # NOTE: Use validation_size=0.1 (not 0) because UserHoldout._process()
        # computes test_size = valid_size / (1 - test_size) for the second
        # train_test_split call. A value of 0 would produce 0.0 which sklearn rejects.
        # With valid_size=0.1 and test_size=0.2:
        #   - First split: 80% train, 20% test
        #   - Second split from train: valid ratio = 0.1/0.8 = 0.125 -> 12.5% of train = 10% of total
        # Result: ~70% train, ~10% validation, ~20% test
        steps.append(UserHoldout(0.1, 0.2))

        pipeline = Pipe(*steps)
        dataset = pipeline.process(dataset)
        n_counts = dataset.num_interactions()
        print(f"  After preprocessing (seed={seed}): {n_counts}")

        seed_datasets.append(dataset)

    # Create experiment plan for this seed with all three algorithms
    # Each seed gets its own plan and evaluator so results are independent
    plan = ExperimentPlan(plan_name=f"SeedComparison_seed{seed}")

    # ALS (ImplicitMFScorer) for implicit feedback
    plan.add_algorithm(
        LensKit.ImplicitMFScorer,
        {"feedback": "implicit"},
    )
    # Item-based KNN for implicit feedback
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {"feedback": "implicit"},
    )
    # Popularity scorer for implicit feedback
    plan.add_algorithm(
        LensKit.PopScorer,
        {"feedback": "implicit"},
    )

    # Evaluator with NDCG and Precision at k=1,5,10
    evaluator = Evaluator(
        NDCG(METRICS),
        Precision(METRICS),
    )

    # Run all algorithms on all datasets for this seed
    run_omnirec(
        datasets=seed_datasets,
        plan=plan,
        evaluator=evaluator,
    )

    # Collect results for this seed
    seed_results = evaluator.get_results()
    for ds_hash_key, df in seed_results.items():
        df_copy = df.copy()
        # Extract dataset name from the hash key (format: "DatasetName-xxxxxx")
        ds_name = ds_hash_key.split("-")[0] if "-" in ds_hash_key else ds_hash_key
        df_copy["dataset"] = ds_name
        df_copy["seed"] = seed
        all_seed_results.append(df_copy)

# =============================================================================
# 3. Combine all results
# =============================================================================
print(f"\n{'=' * 80}")
print("Collecting Results Across All Seeds")
print(f"{'=' * 80}")

if all_seed_results:
    combined_results = pd.concat(all_seed_results, ignore_index=True)
else:
    combined_results = pd.DataFrame()

print(f"\nCombined results shape: {combined_results.shape}")
if len(combined_results) > 0:
    print("\nRaw results preview (first 20 rows):")
    print(combined_results.head(20).to_string())

# =============================================================================
# 4. Statistical Analysis: Mean and Std across seeds
# =============================================================================
print(f"\n{'=' * 80}")
print("Statistical Analysis: Mean +/- Std Across 5 Seeds")
print(f"{'=' * 80}")

if len(combined_results) > 0:
    # Parse algorithm name to short form
    combined_results["algo_short"] = combined_results["algorithm"].apply(
        lambda x: str(x).split(".")[-1].split("-")[0] if "." in str(x) else str(x)
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
        lambda r: f"{r['mean']:.6f} +/- {r['std']:.6f}", axis=1
    )

    print("\n--- Summary Statistics (Mean +/- Std across seeds) ---")
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
                print(f"    {row['name']}@{k_val}: {row['mean']:.6f} +/- {row['std']:.6f}  (n={int(row['count'])})")
            print()

    # Print a compact pivot table
    print(f"\n{'=' * 80}")
    print("Compact Result Table (mean +/- std)")
    print(f"{'=' * 80}")

    # Create a display column
    stat_analysis["display"] = stat_analysis.apply(
        lambda r: f"{r['mean']:.4f}±{r['std']:.4f}", axis=1
    )

    pivot = stat_analysis.pivot_table(
        index=["dataset", "algorithm_label"],
        columns=["name", "k"],
        values="display",
        aggfunc=lambda x: x.iloc[0] if len(x) > 0 else ""
    )
    print(pivot.to_string())

else:
    print("No results available. The experiments may have checkpointed results.")
    print("Check the checkpoints/ directory for saved results.")

print(f"\n{'=' * 80}")
print("Experiment Complete")
print(f"{'=' * 80}")
print(f"\nWorking directory: {working_dir}")
