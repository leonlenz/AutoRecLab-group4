import os
import sys
import warnings
warnings.filterwarnings('ignore')

from typing import TypedDict

import numpy as np
import pandas as pd

# =============================================================================
# OmniRec imports
# =============================================================================
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

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

class DatasetConfig(TypedDict):
    loader: DataSet
    apply_implicit: bool

# Datasets to process
DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "MovieLens100K": {
        "loader": DataSet.MovieLens100K,
        "apply_implicit": True,
    },
    "Amazon2014VideoGames": {
        "loader": DataSet.Amazon2014VideoGames,
        "apply_implicit": True,
    },
    "HetrecLastFM": {
        "loader": DataSet.HetrecLastFM,
        "apply_implicit": False,
    },
}

# =============================================================================
# Main experiment loop
# =============================================================================
all_results = []  # Will hold dicts with seed, dataset, algorithm, metric, k, value

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*80}")
    print(f"RUNNING SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'='*80}")

    # Set the global random state for reproducibility
    set_random_state(seed)

    processed_datasets = []

    for ds_name, ds_config in DATASET_CONFIGS.items():
        print(f"\n--- Loading & preprocessing {ds_name} ---")

        # Step 1: Load dataset
        dataset = RecSysDataSet.use_dataloader(ds_config["loader"])

        # Step 2: Build preprocessing pipeline
        steps = []

        # Add MakeImplicit if needed
        # Requirement: ratings > 3 for MovieLens100K and Amazon (explicit datasets)
        # MakeImplicit(4) keeps ratings >= 4, i.e. strictly greater than 3
        if ds_config["apply_implicit"]:
            steps.append(MakeImplicit(4))

        # Apply 5-core filtering after implicit conversion
        steps.append(CorePruning(5))

        # User-based holdout split
        # FIX: Use a tiny positive validation_size instead of 0 (which crashes)
        # validation_size=0 leads to test_size=0.0 in sklearn's train_test_split
        # A value of 0.001 gives approximately 79.9% train / 0.1% valid / 20% test
        steps.append(UserHoldout(validation_size=0.001, test_size=0.2))

        # Create and run pipeline
        pipe = Pipe(*steps)
        dataset = pipe.process(dataset)

        # Print dataset stats using public SplitData API
        split_data = dataset._data
        n_train = len(split_data.train)
        n_val = len(split_data.val)
        n_test = len(split_data.test)
        n_users = split_data.train['user'].nunique()
        n_items = split_data.train['item'].nunique()
        print(f"  Train interactions: {n_train}, Val interactions: {n_val}, Test interactions: {n_test}")
        print(f"  Users: {n_users}, Items: {n_items}")

        processed_datasets.append(dataset)

    # =============================================================================
    # Build experiment plan with three LensKit algorithms
    # =============================================================================
    print(f"\n--- Setting up experiment plan for seed {seed} ---")

    plan = ExperimentPlan(plan_name=f"SeedExperiment_{seed}")

    # Algorithm 1: PopScorer (default params)
    plan.add_algorithm(
        LensKit.PopScorer,
        {
            "feedback": "implicit"
        }
    )

    # Algorithm 2: ItemKNNScorer (max_nbrs=20, implicit feedback)
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            "max_nbrs": 20,
            "min_nbrs": 1,
            "feedback": "implicit"
        }
    )

    # Algorithm 3: ImplicitMFScorer (ALS, features=20, epochs=10)
    plan.add_algorithm(
        LensKit.ImplicitMFScorer,
        {
            "features": 20,
            "epochs": 10,
            "weight": 40,
            "feedback": "implicit"
        }
    )

    # =============================================================================
    # Evaluation metrics
    # =============================================================================
    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10])
    )

    # =============================================================================
    # Run experiments
    # =============================================================================
    print(f"\n--- Running experiments for seed {seed} ---")
    run_omnirec(
        datasets=processed_datasets,
        plan=plan,
        evaluator=evaluator
    )

    # =============================================================================
    # Extract results
    # =============================================================================
    results_dict = evaluator.get_results()
    for dataset_id, df in results_dict.items():
        # Determine which dataset name this corresponds to
        matched_name = None
        for ds_name in DATASET_CONFIGS:
            if ds_name.lower() in dataset_id.lower():
                matched_name = ds_name
                break
        if matched_name is None:
            matched_name = dataset_id  # fallback

        for _, row in df.iterrows():
            all_results.append({
                "seed": seed,
                "dataset": matched_name,
                "algorithm": row["algorithm"],
                "metric": row["name"],
                "k": row["k"],
                "value": row["value"],
            })

        print(f"  -> Extracted {len(df)} metric rows for {matched_name} (seed {seed})")

# =============================================================================
# Analysis: Compute mean and std across seeds
# =============================================================================
print(f"\n{'='*80}")
print(f"FINAL RESULTS - Aggregated across {len(SEEDS)} seeds")
print(f"{'='*80}")

results_df = pd.DataFrame(all_results)

# Group by dataset, algorithm, metric, k and compute mean + std
summary = (
    results_df
    .groupby(["dataset", "algorithm", "metric", "k"])["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

# Print summary per dataset
for ds_name in DATASET_CONFIGS:
    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*70}")

    ds_summary = summary[summary["dataset"] == ds_name]

    for algo_name in sorted(ds_summary["algorithm"].unique()):
        print(f"\n  Algorithm: {algo_name}")
        print(f"  {'Metric':<15} {'k':<5} {'Mean':<10} {'Std':<10} {'CV(%)':<10}")
        print(f"  {'-'*50}")

        algo_data = ds_summary[ds_summary["algorithm"] == algo_name]
        for _, row in algo_data.sort_values(["metric", "k"]).iterrows():
            mean_val = row["mean"]
            std_val = row["std"]
            cv = (std_val / mean_val * 100) if mean_val > 0 else 0.0
            print(f"  {row['metric']:<15} {row['k']:<5} {mean_val:<10.6f} {std_val:<10.6f} {cv:<10.2f}")

# =============================================================================
# Statistical analysis of seed impact
# =============================================================================
print(f"\n{'='*80}")
print(f"STATISTICAL ANALYSIS: Impact of Data Split Seed Variation")
print(f"{'='*80}")

# Compute coefficient of variation (CV) as a measure of seed impact
pivot_cv = summary.copy()
pivot_cv["cv"] = np.where(
    pivot_cv["mean"] > 0,
    pivot_cv["std"] / pivot_cv["mean"] * 100,
    0.0
)

# Average CV per (dataset, algorithm) combination
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
max_cv_row = pivot_cv.loc[pivot_cv["cv"].idxmax()]
print(f"\nHighest seed sensitivity observed:")
print(f"  Dataset: {max_cv_row['dataset']}")
print(f"  Algorithm: {max_cv_row['algorithm']}")
print(f"  Metric: {max_cv_row['metric']} @ k={max_cv_row['k']}")
print(f"  Mean: {max_cv_row['mean']:.6f}, Std: {max_cv_row['std']:.6f}, CV: {max_cv_row['cv']:.2f}%")

# Min CV across all conditions (excluding zeros)
nonzero_cv = pivot_cv[pivot_cv["cv"] > 0]
if len(nonzero_cv) > 0:
    min_cv_row = nonzero_cv.loc[nonzero_cv["cv"].idxmin()]
    print(f"\nLowest seed sensitivity observed:")
    print(f"  Dataset: {min_cv_row['dataset']}")
    print(f"  Algorithm: {min_cv_row['algorithm']}")
    print(f"  Metric: {min_cv_row['metric']} @ k={min_cv_row['k']}")
    print(f"  Mean: {min_cv_row['mean']:.6f}, Std: {min_cv_row['std']:.6f}, CV: {min_cv_row['cv']:.2f}%")

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

print(f"\n{'='*80}")
print(f"EXPERIMENT COMPLETE")
print(f"{'='*80}")
print(f"\nSeeds used: {SEEDS}")
print(f"Total experiment configurations: {len(SEEDS)} seeds x {len(DATASET_CONFIGS)} datasets x 3 algorithms = {len(SEEDS) * len(DATASET_CONFIGS) * 3}")
print(f"Results stored in results_df with {len(results_df)} individual metric rows.")
