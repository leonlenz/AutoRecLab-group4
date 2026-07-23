#!/usr/bin/env python3
"""Experiment: Impact of data split random seeds on recommender accuracy.

Full factorial: 3 algorithms x 3 datasets x 5 seeds = 45 runs.
"""

import os
import sys
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# OmniRec imports
# ──────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1111]
DATASET_NAMES = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}
# Note: HetrecLastFM already provides implicit feedback (rating=1),
# so MakeImplicit is not needed for it in the pipeline.

ALGORITHMS = [
    (LensKit.PopScorer, "PopScorer"),
    (LensKit.ItemKNNScorer, "ItemKNNScorer"),
    (LensKit.ImplicitMFScorer, "ImplicitMFScorer"),
]

# Standard hyperparameters with feedback='implicit'
ALGO_CONFIGS = {
    "PopScorer": {"feedback": "implicit"},
    "ItemKNNScorer": {"feedback": "implicit"},
    "ImplicitMFScorer": {"feedback": "implicit"},
}

METRIC_KS = [1, 5, 10]

# Working directory
WORKING_DIR = os.path.join(os.getcwd(), "working")
os.makedirs(WORKING_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Collect results across all seeds
# ──────────────────────────────────────────────────────────────────────
# We'll store per-seed results as a list of dicts
all_results = []  # list of dicts with keys: seed, dataset, algorithm, metric, k, value

print("=" * 80)
print("EXPERIMENT: Impact of Data Split Random Seeds on Recommender Accuracy")
print("=" * 80)

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'─' * 80}")
    print(f"SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'─' * 80}")

    # ── Set global random state for reproducibility ──
    set_random_state(seed)

    # ── Create a fresh evaluator for this seed ──
    evaluator = Evaluator(
        NDCG(METRIC_KS),
        Precision(METRIC_KS),
    )

    # ── Build the experiment plan ──
    plan = ExperimentPlan(plan_name=f"SeedImpact-{seed}")

    # Add all three algorithms with standard hyperparameters
    for algo_enum, algo_name in ALGORITHMS:
        plan.add_algorithm(algo_enum, ALGO_CONFIGS[algo_name])

    # ── Preprocess each dataset ──
    processed_datasets = []
    for ds_name, ds_enum in DATASET_NAMES.items():
        print(f"  Loading dataset: {ds_name}")

        # Load raw dataset
        dataset = RecSysDataSet.use_dataloader(ds_enum)

        # Build preprocessing pipeline
        if ds_name == "HetrecLastFM":
            # HetrecLastFM is already implicit (rating=1), no need for MakeImplicit
            pipeline = Pipe(
                CorePruning(5),
                UserHoldout(validation_size=0.0, test_size=0.2),
            )
        else:
            # MovieLens100K and Amazon2014VideoGames: convert ratings > 3 to implicit
            pipeline = Pipe(
                MakeImplicit(3),
                CorePruning(5),
                UserHoldout(validation_size=0.0, test_size=0.2),
            )

        dataset = pipeline.process(dataset)
        processed_datasets.append(dataset)

    # ── Run all algorithms on all datasets ──
    print(f"  Running experiment plan (3 algorithms × {len(processed_datasets)} datasets)...")
    sys.stdout.flush()

    run_omnirec(
        datasets=processed_datasets,
        plan=plan,
        evaluator=evaluator,
    )

    # ── Collect results from the evaluator ──
    results_dict = evaluator.get_results()
    for dataset_key, df in results_dict.items():
        for _, row in df.iterrows():
            all_results.append({
                "seed": seed,
                "dataset": dataset_key,
                "algorithm": row["algorithm"],
                "metric": row["name"],
                "k": row["k"],
                "value": row["value"],
            })

    print(f"  ✓ Seed {seed} complete.")

# ──────────────────────────────────────────────────────────────────────
# Aggregate and Report Results
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("AGGREGATED RESULTS")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Helper to extract short algorithm name from the full algorithm string
def short_algo_name(full_name):
    """Extract algorithm name like 'PopScorer', 'ItemKNNScorer', 'ImplicitMFScorer'."""
    for name_part in ["PopScorer", "ItemKNNScorer", "ImplicitMFScorer"]:
        if name_part in full_name:
            return name_part
    return full_name

results_df["algo_short"] = results_df["algorithm"].apply(short_algo_name)

# Helper to extract short dataset name from the dataset key
def short_ds_name(full_name):
    """Extract a short dataset name."""
    for ds in ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]:
        if ds in full_name:
            return ds
    return full_name

results_df["ds_short"] = results_df["dataset"].apply(short_ds_name)

# Compute per-seed, per-dataset, per-algorithm, per-metric-k statistics
group_cols = ["seed", "ds_short", "algo_short", "metric", "k"]
per_seed_stats = results_df.groupby(group_cols)["value"].first().reset_index()

# Compute mean and std across seeds
agg_group_cols = ["ds_short", "algo_short", "metric", "k"]
stats = per_seed_stats.groupby(agg_group_cols)["value"].agg(["mean", "std"]).reset_index()
stats.columns = list(agg_group_cols) + ["mean", "std"]

# Sort for nice display
stats = stats.sort_values(["ds_short", "algo_short", "metric", "k"])

# ── Print per-(dataset, algorithm) summary tables ──
for ds in ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]:
    print(f"\n{'─' * 80}")
    print(f"Dataset: {ds}")
    print(f"{'─' * 80}")

    ds_stats = stats[stats["ds_short"] == ds]

    for algo in ["PopScorer", "ItemKNNScorer", "ImplicitMFScorer"]:
        algo_stats = ds_stats[ds_stats["algo_short"] == algo]
        if algo_stats.empty:
            continue
        print(f"\n  Algorithm: {algo}")
        print(f"  {'Metric':<12} {'k':>3} {'Mean':>10} {'Std':>10}")
        print(f"  {'─'*12} {'─'*3} {'─'*10} {'─'*10}")
        for _, row in algo_stats.iterrows():
            print(f"  {row['metric']:<12} {row['k']:>3} {row['mean']:>10.6f} {row['std']:>10.6f}")

# ── Print detailed per-seed results table ──
print(f"\n{'=' * 80}")
print("DETAILED PER-SEED RESULTS")
print("=" * 80)

# Pivot for a nice cross-table
pivot_df = per_seed_stats.pivot_table(
    index=["ds_short", "algo_short", "metric", "k"],
    columns="seed",
    values="value",
)
pivot_df = pivot_df.round(6)
print(f"\n{pivot_df.to_string()}")

# ── Statistical summary: Mean ± Std across seeds ──
print(f"\n{'=' * 80}")
print("MEAN ± STD ACROSS 5 SEEDS (Quantifying Random Seed Impact)")
print("=" * 80)

# Create a combined column
stats["mean_std"] = stats.apply(
    lambda r: f"{r['mean']:.6f} ± {r['std']:.6f}", axis=1
)

pivot_stats = stats.pivot_table(
    index=["ds_short", "algo_short"],
    columns=["metric", "k"],
    values="mean_std",
    aggfunc="first",
)

print(f"\n{pivot_stats.to_string()}")

print(f"\n{'=' * 80}")
print("EXPERIMENT COMPLETE")
print("=" * 80)
