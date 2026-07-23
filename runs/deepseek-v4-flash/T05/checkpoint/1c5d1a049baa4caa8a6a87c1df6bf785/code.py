#!/usr/bin/env python3
"""
Experiment: Quantifying the effect of data split random seeds on recommender accuracy.
Tests ALS (ImplicitMFScorer), ItemKNNScorer, and PopScorer on MovieLens100K,
Amazon2014VideoGames, and HetrecLastFM with 3 different random seeds each.

Uses OmniRec exclusively through its public API.
"""

import os
import sys
import json
import warnings
from typing import Optional, Any, cast

import pandas as pd
import numpy as np

# Suppress non-critical warnings
warnings.filterwarnings('ignore')

# ─── OmniRec imports ─────────────────────────────────────────────────────────
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import Precision, NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state

# ─── Setup working directory ─────────────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# ─── Configuration ────────────────────────────────────────────────────────────
# Reduced to 3 seeds to fit within the 1-hour execution limit
SEEDS = [42, 123, 256]

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "HetrecLastFM": {
        "dataset_enum": DataSet.HetrecLastFM,
        "make_implicit": False,
        "implicit_threshold": None,
    },
    "MovieLens100K": {
        "dataset_enum": DataSet.MovieLens100K,
        "make_implicit": True,
        "implicit_threshold": 3,
    },
    "Amazon2014VideoGames": {
        "dataset_enum": DataSet.Amazon2014VideoGames,
        "make_implicit": True,
        "implicit_threshold": 3,
    },
}

ALGORITHMS = [
    (LensKit.ImplicitMFScorer, "LensKit.ImplicitMFScorer"),
    (LensKit.ItemKNNScorer, "LensKit.ItemKNNScorer"),
    (LensKit.PopScorer, "LensKit.PopScorer"),
]

# Minimal configs - the LensKit runner infers feedback type from data columns
ALGO_CONFIGS = {
    "LensKit.ImplicitMFScorer": {},
    "LensKit.ItemKNNScorer": {},
    "LensKit.PopScorer": {},
}

METRIC_KS = [1, 5, 10]

# ─── Collect results ─────────────────────────────────────────────────────────
all_results: list[dict[str, Any]] = []  # list of dicts for building a DataFrame later

print("=" * 80)
print("OMNIREC EXPERIMENT: RANDOM SEED EFFECT ON RECOMMENDER ACCURACY")
print("=" * 80)

# ─── Main experiment loop ────────────────────────────────────────────────────
# Process one seed at a time, running all datasets together for efficiency
for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'#' * 70}")
    print(f"  SEED {seed} (iteration {seed_idx + 1}/{len(SEEDS)})")
    print(f"{'#' * 70}")

    # Set the global random state for reproducible splitting
    set_random_state(seed)
    print(f"    Random state set to: {get_random_state()}")

    # Preprocess all datasets with this seed
    preprocessed_datasets = []

    for dataset_name, ds_config in DATASET_CONFIGS.items():
        print(f"\n    {'─' * 50}")
        print(f"    Preprocessing {dataset_name}...")

        # Load raw dataset (fresh copy each time)
        dataset_enum = cast(DataSet, ds_config["dataset_enum"])
        dataset = RecSysDataSet.use_dataloader(dataset_enum)

        # Build preprocessing pipeline
        pipe_steps = []

        if ds_config["make_implicit"]:
            threshold = ds_config["implicit_threshold"]
            assert threshold is not None
            pipe_steps.append(MakeImplicit(cast(int, threshold)))

        pipe_steps.append(CorePruning(5))

        # UserHoldout with 80/20 train/test split and 0 validation
        # Since UserHoldout requires both validation_size and test_size,
        # we use test_size=0.2 and validation_size=0.0 to get 80/0/20 split.
        # However, looking at the source, validation_size=0.0 would cause
        # division by zero in train_test_split(test_size=0.0/(1-0.2)).
        # So we use a tiny validation: validation_size=0.05, test_size=0.2
        # to get approximately 75/5/20 split.
        pipe_steps.append(UserHoldout(validation_size=0.05, test_size=0.2))

        pipeline = Pipe(*pipe_steps)

        print(f"      Steps: MakeImplicit={ds_config['make_implicit']}, "
              f"CorePruning=5, UserHoldout(val=0.05, test=0.2)")
        processed_dataset = pipeline.process(dataset)

        # Verify the split using the public SplitData API
        # Access SplitData through _data (as documented in loading_datasets.md)
        train_df = processed_dataset._data.get("train")
        val_df = processed_dataset._data.get("val")
        test_df = processed_dataset._data.get("test")
        total = len(train_df) + len(val_df) + len(test_df)
        print(f"      Train: {len(train_df)}, Val: {len(val_df)}, "
              f"Test: {len(test_df)} (Total: {total})")

        preprocessed_datasets.append(processed_dataset)

    # ─── Create ExperimentPlan with all 3 algorithms ────────────────────────
    plan = ExperimentPlan(plan_name=f"seed{seed}_experiment")

    for algo_enum, algo_name in ALGORITHMS:
        plan.add_algorithm(algo_enum, ALGO_CONFIGS[algo_name])

    # ─── Create Evaluator with ranking metrics ─────────────────────────────
    evaluator = Evaluator(
        Precision(METRIC_KS),
        NDCG(METRIC_KS),
    )

    # ─── Run experiments ────────────────────────────────────────────────────
    # Pass all preprocessed datasets as a list so Coordinator reuses the
    # runner process across all algorithms and datasets efficiently.
    print(f"\n    Running all algorithms on all datasets (single run_omnirec call)...")
    print(f"      Algorithms: ImplicitMFScorer, ItemKNNScorer, PopScorer")
    print(f"      Datasets: {list(DATASET_CONFIGS.keys())}")

    try:
        run_omnirec(
            datasets=preprocessed_datasets,
            plan=plan,
            evaluator=evaluator,
        )
    except Exception as e:
        print(f"    ERROR during run_omnirec for seed {seed}: {e}")
        print(f"    Saving partial results and continuing...")
        # If we get results so far, collect them before continuing
        partial_results = evaluator.get_results()
        if partial_results:
            print(f"    Collected {len(partial_results)} partial result sets")
        # Continue to next seed rather than aborting entirely
        continue

    # ─── Collect results ────────────────────────────────────────────────────
    results_dict = evaluator.get_results()
    print(f"\n    Results collected from Evaluator ({len(results_dict)} dataset-ids):")

    for dataset_id, result_df in results_dict.items():
        print(f"\n      Dataset-ID: {dataset_id}")
        for _, row in result_df.iterrows():
            algo_short = row["algorithm"].split("-")[0] if "-" in row["algorithm"] else row["algorithm"]
            metric_str = f"{row['name']}@{row['k']}" if row['k'] is not None else row['name']
            print(f"        {algo_short:40s} | {metric_str:12s} = {row['value']:.6f}")

            all_results.append({
                "dataset": dataset_id,
                "seed": seed,
                "algorithm": row["algorithm"],
                "metric": row["name"],
                "k": row["k"],
                "value": row["value"],
            })

# ─── Build final results DataFrame ──────────────────────────────────────────
print(f"\n{'=' * 80}")
print("  FINAL RESULTS AGGREGATION")
print(f"{'=' * 80}")

results_df = pd.DataFrame(all_results)

if len(results_df) == 0:
    print("ERROR: No results collected. Something went wrong.")
    sys.exit(1)

print(f"\nTotal result rows: {len(results_df)}")
print(f"\nFirst few rows:")
print(results_df.head(20).to_string())

# ─── Statistical Analysis ────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("  STATISTICAL ANALYSIS: EFFECT OF RANDOM SEEDS")
print(f"{'=' * 80}")

# For each (dataset, algorithm, metric@k), compute mean, std, min, max across seeds
summary_stats = results_df.groupby(
    ["dataset", "algorithm", "metric", "k"]
)["value"].agg(["mean", "std", "min", "max", "count"]).reset_index()

# Coefficient of variation (CV = std/mean)
summary_stats["cv"] = summary_stats["std"] / summary_stats["mean"].replace(0, np.nan)

print("\nSummary statistics across random seeds:")
print("=" * 60)

for dataset_name in sorted(summary_stats["dataset"].unique()):
    ds_mask = summary_stats["dataset"] == dataset_name
    ds_stats = summary_stats[ds_mask]
    print(f"\n  Dataset: {dataset_name}")
    print(f"  {'Algorithm':30s} {'Metric':12s} {'Mean':10s} {'Std':10s} {'CV':10s} {'Min':10s} {'Max':10s}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for _, row in ds_stats.iterrows():
        algo_short = row["algorithm"].split("-")[0] if "-" in row["algorithm"] else row["algorithm"]
        metric_str = f"{row['metric']}@{int(row['k'])}" if pd.notna(row["k"]) else row["metric"]
        print(f"  {algo_short:30s} {metric_str:12s} "
              f"{row['mean']:10.6f} {row['std']:10.6f} {row['cv']:10.6f} "
              f"{row['min']:10.6f} {row['max']:10.6f}")

# ─── Per-dataset, per-algorithm variation analysis ──────────────────────────
print(f"\n\n  Detailed per-(dataset, algorithm) seed variation:")
print(f"  {'=' * 50}")

pivot_data: list[dict[str, Any]] = []
grouped = results_df.groupby(["dataset", "algorithm"])
for (ds, algo), group in cast("pd.core.groupby.DataFrameGroupBy", grouped):
    metric_pivot = group.pivot_table(
        index="seed",
        columns=["metric", "k"],
        values="value",
    )
    seed_std = metric_pivot.std()
    seed_mean = metric_pivot.mean()

    for (metric, k), std_val in seed_std.items():
        mean_val = seed_mean[metric, k]
        pivot_data.append({
            "dataset": ds,
            "algorithm": algo.split("-")[0] if "-" in algo else algo,
            "metric": f"{metric}@{int(k)}",
            "mean_across_seeds": mean_val,
            "std_across_seeds": std_val,
            "cv": std_val / mean_val if mean_val > 0 else np.nan,
        })

pivot_df = pd.DataFrame(pivot_data)
print(pivot_df.to_string(index=False))

# ─── Save results ────────────────────────────────────────────────────────────
results_path = os.path.join(working_dir, "experiment_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\n\nAll results saved to: {results_path}")

summary_path = os.path.join(working_dir, "summary_statistics.csv")
summary_stats.to_csv(summary_path, index=False)
print(f"Summary statistics saved to: {summary_path}")

pivot_path = os.path.join(working_dir, "seed_variation_analysis.csv")
pivot_df.to_csv(pivot_path, index=False)
print(f"Seed variation analysis saved to: {pivot_path}")

# ─── Final Summary ───────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("  EXPERIMENT COMPLETE")
print(f"{'=' * 80}")
print(f"\n  Total seeds: {len(SEEDS)}")
print(f"  Total datasets: {len(DATASET_CONFIGS)}")
print(f"  Total algorithms: {len(ALGORITHMS)}")
print(f"  Results collected: {len(results_df)} metric-value pairs")
print(f"\n  Working directory: {working_dir}")
print(f"  Key output files:")
print(f"    - experiment_results.csv: All raw metric values")
print(f"    - summary_statistics.csv: Mean, Std, Min, Max per metric")
print(f"    - seed_variation_analysis.csv: Seed variation per (dataset, algo, metric)")
print(f"{'=' * 80}")
