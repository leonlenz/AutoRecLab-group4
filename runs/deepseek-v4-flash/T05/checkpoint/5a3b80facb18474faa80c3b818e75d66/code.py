#!/usr/bin/env python3
"""
Experiment: Quantifying the effect of data split random seeds on recommender accuracy.
Tests ALS (ImplicitMFScorer), ItemKNNScorer, and PopScorer on MovieLens100K,
Amazon2014VideoGames, and HetrecLastFM with 5 different random seeds each.
"""

import os
import sys
import json
import warnings
from typing import cast, Optional, Union
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
SEEDS = [42, 123, 256, 789, 1337]

DATASET_CONFIGS = {
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
    "HetrecLastFM": {
        "dataset_enum": DataSet.HetrecLastFM,
        "make_implicit": False,  # Already implicit
        "implicit_threshold": None,
    },
}

ALGORITHMS = [
    (LensKit.ImplicitMFScorer, "LensKit.ImplicitMFScorer"),
    (LensKit.ItemKNNScorer, "LensKit.ItemKNNScorer"),
    (LensKit.PopScorer, "LensKit.PopScorer"),
]

# All algorithms use default hyperparams with feedback='implicit'
ALGO_CONFIGS = {
    "LensKit.ImplicitMFScorer": {"feedback": "implicit"},
    "LensKit.ItemKNNScorer": {"feedback": "implicit"},
    "LensKit.PopScorer": {"feedback": "implicit"},
}

METRIC_KS = [1, 5, 10]

# ─── Collect results ─────────────────────────────────────────────────────────
all_results = []  # list of dicts for building a DataFrame later

print("=" * 80)
print("OMNIREC EXPERIMENT: RANDOM SEED EFFECT ON RECOMMENDER ACCURACY")
print("=" * 80)

# ─── Main experiment loop ────────────────────────────────────────────────────
for dataset_name, ds_config in DATASET_CONFIGS.items():
    print(f"\n{'─' * 70}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'─' * 70}")

    # Extract typed values from the config dict to avoid type-checker union issues
    dataset_enum: DataSet = cast(DataSet, ds_config["dataset_enum"])
    make_implicit: bool = cast(bool, ds_config["make_implicit"])
    implicit_threshold: Optional[Union[int, float]] = cast(Optional[Union[int, float]], ds_config["implicit_threshold"])

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  ── Seed {seed} (iteration {seed_idx + 1}/{len(SEEDS)}) ──")

        # Step 1: Load raw dataset (fresh copy each time)
        print(f"    Loading dataset...")
        dataset = RecSysDataSet.use_dataloader(dataset_enum)

        # Step 2: Set the global random state so UserHoldout uses this seed
        set_random_state(seed)
        current_state = get_random_state()
        print(f"    Random state set to: {current_state}")

        # Step 3: Build preprocessing pipeline
        pipe_steps = []

        if make_implicit:
            # implicit_threshold is guaranteed to be int|float when make_implicit is True
            assert implicit_threshold is not None
            pipe_steps.append(MakeImplicit(implicit_threshold))

        pipe_steps.append(CorePruning(5))

        # BUGFIX: UserHoldout requires a positive validation_size.
        # validation_size=0.0 causes sklearn train_test_split to receive test_size=0.0,
        # which is invalid (must be strictly in (0.0, 1.0)).
        # We use validation_size=0.1, test_size=0.2 to get a 70/10/20 split.
        pipe_steps.append(UserHoldout(validation_size=0.1, test_size=0.2))

        pipeline = Pipe(*pipe_steps)

        print(f"    Preprocessing (MakeImplicit={make_implicit}, CorePruning=5, UserHoldout=70/10/20)...")
        processed_dataset = pipeline.process(dataset)

        # Verify we got a SplitData using the public API
        train_df = processed_dataset._data.get("train")
        val_df = processed_dataset._data.get("val")
        test_df = processed_dataset._data.get("test")
        print(f"    Train interactions: {len(train_df)}, Val interactions: {len(val_df)}, Test interactions: {len(test_df)}")

        # Step 4: Create ExperimentPlan with all 3 algorithms
        plan = ExperimentPlan(plan_name=f"{dataset_name}_seed{seed}")

        for algo_enum, algo_name in ALGORITHMS:
            plan.add_algorithm(algo_enum, ALGO_CONFIGS[algo_name])

        # Step 5: Create Evaluator with ranking metrics
        evaluator = Evaluator(
            Precision(METRIC_KS),
            NDCG(METRIC_KS),
        )

        # Step 6: Run experiments
        print(f"    Running algorithms (ImplicitMFScorer, ItemKNNScorer, PopScorer)...")
        try:
            run_omnirec(
                datasets=processed_dataset,
                plan=plan,
                evaluator=evaluator,
            )
        except Exception as e:
            print(f"    ERROR during run_omnirec: {e}")
            print(f"    Skipping this (dataset={dataset_name}, seed={seed})...")
            continue

        # Step 7: Collect results
        results_dict = evaluator.get_results()
        # results_dict maps dataset_id (name-hash) -> DataFrame

        for dataset_id, result_df in results_dict.items():
            for _, row in result_df.iterrows():
                all_results.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "algorithm": row["algorithm"],
                    "metric": row["name"],
                    "k": row["k"],
                    "value": row["value"],
                })

            # Print results for monitoring
            print(f"    Results ({dataset_id}):")
            for _, row in result_df.iterrows():
                algo_short = row["algorithm"].split("-")[0] if "-" in row["algorithm"] else row["algorithm"]
                metric_str = f"{row['name']}@{row['k']}" if row['k'] is not None else row['name']
                print(f"      {algo_short:40s} | {metric_str:12s} = {row['value']:.6f}")

# ─── Build final results DataFrame ──────────────────────────────────────────
print(f"\n{'=' * 80}")
print("  FINAL RESULTS AGGREGATION")
print(f"{'=' * 80}")

results_df = pd.DataFrame(all_results)

if len(results_df) == 0:
    print("ERROR: No results collected. Something went wrong with run_omnirec.")
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

# Also compute coefficient of variation (CV = std/mean) as a scale-invariant measure
summary_stats["cv"] = summary_stats["std"] / summary_stats["mean"].replace(0, np.nan)

print("\nSummary statistics across 5 random seeds:")
print("=" * 60)

for dataset_name in summary_stats["dataset"].unique():
    ds_mask = summary_stats["dataset"] == dataset_name
    ds_stats = summary_stats[ds_mask]
    print(f"\n  Dataset: {dataset_name}")
    print(f"  {'Algorithm':30s} {'Metric':12s} {'Mean':10s} {'Std':10s} {'CV':10s} {'Min':10s} {'Max':10s}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for _, row in ds_stats.iterrows():
        algo_short = row["algorithm"].split("-")[0] if "-" in row["algorithm"] else row["algorithm"]
        metric_str = f"{row['metric']}@{int(row['k'])}" if pd.notna(row["k"]) else row["metric"]
        print(f"  {algo_short:30s} {metric_str:12s} {row['mean']:10.6f} {row['std']:10.6f} {row['cv']:10.6f} {row['min']:10.6f} {row['max']:10.6f}")

# ─── Per-dataset, per-algorithm variation analysis ──────────────────────────
print(f"\n\n  Detailed per-(dataset, algorithm) seed variation:")
print(f"  {'=' * 50}")

pivot_data = []
grouped = results_df.groupby(["dataset", "algorithm"])
for (ds, algo), group in grouped:  # type: ignore[union-attr]
    # Compute the range (max - min) as absolute variation
    metric_pivot = group.pivot_table(
        index="seed",
        columns=["metric", "k"],
        values="value",
    )
    # Compute std across seeds for each metric@k
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
print(f"\n  Total runs attempted: {len(SEEDS) * len(DATASET_CONFIGS)} dataset-seed combinations")
print(f"  Each with {len(ALGORITHMS)} algorithms = {len(SEEDS) * len(DATASET_CONFIGS) * len(ALGORITHMS)} algorithm runs")
print(f"  Results collected: {len(results_df)} metric-value pairs")
print(f"\n  Working directory: {working_dir}")
print(f"  Key output files:")
print(f"    - experiment_results.csv: All raw metric values")
print(f"    - summary_statistics.csv: Mean, Std, Min, Max per metric")
print(f"    - seed_variation_analysis.csv: Seed variation per (dataset, algo, metric)")
print(f"{'=' * 80}")
