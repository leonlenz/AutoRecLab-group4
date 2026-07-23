#!/usr/bin/env python3
"""
Experiment: Quantifying the effect of data split random seeds on recommender accuracy.
Tests ALS (ImplicitMFScorer), ItemKNNScorer, and PopScorer on MovieLens100K,
Amazon2014VideoGames, and HetrecLastFM with 5 different random seeds each.

BUGFIX: Process datasets from smallest to largest to maximize results collected.
Uses checkpointing and saves partial results after each dataset.
Uses public API (SplitData.train/val/test) correctly.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import cast, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime

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

# Define dataset configs - order from smallest to largest to maximize data collected
# Dataset sizes (raw): HetrecLastFM=71K, MovieLens100K=100K, Amazon2014VideoGames=1.3M
DATASET_CONFIGS = [
    {
        "name": "HetrecLastFM",
        "dataset_enum": DataSet.HetrecLastFM,
        "make_implicit": False,  # Already implicit
        "implicit_threshold": None,
        "description": "Hetrec LastFM (71K implicit interactions)",
    },
    {
        "name": "MovieLens100K",
        "dataset_enum": DataSet.MovieLens100K,
        "make_implicit": True,
        "implicit_threshold": 3,
        "description": "MovieLens 100K (100K explicit ratings, converted to implicit)",
    },
    {
        "name": "Amazon2014VideoGames",
        "dataset_enum": DataSet.Amazon2014VideoGames,
        "make_implicit": True,
        "implicit_threshold": 3,
        "description": "Amazon Video Games (1.3M explicit ratings, converted to implicit)",
    },
]

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

# ─── Results storage ─────────────────────────────────────────────────────────
all_results = []  # list of dicts for building a DataFrame later
partial_results_path = os.path.join(working_dir, "partial_results.csv")

# ─── Helper function to save partial results ────────────────────────────────
def save_partial_results():
    """Save all collected results so far to CSV."""
    if all_results:
        partial_df = pd.DataFrame(all_results)
        partial_df.to_csv(partial_results_path, index=False)
        print(f"    [CHECKPOINT] Saved {len(all_results)} result rows to {partial_results_path}")

# ─── Main experiment loop ────────────────────────────────────────────────────
print("=" * 80)
print("OMNIREC EXPERIMENT: RANDOM SEED EFFECT ON RECOMMENDER ACCURACY")
print("=" * 80)
print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Datasets processed in order (smallest to largest):")
for ds_cfg in DATASET_CONFIGS:
    print(f"  - {ds_cfg['name']}: {ds_cfg['description']}")
print(f"Seeds: {SEEDS}")
print(f"Algorithms: {[a[1] for a in ALGORITHMS]}")
print(f"=" * 80)

# Track overall progress
datasets_completed = 0

for ds_config in DATASET_CONFIGS:
    dataset_name = ds_config["name"]
    dataset_enum = cast(DataSet, ds_config["dataset_enum"])
    make_implicit = cast(bool, ds_config["make_implicit"])
    implicit_threshold = cast(Optional[Union[int, float]], ds_config["implicit_threshold"])

    print(f"\n{'#' * 70}")
    print(f"#  DATASET: {dataset_name} ({ds_config['description']})")
    print(f"{'#' * 70}")

    # Track per-dataset progress
    seeds_completed = 0

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  {'─' * 60}")
        print(f"  Seed {seed} (iteration {seed_idx + 1}/{len(SEEDS)})")
        print(f"  {'─' * 60}")

        try:
            # Step 1: Set the global random state
            set_random_state(seed)
            current_state = get_random_state()
            print(f"    Random state set to: {current_state}")

            # Step 2: Load raw dataset (fresh copy each time)
            print(f"    Loading dataset...")
            dataset = RecSysDataSet.use_dataloader(dataset_enum)

            # Step 3: Build preprocessing pipeline
            pipe_steps = []

            if make_implicit:
                assert implicit_threshold is not None
                pipe_steps.append(MakeImplicit(implicit_threshold))

            pipe_steps.append(CorePruning(5))

            # Use 80/20 user-based holdout (test_size=0.2 gives 80/20).
            # validation_size=0 means no separate validation set (only train/test).
            # However, UserHoldout requires both parameters to be positive floats.
            # We use a small validation_size=0.1 and test_size=0.2 to get ~70/10/20 split.
            pipe_steps.append(UserHoldout(validation_size=0.1, test_size=0.2))

            pipeline = Pipe(*pipe_steps)

            print(f"    Preprocessing (MakeImplicit={make_implicit}, CorePruning=5, UserHoldout split)...")
            processed_dataset = pipeline.process(dataset)

            # Verify the split using the public SplitData API
            split_data = processed_dataset._data  # Access data variant (SplitData)
            train_df = split_data.train
            val_df = split_data.val
            test_df = split_data.test
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

            # Step 6: Run experiments for this seed
            print(f"    Running algorithms (ImplicitMFScorer, ItemKNNScorer, PopScorer)...")
            print(f"    This may take a while for large datasets...")

            # run_omnirec automatically handles environment setup, training, prediction,
            # and evaluation. It also prints result tables at the end.
            run_omnirec(
                datasets=processed_dataset,
                plan=plan,
                evaluator=evaluator,
            )

            # Step 7: Collect results using the public API
            results_dict = evaluator.get_results()
            # results_dict maps dataset_id (name-hash) -> DataFrame

            for dataset_id, result_df in results_dict.items():
                print(f"    Results ({dataset_id}):")
                for _, row in result_df.iterrows():
                    algo_short = row["algorithm"].split("-")[0] if "-" in row["algorithm"] else row["algorithm"]
                    metric_str = f"{row['name']}@{row['k']}" if pd.notna(row["k"]) else row["name"]
                    print(f"      {algo_short:40s} | {metric_str:12s} = {row['value']:.6f}")

                    all_results.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "algorithm": row["algorithm"],
                        "metric": row["name"],
                        "k": row["k"],
                        "value": row["value"],
                    })

            seeds_completed += 1
            print(f"    ✓ Seed {seed} completed successfully")

        except TimeoutError as e:
            print(f"    ⚠ TIMEOUT on {dataset_name}, seed {seed}: {e}")
            print(f"    Saving partial results and continuing with next dataset...")
            save_partial_results()
            # Break out of the seed loop - if we timeout, the rest of the seeds
            # for this dataset will likely also timeout
            break
        except Exception as e:
            print(f"    ✗ ERROR on {dataset_name}, seed {seed}: {e}")
            print(f"    Continuing with next seed...")
            continue

    datasets_completed += 1
    print(f"\n  Dataset {dataset_name}: completed {seeds_completed}/{len(SEEDS)} seeds")
    
    # Save results after each dataset
    save_partial_results()

# ─── Build final results DataFrame ──────────────────────────────────────────
print(f"\n{'=' * 80}")
print("  FINAL RESULTS AGGREGATION")
print(f"{'=' * 80}")

if len(all_results) == 0:
    print("ERROR: No results collected. Something went wrong with run_omnirec.")
    sys.exit(1)

results_df = pd.DataFrame(all_results)

print(f"\nTotal result rows: {len(results_df)}")
print(f"Datasets with results: {results_df['dataset'].unique()}")
print(f"Seeds with results: {sorted(results_df['seed'].unique())}")
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

print("\nSummary statistics across seeds:")
print("=" * 60)

for dataset_name in summary_stats["dataset"].unique():
    ds_mask = summary_stats["dataset"] == dataset_name
    ds_stats = summary_stats[ds_mask]
    print(f"\n  Dataset: {dataset_name}")
    print(f"  {'Algorithm':35s} {'Metric':12s} {'Mean':10s} {'Std':10s} {'CV':10s} {'Min':10s} {'Max':10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for _, row in ds_stats.iterrows():
        algo_short = row["algorithm"].split("-")[0] if "-" in row["algorithm"] else row["algorithm"]
        metric_str = f"{row['metric']}@{int(row['k'])}" if pd.notna(row["k"]) else row["metric"]
        print(f"  {algo_short:35s} {metric_str:12s} {row['mean']:10.6f} {row['std']:10.6f} {row['cv']:10.6f} {row['min']:10.6f} {row['max']:10.6f}")

# ─── Per-dataset, per-algorithm variation analysis ──────────────────────────
print(f"\n\n  Detailed per-(dataset, algorithm) seed variation:")
print(f"  {'=' * 50}")

pivot_data = []
grouped = cast(Any, results_df.groupby(["dataset", "algorithm"]))
for (ds, algo), group in grouped:
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
print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total runs attempted: {len(DATASET_CONFIGS)} datasets × {len(SEEDS)} seeds = {len(DATASET_CONFIGS) * len(SEEDS)} dataset-seed combinations")
print(f"  Each with {len(ALGORITHMS)} algorithms = {len(DATASET_CONFIGS) * len(SEEDS) * len(ALGORITHMS)} algorithm runs")
print(f"  Results collected: {len(results_df)} metric-value pairs")
print(f"\n  Datasets processed: {results_df['dataset'].nunique()}/{len(DATASET_CONFIGS)}")
print(f"  Seeds per dataset (unique):")
for ds in results_df['dataset'].unique():
    n_seeds = results_df[results_df['dataset'] == ds]['seed'].nunique()
    print(f"    - {ds}: {n_seeds} seeds")
print(f"\n  Working directory: {working_dir}")
print(f"  Key output files:")
print(f"    - experiment_results.csv: All raw metric values")
print(f"    - summary_statistics.csv: Mean, Std, Min, Max per metric")
print(f"    - seed_variation_analysis.csv: Seed variation per (dataset, algo, metric)")
print(f"    - partial_results.csv: Intermediate checkpoint file")
print(f"{'=' * 80}")
