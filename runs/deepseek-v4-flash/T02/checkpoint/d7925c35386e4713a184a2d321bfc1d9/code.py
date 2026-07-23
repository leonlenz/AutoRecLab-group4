import os
import sys
import warnings
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

# OmniRec imports
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

warnings.filterwarnings("ignore")

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# ─── Configuration ───────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1111]


class DatasetConfig(TypedDict):
    dataset_enum: DataSet
    implicit_threshold: int | None
    already_implicit: bool


DATASETS_CFG: dict[str, DatasetConfig] = {
    "MovieLens100K": {
        "dataset_enum": DataSet.MovieLens100K,
        "implicit_threshold": 3,        # convert explicit -> implicit
        "already_implicit": False,
    },
    "Amazon2014VideoGames": {
        "dataset_enum": DataSet.Amazon2014VideoGames,
        "implicit_threshold": 3,        # convert explicit -> implicit
        "already_implicit": False,
    },
    "HetrecLastFM": {
        "dataset_enum": DataSet.HetrecLastFM,
        "implicit_threshold": None,     # already implicit
        "already_implicit": True,
    },
}

ALGORITHMS = {
    "ALS": (LensKit.ImplicitMFScorer, {"feedback": "implicit"}),
    "ItemKNN": (LensKit.ItemKNNScorer, {"feedback": "implicit"}),
    "Pop": (LensKit.PopScorer, {"feedback": "implicit"}),
}

METRICS = [1, 5, 10]

# ─── Load datasets ───────────────────────────────────────────────────────────
print("=" * 80)
print("Loading datasets...")
print("=" * 80)

loaded_datasets = {}
for ds_name, ds_cfg in DATASETS_CFG.items():
    print(f"\nLoading {ds_name}...")
    dataset = RecSysDataSet.use_dataloader(ds_cfg["dataset_enum"])
    print(f"  Loaded {dataset.num_interactions()} interactions, "
          f"rating range [{dataset.min_rating()}, {dataset.max_rating()}]")
    loaded_datasets[ds_name] = (dataset, ds_cfg)

# ─── Run Experiment Grid ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("Running experiment grid: 3 datasets x 3 algorithms x 5 seeds = 45 conditions")
print("=" * 80)

# Store all results per (dataset, algorithm, seed, metric, k) -> value
all_results = []

# Suppress OmniRec runner output noise
import logging
logging.getLogger().setLevel(logging.WARNING)

for ds_name, (raw_dataset, ds_cfg) in loaded_datasets.items():
    print(f"\n{'─' * 70}")
    print(f"Processing dataset: {ds_name}")
    print(f"{'─' * 70}")

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed_idx + 1}/{len(SEEDS)} (seed={seed})")

        # ── Step 1: Preprocess ──
        # Start from a fresh raw copy for each seed (clone lineage by re-loading)
        # Actually we need to reload to avoid mutation across seeds
        dataset = RecSysDataSet.use_dataloader(ds_cfg["dataset_enum"])

        # Build the preprocessing pipeline
        pipe_steps = []
        if not ds_cfg["already_implicit"]:
            pipe_steps.append(MakeImplicit(ds_cfg["implicit_threshold"]))
        pipe_steps.append(CorePruning(5))

        # Set random state for reproducible split
        set_random_state(seed)
        pipe_steps.append(UserHoldout(validation_size=0, test_size=0.2))

        pipeline = Pipe(*pipe_steps)
        split_dataset = pipeline.process(dataset)

        # ── Step 2: Create experiment plan with all 3 algorithms ──
        plan = ExperimentPlan(f"{ds_name}_seed{seed}")

        for algo_name, (algo_enum, algo_cfg) in ALGORITHMS.items():
            plan.add_algorithm(algo_enum, algo_cfg)

        # ── Step 3: Set up evaluator ──
        evaluator = Evaluator(
            NDCG(METRICS),
            Precision(METRICS),
        )

        # ── Step 4: Run experiments ──
        # Use a unique checkpoint dir per seed/dataset to avoid caching collisions
        chk_dir = os.path.join(working_dir, "checkpoints", ds_name, f"seed_{seed}")
        try:
            evaluator = run_omnirec(
                datasets=split_dataset,
                plan=plan,
                evaluator=evaluator,
            )
        except Exception as e:
            print(f"    WARNING: run_omnirec failed for {ds_name} seed {seed}: {e}")
            continue

        # ── Step 5: Extract results ──
        results_dict = evaluator.get_results()
        for dataset_id, df_results in results_dict.items():
            for _, row in df_results.iterrows():
                all_results.append({
                    "dataset": ds_name,
                    "seed": seed,
                    "algorithm": row["algorithm"],
                    "metric": row["name"],
                    "k": row["k"],
                    "value": row["value"],
                })

        print(f"    Completed: extracted {len(df_results)} metric rows")

# ─── Aggregate and Print Results ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS: Mean and Std across 5 seeds")
print("=" * 80)

if len(all_results) == 0:
    print("No results collected. Something went wrong with experiment execution.")
    sys.exit(1)

results_df = pd.DataFrame(all_results)

# Compute mean and std across seeds for each (dataset, algorithm, metric, k)
agg_results = (
    results_df.groupby(["dataset", "algorithm", "metric", "k"])["value"]
    .agg(["mean", "std"])
    .reset_index()
)

# Sort for nice display
agg_results = agg_results.sort_values(["dataset", "algorithm", "metric", "k"])

# Print results grouped by dataset
for dataset_name in agg_results["dataset"].unique():
    print(f"\n{'#' * 70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'#' * 70}")

    ds_mask = agg_results["dataset"] == dataset_name
    ds_results = agg_results[ds_mask]

    for algo_name in ds_results["algorithm"].unique():
        print(f"\n  Algorithm: {algo_name}")
        algo_mask = ds_results["algorithm"] == algo_name
        algo_df = ds_results[algo_mask]

        for metric_name in ["NDCG", "Precision"]:
            metric_mask = algo_df["metric"] == metric_name
            metric_df = algo_df[metric_mask].sort_values("k")
            if len(metric_df) > 0:
                print(f"    {metric_name}:")
                for _, row in metric_df.iterrows():
                    print(f"      @{int(row['k']):>2d}: mean={row['mean']:.5f}  "
                          f"std={row['std']:.5f}")

# Also save results to CSV for later analysis
output_path = os.path.join(working_dir, "experiment_results.csv")
agg_results.to_csv(output_path, index=False)
print(f"\n\nResults saved to: {output_path}")

# Print a compact summary table
print("\n" + "=" * 80)
print("COMPACT SUMMARY (mean ± std)")
print("=" * 80)

for dataset_name in agg_results["dataset"].unique():
    print(f"\n--- {dataset_name} ---")
    ds_mask = agg_results["dataset"] == dataset_name
    ds_results = agg_results[ds_mask]
    print(f"{'Algorithm':<20} {'Metric':<12} {'k':<5} {'Mean':<10} {'Std':<10}")
    print("-" * 60)

    for _, row in ds_results.iterrows():
        algo_short = row["algorithm"].split("-")[0].split(".")[-1]
        print(f"{algo_short:<20} {row['metric']:<12} {int(row['k']):<5} "
              f"{row['mean']:<10.5f} {row['std']:<10.5f}")

print("\nExperiment complete!")
