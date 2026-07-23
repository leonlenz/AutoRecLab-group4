#!/usr/bin/env python3
"""
Experiment: Quantify how data split random seeds affect recommender system accuracy.

Runs ALS (ImplicitMFScorer), ItemKNN, and Pop on MovieLens100K, Amazon2014VideoGames,
and HetrecLastFM with 5 different random seeds for user-based 80/20 holdout splits.
Reports nDCG@k and Precision@k for k=1,5,10 with mean ± std across seeds.
"""

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
from omnirec.util.util import set_random_state, get_random_state

warnings.filterwarnings("ignore")

# ─── Working directory ──────────────────────────────────────────────────────
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
        "implicit_threshold": 3,
        "already_implicit": False,
    },
    "Amazon2014VideoGames": {
        "dataset_enum": DataSet.Amazon2014VideoGames,
        "implicit_threshold": 3,
        "already_implicit": False,
    },
    "HetrecLastFM": {
        "dataset_enum": DataSet.HetrecLastFM,
        "implicit_threshold": None,
        "already_implicit": True,
    },
}

ALGORITHMS = {
    "ALS": (LensKit.ImplicitMFScorer, {"feedback": "implicit"}),
    "ItemKNN": (LensKit.ItemKNNScorer, {"feedback": "implicit"}),
    "Pop": (LensKit.PopScorer, {"feedback": "implicit"}),
}

METRIC_KS = [1, 5, 10]

# ─── Helper: preprocess & cache each dataset once ────────────────────────────


def preprocess_dataset(ds_name: str, ds_cfg: DatasetConfig) -> RecSysDataSet:
    """Load raw dataset, apply 5-core + optional implicit conversion,
    save to .rsds cache, return the processed RawData dataset."""
    cache_path = Path(working_dir) / f"{ds_name}_preprocessed.rsds"

    if cache_path.exists():
        print(f"  Loading cached preprocessed dataset from {cache_path}")
        return RecSysDataSet.load(str(cache_path))

    print(f"  Preprocessing {ds_name} from scratch...")
    dataset = RecSysDataSet.use_dataloader(ds_cfg["dataset_enum"])

    # Build preprocessing pipeline for conversion + core pruning
    pipe_steps = []
    if not ds_cfg["already_implicit"]:
        # When already_implicit is False, threshold is guaranteed to be set
        assert ds_cfg["implicit_threshold"] is not None
        pipe_steps.append(MakeImplicit(ds_cfg["implicit_threshold"]))
    pipe_steps.append(CorePruning(5))

    pipeline = Pipe(*pipe_steps)
    processed = pipeline.process(dataset)  # Returns RecSysDataSet[RawData]

    processed.save(str(cache_path.with_suffix("")))  # .rsds appended automatically
    print(f"  Saved cached preprocessed dataset to {cache_path}")
    return processed


# ─── Main experiment loop ────────────────────────────────────────────────────

print("=" * 80)
print("EXPERIMENT: Impact of random seed on recommender accuracy")
print("=" * 80)
print(f"\nDatasets: {list(DATASETS_CFG.keys())}")
print(f"Algorithms: {list(ALGORITHMS.keys())}")
print(f"Seeds: {SEEDS}")
print(f"Total conditions: {len(DATASETS_CFG)} × {len(ALGORITHMS)} × {len(SEEDS)} = "
      f"{len(DATASETS_CFG) * len(ALGORITHMS) * len(SEEDS)}")

# ─── Preprocess all datasets once ────────────────────────────────────────────
print("\n" + "=" * 80)
print("Phase 1: Preprocess all datasets (one-time)")
print("=" * 80)

preprocessed_datasets = {}
for ds_name, ds_cfg in DATASETS_CFG.items():
    print(f"\n--- {ds_name} ---")
    preprocessed_datasets[ds_name] = preprocess_dataset(ds_name, ds_cfg)
    print(f"  Done. Interactions: {preprocessed_datasets[ds_name].num_interactions()}")

# ─── Run experiment grid ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("Phase 2: Run experiment grid")
print("=" * 80)

# Suppress OmniRec runner output noise
import logging
logging.getLogger().setLevel(logging.WARNING)

all_results = []  # list of dicts

for ds_name in DATASETS_CFG:
    print(f"\n{'─' * 70}")
    print(f"Dataset: {ds_name}")
    print(f"{'─' * 70}")

    # Load the preprocessed raw data from cache
    base_dataset = preprocessed_datasets[ds_name]

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed_idx + 1}/{len(SEEDS)} (seed={seed})")

        # Load a fresh copy of the preprocessed data and split with this seed
        set_random_state(seed)
        # UserHoldout(validation_size=0, test_size=0.2):
        #   - For each user, 20% of interactions → test, 80% → train
        #   - validation_size=0 → no validation split (empty val set)
        #   - This produces an exact 80/20 train/test split per user
        splitter = UserHoldout(validation_size=0, test_size=0.2)
        split_dataset = splitter.process(base_dataset)

        # Create experiment plan with all 3 algorithms (default hyperparams)
        plan = ExperimentPlan(f"{ds_name}_seed{seed}")
        for algo_name, (algo_enum, algo_cfg) in ALGORITHMS.items():
            plan.add_algorithm(algo_enum, algo_cfg)

        # Set up evaluator
        evaluator = Evaluator(
            NDCG(METRIC_KS),
            Precision(METRIC_KS),
        )

        # Run experiments
        try:
            run_omnirec(
                datasets=split_dataset,
                plan=plan,
                evaluator=evaluator,
            )
        except Exception as e:
            print(f"    WARNING: run_omnirec failed for {ds_name} seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Extract results
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

        n_rows = len(df_results) if 'df_results' in dir() else 0
        print(f"    Completed: extracted results")

    # Save intermediate results after each dataset (in case of timeout)
    if all_results:
        intermediate_df = pd.DataFrame(all_results)
        inter_path = os.path.join(working_dir, f"results_{ds_name}.csv")
        intermediate_df.to_csv(inter_path, index=False)
        print(f"  Saved intermediate results to {inter_path}")

# ─── Aggregate and Report ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("Phase 3: Statistical Analysis")
print("=" * 80)

if not all_results:
    print("No results collected. Something went wrong with experiment execution.")
    sys.exit(1)

results_df = pd.DataFrame(all_results)
print(f"\nTotal result rows: {len(results_df)}")

# Compute mean and std across seeds for each (dataset, algorithm, metric, k)
agg_results = (
    results_df.groupby(["dataset", "algorithm", "metric", "k"])["value"]
    .agg(["mean", "std"])
    .reset_index()
)

agg_results = agg_results.sort_values(["dataset", "algorithm", "metric", "k"])

# Print formatted results
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

# Save full results
output_path = os.path.join(working_dir, "experiment_results.csv")
agg_results.to_csv(output_path, index=False)
print(f"\n\nAggregated results saved to: {output_path}")

# Also save raw per-seed results
raw_path = os.path.join(working_dir, "experiment_results_raw.csv")
results_df.to_csv(raw_path, index=False)
print(f"Raw per-seed results saved to: {raw_path}")

# Compact summary table
print("\n" + "=" * 80)
print("COMPACT SUMMARY (mean ± std)")
print("=" * 80)

for dataset_name in agg_results["dataset"].unique():
    print(f"\n--- {dataset_name} ---")
    ds_results = agg_results[agg_results["dataset"] == dataset_name]
    print(f"{'Algorithm':<25} {'Metric':<12} {'k':<5} {'Mean':<10} {'Std':<10}")
    print("-" * 65)

    for _, row in ds_results.iterrows():
        algo_short = row["algorithm"].split("-")[0].split(".")[-1]
        print(f"{algo_short:<25} {row['metric']:<12} {int(row['k']):<5} "
              f"{row['mean']:<10.5f} {row['std']:<10.5f}")

print("\nExperiment complete!")
