"""
Experiment: Quantify the effect of data split random seeds on recommender system accuracy.
Uses OmniRec framework exclusively with LensKit algorithms.
"""

import os
import sys
import warnings
from typing import Any

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── OmniRec Imports ───────────────────────────────────────────────────────────
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

# ─── Setup Working Directory ───────────────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# ─── Configuration ─────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
ALGORITHMS = {
    LensKit.PopScorer: {},
    LensKit.ItemKNNScorer: {"feedback": "implicit"},
    LensKit.ImplicitMFScorer: {"feedback": "implicit"},
}
DATASET_NAMES = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}

# Metrics: NDCG@k and Precision@k for k=1, 5, 10
EVALUATOR = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

# ─── Tracking Results ──────────────────────────────────────────────────────────
results_rows: list[dict[str, Any]] = []


def preprocess_dataset(
    dataset_name: str,
    data_set_enum: DataSet,
    seed: int,
) -> RecSysDataSet:
    """Load and preprocess a dataset for a specific random seed.

    Steps:
        1. Load raw dataset
        2. For MovieLens100K and Amazon: convert ratings > 3 to implicit
        3. Apply 5-core filtering
        4. Apply user-based 80/20 holdout split
    """
    set_random_state(seed)

    # Load the raw dataset
    dataset = RecSysDataSet.use_dataloader(data_set_enum)

    # Build preprocessing pipeline
    pipeline_steps = []

    # MovieLens100K and Amazon have explicit ratings -> convert to implicit
    if dataset_name in ("MovieLens100K", "Amazon2014VideoGames"):
        pipeline_steps.append(MakeImplicit(3))

    # 5-core filtering
    pipeline_steps.append(CorePruning(5))

    # User-based 80/20 holdout (no validation set)
    # validation_size=0.0 ensures no validation split is created
    pipeline_steps.append(UserHoldout(validation_size=0.0, test_size=0.2))

    pipeline = Pipe(*pipeline_steps)
    processed_dataset = pipeline.process(dataset)

    return processed_dataset


def run_experiment_for_seed(seed: int) -> None:
    """Run all algorithms on all datasets for a given random seed."""
    print(f"\n{'='*70}")
    print(f"  RUNNING EXPERIMENTS FOR SEED = {seed}")
    print(f"{'='*70}\n")

    for dataset_label, dataset_enum in DATASET_NAMES.items():
        print(f"\n  --- Processing {dataset_label} with seed {seed} ---")

        # Preprocess the dataset (fresh for each seed)
        try:
            processed_dataset = preprocess_dataset(
                dataset_name=dataset_label,
                data_set_enum=dataset_enum,
                seed=seed,
            )
        except Exception as e:
            print(f"  ERROR preprocessing {dataset_label} with seed {seed}: {e}")
            continue

        # Print dataset stats after preprocessing
        try:
            train_df = processed_dataset._data.get("train")
            test_df = processed_dataset._data.get("test")
            print(f"    Train: {len(train_df)} interactions, "
                  f"Users: {train_df['user'].nunique()}, "
                  f"Items: {train_df['item'].nunique()}")
            print(f"    Test:  {len(test_df)} interactions, "
                  f"Users: {test_df['user'].nunique()}, "
                  f"Items: {test_df['item'].nunique()}")
        except Exception:
            pass

        # Create experiment plan for this dataset+seed
        plan = ExperimentPlan(
            plan_name=f"{dataset_label}_seed{seed}"
        )

        # Add all three algorithms with default hyperparameters
        for algo_id, algo_config in ALGORITHMS.items():
            plan.add_algorithm(algo_id, algo_config)

        # Run experiments via OmniRec runner
        try:
            run_omnirec(
                datasets=processed_dataset,
                plan=plan,
                evaluator=EVALUATOR,
            )
            print(f"    ✓ Completed {dataset_label} seed {seed}")
        except Exception as e:
            print(f"    ✗ Error running {dataset_label} seed {seed}: {e}")
            import traceback
            traceback.print_exc()


def collect_and_analyze_results() -> None:
    """Collect results from checkpoint files and compute statistics."""
    print(f"\n{'='*70}")
    print(f"  COLLECTING AND ANALYZING RESULTS")
    print(f"{'='*70}\n")

    # Checkpoints directory structure:
    # ./checkpoints/<dataset>/<algorithm>/<config_index>/<fold>/

    all_metrics: list[dict[str, Any]] = []

    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Skipping analysis.")
        print("Results may have been stored differently by the runner.")
        return

    # Walk through checkpoint files to find evaluation results
    for dataset_name in DATASET_NAMES:
        dataset_path = os.path.join(checkpoint_dir, dataset_name)
        if not os.path.exists(dataset_path):
            # Try with seed appended
            continue

        for algo_name in os.listdir(dataset_path):
            algo_path = os.path.join(dataset_path, algo_name)
            if not os.path.isdir(algo_path):
                continue

            for config_idx in os.listdir(algo_path):
                config_path = os.path.join(algo_path, config_idx)
                if not os.path.isdir(config_path):
                    continue

                for fold_name in os.listdir(config_path):
                    fold_path = os.path.join(config_path, fold_name)
                    if not os.path.isdir(fold_path):
                        continue

                    # Look for eval results
                    eval_file = os.path.join(fold_path, "eval.csv")
                    if os.path.exists(eval_file):
                        eval_df = pd.read_csv(eval_file)
                        for _, row in eval_df.iterrows():
                            all_metrics.append({
                                "dataset": dataset_name,
                                "algorithm": algo_name,
                                "config_index": config_idx,
                                "fold": fold_name,
                                "metric": row.get("name", ""),
                                "k": row.get("k", ""),
                                "value": row.get("value", 0.0),
                            })

    if not all_metrics:
        print("No evaluation results found in checkpoints.")
        print("Let's try to find results in alternative locations...")
        # Look more broadly
        for root, dirs, files in os.walk(working_dir):
            for f in files:
                if "eval" in f.lower() or "metric" in f.lower() or "result" in f.lower():
                    filepath = os.path.join(root, f)
                    print(f"  Found: {filepath}")
        return

    # Convert to DataFrame for analysis
    metrics_df = pd.DataFrame(all_metrics)
    print(f"Collected {len(metrics_df)} metric records.\n")

    # Compute mean and std across seeds for each (dataset, algorithm, metric, k)
    print("=" * 90)
    print("  RESULTS: Mean ± Std across 5 random seeds")
    print("=" * 90)

    for dataset_name in sorted(metrics_df["dataset"].unique()):
        print(f"\n{'─'*90}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'─'*90}")

        ds_df = metrics_df[metrics_df["dataset"] == dataset_name]

        for algo_name in sorted(ds_df["algorithm"].unique()):
            print(f"\n  Algorithm: {algo_name}")
            algo_df = ds_df[ds_df["algorithm"] == algo_name]

            for metric_name in sorted(algo_df["metric"].unique()):
                metric_df = algo_df[algo_df["metric"] == metric_name]
                k_values = sorted(metric_df["k"].unique())

                for k_val in k_values:
                    k_df = metric_df[metric_df["k"] == k_val]
                    values = k_df["value"].values

                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                        print(f"    {metric_name}@{k_val}: {mean_val:.6f} ± {std_val:.6f} "
                              f"(n={len(values)} seeds)")


def main():
    print("╔" + "═" * 78 + "╗")
    print("║  Experiment: Quantifying Data Split Random Seed Effects on RS Accuracy  ║")
    print("╚" + "═" * 78 + "╝")
    print(f"  Datasets: {list(DATASET_NAMES.keys())}")
    print(f"  Algorithms: PopScorer, ItemKNNScorer, ImplicitMFScorer")
    print(f"  Seeds: {SEEDS}")
    print(f"  Metrics: NDCG@[1,5,10], Precision@[1,5,10]")
    print(f"  Working Dir: {working_dir}")

    # Run experiments for each seed
    for seed in SEEDS:
        run_experiment_for_seed(seed)

    # Analyze results
    collect_and_analyze_results()

    print(f"\n{'='*70}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved in: {working_dir}/checkpoints/")


if __name__ == "__main__":
    main()
