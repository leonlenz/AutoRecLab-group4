#!/usr/bin/env python3
"""
Experiment: Quantifying Impact of Random Seeds on Recommender System Accuracy
==============================================================================
Tests three LensKit algorithms (Pop, ItemKNN, ImplicitMF) on three datasets
(MovieLens100K, Amazon2014VideoGames, HetrecLastFM) across 5 different random
seeds for data splitting. Measures nDCG@k and Precision@k for k=1,5,10.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

# =============================================================================
# OmniRec imports
# =============================================================================
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.preprocess.pipe import Pipe
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEEDS = [42, 73, 123, 256, 999]  # 5 different seeds
WORKING_DIR = os.path.join(os.getcwd(), "working")
os.makedirs(WORKING_DIR, exist_ok=True)
os.chdir(WORKING_DIR)

DATASET_NAMES = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}

ALGORITHM_CONFIGS = {
    LensKit.PopScorer: {
        "feedback": "implicit",
    },
    LensKit.ItemKNNScorer: {
        "max_nbrs": 30,
        "min_nbrs": 5,
        "feedback": "implicit",
    },
    LensKit.ImplicitMFScorer: {
        "features": 50,
        "epochs": 100,          # CORRECTED: was "iterations" - LensKit uses "epochs"
        "feedback": "implicit",
    },
}

METRICS_KS = [1, 5, 10]


def main():
    print("=" * 80)
    print("EXPERIMENT: Effect of Random Seeds on Recommender Accuracy")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Load and preprocess datasets (once per dataset)
    # -----------------------------------------------------------------------
    raw_datasets = {}
    for ds_name, ds_enum in DATASET_NAMES.items():
        print(f"\n--- Loading dataset: {ds_name} ---")
        ds = RecSysDataSet.use_dataloader(ds_enum)
        print(f"  Loaded: {ds.num_interactions()} interactions")

        # Apply 5-core filtering
        print(f"  Applying 5-core filtering...")
        ds = CorePruning(5).process(ds)
        print(f"  After core pruning: {ds.num_interactions()} interactions")

        # Convert explicit datasets to implicit (ratings > 3 => rating >= 4)
        # MakeImplicit with threshold=4 means keep ratings >= 4 (i.e. ratings > 3)
        if ds_name in ("MovieLens100K", "Amazon2014VideoGames"):
            print(f"  Converting to implicit (threshold=4, i.e. ratings >= 4)...")
            ds = MakeImplicit(4).process(ds)
            print(f"  After implicit conversion: {ds.num_interactions()} interactions")
        else:
            # HetrecLastFM is already implicit
            print(f"  Dataset is already implicit, skipping conversion.")

        raw_datasets[ds_name] = ds

    # -----------------------------------------------------------------------
    # Step 2: Run experiments for each seed
    # -----------------------------------------------------------------------
    # Structure: results[ds_name][seed][algo_name] = {metric: value}
    all_results = {}

    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'=' * 70}")
        print(f"SEED {seed_idx + 1}/{len(RANDOM_SEEDS)}: seed = {seed}")
        print(f"{'=' * 70}")

        for ds_name, raw_ds in raw_datasets.items():
            print(f"\n  --- Processing {ds_name} ---")

            # Set random state for reproducibility
            set_random_state(seed)

            # Apply user-based holdout split.
            # UserHoldout always creates a 3-way split (train/val/test).
            # Using validation_size=0.0 would crash because internally it computes
            # test_size = 0.0/(1-0.2) = 0.0 which sklearn rejects.
            # So we use a small validation_size=0.1, test_size=0.2 giving 70/10/20 split.
            splitter = UserHoldout(validation_size=0.1, test_size=0.2)
            split_ds = splitter.process(raw_ds)

            # Verify split
            train_df = split_ds._data.get("train")
            val_df = split_ds._data.get("val")
            test_df = split_ds._data.get("test")
            print(f"    Train interactions: {len(train_df)}")
            print(f"    Validation interactions: {len(val_df)}")
            print(f"    Test interactions:  {len(test_df)}")
            n_users_train = train_df["user"].nunique()
            n_users_test = test_df["user"].nunique()
            print(f"    Train users: {n_users_train}, Test users: {n_users_test}")

            # Create experiment plan
            plan = ExperimentPlan(f"Seed{seed}_{ds_name}")

            # Add all three algorithms
            for algo_enum, algo_config in ALGORITHM_CONFIGS.items():
                plan.add_algorithm(algo_enum, algo_config)

            # Create evaluator with NDCG and Precision at k=1,5,10
            evaluator = Evaluator(
                NDCG(METRICS_KS),
                Precision(METRICS_KS),
            )

            # Run experiments
            try:
                # run_omnirec modifies dataset and stores results in evaluator
                run_omnirec(split_ds, plan, evaluator)

                # Collect results from evaluator
                results_dict = evaluator.get_results()
                for ds_id, df_results in results_dict.items():
                    # Store results keyed by (ds_name, seed)
                    if ds_name not in all_results:
                        all_results[ds_name] = {}
                    if seed not in all_results[ds_name]:
                        all_results[ds_name][seed] = {}

                    for _, row in df_results.iterrows():
                        algo_str = row["algorithm"]
                        metric_name = row["name"]
                        k_val = row["k"]
                        value = row["value"]

                        if algo_str not in all_results[ds_name][seed]:
                            all_results[ds_name][seed][algo_str] = {}
                        key = f"{metric_name}@{k_val}"
                        all_results[ds_name][seed][algo_str][key] = value

            except Exception as e:
                print(f"    ERROR running experiment: {e}")
                # Re-raise to see full traceback
                raise

    # -----------------------------------------------------------------------
    # Step 3: Statistical Analysis
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    summary_rows = []

    for ds_name in sorted(all_results.keys()):
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        # Collect all algo names from seeds
        algo_names = set()
        for seed_results in all_results[ds_name].values():
            algo_names.update(seed_results.keys())
        algo_names = sorted(algo_names)

        for algo_name in algo_names:
            print(f"\n  Algorithm: {algo_name}")
            print(f"  {'Metric':<15} {'Mean':<10} {'Std':<10} {'CV':<10} {'Seeds':<40}")
            print(f"  {'-'*75}")

            # Collect all metric keys
            metric_keys = set()
            for seed_results in all_results[ds_name].values():
                if algo_name in seed_results:
                    metric_keys.update(seed_results[algo_name].keys())
            metric_keys = sorted(metric_keys)

            for metric_key in metric_keys:
                values = []
                for seed in RANDOM_SEEDS:
                    if (
                        seed in all_results[ds_name]
                        and algo_name in all_results[ds_name][seed]
                        and metric_key in all_results[ds_name][seed][algo_name]
                    ):
                        values.append(all_results[ds_name][seed][algo_name][metric_key])
                    else:
                        values.append(np.nan)

                values_arr = np.array(values, dtype=float)
                valid = values_arr[~np.isnan(values_arr)]

                if len(valid) > 0:
                    mean_val = np.mean(valid)
                    std_val = np.std(valid, ddof=1)  # sample std
                    cv_val = std_val / mean_val if mean_val != 0 else np.nan

                    seeds_str = ", ".join([f"{v:.4f}" for v in values])
                    print(
                        f"  {metric_key:<15} {mean_val:<10.6f} {std_val:<10.6f} {cv_val:<10.6f} {seeds_str:<40}"
                    )

                    summary_rows.append(
                        {
                            "dataset": ds_name,
                            "algorithm": algo_name,
                            "metric": metric_key,
                            "mean": mean_val,
                            "std": std_val,
                            "cv": cv_val,
                            "values": values,
                        }
                    )
                else:
                    print(f"  {metric_key:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    # -----------------------------------------------------------------------
    # Step 4: Save full results to JSON
    # -----------------------------------------------------------------------
    results_path = os.path.join(WORKING_DIR, "experiment_results.json")
    serializable_results = {}
    for ds_name in all_results:
        serializable_results[ds_name] = {}
        for seed in all_results[ds_name]:
            serializable_results[ds_name][str(seed)] = {}
            for algo in all_results[ds_name][seed]:
                serializable_results[ds_name][str(seed)][algo] = {
                    k: float(v) for k, v in all_results[ds_name][seed][algo].items()
                }

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n\nResults saved to: {results_path}")

    # Save summary as CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(WORKING_DIR, "experiment_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")

    # Print summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE (Mean ± Std across 5 seeds)")
    print("=" * 80)
    for ds_name in sorted(all_results.keys()):
        print(f"\nDataset: {ds_name}")
        ds_summary = summary_df[summary_df["dataset"] == ds_name]
        for algo in sorted(ds_summary["algorithm"].unique()):
            print(f"  {algo}:")
            algo_df = ds_summary[ds_summary["algorithm"] == algo]
            for _, row in algo_df.iterrows():
                print(
                    f"    {row['metric']:<12} = {row['mean']:.4f} ± {row['std']:.4f}  (CV={row['cv']:.4f})"
                )

    print("\n\nExperiment complete!")


if __name__ == "__main__":
    main()
