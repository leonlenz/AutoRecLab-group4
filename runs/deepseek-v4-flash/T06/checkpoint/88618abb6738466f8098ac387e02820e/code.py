#!/usr/bin/env python3
"""
Experiment: Quantifying Impact of Random Seeds on Recommender Accuracy
==============================================================================
Tests three LensKit algorithms (Pop, ItemKNN, ImplicitMF/ALS) on three datasets
(MovieLens100K, Amazon2014VideoGames, HetrecLastFM) across 5 different random
seeds for data splitting. Measures nDCG@k and Precision@k for k=1,5,10.
"""
import os
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

DATASET_NAMES = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}

# Per-dataset algorithm configs to handle time constraints
# For MovieLens100K and HetrecLastFM (smaller datasets) - fewer epochs
# For Amazon2014VideoGames (large dataset) - even fewer epochs for ALS
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
        "epochs": 20,  # Reduced from 100 to avoid timeout on large datasets
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
    #
    # Apply MakeImplicit BEFORE CorePruning:
    # After implicit conversion, many users may have fewer interactions.
    # CorePruning(5) after conversion ensures every user has >=5 implicit
    # interactions, preventing UserHoldout from crashing on users with 1 item.
    #
    # Using UserHoldout(validation_size=0.0, test_size=0.2) as specified:
    # This creates an ~80/20 train/test split with no validation set.
    # -----------------------------------------------------------------------
    raw_datasets = {}
    for ds_name, ds_enum in DATASET_NAMES.items():
        print(f"\n--- Loading dataset: {ds_name} ---")
        ds = RecSysDataSet.use_dataloader(ds_enum)
        # num_interactions returns a dict for SplitData, an int for RawData
        n_inter = ds.num_interactions()
        if isinstance(n_inter, dict):
            print(f"  Loaded: {sum(n_inter.values())} interactions")
        else:
            print(f"  Loaded: {n_inter} interactions")

        # Build preprocessing pipeline
        pipeline_steps = []

        # Convert explicit datasets to implicit (ratings > 3 => rating >= 4)
        if ds_name in ("MovieLens100K", "Amazon2014VideoGames"):
            print(f"  Converting to implicit (threshold=4, i.e. ratings > 3)...")
            pipeline_steps.append(MakeImplicit(4))
        else:
            print(f"  Dataset is already implicit (HetrecLastFM), skipping conversion.")

        # Apply 5-core filtering AFTER implicit conversion to guarantee
        # every remaining user has at least 5 interactions
        print(f"  Applying 5-core filtering...")
        pipeline_steps.append(CorePruning(5))

        # Apply all preprocessing steps
        if pipeline_steps:
            pipeline = Pipe(*pipeline_steps)
            ds = pipeline.process(ds)

        n_inter = ds.num_interactions()
        if isinstance(n_inter, dict):
            print(f"  After preprocessing: {sum(n_inter.values())} interactions")
        else:
            print(f"  After preprocessing: {n_inter} interactions")
        raw_datasets[ds_name] = ds

    # -----------------------------------------------------------------------
    # Step 2: Run experiments for each seed
    # -----------------------------------------------------------------------
    # Structure: results[ds_name][seed][algo_name] = {metric_key: value}
    all_results = {}

    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'=' * 70}")
        print(f"SEED {seed_idx + 1}/{len(RANDOM_SEEDS)}: seed = {seed}")
        print(f"{'=' * 70}")

        for ds_name, raw_ds in raw_datasets.items():
            print(f"\n  --- Processing {ds_name} ---")

            # Set random state for reproducibility
            set_random_state(seed)

            # Apply user-based holdout split: ~80% train, ~20% test
            # validation_size=0.0 means no validation set
            splitter = UserHoldout(validation_size=0.0, test_size=0.2)
            split_ds = splitter.process(raw_ds)

            # Print basic stats using the public API
            n_inter = split_ds.num_interactions()
            if isinstance(n_inter, dict):
                train_count = n_inter.get("train", 0)
                val_count = n_inter.get("val", 0)
                test_count = n_inter.get("test", 0)
                print(f"    Train interactions: {train_count}")
                print(f"    Val interactions:   {val_count}")
                print(f"    Test interactions:  {test_count}")

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

            # Run experiments (the run_omnirec function prints tables automatically)
            try:
                run_omnirec(split_ds, plan, evaluator)

                # Collect results from evaluator.get_results()
                # Returns dict[str, DataFrame] with columns:
                #   algorithm, fold, name, k, value
                results_dict = evaluator.get_results()

                for ds_id, df_results in results_dict.items():
                    if ds_name not in all_results:
                        all_results[ds_name] = {}
                    if seed not in all_results[ds_name]:
                        all_results[ds_name][seed] = {}

                    for _, row in df_results.iterrows():
                        algo_str = str(row["algorithm"])
                        metric_name = str(row["name"])
                        k_val = row["k"] if not pd.isna(row["k"]) else None
                        value = float(row["value"])

                        if algo_str not in all_results[ds_name][seed]:
                            all_results[ds_name][seed][algo_str] = {}
                        if k_val is not None:
                            key = f"{metric_name}@{k_val}"
                        else:
                            key = metric_name
                        all_results[ds_name][seed][algo_str][key] = value

            except Exception as e:
                print(f"    ERROR running experiment: {e}")
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
            print(f"  {'Metric':<20} {'Mean':<12} {'Std':<12} {'CV':<12} {'Values per seed':<50}")
            print(f"  {'-' * 106}")

            # Collect all metric keys across seeds for this algo
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
                        f"  {metric_key:<20} {mean_val:<12.6f} {std_val:<12.6f} {cv_val:<12.6f} {seeds_str:<50}"
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
                    print(f"  {metric_key:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

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
    print("SUMMARY TABLE (Mean +/- Std across 5 seeds)")
    print("=" * 80)
    for ds_name in sorted(all_results.keys()):
        print(f"\nDataset: {ds_name}")
        ds_summary = summary_df[summary_df["dataset"] == ds_name]
        for algo in sorted(ds_summary["algorithm"].unique()):
            print(f"  {algo}:")
            algo_df = ds_summary[ds_summary["algorithm"] == algo]
            for _, row in algo_df.iterrows():
                print(
                    f"    {row['metric']:<20} = {row['mean']:.4f} +/- {row['std']:.4f}  (CV={row['cv']:.4f})"
                )

    print("\n\nExperiment complete!")


if __name__ == "__main__":
    main()
