#!/usr/bin/env python3
"""
Experiment: Impact of Data Split Random Seeds on Recommender System Accuracy.

Tests three algorithms (Pop, ItemKNN, ALS) on three datasets (MovieLens100K,
Amazon2014VideoGames, HetrecLastFM) across 5 different data split random seeds.
Reports mean and std of NDCG@k and Precision@k for k=[1,5,10].
"""

import os
import sys
from copy import deepcopy
from typing import cast

import numpy as np
import pandas as pd

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


def main():
    # Create working directory
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # Five explicitly defined random seeds for data splitting
    RANDOM_SEEDS = [42, 123, 456, 789, 1024]

    # Datasets to use (name -> DataSet enum)
    DATASET_SPECS = {
        "MovieLens100K": {
            "enum": DataSet.MovieLens100K,
            "make_implicit": True,  # Apply MakeImplicit(3)
        },
        "Amazon2014VideoGames": {
            "enum": DataSet.Amazon2014VideoGames,
            "make_implicit": True,  # Apply MakeImplicit(3)
        },
        "HetrecLastFM": {
            "enum": DataSet.HetrecLastFM,
            "make_implicit": False,  # Already implicit
        },
    }

    # Algorithms to test (all with feedback='implicit')
    ALGORITHMS = [
        (LensKit.PopScorer, "PopScorer"),
        (LensKit.ItemKNNScorer, "ItemKNNScorer"),
        (LensKit.ImplicitMFScorer, "ImplicitMFScorer"),
    ]

    # Evaluation metrics
    K_VALUES = [1, 5, 10]

    # ------------------------------------------------------------------
    # Storage: results[dataset_name][algorithm_display_name][seed] = DataFrame row
    # We'll collect all results into a single flat structure
    # ------------------------------------------------------------------
    all_records = []  # list of dicts

    # ------------------------------------------------------------------
    # Main experiment loop: for each seed, process all datasets
    # ------------------------------------------------------------------
    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx+1}/{len(RANDOM_SEEDS)}: seed = {seed}")
        print(f"{'='*80}")

        # For each dataset, preprocess and run
        for ds_name, ds_spec in DATASET_SPECS.items():
            print(f"\n--- Processing dataset: {ds_name} with seed {seed} ---")

            # Set random state BEFORE splitting to control the split randomness
            set_random_state(seed)

            # 1. Load dataset - cast to DataSet to satisfy the type checker
            ds_enum: DataSet = cast(DataSet, ds_spec["enum"])
            dataset = RecSysDataSet.use_dataloader(ds_enum)

            # 2. Build preprocessing pipeline
            pipeline_steps = []

            if ds_spec["make_implicit"]:
                # Apply MakeImplicit(3) BEFORE CorePruning
                pipeline_steps.append(MakeImplicit(3))

            # Apply CorePruning(5) to all datasets
            pipeline_steps.append(CorePruning(5))

            # Apply UserHoldout: 80/20 user-based holdout
            # validation_size=0.0 because we don't need validation for this experiment
            pipeline_steps.append(UserHoldout(validation_size=0.0, test_size=0.2))

            # Execute pipeline
            pipeline = Pipe(*pipeline_steps)
            processed_dataset = pipeline.process(dataset)

            # 3. Create experiment plan for this dataset + seed
            plan = ExperimentPlan(f"{ds_name}_seed{seed}")

            # Add all three algorithms with implicit feedback
            for algo_enum, algo_display in ALGORITHMS:
                plan.add_algorithm(
                    algo_enum,
                    {"feedback": "implicit"},
                )

            # 4. Create evaluator with NDCG@k and Precision@k
            evaluator = Evaluator(
                NDCG(K_VALUES),
                Precision(K_VALUES),
            )

            # 5. Run experiments
            print(f"  Running experiments for {ds_name} (seed={seed})...")
            # Note: run_omnirec will print its own tables; suppress by not printing
            run_omnirec(
                datasets=processed_dataset,
                plan=plan,
                evaluator=evaluator,
            )

            # 6. Collect results
            results_dict = evaluator.get_results()
            for dataset_id, df in results_dict.items():
                # Add dataset and seed info to each row
                for _, row in df.iterrows():
                    record = row.to_dict()
                    record["dataset"] = ds_name
                    record["seed"] = seed
                    all_records.append(record)

                print(f"  Collected {len(df)} metric rows for {dataset_id}")

        # End of this seed's experiments

    # ------------------------------------------------------------------
    # Statistical Analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS: Aggregating results across 5 seeds")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_records)

    # Parse algorithm name from the format "LensKit.PopScorer-<hash>"
    # Keep only what's before the hash
    results_df["algo_name"] = results_df["algorithm"].str.extract(
        r'^(LensKit\.[A-Za-z]+)'
    )[0]

    # Map algorithm identifiers to short display names
    algo_short_map = {
        "LensKit.PopScorer": "Pop",
        "LensKit.ItemKNNScorer": "ItemKNN",
        "LensKit.ImplicitMFScorer": "ALS",
    }
    results_df["algo_short"] = results_df["algo_name"].map(algo_short_map)

    # Filter to our metrics of interest
    metric_map = {"NDCG": "NDCG", "Precision": "Precision"}
    results_df = results_df[results_df["name"].isin(metric_map.keys())]

    # Group by dataset, algorithm short name, metric name, and k
    grouped = results_df.groupby(
        ["dataset", "algo_short", "name", "k"], as_index=False
    )["value"].agg(["mean", "std"]).reset_index()

    # Round for display
    grouped["mean"] = grouped["mean"].round(5)
    grouped["std"] = grouped["std"].round(5)

    # Print organized results per dataset
    for ds_name in DATASET_SPECS.keys():
        print(f"\n{'='*80}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*80}")

        ds_group = grouped[grouped["dataset"] == ds_name]

        for metric_name in ["NDCG", "Precision"]:
            print(f"\n  --- {metric_name}@k ---")
            metric_group = ds_group[ds_group["name"] == metric_name]

            # Pivot: algorithms as rows, k values as columns
            for algo in ["Pop", "ItemKNN", "ALS"]:
                algo_group = metric_group[metric_group["algo_short"] == algo]
                if algo_group.empty:
                    continue
                parts = []
                for _, row in algo_group.sort_values("k").iterrows():
                    parts.append(
                        f"k={int(row['k'])}: {row['mean']:.5f} ± {row['std']:.5f}"
                    )
                print(f"    {algo:10s}:  {' | '.join(parts)}")

    # Also save results to CSV for further analysis
    output_path = os.path.join(working_dir, "seed_randomness_experiment_results.csv")
    grouped.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Save raw results too
    raw_output_path = os.path.join(
        working_dir, "seed_randomness_experiment_raw.csv"
    )
    results_df.to_csv(raw_output_path, index=False)
    print(f"Raw results saved to: {raw_output_path}")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
