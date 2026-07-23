#!/usr/bin/env python3
"""
Experiment to quantify how data split random seeds affect recommender system accuracy.
Tests ALS (ImplicitMFScorer), ItemKNNScorer, and PopScorer on MovieLens100K,
Amazon2014VideoGames, and HetrecLastFM datasets across 5 random seeds.
"""

import os
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import numpy as np

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
    # Setup working directory
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    # Define datasets to use
    dataset_configs = [
        {
            'name': 'MovieLens100K',
            'dataset_enum': DataSet.MovieLens100K,
            'needs_implicit_conversion': True,
        },
        {
            'name': 'Amazon2014VideoGames',
            'dataset_enum': DataSet.Amazon2014VideoGames,
            'needs_implicit_conversion': True,
        },
        {
            'name': 'HetrecLastFM',
            'dataset_enum': DataSet.HetrecLastFM,
            'needs_implicit_conversion': False,  # Already implicit
        },
    ]

    # Define random seeds for data splitting reproducibility
    seeds = [42, 123, 456, 789, 1111]

    # Store all results for aggregation
    all_results = []

    for dconfig in dataset_configs:
        dataset_name = dconfig['name']
        dataset_enum = cast(DataSet, dconfig['dataset_enum'])
        needs_implicit = dconfig['needs_implicit_conversion']

        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")

        for seed_idx, seed in enumerate(seeds):
            print(f"\n  --- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")

            # Set random state BEFORE loading and preprocessing for reproducibility
            set_random_state(seed)

            # Load dataset
            dataset = RecSysDataSet.use_dataloader(dataset_enum)

            # Build preprocessing pipeline
            pipeline_steps = []

            if needs_implicit:
                # Convert ratings > 3 to implicit feedback
                # MakeImplicit(4) keeps ratings >= 4 (i.e., > 3)
                pipeline_steps.append(MakeImplicit(4))

            # Apply 5-core filtering
            pipeline_steps.append(CorePruning(5))

            # User-based 80/20 holdout split
            pipeline_steps.append(UserHoldout(validation_size=0.0, test_size=0.2))

            # Execute preprocessing pipeline
            pipeline = Pipe(*pipeline_steps)
            dataset = pipeline.process(dataset)

            print(f"    Dataset after preprocessing: {dataset}")

            # Create experiment plan
            plan = ExperimentPlan(f"{dataset_name}_seed{seed}")

            # Add three algorithms with default hyperparameters and implicit feedback
            plan.add_algorithm(
                LensKit.PopScorer,
                {"feedback": "implicit"}
            )
            plan.add_algorithm(
                LensKit.ItemKNNScorer,
                {"feedback": "implicit"}
            )
            plan.add_algorithm(
                LensKit.ImplicitMFScorer,
                {"feedback": "implicit"}
            )

            # Configure evaluation metrics
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )

            # Run all experiments for this dataset+seed combination
            print(f"    Running experiments...")
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

            # Collect results
            results_dict = evaluator.get_results()
            for dataset_id, result_df in results_dict.items():
                # Add metadata columns
                result_df = result_df.copy()
                result_df['dataset'] = dataset_name
                result_df['seed'] = seed
                all_results.append(result_df)

            print(f"    Completed seed {seed} for {dataset_name}")

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        # Print raw results summary
        print(f"\n{'='*80}")
        print("RAW RESULTS SUMMARY")
        print(f"{'='*80}")
        print(combined_results.to_string())

        # Aggregate: group by dataset, algorithm, name, k and compute mean/std across seeds
        print(f"\n\n{'='*80}")
        print("AGGREGATED RESULTS (mean ± std across 5 seeds)")
        print(f"{'='*80}")

        aggregated = combined_results.groupby(
            ['dataset', 'algorithm', 'name', 'k'], sort=False
        )['value'].agg(['mean', 'std']).reset_index()

        for _, row in aggregated.iterrows():
            metric_name = f"{row['name']}@{int(row['k'])}" if pd.notna(row['k']) else row['name']
            print(f"  {row['dataset']:25s} | {row['algorithm']:35s} | {metric_name:12s} = {row['mean']:.5f} ± {row['std']:.5f}")

        # Also print pivot tables for easier reading
        print(f"\n\n{'='*80}")
        print("PIVOT TABLE - Mean values by Dataset, Algorithm, and Metric")
        print(f"{'='*80}")

        for dataset_name in combined_results['dataset'].unique():
            print(f"\n--- {dataset_name} ---")
            mask = combined_results['dataset'] == dataset_name
            sub = combined_results[mask].copy()
            sub['metric_key'] = sub['name'] + '@' + sub['k'].astype(int).astype(str)
            
            # Pivot to show algorithm vs metric, averaged across seeds
            pivot = sub.pivot_table(
                index='algorithm',
                columns='metric_key',
                values='value',
                aggfunc=['mean', 'std'],
                sort=False
            )
            print(pivot.to_string())

        # Save results to CSV
        results_path = os.path.join(working_dir, 'experiment_results.csv')
        combined_results.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        # Save aggregated results
        agg_path = os.path.join(working_dir, 'aggregated_results.csv')
        aggregated.to_csv(agg_path, index=False)
        print(f"Aggregated results saved to: {agg_path}")

    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
