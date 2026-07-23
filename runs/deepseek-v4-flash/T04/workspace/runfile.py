import os
import sys
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

    # Define the 5 random seeds for reproducibility
    seeds = [42, 123, 456, 789, 1234]

    # =====================================================
    # Step 1: Load all three datasets
    # =====================================================
    print("=" * 80)
    print("Loading datasets...")
    print("=" * 80)

    # MovieLens100K (explicit ratings 1-5)
    ml100k_raw = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print(f"MovieLens100K loaded: {ml100k_raw.num_interactions()} interactions")

    # Amazon2014VideoGames (explicit ratings)
    amz_raw = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
    print(f"Amazon2014VideoGames loaded: {amz_raw.num_interactions()} interactions")

    # HetrecLastFM (already implicit)
    lastfm_raw = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
    print(f"HetrecLastFM loaded: {lastfm_raw.num_interactions()} interactions")

    # =====================================================
    # Step 2: Preprocess datasets (MakeImplicit + CorePruning)
    # =====================================================
    print("\n" + "=" * 80)
    print("Preprocessing datasets...")
    print("=" * 80)

    # MovieLens100K: MakeImplicit(3) then CorePruning(5)
    pipeline_ml = Pipe(
        MakeImplicit(3),
        CorePruning(5),
    )
    ml100k_processed = pipeline_ml.process(ml100k_raw)
    print(f"MovieLens100K after preprocessing: {ml100k_processed.num_interactions()} interactions")

    # Amazon2014VideoGames: MakeImplicit(3) then CorePruning(5)
    pipeline_amz = Pipe(
        MakeImplicit(3),
        CorePruning(5),
    )
    amz_processed = pipeline_amz.process(amz_raw)
    print(f"Amazon2014VideoGames after preprocessing: {amz_processed.num_interactions()} interactions")

    # HetrecLastFM: Only CorePruning(5) (already implicit)
    pipeline_lastfm = Pipe(
        CorePruning(5),
    )
    lastfm_processed = pipeline_lastfm.process(lastfm_raw)
    print(f"HetrecLastFM after preprocessing: {lastfm_processed.num_interactions()} interactions")

    # Store processed base datasets
    base_datasets = {
        "MovieLens100K": ml100k_processed,
        "Amazon2014VideoGames": amz_processed,
        "HetrecLastFM": lastfm_processed,
    }

    # =====================================================
    # Step 3-4-5: For each seed, split and run experiments
    # =====================================================
    print("\n" + "=" * 80)
    print("Running full factorial experiment: 3 datasets x 3 algorithms x 5 seeds")
    print("=" * 80)

    # Store all results for statistical analysis
    all_results = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Processing seed={seed}")
        print(f"{'=' * 60}")

        # Set the random state before splitting
        set_random_state(seed)

        # Split each dataset with the current seed using UserHoldout
        # NOTE: validation_size=0.0 is NOT supported by UserHoldout._process because
        # it computes valid_size/(1-test_size) and passes to sklearn's train_test_split
        # which requires test_size > 0. Using validation_size=0.1 instead (~70/10/20 split).
        splitter = UserHoldout(validation_size=0.1, test_size=0.2)

        split_datasets = {}
        for ds_name, ds_base in base_datasets.items():
            # Set random state again before each split for reproducibility
            set_random_state(seed)
            split_ds = splitter.process(ds_base)
            split_datasets[ds_name] = split_ds
            print(f"  {ds_name}: split into train/validation/test")

        # Create experiment plan with the 3 algorithms
        plan = ExperimentPlan(f"Seed-{seed}-Comparison")

        # Add PopScorer (popularity baseline) with implicit feedback
        plan.add_algorithm(
            LensKit.PopScorer,
            {"feedback": "implicit"}
        )

        # Add ItemKNNScorer with default hyperparameters and implicit feedback
        plan.add_algorithm(
            LensKit.ItemKNNScorer,
            {"feedback": "implicit"}
        )

        # Add ImplicitMFScorer (ALS) with default hyperparameters and implicit feedback
        plan.add_algorithm(
            LensKit.ImplicitMFScorer,
            {"feedback": "implicit"}
        )

        # Create evaluator with NDCG and Precision at k=[1, 5, 10]
        evaluator = Evaluator(
            NDCG([1, 5, 10]),
            Precision([1, 5, 10])
        )

        # Run experiments for all datasets with this plan
        dataset_list = [
            split_datasets["MovieLens100K"],
            split_datasets["Amazon2014VideoGames"],
            split_datasets["HetrecLastFM"],
        ]

        run_omnirec(
            datasets=dataset_list,
            plan=plan,
            evaluator=evaluator
        )

        # Collect results and add seed/dataset metadata
        results_dict = evaluator.get_results()
        for dataset_id, df_result in results_dict.items():
            df_result = df_result.copy()
            df_result["seed"] = seed
            # Extract base dataset name from the dataset_id (e.g., "MovieLens100K-abc123")
            for base_name in ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]:
                if base_name in dataset_id:
                    df_result["dataset"] = base_name
                    break
            all_results.append(df_result)

    # =====================================================
    # Step 6: Statistical Analysis
    # =====================================================
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS: Mean and Std across 5 seeds")
    print("=" * 80)

    # Combine all results
    if all_results:
        full_results = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal result rows: {len(full_results)}")
        print(f"Columns: {list(full_results.columns)}")

        # Group by dataset, algorithm, name, k and compute mean/std
        summary = full_results.groupby(
            ["dataset", "algorithm", "name", "k"]
        )["value"].agg(["mean", "std"]).reset_index()

        # Round values for display
        summary["mean"] = summary["mean"].round(6)
        summary["std"] = summary["std"].round(6)

        # Print summary tables
        for dataset in summary["dataset"].unique():
            print(f"\n{'=' * 70}")
            print(f"Dataset: {dataset}")
            print(f"{'=' * 70}")

            ds_summary = summary[summary["dataset"] == dataset]

            for algo in ds_summary["algorithm"].unique():
                print(f"\n  Algorithm: {algo}")
                algo_ds = ds_summary[ds_summary["algorithm"] == algo]
                # Separate NDCG and Precision
                for metric_name in ["NDCG", "Precision"]:
                    metric_rows = algo_ds[algo_ds["name"] == metric_name]
                    if len(metric_rows) > 0:
                        print(f"    {metric_name}:")
                        for _, row in metric_rows.sort_values("k").iterrows():
                            print(f"      k={int(row['k']):2d}:  mean={row['mean']:.6f},  std={row['std']:.6f}")

        # Save full results to CSV
        full_results.to_csv(os.path.join(working_dir, "full_results.csv"), index=False)
        summary.to_csv(os.path.join(working_dir, "summary_statistics.csv"), index=False)
        print(f"\nFull results saved to: {os.path.join(working_dir, 'full_results.csv')}")
        print(f"Summary statistics saved to: {os.path.join(working_dir, 'summary_statistics.csv')}")
    else:
        print("ERROR: No results collected!")


if __name__ == "__main__":
    main()