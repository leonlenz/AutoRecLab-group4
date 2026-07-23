#!/usr/bin/env python3
"""
Experiment: Quantifying the impact of data split random seeds on recommender system accuracy.

Uses OmniRec (which wraps LensKit internally) to test 3 algorithms (ALS/ImplicitMF, ItemKNN, Pop)
on 3 datasets (MovieLens100K, Amazon2014VideoGames, HetrecLastFM) with 5 different random seeds
for user-based 80/20 holdout. Evaluates NDCG@k and Precision@k for k=1,5,10.
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# OmniRec imports - use exclusively as per requirements
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import RawData
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


# ---------------------------------------------------------------------------
# Helpers for dataset statistics (RecSysDataSet[RawData] has no public
# num_users / num_items methods, so we compute them from the underlying df)
# ---------------------------------------------------------------------------
def _num_users(ds: RecSysDataSet[RawData]) -> int:
    """Return the number of unique users in a RawData-backed dataset."""
    return ds._data.df["user"].nunique()  # type: ignore[attr-defined]


def _num_items(ds: RecSysDataSet[RawData]) -> int:
    """Return the number of unique items in a RawData-backed dataset."""
    return ds._data.df["item"].nunique()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.2          # 80/20 user-based holdout (20% test)
CORE_VALUE = 5           # 5-core filtering
IMPLICIT_THRESHOLD = 3   # Ratings >= 3 become implicit (MakeImplicit(3))

BASE_WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(BASE_WORKING_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
print("=" * 70)
print("EXPERIMENT: Impact of data split random seeds on recommender accuracy")
print("=" * 70)
print(f"\nDatasets: MovieLens100K, Amazon2014VideoGames, HetrecLastFM")
print(f"Algorithms: ALS (ImplicitMF), ItemKNN, Pop")
print(f"Seeds: {SEEDS}")
print(f"Split: User-based 80/20 holdout")
print(f"Preprocessing: MakeImplicit(>=3), 5-core filtering")
print(f"Metrics: NDCG@k and Precision@k for k=1,5,10")
print(f"Working directory: {BASE_WORKING_DIR}")

# This will hold all result records across all seeds
all_results = []  # list of dicts with dataset, algorithm, seed, metric, k, value

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'=' * 60}")
    print(f"SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'=' * 60}")

    # Set the global random state for reproducibility
    set_random_state(seed)
    current_state = get_random_state()
    print(f"  Random state set to: {current_state}")

    # Create seed-specific working directory for checkpoint isolation
    seed_working_dir = os.path.join(BASE_WORKING_DIR, f"seed_{seed}")
    os.makedirs(seed_working_dir, exist_ok=True)
    # Change to seed directory so checkpoints don't collide
    original_cwd = os.getcwd()
    os.chdir(seed_working_dir)

    try:
        # ------------------------------------------------------------------
        # Step 1 & 2: Load and preprocess datasets
        # ------------------------------------------------------------------
        print("\n  --- Loading & Preprocessing Datasets ---")

        datasets = []

        # --- MovieLens100K ---
        print("\n  Processing MovieLens100K...")
        ml100k_ds = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        print(f"    Loaded: {ml100k_ds.num_interactions()} interactions, "
              f"users={_num_users(ml100k_ds)}, items={_num_items(ml100k_ds)}")
        # Preprocess: MakeImplicit(3) -> CorePruning(5) -> UserHoldout(0.0, 0.2)
        ml100k_pipeline = Pipe(
            MakeImplicit(IMPLICIT_THRESHOLD),   # ratings >= 3 become implicit
            CorePruning(CORE_VALUE),            # 5-core filtering
            UserHoldout(0.0, TEST_FRAC),        # 80/20 user-based split
        )
        ml100k_processed = ml100k_pipeline.process(ml100k_ds)
        datasets.append(("MovieLens100K", ml100k_processed))

        # --- Amazon2014VideoGames ---
        print("\n  Processing Amazon2014VideoGames...")
        amazon_ds = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
        print(f"    Loaded: {amazon_ds.num_interactions()} interactions, "
              f"users={_num_users(amazon_ds)}, items={_num_items(amazon_ds)}")
        amazon_pipeline = Pipe(
            MakeImplicit(IMPLICIT_THRESHOLD),   # ratings >= 3 become implicit
            CorePruning(CORE_VALUE),            # 5-core filtering
            UserHoldout(0.0, TEST_FRAC),        # 80/20 user-based split
        )
        amazon_processed = amazon_pipeline.process(amazon_ds)
        datasets.append(("Amazon2014VideoGames", amazon_processed))

        # --- HetrecLastFM ---
        print("\n  Processing HetrecLastFM...")
        lastfm_ds = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
        print(f"    Loaded: {lastfm_ds.num_interactions()} interactions, "
              f"users={_num_users(lastfm_ds)}, items={_num_items(lastfm_ds)}")
        lastfm_pipeline = Pipe(
            CorePruning(CORE_VALUE),            # 5-core filtering (already implicit)
            UserHoldout(0.0, TEST_FRAC),        # 80/20 user-based split
        )
        lastfm_processed = lastfm_pipeline.process(lastfm_ds)
        datasets.append(("HetrecLastFM", lastfm_processed))

        # ------------------------------------------------------------------
        # Step 3: Create experiment plan and evaluator
        # ------------------------------------------------------------------
        print("\n  --- Creating Experiment Plan ---")

        # Create experiment plan with all three algorithms using default hyperparameters
        plan = ExperimentPlan("SeedVariationAnalysis")

        # ALS - ImplicitMFScorer with default params (embedding_size=64, epochs=10, regularization=0.1)
        plan.add_algorithm(LensKit.ImplicitMFScorer, {})

        # ItemKNN with default params (max_nbrs=20, feedback='explicit')
        plan.add_algorithm(LensKit.ItemKNNScorer, {})

        # Pop with default params
        plan.add_algorithm(LensKit.PopScorer, {})

        # Create evaluator with NDCG and Precision at k=1,5,10
        evaluator = Evaluator(
            NDCG([1, 5, 10]),
            Precision([1, 5, 10]),
        )

        # ------------------------------------------------------------------
        # Step 4: Run experiments
        # ------------------------------------------------------------------
        print("\n  --- Running Experiments ---")

        # We need to run on each dataset separately to collect per-dataset results
        for ds_name, ds in datasets:
            print(f"\n    Dataset: {ds_name}")
            try:
                run_omnirec(datasets=ds, plan=plan, evaluator=evaluator)

                # Collect results from evaluator
                results_dict = evaluator.get_results()
                for dataset_id, df in results_dict.items():
                    for _, row in df.iterrows():
                        metric_name = row['name']
                        k_val = row['k']
                        metric_val = row['value']
                        algo_full = row['algorithm']

                        # Extract algorithm name from full identifier (e.g., "LensKit.ImplicitMFScorer-xxxx")
                        algo_short = algo_full.split('-')[0]
                        if '.' in algo_short:
                            algo_short = algo_short.split('.')[1]

                        all_results.append({
                            'dataset': ds_name,
                            'algorithm': algo_short,
                            'seed': seed,
                            'metric': metric_name,
                            'k': k_val,
                            'value': metric_val,
                        })

                        print(f"      {algo_short:20s} | {metric_name:10s}@{k_val:2d} = {metric_val:.6f}")

            except Exception as e:
                print(f"      ERROR processing {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    except Exception as e:
        print(f"  ERROR in seed {seed}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

# ---------------------------------------------------------------------------
# Step 5: Statistical analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Statistical Analysis - Mean and Std across seeds")
print("=" * 70)

if not all_results:
    print("No results collected. Exiting.")
    sys.exit(1)

combined_results = pd.DataFrame(all_results)
print(f"\nTotal result rows collected: {len(combined_results)}")

# Group by dataset, algorithm, metric, k and compute mean & std
agg_results = (
    combined_results
    .groupby(["dataset", "algorithm", "metric", "k"])["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

agg_results.columns = ["Dataset", "Algorithm", "Metric", "k", "Mean", "Std", "Count"]
agg_results = agg_results.sort_values(["Dataset", "Algorithm", "Metric", "k"])

# Print aggregated results
print("\n" + "=" * 70)
print("AGGREGATED RESULTS (Mean ± Std across 5 seeds)")
print("=" * 70)

for dataset_name in agg_results["Dataset"].unique():
    print(f"\n{'─' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'─' * 60}")

    ds_df = agg_results[agg_results["Dataset"] == dataset_name]
    for _, row in ds_df.iterrows():
        print(f"  {row['Algorithm']:15s} | {row['Metric']:10s}@{int(row['k']):2d} | "
              f"Mean={row['Mean']:.6f} ± Std={row['Std']:.6f} (n={int(row['Count'])})")

# Coefficient of Variation
print("\n" + "=" * 70)
print("VARIATION SUMMARY (Coefficient of Variation = Std/Mean)")
print("=" * 70)

agg_results["CV"] = np.where(agg_results["Mean"] > 0,
                              agg_results["Std"] / agg_results["Mean"],
                              np.nan)

for dataset_name in agg_results["Dataset"].unique():
    print(f"\nDataset: {dataset_name}")
    ds_df = agg_results[agg_results["Dataset"] == dataset_name]
    for _, row in ds_df.iterrows():
        cv_val = row.get("CV", np.nan)
        cv_str = f"{cv_val:.4f}" if not (cv_val is None or (isinstance(cv_val, float) and np.isnan(cv_val))) else "N/A"
        print(f"  {row['Algorithm']:15s} | {row['Metric']:10s}@{int(row['k']):2d} | "
              f"CV={cv_str}")

# Save results
results_csv_path = os.path.join(BASE_WORKING_DIR, "seed_variation_results.csv")
combined_results.to_csv(results_csv_path, index=False)
print(f"\nDetailed results saved to: {results_csv_path}")

agg_csv_path = os.path.join(BASE_WORKING_DIR, "seed_variation_aggregated.csv")
agg_results.to_csv(agg_csv_path, index=False)
print(f"Aggregated results saved to: {agg_csv_path}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

if __name__ == "__main__":
    pass
