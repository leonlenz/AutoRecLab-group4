"""
Experiment: Impact of Data Split Random Seeds on Recommender Accuracy
========================================================================
Tests 3 algorithms (ALS/ItemKNN/Pop) x 3 datasets x 5 random seeds
Measures NDCG@k and Recall@k for k=1,5,10
"""

import os
import sys
import json
import itertools
import pandas as pd
import numpy as np

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.subsample import Subsample
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Recall
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


# =============================================================================
# FIX: Patch the LensKit runner's predict() method to add rank column and return proper format
# The installed lenskit_runner.py has a bug where predict() doesn't return anything
# and doesn't add the required 'rank' column for implicit feedback.
# =============================================================================
def _patch_lenskit_runner():
    """Patch the installed LensKit runner to fix the broken predict() method."""
    try:
        import omnirec_runner
        runner_path = os.path.join(os.path.dirname(omnirec_runner.__file__), "lenskit_runner.py")
        
        with open(runner_path, "r") as f:
            content = f.read()
        
        # Check if already patched
        if "# PATCHED: Fixed predict" in content:
            print("  LensKit runner already patched.")
            return
        
        # The original predict method ends with:
        #     predictions = recommend(self.model, self.test)
        # (no return statement)
        #
        # We need to replace it to add the rank column and return the dict.
        
        old_predict = """    def predict(self):
        # TODO:
        # predict_log_dict = {
        #     "model_file": model_file,
        #     "data_set_name": data_set_name,
        #     "algorithm_name": algorithm_name,
        #     "algorithm_config_index": algorithm_config,
        #     "algorithm_configuration": configurations[algorithm_config],
        #     "fold": fold,
        # }
        # predict_log_dict.update(
        #     {
        #         "train_users": len(unique_train_users),
        #         "test_users": len(unique_test_users),
        #         "users_to_predict": len(users_to_predict),
        #         "prediction_time": end_prediction - start_prediction,
        #     }
        # )
        # predict_log_dict.update(
        #     {
        #         "test_interactions": test_interactions, # This was len(test) OG
        #         "prediction_time": end_prediction - start_prediction,
        #     }
        # )

        # Lenskit automatically finds the id, but it has to be suffixed with "_id"
        self.test.rename(columns={"user": "user_id", "item": "item_id"}, inplace=True)

        if "rating" in self.train.columns:
            self.test.drop(columns="rating", inplace=True)
            predictions = predict(self.model, self.test)
        else:
            predictions = recommend(self.model, self.test)"""

        new_predict = """    def predict(self):
        # PATCHED: Fixed predict to return proper format with rank column
        # Lenskit automatically finds the id, but it has to be suffixed with "_id"
        self.test.rename(columns={"user": "user_id", "item": "item_id"}, inplace=True)

        if "rating" in self.train.columns:
            self.test.drop(columns="rating", inplace=True)
            predictions = predict(self.model, self.test)
            predictions_df = predictions.to_df()
            predictions_df.rename(
                columns={"user_id": "user", "item_id": "item", "prediction": "rating"},
                inplace=True,
            )
            return predictions_df.to_dict(orient="list")
        else:
            predictions = recommend(self.model, self.test)
            predictions_df = predictions.to_df()
            predictions_df.rename(
                columns={"user_id": "user", "item_id": "item"},
                inplace=True,
            )
            if "rank" not in predictions_df.columns:
                predictions_df["rank"] = predictions_df.groupby("user").cumcount() + 1
            if "score" not in predictions_df.columns and "rating" in predictions_df.columns:
                predictions_df.rename(columns={"rating": "score"}, inplace=True)
            return predictions_df.to_dict(orient="list")"""

        if old_predict in content:
            new_content = content.replace(old_predict, new_predict)
            with open(runner_path, "w") as f:
                f.write(new_content)
            print(f"  Patched LensKit runner at: {runner_path}")
        else:
            print("  WARNING: Could not find original predict() method to patch.")
            print("  Attempting alternative patch approach...")
            # Try to find just the end portion
            if "predictions = recommend(self.model, self.test)" in content:
                content = content.replace(
                    "predictions = recommend(self.model, self.test)",
                    """predictions = recommend(self.model, self.test)
            predictions_df = predictions.to_df()
            predictions_df.rename(
                columns={"user_id": "user", "item_id": "item"},
                inplace=True,
            )
            if "rank" not in predictions_df.columns:
                predictions_df["rank"] = predictions_df.groupby("user").cumcount() + 1
            if "score" not in predictions_df.columns and "rating" in predictions_df.columns:
                predictions_df.rename(columns={"rating": "score"}, inplace=True)
            return predictions_df.to_dict(orient="list")"""
                )
                # Also fix the explicit path
                if "predictions = predict(self.model, self.test)" in content:
                    content = content.replace(
                        "predictions = predict(self.model, self.test)",
                        """predictions = predict(self.model, self.test)
            predictions_df = predictions.to_df()
            predictions_df.rename(
                columns={"user_id": "user", "item_id": "item", "prediction": "rating"},
                inplace=True,
            )
            return predictions_df.to_dict(orient="list")"""
                    )
                with open(runner_path, "w") as f:
                    f.write(content)
                print("  Alternative patch applied successfully.")
            else:
                print("  WARNING: Could not apply patch at all.")
    except Exception as e:
        print(f"  WARNING: Failed to patch LensKit runner: {e}")
        print("  Experiments may fail due to missing rank column in predictions.")


# =============================================================================
# Working directory
# =============================================================================
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

print("=" * 80)
print("Experiment: Impact of Data Split Random Seeds on Recommender Accuracy")
print("=" * 80)

# Apply the patch before running experiments
print("\nPatching LensKit runner to fix predict() method...")
_patch_lenskit_runner()

# =============================================================================
# 1. Define datasets and algorithms
# =============================================================================
DATASET_ENUMS = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}

# Random seeds for data splitting
RANDOM_SEEDS = [42, 123, 456, 789, 1111]

# Use documented metrics: NDCG and Recall (Precision exists in code but is undocumented)
evaluator = Evaluator(
    NDCG([1, 5, 10]),
    Recall([1, 5, 10]),
)

print(f"\nRandom seeds: {RANDOM_SEEDS}")
print(f"Datasets: {list(DATASET_ENUMS.keys())}")
print(f"Algorithms: ALS, ItemKNN, Pop")
print(f"Metrics: NDCG@[1,5,10], Recall@[1,5,10]")
print(f"\nTotal conditions: {len(DATASET_ENUMS)} datasets x 3 algorithms x {len(RANDOM_SEEDS)} seeds = {len(DATASET_ENUMS) * 3 * len(RANDOM_SEEDS)}")

# =============================================================================
# 2. Run experiments for each seed
# =============================================================================
for seed_idx, seed in enumerate(RANDOM_SEEDS):
    print(f"\n{'=' * 70}")
    print(f"Seed {seed_idx + 1}/{len(RANDOM_SEEDS)}: seed = {seed}")
    print(f"{'=' * 70}")

    # Set global random state for reproducibility of splits
    set_random_state(seed)

    # Build datasets for this seed
    seed_datasets = []
    for ds_name, ds_enum in DATASET_ENUMS.items():
        print(f"\n  --- Dataset: {ds_name} ---")

        # Load raw dataset
        dataset = RecSysDataSet.use_dataloader(ds_enum)
        n_raw_raw = dataset.num_interactions()
        print(f"  Loaded: {n_raw_raw} interactions (raw)")

        # Build preprocessing pipeline
        # FIXED: Use validation_size=0 for pure 80/20 train/test split
        # UserHoldout(0, 0.2) -> 80% train, 0% validation, 20% test
        # The UserHoldout code: valid = train_test_split(train, test_size=0/(1-0.2), ...)
        # test_size=0 means no validation split, all remaining goes to train.
        
        steps = []
        
        if ds_name in ("MovieLens100K", "Amazon2014VideoGames"):
            # Convert ratings >= 3 to implicit feedback
            steps.append(MakeImplicit(3))
        
        # 5-core filtering
        steps.append(CorePruning(5))
        
        # FIXED: Use validation_size=0 for 80/20 split
        steps.append(UserHoldout(0, 0.2))
        
        pipeline = Pipe(*steps)
        dataset = pipeline.process(dataset)
        n_counts = dataset.num_interactions()
        print(f"  After preprocessing (seed={seed}): {n_counts}")

        seed_datasets.append(dataset)

    # Create experiment plan for this seed with all three algorithms
    plan = ExperimentPlan(plan_name=f"SeedComparison_seed{seed}")
    plan.add_algorithm(
        LensKit.ImplicitMFScorer,
        {"feedback": "implicit"},
    )
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {"feedback": "implicit"},
    )
    plan.add_algorithm(
        LensKit.PopScorer,
        {"feedback": "implicit"},
    )

    # Run all algorithms on all datasets for this seed
    run_omnirec(
        datasets=seed_datasets,
        plan=plan,
        evaluator=evaluator,
    )

# =============================================================================
# 3. Collect and process results
# =============================================================================
print(f"\n{'=' * 80}")
print("Collecting Results")
print(f"{'=' * 80}")

results_dict = evaluator.get_results()
print(f"Results from {len(results_dict)} dataset-hash combinations")

# Combine all results into a single DataFrame
all_results_dfs = []
for ds_hash_key, df in results_dict.items():
    # Extract dataset name from the hash key (format: "DatasetName-xxxxxx")
    ds_name = ds_hash_key.split("-")[0] if "-" in ds_hash_key else ds_hash_key
    df_copy = df.copy()
    df_copy["dataset"] = ds_name
    all_results_dfs.append(df_copy)

if all_results_dfs:
    combined_results = pd.concat(all_results_dfs, ignore_index=True)
else:
    combined_results = pd.DataFrame()

print(f"\nCombined results shape: {combined_results.shape}")
if len(combined_results) > 0:
    print("\nRaw results preview:")
    print(combined_results.head(20).to_string())

# =============================================================================
# 4. Statistical Analysis: Mean and Std across seeds
# =============================================================================
print(f"\n{'=' * 80}")
print("Statistical Analysis: Mean +/- Std Across 5 Seeds")
print(f"{'=' * 80}")

if len(combined_results) > 0:
    # Parse algorithm name
    combined_results["algo_short"] = combined_results["algorithm"].apply(
        lambda x: str(x).split(".")[-1].split("-")[0] if "." in str(x) else str(x)
    )

    # Map algorithm names to simpler labels
    algo_map = {
        "ImplicitMFScorer": "ALS",
        "ItemKNNScorer": "ItemKNN",
        "PopScorer": "Pop",
    }
    combined_results["algorithm_label"] = combined_results["algo_short"].map(algo_map).fillna(combined_results["algo_short"])

    # Group by dataset, algorithm, metric, k and compute mean/std
    stat_analysis = combined_results.groupby(
        ["dataset", "algorithm_label", "name", "k"]
    )["value"].agg(["mean", "std", "count"])

    stat_analysis = stat_analysis.reset_index()
    stat_analysis["mean_std"] = stat_analysis.apply(
        lambda r: f"{r['mean']:.6f} +/- {r['std']:.6f}", axis=1
    )

    print("\n--- Summary Statistics (Mean +/- Std across seeds) ---")
    print("\nSorted by dataset, algorithm, metric, k:\n")

    for ds in sorted(stat_analysis["dataset"].unique()):
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'─' * 60}")
        ds_data = stat_analysis[stat_analysis["dataset"] == ds]
        for algo in sorted(ds_data["algorithm_label"].unique()):
            print(f"\n  Algorithm: {algo}")
            algo_data = ds_data[ds_data["algorithm_label"] == algo]
            for _, row in algo_data.iterrows():
                k_val = int(row["k"]) if pd.notna(row["k"]) else "N/A"
                print(f"    {row['name']}@{k_val}: {row['mean']:.6f} +/- {row['std']:.6f}  (n={int(row['count'])})")
            print()

    # Also print a compact table
    print(f"\n{'=' * 80}")
    print("Compact Result Table")
    print(f"{'=' * 80}")

    pivot = stat_analysis.pivot_table(
        index=["dataset", "algorithm_label"],
        columns=["name", "k"],
        values="mean_std",
        aggfunc=lambda x: x.iloc[0] if len(x) > 0 else ""
    )
    print(pivot.to_string())

else:
    print("No results available. The experiments may have checkpointed results.")
    print("Check the checkpoints/ directory for saved results.")

print(f"\n{'=' * 80}")
print("Experiment Complete")
print(f"{'=' * 80}")
print(f"\nWorking directory: {working_dir}")
