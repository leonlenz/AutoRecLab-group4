import os
import sys
import re
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

# ============================================================
# PATCH the installed lenskit_runner.py to fix the KeyError: 'rank' bug
# ============================================================
# The bug: In lenskit_runner.py predict(), it passes self.test (DataFrame with
# item_id column) to lenskit.batch.recommend() instead of self.users_to_predict
# (just user IDs). This causes a missing 'rank' column in the output.
# Fix: Change recommend(self.model, self.test) -> recommend(self.model, self.users_to_predict)
def patch_lenskit_runner():
    """Patch the installed lenskit_runner.py to fix the KeyError: 'rank' bug."""
    try:
        import omnirec_runner
        runner_dir = os.path.dirname(os.path.abspath(omnirec_runner.__file__))
        runner_path = os.path.join(runner_dir, "lenskit_runner.py")
        
        if not os.path.exists(runner_path):
            print(f"WARNING: lenskit_runner.py not found at {runner_path}")
            return
        
        with open(runner_path, "r") as f:
            content = f.read()
        
        # The bug: recommend(self.model, self.test) should be recommend(self.model, self.users_to_predict)
        # Also fix the implicit path (no "rating" column in train)
        # We need to find the exact line and fix it
        old_code = "predictions = recommend(self.model, self.test)"
        new_code = "predictions = recommend(self.model, self.users_to_predict)"
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            with open(runner_path, "w") as f:
                f.write(content)
            print(f"PATCHED lenskit_runner.py: {runner_path}")
            print(f"  Changed: '{old_code}' -> '{new_code}'")
        else:
            # Check if already patched
            if new_code in content:
                print(f"lenskit_runner.py already patched at {runner_path}")
            else:
                print(f"WARNING: Could not find '{old_code}' in lenskit_runner.py")
                print(f"  File contains 'recommend' calls:")
                for i, line in enumerate(content.split('\n')):
                    if 'recommend' in line:
                        print(f"  Line {i+1}: {line.strip()}")
    except ImportError:
        print("WARNING: Could not import omnirec_runner to patch lenskit_runner.py")

# Apply the patch
patch_lenskit_runner()

# ============================================================
# MAIN EXPERIMENT
# ============================================================

# Create working directory
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# Define seeds and parameters
SEEDS = [42, 123, 456, 789, 1111]
METRIC_KS = [1, 5, 10]

# Define datasets with their preprocessing specifications
# MovieLens100K and Amazon2014VideoGames: explicit -> implicit via MakeImplicit(3)
# HetrecLastFM: already implicit, skip MakeImplicit
DATASET_SPECS = [
    {
        "name": "MovieLens100K",
        "dataset_enum": DataSet.MovieLens100K,
        "use_make_implicit": True,
    },
    {
        "name": "Amazon2014VideoGames",
        "dataset_enum": DataSet.Amazon2014VideoGames,
        "use_make_implicit": True,
    },
    {
        "name": "HetrecLastFM",
        "dataset_enum": DataSet.HetrecLastFM,
        "use_make_implicit": False,
    },
]

# Define algorithms with their configs (standard/default hyperparameters)
ALGORITHMS = [
    (LensKit.PopScorer, {"feedback": "implicit"}),
    (LensKit.ItemKNNScorer, {"feedback": "implicit"}),
    (LensKit.ImplicitMFScorer, {"feedback": "implicit"}),
]


def preprocess_dataset(spec, seed):
    """Load and preprocess a single dataset for a given random seed."""
    set_random_state(seed)

    # Load the dataset
    dataset = RecSysDataSet.use_dataloader(spec["dataset_enum"])

    # Build preprocessing pipeline
    steps = []

    # Only apply MakeImplicit to datasets specified as explicit
    if spec["use_make_implicit"]:
        # Convert ratings > 3 to implicit feedback
        steps.append(MakeImplicit(3))

    # Apply 5-core filtering
    steps.append(CorePruning(5))

    # Apply user-based holdout split with 80/20 train/test split.
    # UserHoldout(validation_size=0.0, test_size=0.2) creates train=80%, valid=0%, test=20%
    steps.append(UserHoldout(validation_size=0.0, test_size=0.2))

    # Execute the pipeline
    pipeline = Pipe(*steps)
    processed_dataset = pipeline.process(dataset)

    return processed_dataset


def main():
    # Store all results across seeds
    all_results = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx + 1}/{len(SEEDS)}: seed = {seed}")
        print(f"{'='*80}")

        # Set the global random state for this seed
        set_random_state(seed)

        # Preprocess all three datasets with this seed
        processed_datasets = []
        for spec in DATASET_SPECS:
            print(f"  Preprocessing {spec['name']} with seed {seed}...")
            ds = preprocess_dataset(spec, seed)
            processed_datasets.append(ds)
            print(f"    Done.")

        # Create experiment plan with all three algorithms
        plan = ExperimentPlan(f"Seed-{seed}-Experiment")
        for algo_enum, algo_config in ALGORITHMS:
            plan.add_algorithm(algo_enum, algo_config)

        # Create evaluator with NDCG and Precision at k=[1, 5, 10]
        evaluator = Evaluator(
            NDCG(METRIC_KS),
            Precision(METRIC_KS),
        )

        # Run all experiments: 3 algorithms x 3 datasets
        print(f"  Running experiments for seed {seed}...")
        run_omnirec(processed_datasets, plan, evaluator)
        print(f"  Experiments completed for seed {seed}.")

        # Collect results from evaluator
        results_dict = evaluator.get_results()
        for dataset_key, result_df in results_dict.items():
            result_df = result_df.copy()
            result_df["seed"] = seed
            result_df["dataset_key"] = dataset_key
            all_results.append(result_df)

    # Combine all results into a single DataFrame
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
    else:
        print("ERROR: No results collected!")
        return

    # Print raw results sorted for readability
    print("\n\n")
    print("=" * 80)
    print("RAW RESULTS PER (DATASET, ALGORITHM, SEED)")
    print("=" * 80)

    display_cols = ["dataset_key", "algorithm", "seed", "name", "k", "value"]
    if all(col in combined_results.columns for col in display_cols):
        sorted_results = combined_results[display_cols].sort_values(
            by=["dataset_key", "algorithm", "seed", "name", "k"]
        )
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', 10)
        print(sorted_results.to_string(index=False))
    else:
        print("Columns found:", combined_results.columns.tolist())
        print(combined_results.head(20))

    # Statistical analysis: compute mean and std across seeds
    print("\n\n")
    print("=" * 80)
    print("STATISTICAL ANALYSIS: Mean and Std across 5 seeds")
    print("=" * 80)

    # Extract algorithm base name (without hash suffix) for grouping
    # Algorithm column format: "LensKit.PopScorer-abc123" or similar
    combined_results["algo_base"] = combined_results["algorithm"].str.replace(
        r'-[a-f0-9]+$', '', regex=True
    )

    # Group by dataset_key, algo_base, name, k and compute mean/std
    stats = (
        combined_results.groupby(["dataset_key", "algo_base", "name", "k"])["value"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["dataset_key", "algo_base", "name", "k"])
    )

    stats["mean_str"] = stats["mean"].map("{:.6f}".format)
    stats["std_str"] = stats["std"].map("{:.6f}".format)

    # Print the stats with the formatted strings
    print(stats[["dataset_key", "algo_base", "name", "k", "mean_str", "std_str"]].to_string(index=False))

    # Print per-dataset analysis
    print("\n\n")
    print("=" * 80)
    print("PER-DATASET ANALYSIS")
    print("=" * 80)

    for dataset_key in stats["dataset_key"].unique():
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset_key}")
        print(f"{'─'*70}")

        ds_stats = stats[stats["dataset_key"] == dataset_key]

        for algo in ds_stats["algo_base"].unique():
            print(f"\n  Algorithm: {algo}")
            algo_stats = ds_stats[ds_stats["algo_base"] == algo]
            for _, row in algo_stats.iterrows():
                print(f"    {row['name']}@{row['k']}: mean={row['mean_str']}, std={row['std_str']}")

    # Summary: Impact of random seed variation
    print("\n\n")
    print("=" * 80)
    print("SUMMARY: Impact of Random Seed Variation on Accuracy")
    print("=" * 80)

    # Compute overall std across all metric values for each algorithm-dataset combination
    combined_results["value"] = pd.to_numeric(combined_results["value"], errors="coerce")

    avg_std = (
        combined_results.groupby(["dataset_key", "algo_base"])["value"]
        .agg(lambda x: np.std(x))
        .reset_index()
    )
    avg_std["std_str"] = avg_std["value"].map("{:.6f}".format)

    print("\nOverall std (across seeds, metrics, and k-values) per (dataset, algorithm):")
    for _, row in avg_std.iterrows():
        print(f"  {row['dataset_key']} | {row['algo_base']}: std={row['std_str']}")

    print("\n\nExperiment complete! Results demonstrate how data split random seeds")
    print("affect recommender system accuracy across different algorithms and datasets.")


if __name__ == "__main__":
    main()
