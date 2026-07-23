import os
import sys
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# OmniRec imports
from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

# Create working directory
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# Define random seeds
SEEDS = [42, 123, 456, 789, 1024]

# Define datasets
DATASET_CONFIGS = {
    "MovieLens100K": {
        "enum": DataSet.MovieLens100K,
        "apply_implicit": True,
    },
    "Amazon2014VideoGames": {
        "enum": DataSet.Amazon2014VideoGames,
        "apply_implicit": True,
    },
    "HetrecLastFM": {
        "enum": DataSet.HetrecLastFM,
        "apply_implicit": False,
    },
}

# Algorithm configurations
ALGORITHMS = {
    LensKit.PopScorer: {},
    LensKit.ItemKNNScorer: {"max_nbrs": 30, "min_nbrs": 5},
    LensKit.ImplicitMFScorer: {"features": 50, "iterations": 100},
}

# Store all results
all_results = []

print("=" * 80)
print("EXPERIMENT: Effect of Data Split Seeds on Recommendation Accuracy")
print("=" * 80)

for dataset_name, ds_config in DATASET_CONFIGS.items():
    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'=' * 60}")

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  --- Seed {seed} ({seed_idx + 1}/{len(SEEDS)}) ---")

        # Set random state for reproducibility
        set_random_state(seed)

        # Step 1: Load dataset
        print(f"    Loading {dataset_name}...")
        dataset_enum: DataSet = cast(DataSet, ds_config["enum"])
        dataset = RecSysDataSet.use_dataloader(dataset_enum)

        # Step 2: Build preprocessing pipeline
        preprocess_steps = []

        # Convert to implicit if needed (ratings > 3, i.e., threshold=4)
        if ds_config["apply_implicit"]:
            preprocess_steps.append(MakeImplicit(4))

        # Apply 5-core filtering
        preprocess_steps.append(CorePruning(5))

        # Apply user-based 80/20 holdout split
        # IMPORTANT FIX: validation_size must be > 0.0 because UserHoldout._process()
        # internally computes test_size = valid_size / (1 - test_size) for the second
        # train_test_split call. If validation_size = 0.0, the inner split gets test_size=0.0
        # which sklearn rejects. Using 0.01 gives ~79% train, ~1% val, 20% test.
        preprocess_steps.append(UserHoldout(validation_size=0.01, test_size=0.2))

        # Execute preprocessing pipeline
        pipeline = Pipe(*preprocess_steps)
        print(f"    Applying preprocessing (5-core, implicit conversion if needed, split)...")
        try:
            processed_dataset = pipeline.process(dataset)
        except Exception as e:
            print(f"    ERROR during preprocessing: {e}")
            continue

        print(f"    Dataset stats - interactions: {processed_dataset.num_interactions()}")

        # Step 3: Create experiment plan
        plan = ExperimentPlan(f"{dataset_name}_seed{seed}")
        for algo, algo_config in ALGORITHMS.items():
            # Only pass non-empty configs
            if algo_config:
                plan.add_algorithm(algo, algo_config)
            else:
                plan.add_algorithm(algo)

        # Step 4: Create evaluator with NDCG and Precision at k=1,5,10
        evaluator = Evaluator(
            NDCG([1, 5, 10]),
            Precision([1, 5, 10]),
        )

        # Step 5: Run experiments
        print(f"    Running experiments with {len(ALGORITHMS)} algorithms...")
        try:
            run_omnirec(
                processed_dataset,
                plan,
                evaluator,
            )
        except Exception as e:
            print(f"    ERROR running experiment: {e}")
            continue

        # Step 6: Collect results
        results_dict = evaluator.get_results()
        for dataset_key, df in results_dict.items():
            # Add metadata columns
            df = df.copy()
            df["dataset"] = dataset_name
            df["seed"] = seed
            all_results.append(df)
            print(f"    Collected {len(df)} result rows for {dataset_key}")

# Combine all results
if all_results:
    full_results = pd.concat(all_results, ignore_index=True)
    print(f"\n{'=' * 80}")
    print(f"Total results collected: {len(full_results)} rows")
    print(full_results.head(10))
    print("...")

    # Save raw results
    results_path = os.path.join(working_dir, "all_results.json")
    # Convert to records for JSON serialization
    serializable = full_results.to_dict(orient="records")
    # Convert numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            elif isinstance(o, (np.floating,)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    
    with open(results_path, "w") as f:
        json.dump(serializable, f, cls=NpEncoder, indent=2)
    print(f"Raw results saved to {results_path}")

    # ============================================================
    # Statistical Analysis
    # ============================================================
    print(f"\n{'=' * 80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'=' * 80}")

    # Parse algorithm name (extract base name before config hash)
    full_results["algo_base"] = full_results["algorithm"].str.split("-").str[0]

    # Group by dataset, algorithm base, metric name, and k
    grouped = full_results.groupby(["dataset", "algo_base", "name", "k"])

    # Compute statistics
    stats_rows = []
    for (ds, algo, metric, k), group in grouped:
        values = group["value"].dropna().values
        if len(values) > 0:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1))
            cv_val = std_val / mean_val if mean_val > 0 else 0.0
            stats_rows.append({
                "Dataset": ds,
                "Algorithm": algo,
                "Metric": f"{metric}@{k}" if pd.notna(k) else metric,
                "Mean": round(mean_val, 6),
                "Std": round(std_val, 6),
                "CV (Std/Mean)": round(cv_val, 6),
            })

    stats_df = pd.DataFrame(stats_rows)

    # Print statistical summary
    print("\nPer (Dataset, Algorithm, Metric) Statistics Across 5 Seeds:\n")
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_columns', 10)
    print(stats_df.to_string(index=False))

    # Save statistics
    stats_path = os.path.join(working_dir, "statistical_analysis.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistical analysis saved to {stats_path}")

    # ============================================================
    # Summary: Coefficient of Variation highlights
    # ============================================================
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS - Coefficient of Variation (Seed Sensitivity)")
    print(f"{'=' * 80}")

    # Pivot table: CV across algorithms and datasets for each metric
    for metric_name in ["NDCG", "Precision"]:
        for k_val in [1, 5, 10]:
            mask = (stats_df["Metric"] == f"{metric_name}@{k_val}")
            subset = stats_df[mask]
            if len(subset) > 0:
                print(f"\n  {metric_name}@{k_val} - CV across seeds:")
                for _, row in subset.iterrows():
                    print(f"    {row['Dataset']:25s} | {row['Algorithm']:25s} | CV = {row['CV (Std/Mean)']:.6f}")

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 80}")
else:
    print("No results were collected. Something went wrong.")
    sys.exit(1)
