import os
import math
import statistics as stats
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.util import set_random_state
from omnirec.util.run import run_omnirec


def summarize_seed_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "algorithm", "metric", "k"]
    for keys, g in df.groupby(group_cols, dropna=False):
        dataset, algo, metric, k = keys if isinstance(keys, tuple) else (keys,)
        vals = g["value"].astype(float).tolist()
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algo,
                "metric": metric,
                "k": int(k),
                "mean": mean,
                "std": std,
                "cv": cv,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric", "k"])


def normalize_results_table(res):
    if isinstance(res, pd.DataFrame):
        df = res.copy()
    elif hasattr(res, "to_df"):
        df = res.to_df().copy()
    else:
        df = pd.DataFrame(res)

    rename_map = {}
    for cand in ["metric_name", "metric", "metric_id"]:
        if cand in df.columns:
            rename_map[cand] = "metric"
            break
    for cand in ["algo", "model", "algorithm_name"]:
        if cand in df.columns:
            rename_map[cand] = "algorithm"
            break
    for cand in ["cutoff", "N", "k", "topk"]:
        if cand in df.columns:
            rename_map[cand] = "k"
            break
    for cand in ["score", "value", "metric_value"]:
        if cand in df.columns:
            rename_map[cand] = "value"
            break

    df = df.rename(columns=rename_map)
    return df


def load_dataset_exact(ds_enum):
    return RecSysDataSet.use_dataloader(ds_enum)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]

    # Use documented dataset names available in OmniRec's built-in registry.
    # If your local OmniRec install exposes a slightly different Amazon/LastFM enum name,
    # replace only the enum reference here, not the experimental logic.
    dataset_specs = [
        ("MovieLens100K", DataSet.MovieLens100K, True),
        ("AmazonVideoGames", DataSet.Amazon2023VideoGames, True),
        ("HetrecLastFM", DataSet.HetrecLastFM, False),
    ]

    all_results = []
    metadata = []

    for dataset_name, ds_enum, convert_implicit in dataset_specs:
        print(f"\n=== Loading {dataset_name} ===")
        raw_dataset = load_dataset_exact(ds_enum)
        metadata.append(
            {
                "dataset": dataset_name,
                "raw_interactions": raw_dataset.num_interactions(),
                "min_rating": raw_dataset.min_rating() if hasattr(raw_dataset, "min_rating") else None,
                "max_rating": raw_dataset.max_rating() if hasattr(raw_dataset, "max_rating") else None,
                "loader_enum": str(ds_enum),
            }
        )

        for seed in seeds:
            print(f"\n--- {dataset_name} | seed={seed} ---")
            set_random_state(seed)

            # Reload from raw each seed to avoid any in-place state carryover.
            dataset = load_dataset_exact(ds_enum)

            steps = []
            if convert_implicit:
                steps.append(MakeImplicit(3))
            steps.append(CorePruning(5))
            steps.append(UserHoldout(validation_size=0.0, test_size=0.2))
            pipe = Pipe(*steps)
            split_ds = pipe.process(dataset)

            plan = ExperimentPlan(plan_name=f"{dataset_name}_seed_{seed}")
            plan.add_algorithm(LensKit.ImplicitMFScorer)
            plan.add_algorithm(LensKit.ItemKNNScorer)
            plan.add_algorithm(LensKit.PopScorer)

            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )

            result = run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            res_df = normalize_results_table(result)
            res_df["dataset"] = dataset_name
            res_df["seed"] = seed
            all_results.append(res_df)
            print(res_df)

    results = pd.concat(all_results, ignore_index=True)
    results_path = os.path.join(working_dir, "seed_sensitivity_results.csv")
    results.to_csv(results_path, index=False)

    summary = summarize_seed_sensitivity(results)
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== Dataset metadata ===")
    print(pd.DataFrame(metadata).to_string(index=False))

    print("\n=== Seed sensitivity summary ===")
    print(summary.to_string(index=False))

    print("\nBrief statistical analysis:")
    for keys, g in summary.groupby(["dataset", "algorithm"]):
        dataset, algo = keys if isinstance(keys, tuple) else (keys,)
        cv_series = g["cv"].replace([np.inf, -np.inf], np.nan)
        max_cv = float(cv_series.max()) if not cv_series.isna().all() else float("nan")
        avg_std = float(g["std"].mean())
        print(
            f"- {dataset} / {algo}: max seed CV across metrics@k = {max_cv:.4f}, mean std = {avg_std:.4f}"
        )

    print(f"\nSaved results to: {results_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
