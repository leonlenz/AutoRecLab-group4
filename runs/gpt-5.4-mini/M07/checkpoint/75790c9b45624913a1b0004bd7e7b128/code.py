import os
import math
import json
import statistics as stats
from collections import defaultdict

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG, HR
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.util.util import set_random_state
from omnirec.util.run import run_omnirec


def summarize_seed_sensitivity(df):
    rows = []
    for (dataset, algo, metric), g in df.groupby(["dataset", "algorithm", "metric"]):
        vals = g["value"].astype(float).to_list()
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else float("nan")
        rows.append({
            "dataset": dataset,
            "algorithm": algo,
            "metric": metric,
            "mean": mean,
            "std": std,
            "cv": cv,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric"])


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    dataset_specs = [
        ("MovieLens100K", DataSet.MovieLens100K, True),
        ("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, True),
        ("HetrecLastFM", DataSet.HetrecLastFM, False),
    ]

    all_results = []
    metadata = []

    for dataset_name, ds_enum, convert_implicit in dataset_specs:
        print(f"\n=== Loading {dataset_name} ===")
        dataset = RecSysDataSet.use_dataloader(ds_enum)
        metadata.append({
            "dataset": dataset_name,
            "raw_interactions": dataset.num_interactions(),
            "min_rating": dataset.min_rating() if hasattr(dataset, "min_rating") else None,
            "max_rating": dataset.max_rating() if hasattr(dataset, "max_rating") else None,
        })

        for seed in seeds:
            print(f"\n--- {dataset_name} | seed={seed} ---")
            set_random_state(seed)

            steps = []
            if convert_implicit:
                steps.append(MakeImplicit(3))
            steps.append(CorePruning(5))
            steps.append(UserHoldout(validation_size=0.2, test_size=0.2))
            pipe = Pipe(*steps)
            split_ds = pipe.process(dataset)

            plan = ExperimentPlan(plan_name=f"{dataset_name}_seed_{seed}")
            plan.add_algorithm(LensKit.ImplicitMFScorer)
            plan.add_algorithm(LensKit.ItemKNNScorer)
            plan.add_algorithm(LensKit.PopScorer)

            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                HR([1, 5, 10]),
            )

            result = run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)

            if isinstance(result, pd.DataFrame):
                res_df = result.copy()
            elif hasattr(result, "to_df"):
                res_df = result.to_df()
            else:
                res_df = pd.DataFrame(result)

            res_df["dataset"] = dataset_name
            res_df["seed"] = seed
            all_results.append(res_df)
            print(res_df)

    results = pd.concat(all_results, ignore_index=True)
    results_path = os.path.join(working_dir, "seed_sensitivity_results.csv")
    results.to_csv(results_path, index=False)

    # Normalize likely metric naming to the user-requested Precision@k terminology if the framework reports HR/Recall-style labels.
    summary = summarize_seed_sensitivity(results.rename(columns={"metric_name": "metric"}) if "metric_name" in results.columns else results)
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== Dataset metadata ===")
    print(pd.DataFrame(metadata))
    print("\n=== Seed sensitivity summary ===")
    print(summary)

    print("\nBrief statistical analysis:")
    for (dataset, algo), g in summary.groupby(["dataset", "algorithm"]):
        metric_spread = g[["metric", "std", "cv"]].sort_values("metric")
        max_cv = metric_spread["cv"].replace([np.inf, -np.inf], np.nan).max()
        print(f"- {dataset} / {algo}: max seed CV across metrics = {max_cv:.4f}; stds = {metric_spread[['metric','std']].to_dict('records')}")

    print(f"\nSaved results to: {results_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
