import os
import json
import math
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_dataset(dataset_name):
    mapping = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }
    return RecSysDataSet.use_dataloader(mapping[dataset_name])


def preprocess_dataset(dataset_name, ds):
    ds = CorePruning(5).process(ds)
    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        ds = MakeImplicit(3).process(ds)
    return ds


def build_plan():
    plan = ExperimentPlan("Seed-Sensitivity-Study")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [7, 13, 21, 42, 84]
    datasets = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]
    algorithms = ["ImplicitMFScorer", "ItemKNNScorer", "PopScorer"]

    all_rows = []
    per_run_records = []

    for seed in seeds:
        set_random_state(seed)
        for dataset_name in datasets:
            ds = load_dataset(dataset_name)
            ds = preprocess_dataset(dataset_name, ds)
            ds = UserHoldout(validation_size=0.0, test_size=0.2).process(ds)

            plan = build_plan()
            evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

            print(f"Running seed={seed}, dataset={dataset_name} ...")
            results = run_omnirec(datasets=ds, plan=plan, evaluator=evaluator)

            if isinstance(results, pd.DataFrame):
                res_df = results.copy()
            else:
                res_df = pd.DataFrame(results)

            res_df["seed"] = seed
            res_df["dataset"] = dataset_name
            per_run_records.append(res_df)
            all_rows.append(res_df)
            print(res_df.to_string(index=False))

    results_df = pd.concat(all_rows, ignore_index=True)
    results_path = os.path.join(working_dir, "seed_sensitivity_results.csv")
    results_df.to_csv(results_path, index=False)

    summary_rows = []
    metric_cols = [c for c in results_df.columns if any(m in c.lower() for m in ["ndcg", "precision"])]
    group_cols = ["dataset", "algorithm"]

    for (dataset_name, algo), grp in results_df.groupby(group_cols):
        row = {"dataset": dataset_name, "algorithm": algo, "n_runs": len(grp)}
        for col in metric_cols:
            vals = grp[col].astype(float).to_numpy()
            row[col + "_mean"] = float(np.mean(vals))
            row[col + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[col + "_cv"] = float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 and np.mean(vals) != 0 else 0.0
            row[col + "_min"] = float(np.min(vals))
            row[col + "_max"] = float(np.max(vals))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nAggregate summary by dataset/algorithm:")
    print(summary_df.to_string(index=False))

    print("\nShort statistical analysis:")
    for _, row in summary_df.iterrows():
        metric = "ndcg@10_mean"
        if metric in row and not pd.isna(row[metric]):
            print(
                f"{row['dataset']} / {row['algorithm']}: "
                f"mean nDCG@10={row[metric]:.4f}, std={row['ndcg@10_std']:.4f}, cv={row['ndcg@10_cv']:.3f}"
            )

    meta = {
        "datasets": datasets,
        "seeds": seeds,
        "split": {"type": "UserHoldout", "validation_size": 0.0, "test_size": 0.2},
        "preprocessing": {
            "core_pruning": 5,
            "implicit_threshold": 3,
            "implicit_datasets": ["MovieLens100K", "Amazon2014VideoGames"],
        },
        "algorithms": ["LensKit.ImplicitMFScorer", "LensKit.ItemKNNScorer", "LensKit.PopScorer"],
        "metrics": ["NDCG@1", "NDCG@5", "NDCG@10", "Precision@1", "Precision@5", "Precision@10"],
        "results_path": results_path,
        "summary_path": summary_path,
    }
    with open(os.path.join(working_dir, "experiment_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
