import os
import json
import math
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


def run_single_seed(seed, dataset_name):
    set_random_state(seed)
    ds = load_dataset(dataset_name)
    ds = preprocess_dataset(dataset_name, ds)

    # Verified user-based holdout splitter.
    # The public docs show validation_size and test_size as proportions or counts.
    # We use a small positive validation split to satisfy the API and keep a user-based split.
    ds = UserHoldout(validation_size=0.1, test_size=0.2).process(ds)

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
    return res_df


def normalize_results(df):
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "algorithm":
            rename_map[col] = "algorithm"
        elif lc == "name":
            rename_map[col] = "metric"
        elif lc == "k":
            rename_map[col] = "k"
        elif lc == "value":
            rename_map[col] = "value"
    df = df.rename(columns=rename_map)
    return df


def short_statistical_analysis(results_df):
    rows = []
    if not {"dataset", "algorithm", "metric", "k", "value"}.issubset(results_df.columns):
        return pd.DataFrame(rows)

    for (dataset_name, algo, metric, k), grp in results_df.groupby(["dataset", "algorithm", "metric", "k"], dropna=False):
        vals = grp["value"].astype(float).to_numpy()
        if len(vals) < 2:
            continue
        rows.append({
            "dataset": dataset_name,
            "algorithm": algo,
            "metric": metric,
            "k": int(k),
            "n_runs": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "cv": float(np.std(vals, ddof=1) / np.mean(vals)) if np.mean(vals) != 0 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    return pd.DataFrame(rows)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [7, 13, 21, 42, 84]
    datasets = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]

    all_rows = []
    for seed in seeds:
        for dataset_name in datasets:
            try:
                res_df = run_single_seed(seed, dataset_name)
                res_df = normalize_results(res_df)
                print(res_df.to_string(index=False))
                all_rows.append(res_df)
            except Exception as e:
                print(f"Run failed for seed={seed}, dataset={dataset_name}: {e}")
                raise

    results_df = pd.concat(all_rows, ignore_index=True)
    results_path = os.path.join(working_dir, "seed_sensitivity_results.csv")
    results_df.to_csv(results_path, index=False)

    metric_rows = []
    for (dataset_name, algo, metric, k), grp in results_df.groupby(["dataset", "algorithm", "metric", "k"], dropna=False):
        vals = grp["value"].astype(float).to_numpy()
        metric_rows.append({
            "dataset": dataset_name,
            "algorithm": algo,
            "metric": metric,
            "k": int(k),
            "n_runs": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "cv": float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 and np.mean(vals) != 0 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    summary_df = pd.DataFrame(metric_rows)
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nAggregate summary by dataset/algorithm/metric/k:")
    print(summary_df.to_string(index=False))

    analysis_df = short_statistical_analysis(results_df)
    print("\nShort statistical analysis:")
    print(analysis_df.to_string(index=False))

    meta = {
        "datasets": datasets,
        "seeds": seeds,
        "split": {"type": "UserHoldout", "validation_size": 0.1, "test_size": 0.2},
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
