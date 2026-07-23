import os
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_and_preprocess(dataset_id, make_implicit_threshold=None):
    ds = RecSysDataSet.use_dataloader(dataset_id)
    steps = []
    if make_implicit_threshold is not None:
        steps.append(MakeImplicit(make_implicit_threshold))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def build_plan():
    plan = ExperimentPlan("seed_sensitivity_lenskit_baselines")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def build_split_dataset(dataset, seed):
    set_random_state(seed)
    # Verified API: UserHoldout(validation_size, test_size) expects positive sizes.
    # We use a small validation fraction so the split executes correctly while keeping
    # a user-aware holdout structure; downstream analysis focuses on the test split.
    splitter = UserHoldout(validation_size=0.1, test_size=0.2)
    return splitter.process(dataset)


def standardize_results_df(df, dataset_name, seed):
    out = df.copy()
    out["dataset"] = dataset_name
    out["seed"] = seed
    return out


def summarize_seed_variation(raw_df):
    # Expected raw rows: algorithm, name, k, value, dataset, seed
    rows = []
    for (dataset, algorithm, metric, k), grp in raw_df.groupby(["dataset", "algorithm", "name", "k"]):
        vals = grp["value"].astype(float).to_numpy()
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else np.nan
        rows.append({
            "dataset": dataset,
            "algorithm": algorithm,
            "metric": f"{metric}@{k}",
            "mean": mean,
            "std": std,
            "cv": cv,
            "n": int(len(vals)),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric"])


def short_statistical_analysis(summary_df):
    lines = []
    for (dataset, metric), grp in summary_df.groupby(["dataset", "metric"]):
        best = grp.sort_values("cv").iloc[0]
        lines.append(
            f"{dataset} / {metric}: lowest seed sensitivity was {best['algorithm']} "
            f"(mean={best['mean']:.4f}, std={best['std']:.4f}, cv={best['cv']:.4f}, n={int(best['n'])})."
        )
    return "\n".join(lines)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    dataset_specs = [
        ("MovieLens100K", DataSet.MovieLens100K, 3),
        ("Amazon Video Games", DataSet.Amazon2014VideoGames, 3),
        ("Last.FM", DataSet.HetrecLastFM, None),
    ]

    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    all_raw = []

    for dataset_name, dataset_id, threshold in dataset_specs:
        base_dataset = load_and_preprocess(dataset_id, make_implicit_threshold=threshold)
        for seed in seeds:
            split_dataset = build_split_dataset(base_dataset, seed)
            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
            results = evaluator.get_results()
            for _, result_df in results.items():
                all_raw.append(standardize_results_df(result_df, dataset_name, seed))
            print(f"Finished {dataset_name} seed={seed}")

    raw_results_df = pd.concat(all_raw, ignore_index=True)
    summary_df = summarize_seed_variation(raw_results_df)

    raw_path = os.path.join(working_dir, "seed_sensitivity_raw_results.csv")
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    raw_results_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSeed-variation summary:")
    print(summary_df.to_string(index=False))

    print("\nShort statistical analysis:")
    print(short_statistical_analysis(summary_df))

    print("\nRaw results saved to:", raw_path)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
