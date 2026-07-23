import os
import json
import math
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
from omnirec.runner.plan_components import Grid
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_and_preprocess(dataset_id, threshold=None):
    ds = RecSysDataSet.use_dataloader(dataset_id)
    steps = []
    if threshold is not None:
        steps.append(MakeImplicit(threshold))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def build_split_dataset(dataset, seed):
    set_random_state(seed)
    splitter = UserHoldout(validation_size=0.0, test_size=0.2)
    return splitter.process(dataset)


def build_plan():
    plan = ExperimentPlan("seed_sensitivity_lenskit_baselines")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def summarize_results(df):
    summary_rows = []
    for (algorithm, name, k), grp in df.groupby(["algorithm", "name", "k"]):
        vals = grp["value"].astype(float).to_numpy()
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else np.nan
        summary_rows.append({
            "algorithm": algorithm,
            "metric": f"{name}@{k}",
            "mean": mean,
            "std": std,
            "cv": cv,
            "n": len(vals),
        })
    return pd.DataFrame(summary_rows).sort_values(["algorithm", "metric"])


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    dataset_specs = [
        ("MovieLens100K", DataSet.MovieLens100K, 3),
        ("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, 3),
        ("HetrecLastFM", DataSet.HetrecLastFM, None),
    ]

    all_summaries = []
    all_raw_results = []

    plan = build_plan()

    for dataset_name, dataset_id, threshold in dataset_specs:
        base_dataset = load_and_preprocess(dataset_id, threshold=threshold)
        for seed in seeds:
            split_dataset = build_split_dataset(base_dataset, seed)
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )
            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
            results = evaluator.get_results()
            for ds_key, result_df in results.items():
                out_df = result_df.copy()
                out_df["dataset"] = dataset_name
                out_df["seed"] = seed
                all_raw_results.append(out_df)
                all_summaries.append(summarize_results(out_df.assign(algorithm=out_df["algorithm"])))
            print(f"Finished {dataset_name} seed={seed}")

    raw_results_df = pd.concat(all_raw_results, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)

    raw_path = os.path.join(working_dir, "seed_sensitivity_raw_results.csv")
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    raw_results_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nOverall seed-variation summary:")
    print(summary_df.groupby(["algorithm", "metric"])[["mean", "std", "cv", "n"]].mean())

    print("\nRaw results saved to:", raw_path)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
