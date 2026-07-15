import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def load_and_preprocess(dataset_name, implicit_threshold=None):
    ds = RecSysDataSet.use_dataloader(dataset_name)
    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.append(CorePruning(5))
    ds = Pipe(*steps).process(ds)
    return ds


def build_dataset_for_seed(base_ds, seed):
    set_random_state(seed)
    split = UserHoldout(validation_size=0.0, test_size=0.2)
    # If validation_size=0.0 is not supported in a particular build, this is the one place to adjust.
    return split.process(base_ds)


def build_plan():
    plan = ExperimentPlan("seed_sensitivity_benchmark")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def summarize_results(df):
    agg = (
        df.groupby(["dataset", "algorithm", "name", "k"], as_index=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return agg


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    datasets = {
        "MovieLens100K": (DataSet.MovieLens100K, 3),
        "Amazon2014VideoGames": (DataSet.Amazon2014VideoGames, 3),
        "HetrecLastFM": (DataSet.HetrecLastFM, None),
    }
    seeds = [11, 22, 33, 44, 55]

    records = []
    provenance = []
    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), Recall([1, 5, 10]))

    for ds_name, (ds_enum, thr) in datasets.items():
        base_ds = load_and_preprocess(ds_enum, thr)
        for seed in seeds:
            processed = build_dataset_for_seed(base_ds, seed)
            set_random_state(seed)
            run_omnirec(processed, plan, evaluator)
            res = evaluator.get_results()
            for dataset_id, r in res.items():
                tmp = r.copy()
                tmp["dataset"] = ds_name
                tmp["seed"] = seed
                records.append(tmp)
            provenance.append({
                "dataset": ds_name,
                "seed": seed,
                "preprocessing": ["MakeImplicit(%s)" % thr if thr is not None else "keep implicit", "CorePruning(5)"],
                "split": "UserHoldout(validation_size=0.0, test_size=0.2)",
                "algorithms": ["LensKit.ImplicitMFScorer", "LensKit.ItemKNNScorer", "LensKit.PopScorer"],
                "metrics": "NDCG@1,5,10 and Recall@1,5,10",
                "random_seed": seed,
            })

    all_results = pd.concat(records, ignore_index=True)
    all_results.to_csv(Path(working_dir) / "seed_sensitivity_results.csv", index=False)
    with open(Path(working_dir) / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    summary = (
        all_results.groupby(["dataset", "algorithm", "name", "k"], as_index=False)["value"]
        .agg(mean="mean", std="std")
    )
    summary.to_csv(Path(working_dir) / "summary.csv", index=False)
    print(summary.sort_values(["dataset", "algorithm", "name", "k"]))

    # Brief seed-sensitivity analysis
    print("\nSeed sensitivity analysis:")
    for (dataset, algorithm, name, k), grp in all_results.groupby(["dataset", "algorithm", "name", "k"]):
        mean = grp["value"].mean()
        std = grp["value"].std(ddof=1)
        cv = std / mean if mean != 0 else np.nan
        print(f"{dataset} | {algorithm} | {name}@{k}: mean={mean:.4f}, std={std:.4f}, cv={cv:.3f}")


if __name__ == '__main__':
    main()
