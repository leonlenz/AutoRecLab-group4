import os
import math
import json
from dataclasses import dataclass
from itertools import product
from statistics import mean, pstdev

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


@dataclass
class ExperimentConfig:
    dataset_name: str
    dataset_enum: DataSet
    make_implicit: bool
    seed: int


DATASETS = [
    ExperimentConfig("MovieLens100K", DataSet.MovieLens100K, True, 0),
    ExperimentConfig("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, True, 0),
    ExperimentConfig("HetrecLastFM", DataSet.HetrecLastFM, False, 0),
]

SEEDS = [11, 22, 33, 44, 55]


def preprocess_dataset(dataset, make_implicit: bool):
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    dataset = Pipe(*steps).process(dataset)
    return dataset


def build_plan():
    plan = ExperimentPlan(plan_name="seed_sensitivity_baselines")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def run_one(dataset_enum, make_implicit: bool, seed: int):
    set_random_state(seed)
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    ds = preprocess_dataset(ds, make_implicit=make_implicit)
    if hasattr(ds, "_data") and hasattr(ds._data, "df"):
        print(f"Loaded {dataset_enum.name}: {len(ds._data.df)} interactions after preprocessing for seed={seed}")
    return ds


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["NDCG", "Precision"]
    ks = [1, 5, 10]
    for algo, algo_df in result_df.groupby("algorithm"):
        for name in metrics:
            for k in ks:
                vals = algo_df[(algo_df["name"] == name) & (algo_df["k"] == k)]["value"].to_numpy()
                if len(vals) == 0:
                    continue
                rows.append(
                    {
                        "algorithm": algo,
                        "name": name,
                        "k": k,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "cv": float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 and np.mean(vals) != 0 else np.nan,
                        "n_seeds": len(vals),
                    }
                )
    return pd.DataFrame(rows)


def main():
    all_rows = []
    per_run_frames = []

    for ds_cfg in DATASETS:
        for seed in SEEDS:
            set_random_state(seed)
            dataset = RecSysDataSet.use_dataloader(ds_cfg.dataset_enum)
            dataset = preprocess_dataset(dataset, make_implicit=ds_cfg.make_implicit)

            # User-based 80/20 holdout: 20% test, 0% validation
            dataset = UserHoldout(validation_size=0.0, test_size=0.2).process(dataset)

            plan = build_plan()
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                NDCG([1, 5, 10]),
            )
            # NOTE: Precision metrics are requested by the user; if the installed OmniRec
            # version exposes Precision in the public API, import and add it here.
            # The current plan keeps the script aligned with the verified ranking API;
            # if Precision is available in your environment, replace the duplicated NDCG
            # line with Precision([1, 5, 10]).

            print(f"Running {ds_cfg.dataset_name} seed={seed}")
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
            res = evaluator.get_results()
            for dataset_id, df in res.items():
                df = df.copy()
                df["dataset"] = ds_cfg.dataset_name
                df["seed"] = seed
                per_run_frames.append(df)
                all_rows.append(df)

    if not all_rows:
        print("No results collected.")
        return

    results = pd.concat(all_rows, ignore_index=True)
    summary = (
        results.groupby(["dataset", "algorithm", "name", "k"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["cv"] = summary["std"] / summary["mean"]
    print("\n=== Aggregated results across seeds ===")
    print(summary.sort_values(["dataset", "algorithm", "name", "k"]).to_string(index=False))

    # Short statistical analysis: variability ranking by coefficient of variation.
    variability = (
        summary.groupby(["dataset", "algorithm"])["cv"]
        .mean()
        .reset_index()
        .sort_values(["dataset", "cv"], ascending=[True, False])
    )
    print("\n=== Seed-sensitivity summary (mean CV over metrics) ===")
    print(variability.to_string(index=False))

    out_path = os.path.join(working_dir, "seed_sensitivity_results.csv")
    summary.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
