import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


SEEDS = [7, 13, 21, 37, 101]
K_VALUES = [1, 5, 10]


@dataclass
class DatasetSpec:
    name: str
    loader: DataSet
    make_implicit: bool


DATASET_SPECS = [
    DatasetSpec("MovieLens100K", DataSet.MovieLens100K, True),
    DatasetSpec("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, True),
    DatasetSpec("HetrecLastFM", DataSet.HetrecLastFM, False),
]


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name="SeedSensitivity_ThreeAlgo_ThreeData")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(
        NDCG(K_VALUES),
        Precision(K_VALUES),
    )


def preprocess_dataset(spec: DatasetSpec) -> RecSysDataSet:
    ds = RecSysDataSet.use_dataloader(spec.loader)
    steps = []
    if spec.make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    ds = Pipe(*steps).process(ds)
    return ds


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in {"dataset", "algorithm", "seed"}]
    rows = []
    for (dataset, algorithm), grp in df.groupby(("dataset", "algorithm")):
        row = {"dataset": dataset, "algorithm": algorithm, "n_seeds": len(grp)}
        for col in metric_cols:
            vals = grp[col].astype(float).to_numpy()
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mean = row[f"{col}_mean"]
            row[f"{col}_cv"] = float(row[f"{col}_std"] / mean) if mean != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def short_statistical_analysis(summary: pd.DataFrame) -> None:
    print("\n=== Short statistical analysis of split-seed sensitivity ===")
    for metric in [f"NDCG@{k}" for k in K_VALUES] + [f"Precision@{k}" for k in K_VALUES]:
        best = summary.sort_values(f"{metric}_std", ascending=False).iloc[0]
        worst = summary.sort_values(f"{metric}_std", ascending=True).iloc[0]
        print(
            f"{metric}: highest seed variability = {best['dataset']} / {best['algorithm']} "
            f"(std={best[f'{metric}_std']:.6f}, cv={best[f'{metric}_cv']:.3f}); "
            f"lowest = {worst['dataset']} / {worst['algorithm']} "
            f"(std={worst[f'{metric}_std']:.6f}, cv={worst[f'{metric}_cv']:.3f})"
        )


if __name__ == '__main__':
    results = []
    datasets = []
    for spec in DATASET_SPECS:
        ds = preprocess_dataset(spec)
        datasets.append(ds)
        print(f"Loaded and preprocessed {spec.name}: {ds}")

    plan = build_plan()
    evaluator = build_evaluator()

    for seed in SEEDS:
        set_random_state(seed)
        print(f"\nRunning experiments with seed={seed}")
        run_omnirec(datasets=datasets, plan=plan, evaluator=evaluator)

        for spec in DATASET_SPECS:
            for algo in ["ImplicitMFScorer", "ItemKNNScorer", "PopScorer"]:
                row = {"dataset": spec.name, "algorithm": algo, "seed": seed}
                for metric in ["NDCG", "Precision"]:
                    for k in K_VALUES:
                        row[f"{metric}@{k}"] = np.nan
                results.append(row)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(working_dir, "seed_sensitivity_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved per-seed placeholder results to {results_path}")

    summary_df = summarize_results(results_df)
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(summary_df.to_string(index=False))
    short_statistical_analysis(summary_df)
