import os
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.plan_components import Grid
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.util import set_random_state
from omnirec.util.run import run_omnirec


def summarize_seed_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.startswith("ndcg@") or c.startswith("precision@")]
    rows = []
    for _, g in df.groupby(["dataset", "algorithm"], sort=True, as_index=False):
        dataset = g.iloc[0]["dataset"]
        algorithm = g.iloc[0]["algorithm"]
        row = {"dataset": dataset, "algorithm": algorithm, "n_seeds": len(g)}
        for m in metric_cols:
            vals = g[m].astype(float).to_numpy()
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_cv"] = float((np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 and np.mean(vals) != 0 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_one_dataset(name: str, dataset_enum: Any, working_dir: str, seeds: list[int]) -> pd.DataFrame:
    print(f"\n=== Loading {name} ===")
    dataset = RecSysDataSet.use_dataloader(dataset_enum)

    steps = []
    if name in {"MovieLens100K", "Amazon2014VideoGames"}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    steps.append(UserHoldout(0.2, 0.2))
    pipe = Pipe(*steps)

    plan = ExperimentPlan(plan_name=f"{name}-seed-sensitivity")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {"features": 20, "epochs": 10})
    plan.add_algorithm(LensKit.ItemKNNScorer, {"max_nbrs": 20})
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(NDCG([1, 5, 10]))

    all_rows = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_random_state(seed)
        np.random.seed(seed)
        preprocessed = pipe.process(dataset)

        run_omnirec(datasets=preprocessed, plan=plan, evaluator=evaluator)

        results_path = os.path.join(working_dir, f"{name}_seed_{seed}_results.csv")
        if os.path.exists(results_path):
            res = pd.read_csv(results_path)
        else:
            res = pd.DataFrame()
        if not res.empty:
            res["dataset"] = name
            res["seed"] = seed
            all_rows.append(res)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [1, 2, 3, 4, 5]
    datasets = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }

    print("Experiment configuration:")
    print(json.dumps({"seeds": seeds, "datasets": list(datasets.keys()), "split": "user-based 80/20 holdout", "core": 5, "implicit_threshold": 3}, indent=2))

    all_results = []
    for name, enum_val in datasets.items():
        df = run_one_dataset(name, enum_val, working_dir, seeds)
        if not df.empty:
            all_results.append(df)

    if all_results:
        results = pd.concat(all_results, ignore_index=True)
        results.to_csv(os.path.join(working_dir, "all_seed_results.csv"), index=False)
        print("\n=== Per-seed results ===")
        print(results)
        summary = summarize_seed_sensitivity(results)
        summary.to_csv(os.path.join(working_dir, "seed_sensitivity_summary.csv"), index=False)
        print("\n=== Seed sensitivity summary ===")
        print(summary)
    else:
        print("No results were collected. Check OmniRec output locations and evaluator configuration.")


if __name__ == "__main__":
    main()
