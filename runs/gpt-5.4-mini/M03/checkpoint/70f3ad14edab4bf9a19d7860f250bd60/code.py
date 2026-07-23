import os
import math
import statistics as stats
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import Precision
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_dataset(dataset_name: DataSet) -> RecSysDataSet:
    return RecSysDataSet.use_dataloader(dataset_name)


def preprocess_dataset(dataset: RecSysDataSet, make_implicit: bool, seed: int) -> RecSysDataSet:
    set_random_state(seed)
    steps = [CorePruning(5)]
    if make_implicit:
        steps.insert(0, MakeImplicit(3))
    steps.append(UserHoldout(validation_size=0.2, test_size=0.2))
    return Pipe(*steps).process(dataset)


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan("Seed-Sensitivity Study")
    plan.add_algorithm(LensKit.PopScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )


def extract_results_df(evaluator: Evaluator) -> pd.DataFrame:
    results = evaluator.get_results()
    if isinstance(results, pd.DataFrame):
        return results.copy()
    if isinstance(results, dict):
        frames = []
        for dataset_name, df in results.items():
            if df is None or df.empty:
                continue
            tmp = df.copy()
            tmp["dataset"] = dataset_name
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame(results)


def normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    df = results.copy()
    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"algo", "algorithm_name", "model", "method"}:
            rename_map[c] = "algorithm"
        elif lc in {"ds", "dataset_name"}:
            rename_map[c] = "dataset"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    metric_cols = [c for c in results.columns if c not in {"dataset", "algorithm", "seed"}]
    grouped = results.groupby(["dataset", "algorithm"], dropna=False)[metric_cols]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col for col in summary.columns]
    return summary


def short_stat_analysis(results: pd.DataFrame) -> str:
    if results.empty:
        return "No results were produced."
    metric_cols = [c for c in results.columns if c not in {"dataset", "algorithm", "seed"}]
    lines = []
    for metric in metric_cols:
        vals = [float(v) for v in results[metric].dropna().tolist()]
        if not vals:
            continue
        mean_v = float(stats.mean(vals))
        std_v = float(stats.pstdev(vals)) if len(vals) > 1 else 0.0
        cv = (std_v / mean_v) if mean_v != 0 else math.nan
        lines.append(f"{metric}: mean={mean_v:.4f}, std={std_v:.4f}, cv={cv:.4f}")
    return " | ".join(lines)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    print(f"Working directory: {working_dir}")

    dataset_specs = [
        ("MovieLens100K", DataSet.MovieLens100K, True),
        ("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, True),
        ("HetrecLastFM", DataSet.HetrecLastFM, False),
    ]
    seeds = [11, 22, 33, 44, 55]
    plan = build_plan()
    all_rows: List[Dict[str, object]] = []

    for dataset_label, dataset_enum, make_implicit in dataset_specs:
        base = load_dataset(dataset_enum)
        for seed in seeds:
            processed = preprocess_dataset(base, make_implicit=make_implicit, seed=seed)
            evaluator = build_evaluator()
            run_omnirec(datasets=processed, plan=plan, evaluator=evaluator)
            df = normalize_results(extract_results_df(evaluator))
            if df.empty:
                continue
            if "dataset" not in df.columns:
                df["dataset"] = dataset_label
            if "algorithm" not in df.columns:
                df["algorithm"] = "unknown"
            df["seed"] = seed
            for _, row in df.iterrows():
                all_rows.append(row.to_dict())

    results = pd.DataFrame(all_rows)
    print("\nRaw per-seed results:")
    if results.empty:
        print("No results available.")
        return

    print(results)
    summary = summarize(results)
    print("\nAggregated summary (mean/std):")
    print(summary)
    print("\nSeed sensitivity analysis:")
    print(short_stat_analysis(results))


if __name__ == "__main__":
    main()
