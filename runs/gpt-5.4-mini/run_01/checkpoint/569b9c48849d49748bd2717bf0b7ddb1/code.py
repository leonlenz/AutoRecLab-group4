import os
from statistics import mean, pstdev
from typing import Any, cast

import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_and_preprocess(dataset_enum, make_implicit: bool):
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def make_split_dataset(base_ds, seed: int):
    set_random_state(seed)
    return UserHoldout(validation_size=0.2, test_size=0.2).process(base_ds)


def build_plan():
    plan = ExperimentPlan("seed_sensitivity_lenskit")
    # OmniRec documents LensKit.ImplicitMFScorer as the ALS-style implicit MF wrapper.
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def _get_metric_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in {"dataset", "seed", "algorithm"}]


def summarize_statistical_variability(results_df: pd.DataFrame):
    metric_cols = _get_metric_columns(results_df)
    lines = []
    grouped = cast(Any, results_df.groupby(["dataset", "algorithm"], sort=True))
    for (dataset, algo), grp in grouped:
        lines.append(f"{dataset} / {algo}")
        for metric in metric_cols:
            vals = grp[metric].dropna().tolist()
            if not vals:
                continue
            if len(vals) == 1:
                lines.append(f"  {metric}: single run value={vals[0]:.4f}")
            else:
                lines.append(
                    f"  {metric}: mean={mean(vals):.4f}, std={pstdev(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}"
                )
    return "\n".join(lines)


def extract_results(evaluator: Evaluator, dataset_name: str, seed: int):
    rows = []
    results = evaluator.get_results()

    # evaluator.get_results() returns dataset-level DataFrames keyed by dataset id.
    if isinstance(results, dict):
        iter_dfs = results.values()
    else:
        iter_dfs = [results]

    for df in iter_dfs:
        if not isinstance(df, pd.DataFrame):
            continue
        for algo_name, row in df.iterrows():
            rec = {"dataset": dataset_name, "seed": seed, "algorithm": algo_name}
            rec.update(row.to_dict())
            rows.append(rec)
    return rows


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    # Fixed seeds for reproducibility across preprocessing and splitting.
    seeds = [11, 22, 33, 44, 55]

    datasets = [
        (DataSet.MovieLens100K, True, "MovieLens100K"),
        (DataSet.Amazon2014VideoGames, True, "Amazon2014VideoGames"),
        (DataSet.HetrecLastFM, False, "HetrecLastFM"),
    ]

    plan = build_plan()

    # Requested metrics: nDCG@k and Precision@k for k=1,5,10.
    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    all_rows = []

    for ds_enum, make_implicit, ds_name in datasets:
        base_ds = load_and_preprocess(ds_enum, make_implicit)
        for seed in seeds:
            split_ds = make_split_dataset(base_ds, seed)
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            all_rows.extend(extract_results(evaluator, ds_name, seed))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("seed_sensitivity_results.csv", index=False)

    # Brief aggregate table across seeds.
    metric_cols = _get_metric_columns(results_df)
    summary = (
        results_df.groupby(["dataset", "algorithm"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv("seed_sensitivity_summary.csv", index=False)

    print("Per-seed results:")
    print(results_df.to_string(index=False))

    print("\nSummary across seeds:")
    print(summary.to_string(index=False))

    print("\nBrief statistical note:")
    print(summarize_statistical_variability(results_df))


if __name__ == "__main__":
    main()
