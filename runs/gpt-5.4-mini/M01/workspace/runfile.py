import os
from statistics import mean, pstdev
from typing import Any

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
    # Use a single user-based holdout: 80% train / 20% test, with no validation contribution.
    # UserHoldout requires both sizes; setting validation_size to 0.0 preserves the requested
    # 80/20 train/test split while keeping the user-wise split semantics.
    return UserHoldout(validation_size=0.0, test_size=0.2).process(base_ds)


def build_plan():
    plan = ExperimentPlan("seed_sensitivity_lenskit")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def wide_results_from_evaluator(evaluator: Evaluator, dataset_name: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = evaluator.get_results()
    if not isinstance(results, dict):
        results = {dataset_name: results}

    for _, df in results.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        if {"algorithm", "name", "k", "value"}.issubset(df.columns):
            wide = (
                df.pivot_table(index="algorithm", columns=["name", "k"], values="value", aggfunc="mean")
                .reset_index()
            )
            wide.columns = [
                "algorithm" if c == "algorithm" else f"{c[0]}@{int(c[1])}"
                for c in wide.columns.to_flat_index()
            ]
            for _, row in wide.iterrows():
                rec = {"dataset": dataset_name, "seed": seed, "algorithm": row["algorithm"]}
                for c, v in row.items():
                    if c != "algorithm":
                        rec[c] = v
                rows.append(rec)
        else:
            for algo_name, row in df.iterrows():
                rec = {"dataset": dataset_name, "seed": seed, "algorithm": algo_name}
                rec.update(row.to_dict())
                rows.append(rec)
    return rows


def summarize(results_df: pd.DataFrame) -> str:
    metric_cols = [c for c in results_df.columns if c not in {"dataset", "seed", "algorithm"}]
    lines = []
    for (dataset, algorithm), grp in results_df.groupby(["dataset", "algorithm"], sort=True).items():
        lines.append(f"{dataset} / {algorithm}")
        for metric in metric_cols:
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna().tolist()
            if not vals:
                continue
            if len(vals) == 1:
                lines.append(f"  {metric}: {vals[0]:.4f}")
            else:
                lines.append(
                    f"  {metric}: mean={mean(vals):.4f}, std={pstdev(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}"
                )
    return "\n".join(lines)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    seeds = [11, 22, 33, 44, 55]

    datasets = [
        (DataSet.MovieLens100K, True, "MovieLens100K"),
        (DataSet.Amazon2014VideoGames, True, "Amazon2014VideoGames"),
        (DataSet.HetrecLastFM, False, "HetrecLastFM"),
    ]

    plan = build_plan()
    all_rows = []

    for ds_enum, make_implicit, ds_name in datasets:
        base_ds = load_and_preprocess(ds_enum, make_implicit)
        for seed in seeds:
            split_ds = make_split_dataset(base_ds, seed)
            evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            all_rows.extend(wide_results_from_evaluator(evaluator, ds_name, seed))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("seed_sensitivity_results.csv", index=False)

    if not results_df.empty:
        summary = results_df.groupby(["dataset", "algorithm"]).agg("mean", numeric_only=True).reset_index()
        summary.to_csv("seed_sensitivity_summary.csv", index=False)

    print("Per-seed results:")
    print(results_df.to_string(index=False))

    print("\nBrief statistical note:")
    print(summarize(results_df))


if __name__ == "__main__":
    main()