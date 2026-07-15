import os
from statistics import mean, pstdev

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


def extract_results(results_obj, dataset_name: str, seed: int):
    rows = []
    if isinstance(results_obj, dict):
        dfs = list(results_obj.values())
    else:
        dfs = [results_obj]
    for df in dfs:
        if isinstance(df, pd.DataFrame):
            for algo_name, row in df.iterrows():
                rec = {"dataset": dataset_name, "seed": seed, "algorithm": algo_name}
                rec.update(row.to_dict())
                rows.append(rec)
    return rows


def summarize_statistical_variability(results_df: pd.DataFrame):
    metric_cols = [c for c in results_df.columns if c not in {"dataset", "seed", "algorithm"}]
    lines = []
    for (dataset, algo), grp in results_df.groupby(["dataset", "algorithm"]):
        lines.append(f"{dataset} / {algo}")
        for metric in metric_cols:
            vals = grp[metric].dropna().tolist()
            if len(vals) > 1:
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

    plan = ExperimentPlan("seed_sensitivity_lenskit")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    all_rows = []

    for ds_enum, make_implicit, ds_name in datasets:
        base_ds = load_and_preprocess(ds_enum, make_implicit)
        for seed in seeds:
            split_ds = make_split_dataset(base_ds, seed)
            evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            results = evaluator.get_results()
            all_rows.extend(extract_results(results, ds_name, seed))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("seed_sensitivity_results.csv", index=False)
    print("\nPer-seed results:")
    print(results_df)

    metric_cols = [c for c in results_df.columns if c not in {"dataset", "seed", "algorithm"}]
    summary = results_df.groupby(["dataset", "algorithm"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.to_csv("seed_sensitivity_summary.csv", index=False)
    print("\nSummary across seeds:")
    print(summary)

    print("\nBrief statistical note:")
    print(summarize_statistical_variability(results_df))


if __name__ == "__main__":
    main()
