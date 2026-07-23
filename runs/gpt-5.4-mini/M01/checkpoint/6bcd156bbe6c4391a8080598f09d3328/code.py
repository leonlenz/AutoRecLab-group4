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
    # Verified API: UserHoldout(validation_size, test_size)
    # A pure train/test split is not supported by passing an invalid 0.0 test size.
    # Use a documented holdout configuration that keeps evaluation reproducible.
    return UserHoldout(validation_size=0.0, test_size=0.2).process(base_ds)


def collect_results(results_dict, dataset_name: str, seed: int):
    rows = []
    for _, df in results_dict.items():
        for algo_name, row in df.iterrows():
            rec = {"dataset": dataset_name, "seed": seed, "algorithm": algo_name}
            rec.update(row.to_dict())
            rows.append(rec)
    return rows


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
            all_rows.extend(collect_results(results, ds_name, seed))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("seed_sensitivity_results.csv", index=False)
    print("\nPer-seed results:")
    print(results_df)

    metric_cols = [c for c in results_df.columns if c not in {"dataset", "seed", "algorithm"}]
    summary = (
        results_df.groupby(["dataset", "algorithm"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv("seed_sensitivity_summary.csv", index=False)
    print("\nSummary across seeds:")
    print(summary)

    print("\nBrief statistical note:")
    for _, grp in results_df.groupby(["dataset", "algorithm"]):
        dataset = grp["dataset"].iloc[0]
        algo = grp["algorithm"].iloc[0]
        print(f"{dataset} / {algo}")
        for metric in metric_cols:
            vals = grp[metric].dropna().tolist()
            if len(vals) > 1:
                print(
                    f"  {metric}: mean={mean(vals):.4f}, std={pstdev(vals):.4f}, "
                    f"min={min(vals):.4f}, max={max(vals):.4f}"
                )


if __name__ == "__main__":
    main()
