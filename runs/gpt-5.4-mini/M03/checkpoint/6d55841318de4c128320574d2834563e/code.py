import os
import math
import statistics
from collections import defaultdict

import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_and_preprocess(dataset_name):
    dataset = RecSysDataSet.use_dataloader(dataset_name)
    steps = []
    if dataset_name in {DataSet.MovieLens100K, DataSet.Amazon2014VideoGames}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(dataset)


def split_with_seed(dataset, seed):
    set_random_state(seed)
    # True 80/20 user-based holdout: no validation partition.
    split = UserHoldout(validation_size=0, test_size=0.2)
    return split.process(dataset)


def build_plan():
    plan = ExperimentPlan(plan_name='seed_sensitivity_lenskit_baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def normalize_results_table(results_df, dataset_label, seed):
    df = results_df.copy()
    df = df.rename(columns={"name": "metric"})
    df["dataset"] = dataset_label
    df["seed"] = seed
    return df[["dataset", "seed", "algorithm", "metric", "k", "value"]]


def aggregate_across_seeds(all_results_df):
    agg = (
        all_results_df.groupby(["dataset", "algorithm", "metric", "k"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), min=("value", "min"), max=("value", "max"), n=("value", "count"))
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["cv"] = agg.apply(lambda r: (r["std"] / r["mean"]) if r["mean"] not in (0, None) else math.nan, axis=1)
    return agg.sort_values(["dataset", "algorithm", "metric", "k"])


def short_statistical_analysis(agg_df):
    rows = []
    for (dataset, algorithm), g in agg_df.groupby(["dataset", "algorithm"]):
        top = g.sort_values(["std", "cv"], ascending=False).head(2)
        rows.append({
            "dataset": dataset,
            "algorithm": algorithm,
            "most_variable_metrics": ", ".join([f"{r.metric}@{int(r.k)} (std={r.std:.4f}, cv={r.cv:.3f})" for _, r in top.iterrows()]),
            "avg_std": g["std"].mean(),
            "avg_cv": g["cv"].replace([math.inf, -math.inf], math.nan).mean(),
        })
    return pd.DataFrame(rows)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    seeds = [1, 2, 3, 4, 5]
    datasets = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }

    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    all_rows = []

    for dataset_label, dataset_name in datasets.items():
        print(f"\n=== Loading and preprocessing {dataset_label} ===")
        base = load_and_preprocess(dataset_name)

        for seed in seeds:
            print(f"Running {dataset_label} with seed={seed}...")
            split_ds = split_with_seed(base, seed)

            # run_omnirec accepts a dataset or list of datasets; using one at a time keeps
            # results easy to attribute to the current seed.
            run_omnirec(datasets=[split_ds], plan=plan, evaluator=evaluator)

            results_dict = evaluator.get_results()
            if not results_dict:
                raise RuntimeError(f"No results returned for {dataset_label}, seed={seed}")

            # There should be exactly one dataset entry for this single run.
            dataset_id, results_df = next(iter(results_dict.items()))
            normalized = normalize_results_table(results_df, dataset_label, seed)
            all_rows.append(normalized)

            # Reinitialize evaluator so each seed run is isolated.
            evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    all_results_df = pd.concat(all_rows, ignore_index=True)
    agg_df = aggregate_across_seeds(all_results_df)
    stats_df = short_statistical_analysis(agg_df)

    print("\n=== Aggregated Results Across 5 Seeds ===")
    print(agg_df.to_string(index=False))

    print("\n=== Brief Seed-Sensitivity Analysis ===")
    print(stats_df.to_string(index=False))

    # Save outputs for inspection
    all_results_df.to_csv("seed_level_results.csv", index=False)
    agg_df.to_csv("aggregated_results.csv", index=False)
    stats_df.to_csv("seed_sensitivity_analysis.csv", index=False)


if __name__ == '__main__':
    main()
