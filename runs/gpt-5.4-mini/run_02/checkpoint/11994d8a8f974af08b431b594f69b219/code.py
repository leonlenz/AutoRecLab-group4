import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_and_preprocess(dataset_enum, implicit_threshold=None):
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.append(CorePruning(5))
    ds = Pipe(*steps).process(ds)
    return ds


def split_with_seed(ds, seed):
    set_random_state(seed)
    split = UserHoldout(validation_size=0.01, test_size=0.2)
    return split.process(ds)


def build_plan():
    plan = ExperimentPlan("seed_sensitivity_benchmark")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def normalize_results(res, dataset_name, seed):
    records = []
    if isinstance(res, dict):
        iterable = res.items()
    else:
        iterable = [(None, res)]
    for _, df in iterable:
        tmp = df.copy()
        tmp["dataset"] = dataset_name
        tmp["seed"] = seed
        records.append(tmp)
    return records


def summarize(all_results):
    summary = (
        all_results.groupby(["dataset", "algorithm", "name", "k"], as_index=False)["value"]
        .agg(mean="mean", std="std")
    )
    summary["cv"] = summary["std"] / summary["mean"].replace(0, np.nan)
    return summary


def paired_seed_analysis(all_results):
    print("\nSeed sensitivity analysis (paired across seeds):")
    metrics = all_results[["dataset", "algorithm", "name", "k"]].drop_duplicates()
    for dataset in metrics["dataset"].unique():
        ds_df = all_results[all_results["dataset"] == dataset]
        algs = ds_df["algorithm"].unique().tolist()
        for name in ["NDCG", "Precision"]:
            for k in [1, 5, 10]:
                pivot = ds_df[(ds_df["name"] == name) & (ds_df["k"] == k)].pivot_table(
                    index="seed", columns="algorithm", values="value", aggfunc="mean"
                )
                print(f"{dataset} | {name}@{k}")
                print(pivot)
                if len(algs) >= 2:
                    base = algs[0]
                    for other in algs[1:]:
                        if base in pivot.columns and other in pivot.columns:
                            diff = pivot[other] - pivot[base]
                            diff = diff.dropna()
                            if len(diff) > 1:
                                tstat, pval = stats.ttest_1samp(diff, 0.0)
                                print(f"  paired diff {other} - {base}: mean={diff.mean():.4f}, std={diff.std(ddof=1):.4f}, t={tstat:.3f}, p={pval:.4f}")
                print()


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    datasets = {
        "MovieLens100K": (DataSet.MovieLens100K, 3),
        "Amazon2014VideoGames": (DataSet.Amazon2014VideoGames, 3),
        "HetrecLastFM": (DataSet.HetrecLastFM, None),
    }
    seeds = [11, 22, 33, 44, 55]

    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    provenance = []
    all_records = []

    for ds_name, (ds_enum, thr) in datasets.items():
        base_ds = load_and_preprocess(ds_enum, thr)
        for seed in seeds:
            processed = split_with_seed(base_ds, seed)
            set_random_state(seed)
            run_omnirec(processed, plan, evaluator)
            results = evaluator.get_results()
            all_records.extend(normalize_results(results, ds_name, seed))
            provenance.append({
                "dataset": ds_name,
                "seed": seed,
                "preprocessing": [f"MakeImplicit({thr})" if thr is not None else "implicit feedback kept", "CorePruning(5)"],
                "split": "UserHoldout(validation_size=0.01, test_size=0.2)",
                "algorithms": ["LensKit.ImplicitMFScorer", "LensKit.ItemKNNScorer", "LensKit.PopScorer"],
                "metrics": ["NDCG@1", "NDCG@5", "NDCG@10", "Precision@1", "Precision@5", "Precision@10"],
                "random_seed": seed,
            })

    all_results = pd.concat(all_records, ignore_index=True)
    all_results.to_csv(Path(working_dir) / "seed_sensitivity_results.csv", index=False)
    summary = summarize(all_results)
    summary.to_csv(Path(working_dir) / "summary.csv", index=False)
    with open(Path(working_dir) / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(summary.sort_values(["dataset", "algorithm", "name", "k"]))
    paired_seed_analysis(all_results)


if __name__ == "__main__":
    main()
