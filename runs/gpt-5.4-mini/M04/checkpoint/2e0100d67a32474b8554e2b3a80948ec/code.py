import os
import math
from statistics import mean, pstdev
from collections import defaultdict

import pandas as pd

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_and_preprocess(dataset_enum, implicit_threshold=None):
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def evaluate_results(df):
    # Expected columns from OmniRec evaluator: algorithm, fold, name, k, value
    pivot = df.pivot_table(index=["algorithm", "k"], columns="name", values="value", aggfunc="mean")
    return pivot.reset_index()


def summarize_seed_sensitivity(per_run_rows):
    rows = []
    grouped = defaultdict(list)
    for r in per_run_rows:
        grouped[(r["dataset"], r["algorithm"], r["metric"], r["k"])].append(r["value"])

    for (dataset, algorithm, metric, k), vals in grouped.items():
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "metric": metric,
                "k": k,
                "mean": mean(vals),
                "std": pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "range": max(vals) - min(vals),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric", "k"])


if __name__ == '__main__':
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    print(f'Working directory: {working_dir}')

    seeds = [7, 13, 21, 42, 87]
    print(f'Split seeds: {seeds}')

    dataset_specs = [
        ("MovieLens100K", DataSet.MovieLens100K, 3),
        ("Amazon2014VideoGames", DataSet.Amazon2023VideoGames if hasattr(DataSet, "Amazon2023VideoGames") else DataSet.Amazon2014VideoGames, 3),
        ("HetrecLastFM", DataSet.HetrecLastFM, None),
    ]

    all_seed_rows = []
    summary_rows = []

    for ds_name, ds_enum, threshold in dataset_specs:
        print(f'\n=== Dataset: {ds_name} ===')
        for seed in seeds:
            set_random_state(seed)
            dataset = load_and_preprocess(ds_enum, implicit_threshold=threshold)

            plan = ExperimentPlan(plan_name=f"{ds_name}_seed_{seed}")
            plan.add_algorithm(LensKit.ImplicitMFScorer)
            plan.add_algorithm(LensKit.ItemKNNScorer)
            plan.add_algorithm(LensKit.PopScorer)

            metric_args = [NDCG([1, 5, 10]), Recall([1, 5, 10])]

            evaluator = Evaluator(*metric_args)
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

            results = evaluator.get_results()
            # The dict key is dataset-specific; each dataframe contains algorithm/name/k/value rows.
            for _, res_df in results.items():
                if res_df is None or len(res_df) == 0:
                    continue
                for _, row in res_df.iterrows():
                    all_seed_rows.append(
                        {
                            "dataset": ds_name,
                            "seed": seed,
                            "algorithm": row["algorithm"],
                            "metric": row["name"],
                            "k": int(row["k"]) if not pd.isna(row["k"]) else None,
                            "value": float(row["value"]),
                        }
                    )

    all_seed_df = pd.DataFrame(all_seed_rows)
    if all_seed_df.empty:
        print("No results were produced by OmniRec.")
    else:
        print("\nPer-seed results sample:")
        print(all_seed_df.head(20).to_string(index=False))

        summary_df = summarize_seed_sensitivity(all_seed_rows)
        print("\n=== Summary across seeds ===")
        print(summary_df.to_string(index=False))

        print("\n=== Short statistical analysis ===")
        for ds_name in summary_df["dataset"].unique():
            ds_block = summary_df[summary_df["dataset"] == ds_name]
            mean_range = ds_block["range"].mean()
            max_range = ds_block["range"].max()
            most_sensitive = ds_block.loc[ds_block["range"].idxmax()]
            print(
                f"{ds_name}: average seed-induced range={mean_range:.4f}, max range={max_range:.4f}; "
                f"most sensitive = {most_sensitive['algorithm']} {most_sensitive['metric']}@{int(most_sensitive['k'])} "
                f"(std={most_sensitive['std']:.4f}, range={most_sensitive['range']:.4f})"
            )
