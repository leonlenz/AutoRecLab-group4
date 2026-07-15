import os
import json
import math
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd

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


def _load_dataset(dataset_enum):
    return RecSysDataSet.use_dataloader(dataset_enum)


def _preprocess_for_seed(dataset, dataset_name, seed):
    set_random_state(seed)
    steps = [CorePruning(5)]
    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        steps.insert(0, MakeImplicit(3))
    pipeline = Pipe(*steps, UserHoldout(validation_size=0.0, test_size=0.2))
    return pipeline.process(dataset)


def _extract_metrics(result_obj):
    if isinstance(result_obj, pd.DataFrame):
        return result_obj
    if hasattr(result_obj, "to_df"):
        return result_obj.to_df()
    if hasattr(result_obj, "results"):
        r = result_obj.results
        if isinstance(r, pd.DataFrame):
            return r
    raise TypeError(f"Unsupported result type: {type(result_obj)}")


def summarize(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = float(std / mean) if mean != 0 else float("nan")
    return {"mean": mean, "std": std, "cv": cv, "min": float(np.min(arr)), "max": float(np.max(arr))}


def short_statistical_analysis(summary_rows):
    lines = []
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[(row["dataset"], row["algorithm"])] .append(row)
    for (dataset, algorithm), rows in grouped.items():
        ndcg10 = rows[0]["nDCG@10_mean"]
        prec10 = rows[0]["Precision@10_mean"]
        ndcg10_cv = rows[0]["nDCG@10_cv"]
        prec10_cv = rows[0]["Precision@10_cv"]
        lines.append(
            f"{dataset} / {algorithm}: nDCG@10={ndcg10:.4f} (CV={ndcg10_cv:.2%}), Precision@10={prec10:.4f} (CV={prec10_cv:.2%})."
        )
    return lines


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    datasets = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }

    plan = ExperimentPlan("seed_sensitivity_holdout")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    all_seed_rows = []
    config = {
        "datasets": list(datasets.keys()),
        "seeds": seeds,
        "preprocessing": {
            "core_pruning": 5,
            "implicit_threshold": 3,
            "implicit_datasets": ["MovieLens100K", "Amazon2014VideoGames"],
            "split": {"type": "UserHoldout", "validation_size": 0.0, "test_size": 0.2},
        },
        "algorithms": ["LensKit.ImplicitMFScorer", "LensKit.ItemKNNScorer", "LensKit.PopScorer"],
        "metrics": ["nDCG@1", "nDCG@5", "nDCG@10", "Precision@1", "Precision@5", "Precision@10"],
    }

    print("Running configuration:")
    print(json.dumps(config, indent=2))

    for dataset_name, dataset_enum in datasets.items():
        raw_dataset = _load_dataset(dataset_enum)
        for seed in seeds:
            print(f"\n=== Dataset={dataset_name} Seed={seed} ===")
            processed = _preprocess_for_seed(raw_dataset, dataset_name, seed)
            result = run_omnirec(processed, plan, evaluator)
            df = _extract_metrics(result)
            df = df.copy()
            df["dataset"] = dataset_name
            df["seed"] = seed
            all_seed_rows.append(df)
            print(df)

    combined = pd.concat(all_seed_rows, ignore_index=True)
    combined.to_csv(os.path.join(working_dir, "seed_sensitivity_raw_results.csv"), index=False)

    metric_cols = [c for c in combined.columns if c not in {"dataset", "seed", "algorithm"}]
    summary = []
    for (dataset_name, algorithm), g in combined.groupby(["dataset", "algorithm"]):
        row = {"dataset": dataset_name, "algorithm": algorithm}
        for col in ["nDCG@1", "nDCG@5", "nDCG@10", "Precision@1", "Precision@5", "Precision@10"]:
            if col in g.columns:
                s = summarize(g[col].tolist())
                row[f"{col}_mean"] = s["mean"]
                row[f"{col}_std"] = s["std"]
                row[f"{col}_cv"] = s["cv"]
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(working_dir, "seed_sensitivity_summary.csv"), index=False)

    print("\n=== Aggregate Summary ===")
    print(summary_df)

    print("\n=== Short Statistical Analysis ===")
    for line in short_statistical_analysis(summary):
        print(line)

    with open(os.path.join(working_dir, "reproducibility_report.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
