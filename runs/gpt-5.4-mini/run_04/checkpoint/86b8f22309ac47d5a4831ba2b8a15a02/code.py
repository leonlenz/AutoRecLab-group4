import os
import math
import statistics
from collections import defaultdict

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def _load_dataset(dataset_name):
    return RecSysDataSet.use_dataloader(dataset_name)


def _preprocess_dataset(dataset, make_implicit):
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(dataset)


def _build_plan():
    plan = ExperimentPlan(plan_name="Seed-Sensitivity LensKit Comparison")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def _build_evaluator():
    return Evaluator(
        NDCG([1, 5, 10]),
        Recall([1, 5, 10]),
    )


def _extract_results(results):
    # Try to normalize result objects into a flat list of rows.
    if results is None:
        return []
    if isinstance(results, list):
        return results
    for attr in ("results", "rows", "data", "summary"):
        if hasattr(results, attr):
            obj = getattr(results, attr)
            if callable(obj):
                obj = obj()
            if isinstance(obj, list):
                return obj
    return [results]


def _row_to_dict(row):
    if isinstance(row, dict):
        return row
    out = {}
    for key in dir(row):
        if key.startswith("_"):
            continue
        try:
            val = getattr(row, key)
        except Exception:
            continue
        if callable(val):
            continue
        out[key] = val
    return out


def _find_metric_value(row_dict, metric_name, k):
    target_keys = [
        f"{metric_name}@{k}",
        f"{metric_name}_{k}",
        f"{metric_name.lower()}@{k}",
        f"{metric_name.lower()}_{k}",
    ]
    for key in target_keys:
        if key in row_dict:
            return row_dict[key]
    for key, value in row_dict.items():
        lk = str(key).lower()
        if metric_name.lower() in lk and str(k) in lk:
            return value
    return None


def _summarize(values):
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


if __name__ == "__main__":
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seed_list = [7, 21, 42, 84, 123]
    print("Working directory:", working_dir)
    print("Split seeds:", seed_list)

    datasets = [
        ("MovieLens100K", DataSet.MovieLens100K, True),
        ("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, True),
        ("HetrecLastFM", DataSet.HetrecLastFM, False),
    ]

    plan = _build_plan()
    evaluator = _build_evaluator()

    all_run_records = []
    for seed in seed_list:
        set_random_state(seed)
        for ds_name, ds_enum, make_implicit in datasets:
            print(f"Loading {ds_name}...")
            dataset = _load_dataset(ds_enum)
            dataset = _preprocess_dataset(dataset, make_implicit=make_implicit)
            print(f"Running experiment for dataset={ds_name}, seed={seed}...")
            result = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
            rows = _extract_results(result)
            for row in rows:
                row_dict = _row_to_dict(row)
                row_dict["dataset"] = ds_name
                row_dict["seed"] = seed
                all_run_records.append(row_dict)

    metrics = ["NDCG", "Recall"]
    ks = [1, 5, 10]
    grouped = defaultdict(lambda: defaultdict(list))

    for row in all_run_records:
        algo = row.get("algorithm", row.get("algo", row.get("model", "unknown")))
        dataset = row.get("dataset", "unknown")
        for metric in metrics:
            for k in ks:
                val = _find_metric_value(row, metric, k)
                grouped[(dataset, algo)][f"{metric}@{k}"].append(val)

    print("\n=== Summary across seeds ===")
    for (dataset, algo), metric_map in sorted(grouped.items()):
        print(f"\nDataset: {dataset} | Algorithm: {algo}")
        for metric_name in [f"NDCG@{k}" for k in ks] + [f"Recall@{k}" for k in ks]:
            summary = _summarize(metric_map.get(metric_name, []))
            print(
                f"  {metric_name}: mean={summary['mean']} std={summary['std']} min={summary['min']} max={summary['max']} n={summary['n']}"
            )

    print("\n=== Short statistical analysis ===")
    print("1) Seed sensitivity can be assessed by the across-seed standard deviation for each dataset-algorithm pair.")
    print("2) Lower std indicates more stable ranking performance under different user-holdout splits.")
    print("3) Compare ALS/ItemKNN/Pop within each dataset to identify which method is most sensitive to split randomness.")
    print("4) If one method has consistently higher mean and lower std, it is both stronger and more robust to split seeds.")
