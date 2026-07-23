import os
import json
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.metrics.ranking import Precision
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_dataset(dataset_name):
    mapping = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }
    return RecSysDataSet.use_dataloader(mapping[dataset_name])


def preprocess_dataset(dataset_name, ds):
    ds = CorePruning(5).process(ds)
    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        ds = MakeImplicit(3).process(ds)
    return ds


def can_user_holdout(ds, min_interactions_per_user=3):
    df = ds._data.df
    return int(df.groupby("user").size().min()) >= min_interactions_per_user


def build_plan():
    plan = ExperimentPlan("Seed-Sensitivity-Study")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def normalize_results(df):
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        lc = str(col).lower()
        if lc == "algorithm":
            rename_map[col] = "algorithm"
        elif lc in {"name", "metric_name"}:
            rename_map[col] = "metric"
        elif lc == "k":
            rename_map[col] = "k"
        elif lc in {"value", "score"}:
            rename_map[col] = "value"
    df = df.rename(columns=rename_map)
    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).replace({"NDCG": "nDCG", "Precision": "Precision"})
    return df


def run_single_seed(seed, dataset_name):
    set_random_state(seed)
    ds = load_dataset(dataset_name)
    ds = preprocess_dataset(dataset_name, ds)

    if not can_user_holdout(ds):
        raise RuntimeError(
            f"Dataset {dataset_name} is too sparse after preprocessing for user-based 80/20 holdout. "
            f"Try revisiting preprocessing order or filtering strictness."
        )

    # Documented user-aware holdout; using 20% test and 20% validation gives a valid user-based split.
    # The original crash came from a too-small effective split on sparse users; the safety check above prevents it.
    ds = UserHoldout(validation_size=0.20, test_size=0.20).process(ds)

    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    print(f"Running seed={seed}, dataset={dataset_name} ...")
    run_omnirec(datasets=ds, plan=plan, evaluator=evaluator)

    results_map = evaluator.get_results()
    if not results_map:
        return pd.DataFrame()

    frames = []
    for _, rdf in results_map.items():
        rdf = rdf.copy()
        rdf["seed"] = seed
        rdf["dataset"] = dataset_name
        frames.append(rdf)
    return normalize_results(pd.concat(frames, ignore_index=True))


def short_statistical_analysis(results_df):
    rows = []
    required = {"dataset", "algorithm", "metric", "k", "value", "seed"}
    if not required.issubset(results_df.columns):
        return pd.DataFrame(rows)

    for (dataset_name, algo, metric, k), grp in results_df.groupby(["dataset", "algorithm", "metric", "k"], dropna=False):
        vals = grp["value"].astype(float).to_numpy()
        if len(vals) < 2:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        rows.append({
            "dataset": dataset_name,
            "algorithm": algo,
            "metric": metric,
            "k": int(k),
            "n_runs": int(len(vals)),
            "mean": mean,
            "std": std,
            "cv": float(std / mean) if mean != 0 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "range": float(np.max(vals) - np.min(vals)),
        })
    return pd.DataFrame(rows)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [7, 13, 21, 42, 84]
    datasets = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]

    all_rows = []
    skipped = []
    for seed in seeds:
        for dataset_name in datasets:
            try:
                res_df = run_single_seed(seed, dataset_name)
                if not res_df.empty:
                    print(res_df.to_string(index=False))
                    all_rows.append(res_df)
            except Exception as e:
                skipped.append({"seed": seed, "dataset": dataset_name, "error": str(e)})
                print(f"[WARN] Skipping seed={seed}, dataset={dataset_name}: {e}")

    if not all_rows:
        raise RuntimeError("No evaluation results were produced.")

    results_df = pd.concat(all_rows, ignore_index=True)
    results_path = os.path.join(working_dir, "seed_sensitivity_results.csv")
    results_df.to_csv(results_path, index=False)

    summary_rows = []
    for (dataset_name, algo, metric, k), grp in results_df.groupby(["dataset", "algorithm", "metric", "k"], dropna=False):
        vals = grp["value"].astype(float).to_numpy()
        summary_rows.append({
            "dataset": dataset_name,
            "algorithm": algo,
            "metric": metric,
            "k": int(k),
            "n_runs": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "cv": float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 and np.mean(vals) != 0 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(working_dir, "seed_sensitivity_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    analysis_df = short_statistical_analysis(results_df)
    analysis_path = os.path.join(working_dir, "seed_sensitivity_analysis.csv")
    analysis_df.to_csv(analysis_path, index=False)

    print("\nAggregate summary by dataset/algorithm/metric/k:")
    print(summary_df.to_string(index=False))
    print("\nShort statistical analysis:")
    print(analysis_df.to_string(index=False))

    if skipped:
        skipped_path = os.path.join(working_dir, "seed_sensitivity_skipped_runs.json")
        with open(skipped_path, "w") as f:
            json.dump(skipped, f, indent=2)
    else:
        skipped_path = None

    meta = {
        "datasets": datasets,
        "seeds": seeds,
        "split": {"type": "UserHoldout", "validation_size": 0.20, "test_size": 0.20},
        "preprocessing": {
            "core_pruning": 5,
            "implicit_threshold": 3,
            "implicit_datasets": ["MovieLens100K", "Amazon2014VideoGames"],
        },
        "algorithms": ["LensKit.ImplicitMFScorer", "LensKit.ItemKNNScorer", "LensKit.PopScorer"],
        "metrics": ["NDCG@1", "NDCG@5", "NDCG@10", "Precision@1", "Precision@5", "Precision@10"],
        "results_path": results_path,
        "summary_path": summary_path,
        "analysis_path": analysis_path,
        "skipped_path": skipped_path,
    }
    with open(os.path.join(working_dir, "experiment_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
