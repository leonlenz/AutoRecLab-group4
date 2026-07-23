import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.base import Preprocessor
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

SEEDS = [7, 13, 23, 37, 53]
K_VALUES = [1, 5, 10]
DEFAULT_DATASETS = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]


class MergeValidationIntoTrain(Preprocessor):
    def __init__(self):
        super().__init__()

    def _process(self, dataset):
        data = dataset._data
        train_df = data.train.copy()
        val_df = data.val.copy()
        test_df = data.test.copy()

        merged_train = pd.concat([train_df, val_df], ignore_index=True)
        merged_train = merged_train.drop_duplicates()

        data.train = merged_train
        data.val = train_df.iloc[0:0].copy()
        data.test = test_df
        return dataset


class SaveSeedResults:
    def __init__(self, result_path):
        self.result_path = Path(result_path)

    def exists(self):
        return self.result_path.exists()

    def save(self, df):
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.result_path, index=False)

    def load(self):
        return pd.read_csv(self.result_path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_name(name):
    return str(name).replace(" ", "_").replace("/", "_")


def dataset_spec(name):
    mapping = {
        "MovieLens100K": (DataSet.MovieLens100K, 3),
        "Amazon2014VideoGames": (DataSet.Amazon2014VideoGames, 3),
        "HetrecLastFM": (DataSet.HetrecLastFM, None),
    }
    if name not in mapping:
        raise ValueError(f"Unknown dataset: {name}")
    return mapping[name]


def load_and_preprocess_dataset(dataset_name):
    dataset_enum, implicit_threshold = dataset_spec(dataset_name)
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.append(CorePruning(5))
    ds = Pipe(*steps).process(ds)
    return ds


def make_exact_user_80_20_holdout(dataset, seed, save_path=None):
    set_random_state(seed)
    split_ds = UserHoldout(validation_size=1, test_size=1).process(dataset)
    split_ds = MergeValidationIntoTrain().process(split_ds)
    if save_path is not None:
        split_ds.save(save_path)
    return split_ds


def build_plan():
    plan = ExperimentPlan(plan_name="seed_sensitivity_lenskit_baselines")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def build_evaluator():
    return Evaluator(NDCG(K_VALUES))


def normalize_algorithm_name(name):
    text = str(name)
    if text.startswith("LensKit.ImplicitMFScorer"):
        return "ALS"
    if text.startswith("LensKit.ItemKNNScorer"):
        return "ItemKNN"
    if text.startswith("LensKit.PopScorer"):
        return "Pop"
    return text


def flatten_results_dict(results_dict, dataset_alias, seed):
    frames = []
    for dataset_id, df in results_dict.items():
        tmp = df.copy()
        tmp["dataset_id"] = dataset_id
        tmp["dataset"] = dataset_alias
        tmp["seed"] = seed
        frames.append(tmp)
    if not frames:
        empty_columns: Sequence[str] = ["algorithm", "fold", "name", "k", "value", "dataset_id", "dataset", "seed"]
        return pd.DataFrame.from_records([], columns=empty_columns)
    return pd.concat(frames, ignore_index=True)


def extract_test_df(split_dataset):
    test_df = split_dataset._data.test.copy()
    cols = [c for c in ["user", "item", "rating"] if c in test_df.columns]
    out = test_df[cols].copy()
    if "rating" not in out.columns:
        out["rating"] = 1
    return out[["user", "item", "rating"]].drop_duplicates()


def find_prediction_files(checkpoint_root):
    root = Path(checkpoint_root)
    if not root.exists():
        return []
    return list(root.rglob("predictions.json"))


def load_predictions_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return pd.DataFrame(data["data"])
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def standardize_prediction_df(pred_df):
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    lower_to_orig = {c.lower(): c for c in out.columns}
    if "user" not in out.columns and "user_id" in lower_to_orig:
        out = out.rename(columns={lower_to_orig["user_id"]: "user"})
    if "item" not in out.columns and "item_id" in lower_to_orig:
        out = out.rename(columns={lower_to_orig["item_id"]: "item"})
    if "score" not in out.columns:
        if "prediction" in lower_to_orig:
            out = out.rename(columns={lower_to_orig["prediction"]: "score"})
        elif "value" in lower_to_orig:
            out = out.rename(columns={lower_to_orig["value"]: "score"})
    if "rank" not in out.columns and "rnk" in lower_to_orig:
        out = out.rename(columns={lower_to_orig["rnk"]: "rank"})
    needed = {"user", "item"}
    if not needed.issubset(set(out.columns)):
        return pd.DataFrame()
    return out


def precision_at_k(pred_df, test_df, k):
    preds = standardize_prediction_df(pred_df)
    if preds.empty:
        return np.nan

    rel = test_df[["user", "item"]].drop_duplicates().copy()
    rel["relevant"] = 1

    if "rank" in preds.columns:
        sort_cols = ["user", "rank"]
        ascending = [True, True]
        if "score" in preds.columns:
            sort_cols.append("score")
            ascending.append(False)
        preds = preds.sort_values(sort_cols, ascending=ascending)
    elif "score" in preds.columns:
        preds = preds.sort_values(["user", "score"], ascending=[True, False])
    else:
        preds = preds.sort_values(["user", "item"])

    topk = preds.groupby("user", as_index=False, group_keys=False).head(k).copy()
    topk = topk.merge(rel, on=["user", "item"], how="left")
    topk["relevant"] = topk["relevant"].fillna(0)
    user_prec = topk.groupby("user")["relevant"].sum() / float(k)
    if len(user_prec) == 0:
        return np.nan
    return float(user_prec.mean())


def collect_precision_metrics(split_dataset, checkpoint_root, completed_algorithms):
    test_df = extract_test_df(split_dataset)
    pred_files = find_prediction_files(checkpoint_root)
    metrics_by_algo = {algo: {f"Precision@{k}": np.nan for k in K_VALUES} for algo in completed_algorithms}

    for pred_file in pred_files:
        pred_df = standardize_prediction_df(load_predictions_json(pred_file))
        if pred_df.empty:
            continue
        path_text = str(pred_file)
        algo_match = None
        for algo in completed_algorithms:
            if algo == "ALS" and "ImplicitMFScorer" in path_text:
                algo_match = algo
                break
            if algo == "ItemKNN" and "ItemKNNScorer" in path_text:
                algo_match = algo
                break
            if algo == "Pop" and "PopScorer" in path_text:
                algo_match = algo
                break
        if algo_match is None:
            continue
        for k in K_VALUES:
            metrics_by_algo[algo_match][f"Precision@{k}"] = precision_at_k(pred_df, test_df, k)
    return metrics_by_algo


def aggregate_seed_results(all_metric_rows):
    grouped = all_metric_rows.groupby(["dataset", "algorithm", "metric"], as_index=False).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        min=("value", "min"),
        max=("value", "max"),
        n=("value", "count"),
    )
    grouped["range"] = grouped["max"] - grouped["min"]
    grouped["cv"] = grouped["std"] / grouped["mean"].replace(0, np.nan)
    return grouped.sort_values(["dataset", "algorithm", "metric"]).reset_index(drop=True)


def summarize_seed_sensitivity(summary_df):
    lines = []
    for dataset in sorted(summary_df["dataset"].unique()):
        sub = summary_df[(summary_df["dataset"] == dataset) & (summary_df["metric"].isin(["NDCG@10", "Precision@10"]))]
        if sub.empty:
            continue
        lines.append(f"Dataset: {dataset}")
        for metric in ["NDCG@10", "Precision@10"]:
            msub = sub[sub["metric"] == metric].sort_values("std", ascending=False)
            if msub.empty:
                continue
            most = msub.iloc[0]
            least = msub.iloc[-1]
            lines.append(
                f"  {metric}: most seed-sensitive={most['algorithm']} (std={most['std']:.6f}, range={most['range']:.6f}); least seed-sensitive={least['algorithm']} (std={least['std']:.6f}, range={least['range']:.6f})"
            )
    return "\n".join(lines)


def metric_rows_from_ndcg(ndcg_df, dataset_alias, seed):
    rows = []
    completed_algorithms = []
    for _, row in ndcg_df.iterrows():
        algo_name = normalize_algorithm_name(row["algorithm"])
        if algo_name not in completed_algorithms:
            completed_algorithms.append(algo_name)
        rows.append(
            {
                "dataset": dataset_alias,
                "seed": seed,
                "algorithm": algo_name,
                "metric": f"{row['name']}@{int(row['k'])}" if pd.notna(row["k"]) else str(row["name"]),
                "value": float(row["value"]),
            }
        )
    return rows, completed_algorithms


def run_seed(dataset_alias, base_dataset, seed, working_dir, results_dir, splits_dir):
    seed_result_path = Path(results_dir) / f"{sanitize_name(dataset_alias)}_seed{seed}.csv"
    cache = SaveSeedResults(seed_result_path)
    if cache.exists():
        print(f"Skipping completed dataset={dataset_alias}, seed={seed}: {seed_result_path}")
        return cache.load()

    print(f"\n--- Running dataset={dataset_alias}, seed={seed} ---")
    split_path = os.path.join(splits_dir, f"{sanitize_name(dataset_alias)}_seed{seed}.rsds")
    split_dataset = make_exact_user_80_20_holdout(base_dataset, seed, save_path=split_path)

    plan = build_plan()
    evaluator = build_evaluator()
    run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)

    ndcg_df = flatten_results_dict(evaluator.get_results(), dataset_alias, seed)
    if ndcg_df.empty:
        raise RuntimeError(f"No evaluator results produced for dataset={dataset_alias}, seed={seed}")

    metric_rows, completed_algorithms = metric_rows_from_ndcg(ndcg_df, dataset_alias, seed)
    precision_metrics = collect_precision_metrics(split_dataset, working_dir, completed_algorithms)
    for algo_name, algo_metrics in precision_metrics.items():
        for metric_name, metric_value in algo_metrics.items():
            metric_rows.append(
                {
                    "dataset": dataset_alias,
                    "seed": seed,
                    "algorithm": algo_name,
                    "metric": metric_name,
                    "value": float(metric_value),
                }
            )

    run_df = pd.DataFrame(metric_rows)
    cache.save(run_df)
    print(run_df.sort_values(["algorithm", "metric"]).to_string(index=False))
    return run_df


def load_all_seed_csvs(results_dir):
    files = sorted(Path(results_dir).glob("*_seed*.csv"))
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    if not frames:
        empty_columns: Sequence[str] = ["dataset", "seed", "algorithm", "metric", "value"]
        return pd.DataFrame.from_records([], columns=empty_columns)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    working_dir = os.path.join(os.getcwd(), "working")
    ensure_dir(working_dir)
    results_dir = ensure_dir(os.path.join(working_dir, "results"))
    splits_dir = ensure_dir(os.path.join(working_dir, "splits"))
    os.chdir(working_dir)

    for dataset_alias in args.datasets:
        print(f"\n=== Loading and preprocessing {dataset_alias} ===")
        base_dataset = load_and_preprocess_dataset(dataset_alias)
        print(base_dataset)
        for seed in args.seeds:
            run_seed(dataset_alias, base_dataset, seed, working_dir, results_dir, splits_dir)

    all_runs_df = load_all_seed_csvs(results_dir)
    if all_runs_df.empty:
        raise RuntimeError("No experiment results were collected.")

    summary_df = aggregate_seed_results(all_runs_df)
    all_runs_csv = os.path.join(results_dir, "seed_sensitivity_all_runs.csv")
    summary_csv = os.path.join(results_dir, "seed_sensitivity_summary.csv")
    all_runs_df.to_csv(all_runs_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Mean and variability across available completed seeds ===")
    print(summary_df.to_string(index=False))

    print("\n=== Concise seed-sensitivity analysis ===")
    print(summarize_seed_sensitivity(summary_df))

    completed = all_runs_df.groupby(["dataset", "algorithm", "metric"])["seed"].nunique().reset_index(name="n_seeds")
    incomplete = completed[completed["n_seeds"] < len(SEEDS)]
    if not incomplete.empty:
        print("\nWARNING: Some dataset/algorithm/metric combinations do not yet have all 5 seeds completed.")
        print(incomplete.to_string(index=False))
        print("Rerun the script with the missing datasets/seeds; OmniRec will resume from checkpoints and the script will skip saved per-seed CSVs.")

    print(f"\nSaved per-run results to: {all_runs_csv}")
    print(f"Saved summary results to: {summary_csv}")


if __name__ == "__main__":
    main()
