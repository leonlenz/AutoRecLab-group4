import os
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


KS = [1, 5, 10]
SEEDS = [11, 29, 47, 71, 97]


def load_base_dataset(dataset_key: str) -> RecSysDataSet:
    mapping = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }
    if dataset_key not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_key}")
    return RecSysDataSet.use_dataloader(mapping[dataset_key])


def preprocess_dataset(dataset_key: str) -> RecSysDataSet:
    dataset = load_base_dataset(dataset_key)
    steps = []
    if dataset_key in {"MovieLens100K", "Amazon2014VideoGames"}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    processed = pipe.process(dataset)
    return processed


def split_dataset_for_seed(dataset: RecSysDataSet, seed: int) -> RecSysDataSet:
    set_random_state(seed)
    splitter = UserHoldout(validation_size=0.0, test_size=0.2)
    return splitter.process(dataset)


def make_plan(plan_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name=plan_name)
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer, {"feedback": "implicit"})
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def dataset_id_from_results(results: Dict[str, pd.DataFrame]) -> str:
    if len(results) != 1:
        raise RuntimeError(f"Expected exactly one dataset result, found {len(results)}")
    return next(iter(results.keys()))


def sanitize_algo_name(algo: str) -> str:
    if algo.startswith("LensKit.ImplicitMFScorer"):
        return "ALS"
    if algo.startswith("LensKit.ItemKNNScorer"):
        return "ItemKNN"
    if algo.startswith("LensKit.PopScorer"):
        return "Pop"
    return algo


def find_predictions_file(checkpoints_dir: Path, dataset_id: str, algorithm_id: str) -> Path:
    ds_dir = checkpoints_dir / dataset_id
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset checkpoint directory not found: {ds_dir}")
    algo_dir = ds_dir / algorithm_id
    if not algo_dir.exists():
        raise FileNotFoundError(f"Algorithm checkpoint directory not found: {algo_dir}")
    pred_file = algo_dir / "predictions.json"
    if pred_file.exists():
        return pred_file
    fold_pred = algo_dir / "fold_0" / "predictions.json"
    if fold_pred.exists():
        return fold_pred
    matches = list(algo_dir.rglob("predictions.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not locate predictions.json under {algo_dir}")


def load_predictions(pred_file: Path) -> pd.DataFrame:
    try:
        df = pd.read_json(pred_file)
    except ValueError:
        df = pd.read_json(pred_file, lines=True)
    required = {"user", "item", "rank"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Predictions file {pred_file} missing columns: {missing}")
    return df


def get_split_frames(split_ds: RecSysDataSet) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = split_ds._data.get("train")
    val_df = split_ds._data.get("val")
    test_df = split_ds._data.get("test")
    if train_df is None or test_df is None:
        raise RuntimeError("Split dataset is missing train/test frames")
    if val_df is None:
        val_df = pd.DataFrame(columns=train_df.columns)
    return train_df.copy(), val_df.copy(), test_df.copy()


def precision_at_k(preds: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    test_items = test_df.groupby("user")["item"].apply(set).to_dict()
    ranked = preds.sort_values(["user", "rank", "score"], ascending=[True, True, False])
    ranked = ranked.groupby("user").head(k)
    per_user = []
    for user, grp in ranked.groupby("user"):
        rel = test_items.get(user, set())
        if len(rel) == 0:
            continue
        hit_count = grp["item"].isin(rel).sum()
        per_user.append(hit_count / float(k))
    if not per_user:
        return float("nan")
    return float(np.mean(per_user))


def coefficient_of_variation(values: pd.Series) -> float:
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if mean == 0:
        return float("nan")
    return std / mean


def eta_squared_seed_effect(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_seed = ss_total
    return ss_seed / ss_total


def run_one_condition(dataset_key: str, seed: int, base_processed: RecSysDataSet, working_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_ds = split_dataset_for_seed(base_processed, seed)
    train_df, val_df, test_df = get_split_frames(split_ds)

    split_export = working_dir / "splits" / dataset_key / f"seed_{seed}"
    split_export.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_export / "train.csv", index=False)
    val_df.to_csv(split_export / "val.csv", index=False)
    test_df.to_csv(split_export / "test.csv", index=False)

    run_dir = working_dir / "omnirec_runs" / dataset_key / f"seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        plan = make_plan(plan_name=f"seed_sensitivity_{dataset_key}_{seed}")
        evaluator = Evaluator(NDCG(KS))
        print(f"Running dataset={dataset_key}, seed={seed}")
        run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
        results = evaluator.get_results()
        dataset_id = dataset_id_from_results(results)
        raw_metrics = results[dataset_id].copy()
    finally:
        os.chdir(cwd)

    raw_metrics["dataset"] = dataset_key
    raw_metrics["seed"] = seed
    raw_metrics["algorithm_short"] = raw_metrics["algorithm"].map(sanitize_algo_name)

    checkpoints_dir = run_dir / "checkpoints"
    precision_rows = []
    for algorithm_id in raw_metrics["algorithm"].drop_duplicates().tolist():
        pred_file = find_predictions_file(checkpoints_dir, dataset_id, algorithm_id)
        preds = load_predictions(pred_file)
        algo_short = sanitize_algo_name(algorithm_id)
        for k in KS:
            p_at_k = precision_at_k(preds, test_df, k)
            precision_rows.append(
                {
                    "dataset": dataset_key,
                    "seed": seed,
                    "algorithm": algorithm_id,
                    "algorithm_short": algo_short,
                    "fold": None,
                    "name": "Precision",
                    "k": k,
                    "value": p_at_k,
                }
            )

    precision_df = pd.DataFrame(precision_rows)
    raw_metrics = raw_metrics[["dataset", "seed", "algorithm", "algorithm_short", "fold", "name", "k", "value"]]
    combined = pd.concat([raw_metrics, precision_df], ignore_index=True)
    combined = combined.sort_values(["dataset", "seed", "algorithm_short", "name", "k"]).reset_index(drop=True)

    split_stats = pd.DataFrame(
        [
            {
                "dataset": dataset_key,
                "seed": seed,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_users_train": train_df["user"].nunique(),
                "n_users_test": test_df["user"].nunique(),
                "n_items_train": train_df["item"].nunique(),
                "n_items_test": test_df["item"].nunique(),
            }
        ]
    )

    return combined, split_stats, raw_metrics


def aggregate_results(per_run: pd.DataFrame) -> pd.DataFrame:
    grouped = per_run.groupby(["dataset", "algorithm_short", "name", "k"], dropna=False)["value"]
    agg = grouped.agg(["mean", "std", "min", "max", "median", "count"]).reset_index()
    agg["cv"] = grouped.apply(lambda s: coefficient_of_variation(s)).values
    return agg.sort_values(["dataset", "algorithm_short", "name", "k"]).reset_index(drop=True)


def seed_variability_analysis(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = per_run.groupby(["dataset", "algorithm_short", "name", "k"], dropna=False)
    for key, grp in grouped.__iter__():
        dataset, algorithm, metric, k = cast(Tuple[object, object, object, object], key)
        vals = grp["value"].dropna().astype(float)
        if len(vals) == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        value_range = float(vals.max() - vals.min()) if len(vals) > 0 else float("nan")
        cv = coefficient_of_variation(vals)
        eta2 = eta_squared_seed_effect(vals)
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "metric": metric,
                "k": k,
                "mean": mean,
                "std": std,
                "range": value_range,
                "cv": cv,
                "eta_squared_seed": eta2,
                "n_seeds": int(len(vals)),
                "min_seed_value": float(vals.min()),
                "max_seed_value": float(vals.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric", "k"]).reset_index(drop=True)


def summarize_preprocessing(working_dir: Path, dataset_key: str, processed: RecSysDataSet) -> pd.DataFrame:
    raw_df = processed._data.df.copy()
    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_key,
                "n_interactions": len(raw_df),
                "n_users": raw_df["user"].nunique(),
                "n_items": raw_df["item"].nunique(),
                "min_user_interactions": int(raw_df.groupby("user").size().min()),
                "min_item_interactions": int(raw_df.groupby("item").size().min()),
                "rating_unique_values": ",".join(map(str, sorted(pd.unique(raw_df["rating"])))) if "rating" in raw_df.columns else "",
            }
        ]
    )
    out = working_dir / "preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"{dataset_key}_post_preprocessing_summary.csv", index=False)
    return summary


def print_table(df: pd.DataFrame, title: str, max_rows: int = 20) -> None:
    print(f"\n=== {title} ===")
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)} rows total)")
    else:
        print(df.to_string(index=False))


def main() -> None:
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    working_dir = Path(working_dir)

    datasets = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]

    all_per_run = []
    all_split_stats = []
    preprocessing_summaries = []

    processed_cache: Dict[str, RecSysDataSet] = {}
    for dataset_key in datasets:
        print(f"Loading and preprocessing {dataset_key}")
        processed_cache[dataset_key] = preprocess_dataset(dataset_key)
        preprocessing_summaries.append(summarize_preprocessing(working_dir, dataset_key, processed_cache[dataset_key]))

    for dataset_key in datasets:
        for seed in SEEDS:
            per_run_df, split_stats_df, _ = run_one_condition(
                dataset_key=dataset_key,
                seed=seed,
                base_processed=processed_cache[dataset_key],
                working_dir=working_dir,
            )
            all_per_run.append(per_run_df)
            all_split_stats.append(split_stats_df)
            monitor = per_run_df.pivot_table(index=["dataset", "seed", "algorithm_short"], columns=["name", "k"], values="value")
            print_table(monitor.reset_index(), f"Metrics for {dataset_key} seed {seed}", max_rows=10)

    per_run = pd.concat(all_per_run, ignore_index=True)
    split_stats = pd.concat(all_split_stats, ignore_index=True)
    preprocessing_summary = pd.concat(preprocessing_summaries, ignore_index=True)
    aggregated = aggregate_results(per_run)
    variability = seed_variability_analysis(per_run)

    results_dir = working_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(results_dir / "per_run_results.csv", index=False)
    aggregated.to_csv(results_dir / "aggregated_results.csv", index=False)
    variability.to_csv(results_dir / "seed_variability_analysis.csv", index=False)
    split_stats.to_csv(results_dir / "split_stats.csv", index=False)
    preprocessing_summary.to_csv(results_dir / "preprocessing_summary.csv", index=False)

    compact_agg = aggregated.copy()
    compact_agg["metric"] = compact_agg["name"] + "@" + compact_agg["k"].astype(int).astype(str)
    compact_agg = compact_agg[["dataset", "algorithm_short", "metric", "mean", "std", "cv", "min", "max", "count"]]

    compact_var = variability.copy()
    compact_var["metric"] = compact_var["metric"] + "@" + compact_var["k"].astype(int).astype(str)
    compact_var = compact_var[["dataset", "algorithm", "metric", "mean", "std", "range", "cv", "eta_squared_seed", "n_seeds"]]

    print_table(preprocessing_summary, "Post-preprocessing dataset summaries")
    print_table(split_stats, "Per-seed split statistics", max_rows=30)
    print_table(per_run, "Per-run results", max_rows=40)
    print_table(compact_agg, "Aggregated results over seeds", max_rows=60)
    print_table(compact_var, "Across-seed variability analysis", max_rows=60)

    print("\nArtifacts written to:")
    print(results_dir)
    print("- per_run_results.csv")
    print("- aggregated_results.csv")
    print("- seed_variability_analysis.csv")
    print("- split_stats.csv")
    print("- preprocessing_summary.csv")


if __name__ == '__main__':
    main()
