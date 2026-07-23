import os
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, cast

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
VALIDATION_SIZE = 0.01
TEST_SIZE = 0.20


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
    return pipe.process(dataset)


def get_raw_df(dataset: RecSysDataSet) -> pd.DataFrame:
    data = getattr(dataset, "_data", None)
    if data is None or not hasattr(data, "df"):
        raise RuntimeError("Expected raw dataset with a DataFrame-backed data variant.")
    return data.df.copy()


def split_dataset_for_seed(dataset: RecSysDataSet, seed: int) -> RecSysDataSet:
    set_random_state(seed)
    splitter = UserHoldout(validation_size=VALIDATION_SIZE, test_size=TEST_SIZE)
    return splitter.process(dataset)


def get_split_frames(split_ds: RecSysDataSet) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = getattr(split_ds, "_data", None)
    if data is None:
        raise RuntimeError("Split dataset has no data payload.")

    train_df = None
    val_df = None
    test_df = None

    if hasattr(data, "get"):
        try:
            train_df = data.get("train")
            val_df = data.get("val")
            test_df = data.get("test")
        except Exception:
            train_df = None
            val_df = None
            test_df = None

    if train_df is None and hasattr(data, "train"):
        train_df = data.train
    if val_df is None and hasattr(data, "val"):
        val_df = data.val
    if test_df is None and hasattr(data, "test"):
        test_df = data.test

    if train_df is None or test_df is None:
        raise RuntimeError("Could not retrieve train/test splits from OmniRec split dataset.")
    if val_df is None:
        val_df = pd.DataFrame(columns=train_df.columns)

    return train_df.copy(), val_df.copy(), test_df.copy()


def make_plan(plan_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name=plan_name)
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
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

    candidates = [
        algo_dir / "predictions.json",
        algo_dir / "fold_0" / "predictions.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(algo_dir.rglob("predictions.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches = sorted(matches)
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
        raise ValueError(f"Predictions file {pred_file} missing required columns: {missing}")
    return df.copy()


def precision_at_k(preds: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    test_items = test_df.groupby("user")["item"].apply(set).to_dict()

    sort_cols = [c for c in ["user", "rank", "score"] if c in preds.columns]
    ascending = []
    for c in sort_cols:
        if c in {"user", "rank"}:
            ascending.append(True)
        else:
            ascending.append(False)
    ranked = preds.sort_values(sort_cols, ascending=ascending)
    ranked = ranked.groupby("user", sort=False).head(k)

    per_user = []
    for user, grp in ranked.groupby("user", sort=False):
        rel = test_items.get(user, set())
        if not rel:
            continue
        hit_count = int(grp["item"].isin(rel).sum())
        per_user.append(hit_count / float(k))

    if not per_user:
        return float("nan")
    return float(np.mean(per_user))


def coefficient_of_variation(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(vals) == 0:
        return float("nan")
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if mean == 0.0:
        return float("nan")
    return std / mean


def summarize_preprocessing(dataset_key: str, processed: RecSysDataSet) -> pd.DataFrame:
    raw_df = get_raw_df(processed)
    summary = {
        "dataset": dataset_key,
        "n_interactions": len(raw_df),
        "n_users": int(raw_df["user"].nunique()),
        "n_items": int(raw_df["item"].nunique()),
        "min_user_interactions": int(raw_df.groupby("user").size().min()),
        "min_item_interactions": int(raw_df.groupby("item").size().min()),
    }
    if "rating" in raw_df.columns:
        unique_vals = sorted(pd.unique(raw_df["rating"]))
        summary["rating_unique_values"] = ",".join(map(str, unique_vals))
    else:
        summary["rating_unique_values"] = ""
    return pd.DataFrame([summary])


def print_table(df: pd.DataFrame, title: str, max_rows: int = 20) -> None:
    print(f"\n=== {title} ===")
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)} rows total)")
    else:
        print(df.to_string(index=False))


def run_one_condition(dataset_key: str, seed: int, base_processed: RecSysDataSet, working_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        print(
            f"Running dataset={dataset_key}, seed={seed}, split=train/val/test ~= "
            f"{1.0 - TEST_SIZE - VALIDATION_SIZE:.2f}/{VALIDATION_SIZE:.2f}/{TEST_SIZE:.2f}"
        )
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
                "train_share": len(train_df) / max(len(train_df) + len(val_df) + len(test_df), 1),
                "val_share": len(val_df) / max(len(train_df) + len(val_df) + len(test_df), 1),
                "test_share": len(test_df) / max(len(train_df) + len(val_df) + len(test_df), 1),
                "n_users_train": int(train_df["user"].nunique()),
                "n_users_val": int(val_df["user"].nunique()) if len(val_df) else 0,
                "n_users_test": int(test_df["user"].nunique()),
                "n_items_train": int(train_df["item"].nunique()),
                "n_items_test": int(test_df["item"].nunique()),
            }
        ]
    )

    return combined, split_stats


def aggregate_results(per_run: pd.DataFrame) -> pd.DataFrame:
    grouped = per_run.groupby(["dataset", "algorithm_short", "name", "k"], dropna=False)["value"]
    agg = grouped.agg(["mean", "std", "min", "max", "median", "count"]).reset_index()
    agg["cv"] = grouped.apply(coefficient_of_variation).values
    return agg.sort_values(["dataset", "algorithm_short", "name", "k"]).reset_index(drop=True)


def paired_seed_differences(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = per_run.groupby(["dataset", "name", "k"], dropna=False)
    for group_key in grouped.groups.keys():
        dataset, metric, k = cast(Tuple[object, object, object], group_key if isinstance(group_key, tuple) else (group_key,))
        grp = grouped.get_group(group_key)
        pivot = grp.pivot_table(index="seed", columns="algorithm_short", values="value", aggfunc="mean")
        algos = [c for c in ["ALS", "ItemKNN", "Pop"] if c in pivot.columns]
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                a = algos[i]
                b = algos[j]
                paired = pivot[[a, b]].dropna()
                if len(paired) == 0:
                    continue
                diff = paired[a] - paired[b]
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "k": k,
                        "algo_a": a,
                        "algo_b": b,
                        "mean_diff": float(diff.mean()),
                        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                        "min_diff": float(diff.min()),
                        "max_diff": float(diff.max()),
                        "n_pairs": int(len(diff)),
                    }
                )
    if not rows:
        return pd.DataFrame(
            {
                "dataset": [],
                "metric": [],
                "k": [],
                "algo_a": [],
                "algo_b": [],
                "mean_diff": [],
                "std_diff": [],
                "min_diff": [],
                "max_diff": [],
                "n_pairs": [],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "k", "algo_a", "algo_b"]).reset_index(drop=True)


def seed_variability_analysis(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = per_run.groupby(["dataset", "algorithm_short", "name", "k"], dropna=False)
    for group_key in grouped.groups.keys():
        dataset, algorithm, metric, k = cast(
            Tuple[object, object, object, object],
            group_key if isinstance(group_key, tuple) else (group_key,),
        )
        grp = grouped.get_group(group_key)
        vals = pd.to_numeric(grp["value"], errors="coerce").dropna().astype(float)
        if len(vals) == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        value_range = float(vals.max() - vals.min())
        cv = coefficient_of_variation(vals)
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
                "n_seeds": int(len(vals)),
                "min_seed_value": float(vals.min()),
                "max_seed_value": float(vals.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric", "k"]).reset_index(drop=True)


def write_short_analysis(aggregated: pd.DataFrame, variability: pd.DataFrame, pairwise: pd.DataFrame, out_file: Path) -> None:
    lines: List[str] = []
    lines.append("Seed-sensitivity analysis summary")
    lines.append("================================")
    lines.append("")
    lines.append(
        "This experiment uses OmniRec's LensKit wrappers for ALS (ImplicitMFScorer), ItemKNNScorer, and PopScorer. "
        "Because OmniRec's documented UserHoldout requires a positive validation split, runs use validation_size=0.01 and test_size=0.20; metrics are reported on the test split."
    )
    lines.append("")
    lines.append("Across-seed variability highlights (lower CV/range indicates more stability):")
    if len(variability) == 0:
        lines.append("- No variability results available.")
    else:
        top_stable = variability.sort_values(["cv", "range"], na_position="last").head(9)
        for _, row in top_stable.iterrows():
            lines.append(
                f"- Stable: {row['dataset']} | {row['algorithm']} | {row['metric']}@{int(row['k'])} "
                f"mean={row['mean']:.4f}, std={row['std']:.4f}, cv={row['cv']:.4f}, range={row['range']:.4f}"
            )
        lines.append("")
        top_sensitive = variability.sort_values(["cv", "range"], ascending=[False, False], na_position="last").head(9)
        for _, row in top_sensitive.iterrows():
            lines.append(
                f"- Sensitive: {row['dataset']} | {row['algorithm']} | {row['metric']}@{int(row['k'])} "
                f"mean={row['mean']:.4f}, std={row['std']:.4f}, cv={row['cv']:.4f}, range={row['range']:.4f}"
            )
    lines.append("")
    lines.append("Pairwise seed-matched algorithm differences:")
    if len(pairwise) == 0:
        lines.append("- No pairwise comparisons available.")
    else:
        best = pairwise.sort_values("mean_diff", ascending=False).head(12)
        for _, row in best.iterrows():
            lines.append(
                f"- {row['dataset']} | {row['metric']}@{int(row['k'])}: {row['algo_a']} - {row['algo_b']} "
                f"mean_diff={row['mean_diff']:.4f}, std_diff={row['std_diff']:.4f}, n={int(row['n_pairs'])}"
            )
    lines.append("")
    lines.append("Interpretation guideline:")
    lines.append(
        "If the across-seed standard deviation and coefficient of variation are small relative to mean differences between algorithms, the algorithm ranking is robust to random splitting. "
        "If they are large, conclusions about superiority should be treated cautiously and reported with uncertainty."
    )
    out_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    working_dir = Path(working_dir)

    datasets = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]

    all_per_run: List[pd.DataFrame] = []
    all_split_stats: List[pd.DataFrame] = []
    preprocessing_summaries: List[pd.DataFrame] = []
    processed_cache: Dict[str, RecSysDataSet] = {}

    print("Using OmniRec-only pipeline.")
    print(
        f"UserHoldout requires positive validation_size in OmniRec; using validation_size={VALIDATION_SIZE} and test_size={TEST_SIZE}."
    )

    for dataset_key in datasets:
        print(f"Loading and preprocessing {dataset_key}")
        processed = preprocess_dataset(dataset_key)
        processed_cache[dataset_key] = processed
        preprocessing_summaries.append(summarize_preprocessing(dataset_key, processed))

    for dataset_key in datasets:
        for seed in SEEDS:
            per_run_df, split_stats_df = run_one_condition(
                dataset_key=dataset_key,
                seed=seed,
                base_processed=processed_cache[dataset_key],
                working_dir=working_dir,
            )
            all_per_run.append(per_run_df)
            all_split_stats.append(split_stats_df)

            monitor = per_run_df.pivot_table(
                index=["dataset", "seed", "algorithm_short"],
                columns=["name", "k"],
                values="value",
                aggfunc="mean",
            )
            print_table(monitor.reset_index(), f"Metrics for {dataset_key} seed {seed}", max_rows=10)

    per_run = pd.concat(all_per_run, ignore_index=True)
    split_stats = pd.concat(all_split_stats, ignore_index=True)
    preprocessing_summary = pd.concat(preprocessing_summaries, ignore_index=True)
    aggregated = aggregate_results(per_run)
    variability = seed_variability_analysis(per_run)
    pairwise = paired_seed_differences(per_run)

    results_dir = working_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_run.to_csv(results_dir / "per_run_results.csv", index=False)
    aggregated.to_csv(results_dir / "aggregated_results.csv", index=False)
    variability.to_csv(results_dir / "seed_variability_analysis.csv", index=False)
    pairwise.to_csv(results_dir / "pairwise_seed_differences.csv", index=False)
    split_stats.to_csv(results_dir / "split_stats.csv", index=False)
    preprocessing_summary.to_csv(results_dir / "preprocessing_summary.csv", index=False)
    write_short_analysis(aggregated, variability, pairwise, results_dir / "short_statistical_analysis.txt")

    compact_agg = aggregated.copy()
    compact_agg["metric"] = compact_agg["name"] + "@" + compact_agg["k"].astype(int).astype(str)
    compact_agg = compact_agg[["dataset", "algorithm_short", "metric", "mean", "std", "cv", "min", "max", "count"]]

    compact_var = variability.copy()
    compact_var["metric_at_k"] = compact_var["metric"] + "@" + compact_var["k"].astype(int).astype(str)
    compact_var = compact_var[["dataset", "algorithm", "metric_at_k", "mean", "std", "range", "cv", "n_seeds"]]

    print_table(preprocessing_summary, "Post-preprocessing dataset summaries")
    print_table(split_stats, "Per-seed split statistics", max_rows=30)
    print_table(per_run, "Per-run results", max_rows=40)
    print_table(compact_agg, "Aggregated results over seeds", max_rows=60)
    print_table(compact_var, "Across-seed variability analysis", max_rows=60)
    print_table(pairwise, "Pairwise seed-matched algorithm differences", max_rows=60)

    print("\nArtifacts written to:")
    print(results_dir)
    print("- per_run_results.csv")
    print("- aggregated_results.csv")
    print("- seed_variability_analysis.csv")
    print("- pairwise_seed_differences.csv")
    print("- split_stats.csv")
    print("- preprocessing_summary.csv")
    print("- short_statistical_analysis.txt")


if __name__ == "__main__":
    main()
