import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
VALIDATION_SIZE = 0.001
TEST_SIZE = 0.20
DATASETS = [
    "MovieLens100K",
    "Amazon2014VideoGames",
    "HetrecLastFM",
]


def load_base_dataset(dataset_key: str) -> RecSysDataSet:
    mapping = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }
    return RecSysDataSet.use_dataloader(mapping[dataset_key])


def preprocess_dataset(dataset_key: str) -> RecSysDataSet:
    dataset = load_base_dataset(dataset_key)
    steps = []
    if dataset_key in {"MovieLens100K", "Amazon2014VideoGames"}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(dataset)


def get_raw_df(dataset: RecSysDataSet) -> pd.DataFrame:
    data = dataset._data
    if not hasattr(data, "df"):
        raise RuntimeError("Expected raw dataset with df attribute after preprocessing.")
    return data.df.copy()


def split_dataset_for_seed(dataset: RecSysDataSet, seed: int) -> RecSysDataSet:
    set_random_state(seed)
    splitter = UserHoldout(validation_size=VALIDATION_SIZE, test_size=TEST_SIZE)
    return splitter.process(dataset)


def get_split_frames(split_ds: RecSysDataSet) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = split_ds._data.get("train")
    val_df = split_ds._data.get("val")
    test_df = split_ds._data.get("test")
    return train_df.copy(), val_df.copy(), test_df.copy()


def make_plan(plan_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name=plan_name)
    plan.add_algorithm(
        LensKit.ImplicitMFScorer,
        {
            "features": 32,
            "epochs": 10,
        },
    )
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            "max_nbrs": 50,
            "min_nbrs": 1,
        },
    )
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def dataset_id_from_results(results: Dict[str, pd.DataFrame]) -> str:
    if len(results) != 1:
        raise RuntimeError(f"Expected one dataset result, got {list(results.keys())}")
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
    algo_dir = checkpoints_dir / dataset_id / algorithm_id
    candidates = [
        algo_dir / "predictions.json",
        algo_dir / "fold_0" / "predictions.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    matches = sorted(algo_dir.rglob("predictions.json"))
    if not matches:
        raise FileNotFoundError(f"No predictions.json found under {algo_dir}")
    return matches[0]


def load_predictions(pred_file: Path) -> pd.DataFrame:
    try:
        df = pd.read_json(pred_file)
    except ValueError:
        df = pd.read_json(pred_file, lines=True)
    required = {"user", "item", "rank"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")
    return df.copy()


def precision_at_k(preds: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    test_items = test_df.groupby("user")["item"].apply(set).to_dict()
    sort_cols = [c for c in ["user", "rank", "score"] if c in preds.columns]
    ascending = [True if c in {"user", "rank"} else False for c in sort_cols]
    ranked = preds.sort_values(sort_cols, ascending=ascending).groupby("user", sort=False).head(k)
    per_user = []
    for user, grp in ranked.groupby("user", sort=False):
        rel = test_items.get(user, set())
        if not rel:
            continue
        per_user.append(float(grp["item"].isin(rel).sum()) / float(k))
    return float(np.mean(per_user)) if per_user else float("nan")


def coefficient_of_variation(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(vals) == 0:
        return float("nan")
    mean = float(vals.mean())
    if mean == 0.0:
        return float("nan")
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
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
        "has_rating_col": int("rating" in raw_df.columns),
    }
    return pd.DataFrame([summary])


def print_table(df: pd.DataFrame, title: str, max_rows: int = 20) -> None:
    print(f"\n=== {title} ===")
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)} rows total)")
    else:
        print(df.to_string(index=False))


def persist_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_condition_done(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def condition_done(path: Path) -> bool:
    return path.exists()


def run_one_condition(dataset_key: str, seed: int, base_processed: RecSysDataSet, working_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_ds = split_dataset_for_seed(base_processed, seed)
    train_df, val_df, test_df = get_split_frames(split_ds)

    split_export = working_dir / "splits" / dataset_key / f"seed_{seed}"
    split_export.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_export / "train.csv", index=False)
    val_df.to_csv(split_export / "val.csv", index=False)
    test_df.to_csv(split_export / "test.csv", index=False)

    run_dir = working_dir / "omnirec_runs" / dataset_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        plan = make_plan(plan_name=f"seed_sensitivity_{dataset_key}_{seed}")
        evaluator = Evaluator(NDCG(KS))
        print(
            f"Running dataset={dataset_key}, seed={seed}, split=train/val/test ~= "
            f"{1.0 - TEST_SIZE - VALIDATION_SIZE:.3f}/{VALIDATION_SIZE:.3f}/{TEST_SIZE:.2f}"
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
            precision_rows.append(
                {
                    "dataset": dataset_key,
                    "seed": seed,
                    "algorithm": algorithm_id,
                    "algorithm_short": algo_short,
                    "fold": None,
                    "name": "Precision",
                    "k": k,
                    "value": precision_at_k(preds, test_df, k),
                }
            )

    precision_df = pd.DataFrame(precision_rows)
    raw_metrics = raw_metrics[["dataset", "seed", "algorithm", "algorithm_short", "fold", "name", "k", "value"]]
    combined = pd.concat([raw_metrics, precision_df], ignore_index=True)
    combined = combined.sort_values(["dataset", "seed", "algorithm_short", "name", "k"]).reset_index(drop=True)

    split_stats = pd.DataFrame([
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
    ])

    return combined, split_stats


def aggregate_results(per_run: pd.DataFrame) -> pd.DataFrame:
    grouped = per_run.groupby(["dataset", "algorithm_short", "name", "k"], dropna=False)["value"]
    agg = grouped.agg(["mean", "std", "min", "max", "median", "count"]).reset_index()
    agg["cv"] = grouped.apply(coefficient_of_variation).values
    return agg.sort_values(["dataset", "algorithm_short", "name", "k"]).reset_index(drop=True)


def paired_seed_differences(per_run: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    grouped = per_run.groupby(["dataset", "name", "k"], dropna=False)
    for group_key, grp in grouped:
        if not isinstance(group_key, tuple):
            continue
        dataset, metric, k = group_key
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
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "k", "algo_a", "algo_b"]).reset_index(drop=True) if rows else pd.DataFrame()


def seed_variability_analysis(per_run: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    grouped = per_run.groupby(["dataset", "algorithm_short", "name", "k"], dropna=False)
    for group_key, grp in grouped:
        if not isinstance(group_key, tuple):
            continue
        dataset, algorithm, metric, k = group_key
        vals = pd.to_numeric(grp["value"], errors="coerce").dropna().astype(float)
        if len(vals) == 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "metric": metric,
                "k": k,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "range": float(vals.max() - vals.min()),
                "cv": coefficient_of_variation(vals),
                "n_seeds": int(len(vals)),
                "min_seed_value": float(vals.min()),
                "max_seed_value": float(vals.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric", "k"]).reset_index(drop=True) if rows else pd.DataFrame()


def write_short_analysis(aggregated: pd.DataFrame, variability: pd.DataFrame, pairwise: pd.DataFrame, out_file: Path) -> None:
    lines: List[str] = []
    lines.append("Seed-sensitivity analysis summary")
    lines.append("================================")
    lines.append("")
    lines.append(
        "This run uses OmniRec LensKit wrappers for ALS (ImplicitMFScorer), ItemKNNScorer, and PopScorer. "
        "OmniRec documents UserHoldout as a train/validation/test splitter, so this implementation uses a near-zero validation split (0.001) and test_size=0.20 to approximate the requested 80/20 user holdout while remaining within the documented OmniRec API."
    )
    lines.append("")
    lines.append("Across-seed variability highlights:")
    if len(variability) > 0:
        stable = variability.sort_values(["cv", "range"], na_position="last").head(9)
        for _, row in stable.iterrows():
            lines.append(
                f"- Stable: {row['dataset']} | {row['algorithm']} | {row['metric']}@{int(row['k'])} mean={row['mean']:.4f}, std={row['std']:.4f}, cv={row['cv']:.4f}, range={row['range']:.4f}"
            )
        lines.append("")
        sensitive = variability.sort_values(["cv", "range"], ascending=[False, False], na_position="last").head(9)
        for _, row in sensitive.iterrows():
            lines.append(
                f"- Sensitive: {row['dataset']} | {row['algorithm']} | {row['metric']}@{int(row['k'])} mean={row['mean']:.4f}, std={row['std']:.4f}, cv={row['cv']:.4f}, range={row['range']:.4f}"
            )
    else:
        lines.append("- No variability results available.")
    lines.append("")
    lines.append("Pairwise seed-matched algorithm differences:")
    if len(pairwise) > 0:
        best = pairwise.sort_values("mean_diff", ascending=False).head(12)
        for _, row in best.iterrows():
            lines.append(
                f"- {row['dataset']} | {row['metric']}@{int(row['k'])}: {row['algo_a']} - {row['algo_b']} mean_diff={row['mean_diff']:.4f}, std_diff={row['std_diff']:.4f}, n={int(row['n_pairs'])}"
            )
    else:
        lines.append("- No pairwise comparisons available.")
    out_file.write_text("\n".join(lines), encoding="utf-8")


def recompute_and_write_all(results_dir: Path) -> None:
    per_run_path = results_dir / "per_run_results.csv"
    split_path = results_dir / "split_stats.csv"
    prep_path = results_dir / "preprocessing_summary.csv"
    if not per_run_path.exists():
        return
    per_run = pd.read_csv(per_run_path)
    aggregated = aggregate_results(per_run)
    variability = seed_variability_analysis(per_run)
    pairwise = paired_seed_differences(per_run)
    aggregated.to_csv(results_dir / "aggregated_results.csv", index=False)
    variability.to_csv(results_dir / "seed_variability_analysis.csv", index=False)
    pairwise.to_csv(results_dir / "pairwise_seed_differences.csv", index=False)
    write_short_analysis(aggregated, variability, pairwise, results_dir / "short_statistical_analysis.txt")

    print_table(pd.read_csv(prep_path) if prep_path.exists() else pd.DataFrame(), "Post-preprocessing dataset summaries", max_rows=20)
    print_table(pd.read_csv(split_path) if split_path.exists() else pd.DataFrame(), "Per-seed split statistics", max_rows=30)
    print_table(per_run, "Per-run results", max_rows=40)
    print_table(aggregated, "Aggregated results over seeds", max_rows=60)
    print_table(variability, "Across-seed variability analysis", max_rows=60)
    print_table(pairwise, "Pairwise seed-matched algorithm differences", max_rows=60)


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    working_dir = Path(working_dir)
    results_dir = working_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = results_dir / "per_run_results.csv"
    split_stats_path = results_dir / "split_stats.csv"
    prep_summary_path = results_dir / "preprocessing_summary.csv"

    existing_per_run = load_existing(per_run_path)
    existing_split = load_existing(split_stats_path)
    existing_prep = load_existing(prep_summary_path)

    print("Using OmniRec-only pipeline.")
    print(
        f"OmniRec documents UserHoldout(validation_size, test_size) for train/val/test splitting; using validation_size={VALIDATION_SIZE} and test_size={TEST_SIZE}."
    )
    print("This script is resumable: completed dataset/seed conditions are skipped and prior checkpoints are preserved.")

    processed_cache: Dict[str, RecSysDataSet] = {}

    for dataset_key in DATASETS:
        print(f"Loading and preprocessing {dataset_key}")
        processed = preprocess_dataset(dataset_key)
        processed_cache[dataset_key] = processed
        if existing_prep.empty or dataset_key not in set(existing_prep.get("dataset", pd.Series(dtype=str)).astype(str)):
            prep_row = summarize_preprocessing(dataset_key, processed)
            existing_prep = pd.concat([existing_prep, prep_row], ignore_index=True)
            existing_prep = existing_prep.drop_duplicates(subset=["dataset"], keep="last")
            persist_csv(existing_prep, prep_summary_path)

    for dataset_key in DATASETS:
        for seed in SEEDS:
            done_file = results_dir / "done_flags" / dataset_key / f"seed_{seed}.json"
            if condition_done(done_file):
                print(f"Skipping completed condition dataset={dataset_key}, seed={seed}")
                continue
            try:
                per_run_df, split_stats_df = run_one_condition(
                    dataset_key=dataset_key,
                    seed=seed,
                    base_processed=processed_cache[dataset_key],
                    working_dir=working_dir,
                )
            except Exception as e:
                print(f"Condition failed dataset={dataset_key}, seed={seed}: {e}")
                recompute_and_write_all(results_dir)
                raise

            existing_per_run = pd.concat([existing_per_run, per_run_df], ignore_index=True)
            existing_per_run = existing_per_run.drop_duplicates(
                subset=["dataset", "seed", "algorithm", "name", "k"], keep="last"
            )
            persist_csv(existing_per_run, per_run_path)

            existing_split = pd.concat([existing_split, split_stats_df], ignore_index=True)
            existing_split = existing_split.drop_duplicates(subset=["dataset", "seed"], keep="last")
            persist_csv(existing_split, split_stats_path)

            save_condition_done(
                done_file,
                {
                    "dataset": dataset_key,
                    "seed": seed,
                    "status": "complete",
                },
            )

            monitor = per_run_df.pivot_table(
                index=["dataset", "seed", "algorithm_short"],
                columns=["name", "k"],
                values="value",
                aggfunc="mean",
            )
            print_table(monitor.reset_index(), f"Metrics for {dataset_key} seed {seed}", max_rows=10)
            recompute_and_write_all(results_dir)

    recompute_and_write_all(results_dir)

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
