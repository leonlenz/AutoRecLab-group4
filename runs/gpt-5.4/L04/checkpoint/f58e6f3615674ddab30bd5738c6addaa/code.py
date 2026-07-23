import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, cast
from collections.abc import Hashable

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
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

KS = [1, 5, 10]
SEEDS = [11, 29, 47, 71, 97]
# OmniRec documents UserHoldout(validation_size, test_size) only.
# Use the smallest practical positive validation split to approximate a 2-way 80/20 user holdout.
VALIDATION_SIZE = 0.0001
TEST_SIZE = 0.20
EMPTY_RESULT_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "seed",
    "algorithm",
    "algorithm_short",
    "fold",
    "name",
    "k",
    "value",
)

DATASETS = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}

ALGOS = {
    "ALS": LensKit.ImplicitMFScorer,
    "ItemKNN": LensKit.ItemKNNScorer,
    "Pop": LensKit.PopScorer,
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def print_table(df: pd.DataFrame, title: str, max_rows: int = 20) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("<empty>")
        return
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)} rows total)")
    else:
        print(df.to_string(index=False))



def get_variant_frames(dataset: RecSysDataSet) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = getattr(dataset, "_data", None)
    if data is None:
        raise RuntimeError("Dataset has no internal data payload.")

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
        raise RuntimeError("Could not retrieve train/test DataFrames from split dataset.")
    if val_df is None:
        val_df = pd.DataFrame(columns=train_df.columns)

    return train_df.copy(), val_df.copy(), test_df.copy()



def get_raw_df(dataset: RecSysDataSet) -> pd.DataFrame:
    data = getattr(dataset, "_data", None)
    if data is None or not hasattr(data, "df"):
        raise RuntimeError("Expected a raw-data dataset with a DataFrame-backed payload.")
    return data.df.copy()



def load_or_preprocess_dataset(dataset_key: str, cache_dir: Path) -> RecSysDataSet:
    ds_cache = cache_dir / f"{dataset_key}_processed.rsds"
    if ds_cache.exists():
        print(f"Loading cached processed dataset: {ds_cache}")
        return RecSysDataSet.load(ds_cache)

    print(f"Loading and preprocessing {dataset_key}")
    dataset = RecSysDataSet.use_dataloader(DATASETS[dataset_key])
    steps = []
    if dataset_key in {"MovieLens100K", "Amazon2014VideoGames"}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    processed = Pipe(*steps).process(dataset)
    processed.save(ds_cache)
    return processed



def summarize_processed_dataset(dataset_key: str, dataset: RecSysDataSet) -> pd.DataFrame:
    df = get_raw_df(dataset)
    return pd.DataFrame([
        {
            "dataset": dataset_key,
            "n_interactions": int(len(df)),
            "n_users": int(df["user"].nunique()),
            "n_items": int(df["item"].nunique()),
            "min_user_interactions": int(df.groupby("user").size().min()),
            "min_item_interactions": int(df.groupby("item").size().min()),
            "columns": ",".join(df.columns.astype(str).tolist()),
        }
    ])



def load_or_make_split(dataset_key: str, processed: RecSysDataSet, seed: int, split_dir: Path) -> RecSysDataSet:
    split_path = split_dir / dataset_key / f"seed_{seed}.rsds"
    if split_path.exists():
        print(f"Loading cached split for {dataset_key} seed={seed}")
        return RecSysDataSet.load(split_path)

    ensure_dir(split_path.parent)
    set_random_state(seed)
    splitter = UserHoldout(validation_size=VALIDATION_SIZE, test_size=TEST_SIZE)
    split_ds = splitter.process(processed)
    split_ds.save(split_path)

    train_df, val_df, test_df = get_variant_frames(split_ds)
    export_dir = ensure_dir(split_dir / dataset_key / f"seed_{seed}")
    train_df.to_csv(export_dir / "train.csv", index=False)
    val_df.to_csv(export_dir / "val.csv", index=False)
    test_df.to_csv(export_dir / "test.csv", index=False)
    return split_ds



def split_stats(dataset_key: str, seed: int, split_ds: RecSysDataSet) -> pd.DataFrame:
    train_df, val_df, test_df = get_variant_frames(split_ds)
    total = max(len(train_df) + len(val_df) + len(test_df), 1)
    return pd.DataFrame([
        {
            "dataset": dataset_key,
            "seed": seed,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "train_share": len(train_df) / total,
            "val_share": len(val_df) / total,
            "test_share": len(test_df) / total,
            "n_users_train": int(train_df["user"].nunique()),
            "n_users_val": int(val_df["user"].nunique()) if len(val_df) else 0,
            "n_users_test": int(test_df["user"].nunique()),
            "n_items_train": int(train_df["item"].nunique()),
            "n_items_test": int(test_df["item"].nunique()),
        }
    ])



def result_file(results_dir: Path, dataset_key: str, seed: int, algo_short: str) -> Path:
    return results_dir / dataset_key / f"seed_{seed}" / f"{algo_short}.csv"



def run_one_algorithm(dataset_key: str, seed: int, split_ds: RecSysDataSet, algo_short: str, algo_enum: str, run_root: Path, out_root: Path) -> pd.DataFrame:
    out_file = result_file(out_root, dataset_key, seed, algo_short)
    ensure_dir(out_file.parent)
    if out_file.exists():
        print(f"Skipping completed run: {dataset_key} seed={seed} algo={algo_short}")
        return pd.read_csv(out_file)

    run_dir = ensure_dir(run_root / dataset_key / f"seed_{seed}" / algo_short)
    plan = ExperimentPlan(plan_name=f"seed_sensitivity_{dataset_key}_{seed}_{algo_short}")
    plan.add_algorithm(algo_enum)
    evaluator = Evaluator(NDCG(KS), Precision(KS))

    print(f"Running dataset={dataset_key}, seed={seed}, algo={algo_short}")
    cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
        results = evaluator.get_results()
    finally:
        os.chdir(cwd)

    if len(results) != 1:
        raise RuntimeError(f"Expected one dataset result, found {list(results.keys())}")

    ds_id = next(iter(results.keys()))
    df = results[ds_id].copy()
    df["dataset"] = dataset_key
    df["seed"] = seed
    df["algorithm_short"] = algo_short
    cols = [c for c in ["dataset", "seed", "algorithm", "algorithm_short", "fold", "name", "k", "value"] if c in df.columns]
    df = df[cols].sort_values(["dataset", "seed", "algorithm_short", "name", "k"]).reset_index(drop=True)
    df.to_csv(out_file, index=False)
    return df



def load_all_completed_results(out_root: Path) -> pd.DataFrame:
    files = sorted(out_root.rglob("*.csv"))
    dfs = []
    for file in files:
        try:
            dfs.append(pd.read_csv(file))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in EMPTY_RESULT_COLUMNS})
    return pd.concat(dfs, ignore_index=True)



def coefficient_of_variation(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(vals) == 0:
        return float("nan")
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if mean == 0.0:
        return float("nan")
    return std / mean



def aggregate_results(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()
    grouped = per_run.groupby(["dataset", "algorithm_short", "name", "k"], dropna=False)["value"]
    agg = grouped.agg(["mean", "std", "min", "max", "median", "count"]).reset_index()
    agg["cv"] = grouped.apply(coefficient_of_variation).values
    return agg.sort_values(["dataset", "algorithm_short", "name", "k"]).reset_index(drop=True)



def seed_variability_analysis(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    grouped = per_run.groupby(["dataset", "algorithm_short", "name", "k"], dropna=False)
    for key in grouped.groups:
        if not isinstance(key, tuple) or len(key) != 4:
            continue
        dataset, algorithm, metric, k = cast(Tuple[Hashable, Hashable, Hashable, Hashable], key)
        grp = grouped.get_group(key)
        vals = pd.to_numeric(grp["value"], errors="coerce").dropna().astype(float)
        if len(vals) == 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "metric": metric,
                "k": int(cast(int | str, k)),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "range": float(vals.max() - vals.min()),
                "cv": coefficient_of_variation(vals),
                "n_seeds": int(len(vals)),
                "min_seed_value": float(vals.min()),
                "max_seed_value": float(vals.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric", "k"]).reset_index(drop=True)



def paired_seed_differences(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    grouped = per_run.groupby(["dataset", "name", "k"], dropna=False)
    for key in grouped.groups:
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        dataset, metric, k = cast(Tuple[Hashable, Hashable, Hashable], key)
        grp = grouped.get_group(key)
        pivot = grp.pivot_table(index="seed", columns="algorithm_short", values="value", aggfunc="mean")
        algos = [a for a in ["ALS", "ItemKNN", "Pop"] if a in pivot.columns]
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                a, b = algos[i], algos[j]
                paired = pivot[[a, b]].dropna()
                if paired.empty:
                    continue
                diff = paired[a] - paired[b]
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "k": int(cast(int | str, k)),
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
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "k", "algo_a", "algo_b"]).reset_index(drop=True)



def coverage_table(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected = len(SEEDS)
    for dataset_key in DATASETS:
        for algo_short in ALGOS:
            grp = per_run[(per_run["dataset"] == dataset_key) & (per_run["algorithm_short"] == algo_short)]
            done = int(grp["seed"].nunique()) if not grp.empty else 0
            rows.append(
                {
                    "dataset": dataset_key,
                    "algorithm": algo_short,
                    "completed_seeds": done,
                    "expected_seeds": expected,
                    "complete": done == expected,
                }
            )
    return pd.DataFrame(rows)



def write_short_analysis(aggregated: pd.DataFrame, variability: pd.DataFrame, pairwise: pd.DataFrame, coverage: pd.DataFrame, out_file: Path) -> None:
    lines: List[str] = []
    lines.append("Seed-sensitivity analysis summary")
    lines.append("================================")
    lines.append("")
    lines.append(
        "This script uses OmniRec only, with LensKit wrappers for ALS, ItemKNN, and Pop. "
        "OmniRec documents UserHoldout(validation_size, test_size) as a train/validation/test splitter, so the script uses validation_size=0.0001 and test_size=0.20 to approximate a user-based 80/20 holdout while remaining within documented API usage."
    )
    lines.append("")
    lines.append("Completion coverage:")
    for _, row in coverage.iterrows():
        lines.append(
            f"- {row['dataset']} | {row['algorithm']}: {int(row['completed_seeds'])}/{int(row['expected_seeds'])} seeds complete"
        )
    lines.append("")
    if not variability.empty:
        lines.append("Most stable completed conditions:")
        for _, row in variability.sort_values(["cv", "range"], na_position="last").head(9).iterrows():
            lines.append(
                f"- {row['dataset']} | {row['algorithm']} | {row['metric']}@{int(row['k'])}: mean={row['mean']:.4f}, std={row['std']:.4f}, cv={row['cv']:.4f}, range={row['range']:.4f}"
            )
        lines.append("")
        lines.append("Most seed-sensitive completed conditions:")
        for _, row in variability.sort_values(["cv", "range"], ascending=[False, False], na_position="last").head(9).iterrows():
            lines.append(
                f"- {row['dataset']} | {row['algorithm']} | {row['metric']}@{int(row['k'])}: mean={row['mean']:.4f}, std={row['std']:.4f}, cv={row['cv']:.4f}, range={row['range']:.4f}"
            )
    else:
        lines.append("No completed variability results yet.")
    lines.append("")
    if not pairwise.empty:
        lines.append("Pairwise seed-matched differences:")
        for _, row in pairwise.sort_values("mean_diff", ascending=False).head(12).iterrows():
            lines.append(
                f"- {row['dataset']} | {row['metric']}@{int(row['k'])}: {row['algo_a']} - {row['algo_b']} mean_diff={row['mean_diff']:.4f}, std_diff={row['std_diff']:.4f}, n={int(row['n_pairs'])}"
            )
    else:
        lines.append("No pairwise comparisons available yet.")
    lines.append("")
    lines.append(
        "Interpretation: compare between-algorithm differences to the across-seed standard deviation/range. "
        "If mean differences are much larger than seed variability, conclusions are robust; otherwise, report rankings cautiously."
    )
    out_file.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    working_dir = Path(os.path.join(os.getcwd(), "working"))
    ensure_dir(working_dir)

    cache_dir = ensure_dir(working_dir / "cache")
    split_dir = ensure_dir(working_dir / "splits")
    run_root = ensure_dir(working_dir / "omnirec_runs")
    run_results_dir = ensure_dir(working_dir / "per_condition_results")
    final_results_dir = ensure_dir(working_dir / "results")

    print("Using OmniRec-only pipeline with documented APIs.")
    print(
        f"OmniRec UserHoldout is train/validation/test; using validation_size={VALIDATION_SIZE} and test_size={TEST_SIZE} to approximate 80/20 user holdout."
    )
    print("This script is resumable: completed algorithm runs are skipped, and OmniRec checkpointing resumes interrupted phases.")

    processed_cache: Dict[str, RecSysDataSet] = {}
    preprocess_summaries: List[pd.DataFrame] = []
    split_summary_rows: List[pd.DataFrame] = []

    for dataset_key in DATASETS:
        processed = load_or_preprocess_dataset(dataset_key, cache_dir)
        processed_cache[dataset_key] = processed
        preprocess_summaries.append(summarize_processed_dataset(dataset_key, processed))

    for dataset_key in DATASETS:
        for seed in SEEDS:
            split_ds = load_or_make_split(dataset_key, processed_cache[dataset_key], seed, split_dir)
            split_summary_rows.append(split_stats(dataset_key, seed, split_ds))
            for algo_short, algo_enum in ALGOS.items():
                try:
                    df = run_one_algorithm(
                        dataset_key=dataset_key,
                        seed=seed,
                        split_ds=split_ds,
                        algo_short=algo_short,
                        algo_enum=algo_enum,
                        run_root=run_root,
                        out_root=run_results_dir,
                    )
                    monitor = df.pivot_table(
                        index=["dataset", "seed", "algorithm_short"],
                        columns=["name", "k"],
                        values="value",
                        aggfunc="mean",
                    ).reset_index()
                    print_table(monitor, f"Metrics for {dataset_key} seed {seed} algo {algo_short}", max_rows=10)
                except TimeoutError:
                    print(f"Timeout during {dataset_key} seed={seed} algo={algo_short}; rerun script to resume.")
                    raise
                except Exception as e:
                    print(f"Run failed for {dataset_key} seed={seed} algo={algo_short}: {e}")

    per_run = load_all_completed_results(run_results_dir)
    preprocessing_summary = pd.concat(preprocess_summaries, ignore_index=True) if preprocess_summaries else pd.DataFrame()
    split_stats_df = pd.concat(split_summary_rows, ignore_index=True) if split_summary_rows else pd.DataFrame()
    aggregated = aggregate_results(per_run)
    variability = seed_variability_analysis(per_run)
    pairwise = paired_seed_differences(per_run)
    coverage = coverage_table(per_run)

    per_run.to_csv(final_results_dir / "per_run_results.csv", index=False)
    preprocessing_summary.to_csv(final_results_dir / "preprocessing_summary.csv", index=False)
    split_stats_df.to_csv(final_results_dir / "split_stats.csv", index=False)
    aggregated.to_csv(final_results_dir / "aggregated_results.csv", index=False)
    variability.to_csv(final_results_dir / "seed_variability_analysis.csv", index=False)
    pairwise.to_csv(final_results_dir / "pairwise_seed_differences.csv", index=False)
    coverage.to_csv(final_results_dir / "completion_coverage.csv", index=False)
    write_short_analysis(aggregated, variability, pairwise, coverage, final_results_dir / "short_statistical_analysis.txt")

    print_table(preprocessing_summary, "Post-preprocessing dataset summaries")
    print_table(split_stats_df, "Per-seed split statistics", max_rows=30)
    print_table(coverage, "Completion coverage", max_rows=30)
    print_table(per_run, "Per-run results", max_rows=40)
    print_table(aggregated, "Aggregated results over seeds", max_rows=60)
    print_table(variability, "Across-seed variability analysis", max_rows=60)
    print_table(pairwise, "Pairwise seed-matched algorithm differences", max_rows=60)

    print("\nArtifacts written to:")
    print(final_results_dir)
    for name in [
        "per_run_results.csv",
        "preprocessing_summary.csv",
        "split_stats.csv",
        "aggregated_results.csv",
        "seed_variability_analysis.csv",
        "pairwise_seed_differences.csv",
        "completion_coverage.csv",
        "short_statistical_analysis.txt",
    ]:
        print(f"- {name}")


if __name__ == "__main__":
    main()
