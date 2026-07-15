import itertools
import json
import os
import time
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import SplitData
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


METRIC_KS: List[int] = [1, 5, 10]
METRIC_COLS: List[str] = [
    "ndcg@1",
    "ndcg@5",
    "ndcg@10",
    "precision@1",
    "precision@5",
    "precision@10",
]

BASE_RESULT_COLUMNS: Tuple[str, str, str] = ("dataset", "seed", "algorithm")
PER_RUN_COLUMNS: pd.Index = pd.Index(list(BASE_RESULT_COLUMNS) + METRIC_COLS)

DATASETS: Dict[str, DataSet] = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}

DATASET_ORDER: List[str] = ["HetrecLastFM", "MovieLens100K", "Amazon2014VideoGames"]
SEEDS: List[int] = [2027, 3109, 4513, 7127, 9901]

ALGORITHMS: Dict[str, LensKit] = {
    "Pop": LensKit.PopScorer,
    "ItemKNN": LensKit.ItemKNNScorer,
    "ALS": LensKit.ImplicitMFScorer,
}
ALGO_ORDER: List[str] = ["Pop", "ItemKNN", "ALS"]
ALGO_MARKERS: Dict[str, str] = {
    "Pop": "PopScorer",
    "ItemKNN": "ItemKNNScorer",
    "ALS": "ImplicitMFScorer",
}


def init_per_run_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PER_RUN_COLUMNS)


def export_raw_dataframe_via_rsds(dataset: RecSysDataSet, export_path_no_ext: str) -> pd.DataFrame:
    dataset.save(export_path_no_ext)
    rsds_path = export_path_no_ext if export_path_no_ext.endswith(".rsds") else export_path_no_ext + ".rsds"
    with zipfile.ZipFile(rsds_path, "r") as zf:
        if "data.csv" not in zf.namelist():
            raise RuntimeError("Expected data.csv in RSDS export for RawData dataset.")
        df = pd.read_csv(zf.open("data.csv"))
    os.remove(rsds_path)
    return df


def preprocess_dataset(dataset_name: str, dataset_enum: DataSet) -> RecSysDataSet:
    dataset = RecSysDataSet.use_dataloader(dataset_enum)

    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        pipe = Pipe(
            MakeImplicit(4),
            CorePruning(5),
        )
    elif dataset_name == "HetrecLastFM":
        pipe = Pipe(
            MakeImplicit(1),
            CorePruning(5),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return pipe.process(dataset)


def user_random_holdout_80_20(df: pd.DataFrame, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, user_df in df.groupby("user", sort=False):
        n = len(user_df)
        if n < 2:
            continue

        n_test = max(1, int(round(0.2 * n)))
        if n_test >= n:
            n_test = n - 1

        test_idx = rng.choice(np.arange(n), size=n_test, replace=False)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True

        train_parts.append(user_df.iloc[~test_mask])
        test_parts.append(user_df.iloc[test_mask])

    if not train_parts or not test_parts:
        raise RuntimeError("Split failed: empty train/test after user-based holdout.")

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    val_df = train_df.iloc[0:0].copy()

    return SplitData(train=train_df, val=val_df, test=test_df)


def build_single_algo_plan(algo: LensKit) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name="seed_sensitivity_lenskit_implicit")
    plan.add_algorithm(algo)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(
        NDCG(METRIC_KS),
        Precision(METRIC_KS),
    )


def _metric_value(metric_df: pd.DataFrame, metric_name: str, k: int) -> float:
    x = metric_df[(metric_df["name"] == metric_name) & (metric_df["k"] == float(k))]["value"]
    if x.empty:
        x = metric_df[(metric_df["name"] == metric_name) & (metric_df["k"] == k)]["value"]
    if x.empty:
        return float("nan")
    return float(x.mean())


def extract_condition_row(
    evaluator_results: Dict[str, pd.DataFrame],
    dataset_name: str,
    algo_label: str,
    seed: int,
) -> pd.DataFrame:
    matching_dataset_keys = [k for k in evaluator_results.keys() if str(k).startswith(dataset_name)]
    if not matching_dataset_keys:
        return init_per_run_df()

    dataset_key = matching_dataset_keys[-1]
    df = evaluator_results[dataset_key].copy()
    if df.empty:
        return init_per_run_df()

    df = df[df["name"].isin(["NDCG", "Precision"])].copy()
    if df.empty:
        return init_per_run_df()

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df[df["k"].isin([1.0, 5.0, 10.0])].copy()
    if df.empty:
        return init_per_run_df()

    marker = ALGO_MARKERS[algo_label]
    df = df[df["algorithm"].astype(str).str.contains(marker, regex=False)].copy()
    if df.empty:
        return init_per_run_df()

    last_algo_id = str(df["algorithm"].iloc[-1])
    df = df[df["algorithm"].astype(str) == last_algo_id].copy()

    row = {
        "dataset": dataset_name,
        "seed": int(seed),
        "algorithm": algo_label,
        "ndcg@1": _metric_value(df, "NDCG", 1),
        "ndcg@5": _metric_value(df, "NDCG", 5),
        "ndcg@10": _metric_value(df, "NDCG", 10),
        "precision@1": _metric_value(df, "Precision", 1),
        "precision@5": _metric_value(df, "Precision", 5),
        "precision@10": _metric_value(df, "Precision", 10),
    }
    return pd.DataFrame([row])


def merge_per_run(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = new_rows.copy()
    elif new_rows.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new_rows], axis=0, ignore_index=True)

    if merged.empty:
        return merged

    merged = merged.drop_duplicates(subset=["dataset", "seed", "algorithm"], keep="last")
    merged = merged.sort_values(["dataset", "algorithm", "seed"]).reset_index(drop=True)
    return merged


def determine_pending_conditions(per_run_df: pd.DataFrame) -> List[Tuple[str, int, str]]:
    pending: List[Tuple[str, int, str]] = []

    for seed in SEEDS:
        for dataset_name in DATASET_ORDER:
            for algo_label in ALGO_ORDER:
                if per_run_df.empty:
                    pending.append((dataset_name, seed, algo_label))
                    continue

                mask = (
                    (per_run_df["dataset"] == dataset_name)
                    & (per_run_df["seed"] == seed)
                    & (per_run_df["algorithm"] == algo_label)
                )
                if not bool(mask.any()):
                    pending.append((dataset_name, seed, algo_label))

    return pending


def summarize_seed_variation(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()

    grouped = per_run_df.groupby(["dataset", "algorithm"], as_index=False)
    summary_parts = [grouped.agg(seed_count=("seed", "nunique"))]

    for metric in METRIC_COLS:
        part = grouped.agg(
            **{
                f"{metric}_mean": (metric, "mean"),
                f"{metric}_std": (metric, lambda x: x.std(ddof=1)),
            }
        )
        summary_parts.append(part)

    summary = summary_parts[0]
    for part in summary_parts[1:]:
        summary = summary.merge(part, on=["dataset", "algorithm"], how="left")

    return summary.sort_values(["dataset", "algorithm"]).reset_index(drop=True)


def compute_seed_sensitivity_analysis(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if per_run_df.empty:
        return pd.DataFrame()

    for dataset_name, ds_df in per_run_df.groupby("dataset"):
        for metric in METRIC_COLS:
            seed_std_by_algo = ds_df.groupby("algorithm")[metric].std(ddof=1).dropna()
            mean_seed_std = float(seed_std_by_algo.mean()) if not seed_std_by_algo.empty else np.nan

            algo_means = ds_df.groupby("algorithm")[metric].mean().to_dict()
            pairs = list(itertools.combinations(sorted(algo_means.keys()), 2))
            pair_diffs = [abs(algo_means[a] - algo_means[b]) for a, b in pairs]
            mean_algo_gap = float(np.mean(pair_diffs)) if pair_diffs else np.nan

            if np.isnan(mean_seed_std) or np.isnan(mean_algo_gap) or mean_algo_gap == 0:
                ratio = np.nan
            else:
                ratio = float(mean_seed_std / mean_algo_gap)

            rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "mean_seed_std_across_algorithms": mean_seed_std,
                    "mean_between_algorithm_gap": mean_algo_gap,
                    "seed_std_over_algo_gap_ratio": ratio,
                }
            )

    return pd.DataFrame(rows).sort_values(["dataset", "metric"]).reset_index(drop=True)


def print_short_statistical_analysis(sensitivity_df: pd.DataFrame) -> None:
    print("\n=== Short Statistical Analysis: Split-Seed Sensitivity ===")
    if sensitivity_df.empty:
        print("No sensitivity data available yet.")
        return

    for dataset_name, ds_df in sensitivity_df.groupby("dataset"):
        ratios = ds_df["seed_std_over_algo_gap_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        if ratios.empty:
            print(f"- {dataset_name}: insufficient complete data for stable ratio.")
            continue

        avg_ratio = float(ratios.mean())
        if avg_ratio < 0.33:
            interpretation = "low seed sensitivity relative to algorithm gaps"
        elif avg_ratio < 0.67:
            interpretation = "moderate seed sensitivity"
        else:
            interpretation = "high seed sensitivity (random split effects comparable to model gaps)"

        print(f"- {dataset_name}: avg seed-std/algo-gap ratio = {avg_ratio:.3f} -> {interpretation}.")


def persist_outputs(per_run_df: pd.DataFrame, output_dir: str, verbose: bool = False) -> None:
    per_run_csv = os.path.join(output_dir, "per_run_results_dataset_algorithm_seed.csv")
    summary_csv = os.path.join(output_dir, "seed_variation_summary_mean_std.csv")
    sensitivity_csv = os.path.join(output_dir, "seed_sensitivity_vs_algorithm_gap.csv")

    per_run_df.to_csv(per_run_csv, index=False)

    if per_run_df.empty:
        return

    summary_df = summarize_seed_variation(per_run_df)
    sensitivity_df = compute_seed_sensitivity_analysis(per_run_df)

    summary_df.to_csv(summary_csv, index=False)
    sensitivity_df.to_csv(sensitivity_csv, index=False)

    if verbose:
        print("\n===== Per-Run Results (available so far) =====")
        print(per_run_df.to_string(index=False))

        print("\n===== Mean/Std Across Seeds (per dataset x algorithm) =====")
        print(summary_df.to_string(index=False))

        print("\n===== Seed Sensitivity vs Algorithm Gap =====")
        print(sensitivity_df.to_string(index=False))

        print_short_statistical_analysis(sensitivity_df)

        print("\nSaved outputs:")
        print("-", per_run_csv)
        print("-", summary_csv)
        print("-", sensitivity_csv)
        print("-", os.path.join(output_dir, "split_seeds.json"))


def main() -> None:
    original_cwd = os.getcwd()
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    output_dir = os.path.join(working_dir, "seed_sensitivity_outputs")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "split_seeds.json"), "w", encoding="utf-8") as f:
        json.dump({"split_seeds": SEEDS}, f, indent=2)

    per_run_path = os.path.join(output_dir, "per_run_results_dataset_algorithm_seed.csv")
    if os.path.exists(per_run_path):
        per_run_df = pd.read_csv(per_run_path)
        print(f"Loaded existing per-run file: {per_run_path} ({len(per_run_df)} rows)")
    else:
        per_run_df = init_per_run_df()

    pending = determine_pending_conditions(per_run_df)
    print(f"Pending conditions: {len(pending)}")
    if not pending:
        print("All conditions already completed. Regenerating summaries only.")
        persist_outputs(per_run_df, output_dir, verbose=True)
        return

    needed_datasets = sorted({ds for ds, _, _ in pending})

    base_datasets: Dict[str, RecSysDataSet] = {}
    base_frames: Dict[str, pd.DataFrame] = {}

    for dataset_name in needed_datasets:
        print(f"\n===== Preparing dataset: {dataset_name} =====")
        ds = preprocess_dataset(dataset_name, DATASETS[dataset_name])
        tmp_path = os.path.join(output_dir, f"{dataset_name}_preprocessed_raw")
        df = export_raw_dataframe_via_rsds(ds, tmp_path)
        print(f"Preprocessed interactions for {dataset_name}: {len(df)}")
        base_datasets[dataset_name] = ds
        base_frames[dataset_name] = df

    max_runtime_seconds = int(os.getenv("MAX_RUNTIME_SECONDS", "3000"))
    max_conditions_per_run = int(os.getenv("MAX_CONDITIONS_PER_RUN", "999999"))
    start_time = time.time()
    completed_this_invocation = 0

    os.chdir(working_dir)
    try:
        for dataset_name, seed, algo_label in pending:
            elapsed = time.time() - start_time
            if elapsed >= max_runtime_seconds:
                print(
                    f"\nStopping before hard timeout (elapsed={elapsed:.1f}s, budget={max_runtime_seconds}s)."
                )
                break
            if completed_this_invocation >= max_conditions_per_run:
                print(
                    f"\nReached MAX_CONDITIONS_PER_RUN={max_conditions_per_run}. Stopping cleanly."
                )
                break

            print(
                f"\n========== Condition: dataset={dataset_name}, seed={seed}, algorithm={algo_label} =========="
            )

            set_random_state(seed)
            split_data = user_random_holdout_80_20(base_frames[dataset_name], seed)
            split_dataset = base_datasets[dataset_name].replace_data(split_data)

            plan = build_single_algo_plan(ALGORITHMS[algo_label])
            evaluator = build_evaluator()

            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)

            row_df = extract_condition_row(
                evaluator_results=evaluator.get_results(),
                dataset_name=dataset_name,
                algo_label=algo_label,
                seed=seed,
            )

            if row_df.empty:
                print("Warning: no metrics row extracted for this condition.")
            else:
                print("Condition result:")
                print(row_df.to_string(index=False))
                per_run_df = merge_per_run(per_run_df, row_df)

            persist_outputs(per_run_df, output_dir, verbose=False)
            completed_this_invocation += 1

    finally:
        os.chdir(original_cwd)

    if per_run_df.empty:
        raise RuntimeError("No experiment rows were collected.")

    pending_after = determine_pending_conditions(per_run_df)
    print(f"\nCompleted this invocation: {completed_this_invocation}")
    print(f"Remaining pending conditions: {len(pending_after)}")

    if pending_after:
        print("Resume by rerunning the script; completed conditions are already checkpointed.")
        print("Tip: increase MAX_RUNTIME_SECONDS or run multiple invocations to finish all 45 conditions.")
    else:
        print("All 45 conditions completed.")

    persist_outputs(per_run_df, output_dir, verbose=True)


if __name__ == "__main__":
    main()
