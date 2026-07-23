import json
import os
import time
import zipfile
from typing import Dict, List, Tuple

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

DATASETS: Dict[str, DataSet] = {
    "MovieLens100K": DataSet.MovieLens100K,
    "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
    "HetrecLastFM": DataSet.HetrecLastFM,
}
DATASET_ORDER: List[str] = ["MovieLens100K", "Amazon2014VideoGames", "HetrecLastFM"]

SEEDS: List[int] = [2027, 3109, 4513, 7127, 9901]

ALGORITHMS: Dict[str, LensKit] = {
    "ALS": LensKit.ImplicitMFScorer,
    "ItemKNN": LensKit.ItemKNNScorer,
    "Pop": LensKit.PopScorer,
}
ALGO_ORDER: List[str] = ["ALS", "ItemKNN", "Pop"]
ALGO_MARKERS: Dict[str, str] = {
    "ALS": "LensKit.ImplicitMFScorer",
    "ItemKNN": "LensKit.ItemKNNScorer",
    "Pop": "LensKit.PopScorer",
}

RESULT_COLUMNS: List[str] = ["dataset", "seed", "algorithm"] + METRIC_COLS


def init_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(RESULT_COLUMNS))


def preprocess_dataset(dataset_name: str, dataset_enum: DataSet) -> RecSysDataSet:
    dataset = RecSysDataSet.use_dataloader(dataset_enum)

    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        # MakeImplicit(int) keeps rating >= threshold, so threshold=4 means ratings > 3
        pipeline = Pipe(MakeImplicit(4), CorePruning(5))
    elif dataset_name == "HetrecLastFM":
        pipeline = Pipe(CorePruning(5))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return pipeline.process(dataset)


def export_raw_dataframe_via_rsds(dataset: RecSysDataSet, export_path_no_ext: str) -> pd.DataFrame:
    dataset.save(export_path_no_ext)
    rsds_path = export_path_no_ext if export_path_no_ext.endswith(".rsds") else export_path_no_ext + ".rsds"

    with zipfile.ZipFile(rsds_path, "r") as zf:
        if "data.csv" not in zf.namelist():
            raise RuntimeError("Expected data.csv in RSDS export for RawData dataset.")
        df = pd.read_csv(zf.open("data.csv"))

    os.remove(rsds_path)
    return df


def user_based_holdout_80_20(df: pd.DataFrame, seed: int) -> SplitData:
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

        idx = np.arange(n)
        chosen = rng.choice(idx, size=n_test, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[chosen] = True

        train_parts.append(user_df.iloc[~mask])
        test_parts.append(user_df.iloc[mask])

    if not train_parts or not test_parts:
        raise RuntimeError("Split failed: empty train/test after user-based holdout.")

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    val_df = train_df.iloc[0:0].copy()

    return SplitData(train=train_df, val=val_df, test=test_df)


def build_lenskit_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name="LensKit-Seed-Sensitivity-Implicit")
    for _, algo in ALGORITHMS.items():
        plan.add_algorithm(algo)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(NDCG(METRIC_KS), Precision(METRIC_KS))


def _metric_value(df: pd.DataFrame, metric_name: str, k: int) -> float:
    x = df[(df["name"] == metric_name) & (df["k"] == float(k))]["value"]
    if x.empty:
        x = df[(df["name"] == metric_name) & (df["k"] == k)]["value"]
    if x.empty:
        return float("nan")
    return float(x.mean())


def extract_rows_for_dataset_seed(
    evaluator_results: Dict[str, pd.DataFrame], dataset_name: str, seed: int
) -> pd.DataFrame:
    keys = [k for k in evaluator_results if str(k).startswith(dataset_name)]
    if not keys:
        return init_results_df()

    dataset_key = keys[-1]
    df = evaluator_results[dataset_key].copy()
    if df.empty:
        return init_results_df()

    df = df[df["name"].isin(["NDCG", "Precision"])].copy()
    if df.empty:
        return init_results_df()

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df[df["k"].isin([1.0, 5.0, 10.0])].copy()
    if df.empty:
        return init_results_df()

    out_rows = []
    for algo_label in ALGO_ORDER:
        marker = ALGO_MARKERS[algo_label]
        algo_df = df[df["algorithm"].astype(str).str.contains(marker, regex=False)].copy()
        if algo_df.empty:
            continue

        algo_id = str(algo_df["algorithm"].iloc[-1])
        algo_df = algo_df[algo_df["algorithm"].astype(str) == algo_id].copy()

        out_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "algorithm": algo_label,
                "ndcg@1": _metric_value(algo_df, "NDCG", 1),
                "ndcg@5": _metric_value(algo_df, "NDCG", 5),
                "ndcg@10": _metric_value(algo_df, "NDCG", 10),
                "precision@1": _metric_value(algo_df, "Precision", 1),
                "precision@5": _metric_value(algo_df, "Precision", 5),
                "precision@10": _metric_value(algo_df, "Precision", 10),
            }
        )

    if not out_rows:
        return init_results_df()

    return pd.DataFrame(out_rows)


def merge_results(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
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


def pending_dataset_seed_runs(per_run_df: pd.DataFrame) -> List[Tuple[str, int]]:
    pending: List[Tuple[str, int]] = []
    for dataset_name in DATASET_ORDER:
        for seed in SEEDS:
            if per_run_df.empty:
                pending.append((dataset_name, seed))
                continue

            subset = per_run_df[(per_run_df["dataset"] == dataset_name) & (per_run_df["seed"] == seed)]
            have = set(subset["algorithm"].tolist())
            need = set(ALGO_ORDER)
            if not need.issubset(have):
                pending.append((dataset_name, seed))

    return pending


def summarize_seed_variation(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()

    rows = []
    grouped = per_run_df.groupby(["dataset", "algorithm"], as_index=False)
    for _, g in grouped:
        dataset_name = str(g["dataset"].iloc[0])
        algo_name = str(g["algorithm"].iloc[0])
        row = {
            "dataset": dataset_name,
            "algorithm": algo_name,
            "seed_count": int(g["seed"].nunique()),
        }
        for metric in METRIC_COLS:
            vals = pd.to_numeric(g[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1))
            row[f"{metric}_min"] = float(vals.min())
            row[f"{metric}_max"] = float(vals.max())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["dataset", "algorithm"]).reset_index(drop=True)


def compute_seed_sensitivity_analysis(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()

    rows = []
    for dataset_name, ds_df in per_run_df.groupby("dataset"):
        for metric in METRIC_COLS:
            seed_std_by_algo = ds_df.groupby("algorithm")[metric].std(ddof=1).dropna()
            mean_seed_std = float(seed_std_by_algo.mean()) if not seed_std_by_algo.empty else np.nan

            algo_means = ds_df.groupby("algorithm")[metric].mean().to_dict()
            algo_values = list(algo_means.values())
            between_algo_std = float(np.std(algo_values, ddof=1)) if len(algo_values) > 1 else np.nan

            pair_gaps = []
            algo_names = sorted(algo_means.keys())
            for i in range(len(algo_names)):
                for j in range(i + 1, len(algo_names)):
                    pair_gaps.append(abs(algo_means[algo_names[i]] - algo_means[algo_names[j]]))
            mean_algo_gap = float(np.mean(pair_gaps)) if pair_gaps else np.nan

            if np.isnan(mean_seed_std) or np.isnan(mean_algo_gap) or mean_algo_gap == 0:
                ratio = np.nan
            else:
                ratio = float(mean_seed_std / mean_algo_gap)

            if np.isnan(mean_seed_std) or np.isnan(between_algo_std):
                seed_share = np.nan
            else:
                within_var = mean_seed_std ** 2
                between_var = between_algo_std ** 2
                denom = within_var + between_var
                seed_share = float(within_var / denom) if denom > 0 else np.nan

            rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "mean_seed_std_across_algorithms": mean_seed_std,
                    "between_algorithm_std_of_means": between_algo_std,
                    "mean_between_algorithm_gap": mean_algo_gap,
                    "seed_std_over_algo_gap_ratio": ratio,
                    "seed_variance_share": seed_share,
                }
            )

    return pd.DataFrame(rows).sort_values(["dataset", "metric"]).reset_index(drop=True)


def print_short_statistical_analysis(sensitivity_df: pd.DataFrame) -> None:
    print("\n=== Short Statistical Analysis: Split-Seed Sensitivity ===")
    if sensitivity_df.empty:
        print("No sensitivity data available.")
        return

    for dataset_name, ds_df in sensitivity_df.groupby("dataset"):
        ratios = ds_df["seed_std_over_algo_gap_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        shares = ds_df["seed_variance_share"].replace([np.inf, -np.inf], np.nan).dropna()

        if ratios.empty:
            print(f"- {dataset_name}: insufficient data to estimate seed/algo ratio.")
            continue

        avg_ratio = float(ratios.mean())
        avg_share = float(shares.mean()) if not shares.empty else np.nan

        if avg_ratio < 0.33:
            interpretation = "low seed sensitivity relative to between-algorithm differences"
        elif avg_ratio < 0.67:
            interpretation = "moderate seed sensitivity"
        else:
            interpretation = "high seed sensitivity (split randomness comparable to model gaps)"

        if np.isnan(avg_share):
            share_txt = "n/a"
        else:
            share_txt = f"{avg_share:.3f}"

        print(
            f"- {dataset_name}: mean(seed_std/algo_gap)={avg_ratio:.3f}, "
            f"mean(seed_variance_share)={share_txt} -> {interpretation}."
        )


def persist_outputs(per_run_df: pd.DataFrame, output_dir: str, verbose: bool = False) -> None:
    per_run_csv = os.path.join(output_dir, "per_run_results_dataset_seed_algorithm.csv")
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
        print("\n===== Per-Run Results =====")
        print(per_run_df.to_string(index=False))

        print("\n===== Mean/Std Across Seeds (dataset x algorithm) =====")
        print(summary_df.to_string(index=False))

        print("\n===== Seed Sensitivity vs Algorithm Gaps =====")
        print(sensitivity_df.to_string(index=False))

        print_short_statistical_analysis(sensitivity_df)

        print("\nSaved outputs:")
        print("-", per_run_csv)
        print("-", summary_csv)
        print("-", sensitivity_csv)
        print("-", os.path.join(output_dir, "split_seeds.json"))


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[RESULT_COLUMNS]
    return out


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    output_dir = os.path.join(working_dir, "seed_sensitivity_outputs")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "split_seeds.json"), "w", encoding="utf-8") as f:
        json.dump({"split_seeds": SEEDS}, f, indent=2)

    per_run_path = os.path.join(output_dir, "per_run_results_dataset_seed_algorithm.csv")
    if os.path.exists(per_run_path):
        per_run_df = pd.read_csv(per_run_path)
        per_run_df = ensure_columns(per_run_df)
        print(f"Loaded existing results: {per_run_path} ({len(per_run_df)} rows)")
    else:
        per_run_df = init_results_df()

    pending = pending_dataset_seed_runs(per_run_df)
    print(f"Pending dataset-seed runs: {len(pending)}")
    if not pending:
        print("All dataset-seed runs already completed. Rebuilding summaries only.")
        persist_outputs(per_run_df, output_dir, verbose=True)
        return

    needed_datasets = sorted({d for d, _ in pending})
    base_datasets: Dict[str, RecSysDataSet] = {}
    base_frames: Dict[str, pd.DataFrame] = {}

    for dataset_name in needed_datasets:
        print(f"\n===== Preparing dataset: {dataset_name} =====")
        ds = preprocess_dataset(dataset_name, DATASETS[dataset_name])
        tmp_base = os.path.join(output_dir, f"{dataset_name}_preprocessed_raw")
        df = export_raw_dataframe_via_rsds(ds, tmp_base)
        print(f"Preprocessed interactions: {len(df)}")
        base_datasets[dataset_name] = ds
        base_frames[dataset_name] = df

    max_runtime_seconds = int(os.getenv("MAX_RUNTIME_SECONDS", "3300"))
    max_dataset_seed_runs = int(os.getenv("MAX_DATASET_SEED_RUNS", "999999"))

    start = time.time()
    executed = 0

    original_cwd = os.getcwd()
    os.chdir(working_dir)
    try:
        for dataset_name, seed in pending:
            elapsed = time.time() - start
            if elapsed >= max_runtime_seconds:
                print(
                    f"\nStopping before timeout budget: elapsed={elapsed:.1f}s, budget={max_runtime_seconds}s"
                )
                break
            if executed >= max_dataset_seed_runs:
                print(f"\nReached MAX_DATASET_SEED_RUNS={max_dataset_seed_runs}; stopping cleanly.")
                break

            print(f"\n========== Run: dataset={dataset_name}, seed={seed} ==========")

            set_random_state(seed)
            split_data = user_based_holdout_80_20(base_frames[dataset_name], seed)
            split_dataset = base_datasets[dataset_name].replace_data(split_data)

            plan = build_lenskit_plan()
            evaluator = build_evaluator()

            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)

            rows_df = extract_rows_for_dataset_seed(
                evaluator_results=evaluator.get_results(),
                dataset_name=dataset_name,
                seed=seed,
            )

            if rows_df.empty:
                print("Warning: no rows extracted for this run.")
            else:
                print("Run results:")
                print(rows_df.to_string(index=False))
                per_run_df = merge_results(per_run_df, rows_df)

            persist_outputs(per_run_df, output_dir, verbose=False)
            executed += 1

    finally:
        os.chdir(original_cwd)

    if per_run_df.empty:
        raise RuntimeError("No experiment results were collected.")

    remaining = pending_dataset_seed_runs(per_run_df)
    print(f"\nCompleted dataset-seed runs this invocation: {executed}")
    print(f"Remaining pending dataset-seed runs: {len(remaining)}")

    if remaining:
        print("You can resume by rerunning the script; results are checkpointed.")
    else:
        print("All runs complete.")

    persist_outputs(per_run_df, output_dir, verbose=True)


if __name__ == "__main__":
    main()
