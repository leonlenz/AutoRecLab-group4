import itertools
import json
import os
import zipfile
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import SplitData
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.filter import RatingFilter
from omnirec.preprocess.pipe import Pipe
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


METRIC_COLS: List[str] = [
    "ndcg@1",
    "ndcg@5",
    "ndcg@10",
    "precision@1",
    "precision@5",
    "precision@10",
]

ALGO_LABELS: Set[str] = {"ALS", "ItemKNN", "Pop"}


def init_per_run_df() -> pd.DataFrame:
    columns = pd.Index(["dataset", "seed", "algorithm", *METRIC_COLS])
    return pd.DataFrame(columns=columns)


def export_raw_dataframe_via_rsds(dataset: RecSysDataSet, export_path: str) -> pd.DataFrame:
    dataset.save(export_path)
    rsds_path = export_path if export_path.endswith(".rsds") else export_path + ".rsds"
    with zipfile.ZipFile(rsds_path, "r") as zf:
        if "data.csv" not in zf.namelist():
            raise RuntimeError("Expected RawData with data.csv in RSDS export, but data.csv was not found.")
        df = pd.read_csv(zf.open("data.csv"))
    os.remove(rsds_path)
    return df


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

        test_local_idx = rng.choice(np.arange(n), size=n_test, replace=False)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_local_idx] = True

        test_parts.append(user_df.iloc[test_mask])
        train_parts.append(user_df.iloc[~test_mask])

    if not train_parts or not test_parts:
        raise RuntimeError("Split failed: produced empty train or test partition.")

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    val_df = df.iloc[0:0].copy().reset_index(drop=True)

    return SplitData(train=train_df, val=val_df, test=test_df)


def preprocess_dataset(dataset_name: str, dataset_enum: DataSet) -> RecSysDataSet:
    dataset = RecSysDataSet.use_dataloader(dataset_enum)

    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        pipe = Pipe(
            RatingFilter(lower=4),
            MakeImplicit(4),
            CorePruning(5),
        )
    elif dataset_name == "HetrecLastFM":
        pipe = Pipe(
            MakeImplicit(1),
            CorePruning(5),
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return pipe.process(dataset)


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name="seed_sensitivity_lenskit_implicit")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )


def algorithm_label_from_result_name(result_algo_name: str) -> str:
    if "ImplicitMFScorer" in result_algo_name:
        return "ALS"
    if "ItemKNNScorer" in result_algo_name:
        return "ItemKNN"
    if "PopScorer" in result_algo_name:
        return "Pop"
    return "Unknown"


def parse_dataset_name_from_result_key(dataset_key: str) -> str:
    if "-" in dataset_key:
        return dataset_key.split("-", 1)[0]
    return dataset_key


def collect_seed_rows(
    evaluator_results: Dict[str, pd.DataFrame],
    seed: int,
    datasets_filter: Optional[Set[str]] = None,
) -> pd.DataFrame:
    collected: List[Dict[str, float | int | str]] = []

    for dataset_key, df in evaluator_results.items():
        dataset_name = parse_dataset_name_from_result_key(dataset_key)
        if datasets_filter is not None and dataset_name not in datasets_filter:
            continue

        working = df.copy()
        if working.empty:
            continue

        working = working[working["name"].isin(["NDCG", "Precision"])].copy()
        if working.empty:
            continue

        working["k"] = pd.to_numeric(working["k"], errors="coerce")
        working = working[working["k"].isin([1, 5, 10])].copy()
        if working.empty:
            continue

        working = working[working["algorithm"].astype(str).str.endswith(f"-{seed}")].copy()
        if working.empty:
            continue

        pivot = working.pivot_table(
            index="algorithm",
            columns=["name", "k"],
            values="value",
            aggfunc="mean",
        )

        for algo_name, row in pivot.iterrows():
            algo_label = algorithm_label_from_result_name(str(algo_name))
            if algo_label == "Unknown":
                continue

            out = {
                "dataset": dataset_name,
                "seed": int(seed),
                "algorithm": algo_label,
                "ndcg@1": float(row.get(("NDCG", 1.0), np.nan)),
                "ndcg@5": float(row.get(("NDCG", 5.0), np.nan)),
                "ndcg@10": float(row.get(("NDCG", 10.0), np.nan)),
                "precision@1": float(row.get(("Precision", 1.0), np.nan)),
                "precision@5": float(row.get(("Precision", 5.0), np.nan)),
                "precision@10": float(row.get(("Precision", 10.0), np.nan)),
            }
            collected.append(out)

    return pd.DataFrame(collected)


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


def determine_pending_conditions(
    per_run_df: pd.DataFrame,
    dataset_names: Sequence[str],
    seeds: Sequence[int],
) -> List[Tuple[str, int]]:
    pending: List[Tuple[str, int]] = []

    if per_run_df.empty:
        for ds in dataset_names:
            for seed in seeds:
                pending.append((ds, seed))
        return pending

    for ds in dataset_names:
        for seed in seeds:
            subset = per_run_df[(per_run_df["dataset"] == ds) & (per_run_df["seed"] == seed)]
            have = set(subset["algorithm"].astype(str).unique())
            if not ALGO_LABELS.issubset(have):
                pending.append((ds, seed))

    return pending


def summarize_seed_variation(per_run_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {}
    for m in METRIC_COLS:
        agg_spec[m + "_mean"] = (m, "mean")
        agg_spec[m + "_std"] = (m, "std")

    summary = (
        per_run_df.groupby(["dataset", "algorithm"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["dataset", "algorithm"])
        .reset_index(drop=True)
    )
    return summary


def compute_seed_sensitivity_analysis(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset_name, ds_df in per_run_df.groupby("dataset"):
        for metric in METRIC_COLS:
            seed_std_by_algo = ds_df.groupby("algorithm")[metric].std(ddof=1)
            mean_seed_std = float(seed_std_by_algo.mean()) if not seed_std_by_algo.empty else np.nan

            algo_means = ds_df.groupby("algorithm")[metric].mean().to_dict()
            pairs = list(itertools.combinations(sorted(algo_means.keys()), 2))
            pair_diffs = [abs(algo_means[a] - algo_means[b]) for a, b in pairs]
            mean_algo_gap = float(np.mean(pair_diffs)) if pair_diffs else np.nan

            if np.isnan(mean_seed_std) or np.isnan(mean_algo_gap) or mean_algo_gap == 0:
                sensitivity_ratio = np.nan
            else:
                sensitivity_ratio = float(mean_seed_std / mean_algo_gap)

            rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "mean_seed_std_across_algorithms": mean_seed_std,
                    "mean_between_algorithm_gap": mean_algo_gap,
                    "seed_std_over_algo_gap_ratio": sensitivity_ratio,
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["dataset", "metric"]).reset_index(drop=True)
    return out_df


def print_short_statistical_analysis(sensitivity_df: pd.DataFrame) -> None:
    print("\n=== Short Statistical Analysis: Split-Seed Sensitivity ===")
    for dataset_name, ds_df in sensitivity_df.groupby("dataset"):
        ratio_vals = ds_df["seed_std_over_algo_gap_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        if ratio_vals.empty:
            print(f"- {dataset_name}: insufficient data for stable sensitivity ratio.")
            continue

        avg_ratio = float(ratio_vals.mean())
        if avg_ratio < 0.33:
            interpretation = "low seed sensitivity relative to algorithm differences"
        elif avg_ratio < 0.67:
            interpretation = "moderate seed sensitivity"
        else:
            interpretation = "high seed sensitivity (split randomness is comparable to model gaps)"

        print(
            f"- {dataset_name}: average seed-std/algo-gap ratio = {avg_ratio:.3f} -> {interpretation}."
        )


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
        print("\n===== Final Per-Run Results =====")
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
    working_dir = os.path.join(original_cwd, "working")
    os.makedirs(working_dir, exist_ok=True)

    output_dir = os.path.join(working_dir, "seed_sensitivity_outputs")
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "MovieLens100K": DataSet.MovieLens100K,
        "Amazon2014VideoGames": DataSet.Amazon2014VideoGames,
        "HetrecLastFM": DataSet.HetrecLastFM,
    }

    seeds = [2027, 3109, 4513, 7127, 9901]
    with open(os.path.join(output_dir, "split_seeds.json"), "w", encoding="utf-8") as f:
        json.dump({"split_seeds": seeds}, f, indent=2)

    print("Using split seeds:", seeds)

    per_run_path = os.path.join(output_dir, "per_run_results_dataset_algorithm_seed.csv")
    if os.path.exists(per_run_path):
        per_run_df = pd.read_csv(per_run_path)
        print(f"Loaded existing per-run results from {per_run_path} ({len(per_run_df)} rows).")
    else:
        per_run_df = init_per_run_df()

    pending = determine_pending_conditions(per_run_df, list(datasets.keys()), seeds)
    if not pending:
        print("All dataset-seed-algorithm combinations already completed. Regenerating summaries only.")
        persist_outputs(per_run_df, output_dir, verbose=True)
        return

    print(f"Pending dataset-seed conditions: {len(pending)}")

    needed_dataset_names = sorted({ds for ds, _ in pending})
    base_datasets: Dict[str, RecSysDataSet] = {}
    base_frames: Dict[str, pd.DataFrame] = {}

    for dataset_name in needed_dataset_names:
        print(f"\n===== Preparing dataset: {dataset_name} =====")
        base_dataset = preprocess_dataset(dataset_name, datasets[dataset_name])
        rsds_temp_path = os.path.join(output_dir, f"{dataset_name}_preprocessed_raw")
        base_df = export_raw_dataframe_via_rsds(base_dataset, rsds_temp_path)
        print(f"Preprocessed interactions for {dataset_name}: {len(base_df)}")
        base_datasets[dataset_name] = base_dataset
        base_frames[dataset_name] = base_df

    plan = build_plan()
    evaluator = build_evaluator()

    os.chdir(working_dir)
    try:
        for seed in seeds:
            datasets_for_seed = [ds for ds, s in pending if s == seed]
            if not datasets_for_seed:
                continue

            print(f"\n========== Running seed {seed} on datasets: {datasets_for_seed} ==========")
            set_random_state(seed)

            split_datasets: List[RecSysDataSet] = []
            for dataset_name in datasets_for_seed:
                split_data = user_random_holdout_80_20(base_frames[dataset_name], seed)
                split_dataset = base_datasets[dataset_name].replace_data(split_data)
                split_datasets.append(split_dataset)

            run_omnirec(datasets=split_datasets, plan=plan, evaluator=evaluator)

            seed_rows = collect_seed_rows(
                evaluator.get_results(),
                seed=seed,
                datasets_filter=set(datasets_for_seed),
            )

            if seed_rows.empty:
                print(f"Warning: no result rows captured for seed={seed}")
            else:
                print("Per-run result snapshot:")
                print(seed_rows.sort_values(["dataset", "algorithm"]).to_string(index=False))

            per_run_df = merge_per_run(per_run_df, seed_rows)
            persist_outputs(per_run_df, output_dir, verbose=False)

    finally:
        os.chdir(original_cwd)

    if per_run_df.empty:
        raise RuntimeError("No experiment results were produced.")

    still_pending = determine_pending_conditions(per_run_df, list(datasets.keys()), seeds)
    if still_pending:
        print("\nExperiment ended with remaining pending conditions (safe to re-run and resume):")
        for ds, seed in still_pending:
            print(f"- dataset={ds}, seed={seed}")

    persist_outputs(per_run_df, output_dir, verbose=True)


if __name__ == "__main__":
    main()
