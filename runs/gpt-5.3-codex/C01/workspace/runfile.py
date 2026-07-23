import itertools
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
from omnirec.recsys_data_set import DatasetMeta
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

ALGORITHMS: Dict[str, LensKit] = {
    "Pop": LensKit.PopScorer,
    "ItemKNN": LensKit.ItemKNNScorer,
    "ALS": LensKit.ImplicitMFScorer,
}
ALGO_ORDER: List[str] = ["Pop", "ItemKNN", "ALS"]


def generate_split_seeds(n: int = 5, seed: int = 2026) -> List[int]:
    rng = np.random.default_rng(seed)
    seeds = rng.choice(np.arange(1000, 99999), size=n, replace=False)
    return [int(x) for x in seeds]


def init_per_run_df() -> pd.DataFrame:
    cols: List[str] = ["dataset", "seed", "algorithm", *METRIC_COLS]
    return pd.DataFrame(columns=pd.Index(cols))


def normalize_per_run_df(df: pd.DataFrame) -> pd.DataFrame:
    target_cols = ["dataset", "seed", "algorithm", *METRIC_COLS]
    out = df.copy()
    for c in target_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[target_cols]


def export_raw_dataframe_via_rsds(dataset: RecSysDataSet, export_path_no_ext: str) -> pd.DataFrame:
    dataset.save(export_path_no_ext)
    rsds_path = export_path_no_ext if export_path_no_ext.endswith(".rsds") else export_path_no_ext + ".rsds"
    try:
        with zipfile.ZipFile(rsds_path, "r") as zf:
            if "data.csv" not in zf.namelist():
                raise RuntimeError("Expected data.csv in RSDS export for RawData variant.")
            df = pd.read_csv(zf.open("data.csv"))
    finally:
        if os.path.exists(rsds_path):
            os.remove(rsds_path)
    return df


def preprocess_dataset(dataset_name: str, dataset_enum: DataSet) -> RecSysDataSet:
    dataset = RecSysDataSet.use_dataloader(dataset_enum)

    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        pipe = Pipe(
            CorePruning(5),
            MakeImplicit(4),
            CorePruning(5),
        )
    elif dataset_name == "HetrecLastFM":
        pipe = Pipe(
            CorePruning(5),
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

        perm = rng.permutation(n)
        test_idx = perm[:n_test]

        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True

        train_parts.append(user_df.iloc[~mask])
        test_parts.append(user_df.iloc[mask])

    if not train_parts or not test_parts:
        raise RuntimeError("User holdout produced empty train or test split.")

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    val_df = train_df.iloc[0:0].copy()

    return SplitData(train=train_df, val=val_df, test=test_df)


def build_seed_alias(dataset_name: str, seed: int) -> str:
    return f"{dataset_name}__seed_{seed}"


def build_seed_dataset(base_dataset: RecSysDataSet, split_data: SplitData, alias_name: str) -> RecSysDataSet:
    base_meta = base_dataset.meta
    meta = DatasetMeta(canon_pth=base_meta.canon_pth, raw_dir=base_meta.raw_dir, name=alias_name)
    return RecSysDataSet(split_data, meta)


def build_plan_all_lenskit_algorithms() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name="seed_sensitivity_lenskit_backend")
    for algo_label in ALGO_ORDER:
        plan.add_algorithm(ALGORITHMS[algo_label])
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(NDCG(METRIC_KS), Precision(METRIC_KS))


def metric_from_eval_df(df: pd.DataFrame, metric_name: str, k: int) -> float:
    k_numeric = pd.to_numeric(df["k"], errors="coerce")
    mask = (df["name"] == metric_name) & (k_numeric == float(k))
    vals = df.loc[mask, "value"]
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def extract_rows_for_alias(
    evaluator_results: Dict[str, pd.DataFrame],
    alias_name: str,
    dataset_name: str,
    seed: int,
) -> pd.DataFrame:
    if not evaluator_results:
        return init_per_run_df()

    matching = [k for k in evaluator_results.keys() if str(k).startswith(alias_name)]
    if not matching:
        return init_per_run_df()

    dataset_key = sorted(matching)[-1]
    df = evaluator_results[dataset_key].copy()
    if df.empty:
        return init_per_run_df()

    rows = []
    for algo_label in ALGO_ORDER:
        algo_id = ALGORITHMS[algo_label].value
        algo_df = df[df["algorithm"].astype(str).str.startswith(algo_id)].copy()
        if algo_df.empty:
            continue

        row = {
            "dataset": dataset_name,
            "seed": int(seed),
            "algorithm": algo_label,
            "ndcg@1": metric_from_eval_df(algo_df, "NDCG", 1),
            "ndcg@5": metric_from_eval_df(algo_df, "NDCG", 5),
            "ndcg@10": metric_from_eval_df(algo_df, "NDCG", 10),
            "precision@1": metric_from_eval_df(algo_df, "Precision", 1),
            "precision@5": metric_from_eval_df(algo_df, "Precision", 5),
            "precision@10": metric_from_eval_df(algo_df, "Precision", 10),
        }
        rows.append(row)

    if not rows:
        return init_per_run_df()

    return pd.DataFrame(rows)


def merge_per_run(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = new_rows.copy()
    elif new_rows.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new_rows], axis=0, ignore_index=True)

    if merged.empty:
        return merged

    merged = normalize_per_run_df(merged)
    merged = merged.drop_duplicates(subset=["dataset", "seed", "algorithm"], keep="last")
    merged = merged.sort_values(["dataset", "algorithm", "seed"]).reset_index(drop=True)
    return merged


def summarize_seed_variation(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()

    grouped = per_run_df.groupby(["dataset", "algorithm"], as_index=False)
    parts = [grouped.agg(seed_count=("seed", "nunique"))]

    for metric in METRIC_COLS:
        p = grouped.agg(
            **{
                f"{metric}_mean": (metric, "mean"),
                f"{metric}_std": (metric, lambda x: float(np.std(x, ddof=1))),
                f"{metric}_min": (metric, "min"),
                f"{metric}_max": (metric, "max"),
            }
        )
        parts.append(p)

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["dataset", "algorithm"], how="left")

    return out.sort_values(["dataset", "algorithm"]).reset_index(drop=True)


def compute_seed_sensitivity_analysis(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()

    rows = []
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
        print("No sensitivity data available.")
        return

    for dataset_name, ds_df in sensitivity_df.groupby("dataset"):
        ratios = ds_df["seed_std_over_algo_gap_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        if ratios.empty:
            print(f"- {dataset_name}: insufficient complete data for stable ratio.")
            continue

        avg_ratio = float(ratios.mean())
        if avg_ratio < 0.33:
            interpretation = "low seed sensitivity relative to algorithm differences"
        elif avg_ratio < 0.67:
            interpretation = "moderate seed sensitivity"
        else:
            interpretation = "high seed sensitivity (split randomness comparable to model differences)"

        print(f"- {dataset_name}: avg ratio={avg_ratio:.3f} -> {interpretation}.")


def persist_outputs(per_run_df: pd.DataFrame, output_dir: str, verbose: bool = False) -> None:
    per_run_csv = os.path.join(output_dir, "per_run_results_dataset_seed_algorithm.csv")
    summary_csv = os.path.join(output_dir, "seed_variation_summary_mean_std_min_max.csv")
    sensitivity_csv = os.path.join(output_dir, "seed_sensitivity_vs_algorithm_gap.csv")

    normalize_per_run_df(per_run_df).to_csv(per_run_csv, index=False)

    if per_run_df.empty:
        return

    summary_df = summarize_seed_variation(per_run_df)
    sensitivity_df = compute_seed_sensitivity_analysis(per_run_df)

    summary_df.to_csv(summary_csv, index=False)
    sensitivity_df.to_csv(sensitivity_csv, index=False)

    if verbose:
        print("\n===== Per-run results =====")
        print(per_run_df.to_string(index=False))

        print("\n===== Mean/Std/Min/Max across seeds =====")
        print(summary_df.to_string(index=False))

        print("\n===== Seed sensitivity vs algorithm gap =====")
        print(sensitivity_df.to_string(index=False))

        print_short_statistical_analysis(sensitivity_df)

        print("\nSaved outputs:")
        print("-", per_run_csv)
        print("-", summary_csv)
        print("-", sensitivity_csv)


def condition_completed(per_run_df: pd.DataFrame, dataset_name: str, seed: int) -> bool:
    if per_run_df.empty:
        return False
    rows = per_run_df[(per_run_df["dataset"] == dataset_name) & (per_run_df["seed"] == seed)]
    return set(rows["algorithm"].tolist()) == set(ALGO_ORDER)


def determine_pending_conditions(per_run_df: pd.DataFrame, seeds: List[int]) -> List[Tuple[str, int]]:
    pending: List[Tuple[str, int]] = []
    for dataset_name in DATASET_ORDER:
        for seed in seeds:
            if not condition_completed(per_run_df, dataset_name, seed):
                pending.append((dataset_name, seed))
    return pending


def run_dataset_seed_batch(
    dataset_name: str,
    pending_seeds: List[int],
    base_dataset: RecSysDataSet,
    base_df: pd.DataFrame,
    working_dir: str,
) -> pd.DataFrame:
    split_datasets: List[RecSysDataSet] = []
    alias_to_seed: Dict[str, int] = {}

    for seed in pending_seeds:
        set_random_state(seed)
        split_data = user_random_holdout_80_20(base_df, seed)
        alias_name = build_seed_alias(dataset_name, seed)
        split_dataset = build_seed_dataset(base_dataset, split_data, alias_name)
        split_datasets.append(split_dataset)
        alias_to_seed[alias_name] = seed

    plan = build_plan_all_lenskit_algorithms()
    evaluator = build_evaluator()

    run_dir = os.path.join(working_dir, "runs", dataset_name, f"seed_batch_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        run_omnirec(datasets=split_datasets, plan=plan, evaluator=evaluator)
    finally:
        os.chdir(original_cwd)

    results = evaluator.get_results()
    batch_rows = init_per_run_df()

    for alias_name, seed in alias_to_seed.items():
        rows = extract_rows_for_alias(
            evaluator_results=results,
            alias_name=alias_name,
            dataset_name=dataset_name,
            seed=seed,
        )
        batch_rows = merge_per_run(batch_rows, rows)

    return batch_rows


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    output_dir = os.path.join(working_dir, "seed_sensitivity_outputs")
    os.makedirs(output_dir, exist_ok=True)

    seeds_path = os.path.join(output_dir, "split_seeds.json")
    if os.path.exists(seeds_path):
        with open(seeds_path, "r", encoding="utf-8") as f:
            seeds = [int(x) for x in json.load(f)["split_seeds"]]
    else:
        seeds = generate_split_seeds(n=5, seed=2026)
        with open(seeds_path, "w", encoding="utf-8") as f:
            json.dump({"split_seeds": seeds}, f, indent=2)

    print(f"Using split seeds: {seeds}")

    per_run_path = os.path.join(output_dir, "per_run_results_dataset_seed_algorithm.csv")
    if os.path.exists(per_run_path):
        per_run_df = normalize_per_run_df(pd.read_csv(per_run_path))
        print(f"Loaded existing results: {per_run_path} ({len(per_run_df)} rows)")
    else:
        per_run_df = init_per_run_df()

    pending = determine_pending_conditions(per_run_df, seeds)
    print(f"Pending dataset-seed conditions: {len(pending)} / {len(DATASET_ORDER) * len(seeds)}")

    if not pending:
        print("All conditions already completed. Rebuilding summaries only.")
        persist_outputs(per_run_df, output_dir, verbose=True)
        return

    needed_datasets = [d for d in DATASET_ORDER if any(pdset == d for pdset, _ in pending)]

    base_datasets: Dict[str, RecSysDataSet] = {}
    base_frames: Dict[str, pd.DataFrame] = {}

    for dataset_name in needed_datasets:
        print(f"\n===== Preparing dataset: {dataset_name} =====")
        ds = preprocess_dataset(dataset_name, DATASETS[dataset_name])
        tmp_export = os.path.join(output_dir, f"{dataset_name}_preprocessed_raw")
        df = export_raw_dataframe_via_rsds(ds, tmp_export)
        print(f"Preprocessed interactions: {len(df)}")
        base_datasets[dataset_name] = ds
        base_frames[dataset_name] = df

    max_runtime_seconds = int(os.getenv("MAX_RUNTIME_SECONDS", "0"))
    start_time = time.time()
    completed_batches = 0

    for dataset_name in DATASET_ORDER:
        pending_seeds = [seed for dset, seed in pending if dset == dataset_name]
        if not pending_seeds:
            continue

        if max_runtime_seconds > 0:
            elapsed = time.time() - start_time
            if elapsed >= max_runtime_seconds:
                print(
                    f"\nStopping due to MAX_RUNTIME_SECONDS before dataset batch {dataset_name}: "
                    f"elapsed={elapsed:.1f}s, budget={max_runtime_seconds}s"
                )
                break

        print(f"\n========== Dataset batch: {dataset_name} | pending seeds: {pending_seeds} ==========")
        batch_rows = run_dataset_seed_batch(
            dataset_name=dataset_name,
            pending_seeds=pending_seeds,
            base_dataset=base_datasets[dataset_name],
            base_df=base_frames[dataset_name],
            working_dir=working_dir,
        )

        if batch_rows.empty:
            print("Warning: no rows extracted for this dataset batch.")
        else:
            print("Batch metrics:")
            print(batch_rows.to_string(index=False))
            per_run_df = merge_per_run(per_run_df, batch_rows)

        persist_outputs(per_run_df, output_dir, verbose=False)
        completed_batches += 1

    if per_run_df.empty:
        raise RuntimeError("No experiment results collected.")

    pending_after = determine_pending_conditions(per_run_df, seeds)
    print(f"\nCompleted dataset batches this invocation: {completed_batches}")
    print(f"Remaining pending conditions: {len(pending_after)}")

    persist_outputs(per_run_df, output_dir, verbose=True)

    if pending_after:
        raise RuntimeError(
            "Experiment incomplete: some dataset-seed-algorithm conditions are still missing. "
            "Rerun the script to resume from checkpoints."
        )

    expected_rows = len(DATASET_ORDER) * len(seeds) * len(ALGO_ORDER)
    actual_rows = len(per_run_df)
    print(f"\nAll dataset-seed-algorithm conditions completed ({actual_rows}/{expected_rows} rows).")


if __name__ == "__main__":
    main()