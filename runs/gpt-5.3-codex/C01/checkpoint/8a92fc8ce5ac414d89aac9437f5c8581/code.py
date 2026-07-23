import os
import re
import shutil
import itertools
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import RawData, SplitData
from omnirec.preprocess.base import Preprocessor
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.pipe import Pipe
from omnirec.util.util import set_random_state, get_random_state
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec


class UserHoldout80_20(Preprocessor[RawData, SplitData]):
    """
    Exact user-based 80/20 train-test split for OmniRec datasets.
    Produces SplitData(train, val, test) with an empty validation split.
    """

    def __init__(self, test_size: float = 0.2) -> None:
        super().__init__()
        if not (0 < test_size < 1):
            raise ValueError("test_size must be in (0, 1).")
        self.test_size = test_size

    def _process(self, dataset: RecSysDataSet[RawData]) -> RecSysDataSet[SplitData]:
        df = dataset._data.df.reset_index(drop=True)

        train_indices = []
        test_indices = []
        rs = get_random_state()

        for _, user_idx in df.groupby("user").indices.items():
            tr_idx, te_idx = train_test_split(
                user_idx,
                test_size=self.test_size,
                random_state=rs,
                shuffle=True,
            )
            train_indices.extend(tr_idx.tolist())
            test_indices.extend(te_idx.tolist())

        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()
        val_df = train_df.iloc[0:0].copy()

        return dataset.replace_data(SplitData(train=train_df, val=val_df, test=test_df))


@contextmanager
def pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def build_pipeline(dataset_name: str) -> Pipe:
    # ratings > 3  => threshold 4 for explicit datasets
    if dataset_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        return Pipe(
            MakeImplicit(4),
            CorePruning(5),
            UserHoldout80_20(test_size=0.2),
        )

    # LastFM is implicit by nature; MakeImplicit(1) removes rating column while preserving all interactions
    return Pipe(
        MakeImplicit(1),
        CorePruning(5),
        UserHoldout80_20(test_size=0.2),
    )


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan("SeedSensitivity-LensKit-Baselines")

    # Standard/default hyperparameters (no tuning)
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})

    return plan


def parse_algorithm_base(algorithm_with_hash: str) -> str:
    if "-" not in algorithm_with_hash:
        return algorithm_with_hash
    return algorithm_with_hash.rsplit("-", 1)[0]


def normalize_dataset_id(dataset_id: str) -> str:
    # Dataset IDs in results are typically like <datasetName>-<hash>
    m = re.match(r"^(.*)-([0-9a-f]{8,64})$", str(dataset_id))
    if m:
        return m.group(1)
    return str(dataset_id)


def pivot_per_run(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = df_long.pivot_table(
        index=["dataset", "seed", "algorithm"],
        columns=["name", "k"],
        values="value",
        aggfunc="first",
    ).reset_index()

    flat_cols = []
    for c in wide.columns:
        if isinstance(c, tuple):
            if c[0] in {"dataset", "seed", "algorithm"}:
                flat_cols.append(c[0])
            else:
                flat_cols.append(f"{c[0]}@{int(c[1])}")
        else:
            flat_cols.append(c)
    wide.columns = flat_cols

    metric_cols = ["NDCG@1", "NDCG@5", "NDCG@10", "Precision@1", "Precision@5", "Precision@10"]
    for mc in metric_cols:
        if mc not in wide.columns:
            wide[mc] = np.nan

    wide = wide[["dataset", "seed", "algorithm"] + metric_cols]
    wide = wide.sort_values(["dataset", "algorithm", "seed"]).reset_index(drop=True)
    return wide


def summarize_seed_variation(run_wide: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["NDCG@1", "NDCG@5", "NDCG@10", "Precision@1", "Precision@5", "Precision@10"]
    summary = run_wide.groupby(["dataset", "algorithm"])[metric_cols].agg(["mean", "std"]).reset_index()

    flat_cols = []
    for c in summary.columns:
        if isinstance(c, tuple):
            if c[0] in {"dataset", "algorithm"}:
                flat_cols.append(c[0])
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    summary.columns = flat_cols

    return summary.sort_values(["dataset", "algorithm"]).reset_index(drop=True)


def statistical_sensitivity_analysis(run_wide: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["NDCG@1", "NDCG@5", "NDCG@10", "Precision@1", "Precision@5", "Precision@10"]
    records = []

    for dataset_name, ds_df in run_wide.groupby("dataset"):
        for metric in metric_cols:
            mean_by_alg = ds_df.groupby("algorithm")[metric].mean().dropna()
            std_by_alg = ds_df.groupby("algorithm")[metric].std(ddof=1).dropna()

            avg_seed_std = float(std_by_alg.mean()) if len(std_by_alg) > 0 else np.nan

            if len(mean_by_alg) >= 2:
                pairwise = [
                    abs(float(mean_by_alg[a] - mean_by_alg[b]))
                    for a, b in itertools.combinations(mean_by_alg.index.tolist(), 2)
                ]
                avg_alg_gap = float(np.mean(pairwise))
            else:
                avg_alg_gap = np.nan

            ratio = np.nan
            if pd.notna(avg_seed_std) and pd.notna(avg_alg_gap) and avg_alg_gap > 0:
                ratio = avg_seed_std / avg_alg_gap

            records.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "avg_seed_std": avg_seed_std,
                    "avg_pairwise_algorithm_gap": avg_alg_gap,
                    "seed_std_to_algorithm_gap_ratio": ratio,
                }
            )

    return pd.DataFrame(records)


def run_single_condition(dataset_enum: DataSet, dataset_name: str, seed: int, working_dir: str) -> pd.DataFrame:
    print(f"\n=== Running: dataset={dataset_name}, seed={seed} ===")

    set_random_state(seed)

    dataset = RecSysDataSet.use_dataloader(dataset_enum)
    pipe = build_pipeline(dataset_name)
    dataset = pipe.process(dataset)

    print(f"Split interaction counts: {dataset.num_interactions()}")

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    plan = build_plan()

    run_dir = os.path.join(working_dir, "runs", dataset_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Isolate checkpoints per condition to avoid cache contamination across seeds/datasets
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)

    with pushd(run_dir):
        maybe_eval = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
        if maybe_eval is not None and hasattr(maybe_eval, "get_results"):
            evaluator = maybe_eval

    result_frames = []
    for dataset_id, df in evaluator.get_results().items():
        cdf = df.copy()
        cdf["dataset_id"] = str(dataset_id)
        cdf["dataset"] = dataset_name
        cdf["dataset_from_result_id"] = normalize_dataset_id(str(dataset_id))
        cdf["seed"] = seed
        result_frames.append(cdf)

    if not result_frames:
        raise RuntimeError(f"No results returned for dataset={dataset_name}, seed={seed}")

    out = pd.concat(result_frames, ignore_index=True)
    out["algorithm_full"] = out["algorithm"].astype(str).map(parse_algorithm_base)

    alg_map = {
        "LensKit.ImplicitMFScorer": "ALS",
        "LensKit.ItemKNNScorer": "ItemKNN",
        "LensKit.PopScorer": "Pop",
    }
    out["algorithm"] = out["algorithm_full"].map(alg_map).fillna(out["algorithm_full"])

    out["k"] = pd.to_numeric(out["k"], errors="coerce")

    keep = out["name"].isin(["NDCG", "Precision"]) & out["k"].isin([1, 5, 10])
    out = out.loc[keep].copy()

    # Monitoring print for this run
    monitor = pivot_per_run(out[["dataset", "seed", "algorithm", "name", "k", "value"]])
    print("Per-run metrics (this dataset-seed condition):")
    print(monitor.to_string(index=False))

    return out


def main() -> None:
    working_dir = os.path.join(os.getcwd(), 'working'); os.makedirs(working_dir, exist_ok=True)
    results_dir = os.path.join(working_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    seeds = [13, 42, 123, 2024, 31415]
    print(f"Using random seeds: {seeds}")

    datasets = [
        ("MovieLens100K", DataSet.MovieLens100K),
        ("Amazon2014VideoGames", DataSet.Amazon2014VideoGames),
        ("HetrecLastFM", DataSet.HetrecLastFM),
    ]

    all_long = []

    for dataset_name, dataset_enum in datasets:
        for seed in seeds:
            run_long = run_single_condition(
                dataset_enum=dataset_enum,
                dataset_name=dataset_name,
                seed=seed,
                working_dir=working_dir,
            )
            all_long.append(run_long)

    all_long_df = pd.concat(all_long, ignore_index=True)
    all_long_df = all_long_df.sort_values(["dataset", "algorithm", "seed", "name", "k"]).reset_index(drop=True)

    run_wide_df = pivot_per_run(all_long_df[["dataset", "seed", "algorithm", "name", "k", "value"]])
    summary_df = summarize_seed_variation(run_wide_df)
    sensitivity_df = statistical_sensitivity_analysis(run_wide_df)

    long_path = os.path.join(results_dir, "per_run_results_long.csv")
    wide_path = os.path.join(results_dir, "per_run_results_wide.csv")
    summary_path = os.path.join(results_dir, "seed_variation_summary.csv")
    sensitivity_path = os.path.join(results_dir, "seed_sensitivity_analysis.csv")

    all_long_df.to_csv(long_path, index=False)
    run_wide_df.to_csv(wide_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sensitivity_df.to_csv(sensitivity_path, index=False)

    print("\n=== Final Per-Run Results (dataset x algorithm x seed) ===")
    print(run_wide_df.to_string(index=False))

    print("\n=== Seed Variation Summary (mean/std across 5 seeds) ===")
    print(summary_df.to_string(index=False))

    print("\n=== Short Statistical Seed-Sensitivity Analysis ===")
    print(sensitivity_df.to_string(index=False))

    # Quick human-readable interpretation focused on @10 metrics
    print("\nInterpretation (focused on @10):")
    focus = sensitivity_df[sensitivity_df["metric"].isin(["NDCG@10", "Precision@10"])]
    for _, row in focus.iterrows():
        ds = row["dataset"]
        metric = row["metric"]
        ratio = row["seed_std_to_algorithm_gap_ratio"]
        if pd.isna(ratio):
            tag = "insufficient data for ratio"
        elif ratio < 0.5:
            tag = "algorithm differences dominate seed noise"
        elif ratio < 1.0:
            tag = "seed effects are moderate"
        else:
            tag = "seed effects are large relative to algorithm gaps"
        print(f"- {ds} | {metric}: ratio={ratio:.4f} -> {tag}")

    print("\nSaved files:")
    print(f"- {long_path}")
    print(f"- {wide_path}")
    print(f"- {summary_path}")
    print(f"- {sensitivity_path}")


if __name__ == '__main__':
    main()
