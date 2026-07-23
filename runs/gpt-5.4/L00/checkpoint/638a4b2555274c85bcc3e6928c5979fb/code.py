import os
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


SEEDS = [11, 23, 37, 47, 59]
KS = [1, 5, 10]
ALGORITHM_LABELS = {
    "LensKit.PopScorer": "Pop",
    "LensKit.ItemKNNScorer": "ItemKNN",
    "LensKit.ImplicitMFScorer": "ALS",
}
DATASET_SPECS = [
    ("MovieLens100K", DataSet.MovieLens100K, True),
    ("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, True),
    ("HetrecLastFM", DataSet.HetrecLastFM, False),
]


def extract_raw_dataframe(dataset):
    for attr in ["df", "data", "_data"]:
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            if isinstance(val, pd.DataFrame):
                return val.copy()
            if hasattr(val, "df") and isinstance(val.df, pd.DataFrame):
                return val.df.copy()
    raise AttributeError("Could not locate a pandas DataFrame inside RecSysDataSet.")


def rebuild_dataset_from_df(dataset, df):
    if hasattr(dataset, "replace_data") and hasattr(dataset, "_data"):
        data_obj = dataset._data
        if hasattr(data_obj, "df"):
            new_data = type(data_obj)(df)
            return dataset.replace_data(new_data)
    raise AttributeError("Could not rebuild RecSysDataSet from filtered DataFrame.")


def load_and_prepare_raw_dataset(dataset_enum, convert_gt3):
    dataset = RecSysDataSet.use_dataloader(dataset_enum)
    df = extract_raw_dataframe(dataset)
    if convert_gt3:
        if "rating" not in df.columns:
            raise ValueError(f"Expected 'rating' column for dataset {dataset_enum}.")
        df = df.loc[df["rating"] > 3].copy()
        if df.empty:
            raise ValueError(f"Filtering ratings > 3 produced an empty dataset for {dataset_enum}.")
        keep_cols = [c for c in ["user", "item", "timestamp"] if c in df.columns]
        df = df[keep_cols].copy()
    else:
        if "rating" in df.columns:
            keep_cols = [c for c in ["user", "item", "timestamp"] if c in df.columns]
            df = df[keep_cols].copy()
    dataset = rebuild_dataset_from_df(dataset, df)
    return dataset


def preprocess_for_seed(dataset, seed):
    set_random_state(seed)
    pipe = Pipe(
        CorePruning(5),
        UserHoldout(validation_size=0.0, test_size=0.2),
    )
    return pipe.process(dataset)


def build_plan(plan_name):
    plan = ExperimentPlan(plan_name=plan_name)
    plan.add_algorithm(LensKit.PopScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer, {"feedback": "implicit"})
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    return plan


def run_single_experiment(dataset_name, split_dataset, working_dir, seed):
    seed_dir = os.path.join(working_dir, f"{dataset_name}_seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    os.chdir(seed_dir)
    try:
        plan = build_plan(f"seed-effect-{dataset_name}-seed-{seed}")
        evaluator = Evaluator(NDCG(KS), Precision(KS))
        run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
        results = evaluator.get_results()
        frames = []
        for ds_id, df in results.items():
            tmp = df.copy()
            tmp["dataset_id"] = ds_id
            frames.append(tmp)
        if not frames:
            raise RuntimeError(f"No evaluation results found for {dataset_name}, seed={seed}")
        out = pd.concat(frames, ignore_index=True)
        out["seed"] = seed
        out["dataset"] = dataset_name
        return out
    finally:
        os.chdir(working_dir)


def normalize_algorithm_name(algo_string):
    for key, label in ALGORITHM_LABELS.items():
        if str(algo_string).startswith(key):
            return label
    return str(algo_string)


def compute_seed_statistics(df):
    rows = []
    grouped = df.groupby(["dataset", "algorithm", "name", "k"], dropna=False)
    for (dataset, algorithm, metric, k), grp in grouped:
        values = grp["value"].astype(float).to_numpy()
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = std / sqrt(n) if n > 0 else np.nan
        tcrit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else np.nan
        ci_low = mean - tcrit * sem if n > 1 else np.nan
        ci_high = mean + tcrit * sem if n > 1 else np.nan
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        value_range = max_v - min_v
        cv = std / mean if mean != 0 else np.nan
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "metric": metric,
                "k": k,
                "n_seeds": n,
                "mean": mean,
                "std": std,
                "min": min_v,
                "max": max_v,
                "range": value_range,
                "cv": cv,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "metric", "k"]).reset_index(drop=True)


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    all_results = []

    for dataset_name, dataset_enum, convert_gt3 in DATASET_SPECS:
        print_section(f"Loading base dataset: {dataset_name}")
        base_dataset = load_and_prepare_raw_dataset(dataset_enum, convert_gt3)

        for seed in SEEDS:
            print_section(f"Running dataset={dataset_name}, seed={seed}")
            split_dataset = preprocess_for_seed(base_dataset, seed)
            result_df = run_single_experiment(dataset_name, split_dataset, working_dir, seed)
            all_results.append(result_df)

            view = result_df[["dataset", "seed", "algorithm", "name", "k", "value"]].copy()
            view["algorithm"] = view["algorithm"].map(normalize_algorithm_name)
            print(view.sort_values(["algorithm", "name", "k"]).to_string(index=False))

    results_df = pd.concat(all_results, ignore_index=True)
    results_df["algorithm"] = results_df["algorithm"].map(normalize_algorithm_name)
    results_df = results_df[["dataset", "seed", "algorithm", "name", "k", "value", "fold", "dataset_id"]]
    results_path = Path(working_dir) / "seed_effect_all_results.csv"
    results_df.to_csv(results_path, index=False)

    per_seed_table = (
        results_df.pivot_table(
            index=["dataset", "algorithm", "seed"],
            columns=["name", "k"],
            values="value",
            aggfunc="first",
        )
        .sort_index()
    )
    per_seed_table.columns = [f"{m}@{int(k)}" for m, k in per_seed_table.columns]
    per_seed_table = per_seed_table.reset_index()
    per_seed_path = Path(working_dir) / "seed_effect_per_seed_results.csv"
    per_seed_table.to_csv(per_seed_path, index=False)

    summary_df = compute_seed_statistics(results_df)
    summary_path = Path(working_dir) / "seed_effect_summary_stats.csv"
    summary_df.to_csv(summary_path, index=False)

    print_section("Per-seed results")
    print(per_seed_table.to_string(index=False))

    print_section("Seed variability summary")
    print(summary_df.to_string(index=False))

    variability_rank = (
        summary_df.groupby(["dataset", "algorithm"], as_index=False)["cv"]
        .mean()
        .rename(columns={"cv": "mean_cv_across_metrics"})
        .sort_values(["dataset", "mean_cv_across_metrics", "algorithm"])
    )
    variability_rank_path = Path(working_dir) / "seed_effect_variability_rank.csv"
    variability_rank.to_csv(variability_rank_path, index=False)

    print_section("Algorithm ranking by average coefficient of variation across metrics")
    print(variability_rank.to_string(index=False))

    print_section("Short statistical analysis")
    for dataset in summary_df["dataset"].drop_duplicates():
        print(f"Dataset: {dataset}")
        ds = summary_df[summary_df["dataset"] == dataset].copy()
        for algorithm in ds["algorithm"].drop_duplicates():
            sub = ds[ds["algorithm"] == algorithm]
            avg_std = sub["std"].mean()
            avg_range = sub["range"].mean()
            avg_cv = sub["cv"].mean()
            print(
                f"  {algorithm}: mean std={avg_std:.6f}, mean range={avg_range:.6f}, "
                f"mean CV={avg_cv:.6f} across NDCG/Precision at k=1,5,10"
            )
        print("")

    print_section("Saved files")
    for path in [results_path, per_seed_path, summary_path, variability_rank_path]:
        print(str(path))


if __name__ == "__main__":
    main()
