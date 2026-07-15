import os
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.filter import RatingFilter
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

warnings.filterwarnings("ignore")

SEEDS = [7, 19, 42, 77, 123]
KS = [1, 5, 10]


class SimpleSplitData:
    def __init__(self, name: str, train: pd.DataFrame, test: pd.DataFrame, work_dir: str):
        self.name = name
        self.variant = "SplitData"
        self.lineage_steps = 0
        self.interactions = len(train) + len(test)
        self.columns = len(train.columns)
        self._work_dir = Path(work_dir)
        self._dataset_dir = self._work_dir / f"split_{sanitize_name(name)}"
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = self._dataset_dir / "train.csv"
        self.test_path = self._dataset_dir / "test.csv"
        self.validation_path = self._dataset_dir / "validation.csv"
        train.to_csv(self.train_path, index=False)
        test.to_csv(self.test_path, index=False)
        pd.DataFrame(columns=train.columns).to_csv(self.validation_path, index=False)

    def save(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        pd.read_csv(self.train_path).to_csv(target / "train.csv", index=False)
        pd.read_csv(self.test_path).to_csv(target / "test.csv", index=False)
        pd.read_csv(self.validation_path).to_csv(target / "validation.csv", index=False)

    def __repr__(self) -> str:
        return (
            f"SimpleSplitData(name={self.name!r}, variant='SplitData', "
            f"interactions={self.interactions}, columns={self.columns})"
        )


def sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def ensure_interaction_df(ds: Any) -> pd.DataFrame:
    candidates = [
        "interactions_df",
        "data",
        "df",
        "interactions",
    ]
    for attr in candidates:
        if hasattr(ds, attr):
            value = getattr(ds, attr)
            if isinstance(value, pd.DataFrame):
                return value.copy()
            if callable(value):
                try:
                    out = value()
                    if isinstance(out, pd.DataFrame):
                        return out.copy()
                except Exception:
                    pass
    tmp_dir = Path(os.getcwd()) / "working" / "tmp_exports" / sanitize_name(getattr(ds, "name", "dataset"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds.save(str(tmp_dir))
        for fname in ["data.csv", "interactions.csv", "raw.csv", "dataset.csv"]:
            f = tmp_dir / fname
            if f.exists():
                return pd.read_csv(f)
    except Exception:
        pass
    raise RuntimeError("Could not access dataset interactions as a pandas DataFrame using public methods/exports.")


def load_and_preprocess_dataset(dataset_enum: Any, strict_gt3_to_implicit: bool) -> Any:
    print(f"\nLoading dataset: {dataset_enum}")
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    print(ds)

    steps: List[Any] = []
    if strict_gt3_to_implicit:
        steps.append(RatingFilter(lower=4))
        steps.append(MakeImplicit(4))
    else:
        steps.append(MakeImplicit(1))
    steps.append(CorePruning(5))
    ds = Pipe(*steps).process(ds)
    return ds


def exact_user_holdout_df(df: pd.DataFrame, seed: int, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "user" not in df.columns or "item" not in df.columns:
        raise ValueError("Expected interaction columns 'user' and 'item'.")
    rng = np.random.default_rng(seed)
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    for _, udf in df.groupby("user", sort=False):
        udf = udf.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        n = len(udf)
        if n == 1:
            train_parts.append(udf)
            continue
        n_test = max(1, int(round(n * test_ratio)))
        if n_test >= n:
            n_test = n - 1
        test_parts.append(udf.iloc[:n_test])
        train_parts.append(udf.iloc[n_test:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train, test


def make_split_dataset(base_ds: Any, dataset_label: str, seed: int, working_dir: str) -> SimpleSplitData:
    set_random_state(seed)
    df = ensure_interaction_df(base_ds)
    train_df, test_df = exact_user_holdout_df(df, seed=seed, test_ratio=0.2)
    split_dir = os.path.join(working_dir, "splits", sanitize_name(dataset_label), f"seed_{seed}")
    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, "train_preview.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test_preview.csv"), index=False)
    return SimpleSplitData(dataset_label, train_df, test_df, split_dir)


def build_single_algo_plan(plan_name: str, algo_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name=plan_name)
    if algo_name == "Pop":
        plan.add_algorithm(LensKit.PopScorer, {})
    elif algo_name == "ItemKNN":
        plan.add_algorithm(LensKit.ItemKNNScorer, {"max_nbrs": 20})
    elif algo_name == "ALS":
        plan.add_algorithm(LensKit.ImplicitMFScorer, {"features": 20, "epochs": 10})
    else:
        raise ValueError(f"Unknown algo_name: {algo_name}")
    return plan


def extract_algorithm_family(algorithm_id: str) -> str:
    base = str(algorithm_id).split("-")[0]
    if base.startswith("LensKit."):
        base = base.split(".", 1)[1]
    mapping = {
        "ImplicitMFScorer": "ALS",
        "PopScorer": "Pop",
        "ItemKNNScorer": "ItemKNN",
    }
    return mapping.get(base, base)


def run_one_experiment(split_ds: Any, dataset_label: str, seed: int, algo_name: str, working_dir: str, original_cwd: str) -> pd.DataFrame:
    run_dir = os.path.join(working_dir, "runs", sanitize_name(dataset_label), f"seed_{seed}", sanitize_name(algo_name))
    os.makedirs(run_dir, exist_ok=True)
    out_csv = os.path.join(run_dir, "seed_results.csv")
    out_json = os.path.join(run_dir, "evaluator_results.json")
    if os.path.exists(out_csv):
        print(f"Skipping completed run: dataset={dataset_label} seed={seed} algo={algo_name}")
        return pd.read_csv(out_csv)

    os.chdir(run_dir)
    set_random_state(seed)
    evaluator = Evaluator(NDCG(KS), Precision(KS))
    plan = build_single_algo_plan(f"seed_sensitivity_{sanitize_name(dataset_label)}_{seed}_{algo_name}", algo_name)

    print(f"Running dataset={dataset_label} seed={seed} algo={algo_name}")
    run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
    evaluator.save_results(Path(out_json))

    result_frames: List[pd.DataFrame] = []
    for result_id, df in evaluator.get_results().items():
        tmp = df.copy()
        tmp["dataset_result_id"] = result_id
        tmp["dataset"] = dataset_label
        tmp["seed"] = seed
        tmp["algorithm_name"] = tmp["algorithm"].map(extract_algorithm_family)
        tmp["requested_algo"] = algo_name
        result_frames.append(tmp)

    os.chdir(original_cwd)
    if not result_frames:
        raise RuntimeError(f"No evaluation results returned for dataset={dataset_label}, seed={seed}, algo={algo_name}")

    res = pd.concat(result_frames, ignore_index=True)
    res.to_csv(out_csv, index=False)
    print(res)
    return res


def summarize_seed_sensitivity(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = results_df.copy()
    agg = (
        df.groupby(["dataset", "algorithm_name", "name", "k"], as_index=False)["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    agg.columns = ["dataset", "algorithm_name", "name", "k", "mean", "std", "min", "max", "count"]
    agg["range"] = agg["max"] - agg["min"]
    agg["cv"] = np.where(agg["mean"].abs() > 1e-12, agg["std"] / agg["mean"].abs(), np.nan)

    seed_level = df.groupby(["dataset", "algorithm_name", "name", "k", "seed"], as_index=False)["value"].mean()

    variability_by_algo_dataset = (
        agg.groupby(["dataset", "algorithm_name"], as_index=False)
        .agg(avg_seed_std=("std", "mean"), avg_seed_cv=("cv", "mean"), avg_seed_range=("range", "mean"))
        .sort_values(["dataset", "avg_seed_std"], ascending=[True, False])
    )
    return agg, seed_level, variability_by_algo_dataset


def paired_seed_comparisons(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, sub in seed_level_df.groupby(["dataset", "name", "k"]):
        dataset = sub["dataset"].iloc[0]
        metric_name = sub["name"].iloc[0]
        k = sub["k"].iloc[0]
        pivot = sub.pivot_table(index="seed", columns="algorithm_name", values="value", aggfunc="mean")
        algos = list(pivot.columns)
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                a, b = algos[i], algos[j]
                pair = pivot[[a, b]].dropna()
                if len(pair) == 0:
                    continue
                diff = pair[a] - pair[b]
                rows.append(
                    {
                        "dataset": dataset,
                        "name": metric_name,
                        "k": k,
                        "algo_a": a,
                        "algo_b": b,
                        "n_seeds": len(pair),
                        "mean_diff": float(diff.mean()),
                        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else np.nan,
                        "wins_a": int((diff > 0).sum()),
                        "wins_b": int((diff < 0).sum()),
                        "ties": int((diff == 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def statistical_summary(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = list(seed_level_df.groupby(["dataset", "algorithm_name", "name", "k"]))
    for key, sub in grouped:
        dataset, algo, metric_name, k = cast(Tuple[Any, Any, Any, Any], key)
        vals = sub["value"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "algorithm_name": algo,
                "name": metric_name,
                "k": k,
                "n_seeds": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "range": float(vals.max() - vals.min()),
                "cv": float(vals.std(ddof=1) / abs(vals.mean())) if len(vals) > 1 and abs(vals.mean()) > 1e-12 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def print_short_interpretation(agg: pd.DataFrame, variability: pd.DataFrame) -> None:
    print("\n=== Aggregated results across seeds ===")
    print(agg.sort_values(["dataset", "name", "k", "mean"], ascending=[True, True, True, False]).to_string(index=False))

    print("\n=== Seed sensitivity summary (lower std/cv => less sensitive) ===")
    print(variability.to_string(index=False))

    print("\n=== Short statistical interpretation ===")
    for dataset, sub in variability.groupby("dataset"):
        most_sensitive = sub.sort_values("avg_seed_std", ascending=False).iloc[0]
        least_sensitive = sub.sort_values("avg_seed_std", ascending=True).iloc[0]
        print(
            f"Dataset {dataset}: most seed-sensitive algorithm = {most_sensitive['algorithm_name']} "
            f"(avg std={most_sensitive['avg_seed_std']:.6f}, avg cv={most_sensitive['avg_seed_cv']:.6f}); "
            f"least seed-sensitive algorithm = {least_sensitive['algorithm_name']} "
            f"(avg std={least_sensitive['avg_seed_std']:.6f}, avg cv={least_sensitive['avg_seed_cv']:.6f})."
        )


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    original_cwd = os.getcwd()

    pd.DataFrame({"seed": SEEDS}).to_csv(os.path.join(working_dir, "split_seeds.csv"), index=False)

    dataset_specs = [
        (DataSet.MovieLens100K, True, "MovieLens100K"),
        (DataSet.Amazon2014VideoGames, True, "Amazon2014VideoGames"),
        (DataSet.HetrecLastFM, False, "HetrecLastFM"),
    ]
    algo_order = ["Pop", "ItemKNN", "ALS"]

    metadata_rows: List[Dict[str, Any]] = []
    base_datasets: Dict[str, Any] = {}

    for ds_enum, make_impl, label in dataset_specs:
        ds = load_and_preprocess_dataset(ds_enum, make_impl)
        base_datasets[label] = ds
        metadata_rows.append(
            {
                "dataset": label,
                "source_enum": str(ds_enum),
                "strict_gt3_to_implicit": make_impl,
                "core": 5,
                "implicit_conversion": True,
            }
        )

    pd.DataFrame(metadata_rows).to_csv(os.path.join(working_dir, "dataset_metadata.csv"), index=False)

    all_results: List[pd.DataFrame] = []
    for label, base_ds in base_datasets.items():
        for seed in SEEDS:
            split_ds = make_split_dataset(base_ds, label, seed, working_dir)
            print(split_ds)
            for algo_name in algo_order:
                os.chdir(original_cwd)
                try:
                    res = run_one_experiment(split_ds, label, seed, algo_name, working_dir, original_cwd)
                    all_results.append(res)
                except Exception as e:
                    os.chdir(original_cwd)
                    err_row = pd.DataFrame([
                        {
                            "dataset": label,
                            "seed": seed,
                            "algorithm_name": algo_name,
                            "name": "ERROR",
                            "k": np.nan,
                            "value": np.nan,
                            "error": str(e),
                        }
                    ])
                    err_path = os.path.join(working_dir, "runs", sanitize_name(label), f"seed_{seed}", sanitize_name(algo_name), "error.csv")
                    os.makedirs(os.path.dirname(err_path), exist_ok=True)
                    err_row.to_csv(err_path, index=False)
                    print(f"Run failed for dataset={label} seed={seed} algo={algo_name}: {e}")

    os.chdir(original_cwd)

    valid_results = [df for df in all_results if "value" in df.columns]
    if not valid_results:
        raise RuntimeError("No successful experiment results were produced.")

    results_df = pd.concat(valid_results, ignore_index=True)
    results_df = results_df[results_df["name"].isin(["NDCG", "Precision"])].copy()
    results_df.to_csv(os.path.join(working_dir, "all_seed_results.csv"), index=False)

    agg, seed_level, variability = summarize_seed_sensitivity(results_df)
    agg.to_csv(os.path.join(working_dir, "aggregated_results.csv"), index=False)
    seed_level.to_csv(os.path.join(working_dir, "seed_level_results.csv"), index=False)
    variability.to_csv(os.path.join(working_dir, "seed_sensitivity_summary.csv"), index=False)

    pairwise = paired_seed_comparisons(seed_level)
    pairwise.to_csv(os.path.join(working_dir, "pairwise_seed_comparisons.csv"), index=False)

    seed_effects = statistical_summary(seed_level)
    seed_effects.to_csv(os.path.join(working_dir, "seed_effect_sizes.csv"), index=False)

    print_short_interpretation(agg, variability)

    print("\n=== Pairwise seed-wise comparisons ===")
    if len(pairwise):
        print(pairwise.to_string(index=False))
    else:
        print("No pairwise comparisons available.")

    print("\n=== Seed effect summary ===")
    if len(seed_effects):
        print(seed_effects.to_string(index=False))
    else:
        print("No statistical analysis available.")

    report = {
        "seeds": SEEDS,
        "datasets": [x[2] for x in dataset_specs],
        "metrics": {"NDCG": KS, "Precision": KS},
        "algorithms": algo_order,
        "notes": [
            "Implemented via OmniRec exclusively, using OmniRec's LensKit runner algorithms.",
            "LensKit.ImplicitMFScorer is OmniRec's LensKit implicit ALS-equivalent algorithm.",
            "MovieLens100K and Amazon2014VideoGames use strict >3 filtering via RatingFilter(lower=4) before MakeImplicit(4).",
            "HetrecLastFM is explicitly converted to implicit feedback with MakeImplicit(1).",
            "Exact user-based 80/20 holdout is created with pandas after OmniRec preprocessing to satisfy the requested split while keeping all recommendation training/evaluation inside OmniRec.",
            "run_omnirec checkpointing and per-run CSV outputs allow safe resume after interruption.",
            "To meet runtime limits, expensive LensKit models use conservative standard-like settings: ItemKNN max_nbrs=20, ALS features=20 epochs=10.",
        ],
        "output_files": [
            "split_seeds.csv",
            "dataset_metadata.csv",
            "all_seed_results.csv",
            "aggregated_results.csv",
            "seed_level_results.csv",
            "seed_sensitivity_summary.csv",
            "pairwise_seed_comparisons.csv",
            "seed_effect_sizes.csv",
        ],
    }
    with open(os.path.join(working_dir, "experiment_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinished. All outputs saved under: {working_dir}")


if __name__ == "__main__":
    main()
