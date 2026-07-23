import os
import json
import warnings
from importlib import import_module
from typing import Any, Dict, List, Tuple

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
from omnirec.util.util import set_random_state, get_random_state

warnings.filterwarnings("ignore")


def sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def load_and_preprocess_dataset(dataset_enum: Any, make_implicit: bool, working_dir: str) -> Any:
    print(f"\nLoading dataset: {dataset_enum}")
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    print(ds)
    steps: List[Any] = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    ds = pipe.process(ds)
    out_path = os.path.join(working_dir, f"preprocessed_{sanitize_name(dataset_enum)}")
    try:
        ds.save(out_path)
    except Exception as e:
        print(f"Warning: could not save preprocessed dataset for {dataset_enum}: {e}")
    return ds


def make_user_holdout_dataset(base_ds: Any, seed: int) -> Any:
    set_random_state(seed)
    splitter = UserHoldout(validation_size=0.0, test_size=0.2)
    split_ds = splitter.process(base_ds)
    return split_ds


def build_plan(plan_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name)
    plan.add_algorithm(LensKit.PopScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    return plan


def parse_dataset_name_from_result_id(result_id: str, known_names: List[str]) -> str:
    for name in known_names:
        if str(name) in result_id:
            return str(name)
    return result_id.split("-")[0]


def run_one_experiment(split_ds: Any, dataset_label: str, seed: int, working_dir: str) -> pd.DataFrame:
    set_random_state(seed)
    current_seed = get_random_state()
    print(f"Running dataset={dataset_label} seed={current_seed}")

    run_dir = os.path.join(working_dir, "runs", sanitize_name(dataset_label), f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )
    plan = build_plan(f"seed_sensitivity_{sanitize_name(dataset_label)}_{seed}")

    run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)

    result_frames: List[pd.DataFrame] = []
    results_dict = evaluator.get_results()
    for result_id, df in results_dict.items():
        tmp = df.copy()
        tmp["dataset_result_id"] = result_id
        tmp["dataset"] = dataset_label
        tmp["seed"] = seed
        result_frames.append(tmp)

    if not result_frames:
        raise RuntimeError(f"No evaluation results returned for dataset={dataset_label}, seed={seed}")

    res = pd.concat(result_frames, ignore_index=True)
    res.to_csv(os.path.join(run_dir, "seed_results.csv"), index=False)
    print(res)
    return res


def extract_algorithm_family(algorithm_id: str) -> str:
    base = str(algorithm_id).split("-")[0]
    if base.startswith("LensKit."):
        base = base.split(".", 1)[1]
    return base


def summarize_seed_sensitivity(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = results_df.copy()
    df["algorithm_family"] = df["algorithm"].map(extract_algorithm_family)

    agg = (
        df.groupby(["dataset", "algorithm_family", "name", "k"], as_index=False)["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    agg.columns = ["dataset", "algorithm_family", "name", "k", "mean", "std", "min", "max", "count"]
    agg["range"] = agg["max"] - agg["min"]
    agg["cv"] = np.where(agg["mean"].abs() > 1e-12, agg["std"] / agg["mean"].abs(), np.nan)

    seed_level = (
        df.groupby(["dataset", "algorithm_family", "name", "k", "seed"], as_index=False)["value"]
        .mean()
    )

    variability_by_algo_dataset = (
        agg.groupby(["dataset", "algorithm_family"], as_index=False)
        .agg(
            avg_seed_std=("std", "mean"),
            avg_seed_cv=("cv", "mean"),
            avg_seed_range=("range", "mean"),
        )
        .sort_values(["dataset", "avg_seed_std"], ascending=[True, False])
    )

    return agg, seed_level, variability_by_algo_dataset


def paired_seed_comparisons(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = list(seed_level_df.groupby(["dataset", "name", "k"]))
    for _, sub in grouped:
        dataset = sub["dataset"].iloc[0]
        metric_name = sub["name"].iloc[0]
        k = sub["k"].iloc[0]
        pivot = sub.pivot_table(index="seed", columns="algorithm_family", values="value", aggfunc="mean")
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


def run_ols_if_available(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    try:
        smf = import_module("statsmodels.formula.api")
        sm = import_module("statsmodels.api")
    except Exception:
        rows: List[Dict[str, Any]] = []
        grouped = list(seed_level_df.groupby(["name", "k"]))
        for _, sub in grouped:
            metric_name = sub["name"].iloc[0]
            k = sub["k"].iloc[0]
            grand_mean = sub["value"].mean()
            ss_total = ((sub["value"] - grand_mean) ** 2).sum()
            algo_means = sub.groupby("algorithm_family")["value"].mean()
            dataset_means = sub.groupby("dataset")["value"].mean()
            ss_algo = sum(sub.groupby("algorithm_family").size()[a] * (m - grand_mean) ** 2 for a, m in algo_means.items())
            ss_dataset = sum(sub.groupby("dataset").size()[d] * (m - grand_mean) ** 2 for d, m in dataset_means.items())
            rows.append(
                {
                    "name": metric_name,
                    "k": k,
                    "method": "descriptive_variance_decomposition",
                    "ss_total": float(ss_total),
                    "ss_algorithm": float(ss_algo),
                    "ss_dataset": float(ss_dataset),
                    "prop_algorithm": float(ss_algo / ss_total) if ss_total > 0 else np.nan,
                    "prop_dataset": float(ss_dataset / ss_total) if ss_total > 0 else np.nan,
                }
            )
        return pd.DataFrame(rows)

    out: List[pd.DataFrame] = []
    grouped = list(seed_level_df.groupby(["name", "k"]))
    for _, sub in grouped:
        metric_name = sub["name"].iloc[0]
        k = sub["k"].iloc[0]
        try:
            model = smf.ols("value ~ C(dataset) + C(algorithm_family)", data=sub).fit()
            anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
            anova["name"] = metric_name
            anova["k"] = k
            anova["method"] = "ols_anova_type2"
            out.append(anova)
        except Exception as e:
            out.append(pd.DataFrame([{
                "term": "error",
                "sum_sq": np.nan,
                "df": np.nan,
                "F": np.nan,
                "PR(>F)": np.nan,
                "name": metric_name,
                "k": k,
                "method": f"ols_failed: {e}",
            }]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


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
            f"Dataset {dataset}: most seed-sensitive algorithm = {most_sensitive['algorithm_family']} "
            f"(avg std={most_sensitive['avg_seed_std']:.6f}, avg cv={most_sensitive['avg_seed_cv']:.6f}); "
            f"least seed-sensitive algorithm = {least_sensitive['algorithm_family']} "
            f"(avg std={least_sensitive['avg_seed_std']:.6f}, avg cv={least_sensitive['avg_seed_cv']:.6f})."
        )


def main() -> None:
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    original_cwd = os.getcwd()

    seed_list = [7, 19, 42, 77, 123]
    pd.DataFrame({"seed": seed_list}).to_csv(os.path.join(working_dir, "split_seeds.csv"), index=False)

    dataset_specs = [
        (DataSet.MovieLens100K, True, "MovieLens100K"),
        (DataSet.Amazon2023VideoGames, True, "Amazon2023VideoGames"),
        (DataSet.HetrecLastFM, False, "HetrecLastFM"),
    ]

    metadata_rows: List[Dict[str, Any]] = []
    base_datasets: Dict[str, Any] = {}

    for ds_enum, make_impl, label in dataset_specs:
        ds = load_and_preprocess_dataset(ds_enum, make_impl, working_dir)
        base_datasets[label] = ds
        metadata_rows.append({
            "dataset": label,
            "source_enum": str(ds_enum),
            "make_implicit": make_impl,
            "core": 5,
        })

    pd.DataFrame(metadata_rows).to_csv(os.path.join(working_dir, "dataset_metadata.csv"), index=False)

    all_results: List[pd.DataFrame] = []
    for label, base_ds in base_datasets.items():
        for seed in seed_list:
            os.chdir(original_cwd)
            split_ds = make_user_holdout_dataset(base_ds, seed)
            res = run_one_experiment(split_ds, label, seed, working_dir)
            all_results.append(res)

    os.chdir(original_cwd)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df["algorithm_family"] = results_df["algorithm"].map(extract_algorithm_family)
    results_df.to_csv(os.path.join(working_dir, "all_seed_results.csv"), index=False)

    agg, seed_level, variability = summarize_seed_sensitivity(results_df)
    agg.to_csv(os.path.join(working_dir, "aggregated_results.csv"), index=False)
    seed_level.to_csv(os.path.join(working_dir, "seed_level_results.csv"), index=False)
    variability.to_csv(os.path.join(working_dir, "seed_sensitivity_summary.csv"), index=False)

    pairwise = paired_seed_comparisons(seed_level)
    pairwise.to_csv(os.path.join(working_dir, "pairwise_seed_comparisons.csv"), index=False)

    anova_df = run_ols_if_available(seed_level)
    anova_df.to_csv(os.path.join(working_dir, "statistical_analysis.csv"), index=False)

    print_short_interpretation(agg, variability)

    print("\n=== Pairwise seed-wise comparisons ===")
    if len(pairwise):
        print(pairwise.to_string(index=False))
    else:
        print("No pairwise comparisons available.")

    print("\n=== Statistical analysis table ===")
    if len(anova_df):
        print(anova_df.to_string(index=False))
    else:
        print("No statistical analysis available.")

    report = {
        "seeds": seed_list,
        "datasets": [x[2] for x in dataset_specs],
        "notes": [
            "Implemented via OmniRec exclusively, using OmniRec's LensKit runner algorithms.",
            "Amazon2023VideoGames is used because that is the available OmniRec built-in Video Games dataset enum in this environment.",
            "MovieLens100K and Amazon2023VideoGames are converted to implicit with MakeImplicit(3), which keeps ratings >= 3 per OmniRec docs.",
            "UserHoldout(validation_size=0.0, test_size=0.2) is used to approximate a pure 80/20 user-based holdout.",
        ],
        "output_files": [
            "split_seeds.csv",
            "dataset_metadata.csv",
            "all_seed_results.csv",
            "aggregated_results.csv",
            "seed_level_results.csv",
            "seed_sensitivity_summary.csv",
            "pairwise_seed_comparisons.csv",
            "statistical_analysis.csv",
        ],
    }
    with open(os.path.join(working_dir, "experiment_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinished. All outputs saved under: {working_dir}")


if __name__ == '__main__':
    main()
