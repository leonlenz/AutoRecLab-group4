import os
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.filter import RatingFilter
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

SEEDS = [11, 23, 37, 47, 59]
KS = [1, 5, 10]
WORK_SUBDIR = "working"

DATASET_SPECS = [
    ("MovieLens100K", DataSet.MovieLens100K, True),
    ("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, True),
    ("HetrecLastFM", DataSet.HetrecLastFM, False),
]

ALGO_PREFIX_MAP = {
    "LensKit.PopScorer": "Pop",
    "LensKit.ItemKNNScorer": "ItemKNN",
    "LensKit.ImplicitMFScorer": "ALS",
}


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def normalize_algorithm_name(algo_string):
    algo_string = str(algo_string)
    for prefix, short in ALGO_PREFIX_MAP.items():
        if algo_string.startswith(prefix):
            return short
    return algo_string


def build_base_dataset(dataset_enum, convert_to_implicit):
    dataset = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if convert_to_implicit:
        steps.append(RatingFilter(lower=4))
        steps.append(MakeImplicit(1))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    return pipe.process(dataset)


def build_seed_split(dataset, seed):
    set_random_state(seed)
    splitter = UserHoldout(validation_size=0, test_size=0.2)
    return splitter.process(dataset)


def build_plan(plan_name):
    plan = ExperimentPlan(plan_name=plan_name)
    plan.add_algorithm(LensKit.PopScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer, {"feedback": "implicit"})
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    return plan


def load_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return pd.DataFrame(obj["data"])
        try:
            return pd.DataFrame(obj)
        except Exception:
            pass
    raise ValueError(f"Unsupported JSON prediction format in {path}")


def find_prediction_files(root_dir):
    return list(Path(root_dir).rglob("predictions.json"))


def infer_algorithm_from_prediction_path(path_obj):
    for part in path_obj.parts:
        for prefix, short in ALGO_PREFIX_MAP.items():
            if prefix in part:
                return short
    return None


def prepare_predictions(pred_df):
    cols = pred_df.columns.tolist()
    if "user" not in cols and "user_id" in cols:
        pred_df = pred_df.rename(columns={"user_id": "user"})
    if "item" not in cols and "item_id" in cols:
        pred_df = pred_df.rename(columns={"item_id": "item"})
    if "score" not in pred_df.columns:
        if "prediction" in pred_df.columns:
            pred_df = pred_df.rename(columns={"prediction": "score"})
        elif "rank" in pred_df.columns:
            pred_df = pred_df.rename(columns={"rank": "score"})
        else:
            raise ValueError(f"Prediction file missing score-like column: {pred_df.columns.tolist()}")
    pred_df = pred_df[["user", "item", "score"]].copy()
    pred_df["score"] = pd.to_numeric(pred_df["score"], errors="coerce")
    pred_df = pred_df.dropna(subset=["user", "item", "score"])
    return pred_df


def get_test_truth_from_split(split_dataset):
    test_df = split_dataset.get_test_data().copy()
    cols = test_df.columns.tolist()
    if "user" not in cols and "user_id" in cols:
        test_df = test_df.rename(columns={"user_id": "user"})
    if "item" not in cols and "item_id" in cols:
        test_df = test_df.rename(columns={"item_id": "item"})
    return test_df[["user", "item"]].drop_duplicates()


def precision_at_k_from_frames(pred_df, truth_df, ks):
    truth = truth_df.groupby("user")["item"].apply(set).to_dict()
    pred_df = pred_df.sort_values(["user", "score"], ascending=[True, False]).copy()
    rows = []
    for k in ks:
        vals = []
        for user, grp in pred_df.groupby("user", sort=False):
            rel = truth.get(user)
            if not rel:
                continue
            topk = grp.head(k)["item"].tolist()
            if len(topk) < k:
                continue
            hits = sum(1 for item in topk if item in rel)
            vals.append(hits / float(k))
        rows.append({"name": "Precision", "k": int(k), "value": float(np.mean(vals)) if vals else np.nan})
    return pd.DataFrame(rows)


def collect_precision_results(checkpoint_dir, truth_df, dataset_name, seed):
    pred_files = find_prediction_files(checkpoint_dir)
    rows = []
    seen = set()
    for pred_path in pred_files:
        algo = infer_algorithm_from_prediction_path(pred_path)
        if algo is None:
            continue
        try:
            pred_df = load_json_records(pred_path)
            pred_df = prepare_predictions(pred_df)
            metric_df = precision_at_k_from_frames(pred_df, truth_df, KS)
        except Exception:
            continue
        for _, row in metric_df.iterrows():
            key = (algo, int(row["k"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "algorithm": algo,
                    "name": row["name"],
                    "k": int(row["k"]),
                    "value": float(row["value"]),
                    "fold": np.nan,
                }
            )
    return pd.DataFrame(rows)


def collect_ndcg_results(evaluator, dataset_name, seed):
    results = evaluator.get_results()
    frames = []
    for dataset_id, df in results.items():
        tmp = df.copy()
        tmp["dataset_id"] = dataset_id
        tmp["dataset"] = dataset_name
        tmp["seed"] = seed
        frames.append(tmp)
    if not frames:
        empty_columns = pd.Index(["dataset", "dataset_id", "seed", "algorithm", "name", "k", "value", "fold"])
        return pd.DataFrame(columns=empty_columns)
    out = pd.concat(frames, ignore_index=True)
    out["algorithm"] = out["algorithm"].map(normalize_algorithm_name)
    out = out[out["name"].astype(str).str.lower() == "ndcg"].copy()
    out = out[["dataset", "dataset_id", "seed", "algorithm", "name", "k", "value", "fold"]]
    out = out.drop_duplicates(subset=["dataset", "seed", "algorithm", "name", "k", "value", "fold"])
    return out


def run_single_experiment(dataset_name, split_dataset, working_dir, seed):
    seed_dir = os.path.join(working_dir, f"{dataset_name}_seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    old_cwd = os.getcwd()
    os.chdir(seed_dir)
    try:
        plan = build_plan(f"seed-effect-{dataset_name}-seed-{seed}")
        evaluator = Evaluator(NDCG(KS))
        run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
        ndcg_df = collect_ndcg_results(evaluator, dataset_name, seed)
    finally:
        os.chdir(old_cwd)

    truth_df = get_test_truth_from_split(split_dataset)
    prec_df = collect_precision_results(seed_dir, truth_df, dataset_name, seed)

    if ndcg_df.empty and prec_df.empty:
        raise RuntimeError(f"No evaluation outputs discovered in {seed_dir}")

    frames = []
    if not ndcg_df.empty:
        frames.append(ndcg_df)
    if not prec_df.empty:
        if "dataset_id" not in prec_df.columns:
            prec_df["dataset_id"] = dataset_name
        frames.append(prec_df[["dataset", "dataset_id", "seed", "algorithm", "name", "k", "value", "fold"]])
    result_df = pd.concat(frames, ignore_index=True)
    result_df["algorithm"] = result_df["algorithm"].map(normalize_algorithm_name)
    return result_df.sort_values(["algorithm", "name", "k"]).reset_index(drop=True)


def compute_seed_statistics(df):
    rows = []
    grouped = df.groupby(["dataset", "algorithm", "name", "k"], dropna=False)
    for (dataset, algorithm, metric, k), grp in grouped:
        values = grp["value"].astype(float).dropna().to_numpy()
        if len(values) == 0:
            continue
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = std / sqrt(n) if n > 1 else np.nan
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
                "k": int(k),
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


def paired_tests(df):
    rows = []
    pairs = [("ALS", "ItemKNN"), ("ALS", "Pop"), ("ItemKNN", "Pop")]
    for (dataset, metric, k), grp in df.groupby(["dataset", "name", "k"]):
        pivot = grp.pivot_table(index="seed", columns="algorithm", values="value", aggfunc="first")
        for a, b in pairs:
            if a in pivot.columns and b in pivot.columns:
                sub = pivot[[a, b]].dropna()
                if len(sub) >= 2:
                    t_stat, p_val = stats.ttest_rel(sub[a], sub[b])
                    diff = float((sub[a] - sub[b]).mean())
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "k": int(k),
                            "algo_a": a,
                            "algo_b": b,
                            "n_seeds": int(len(sub)),
                            "mean_diff": diff,
                            "t_stat": float(t_stat),
                            "p_value": float(p_val),
                        }
                    )
    if not rows:
        empty_columns = pd.Index(["dataset", "metric", "k", "algo_a", "algo_b", "n_seeds", "mean_diff", "t_stat", "p_value"])
        return pd.DataFrame(columns=empty_columns)
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "k", "algo_a", "algo_b"]).reset_index(drop=True)


def short_analysis(summary_df):
    lines = []
    for dataset in summary_df["dataset"].drop_duplicates():
        ds = summary_df[summary_df["dataset"] == dataset].copy()
        lines.append(f"Dataset: {dataset}")
        rank = ds.groupby("algorithm", as_index=False)["mean"].mean().sort_values("mean", ascending=False)
        for _, r in rank.iterrows():
            sub = ds[ds["algorithm"] == r["algorithm"]]
            avg_std = sub["std"].mean()
            avg_cv = sub["cv"].mean()
            lines.append(
                f"  {r['algorithm']}: mean score across metrics={r['mean']:.6f}, avg std across seeds={avg_std:.6f}, avg CV={avg_cv:.6f}"
            )
        most_var = ds.sort_values("cv", ascending=False).head(1)
        if not most_var.empty:
            r = most_var.iloc[0]
            lines.append(
                f"  Highest relative seed sensitivity: {r['algorithm']} on {r['metric']}@{int(r['k'])} with CV={r['cv']:.6f}"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    working_dir = os.path.join(os.getcwd(), WORK_SUBDIR)
    os.makedirs(working_dir, exist_ok=True)

    all_results = []

    for dataset_name, dataset_enum, convert_to_implicit in DATASET_SPECS:
        print_section(f"Loading and preprocessing base dataset: {dataset_name}")
        base_dataset = build_base_dataset(dataset_enum, convert_to_implicit)

        for seed in SEEDS:
            print_section(f"Running dataset={dataset_name}, seed={seed}")
            split_dataset = build_seed_split(base_dataset, seed)
            result_df = run_single_experiment(dataset_name, split_dataset, working_dir, seed)
            result_df["algorithm"] = result_df["algorithm"].map(normalize_algorithm_name)
            all_results.append(result_df)
            print(result_df[["dataset", "seed", "algorithm", "name", "k", "value"]].sort_values(["algorithm", "name", "k"]).to_string(index=False))

    results_df = pd.concat(all_results, ignore_index=True)
    results_df["algorithm"] = results_df["algorithm"].map(normalize_algorithm_name)
    results_df = results_df[["dataset", "seed", "algorithm", "name", "k", "value", "fold", "dataset_id"]]
    results_df = results_df.sort_values(["dataset", "algorithm", "seed", "name", "k"]).reset_index(drop=True)

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

    test_df = paired_tests(results_df)
    tests_path = Path(working_dir) / "seed_effect_paired_tests.csv"
    test_df.to_csv(tests_path, index=False)

    variability_rank = (
        summary_df.groupby(["dataset", "algorithm"], as_index=False)["cv"]
        .mean()
        .rename(columns={"cv": "mean_cv_across_metrics"})
        .sort_values(["dataset", "mean_cv_across_metrics", "algorithm"])
    )
    variability_rank_path = Path(working_dir) / "seed_effect_variability_rank.csv"
    variability_rank.to_csv(variability_rank_path, index=False)

    print_section("Per-seed results")
    print(per_seed_table.to_string(index=False))

    print_section("Seed variability summary")
    print(summary_df.to_string(index=False))

    print_section("Paired t-tests across seeds")
    if len(test_df):
        print(test_df.to_string(index=False))
    else:
        print("No paired tests could be computed.")

    print_section("Algorithm ranking by average coefficient of variation across metrics")
    print(variability_rank.to_string(index=False))

    print_section("Short statistical analysis")
    print(short_analysis(summary_df))

    print_section("Saved files")
    for path in [results_path, per_seed_path, summary_path, tests_path, variability_rank_path]:
        print(str(path))


if __name__ == "__main__":
    main()
