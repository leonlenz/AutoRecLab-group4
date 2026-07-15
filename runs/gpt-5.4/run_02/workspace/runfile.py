import os
import json
import warnings
from importlib import import_module
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import SplitData
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.filter import RatingFilter
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state

warnings.filterwarnings("ignore")

SEEDS = [7, 19, 42, 77, 123]
KS = [1, 5, 10]


def sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def to_implicit_df(df: pd.DataFrame, strict_gt_3: bool) -> pd.DataFrame:
    out = df.copy()
    if strict_gt_3 and "rating" in out.columns:
        out = out[out["rating"] >= 4].copy()
    if "rating" in out.columns:
        out = out.drop(columns=["rating"])
    return out.reset_index(drop=True)


def load_and_preprocess_dataset(dataset_enum: Any, strict_gt_3: bool, label: str) -> Any:
    print(f"\nLoading dataset: {label}")
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    print(ds)

    if strict_gt_3:
        pipe = Pipe(RatingFilter(lower=4), CorePruning(5))
    else:
        pipe = Pipe(CorePruning(5))
    ds = pipe.process(ds)

    raw_df = ds._data.df.copy()
    raw_df = to_implicit_df(raw_df, strict_gt_3=False)
    ds = ds.replace_data(ds._data.__class__(raw_df))
    return ds


def user_holdout_split_df(df: pd.DataFrame, seed: int, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts: List[pd.DataFrame] = []
    valid_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    work = df.reset_index(drop=True).copy()
    if "user" not in work.columns or "item" not in work.columns:
        raise ValueError("Expected columns 'user' and 'item' in interactions dataframe.")

    for _, udf in work.groupby("user", sort=False):
        n = len(udf)
        if n < 2:
            continue
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_size)))
        n_test = min(n_test, n - 1)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        if len(train_idx) == 0:
            train_idx = idx[:-1]
            test_idx = idx[-1:]
        train_parts.append(udf.iloc[train_idx])
        test_parts.append(udf.iloc[test_idx])

    if not train_parts or not test_parts:
        raise RuntimeError("User-based split failed to create train/test partitions.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    valid_df = train_df.iloc[0:0].copy()
    return train_df, valid_df, test_df


def make_split_dataset(base_ds: Any, seed: int) -> Any:
    base_df = base_ds._data.df.copy()
    train_df, valid_df, test_df = user_holdout_split_df(base_df, seed=seed, test_size=0.2)
    split_data = SplitData(train_df, valid_df, test_df)
    return base_ds.replace_data(split_data)


def build_plan(plan_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name)
    plan.add_algorithm(LensKit.PopScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    return plan


def metric_precision_at_k(recs: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    if recs is None or len(recs) == 0:
        return float("nan")
    rec_cols = set(recs.columns)
    user_col = "user" if "user" in rec_cols else "user_id"
    item_col = "item" if "item" in rec_cols else "item_id"
    rank_col = None
    for c in ["rank", "Rank", "rnk"]:
        if c in rec_cols:
            rank_col = c
            break
    if rank_col is not None:
        topk = recs[recs[rank_col] <= k].copy()
    else:
        topk = recs.groupby(user_col, sort=False).head(k).copy()

    truth = test_df[["user", "item"]].drop_duplicates().copy()
    truth.columns = [user_col, item_col]
    merged = topk[[user_col, item_col]].merge(truth, on=[user_col, item_col], how="left", indicator=True)
    merged["hit"] = (merged["_merge"] == "both").astype(float)
    by_user = merged.groupby(user_col)["hit"].sum() / float(k)
    return float(by_user.mean()) if len(by_user) else float("nan")


def try_collect_precision_from_checkpoints(run_dir: str, split_ds: Any, dataset_label: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    test_df = split_ds._data.get("test").copy()

    for root, _, files in os.walk(run_dir):
        pred_files = [f for f in files if f.endswith(".csv") and ("pred" in f.lower() or "rec" in f.lower())]
        for pf in pred_files:
            path = os.path.join(root, pf)
            try:
                recs = pd.read_csv(path)
            except Exception:
                continue
            needed = set(recs.columns)
            if not ({"user", "item"}.issubset(needed) or {"user_id", "item_id"}.issubset(needed)):
                continue
            algo = os.path.basename(os.path.dirname(root))
            for k in KS:
                val = metric_precision_at_k(recs, test_df, k)
                rows.append(
                    {
                        "algorithm": algo,
                        "name": "Precision",
                        "k": k,
                        "value": val,
                        "dataset_result_id": f"{dataset_label}-{algo}",
                        "dataset": dataset_label,
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def run_one_experiment(split_ds: Any, dataset_label: str, seed: int, working_dir: str) -> pd.DataFrame:
    set_random_state(seed)
    current_seed = get_random_state()
    print(f"Running dataset={dataset_label} seed={current_seed}")

    run_dir = os.path.join(working_dir, "runs", sanitize_name(dataset_label), f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(run_dir)

    evaluator = Evaluator(NDCG(KS))
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

    ndcg_res = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    prec_res = try_collect_precision_from_checkpoints(run_dir, split_ds, dataset_label, seed)
    res = pd.concat([ndcg_res, prec_res], ignore_index=True) if len(prec_res) else ndcg_res
    if len(res) == 0:
        raise RuntimeError(f"No evaluation results returned for dataset={dataset_label}, seed={seed}")
    res.to_csv(os.path.join(run_dir, "seed_results.csv"), index=False)
    print(res)
    os.chdir(original_cwd)
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
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    original_cwd = os.getcwd()

    pd.DataFrame({"seed": SEEDS}).to_csv(os.path.join(working_dir, "split_seeds.csv"), index=False)

    dataset_specs = [
        (DataSet.MovieLens100K, True, "MovieLens100K"),
        (DataSet.Amazon2014VideoGames, True, "Amazon2014VideoGames"),
        (DataSet.HetrecLastFM, False, "HetrecLastFM"),
    ]

    metadata_rows: List[Dict[str, Any]] = []
    base_datasets: Dict[str, Any] = {}

    for ds_enum, strict_gt_3, label in dataset_specs:
        ds = load_and_preprocess_dataset(ds_enum, strict_gt_3, label)
        base_datasets[label] = ds
        base_df = ds._data.df.copy()
        metadata_rows.append(
            {
                "dataset": label,
                "source_enum": str(ds_enum),
                "strict_gt_3_to_implicit": strict_gt_3,
                "core": 5,
                "interactions_after_preprocess": len(base_df),
                "columns": ",".join(base_df.columns.astype(str).tolist()),
            }
        )

    pd.DataFrame(metadata_rows).to_csv(os.path.join(working_dir, "dataset_metadata.csv"), index=False)

    all_results: List[pd.DataFrame] = []
    for label, base_ds in base_datasets.items():
        for seed in SEEDS:
            os.chdir(original_cwd)
            split_ds = make_split_dataset(base_ds, seed)
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
        "seeds": SEEDS,
        "datasets": [x[2] for x in dataset_specs],
        "notes": [
            "Implemented via OmniRec exclusively, using OmniRec's LensKit runner algorithms.",
            "ALS is represented by OmniRec's LensKit.ImplicitMFScorer, which OmniRec documents as implicit-feedback matrix factorization (ALS).",
            "Amazon2014VideoGames is the required dataset enum available in OmniRec docs.",
            "Strict >3 conversion is implemented with RatingFilter(lower=4), then the rating column is dropped to create implicit interactions.",
            "A custom user-based 80/20 holdout is used because OmniRec UserHoldout crashes with validation_size=0.0.",
            "NDCG is computed by OmniRec Evaluator; Precision is additionally reconstructed from saved recommendation outputs when available.",
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


if __name__ == "__main__":
    main()