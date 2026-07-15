import os
import json
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple, cast, runtime_checkable

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

try:
    from omnirec.metrics.ranking import Precision
except Exception as e:
    raise ImportError(
        "This experiment requires omnirec.metrics.ranking.Precision. "
        "The installed OmniRec build does not expose it, so the requested "
        "Precision@k evaluation cannot be run without deviating from the public OmniRec API."
    ) from e


SEEDS = [11, 22, 33, 44, 55]
KS = [1, 5, 10]


@runtime_checkable
class _HasDf(Protocol):
    df: pd.DataFrame


def make_working_dirs(base_dir: str) -> Dict[str, Path]:
    working_dir = Path(base_dir) / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    out_dir = working_dir / "seed_sensitivity_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "working": working_dir,
        "out": out_dir,
        "raw_results": out_dir / "raw_results.csv",
        "summary_results": out_dir / "summary_results.csv",
        "rank_variability": out_dir / "rank_variability.csv",
        "report": out_dir / "report.txt",
        "results_json": out_dir / "all_results.json",
    }


def strict_gt3_to_implicit(dataset: RecSysDataSet, dataset_name: str) -> RecSysDataSet:
    data = cast(Any, dataset)._data
    if not isinstance(data, _HasDf):
        raise ValueError(f"Expected raw dataset for {dataset_name}, but got non-raw variant: {type(data)}")

    df = data.df.copy()
    if "rating" not in df.columns:
        return dataset

    df = df[df["rating"] > 3].copy()
    if df.empty:
        raise ValueError(f"After filtering ratings > 3, dataset {dataset_name} became empty.")
    df["rating"] = 1

    return RecSysDataSet(df, meta=dataset.meta)


def ensure_implicit_lastfm(dataset: RecSysDataSet, dataset_name: str) -> RecSysDataSet:
    data = cast(Any, dataset)._data
    if not isinstance(data, _HasDf):
        raise ValueError(f"Expected raw dataset for {dataset_name}, but got non-raw variant: {type(data)}")

    df = data.df.copy()
    if "rating" not in df.columns:
        return dataset

    df["rating"] = 1
    return RecSysDataSet(df, meta=dataset.meta)


def load_and_prepare_raw(dataset_enum: DataSet, short_name: str) -> RecSysDataSet:
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    print(f"Loaded {short_name}: {ds}")

    if short_name in {"MovieLens100K", "Amazon2014VideoGames"}:
        ds = strict_gt3_to_implicit(ds, short_name)
    elif short_name == "HetrecLastFM":
        ds = ensure_implicit_lastfm(ds, short_name)
    else:
        raise ValueError(f"Unknown dataset short name: {short_name}")

    return ds


def preprocess_for_seed(raw_ds: RecSysDataSet, seed: int) -> RecSysDataSet:
    set_random_state(seed)
    pipe = Pipe(
        CorePruning(5),
        UserHoldout(0.0, 0.2),
    )
    return pipe.process(raw_ds)


def build_plan(seed: int) -> ExperimentPlan:
    plan = ExperimentPlan(f"seed_effect_baseline_seed_{seed}")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(
        NDCG(KS),
        Precision(KS),
    )


def collect_results(results_dict: Dict[str, pd.DataFrame], dataset_label: str, seed: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dataset_id, df in results_dict.items():
        tmp = df.copy()
        tmp["dataset_id"] = dataset_id
        tmp["dataset"] = dataset_label
        tmp["seed"] = seed
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=pd.Index(["algorithm", "fold", "name", "k", "value", "dataset_id", "dataset", "seed"]))
    out = pd.concat(frames, ignore_index=True)
    return out


def parse_algorithm_name(algorithm_value: str) -> str:
    if not isinstance(algorithm_value, str):
        return str(algorithm_value)
    return algorithm_value.split("-")[0]


def summarize_seed_effects(raw_results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = raw_results.copy()
    df["algorithm_base"] = df["algorithm"].map(parse_algorithm_name)

    summary = (
        df.groupby(["dataset", "algorithm_base", "name", "k"], dropna=False)["value"]
        .agg(["mean", "std", "min", "max", "median", "count"])
        .reset_index()
    )
    summary["cv"] = np.where(summary["mean"].abs() > 0, summary["std"] / summary["mean"].abs(), np.nan)
    summary["range"] = summary["max"] - summary["min"]

    rank_rows: List[pd.DataFrame] = []
    grouped = df.groupby(["dataset", "name", "k"], dropna=False)
    for group_key, g in grouped:
        dataset, metric, k = cast(Tuple[Any, Any, Any], group_key)
        pivot = g.pivot_table(index="seed", columns="algorithm_base", values="value", aggfunc="mean")
        if pivot.empty:
            continue
        ranks = pivot.rank(axis=1, ascending=False, method="average")
        rank_stats = pd.DataFrame({
            "rank_mean": ranks.mean(axis=0),
            "rank_std": ranks.std(axis=0),
            "rank_min": ranks.min(axis=0),
            "rank_max": ranks.max(axis=0),
        }).reset_index().rename(columns={"index": "algorithm_base"})
        rank_stats["dataset"] = dataset
        rank_stats["name"] = metric
        rank_stats["k"] = k
        rank_rows.append(rank_stats)

    rank_df = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame(
        columns=pd.Index(["algorithm_base", "rank_mean", "rank_std", "rank_min", "rank_max", "dataset", "name", "k"])
    )
    return summary, rank_df


def build_report(raw_results: pd.DataFrame, summary: pd.DataFrame, rank_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("Seed Sensitivity Experiment Report")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Total evaluation rows: {len(raw_results)}")
    lines.append(f"Seeds: {SEEDS}")
    lines.append("")

    for dataset in sorted(summary["dataset"].unique()):
        lines.append(f"Dataset: {dataset}")
        dsum = summary[summary["dataset"] == dataset].copy()
        for algo in sorted(dsum["algorithm_base"].unique()):
            lines.append(f"  Algorithm: {algo}")
            asub = dsum[dsum["algorithm_base"] == algo].sort_values(["name", "k"])
            for _, row in asub.iterrows():
                lines.append(
                    f"    {row['name']}@{int(row['k'])}: mean={row['mean']:.6f}, std={0.0 if pd.isna(row['std']) else row['std']:.6f}, "
                    f"min={row['min']:.6f}, max={row['max']:.6f}, cv={np.nan if pd.isna(row['cv']) else row['cv']:.6f}"
                )
        lines.append("")

    lines.append("Short statistical analysis")
    lines.append("-" * 40)
    if summary.empty:
        lines.append("No results available.")
        return "\n".join(lines)

    most_variable = summary.sort_values("cv", ascending=False, na_position="last").head(10)
    lines.append("Highest relative variability across seeds (top 10 by coefficient of variation):")
    for _, row in most_variable.iterrows():
        lines.append(
            f"  {row['dataset']} | {row['algorithm_base']} | {row['name']}@{int(row['k'])} | cv={np.nan if pd.isna(row['cv']) else row['cv']:.6f} | range={row['range']:.6f}"
        )

    lines.append("")
    lines.append("Ranking stability across seeds:")
    if rank_df.empty:
        lines.append("  No ranking-variability table available.")
    else:
        grouped_rank = rank_df.groupby(["dataset", "name", "k"], dropna=False)
        for group_key, g in grouped_rank:
            dataset, metric, k = cast(Tuple[Any, Any, Any], group_key)
            lines.append(f"  {dataset} | {metric}@{int(k)}")
            for _, row in g.sort_values("rank_mean").iterrows():
                lines.append(
                    f"    {row['algorithm_base']}: mean-rank={row['rank_mean']:.3f}, std-rank={0.0 if pd.isna(row['rank_std']) else row['rank_std']:.3f}, "
                    f"rank-range=[{row['rank_min']:.1f}, {row['rank_max']:.1f}]"
                )

    lines.append("")
    lines.append("Interpretation guidance:")
    lines.append("Smaller standard deviation, range, and coefficient of variation indicate lower sensitivity to the random holdout seed.")
    lines.append("If rank ranges are wide, algorithm conclusions depend more strongly on the chosen split seed.")
    return "\n".join(lines)


def main() -> None:
    paths = make_working_dirs(os.getcwd())

    datasets: List[Tuple[DataSet, str]] = [
        (DataSet.MovieLens100K, "MovieLens100K"),
        (DataSet.Amazon2014VideoGames, "Amazon2014VideoGames"),
        (DataSet.HetrecLastFM, "HetrecLastFM"),
    ]

    raw_prepared: Dict[str, RecSysDataSet] = {}
    for ds_enum, short_name in datasets:
        raw_prepared[short_name] = load_and_prepare_raw(ds_enum, short_name)

    all_results: List[pd.DataFrame] = []
    all_results_json: Dict[str, Dict[str, Any]] = {}

    for short_name in [name for _, name in datasets]:
        base_ds = raw_prepared[short_name]
        for seed in SEEDS:
            print(f"\n=== Running dataset={short_name}, seed={seed} ===")
            split_ds = preprocess_for_seed(base_ds, seed)
            print(split_ds.format_details())

            plan = build_plan(seed)
            evaluator = build_evaluator()
            set_random_state(seed)
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)

            result_dict = evaluator.get_results()
            run_df = collect_results(result_dict, short_name, seed)
            if run_df.empty:
                raise RuntimeError(f"No evaluation results returned for dataset={short_name}, seed={seed}")
            run_df["algorithm_base"] = run_df["algorithm"].map(parse_algorithm_name)
            all_results.append(run_df)

            all_results_json[f"{short_name}::seed::{seed}"] = {
                dsid: df.to_dict(orient="records") for dsid, df in result_dict.items()
            }

            per_seed_table = run_df.sort_values(["algorithm_base", "name", "k"])
            print(per_seed_table[["dataset", "seed", "algorithm_base", "name", "k", "value"]].to_string(index=False))

    raw_results = pd.concat(all_results, ignore_index=True)
    raw_results.to_csv(paths["raw_results"], index=False)

    summary, rank_df = summarize_seed_effects(raw_results)
    summary.to_csv(paths["summary_results"], index=False)
    rank_df.to_csv(paths["rank_variability"], index=False)

    report = build_report(raw_results, summary, rank_df)
    paths["report"].write_text(report, encoding="utf-8")

    with open(paths["results_json"], "w", encoding="utf-8") as f:
        json.dump(all_results_json, f, indent=2)

    print("\nSaved outputs:")
    print(paths["raw_results"])
    print(paths["summary_results"])
    print(paths["rank_variability"])
    print(paths["report"])
    print(paths["results_json"])
    print("\n" + report)


if __name__ == "__main__":
    main()
