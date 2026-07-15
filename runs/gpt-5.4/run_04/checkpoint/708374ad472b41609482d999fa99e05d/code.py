import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_base_dataset(dataset_enum, make_implicit: bool) -> RecSysDataSet:
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    ds = pipe.process(ds)
    return ds


def build_seed_split(ds: RecSysDataSet, seed: int) -> RecSysDataSet:
    set_random_state(seed)
    splitter = UserHoldout(0.0, 0.2)
    return splitter.process(ds)


def build_plan(plan_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name=plan_name)
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer, {"feedback": "implicit"})
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def get_algo_family(algo_id: str) -> str:
    if algo_id.startswith("LensKit.ImplicitMFScorer"):
        return "ALS"
    if algo_id.startswith("LensKit.ItemKNNScorer"):
        return "ItemKNN"
    if algo_id.startswith("LensKit.PopScorer"):
        return "Pop"
    return algo_id


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }
    return table.get(df, 1.96)


def summarize_seed_variability(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["dataset", "algorithm", "metric", "k"], as_index=False)
    rows = []
    for _, g in grouped:
        first = g.iloc[0]
        dataset = first["dataset"]
        algorithm = first["algorithm"]
        metric = first["metric"]
        k = first["k"]
        values = g["value"].astype(float).to_numpy()
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        tcrit = t_critical_95(n - 1) if n > 1 else np.nan
        ci_low = mean - tcrit * sem if n > 1 else mean
        ci_high = mean + tcrit * sem if n > 1 else mean
        cv = std / mean if mean != 0 else np.nan
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "metric": metric,
                "k": int(k),
                "n_seeds": int(n),
                "mean": mean,
                "std": std,
                "cv": cv,
                "min": float(np.min(values)) if n else np.nan,
                "max": float(np.max(values)) if n else np.nan,
                "range": float(np.max(values) - np.min(values)) if n else np.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "k", "algorithm"]).reset_index(drop=True)


def pairwise_seed_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["dataset", "algorithm", "metric", "k"], as_index=False)
    for _, g in grouped:
        first = g.iloc[0]
        dataset = first["dataset"]
        algorithm = first["algorithm"]
        metric = first["metric"]
        k = first["k"]
        vals = g.sort_values("seed")["value"].astype(float).to_numpy()
        if len(vals) > 1:
            diffs = np.diff(vals)
            rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "metric": metric,
                    "k": int(k),
                    "mean_abs_consecutive_delta": float(np.mean(np.abs(diffs))),
                    "max_abs_consecutive_delta": float(np.max(np.abs(diffs))),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "k", "algorithm"]).reset_index(drop=True)


def cross_algo_ranking(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["dataset", "metric", "k", "seed"], as_index=False)
    for _, g in grouped:
        first = g.iloc[0]
        dataset = first["dataset"]
        metric = first["metric"]
        k = first["k"]
        seed = first["seed"]
        gg = g.sort_values("value", ascending=False).reset_index(drop=True)
        best = gg.iloc[0]
        worst = gg.iloc[-1]
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "k": int(k),
                "seed": int(seed),
                "best_algorithm": best["algorithm"],
                "best_value": float(best["value"]),
                "worst_algorithm": worst["algorithm"],
                "worst_value": float(worst["value"]),
                "spread": float(best["value"] - worst["value"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "k", "seed"]).reset_index(drop=True)


def print_compact_summary(agg: pd.DataFrame) -> None:
    print("\nAggregated mean ± std across seeds")
    for dataset in agg["dataset"].drop_duplicates().tolist():
        print(f"\n=== {dataset} ===")
        sub = agg[agg["dataset"] == dataset].copy()
        sub["summary"] = sub.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
        pivot = sub.pivot_table(index=["algorithm"], columns=["metric", "k"], values="summary", aggfunc="first")
        print(pivot)


def run_seed_experiment(
    dataset_name: str,
    dataset_enum,
    make_implicit: bool,
    seed: int,
    artifacts_dir: Path,
) -> pd.DataFrame:
    print(f"\n[{dataset_name}] Preparing seed={seed}")
    base_ds = build_base_dataset(dataset_enum, make_implicit)
    split_ds = build_seed_split(base_ds, seed)

    split_path = artifacts_dir / f"{dataset_name}_seed{seed}.rsds"
    split_ds.save(split_path)

    reloaded_ds = RecSysDataSet.load(split_path)
    plan = build_plan(f"seed_sensitivity_{dataset_name}_seed_{seed}")
    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    print(f"[{dataset_name}] Running OmniRec for seed={seed}")
    run_omnirec(datasets=reloaded_ds, plan=plan, evaluator=evaluator)

    results = evaluator.get_results()
    frames = []
    for dataset_id, df in results.items():
        temp = df.copy()
        temp["dataset_id"] = dataset_id
        frames.append(temp)
    if not frames:
        raise RuntimeError(f"No evaluation results returned for {dataset_name} seed={seed}")

    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = dataset_name
    out["seed"] = seed
    out["algorithm_full"] = out["algorithm"]
    out["algorithm"] = out["algorithm"].map(get_algo_family)
    out = out[["dataset", "dataset_id", "seed", "algorithm", "algorithm_full", "fold", "name", "k", "value"]]
    out = out.rename(columns={"name": "metric"})
    out = out.sort_values(["dataset", "seed", "algorithm", "metric", "k"]).reset_index(drop=True)

    print(f"[{dataset_name}] Completed seed={seed}")
    print(out[["algorithm", "metric", "k", "value"]].to_string(index=False))
    return out


def main() -> None:
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    work = Path(working_dir)
    results_dir = work / "results"
    splits_dir = work / "splits"
    ensure_dir(results_dir)
    ensure_dir(splits_dir)

    datasets: List[Tuple[str, object, bool]] = [
        ("MovieLens100K", DataSet.MovieLens100K, True),
        ("Amazon2014VideoGames", DataSet.Amazon2014VideoGames, True),
        ("HetrecLastFM", DataSet.HetrecLastFM, False),
    ]
    seeds = [11, 22, 33, 44, 55]

    all_runs = []
    for dataset_name, dataset_enum, make_implicit in datasets:
        for seed in seeds:
            run_df = run_seed_experiment(
                dataset_name=dataset_name,
                dataset_enum=dataset_enum,
                make_implicit=make_implicit,
                seed=seed,
                artifacts_dir=splits_dir,
            )
            all_runs.append(run_df)
            run_csv = results_dir / f"per_run_{dataset_name}_seed{seed}.csv"
            run_df.to_csv(run_csv, index=False)

    per_run = pd.concat(all_runs, ignore_index=True)
    per_run = per_run.sort_values(["dataset", "seed", "algorithm", "metric", "k"]).reset_index(drop=True)

    agg = summarize_seed_variability(per_run)
    deltas = pairwise_seed_deltas(per_run)
    rankings = cross_algo_ranking(per_run)

    per_run.to_csv(results_dir / "per_run_results.csv", index=False)
    agg.to_csv(results_dir / "aggregated_results.csv", index=False)
    deltas.to_csv(results_dir / "seed_delta_analysis.csv", index=False)
    rankings.to_csv(results_dir / "per_seed_algorithm_ranking.csv", index=False)

    print("\nSaved:")
    print(results_dir / "per_run_results.csv")
    print(results_dir / "aggregated_results.csv")
    print(results_dir / "seed_delta_analysis.csv")
    print(results_dir / "per_seed_algorithm_ranking.csv")

    print_compact_summary(agg)

    print("\nShort statistical analysis:")
    analysis = agg.copy()
    analysis["std_rank"] = analysis.groupby(["dataset", "metric", "k"])["std"].rank(method="dense", ascending=False)
    for dataset in analysis["dataset"].drop_duplicates().tolist():
        print(f"\nDataset: {dataset}")
        sub = analysis[analysis["dataset"] == dataset].sort_values(["metric", "k", "std"], ascending=[True, True, False])
        top = sub[["algorithm", "metric", "k", "mean", "std", "cv", "range", "ci95_low", "ci95_high"]]
        print(top.to_string(index=False))


if __name__ == '__main__':
    main()
