import os
import json
from dataclasses import dataclass, asdict
from typing import Any, List

import numpy as np
import pandas as pd
from scipy import stats

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
from omnirec.util.util import set_random_state

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


@dataclass
class RunRecord:
    dataset: str
    seed: int
    algorithm: str
    metric: str
    value: float


def preprocess_dataset(dataset_name: str) -> Any:
    ds = RecSysDataSet.use_dataloader(getattr(DataSet, dataset_name))
    steps = []
    if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name='SeedSensitivity-LensKit-Baselines')
    # ALS-style implicit matrix factorization in LensKit runner
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))


def extract_results(results_obj: Any, dataset_name: str, seed: int) -> List[RunRecord]:
    records: List[RunRecord] = []
    if isinstance(results_obj, dict):
        dfs = list(results_obj.values())
    else:
        dfs = [results_obj]

    for df in dfs:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if not {'algorithm', 'name', 'k', 'value'}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            metric_name = str(row['name'])
            k = row['k']
            metric = f"{metric_name}@{int(k)}" if pd.notnull(k) else metric_name
            records.append(
                RunRecord(
                    dataset=dataset_name,
                    seed=seed,
                    algorithm=str(row['algorithm']),
                    metric=metric,
                    value=float(row['value']),
                )
            )
    return records


def short_statistical_analysis(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, grp in results.groupby(['dataset', 'algorithm', 'metric']):
        dataset, algorithm, metric = grp.name
        vals = grp['value'].astype(float).values
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = std / mean if mean != 0 else np.nan
        if len(vals) > 1:
            ci = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=stats.sem(vals))
            p_vs_zero = float(stats.ttest_1samp(vals, 0.0).pvalue)
        else:
            ci = (np.nan, np.nan)
            p_vs_zero = np.nan
        rows.append(
            {
                'dataset': dataset,
                'algorithm': algorithm,
                'metric': metric,
                'mean': mean,
                'std': std,
                'cv': cv,
                'ci95_low': float(ci[0]) if np.isfinite(ci[0]) else np.nan,
                'ci95_high': float(ci[1]) if np.isfinite(ci[1]) else np.nan,
                'p_value_one_sample_vs_0': p_vs_zero,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    seeds = [11, 22, 33, 44, 55]
    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']

    plan = build_plan()
    evaluator = build_evaluator()

    all_records: List[RunRecord] = []
    run_metadata = []

    for dataset_name in dataset_names:
        for seed in seeds:
            set_random_state(seed)
            processed = preprocess_dataset(dataset_name)
            split = UserHoldout(validation_size=0.0, test_size=0.2)
            split_ds = split.process(processed)
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)

            results_obj = evaluator.get_results()
            all_records.extend(extract_results(results_obj, dataset_name, seed))

            run_metadata.append(
                {
                    'dataset': dataset_name,
                    'seed': seed,
                    'preprocessing': 'MakeImplicit(3) for MovieLens100K/Amazon2014VideoGames; CorePruning(5)',
                    'split': 'UserHoldout(validation_size=0.0, test_size=0.2)',
                    'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
                    'metrics': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Precision@1', 'Precision@5', 'Precision@10'],
                }
            )

    results_df = pd.DataFrame([asdict(r) for r in all_records])
    results_df.to_csv(os.path.join(working_dir, 'all_seed_results_long.csv'), index=False)

    if not results_df.empty:
        summary = results_df.groupby(['dataset', 'algorithm', 'metric'])['value'].agg(['mean', 'std', 'count']).reset_index()
        summary.to_csv(os.path.join(working_dir, 'summary_mean_std.csv'), index=False)
        stats_df = short_statistical_analysis(results_df)
        stats_df.to_csv(os.path.join(working_dir, 'seed_variation_analysis.csv'), index=False)
        print(summary.to_string(index=False))
        print('\nShort statistical analysis:')
        print(stats_df.to_string(index=False))
    else:
        print('No results were collected. Check OmniRec evaluator results and dataset preprocessing.')

    with open(os.path.join(working_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(
            {
                'datasets': dataset_names,
                'seeds': seeds,
                'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
                'preprocessing': {
                    'MovieLens100K': ['MakeImplicit(3)', 'CorePruning(5)'],
                    'Amazon2014VideoGames': ['MakeImplicit(3)', 'CorePruning(5)'],
                    'HetrecLastFM': ['CorePruning(5)'],
                },
                'split': {'type': 'UserHoldout', 'validation_size': 0.0, 'test_size': 0.2},
                'metrics': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Precision@1', 'Precision@5', 'Precision@10'],
                'seed_control': 'omnirec global random state',
            },
            f,
            indent=2,
        )


if __name__ == '__main__':
    main()
