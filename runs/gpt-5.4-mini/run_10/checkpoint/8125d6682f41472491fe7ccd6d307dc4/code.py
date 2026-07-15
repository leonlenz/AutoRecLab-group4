import os
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

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
from omnirec.util.util import set_random_state

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


@dataclass
class ExperimentRecord:
    dataset: str
    seed: int
    algorithm: str
    preprocessing: str
    split: str
    metric: str
    value: float


def preprocess_dataset(dataset_name: str) -> Any:
    ds = RecSysDataSet.use_dataloader(getattr(DataSet, dataset_name))
    pipeline_steps = [CorePruning(5)]
    if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        pipeline_steps.insert(0, MakeImplicit(3))
    pipeline = Pipe(*pipeline_steps)
    return pipeline.process(ds)


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name='SeedSensitivity-LensKit-Baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in results.columns if any(m in c for m in ['NDCG', 'Precision'])]
    group_cols = ['dataset', 'algorithm']
    summary = results.groupby(group_cols)[metric_cols].agg(['mean', 'std'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    summary = summary.reset_index()
    return summary


def main() -> None:
    seeds = [11, 22, 33, 44, 55]
    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']
    all_records: List[ExperimentRecord] = []
    all_run_tables: List[pd.DataFrame] = []

    plan = build_plan()
    evaluator = build_evaluator()

    for dataset_name in dataset_names:
        for seed in seeds:
            set_random_state(seed)
            processed = preprocess_dataset(dataset_name)
            split = UserHoldout(validation_size=0.0, test_size=0.2)
            processed = split.process(processed)
            run_omnirec(datasets=processed, plan=plan, evaluator=evaluator)

            result_path = os.path.join(working_dir, f'{dataset_name}_seed_{seed}_results.csv')
            if os.path.exists(result_path):
                df = pd.read_csv(result_path)
                all_run_tables.append(df)
                for _, row in df.iterrows():
                    for metric_col in [c for c in df.columns if 'NDCG' in c or 'Precision' in c]:
                        all_records.append(
                            ExperimentRecord(
                                dataset=dataset_name,
                                seed=seed,
                                algorithm=str(row.get('Algorithm', row.get('algorithm', 'unknown'))),
                                preprocessing='CorePruning(5);' + ('MakeImplicit(3)' if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'} else 'implicit-native'),
                                split='UserHoldout(validation=0.0,test=0.2)',
                                metric=metric_col,
                                value=float(row[metric_col]),
                            )
                        )

    if all_run_tables:
        combined = pd.concat(all_run_tables, ignore_index=True)
        combined.to_csv(os.path.join(working_dir, 'all_seed_results.csv'), index=False)
        summary = summarize_results(combined)
        summary.to_csv(os.path.join(working_dir, 'summary_mean_std.csv'), index=False)
        print(summary.to_string(index=False))

        analysis = []
        for _, row in summary.iterrows():
            metric_means = [v for k, v in row.items() if str(k).endswith('_mean') and pd.notnull(v)]
            metric_stds = [v for k, v in row.items() if str(k).endswith('_std') and pd.notnull(v)]
            if metric_means and metric_stds:
                avg_mean = float(np.mean(metric_means))
                avg_std = float(np.mean(metric_stds))
                cv = avg_std / avg_mean if avg_mean != 0 else np.nan
                analysis.append({'dataset': row['dataset'], 'algorithm': row['algorithm'], 'avg_metric_mean': avg_mean, 'avg_metric_std': avg_std, 'avg_cv': cv})
        pd.DataFrame(analysis).to_csv(os.path.join(working_dir, 'seed_variation_analysis.csv'), index=False)

    with open(os.path.join(working_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(
            {
                'datasets': dataset_names,
                'seeds': seeds,
                'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
                'preprocessing': {
                    'core_pruning': 5,
                    'implicit_threshold': 3,
                    'user_holdout': {'validation_size': 0.0, 'test_size': 0.2},
                },
                'metrics': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Precision@1', 'Precision@5', 'Precision@10'],
                'seed_control': 'omnirec global random state',
            },
            f,
            indent=2,
        )


if __name__ == '__main__':
    main()
