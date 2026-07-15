import os
import statistics
from collections import defaultdict

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

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

SEEDS = [7, 13, 29, 42, 101]
DATASETS = {
    'MovieLens100K': DataSet.MovieLens100K,
    'Amazon Video Games': DataSet.Amazon2014VideoGames,
    'Last.FM': DataSet.HetrecLastFM,
}


def preprocess_dataset(dataset_name, dataset_enum):
    dataset = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if dataset_name in {'MovieLens100K', 'Amazon Video Games'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    pipeline = Pipe(*steps)
    return pipeline.process(dataset)


def build_plan():
    plan = ExperimentPlan('Seed-Sensitivity-Study')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def summarize(values):
    values = [float(v) for v in values if pd.notna(v)]
    if not values:
        return {'mean': np.nan, 'std': np.nan, 'cv': np.nan, 'min': np.nan, 'max': np.nan}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = float(std / mean) if mean != 0 else np.nan
    return {'mean': mean, 'std': std, 'cv': cv, 'min': float(np.min(values)), 'max': float(np.max(values))}


def metric_rows_from_results(dataset_name, seed, results_dict):
    rows = []
    for dataset_id, df in results_dict.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({
                'dataset': dataset_name,
                'dataset_id': dataset_id,
                'seed': seed,
                'algorithm': row.get('algorithm'),
                'metric': row.get('name'),
                'k': row.get('k'),
                'value': row.get('value'),
            })
    return rows


def main():
    all_rows = []

    for dataset_name, dataset_enum in DATASETS.items():
        print(f'Loading and preprocessing {dataset_name}')
        processed = preprocess_dataset(dataset_name, dataset_enum)

        for seed in SEEDS:
            print(f'  Running seed={seed}')
            set_random_state(seed)
            split_dataset = UserHoldout(validation_size=0.0, test_size=0.2).process(processed)

            plan = build_plan()
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )
            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)

            result_rows = metric_rows_from_results(dataset_name, seed, evaluator.get_results())
            all_rows.extend(result_rows)

    results = pd.DataFrame(all_rows)
    results_path = os.path.join(working_dir, 'seed_effect_results.csv')
    results.to_csv(results_path, index=False)
    print(f'Saved per-run results to {results_path}')

    if results.empty:
        print('No results were collected.')
        return

    summary_rows = []
    grouped_results = results.groupby(['dataset', 'algorithm', 'metric', 'k'])
    for group_key, grp in grouped_results:
        dataset_name, algorithm, metric, k = group_key
        stats = summarize(grp['value'].tolist())
        summary_rows.append({
            'dataset': dataset_name,
            'algorithm': algorithm,
            'metric': metric,
            'k': k,
            **stats,
        })
    summary = pd.DataFrame(summary_rows).sort_values(['dataset', 'algorithm', 'metric', 'k'])
    summary_path = os.path.join(working_dir, 'seed_effect_summary.csv')
    summary.to_csv(summary_path, index=False)

    print('\nPer-dataset, per-algorithm, per-seed results:')
    print(results.sort_values(['dataset', 'algorithm', 'seed', 'metric', 'k']).to_string(index=False))

    print('\nSummary across seeds:')
    print(summary.to_string(index=False))

    print('\nShort statistical analysis:')
    analysis = []
    grouped_results = results.groupby(['dataset', 'algorithm', 'metric', 'k'])
    for group_key, grp in grouped_results:
        dataset_name, algorithm, metric, k = group_key
        analysis.append((dataset_name, algorithm, metric, k, float(np.std(grp['value'].tolist(), ddof=1)) if len(grp) > 1 else 0.0))
    analysis.sort(key=lambda x: x[-1], reverse=True)
    for dataset_name, algorithm, metric, k, std in analysis[:10]:
        print(f'- Highest variability candidate: {dataset_name} | {algorithm} | {metric}@{k} | std={std:.6f}')

    print('Seeds used:', SEEDS)


if __name__ == '__main__':
    main()
