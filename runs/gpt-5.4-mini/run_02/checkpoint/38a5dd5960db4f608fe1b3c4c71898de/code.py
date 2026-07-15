import os
import json
import statistics as stats
from pathlib import Path

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


def build_dataset(dataset_enum, implicit_threshold=None, seed=42):
    set_random_state(seed)
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.append(CorePruning(5))
    steps.append(UserHoldout(0.1, 0.2))
    pipe = Pipe(*steps)
    return pipe.process(ds)


def make_plan():
    plan = ExperimentPlan(plan_name='SeedSensitivity_LensKit_Baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def normalize_results(results_dict, ds_name, seed):
    rows = []
    for _, df in results_dict.items():
        tmp = df.copy()
        tmp['dataset'] = ds_name
        tmp['seed'] = seed
        rows.append(tmp)
    return rows


def run_experiment():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    datasets = {
        'MovieLens100K': (DataSet.MovieLens100K, 3),
        'Amazon2014VideoGames': (DataSet.Amazon2014VideoGames, 3),
        'HetrecLastFM': (DataSet.HetrecLastFM, None),
    }
    seeds = [11, 22, 33, 44, 55]

    all_run_records = []
    metadata = []

    for ds_name, (ds_enum, thr) in datasets.items():
        print(f'\n=== Dataset: {ds_name} ===')
        for seed in seeds:
            print(f'-- Seed {seed}')
            dataset = build_dataset(ds_enum, implicit_threshold=thr, seed=seed)
            plan = make_plan()
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )
            run_omnirec(dataset, plan, evaluator)
            results = evaluator.get_results()
            all_run_records.extend(normalize_results(results, ds_name, seed))
            metadata.append({
                'dataset': ds_name,
                'seed': seed,
                'preprocessing': {
                    'implicit_threshold': thr,
                    'core_filter': 5,
                    'split': {'type': 'UserHoldout', 'validation_size': 0.1, 'test_size': 0.2},
                },
                'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
                'metrics': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Precision@1', 'Precision@5', 'Precision@10'],
            })
            print(results)

    out_dir = Path(working_dir)
    with (out_dir / 'experiment_metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    if not all_run_records:
        print('No results were produced.')
        return

    full = pd.concat(all_run_records, ignore_index=True)
    full.to_csv(out_dir / 'raw_results.csv', index=False)

    summary = full.groupby(['dataset', 'algorithm', 'name', 'k'], as_index=False)['value'].agg(['mean', 'std', 'count'])
    summary.to_csv(out_dir / 'summary_by_dataset_algorithm_metric.csv')
    print('\n=== Aggregated Summary ===')
    print(summary)

    print('\n=== Short Statistical Analysis ===')
    for (dataset, alg, metric, k), grp in full.groupby(['dataset', 'algorithm', 'name', 'k']):
        vals = grp['value'].tolist()
        mean_v = stats.mean(vals)
        std_v = stats.pstdev(vals) if len(vals) > 1 else 0.0
        cv = std_v / mean_v if mean_v != 0 else float('inf')
        print(f'{dataset} | {alg} | {metric}@{k}: mean={mean_v:.4f}, std={std_v:.4f}, cv={cv:.3f}')

    ndcg10 = full[(full['name'] == 'NDCG') & (full['k'] == 10)]
    pairwise = ndcg10.groupby(['dataset', 'algorithm'])['value'].mean().unstack()
    print('\n=== Pairwise Comparison on mean NDCG@10 ===')
    print(pairwise)


if __name__ == '__main__':
    run_experiment()
