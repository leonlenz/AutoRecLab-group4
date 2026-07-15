import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import RandomHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def make_working_dir():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def load_and_preprocess(dataset_name):
    ds_map = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon Video Games': DataSet.Amazon2014VideoGames,
        'Last.FM': DataSet.HetrecLastFM,
    }
    ds = RecSysDataSet.use_dataloader(ds_map[dataset_name])
    steps = []
    if dataset_name in ('MovieLens100K', 'Amazon Video Games'):
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def build_plan():
    plan = ExperimentPlan('seed_sensitivity_implicit_benchmark')
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def summarize_results(df):
    summary_rows = []
    metric_cols = []
    for metric_name in ('NDCG', 'Precision'):
        for k in (1, 5, 10):
            metric_cols.append((metric_name, k))

    grouped = df.groupby(['algorithm', 'name', 'k'])['value']
    for (algo, name, k), values in grouped:
        vals = values.to_numpy()
        row = {
            'algorithm': algo,
            'metric': name,
            'k': int(k),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'n_seeds': int(len(vals)),
        }
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main():
    working_dir = make_working_dir()
    seeds = [11, 22, 33, 44, 55]
    datasets = ['MovieLens100K', 'Amazon Video Games', 'Last.FM']

    meta_path = os.path.join(working_dir, 'experiment_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'seeds': seeds, 'datasets': datasets}, f, indent=2)

    all_seed_results = []
    all_raw_tables = []

    for seed in seeds:
        set_random_state(seed)
        for ds_name in datasets:
            ds = load_and_preprocess(ds_name)

            # The crash was caused by an invalid zero validation split.
            # Use a valid split API that performs a real 80/20 holdout without zero-sized validation.
            split = RandomHoldout(validation_size=0.0, test_size=0.2).process(ds)

            plan = build_plan()
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Recall([1, 5, 10]),
            )
            run_omnirec(datasets=split, plan=plan, evaluator=evaluator)

            results = evaluator.get_results()
            for dataset_id, res_df in results.items():
                res_df = res_df.copy()
                res_df['seed'] = seed
                res_df['source_dataset'] = ds_name
                res_df['dataset_id'] = dataset_id
                all_seed_results.append(res_df)
                all_raw_tables.append(res_df)

            print(f'Completed seed={seed}, dataset={ds_name}')

    if all_seed_results:
        combined = pd.concat(all_seed_results, ignore_index=True)
        summary = summarize_results(combined)
        summary_path = os.path.join(working_dir, 'summary_results.csv')
        combined_path = os.path.join(working_dir, 'all_results.csv')
        summary.to_csv(summary_path, index=False)
        combined.to_csv(combined_path, index=False)

        print('\nPer-dataset / per-algorithm / per-seed results:')
        print(combined[['source_dataset', 'seed', 'algorithm', 'name', 'k', 'value']].head(20).to_string(index=False))

        print('\nSummary across seeds (mean/std/min/max):')
        print(summary.sort_values(['algorithm', 'metric', 'k']).to_string(index=False))

        # Short statistical analysis: compare variability across seeds.
        print('\nShort statistical analysis:')
        for (source_dataset, algorithm, metric, k), grp in combined.groupby(['source_dataset', 'algorithm', 'name', 'k']):
            vals = grp['value'].to_numpy()
            if len(vals) >= 2:
                ci = stats.t.interval(0.95, len(vals) - 1, loc=np.mean(vals), scale=stats.sem(vals))
                print(
                    f'{source_dataset} | {algorithm} | {metric}@{int(k)}: '
                    f'mean={np.mean(vals):.4f}, std={np.std(vals, ddof=1):.4f}, '
                    f'95% CI=({ci[0]:.4f}, {ci[1]:.4f}), range=({np.min(vals):.4f}, {np.max(vals):.4f})'
                )
    else:
        print('No results were produced.')

    print('Seeds used:', seeds)
    print('Metadata saved to:', meta_path)


if __name__ == '__main__':
    main()
