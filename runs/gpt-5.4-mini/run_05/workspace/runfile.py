import os
from collections import defaultdict

import pandas as pd

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
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


SEEDS = [11, 22, 33, 44, 55]
K_VALUES = [1, 5, 10]


def load_and_preprocess_dataset(dataset_name):
    raw = RecSysDataSet.use_dataloader(dataset_name)
    if dataset_name in (DataSet.MovieLens100K, DataSet.Amazon2014VideoGames):
        pipe = Pipe(
            MakeImplicit(3),
            CorePruning(5),
        )
    else:
        pipe = Pipe(
            CorePruning(5),
        )
    return pipe.process(raw)


def make_seeded_split(dataset, seed):
    set_random_state(seed)
    split_pipe = Pipe(
        UserHoldout(0.8, 0.2),
    )
    return split_pipe.process(dataset)


def build_experiment_plan():
    plan = ExperimentPlan(plan_name='SeedSensitivity_ImplicitTopK')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def summarize_results(results_df):
    summary = (
        results_df.groupby(['dataset', 'algorithm', 'name', 'k'], as_index=False)['value']
        .agg(['mean', 'std'])
        .reset_index()
    )
    summary.columns = ['dataset', 'algorithm', 'name', 'k', 'mean', 'std']
    summary['std'] = summary['std'].fillna(0.0)
    summary['cv'] = summary.apply(lambda r: 0.0 if r['mean'] == 0 else r['std'] / abs(r['mean']), axis=1)
    return summary


def main():
    datasets = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': DataSet.Amazon2014VideoGames,
        'HetrecLastFM': DataSet.HetrecLastFM,
    }

    all_run_rows = []
    per_run_tables = []

    plan = build_experiment_plan()

    for dataset_name, dataset_enum in datasets.items():
        print(f'Loading and preprocessing {dataset_name}...')
        base_dataset = load_and_preprocess_dataset(dataset_enum)

        for seed in SEEDS:
            print(f'Running {dataset_name} with seed={seed}...')
            split_dataset = make_seeded_split(base_dataset, seed)
            evaluator = Evaluator(NDCG(K_VALUES), Recall(K_VALUES))

            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)

            results_map = evaluator.get_results()
            for ds_key, df in results_map.items():
                df = df.copy()
                df['dataset'] = dataset_name
                df['seed'] = seed
                all_run_rows.append(df)
                per_run_tables.append((dataset_name, seed, df))
                print(f'[{dataset_name} | seed={seed}]')
                print(df.to_string(index=False))

    if not all_run_rows:
        print('No results were produced.')
        return

    results_df = pd.concat(all_run_rows, ignore_index=True)
    summary = summarize_results(results_df)

    print('\n=== Aggregated Seed-Sensitivity Summary ===')
    print(summary.sort_values(['dataset', 'algorithm', 'name', 'k']).to_string(index=False))

    print('\n=== Short Statistical Analysis ===')
    for (dataset, algorithm, metric, k), group in results_df.groupby(['dataset', 'algorithm', 'name', 'k']):
        values = group['value']
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        cv = 0.0 if mean == 0 else std / abs(mean)
        print(
            f'{dataset} | {algorithm} | {metric}@{k}: mean={mean:.4f}, std={std:.4f}, CV={cv:.3f} over {len(values)} seeds'
        )

    print('\nInterpretation: higher std/CV indicates stronger sensitivity to data-split randomness. '
          'Compare ALS, ItemKNN, and Pop across datasets to see where seed variation is largest.')


if __name__ == '__main__':
    main()