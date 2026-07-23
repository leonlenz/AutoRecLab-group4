import os
import json
import statistics

import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import NDCG
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def build_dataset(dataset_name):
    if dataset_name == 'MovieLens100K':
        ds = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        pipe = Pipe(MakeImplicit(3), CorePruning(5), UserHoldout(0.2, 0.0))
    elif dataset_name == 'Amazon2014VideoGames':
        ds = RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
        pipe = Pipe(MakeImplicit(3), CorePruning(5), UserHoldout(0.2, 0.0))
    elif dataset_name == 'HetrecLastFM':
        ds = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
        pipe = Pipe(CorePruning(5), UserHoldout(0.2, 0.0))
    else:
        raise ValueError(f'Unexpected dataset: {dataset_name}')
    return pipe.process(ds)


def build_plan():
    plan = ExperimentPlan(plan_name='Seed Sensitivity Study')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def run_for_seed(seed, dataset_names, working_dir):
    set_random_state(seed)
    datasets = {name: build_dataset(name) for name in dataset_names}
    plan = build_plan()
    evaluator = Evaluator(
        NDCG([1, 5, 10])
    )
    result = run_omnirec(datasets=datasets, plan=plan, evaluator=evaluator)
    return result


def extract_results(result_obj, seed):
    if isinstance(result_obj, pd.DataFrame):
        df = result_obj.copy()
    elif hasattr(result_obj, 'to_dataframe'):
        df = result_obj.to_dataframe().copy()
    else:
        df = pd.DataFrame(result_obj)
    df['seed'] = seed
    return df


def summarize_results(df):
    metric_cols = [c for c in df.columns if any(m in c.lower() for m in ['precision', 'ndcg'])]
    group_cols = ['dataset', 'algorithm']
    summary = df.groupby(group_cols)[metric_cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    analysis_rows = []
    for (dataset, algo), grp in df.groupby(group_cols):
        row = {'dataset': dataset, 'algorithm': algo, 'n_seeds': grp['seed'].nunique()}
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors='coerce').dropna().tolist()
            if len(vals) >= 2:
                row[f'{col}_mean'] = float(statistics.mean(vals))
                row[f'{col}_std'] = float(statistics.stdev(vals))
                row[f'{col}_cv'] = float(statistics.stdev(vals) / statistics.mean(vals)) if statistics.mean(vals) else float('nan')
            elif len(vals) == 1:
                row[f'{col}_mean'] = float(vals[0])
                row[f'{col}_std'] = 0.0
                row[f'{col}_cv'] = 0.0
        analysis_rows.append(row)
    analysis = pd.DataFrame(analysis_rows)
    return summary, analysis


if __name__ == '__main__':
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']
    seeds = [11, 22, 33, 44, 55]

    all_results = []
    for seed in seeds:
        print(f'Running seed {seed}...')
        result = run_for_seed(seed, dataset_names, working_dir)
        seed_df = extract_results(result, seed)
        all_results.append(seed_df)
        seed_df.to_csv(os.path.join(working_dir, f'results_seed_{seed}.csv'), index=False)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(os.path.join(working_dir, 'all_results.csv'), index=False)

    summary, analysis = summarize_results(results)
    summary.to_csv(os.path.join(working_dir, 'summary_stats.csv'), index=False)
    analysis.to_csv(os.path.join(working_dir, 'seed_sensitivity_analysis.csv'), index=False)

    print('\n=== Summary Statistics ===')
    print(summary)
    print('\n=== Seed Sensitivity Analysis ===')
    print(analysis)

    if not analysis.empty:
        variability_cols = [c for c in analysis.columns if c.endswith('_cv')]
        if variability_cols:
            worst = analysis.loc[analysis[variability_cols].max(axis=1).idxmax()]
            print('\nMost seed-sensitive configuration:')
            print(worst[['dataset', 'algorithm'] + variability_cols].to_dict())

    report = {
        'seeds': seeds,
        'datasets': dataset_names,
        'algorithms': ['ALS', 'ItemKNN', 'Pop'],
        'metrics': ['NDCG@1/5/10']
    }
    with open(os.path.join(working_dir, 'run_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
