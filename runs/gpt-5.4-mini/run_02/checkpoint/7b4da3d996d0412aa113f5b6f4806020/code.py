import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


SEEDS = [11, 22, 33, 44, 55]
K_VALUES = [1, 5, 10]


def load_and_preprocess(dataset_enum, implicit_threshold=None):
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def user_holdout_80_20(dataset, seed):
    set_random_state(seed)
    df = dataset._data.df.copy()
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for _, grp in df.groupby('user', sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            train_parts.append(grp)
            continue
        perm = rng.permutation(idx)
        n_test = max(1, int(round(len(perm) * 0.2)))
        if n_test >= len(perm):
            n_test = len(perm) - 1
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        test_parts.append(df.loc[test_idx])
        train_parts.append(df.loc[train_idx])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()
    return train_df, test_df


def normalize_results(results, dataset_name, seed):
    records = []
    if isinstance(results, dict):
        items = results.items()
    else:
        items = [(None, results)]
    for _, df in items:
        tmp = df.copy()
        tmp['dataset'] = dataset_name
        tmp['seed'] = seed
        records.append(tmp)
    return records


def summarize(all_results):
    summary = all_results.groupby(['dataset', 'algorithm', 'name', 'k'], as_index=False)['value'].agg(['mean', 'std']).reset_index()
    summary['cv'] = summary['std'] / summary['mean'].replace(0, np.nan)
    summary.columns = ['dataset', 'algorithm', 'name', 'k', 'mean', 'std', 'cv']
    return summary


def paired_seed_analysis(all_results):
    lines = []
    for dataset in all_results['dataset'].unique():
        ds_df = all_results[all_results['dataset'] == dataset]
        for metric_name in ['NDCG', 'Precision']:
            for k in K_VALUES:
                pivot = ds_df[(ds_df['name'] == metric_name) & (ds_df['k'] == k)].pivot_table(
                    index='seed', columns='algorithm', values='value', aggfunc='mean'
                )
                lines.append(f'{dataset} | {metric_name}@{k}')
                lines.append(str(pivot))
                algs = [c for c in pivot.columns if pivot[c].notna().any()]
                if len(algs) >= 2:
                    base = algs[0]
                    for other in algs[1:]:
                        diff = (pivot[other] - pivot[base]).dropna()
                        if len(diff) > 1:
                            tstat, pval = stats.ttest_1samp(diff, 0.0)
                            lines.append(
                                f'paired diff {other} - {base}: mean={diff.mean():.4f}, std={diff.std(ddof=1):.4f}, t={tstat:.3f}, p={pval:.4f}'
                            )
                lines.append('')
    return '\n'.join(lines)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    datasets = {
        'MovieLens100K': (DataSet.MovieLens100K, 3),
        'Amazon2014VideoGames': (DataSet.Amazon2014VideoGames, 3),
        'HetrecLastFM': (DataSet.HetrecLastFM, None),
    }

    plan = ExperimentPlan('seed_sensitivity_benchmark')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    provenance = []
    all_records = []

    for ds_name, (ds_enum, thr) in datasets.items():
        base_ds = load_and_preprocess(ds_enum, thr)
        for seed in SEEDS:
            set_random_state(seed)
            train_df, test_df = user_holdout_80_20(base_ds, seed)
            split_ds = base_ds.replace_data(type(base_ds._data)(train_df, None, test_df) if hasattr(base_ds._data, '__class__') else base_ds)
            # Fallback to direct dataframe-based split if replace_data expects SplitData
            try:
                split_ds = base_ds.replace_data(type(base_ds._data)(train_df, None, test_df))
            except Exception:
                pass

            set_random_state(seed)
            run_omnirec(split_ds, plan, evaluator)
            results = evaluator.get_results()
            all_records.extend(normalize_results(results, ds_name, seed))

            provenance.append({
                'dataset': ds_name,
                'seed': seed,
                'preprocessing': [f'MakeImplicit({thr})' if thr is not None else 'implicit feedback kept', 'CorePruning(5)'],
                'split': 'user-based 80/20 holdout',
                'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
                'metrics': [f'NDCG@{k}' for k in K_VALUES] + [f'Precision@{k}' for k in K_VALUES],
            })

    all_results = pd.concat(all_records, ignore_index=True)
    all_results.to_csv(Path(working_dir) / 'seed_sensitivity_results.csv', index=False)
    summary = summarize(all_results)
    summary.to_csv(Path(working_dir) / 'summary.csv', index=False)

    with open(Path(working_dir) / 'provenance.json', 'w') as f:
        json.dump(provenance, f, indent=2)

    print(summary.sort_values(['dataset', 'algorithm', 'name', 'k']))
    print('\nSeed sensitivity analysis (paired across seeds):')
    print(paired_seed_analysis(all_results))


if __name__ == '__main__':
    main()
