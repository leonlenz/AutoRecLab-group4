import os
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
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


def normalize_results(df):
    out = df.copy()
    if 'algorithm' not in out.columns:
        return out
    out['algorithm'] = out['algorithm'].astype(str).str.replace(r'-[0-9a-fA-F]+$', '', regex=True)
    return out


def summarize(all_rows):
    df = pd.DataFrame(all_rows)
    summary = (
        df.groupby(['dataset', 'algorithm', 'name', 'k'])['value']
        .agg(['mean', 'std'])
        .reset_index()
        .sort_values(['dataset', 'algorithm', 'name', 'k'])
    )
    return df, summary


def short_statistical_analysis(df):
    lines = []
    for (dataset, metric, k), g in df.groupby(['dataset', 'name', 'k']):
        pivot = g.pivot_table(index='seed', columns='algorithm', values='value', aggfunc='mean')
        cols = [c for c in ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'] if c in pivot.columns]
        if len(cols) < 2:
            continue
        arrays = [pivot[c].dropna().values for c in cols]
        if len(arrays) >= 2 and all(len(a) > 0 for a in arrays):
            try:
                stat, p = stats.friedmanchisquare(*arrays)
                lines.append(f'{dataset} | {metric}@{k}: Friedman chi2={stat:.3f}, p={p:.4f} over {len(pivot)} seeds')
            except Exception:
                pass
    return lines


def main():
    working_dir = make_working_dir()
    seeds = [11, 22, 33, 44, 55]
    datasets = ['MovieLens100K', 'Amazon Video Games', 'Last.FM']

    meta_path = os.path.join(working_dir, 'experiment_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'seeds': seeds, 'datasets': datasets}, f, indent=2)

    all_rows = []
    for seed in seeds:
        set_random_state(seed)
        for ds_name in datasets:
            ds = load_and_preprocess(ds_name)
            split = UserHoldout(validation_size=0.1, test_size=0.2).process(ds)
            plan = build_plan()
            evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))
            run_omnirec(datasets=split, plan=plan, evaluator=evaluator)
            results = evaluator.get_results()
            for dataset_id, res_df in results.items():
                res_df = normalize_results(res_df)
                res_df = res_df[['algorithm', 'name', 'k', 'value']].copy()
                res_df['seed'] = seed
                res_df['dataset'] = ds_name
                all_rows.extend(res_df.to_dict('records'))

    raw_df, summary_df = summarize(all_rows)
    raw_path = os.path.join(working_dir, 'all_seed_results.csv')
    summary_path = os.path.join(working_dir, 'summary_results.csv')
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print('\nPer-seed results:')
    print(raw_df.head(20).to_string(index=False))
    print('\nSummary (mean/std across seeds):')
    print(summary_df.to_string(index=False))
    print('\nShort statistical analysis:')
    for line in short_statistical_analysis(raw_df):
        print(line)
    print(f'\nMetadata saved to: {meta_path}')
    print(f'Raw results saved to: {raw_path}')
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    main()