import itertools
import json
import os
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import SplitData
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.filter import RatingFilter
from omnirec.preprocess.pipe import Pipe
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def export_raw_dataframe_via_rsds(dataset: RecSysDataSet, export_path: str) -> pd.DataFrame:
    dataset.save(export_path)
    rsds_path = export_path if export_path.endswith('.rsds') else export_path + '.rsds'
    with zipfile.ZipFile(rsds_path, 'r') as zf:
        if 'data.csv' not in zf.namelist():
            raise RuntimeError('Expected RawData with data.csv in RSDS export, but data.csv was not found.')
        df = pd.read_csv(zf.open('data.csv'))
    os.remove(rsds_path)
    return df


def user_random_holdout_80_20(df: pd.DataFrame, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for _, user_df in df.groupby('user', sort=False):
        n = len(user_df)
        if n < 2:
            continue

        n_test = max(1, int(round(0.2 * n)))
        if n_test >= n:
            n_test = n - 1

        idx = np.arange(n)
        test_local_idx = rng.choice(idx, size=n_test, replace=False)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_local_idx] = True

        test_parts.append(user_df.iloc[test_mask])
        train_parts.append(user_df.iloc[~test_mask])

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    val_df = df.iloc[0:0].copy().reset_index(drop=True)

    return SplitData(train=train_df, val=val_df, test=test_df)


def preprocess_dataset(dataset_name: str, dataset_enum: DataSet) -> RecSysDataSet:
    dataset = RecSysDataSet.use_dataloader(dataset_enum)

    if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        pipe = Pipe(
            RatingFilter(lower=4),
            MakeImplicit(4),
            CorePruning(5),
        )
    elif dataset_name == 'HetrecLastFM':
        pipe = Pipe(
            MakeImplicit(1),
            CorePruning(5),
        )
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')

    return pipe.process(dataset)


def algorithm_label_from_result_name(result_algo_name: str) -> str:
    if 'ImplicitMFScorer' in result_algo_name:
        return 'ALS'
    if 'ItemKNNScorer' in result_algo_name:
        return 'ItemKNN'
    if 'PopScorer' in result_algo_name:
        return 'Pop'
    return 'Unknown'


def run_single_condition(split_dataset: RecSysDataSet, dataset_name: str, seed: int, run_dir: str) -> pd.DataFrame:
    plan = ExperimentPlan(plan_name=f'seed_sensitivity_{dataset_name}_seed_{seed}')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    os.makedirs(run_dir, exist_ok=True)
    original_cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
    finally:
        os.chdir(original_cwd)

    collected = []
    all_results = evaluator.get_results()
    for _, df in all_results.items():
        filtered = df[
            (df['name'].isin(['NDCG', 'Precision']))
            & (df['k'].isin([1, 5, 10]))
        ].copy()

        if filtered.empty:
            continue

        pivot = filtered.pivot_table(
            index='algorithm',
            columns=['name', 'k'],
            values='value',
            aggfunc='mean'
        )

        for algo_name, row in pivot.iterrows():
            algo_label = algorithm_label_from_result_name(algo_name)
            if algo_label == 'Unknown':
                continue

            out = {
                'dataset': dataset_name,
                'seed': seed,
                'algorithm': algo_label,
                'ndcg@1': float(row.get(('NDCG', 1), np.nan)),
                'ndcg@5': float(row.get(('NDCG', 5), np.nan)),
                'ndcg@10': float(row.get(('NDCG', 10), np.nan)),
                'precision@1': float(row.get(('Precision', 1), np.nan)),
                'precision@5': float(row.get(('Precision', 5), np.nan)),
                'precision@10': float(row.get(('Precision', 10), np.nan)),
            }
            collected.append(out)

    result_df = pd.DataFrame(collected)
    if not result_df.empty:
        print('\nPer-run result snapshot:')
        print(result_df.to_string(index=False))
    return result_df


def summarize_seed_variation(per_run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']
    agg_spec = {}
    for m in metric_cols:
        agg_spec[m + '_mean'] = (m, 'mean')
        agg_spec[m + '_std'] = (m, 'std')

    summary = (
        per_run_df
        .groupby(['dataset', 'algorithm'], as_index=False)
        .agg(**agg_spec)
        .sort_values(['dataset', 'algorithm'])
        .reset_index(drop=True)
    )
    return summary


def compute_seed_sensitivity_analysis(per_run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']
    rows = []

    for dataset_name, ds_df in per_run_df.groupby('dataset'):
        for metric in metric_cols:
            # Mean seed-driven std across algorithms
            seed_std_by_algo = ds_df.groupby('algorithm')[metric].std(ddof=1)
            mean_seed_std = float(seed_std_by_algo.mean()) if not seed_std_by_algo.empty else np.nan

            # Average between-algorithm gap using algorithm means
            algo_means = ds_df.groupby('algorithm')[metric].mean().to_dict()
            pairs = list(itertools.combinations(sorted(algo_means.keys()), 2))
            pair_diffs = [abs(algo_means[a] - algo_means[b]) for a, b in pairs]
            mean_algo_gap = float(np.mean(pair_diffs)) if pair_diffs else np.nan

            if np.isnan(mean_seed_std) or np.isnan(mean_algo_gap) or mean_algo_gap == 0:
                sensitivity_ratio = np.nan
            else:
                sensitivity_ratio = float(mean_seed_std / mean_algo_gap)

            rows.append({
                'dataset': dataset_name,
                'metric': metric,
                'mean_seed_std_across_algorithms': mean_seed_std,
                'mean_between_algorithm_gap': mean_algo_gap,
                'seed_std_over_algo_gap_ratio': sensitivity_ratio,
            })

    out_df = pd.DataFrame(rows).sort_values(['dataset', 'metric']).reset_index(drop=True)
    return out_df


def print_short_statistical_analysis(sensitivity_df: pd.DataFrame) -> None:
    print('\n=== Short Statistical Analysis: Split-Seed Sensitivity ===')
    for dataset_name, ds_df in sensitivity_df.groupby('dataset'):
        ratio_mean = ds_df['seed_std_over_algo_gap_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        if ratio_mean.empty:
            print(f'- {dataset_name}: insufficient data for stable sensitivity ratio.')
            continue

        avg_ratio = ratio_mean.mean()
        if avg_ratio < 0.33:
            interpretation = 'low seed sensitivity relative to algorithm differences'
        elif avg_ratio < 0.67:
            interpretation = 'moderate seed sensitivity'
        else:
            interpretation = 'high seed sensitivity (split randomness is comparable to model gaps)'

        print(f'- {dataset_name}: average seed-std/algo-gap ratio = {avg_ratio:.3f} -> {interpretation}.')


def main() -> None:
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    output_dir = os.path.join(working_dir, 'seed_sensitivity_outputs')
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': DataSet.Amazon2014VideoGames,
        'HetrecLastFM': DataSet.HetrecLastFM,
    }

    seeds = [2027, 3109, 4513, 7127, 9901]
    with open(os.path.join(output_dir, 'split_seeds.json'), 'w', encoding='utf-8') as f:
        json.dump({'split_seeds': seeds}, f, indent=2)

    print('Using split seeds:', seeds)

    all_runs: List[pd.DataFrame] = []

    for dataset_name, dataset_enum in datasets.items():
        print(f'\n===== Preparing dataset: {dataset_name} =====')
        base_dataset = preprocess_dataset(dataset_name, dataset_enum)

        rsds_temp_path = os.path.join(output_dir, f'{dataset_name}_preprocessed_raw')
        base_df = export_raw_dataframe_via_rsds(base_dataset, rsds_temp_path)

        print(f'Preprocessed interactions for {dataset_name}: {len(base_df)}')

        for seed in seeds:
            print(f'\n--- Running dataset={dataset_name}, seed={seed} ---')
            set_random_state(seed)

            split_data = user_random_holdout_80_20(base_df, seed)
            split_dataset = base_dataset.replace_data(split_data)

            run_dir = os.path.join(working_dir, 'runs', dataset_name, f'seed_{seed}')
            run_result = run_single_condition(split_dataset, dataset_name, seed, run_dir)

            if run_result.empty:
                print(f'Warning: no results captured for dataset={dataset_name}, seed={seed}')
            else:
                all_runs.append(run_result)

    if not all_runs:
        raise RuntimeError('No experiment results were produced.')

    per_run_df = pd.concat(all_runs, axis=0).reset_index(drop=True)
    per_run_df = per_run_df.sort_values(['dataset', 'algorithm', 'seed']).reset_index(drop=True)

    summary_df = summarize_seed_variation(per_run_df)
    sensitivity_df = compute_seed_sensitivity_analysis(per_run_df)

    per_run_csv = os.path.join(output_dir, 'per_run_results_dataset_algorithm_seed.csv')
    summary_csv = os.path.join(output_dir, 'seed_variation_summary_mean_std.csv')
    sensitivity_csv = os.path.join(output_dir, 'seed_sensitivity_vs_algorithm_gap.csv')

    per_run_df.to_csv(per_run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    sensitivity_df.to_csv(sensitivity_csv, index=False)

    print('\n===== Final Per-Run Results =====')
    print(per_run_df.to_string(index=False))

    print('\n===== Mean/Std Across Seeds (per dataset x algorithm) =====')
    print(summary_df.to_string(index=False))

    print('\n===== Seed Sensitivity vs Algorithm Gap =====')
    print(sensitivity_df.to_string(index=False))

    print_short_statistical_analysis(sensitivity_df)

    print('\nSaved outputs:')
    print('-', per_run_csv)
    print('-', summary_csv)
    print('-', sensitivity_csv)
    print('-', os.path.join(output_dir, 'split_seeds.json'))


if __name__ == '__main__':
    main()
