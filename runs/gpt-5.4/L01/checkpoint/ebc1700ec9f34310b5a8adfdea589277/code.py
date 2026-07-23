import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

SEEDS = [7, 13, 23, 37, 53]
K_VALUES = [1, 5, 10]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_name(name):
    return str(name).replace(' ', '_').replace('/', '_')


def load_and_preprocess_dataset(dataset_enum, dataset_label, implicit_threshold=None):
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    ds = pipe.process(ds)
    print(f'Preprocessed {dataset_label}: {ds}')
    return ds


def make_seed_split(preprocessed_dataset, seed):
    set_random_state(seed)
    splitter = UserHoldout(validation_size=0.2, test_size=0.2)
    return splitter.process(preprocessed_dataset)


def build_plan():
    plan = ExperimentPlan('seed_sensitivity_lenskit_baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def evaluator_with_ndcg():
    return Evaluator(NDCG(K_VALUES))


def flatten_results_dict(results_dict, dataset_alias, seed):
    frames = []
    for dataset_id, df in results_dict.items():
        tmp = df.copy()
        tmp['dataset_id'] = dataset_id
        tmp['dataset'] = dataset_alias
        tmp['seed'] = seed
        frames.append(tmp)
    if not frames:
        empty_columns = pd.Index([
            'algorithm', 'fold', 'name', 'k', 'value', 'dataset_id', 'dataset', 'seed'
        ])
        return pd.DataFrame(columns=empty_columns)
    return pd.concat(frames, ignore_index=True)


def load_predictions_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            return pd.DataFrame(data['data'])
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def find_prediction_files(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return []
    return list(root.rglob('predictions.json'))


def normalize_algorithm_name(name):
    text = str(name)
    if text.startswith('LensKit.ImplicitMFScorer'):
        return 'ALS'
    if text.startswith('LensKit.ItemKNNScorer'):
        return 'ItemKNN'
    if text.startswith('LensKit.PopScorer'):
        return 'Pop'
    return text.split('-', 1)[0]


def extract_test_df_from_split(split_dataset):
    test_df = split_dataset._data.get('test').copy()
    if 'rating' not in test_df.columns:
        test_df['rating'] = 1
    return test_df[['user', 'item', 'rating']].drop_duplicates()


def standardize_prediction_df(pred_df):
    if pred_df.empty:
        return pred_df
    cols = {c.lower(): c for c in pred_df.columns}
    out = pred_df.copy()

    if 'user' not in out.columns and 'user_id' in cols:
        out = out.rename(columns={cols['user_id']: 'user'})
    if 'item' not in out.columns and 'item_id' in cols:
        out = out.rename(columns={cols['item_id']: 'item'})
    if 'score' not in out.columns:
        if 'prediction' in cols:
            out = out.rename(columns={cols['prediction']: 'score'})
        elif 'value' in cols:
            out = out.rename(columns={cols['value']: 'score'})
    if 'rank' not in out.columns and 'rnk' in cols:
        out = out.rename(columns={cols['rnk']: 'rank'})

    needed = {'user', 'item'}
    if not needed.issubset(set(out.columns)):
        return pd.DataFrame()
    return out


def precision_at_k(pred_df, test_df, k):
    if pred_df.empty:
        return float('nan')

    preds = standardize_prediction_df(pred_df)
    if preds.empty:
        return float('nan')

    rel = test_df[['user', 'item']].drop_duplicates().copy()
    rel['relevant'] = 1

    if 'rank' in preds.columns:
        sort_cols = ['user', 'rank']
        ascending = [True, True]
        if 'score' in preds.columns:
            sort_cols.append('score')
            ascending.append(False)
        preds = preds.sort_values(sort_cols, ascending=ascending)
    elif 'score' in preds.columns:
        preds = preds.sort_values(['user', 'score'], ascending=[True, False])
    else:
        preds = preds.sort_values(['user', 'item'])

    preds = preds.groupby('user', as_index=False, group_keys=False).head(k).copy()
    preds = preds.merge(rel, on=['user', 'item'], how='left')
    preds['relevant'] = preds['relevant'].fillna(0)

    user_prec = preds.groupby('user')['relevant'].sum() / float(k)
    if len(user_prec) == 0:
        return float('nan')
    return float(user_prec.mean())


def collect_precision_metrics(split_dataset, checkpoint_root, completed_algorithms):
    test_df = extract_test_df_from_split(split_dataset)
    pred_files = find_prediction_files(checkpoint_root)
    metrics_by_algo = {algo: {f'Precision@{k}': float('nan') for k in K_VALUES} for algo in completed_algorithms}

    for pred_file in pred_files:
        pred_df = load_predictions_json(pred_file)
        pred_df = standardize_prediction_df(pred_df)
        if pred_df.empty:
            continue

        path_text = str(pred_file)
        algo_match = None
        for algo in completed_algorithms:
            if algo == 'ALS' and 'ImplicitMFScorer' in path_text:
                algo_match = algo
                break
            if algo == 'ItemKNN' and 'ItemKNNScorer' in path_text:
                algo_match = algo
                break
            if algo == 'Pop' and 'PopScorer' in path_text:
                algo_match = algo
                break
        if algo_match is None:
            continue

        for k in K_VALUES:
            metrics_by_algo[algo_match][f'Precision@{k}'] = precision_at_k(pred_df, test_df, k)

    return metrics_by_algo


def aggregate_seed_results(all_metric_rows):
    grouped = all_metric_rows.groupby(['dataset', 'algorithm', 'metric'], as_index=False).agg(
        mean=('value', 'mean'),
        std=('value', 'std'),
        min=('value', 'min'),
        max=('value', 'max'),
        n=('value', 'count')
    )
    grouped['cv'] = grouped['std'] / grouped['mean'].replace(0, np.nan)
    grouped['range'] = grouped['max'] - grouped['min']
    return grouped.sort_values(['dataset', 'algorithm', 'metric']).reset_index(drop=True)


def summarize_seed_sensitivity(summary_df):
    lines = []
    for dataset in sorted(summary_df['dataset'].unique()):
        sub = summary_df[(summary_df['dataset'] == dataset) & (summary_df['metric'].isin(['NDCG@10', 'Precision@10']))]
        if sub.empty:
            continue
        lines.append(f'Dataset: {dataset}')
        for metric in ['NDCG@10', 'Precision@10']:
            msub = sub[sub['metric'] == metric].sort_values('std', ascending=False)
            if msub.empty:
                continue
            most = msub.iloc[0]
            least = msub.iloc[-1]
            lines.append(
                f"  {metric}: most seed-sensitive={most['algorithm']} (std={most['std']:.6f}, range={most['range']:.6f}); "
                f"least seed-sensitive={least['algorithm']} (std={least['std']:.6f}, range={least['range']:.6f})"
            )
    return '\n'.join(lines)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    ensure_dir(working_dir)
    results_dir = ensure_dir(os.path.join(working_dir, 'results'))
    splits_dir = ensure_dir(os.path.join(working_dir, 'splits'))
    os.chdir(working_dir)

    datasets = [
        ('MovieLens100K', DataSet.MovieLens100K, 3),
        ('Amazon2014VideoGames', DataSet.Amazon2014VideoGames, 3),
        ('HetrecLastFM', DataSet.HetrecLastFM, None),
    ]

    plan = build_plan()
    all_run_rows = []

    for dataset_alias, dataset_enum, implicit_threshold in datasets:
        print(f'\n=== Loading and preprocessing {dataset_alias} ===')
        base_dataset = load_and_preprocess_dataset(dataset_enum, dataset_alias, implicit_threshold)

        for seed in SEEDS:
            print(f'\n--- Running dataset={dataset_alias}, seed={seed} ---')
            split_dataset = make_seed_split(base_dataset, seed)
            split_path = os.path.join(splits_dir, f'{sanitize_name(dataset_alias)}_seed{seed}.rsds')
            split_dataset.save(split_path)

            evaluator = evaluator_with_ndcg()
            checkpoint_root = os.path.join(working_dir, 'checkpoints', sanitize_name(dataset_alias), f'seed_{seed}')
            ensure_dir(checkpoint_root)

            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)

            ndcg_df = flatten_results_dict(evaluator.get_results(), dataset_alias, seed)
            print(ndcg_df)

            metric_rows = []
            completed_algorithms = []
            for _, row in ndcg_df.iterrows():
                algo_name = normalize_algorithm_name(row['algorithm'])
                if algo_name not in completed_algorithms:
                    completed_algorithms.append(algo_name)
                metric_rows.append(
                    {
                        'dataset': dataset_alias,
                        'seed': seed,
                        'algorithm': algo_name,
                        'metric': f"{row['name']}@{int(row['k'])}" if pd.notna(row['k']) else str(row['name']),
                        'value': float(row['value']),
                    }
                )

            precision_metrics = collect_precision_metrics(split_dataset, working_dir, completed_algorithms)
            for algo_name, algo_metrics in precision_metrics.items():
                for metric_name, metric_value in algo_metrics.items():
                    metric_rows.append(
                        {
                            'dataset': dataset_alias,
                            'seed': seed,
                            'algorithm': algo_name,
                            'metric': metric_name,
                            'value': float(metric_value),
                        }
                    )

            run_df = pd.DataFrame(metric_rows)
            if not run_df.empty:
                print(run_df.sort_values(['algorithm', 'metric']).to_string(index=False))
                all_run_rows.append(run_df)

    if not all_run_rows:
        raise RuntimeError('No experiment results were collected.')

    all_runs_df = pd.concat(all_run_rows, ignore_index=True)
    all_runs_csv = os.path.join(results_dir, 'seed_sensitivity_all_runs.csv')
    all_runs_df.to_csv(all_runs_csv, index=False)

    summary_df = aggregate_seed_results(all_runs_df)
    summary_csv = os.path.join(results_dir, 'seed_sensitivity_summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    print('\n=== Mean and variability across seeds ===')
    print(summary_df.to_string(index=False))

    analysis = summarize_seed_sensitivity(summary_df)
    print('\n=== Concise seed-sensitivity analysis ===')
    print(analysis)

    print('\nNote: OmniRec UserHoldout requires both validation_size and test_size to be positive. This bugfix uses a valid non-zero validation split to avoid the documented API crash.')
    print(f'\nSaved per-run results to: {all_runs_csv}')
    print(f'Saved summary results to: {summary_csv}')


if __name__ == '__main__':
    main()
