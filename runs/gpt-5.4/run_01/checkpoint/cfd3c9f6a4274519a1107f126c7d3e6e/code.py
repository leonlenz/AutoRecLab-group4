import os
import json
import math
import shutil
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.base import Preprocessor
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


class StrictImplicitGT3(Preprocessor):
    def __init__(self):
        super().__init__()

    def _process(self, dataset):
        data = getattr(dataset, '_data', None)
        if data is None or not hasattr(data, 'df'):
            raise ValueError('StrictImplicitGT3 expects a RawData dataset with a df attribute.')
        df = data.df.copy()
        if 'rating' not in df.columns:
            raise ValueError('StrictImplicitGT3 requires a rating column.')
        df = df.loc[df['rating'] > 3, ['user', 'item']].copy()
        data.df = df
        return dataset


class DropToImplicit(Preprocessor):
    def __init__(self):
        super().__init__()

    def _process(self, dataset):
        data = getattr(dataset, '_data', None)
        if data is None or not hasattr(data, 'df'):
            raise ValueError('DropToImplicit expects a RawData dataset with a df attribute.')
        df = data.df.copy()
        keep_cols = [c for c in ['user', 'item'] if c in df.columns]
        if len(keep_cols) != 2:
            raise ValueError('DropToImplicit requires user and item columns.')
        data.df = df[keep_cols].copy()
        return dataset


class MaterializeSplit(Preprocessor):
    def __init__(self, split_root, dataset_name, seed):
        super().__init__()
        self.split_root = Path(split_root)
        self.dataset_name = dataset_name
        self.seed = seed

    def _process(self, dataset):
        data = getattr(dataset, '_data', None)
        if data is None:
            raise ValueError('MaterializeSplit expects a split dataset.')
        train = getattr(data, 'train', None)
        test = getattr(data, 'test', None)
        if train is None or test is None:
            raise ValueError('MaterializeSplit requires public split fields train and test to exist after UserHoldout.')
        out_dir = self.split_root / self.dataset_name / f'seed_{self.seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / 'train.csv'
        test_path = out_dir / 'test.csv'
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        val = getattr(data, 'val', None)
        if val is not None:
            val.to_csv(out_dir / 'val.csv', index=False)
        return dataset


def build_dataset(dataset_enum, dataset_name, seed, split_root):
    print(f'Loading and preprocessing {dataset_name} for seed={seed}...')
    set_random_state(seed)
    ds = RecSysDataSet.use_dataloader(dataset_enum)

    steps = []
    if dataset_name in ('MovieLens100K', 'Amazon2014VideoGames'):
        steps.append(StrictImplicitGT3())
    else:
        steps.append(DropToImplicit())
    steps.append(CorePruning(5))
    steps.append(UserHoldout(validation_size=0.0, test_size=0.2))
    steps.append(MaterializeSplit(split_root=split_root, dataset_name=dataset_name, seed=seed))

    pipe = Pipe(*steps)
    ds = pipe.process(ds)
    return ds


def run_for_seed_dataset(ds, dataset_name, seed, checkpoint_root):
    print(f'Running OmniRec for {dataset_name}, seed={seed}...')
    plan = ExperimentPlan(plan_name=f'seed_sensitivity_{dataset_name}_seed_{seed}')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    run_omnirec(datasets=ds, plan=plan, evaluator=evaluator)


def find_prediction_files(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        return []
    return list(checkpoint_root.rglob('predictions.json'))


def infer_dataset_and_algorithm_from_path(pred_path):
    parts = list(Path(pred_path).parts)
    dataset_part = None
    algorithm_part = None
    for p in parts:
        if p.startswith('MovieLens100K-') or p.startswith('Amazon2014VideoGames-') or p.startswith('HetrecLastFM-'):
            dataset_part = p
        if p.startswith('LensKit.'):
            algorithm_part = p
    if dataset_part is None or algorithm_part is None:
        return None, None
    dataset_name = dataset_part.split('-')[0]
    algo_name = algorithm_part.split('-')[0].replace('LensKit.', '')
    return dataset_name, algo_name


def read_predictions(pred_path):
    pred_path = Path(pred_path)
    try:
        return pd.read_json(pred_path)
    except ValueError:
        with open(pred_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        return pd.DataFrame(obj)


def precision_at_k(pred_df, test_df, k):
    if pred_df.empty:
        return np.nan
    rel = test_df.groupby('user')['item'].apply(set).to_dict()
    work = pred_df[pred_df['rank'] <= k].copy()
    if work.empty:
        return np.nan
    work['hit'] = work.apply(lambda r: 1 if r['user'] in rel and r['item'] in rel[r['user']] else 0, axis=1)
    per_user = work.groupby('user')['hit'].sum() / float(k)
    return float(per_user.mean()) if len(per_user) else np.nan


def ndcg_at_k(pred_df, test_df, k):
    if pred_df.empty:
        return np.nan
    rel = test_df.groupby('user')['item'].apply(set).to_dict()
    ideal_sizes = test_df.groupby('user').size().to_dict()
    work = pred_df[pred_df['rank'] <= k].copy()
    if work.empty:
        return np.nan
    work['gain'] = work.apply(lambda r: 1.0 / math.log2(r['rank'] + 1) if r['user'] in rel and r['item'] in rel[r['user']] else 0.0, axis=1)
    dcg = work.groupby('user')['gain'].sum()
    ndcgs = []
    for user, user_dcg in dcg.items():
        m = min(k, int(ideal_sizes.get(user, 0)))
        if m <= 0:
            continue
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, m + 1))
        ndcgs.append(user_dcg / idcg if idcg > 0 else np.nan)
    return float(np.nanmean(ndcgs)) if ndcgs else np.nan


def summarize_results(df):
    metric_cols = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Precision@1', 'Precision@5', 'Precision@10']
    rows = []
    grouped = df.groupby(['dataset', 'algorithm'])
    for (dataset, algorithm), g in grouped:
        row = {'dataset': dataset, 'algorithm': algorithm, 'n_seeds': int(g['seed'].nunique())}
        for m in metric_cols:
            vals = g[m].astype(float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f'{m}_mean'] = mean
            row[f'{m}_std'] = std
            row[f'{m}_cv'] = float(std / mean) if mean not in (0.0, np.nan) and not np.isnan(mean) else np.nan
            row[f'{m}_min'] = float(vals.min())
            row[f'{m}_max'] = float(vals.max())
            row[f'{m}_range'] = float(vals.max() - vals.min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['dataset', 'algorithm']).reset_index(drop=True)


def rank_seed_sensitivity(summary_df):
    metric_bases = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Precision@1', 'Precision@5', 'Precision@10']
    rows = []
    for dataset, g in summary_df.groupby('dataset'):
        for metric in metric_bases:
            sub = g[['algorithm', f'{metric}_std', f'{metric}_cv', f'{metric}_range']].copy()
            sub = sub.sort_values([f'{metric}_cv', f'{metric}_std', f'{metric}_range'], ascending=False)
            sub['rank'] = range(1, len(sub) + 1)
            sub['dataset'] = dataset
            sub['metric'] = metric
            rows.append(sub[['dataset', 'metric', 'rank', 'algorithm', f'{metric}_std', f'{metric}_cv', f'{metric}_range']])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def concise_text_summary(summary_df):
    metrics_focus = ['NDCG@10_cv', 'Precision@10_cv']
    lines = []
    for dataset, g in summary_df.groupby('dataset'):
        lines.append(f'Dataset: {dataset}')
        for mf in metrics_focus:
            tmp = g[['algorithm', mf]].sort_values(mf, ascending=False)
            ordered = ', '.join([f"{r.algorithm} ({r[mf]:.4f})" for _, r in tmp.iterrows()])
            lines.append(f'  Seed sensitivity by {mf}: {ordered}')
    return '\n'.join(lines)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    split_root = Path(working_dir) / 'materialized_splits'
    outputs_root = Path(working_dir) / 'outputs'
    checkpoint_root = Path(working_dir) / 'checkpoints'
    outputs_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    datasets = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': DataSet.Amazon2014VideoGames,
        'HetrecLastFM': DataSet.HetrecLastFM,
    }
    seeds = [7, 19, 42, 77, 123]

    split_index_rows = []
    for dataset_name, dataset_enum in datasets.items():
        for seed in seeds:
            ds = build_dataset(dataset_enum, dataset_name, seed, split_root)
            split_dir = split_root / dataset_name / f'seed_{seed}'
            train_df = pd.read_csv(split_dir / 'train.csv')
            test_df = pd.read_csv(split_dir / 'test.csv')
            split_index_rows.append({
                'dataset': dataset_name,
                'seed': seed,
                'train_interactions': len(train_df),
                'test_interactions': len(test_df),
                'train_users': train_df['user'].nunique(),
                'test_users': test_df['user'].nunique(),
                'train_items': train_df['item'].nunique(),
                'test_items': test_df['item'].nunique(),
            })
            run_for_seed_dataset(ds, dataset_name, seed, checkpoint_root)

    split_index = pd.DataFrame(split_index_rows)
    split_index.to_csv(outputs_root / 'split_index.csv', index=False)

    prediction_files = find_prediction_files(checkpoint_root)
    if not prediction_files:
        raise RuntimeError(f'No predictions.json files found under {checkpoint_root}')

    result_rows = []
    for pred_path in prediction_files:
        dataset_name, algorithm = infer_dataset_and_algorithm_from_path(pred_path)
        if dataset_name is None:
            continue
        matched_seed = None
        pred_path_str = str(pred_path)
        for seed in seeds:
            if f'seed_{seed}' in pred_path_str:
                matched_seed = seed
                break
        if matched_seed is None:
            continue
        pred_df = read_predictions(pred_path)
        if pred_df.empty:
            continue
        pred_df = pred_df[['user', 'item', 'score', 'rank']].copy()
        split_dir = split_root / dataset_name / f'seed_{matched_seed}'
        test_df = pd.read_csv(split_dir / 'test.csv')
        metrics = {
            'NDCG@1': ndcg_at_k(pred_df, test_df, 1),
            'NDCG@5': ndcg_at_k(pred_df, test_df, 5),
            'NDCG@10': ndcg_at_k(pred_df, test_df, 10),
            'Precision@1': precision_at_k(pred_df, test_df, 1),
            'Precision@5': precision_at_k(pred_df, test_df, 5),
            'Precision@10': precision_at_k(pred_df, test_df, 10),
        }
        row = {
            'dataset': dataset_name,
            'algorithm': algorithm,
            'seed': matched_seed,
            'prediction_file': str(pred_path),
        }
        row.update(metrics)
        print(f"Collected {dataset_name} | {algorithm} | seed={matched_seed} | "
              f"NDCG@10={metrics['NDCG@10']:.4f} | Precision@10={metrics['Precision@10']:.4f}")
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError('No result rows were collected from checkpoint predictions.')

    algo_map = {
        'ImplicitMFScorer': 'ALS',
        'ItemKNNScorer': 'ItemKNN',
        'PopScorer': 'Pop',
    }
    results['algorithm'] = results['algorithm'].map(lambda x: algo_map.get(x, x))
    results = results.sort_values(['dataset', 'algorithm', 'seed']).reset_index(drop=True)
    results.to_csv(outputs_root / 'seed_level_results.csv', index=False)

    summary = summarize_results(results)
    summary.to_csv(outputs_root / 'summary_by_dataset_algorithm.csv', index=False)

    sensitivity = rank_seed_sensitivity(summary)
    sensitivity.to_csv(outputs_root / 'seed_sensitivity_rankings.csv', index=False)

    print('\n=== Seed-level results ===')
    print(results.to_string(index=False))
    print('\n=== Aggregated summary ===')
    print(summary.to_string(index=False))
    print('\n=== Concise comparison ===')
    print(concise_text_summary(summary))
    print(f"\nArtifacts written to: {outputs_root}")


if __name__ == '__main__':
    main()
