import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import NDCG
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)
os.chdir(WORKING_DIR)

SEEDS = [11, 23, 37, 53, 71]
KS = [1, 5, 10]
TINY_VALIDATION = 1e-6


def sanitize_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name)


def snapshot_checkpoint_dirs() -> set:
    cp = Path('checkpoints')
    if not cp.exists():
        return set()
    return {str(p.resolve()) for p in cp.glob('*') if p.is_dir()}


def find_new_dataset_dir(before: set, after: set) -> Path:
    new_dirs = sorted(list(after - before))
    if len(new_dirs) == 1:
        return Path(new_dirs[0])
    if len(new_dirs) > 1:
        return Path(new_dirs[-1])
    if after:
        return Path(sorted(list(after))[-1])
    raise FileNotFoundError('No checkpoint dataset directory found.')


def load_predictions_from_dataset_dir(dataset_dir: Path) -> Dict[str, pd.DataFrame]:
    pred_map = {}
    for algo_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        pred_file = algo_dir / 'predictions.json'
        if pred_file.exists():
            with open(pred_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pred_map[algo_dir.name] = pd.DataFrame(data)
    return pred_map


def map_algo_dir_to_label(algo_dir_name: str) -> str:
    lower = algo_dir_name.lower()
    if 'implicitmfscorer' in lower:
        return 'ALS'
    if 'itemknnscorer' in lower:
        return 'ItemKNN'
    if 'popscorer' in lower:
        return 'Pop'
    return algo_dir_name


def get_public_raw_frame(saved_path: str) -> pd.DataFrame:
    loaded = RecSysDataSet.load(saved_path)
    data_obj = getattr(loaded, 'data', None)
    if data_obj is not None and hasattr(data_obj, 'df'):
        return data_obj.df.copy()
    if hasattr(loaded, '_data') and hasattr(loaded._data, 'df'):
        return loaded._data.df.copy()
    raise ValueError('Unable to access raw dataframe after loading dataset.')


def get_public_split_frames(saved_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loaded = RecSysDataSet.load(saved_path)
    data_obj = getattr(loaded, 'data', None)
    if data_obj is not None and hasattr(data_obj, 'train') and hasattr(data_obj, 'test'):
        train = data_obj.train.copy()
        valid = data_obj.val.copy() if hasattr(data_obj, 'val') else pd.DataFrame(columns=train.columns)
        test = data_obj.test.copy()
        return train, valid, test
    if hasattr(loaded, '_data') and hasattr(loaded._data, 'train') and hasattr(loaded._data, 'test'):
        train = loaded._data.train.copy()
        valid = loaded._data.val.copy() if hasattr(loaded._data, 'val') else pd.DataFrame(columns=train.columns)
        test = loaded._data.test.copy()
        return train, valid, test
    raise ValueError('Unable to access split data after loading dataset.')


def load_and_preprocess_raw(dataset_name: str) -> Tuple[RecSysDataSet, str]:
    if dataset_name == 'MovieLens100K':
        ds_enum = DataSet.MovieLens100K
        pipeline = Pipe(MakeImplicit(4), CorePruning(5))
    elif dataset_name == 'Amazon2014VideoGames':
        ds_enum = DataSet.Amazon2014VideoGames
        pipeline = Pipe(MakeImplicit(4), CorePruning(5))
    elif dataset_name == 'HetrecLastFM':
        ds_enum = DataSet.HetrecLastFM
        pipeline = Pipe(CorePruning(5))
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    ds = RecSysDataSet.use_dataloader(ds_enum)
    ds = pipeline.process(ds)
    save_path = os.path.join(WORKING_DIR, f'{sanitize_name(dataset_name)}_preprocessed')
    ds.save(save_path)
    return ds, save_path + '.rsds'


def build_split_dataset(base_dataset: RecSysDataSet, dataset_name: str, seed: int) -> Tuple[RecSysDataSet, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    set_random_state(seed)
    splitter = UserHoldout(validation_size=TINY_VALIDATION, test_size=0.2)
    split_ds = splitter.process(base_dataset)
    split_save = os.path.join(WORKING_DIR, f'{sanitize_name(dataset_name)}_seed_{seed}_split')
    split_ds.save(split_save)
    train_df, valid_df, test_df = get_public_split_frames(split_save + '.rsds')
    if len(valid_df) > 0:
        merged_train = pd.concat([train_df, valid_df], ignore_index=True)
    else:
        merged_train = train_df.copy()
    return split_ds, merged_train, valid_df, test_df, split_save + '.rsds'


def build_plan(plan_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name)
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def compute_precision_at_k(predictions: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    if predictions.empty or test_df.empty:
        return float('nan')
    truth = test_df.groupby('user')['item'].apply(set).to_dict()
    user_scores = []
    for user, group in predictions.groupby('user'):
        if user not in truth:
            continue
        topk = group.sort_values('rank').head(k)
        recs = topk['item'].tolist()
        hits = sum(1 for item in recs if item in truth[user])
        user_scores.append(hits / float(k))
    return float(np.mean(user_scores)) if user_scores else float('nan')


def compute_seed_variability(group: pd.DataFrame, metric_col: str) -> Dict[str, float]:
    vals = group[metric_col].dropna().astype(float).values
    if len(vals) == 0:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'cv': np.nan, 'range': np.nan}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    cv = float(std / mean) if mean != 0 else np.nan
    vrange = float(vmax - vmin)
    return {'mean': mean, 'std': std, 'min': vmin, 'max': vmax, 'cv': cv, 'range': vrange}


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    metric_cols = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']
    for group_key, g in results_df.groupby(['dataset', 'algorithm']):
        dataset, algorithm = cast(Tuple[object, object], group_key)
        row = {'dataset': dataset, 'algorithm': algorithm}
        for metric in metric_cols:
            stats = compute_seed_variability(g, metric)
            for stat_name, stat_value in stats.items():
                row[f'{metric}_{stat_name}'] = stat_value
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def short_statistical_analysis(summary_df: pd.DataFrame) -> pd.DataFrame:
    analysis_rows = []
    for _, row in summary_df.iterrows():
        record = {'dataset': row['dataset'], 'algorithm': row['algorithm']}
        for metric in ['ndcg@10', 'precision@10']:
            cv = row.get(f'{metric}_cv', np.nan)
            spread = row.get(f'{metric}_range', np.nan)
            if pd.isna(cv):
                sensitivity = 'unknown'
            elif cv < 0.02:
                sensitivity = 'very low'
            elif cv < 0.05:
                sensitivity = 'low'
            elif cv < 0.10:
                sensitivity = 'moderate'
            else:
                sensitivity = 'high'
            record[f'{metric}_cv'] = cv
            record[f'{metric}_range'] = spread
            record[f'{metric}_seed_sensitivity'] = sensitivity
        analysis_rows.append(record)
    return pd.DataFrame(analysis_rows)


def collect_ndcg_results(eval_results, algo_dir_name: str, algo_label: str) -> pd.DataFrame:
    if isinstance(eval_results, dict):
        frames = [df.copy() for df in eval_results.values()]
        eval_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        eval_df = pd.DataFrame(eval_results)
    if eval_df.empty:
        return eval_df
    sub = eval_df[eval_df['algorithm'].astype(str).str.contains(algo_dir_name.split('-')[0], case=False, na=False)]
    if sub.empty:
        sub = eval_df[eval_df['algorithm'].astype(str).str.contains(algo_label, case=False, na=False)]
    return sub


def run_one_dataset_seed(dataset_name: str, base_dataset: RecSysDataSet, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_ds, train_df, valid_df, test_df, split_save_path = build_split_dataset(base_dataset, dataset_name, seed)

    print(f'\n=== Dataset={dataset_name} Seed={seed} ===')
    print(f'Train interactions used for analysis: {len(train_df):,}')
    print(f'Validation interactions: {len(valid_df):,}')
    print(f'Test interactions: {len(test_df):,}')

    evaluator = Evaluator(NDCG(KS))
    plan = build_plan(f'{sanitize_name(dataset_name)}-seed-{seed}')

    before = snapshot_checkpoint_dirs()
    run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
    after = snapshot_checkpoint_dirs()
    dataset_dir = find_new_dataset_dir(before, after)
    predictions = load_predictions_from_dataset_dir(dataset_dir)
    eval_results = evaluator.get_results()

    metric_rows = []
    for algo_dir_name, pred_df in predictions.items():
        algo_label = map_algo_dir_to_label(algo_dir_name)
        sub = collect_ndcg_results(eval_results, algo_dir_name, algo_label)
        row = {'dataset': dataset_name, 'seed': seed, 'algorithm': algo_label}
        for k in KS:
            ndcg_val = np.nan
            if not sub.empty and {'name', 'k', 'value'}.issubset(sub.columns):
                ndcg_match = sub[(sub['name'] == 'NDCG') & (sub['k'] == k)]
                if not ndcg_match.empty:
                    ndcg_val = float(ndcg_match['value'].iloc[0])
            row[f'ndcg@{k}'] = ndcg_val
            row[f'precision@{k}'] = compute_precision_at_k(pred_df, test_df, k)
        metric_rows.append(row)
        print(
            f"{dataset_name} | seed={seed} | algo={algo_label} | "
            f"NDCG@1={row['ndcg@1']:.6f} NDCG@5={row['ndcg@5']:.6f} NDCG@10={row['ndcg@10']:.6f} | "
            f"P@1={row['precision@1']:.6f} P@5={row['precision@5']:.6f} P@10={row['precision@10']:.6f}"
        )

    return pd.DataFrame(metric_rows), train_df, test_df


if __name__ == '__main__':
    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']

    print('Loading and preprocessing datasets...')
    base_datasets = {}
    preprocessing_stats = []

    for dataset_name in dataset_names:
        ds, saved_path = load_and_preprocess_raw(dataset_name)
        base_datasets[dataset_name] = ds
        raw_df = get_public_raw_frame(saved_path)
        preprocessing_stats.append({
            'dataset': dataset_name,
            'num_interactions': len(raw_df),
            'num_users': raw_df['user'].nunique(),
            'num_items': raw_df['item'].nunique(),
        })
        print(f"Prepared {dataset_name}: interactions={len(raw_df):,}, users={raw_df['user'].nunique():,}, items={raw_df['item'].nunique():,}")

    prep_stats_df = pd.DataFrame(preprocessing_stats)
    prep_stats_df.to_csv(os.path.join(WORKING_DIR, 'preprocessing_stats.csv'), index=False)

    all_result_rows = []
    split_stats_rows = []

    for dataset_name in dataset_names:
        base_ds = base_datasets[dataset_name]
        for seed in SEEDS:
            run_df, train_df, test_df = run_one_dataset_seed(dataset_name, base_ds, seed)
            all_result_rows.append(run_df)
            split_stats_rows.append({
                'dataset': dataset_name,
                'seed': seed,
                'train_interactions': len(train_df),
                'test_interactions': len(test_df),
                'train_users': train_df['user'].nunique(),
                'test_users': test_df['user'].nunique(),
                'train_items': train_df['item'].nunique(),
                'test_items': test_df['item'].nunique(),
            })

    results_df = pd.concat(all_result_rows, ignore_index=True)
    split_stats_df = pd.DataFrame(split_stats_rows)
    summary_df = summarize_results(results_df)
    analysis_df = short_statistical_analysis(summary_df)

    results_path = os.path.join(WORKING_DIR, 'seed_sensitivity_per_seed_results.csv')
    split_path = os.path.join(WORKING_DIR, 'seed_split_stats.csv')
    summary_path = os.path.join(WORKING_DIR, 'seed_sensitivity_summary.csv')
    analysis_path = os.path.join(WORKING_DIR, 'seed_sensitivity_analysis.csv')

    results_df.to_csv(results_path, index=False)
    split_stats_df.to_csv(split_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    analysis_df.to_csv(analysis_path, index=False)

    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 200)

    print('\n=== Preprocessing Stats ===')
    print(prep_stats_df.to_string(index=False))

    print('\n=== Per-seed Results ===')
    print(results_df.sort_values(['dataset', 'algorithm', 'seed']).to_string(index=False))

    print('\n=== Variability Summary Across Seeds ===')
    print(summary_df.sort_values(['dataset', 'algorithm']).to_string(index=False))

    print('\n=== Short Statistical Analysis ===')
    print(analysis_df.sort_values(['dataset', 'algorithm']).to_string(index=False))

    print('\nSaved files:')
    print(results_path)
    print(split_path)
    print(summary_path)
    print(analysis_path)
    print(os.path.join(WORKING_DIR, 'preprocessing_stats.csv'))
