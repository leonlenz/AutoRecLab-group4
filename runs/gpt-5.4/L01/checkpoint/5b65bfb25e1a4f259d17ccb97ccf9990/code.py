import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import SplitData, SplitDataDict
from omnirec.preprocess.base import Preprocessor
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


class StrictImplicitGT3(Preprocessor):
    def __init__(self):
        super().__init__()

    def _process(self, dataset):
        df = dataset._data.df.copy()
        if 'rating' not in df.columns:
            raise ValueError('StrictImplicitGT3 requires a rating column.')
        df = df.loc[df['rating'] > 3, ['user', 'item']].copy()
        dataset._data.df = df.reset_index(drop=True)
        return dataset


class DropToImplicit(Preprocessor):
    def __init__(self):
        super().__init__()

    def _process(self, dataset):
        df = dataset._data.df.copy()
        cols = [c for c in ['user', 'item'] if c in df.columns]
        if cols != ['user', 'item']:
            raise ValueError('DropToImplicit requires user and item columns.')
        dataset._data.df = df[['user', 'item']].copy().reset_index(drop=True)
        return dataset


class UserTrainTestHoldout(Preprocessor):
    def __init__(self, test_size=0.2):
        super().__init__()
        self.test_size = float(test_size)

    def _process(self, dataset):
        if not (0.0 < self.test_size < 1.0):
            raise ValueError('test_size must be in (0, 1).')

        df = dataset._data.df.copy().reset_index(drop=True)
        if 'user' not in df.columns or 'item' not in df.columns:
            raise ValueError('UserTrainTestHoldout requires user and item columns.')

        rng = np.random.default_rng(get_random_state())
        train_parts = []
        test_parts = []

        for user, udf in df.groupby('user', sort=False):
            udf = udf.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
            n = len(udf)
            n_test = max(1, int(np.floor(n * self.test_size)))
            if n_test >= n:
                n_test = n - 1
            if n_test <= 0:
                raise ValueError(f'User {user} has too few interactions for holdout after preprocessing.')

            test_parts.append(udf.iloc[:n_test].copy())
            train_parts.append(udf.iloc[n_test:].copy())

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        val_df = train_df.iloc[0:0].copy()

        split_dict: SplitDataDict = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
        }
        dataset._data = SplitData(train=split_dict['train'], val=split_dict['val'], test=split_dict['test'])
        return dataset


class MaterializeSplit(Preprocessor):
    def __init__(self, split_root, dataset_name, seed):
        super().__init__()
        self.split_root = Path(split_root)
        self.dataset_name = dataset_name
        self.seed = seed

    def _process(self, dataset):
        out_dir = self.split_root / self.dataset_name / f'seed_{self.seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset._data.train.to_csv(out_dir / 'train.csv', index=False)
        dataset._data.test.to_csv(out_dir / 'test.csv', index=False)
        dataset._data.val.to_csv(out_dir / 'val.csv', index=False)
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
    steps.append(UserTrainTestHoldout(test_size=0.2))
    steps.append(MaterializeSplit(split_root=split_root, dataset_name=dataset_name, seed=seed))

    pipe = Pipe(*steps)
    ds = pipe.process(ds)
    return ds


def run_for_seed_dataset(ds, dataset_name, seed):
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


def find_prediction_files(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []
    return list(root_dir.rglob('predictions.json'))


def read_predictions(pred_path):
    pred_path = Path(pred_path)
    try:
        return pd.read_json(pred_path)
    except Exception:
        with open(pred_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        return pd.DataFrame(obj)


def infer_dataset_algorithm_seed(pred_path, dataset_names, seeds):
    pred_path = Path(pred_path)
    path_str = str(pred_path)
    dataset = None
    for d in dataset_names:
        if d in path_str:
            dataset = d
            break

    algorithm = None
    for candidate in ['ImplicitMFScorer', 'ItemKNNScorer', 'PopScorer']:
        if candidate in path_str:
            algorithm = candidate
            break

    seed = None
    for s in seeds:
        if f'seed_{s}' in path_str:
            seed = s
            break

    return dataset, algorithm, seed


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
            row[f'{m}_cv'] = float(std / mean) if pd.notna(mean) and mean != 0.0 else np.nan
            row[f'{m}_min'] = float(vals.min())
            row[f'{m}_max'] = float(vals.max())
            row[f'{m}_range'] = float(vals.max() - vals.min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['dataset', 'algorithm']).reset_index(drop=True)


def paired_seed_differences(results_df):
    metrics = ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Precision@1', 'Precision@5', 'Precision@10']
    rows = []
    for dataset, dfg in results_df.groupby('dataset'):
        pivot_source = dfg[['dataset', 'algorithm', 'seed'] + metrics].copy()
        for metric in metrics:
            sub = pivot_source[['algorithm', 'seed', metric]].pivot(index='seed', columns='algorithm', values=metric)
            algos = list(sub.columns)
            for i in range(len(algos)):
                for j in range(i + 1, len(algos)):
                    a, b = algos[i], algos[j]
                    diff = (sub[a] - sub[b]).dropna()
                    if len(diff) == 0:
                        continue
                    rows.append({
                        'dataset': dataset,
                        'metric': metric,
                        'algo_a': a,
                        'algo_b': b,
                        'n_pairs': int(len(diff)),
                        'mean_diff': float(diff.mean()),
                        'std_diff': float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                        'min_diff': float(diff.min()),
                        'max_diff': float(diff.max()),
                    })
    return pd.DataFrame(rows)


def concise_text_summary(summary_df):
    lines = []
    for dataset, g in summary_df.groupby('dataset'):
        lines.append(f'Dataset: {dataset}')
        tmp1 = g[['algorithm', 'NDCG@10_mean', 'NDCG@10_std', 'NDCG@10_cv']].sort_values('NDCG@10_mean', ascending=False)
        lines.append('  NDCG@10 mean±std: ' + ', '.join(
            f"{r.algorithm} ({r['NDCG@10_mean']:.4f}±{r['NDCG@10_std']:.4f}, cv={r['NDCG@10_cv']:.4f})" for _, r in tmp1.iterrows()
        ))
        tmp2 = g[['algorithm', 'Precision@10_mean', 'Precision@10_std', 'Precision@10_cv']].sort_values('Precision@10_mean', ascending=False)
        lines.append('  Precision@10 mean±std: ' + ', '.join(
            f"{r.algorithm} ({r['Precision@10_mean']:.4f}±{r['Precision@10_std']:.4f}, cv={r['Precision@10_cv']:.4f})" for _, r in tmp2.iterrows()
        ))
        sens = g[['algorithm', 'NDCG@10_cv', 'Precision@10_cv']].copy()
        sens['avg_cv'] = sens[['NDCG@10_cv', 'Precision@10_cv']].mean(axis=1)
        sens = sens.sort_values('avg_cv', ascending=False)
        lines.append('  Most seed-sensitive overall: ' + ', '.join(f"{r.algorithm} ({r.avg_cv:.4f})" for _, r in sens.iterrows()))
    return '\n'.join(lines)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    outputs_root = Path(working_dir) / 'outputs'
    split_root = Path(working_dir) / 'materialized_splits'
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
            run_for_seed_dataset(ds, dataset_name, seed)

    split_index = pd.DataFrame(split_index_rows)
    split_index.to_csv(outputs_root / 'split_index.csv', index=False)

    prediction_files = find_prediction_files(Path(working_dir))
    if not prediction_files:
        raise RuntimeError(f'No predictions.json files found under {working_dir}')

    result_rows = []
    for pred_path in prediction_files:
        dataset_name, algorithm, seed = infer_dataset_algorithm_seed(pred_path, list(datasets.keys()), seeds)
        if dataset_name is None or algorithm is None or seed is None:
            continue
        pred_df = read_predictions(pred_path)
        if pred_df.empty:
            continue
        required_cols = {'user', 'item', 'rank'}
        if not required_cols.issubset(set(pred_df.columns)):
            continue
        cols = [c for c in ['user', 'item', 'score', 'rank'] if c in pred_df.columns]
        pred_df = pred_df[cols].copy()
        split_dir = split_root / dataset_name / f'seed_{seed}'
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
            'seed': seed,
            'prediction_file': str(pred_path),
        }
        row.update(metrics)
        print(
            f"Collected {dataset_name} | {algorithm} | seed={seed} | "
            f"NDCG@10={metrics['NDCG@10']:.4f} | Precision@10={metrics['Precision@10']:.4f}"
        )
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError('No result rows were collected from prediction files.')

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

    paired = paired_seed_differences(results)
    paired.to_csv(outputs_root / 'paired_seed_differences.csv', index=False)

    summary_text = concise_text_summary(summary)
    with open(outputs_root / 'concise_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')

    print('\n=== Seed-level results ===')
    print(results.to_string(index=False))
    print('\n=== Aggregated summary ===')
    print(summary.to_string(index=False))
    if not paired.empty:
        print('\n=== Short statistical analysis (paired differences across identical seeds) ===')
        print(paired.to_string(index=False))
    print('\n=== Concise comparison ===')
    print(summary_text)
    print(f"\nArtifacts written to: {outputs_root}")


if __name__ == '__main__':
    main()
