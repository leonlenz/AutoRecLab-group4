import os
import io
import json
import math
import hashlib
from pathlib import Path
from zipfile import ZipFile
import zipfile
from typing import Any, Dict, List, Set, Tuple, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG, register_dataloader
from omnirec.data_loaders.base import Loader, DatasetInfo
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)
RAW_DIR = os.path.join(WORKING_DIR, 'raw')
os.makedirs(RAW_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
KS = [1, 5, 10]


class MovieLens100KImplicitGT3(Loader):
    @staticmethod
    def info(name: str) -> DatasetInfo:
        return DatasetInfo(
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229",
        )

    @staticmethod
    def load(source_dir: Path, name: str) -> pd.DataFrame:
        with ZipFile(source_dir / 'ml-100k.zip') as zipf:
            with zipf.open('ml-100k/u.data') as file:
                df = pd.read_csv(
                    file,
                    sep='\t',
                    names=['user', 'item', 'rating', 'timestamp'],
                )
        df = df[df['rating'] > 3].copy()
        df['rating'] = 1
        return df[['user', 'item', 'rating', 'timestamp']]


class Amazon2014VideoGamesImplicitGT3(Loader):
    @staticmethod
    def info(name: str) -> DatasetInfo:
        return DatasetInfo(
            "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games.csv",
            None,
        )

    @staticmethod
    def load(source_dir: Path, name: str) -> pd.DataFrame:
        csv_path = source_dir / 'Video_Games.csv'
        if not csv_path.exists():
            gz_path = source_dir / 'Video_Games.csv.gz'
            json_gz_path = source_dir / 'Video_Games_5.json.gz'
            if gz_path.exists():
                df = pd.read_csv(gz_path, header=None)
            elif json_gz_path.exists():
                import gzip
                rows: List[Tuple[Any, Any, Any, Any]] = []
                with gzip.open(json_gz_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        obj = json.loads(line)
                        user = obj.get('reviewerID') or obj.get('user')
                        item = obj.get('asin') or obj.get('item')
                        rating = obj.get('overall') or obj.get('rating')
                        ts = obj.get('unixReviewTime') or obj.get('timestamp')
                        if user is not None and item is not None and rating is not None:
                            rows.append((user, item, rating, ts if ts is not None else 0))
                df = pd.DataFrame(rows, columns=['user', 'item', 'rating', 'timestamp'])
            else:
                candidates = list(source_dir.glob('*Video*Game*'))
                if not candidates:
                    raise FileNotFoundError(
                        f'Could not locate Amazon Video Games raw file under {source_dir}. '
                        'Expected OmniRec-downloaded files such as Video_Games.csv.'
                    )
                path = candidates[0]
                if path.suffix == '.csv':
                    df = pd.read_csv(path, header=None)
                else:
                    raise FileNotFoundError(f'Unsupported Amazon raw file: {path}')
        else:
            df = pd.read_csv(csv_path, header=None)

        if list(df.columns) != ['user', 'item', 'rating', 'timestamp']:
            if df.shape[1] >= 4:
                df = df.iloc[:, :4].copy()
                df.columns = ['user', 'item', 'rating', 'timestamp']
            else:
                raise ValueError('Amazon Video Games raw file does not have at least 4 columns.')

        df = df[df['rating'] > 3].copy()
        df['rating'] = 1
        return df[['user', 'item', 'rating', 'timestamp']]


class HetrecLastFMImplicitAll(Loader):
    @staticmethod
    def info(name: str) -> DatasetInfo:
        return DatasetInfo(
            "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
            "6738f48195667ff03caaab4d32ca9a3133d8cc026b7c3cdaf6ce1010e913c59c",
        )

    @staticmethod
    def load(source_dir: Path, name: str) -> pd.DataFrame:
        with zipfile.ZipFile(source_dir / 'hetrec2011-lastfm-2k.zip', 'r') as zipf:
            with zipf.open('user_taggedartists-timestamps.dat') as file:
                with io.TextIOWrapper(file, encoding='utf-8') as text_file:
                    df = pd.read_csv(
                        text_file,
                        sep='\t',
                        header=0,
                        usecols=['userID', 'artistID', 'timestamp'],
                    )
        df = df.rename(columns={'userID': 'user', 'artistID': 'item'})
        df['rating'] = 1
        return df[['user', 'item', 'rating', 'timestamp']]


def ensure_registration() -> None:
    register_dataloader('MovieLens100KImplicitGT3', MovieLens100KImplicitGT3)
    register_dataloader('Amazon2014VideoGamesImplicitGT3', Amazon2014VideoGamesImplicitGT3)
    register_dataloader('HetrecLastFMImplicitAll', HetrecLastFMImplicitAll)


def safe_name(x: str) -> str:
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in x)


def prepare_dataset(loader_name: str, seed: int) -> RecSysDataSet:
    set_random_state(seed)
    ds = RecSysDataSet.use_dataloader(
        loader_name,
        raw_dir=os.path.join(RAW_DIR, safe_name(loader_name)),
    )
    pipe = Pipe(
        CorePruning(5),
        UserHoldout(0.0, 0.2),
    )
    ds = pipe.process(ds)
    return ds


def build_plan(plan_name: str) -> ExperimentPlan:
    plan = ExperimentPlan(plan_name)
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {'feedback': 'implicit'})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def evaluator() -> Evaluator:
    return Evaluator(NDCG([1, 5, 10]))


def latest_file(root: str, filename: str) -> str:
    matches = []
    for dirpath, _, files in os.walk(root):
        if filename in files:
            p = os.path.join(dirpath, filename)
            matches.append((os.path.getmtime(p), p))
    if not matches:
        raise FileNotFoundError(f'Could not find {filename} under {root}')
    matches.sort()
    return matches[-1][1]


def read_predictions_file(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path)
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return pd.DataFrame(data)
        return pd.DataFrame(data)
    raise ValueError(f'Unsupported predictions file format: {path}')


def collect_run_files(checkpoint_root: str) -> Tuple[str, str]:
    pred_candidates = []
    test_candidates = []
    for dirpath, _, files in os.walk(checkpoint_root):
        for fn in files:
            low = fn.lower()
            full = os.path.join(dirpath, fn)
            if 'predict' in low and (low.endswith('.json') or low.endswith('.csv')):
                pred_candidates.append((os.path.getmtime(full), full))
            if low == 'test.csv' or low.endswith(os.path.sep + 'test.csv'):
                test_candidates.append((os.path.getmtime(full), full))
    if not pred_candidates:
        for dirpath, _, files in os.walk(checkpoint_root):
            for fn in files:
                low = fn.lower()
                full = os.path.join(dirpath, fn)
                if low in ('predictions.json', 'predictions.csv', 'prediction.json', 'prediction.csv'):
                    pred_candidates.append((os.path.getmtime(full), full))
    if not pred_candidates:
        raise FileNotFoundError(f'No prediction artifact found under {checkpoint_root}')
    if not test_candidates:
        raise FileNotFoundError(f'No test.csv artifact found under {checkpoint_root}')
    pred_candidates.sort()
    test_candidates.sort()
    return pred_candidates[-1][1], test_candidates[-1][1]


def ndcg_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    recs = recommended_items[:k]
    dcg = 0.0
    for idx, item in enumerate(recs, start=1):
        if item in relevant_items:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return np.nan
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else np.nan


def precision_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    recs = recommended_items[:k]
    if k == 0:
        return np.nan
    hits = sum(1 for item in recs if item in relevant_items)
    return hits / float(k)


def normalize_prediction_columns(pred: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cols = {c.lower(): c for c in pred.columns}
    if 'user' not in pred.columns:
        if 'user_id' in pred.columns:
            rename_map['user_id'] = 'user'
        elif 'userid' in cols:
            rename_map[cols['userid']] = 'user'
    if 'item' not in pred.columns:
        if 'item_id' in pred.columns:
            rename_map['item_id'] = 'item'
        elif 'itemid' in cols:
            rename_map[cols['itemid']] = 'item'
    if 'score' not in pred.columns:
        if 'prediction' in pred.columns:
            rename_map['prediction'] = 'score'
        elif 'rating' in pred.columns:
            rename_map['rating'] = 'score'
    pred = pred.rename(columns=rename_map)
    needed = {'user', 'item'}
    if not needed.issubset(set(pred.columns)):
        raise ValueError(f'Prediction file missing required columns. Found: {list(pred.columns)}')
    if 'rank' not in pred.columns:
        score_col = 'score' if 'score' in pred.columns else None
        if score_col is None:
            pred['score'] = 0.0
            score_col = 'score'
        pred = pred.sort_values(['user', score_col], ascending=[True, False]).copy()
        pred['rank'] = pred.groupby('user').cumcount() + 1
    else:
        pred = pred.sort_values(['user', 'rank']).copy()
    return pred


def compute_metrics_from_artifacts(pred_path: str, test_path: str) -> Dict[str, float]:
    pred = read_predictions_file(pred_path)
    pred = normalize_prediction_columns(pred)
    test = pd.read_csv(test_path)

    if 'user' not in test.columns and 'user_id' in test.columns:
        test = test.rename(columns={'user_id': 'user'})
    if 'item' not in test.columns and 'item_id' in test.columns:
        test = test.rename(columns={'item_id': 'item'})

    gt = test.groupby('user')['item'].apply(lambda s: set(map(int, s.tolist())))
    recs = pred.groupby('user')['item'].apply(lambda s: list(map(int, s.tolist())))
    common_users = sorted(set(gt.index).intersection(set(recs.index)))

    out = {}
    for k in KS:
        ndcgs = []
        precs = []
        for u in common_users:
            rel = gt.loc[u]
            rec = recs.loc[u]
            nd = ndcg_at_k(rec, rel, k)
            pr = precision_at_k(rec, rel, k)
            if not np.isnan(nd):
                ndcgs.append(nd)
            if not np.isnan(pr):
                precs.append(pr)
        out[f'nDCG@{k}'] = float(np.mean(ndcgs)) if ndcgs else np.nan
        out[f'Precision@{k}'] = float(np.mean(precs)) if precs else np.nan
    out['users_evaluated'] = int(len(common_users))
    return out


def summarize(group: pd.DataFrame, metric: str) -> Dict[str, float]:
    vals = group[metric].astype(float).values
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    rng = float(np.max(vals) - np.min(vals)) if len(vals) > 0 else 0.0
    cv = float(std / mean) if mean != 0 else np.nan
    return {'mean': mean, 'std': std, 'range': rng, 'cv': cv}


def run_one(dataset_label: str, loader_name: str, seed: int) -> List[Dict[str, object]]:
    ds = prepare_dataset(loader_name, seed)
    plan_name = f'{dataset_label}_seed_{seed}'
    plan = build_plan(plan_name)
    ev = evaluator()

    print(f'\n=== Running dataset={dataset_label} seed={seed} ===')
    print(ds)
    run_omnirec(datasets=ds, plan=plan, evaluator=ev)

    rows = []
    algo_map = {
        'ALS': 'ImplicitMFScorer',
        'ItemKNN': 'ItemKNNScorer',
        'Pop': 'PopScorer',
    }
    ckpt_root = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.exists(ckpt_root):
        ckpt_root = CHECKPOINT_DIR

    for algo_name, algo_key in algo_map.items():
        candidate_dirs = []
        for dirpath, dirnames, _ in os.walk(ckpt_root):
            if algo_key.lower() in dirpath.lower() and dataset_label.lower() in dirpath.lower():
                candidate_dirs.append(dirpath)
        if not candidate_dirs:
            for dirpath, dirnames, _ in os.walk(ckpt_root):
                if algo_key.lower() in dirpath.lower():
                    candidate_dirs.append(dirpath)
        if not candidate_dirs:
            raise FileNotFoundError(f'Could not locate checkpoint directory for {dataset_label} / {algo_name}')
        candidate_dirs = sorted(set(candidate_dirs), key=lambda p: os.path.getmtime(p))
        run_dir = candidate_dirs[-1]
        pred_path, test_path = collect_run_files(run_dir)
        metrics = compute_metrics_from_artifacts(pred_path, test_path)
        row = {
            'dataset': dataset_label,
            'seed': seed,
            'algorithm': algo_name,
        }
        row.update(metrics)
        print(f"{dataset_label} | seed={seed} | algo={algo_name} | " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items() if isinstance(v, float)))
        rows.append(row)
    return rows


def print_statistical_analysis(results: pd.DataFrame) -> None:
    print('\n\n===== Per-seed results =====')
    print(results.sort_values(['dataset', 'algorithm', 'seed']).to_string(index=False))

    summary_rows = []
    for grp_key, grp in results.groupby(['dataset', 'algorithm']):
        dataset, algorithm = cast(Tuple[Any, Any], grp_key)
        row = {'dataset': dataset, 'algorithm': algorithm}
        for metric in ['nDCG@1', 'nDCG@5', 'nDCG@10', 'Precision@1', 'Precision@5', 'Precision@10']:
            s = summarize(grp, metric)
            row[f'{metric}_mean'] = s['mean']
            row[f'{metric}_std'] = s['std']
            row[f'{metric}_range'] = s['range']
            row[f'{metric}_cv'] = s['cv']
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    print('\n===== Seed sensitivity summary =====')
    print(summary.to_string(index=False))

    print('\n===== Short statistical analysis =====')
    for grp_key, grp in results.groupby(['dataset', 'algorithm']):
        dataset, algorithm = cast(Tuple[Any, Any], grp_key)
        msg_parts = []
        for metric in ['nDCG@10', 'Precision@10']:
            s = summarize(grp, metric)
            strength = 'low'
            if not np.isnan(s['cv']):
                if s['cv'] >= 0.10:
                    strength = 'high'
                elif s['cv'] >= 0.05:
                    strength = 'moderate'
            msg_parts.append(
                f"{metric}: mean={s['mean']:.4f}, std={s['std']:.4f}, range={s['range']:.4f}, cv={s['cv']:.4f} ({strength} seed sensitivity)"
            )
        print(f'- {dataset} / {algorithm}: ' + '; '.join(msg_parts))

    out_csv = os.path.join(WORKING_DIR, 'seed_sensitivity_results.csv')
    out_summary = os.path.join(WORKING_DIR, 'seed_sensitivity_summary.csv')
    results.to_csv(out_csv, index=False)
    summary.to_csv(out_summary, index=False)
    print(f'\nSaved detailed results to: {out_csv}')
    print(f'Saved summary results to: {out_summary}')


if __name__ == '__main__':
    ensure_registration()

    datasets = [
        ('MovieLens100K', 'MovieLens100KImplicitGT3'),
        ('Amazon2014VideoGames', 'Amazon2014VideoGamesImplicitGT3'),
        ('HetrecLastFM', 'HetrecLastFMImplicitAll'),
    ]

    all_rows = []
    for dataset_label, loader_name in datasets:
        for seed in SEEDS:
            try:
                all_rows.extend(run_one(dataset_label, loader_name, seed))
            except Exception as e:
                print(f'FAILED for dataset={dataset_label}, seed={seed}: {e}')
                raise

    results_df = pd.DataFrame(all_rows)
    print_statistical_analysis(results_df)
