import os
import json
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.filter import RatingFilter
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

SEEDS = [11, 23, 37, 53, 71]
KS = [1, 5, 10]
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.2

DATASETS = {
    'MovieLens100K': {
        'loader': DataSet.MovieLens100K,
        'strict_gt3_to_implicit': True,
    },
    'Amazon2014VideoGames': {
        'loader': DataSet.Amazon2014VideoGames,
        'strict_gt3_to_implicit': True,
    },
    'HetrecLastFM': {
        'loader': DataSet.HetrecLastFM,
        'strict_gt3_to_implicit': False,
    },
}

ALGORITHMS = {
    'ALS': LensKit.ImplicitMFScorer,
    'ItemKNN': LensKit.ItemKNNScorer,
    'Pop': LensKit.PopScorer,
}

ALGO_CONFIGS: Dict[str, Dict[str, Any]] = {
    'ALS': {},
    'ItemKNN': {},
    'Pop': {},
}


def make_implicit_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['user', 'item']
    out = df.copy()
    out['rating'] = 1.0
    if 'timestamp' in out.columns:
        cols = ['user', 'item', 'rating', 'timestamp']
    else:
        cols = ['user', 'item', 'rating']
    return out[cols].reset_index(drop=True)


def get_split_frames(split_ds: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = split_ds._data.get('train').copy()
    try:
        valid_df = split_ds._data.get('val')
    except Exception:
        valid_df = None
    if valid_df is None:
        try:
            valid_df = split_ds._data.get('valid')
        except Exception:
            valid_df = None
    if valid_df is None:
        valid_df = train_df.iloc[0:0].copy()
    else:
        valid_df = valid_df.copy()
    test_df = split_ds._data.get('test').copy()
    return train_df, valid_df, test_df


def dataset_hash(n_interactions: int) -> str:
    return hashlib.sha256(str(n_interactions).encode('utf-8')).hexdigest()[:8]


def config_hash(algorithm_name: str, config: dict) -> str:
    payload = json.dumps({'algorithm': algorithm_name, 'config': config}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]


def find_prediction_file(algorithm_enum: Any, config: dict, train_df: pd.DataFrame) -> Path:
    checkpoints_root = Path(working_dir) / 'checkpoints'
    algo_name = algorithm_enum.value.split('.')[-1]
    expected = checkpoints_root / dataset_hash(len(train_df)) / f'{algo_name}-{config_hash(algo_name, config)}' / 'predictions.json'
    if expected.exists():
        return expected

    candidates = list(checkpoints_root.rglob('predictions.json'))
    candidates = [p for p in candidates if p.parent.name.startswith(algo_name + '-')]
    if not candidates:
        raise FileNotFoundError(f'No predictions.json found for algorithm {algo_name}')
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_predictions(pred_path: Path) -> pd.DataFrame:
    with open(pred_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    preds = pd.DataFrame(payload)
    required = {'user', 'item'}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(f'Prediction file missing required columns: {missing}')
    if 'rank' not in preds.columns:
        if 'score' in preds.columns:
            preds = preds.sort_values(['user', 'score'], ascending=[True, False]).copy()
        else:
            preds = preds.copy()
        preds['rank'] = preds.groupby('user').cumcount() + 1
    return preds


def ndcg_at_k_for_user(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return np.nan
    recs = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(recs, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def precision_at_k_for_user(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return np.nan
    recs = recommended[:k]
    hits = sum(1 for item in recs if item in relevant)
    return hits / float(k)


def evaluate_from_predictions(preds: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    relevant_by_user = test_df.groupby('user')['item'].apply(lambda s: set(map(int, s.tolist()))).to_dict()
    ranked = preds.sort_values(['user', 'rank'], ascending=[True, True]).copy()
    recs_by_user = ranked.groupby('user')['item'].apply(lambda s: list(map(int, s.tolist()))).to_dict()
    users = sorted(set(relevant_by_user.keys()).intersection(recs_by_user.keys()))

    metrics: Dict[str, float] = {'n_eval_users': float(len(users))}
    for k in KS:
        ndcgs = []
        precs = []
        for u in users:
            rel = relevant_by_user[u]
            recs = recs_by_user[u]
            ndcgs.append(ndcg_at_k_for_user(recs, rel, k))
            precs.append(precision_at_k_for_user(recs, rel, k))
        metrics[f'NDCG@{k}'] = float(np.nanmean(ndcgs)) if ndcgs else np.nan
        metrics[f'Precision@{k}'] = float(np.nanmean(precs)) if precs else np.nan
    return metrics


def summarize_group(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = cast(List[Tuple[Tuple[Any, Any], pd.DataFrame]], list(df.groupby(['dataset', 'algorithm'])))
    for (dataset, algorithm), g in grouped:
        row: Dict[str, Any] = {'dataset': dataset, 'algorithm': algorithm, 'n_seeds': int(len(g))}
        for col in metric_cols:
            vals = g[col].astype(float).to_numpy()
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{col}_mean'] = mean
            row[f'{col}_std'] = std
            row[f'{col}_cv'] = float(std / mean) if mean != 0 else np.nan
            row[f'{col}_min'] = float(np.min(vals))
            row[f'{col}_max'] = float(np.max(vals))
            row[f'{col}_range'] = float(np.max(vals) - np.min(vals))
            row[f'{col}_ci95_halfwidth'] = float(2.776 * std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def rank_stability(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset, g in df.groupby('dataset'):
        pivot = g.pivot(index='seed', columns='algorithm', values=metric)
        if pivot.shape[0] < 2:
            continue
        ranks = pivot.rank(axis=1, ascending=False, method='average')
        std_by_algo = ranks.std(axis=0, ddof=1)
        for algo in ranks.columns:
            rows.append({
                'dataset': dataset,
                'metric': metric,
                'algorithm': algo,
                'rank_std_across_seeds': float(std_by_algo[algo]),
                'mean_rank': float(ranks[algo].mean()),
            })
    return pd.DataFrame(rows)


def seed_effect_report(agg_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    cols = ['dataset', 'algorithm', f'{metric}_std', f'{metric}_cv', f'{metric}_range']
    return agg_df[cols].sort_values(['dataset', f'{metric}_std'], ascending=[True, False]).reset_index(drop=True)


def preprocess_dataset(dataset_name: str, spec: Dict[str, Any]) -> Any:
    print(f'\nLoading dataset: {dataset_name}')
    ds = RecSysDataSet.use_dataloader(spec['loader'])
    print(ds)

    steps = []
    if spec['strict_gt3_to_implicit']:
        steps.append(RatingFilter(lower=4))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    ds = pipe.process(ds)

    if spec['strict_gt3_to_implicit']:
        raw_df = ds._data.df.copy()
        ds._data.df = make_implicit_frame(raw_df)
    else:
        raw_df = ds._data.df.copy()
        if 'rating' not in raw_df.columns:
            ds._data.df = make_implicit_frame(raw_df)

    print(ds.format_lineage())
    print(f'Interactions after preprocessing: {ds.num_interactions():,}')
    return ds


def split_dataset(raw_ds: Any, seed: int) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_random_state(seed)
    splitter = UserHoldout(validation_size=VALIDATION_SIZE, test_size=TEST_SIZE)
    split_ds = splitter.process(raw_ds)
    train_df, valid_df, test_df = get_split_frames(split_ds)
    train_df = make_implicit_frame(train_df)
    valid_df = make_implicit_frame(valid_df) if len(valid_df) else train_df.iloc[0:0].copy()
    test_df = make_implicit_frame(test_df)
    return split_ds, train_df, valid_df, test_df


def save_split_dataset(split_ds: Any, dataset_name: str, seed: int) -> Path:
    path = Path(working_dir) / f'{dataset_name}_seed_{seed}.rsds'
    split_ds.save(str(path))
    if path.suffix != '.rsds':
        path = Path(str(path) + '.rsds')
    return path


def run_seed_experiment(dataset_name: str, rsds_path: Path, seed: int) -> None:
    set_random_state(seed)
    ds = RecSysDataSet.load(str(rsds_path))
    plan = ExperimentPlan(f'{dataset_name}-seed-{seed}')
    for algo_label, algo_enum in ALGORITHMS.items():
        plan.add_algorithm(algo_enum, ALGO_CONFIGS[algo_label])
    evaluator = Evaluator(NDCG(KS))
    run_omnirec(datasets=ds, plan=plan, evaluator=evaluator)


def main() -> None:
    print('Working directory:', working_dir)
    print('Seeds:', SEEDS)
    print('Validation/Test split:', VALIDATION_SIZE, TEST_SIZE)

    notes = [
        'Fixed the OmniRec UserHoldout crash by using a positive validation split: UserHoldout(validation_size=0.1, test_size=0.2).',
        'For MovieLens100K and Amazon2014VideoGames, strict rating > 3 is implemented as RatingFilter(lower=4), then interactions are converted to implicit format.',
        'All recommender training and prediction generation uses OmniRec with LensKit-backed algorithms via omnirec.runner.algos.LensKit.',
        'Precision@k is computed from OmniRec prediction outputs because OmniRec documents ranking metrics such as NDCG/HR/Recall but not Precision.',
    ]
    print('\nImplementation notes:')
    for note in notes:
        print('-', note)

    all_rows: List[Dict[str, Any]] = []

    for dataset_name, spec in DATASETS.items():
        raw_ds = preprocess_dataset(dataset_name, spec)

        for seed in SEEDS:
            print(f'\n=== Dataset={dataset_name} Seed={seed} ===')
            split_ds, train_df, valid_df, test_df = split_dataset(raw_ds, seed)
            print(f'Train: {len(train_df):,} | Valid: {len(valid_df):,} | Test: {len(test_df):,}')
            rsds_path = save_split_dataset(split_ds, dataset_name, seed)
            run_seed_experiment(dataset_name, rsds_path, seed)

            for algo_label, algo_enum in ALGORITHMS.items():
                pred_path = find_prediction_file(algo_enum, ALGO_CONFIGS[algo_label], train_df)
                preds = load_predictions(pred_path)
                metrics = evaluate_from_predictions(preds, test_df)
                row: Dict[str, Any] = {
                    'dataset': dataset_name,
                    'seed': seed,
                    'algorithm': algo_label,
                    'train_interactions': int(len(train_df)),
                    'valid_interactions': int(len(valid_df)),
                    'test_interactions': int(len(test_df)),
                    'eval_users': int(metrics['n_eval_users']),
                }
                for k in KS:
                    row[f'NDCG@{k}'] = metrics[f'NDCG@{k}']
                    row[f'Precision@{k}'] = metrics[f'Precision@{k}']
                all_rows.append(row)
                print(
                    f"{algo_label}: " + ', '.join([
                        f"NDCG@{k}={row[f'NDCG@{k}']:.4f}, P@{k}={row[f'Precision@{k}']:.4f}"
                        for k in KS
                    ])
                )

    per_seed_df = pd.DataFrame(all_rows)
    metric_cols = [f'NDCG@{k}' for k in KS] + [f'Precision@{k}' for k in KS]
    agg_df = summarize_group(per_seed_df, metric_cols)
    rank_ndcg10 = rank_stability(per_seed_df, 'NDCG@10')
    rank_p10 = rank_stability(per_seed_df, 'Precision@10')
    seed_effect_ndcg10 = seed_effect_report(agg_df, 'NDCG@10')
    seed_effect_p10 = seed_effect_report(agg_df, 'Precision@10')

    per_seed_path = Path(working_dir) / 'per_seed_results.csv'
    agg_path = Path(working_dir) / 'aggregated_results.csv'
    rank_ndcg10_path = Path(working_dir) / 'rank_stability_ndcg10.csv'
    rank_p10_path = Path(working_dir) / 'rank_stability_p10.csv'
    seed_ndcg10_path = Path(working_dir) / 'seed_effect_ndcg10.csv'
    seed_p10_path = Path(working_dir) / 'seed_effect_p10.csv'
    report_path = Path(working_dir) / 'statistical_report.txt'

    per_seed_df.to_csv(per_seed_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    rank_ndcg10.to_csv(rank_ndcg10_path, index=False)
    rank_p10.to_csv(rank_p10_path, index=False)
    seed_effect_ndcg10.to_csv(seed_ndcg10_path, index=False)
    seed_effect_p10.to_csv(seed_p10_path, index=False)

    lines: List[str] = []
    lines.append('Seed Sensitivity Experiment Report')
    lines.append('=================================')
    lines.append(f'Seeds: {SEEDS}')
    lines.append(f'UserHoldout(validation_size={VALIDATION_SIZE}, test_size={TEST_SIZE})')
    lines.append('')
    lines.append('Key implementation notes:')
    for note in notes:
        lines.append(f'- {note}')
    lines.append('')
    lines.append('Interpretation guide:')
    lines.append('- Std and range across seeds quantify split-seed sensitivity directly.')
    lines.append('- CV normalizes variability by mean performance.')
    lines.append('- 95% CI half-widths use t_{0.975, df=4}=2.776 and should be read cautiously with only 5 seeds.')
    lines.append('- Rank stability indicates whether algorithm ordering changes across random seeds.')
    lines.append('')

    if len(seed_effect_ndcg10):
        lines.append('Highest split sensitivity by dataset for NDCG@10:')
        for dataset in sorted(seed_effect_ndcg10['dataset'].unique()):
            top = seed_effect_ndcg10[seed_effect_ndcg10['dataset'] == dataset].iloc[0]
            lines.append(
                f"- {dataset}: {top['algorithm']} (std={top['NDCG@10_std']:.4f}, cv={top['NDCG@10_cv']:.4f}, range={top['NDCG@10_range']:.4f})"
            )
        lines.append('')

    if len(seed_effect_p10):
        lines.append('Highest split sensitivity by dataset for Precision@10:')
        for dataset in sorted(seed_effect_p10['dataset'].unique()):
            top = seed_effect_p10[seed_effect_p10['dataset'] == dataset].iloc[0]
            lines.append(
                f"- {dataset}: {top['algorithm']} (std={top['Precision@10_std']:.4f}, cv={top['Precision@10_cv']:.4f}, range={top['Precision@10_range']:.4f})"
            )
        lines.append('')

    if len(rank_ndcg10):
        lines.append('Rank stability for NDCG@10:')
        for _, r in rank_ndcg10.sort_values(['dataset', 'rank_std_across_seeds', 'algorithm']).iterrows():
            lines.append(
                f"- {r['dataset']} / {r['algorithm']}: mean_rank={r['mean_rank']:.2f}, rank_std={r['rank_std_across_seeds']:.4f}"
            )
        lines.append('')

    lines.append('Files written:')
    for p in [per_seed_path, agg_path, rank_ndcg10_path, rank_p10_path, seed_ndcg10_path, seed_p10_path, report_path]:
        lines.append(f'- {p}')

    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nSaved outputs:')
    for p in [per_seed_path, agg_path, rank_ndcg10_path, rank_p10_path, seed_ndcg10_path, seed_p10_path, report_path]:
        print(p)

    if len(agg_df):
        preview_cols = ['dataset', 'algorithm'] + [f'NDCG@{k}_mean' for k in KS] + [f'Precision@{k}_mean' for k in KS]
        print('\nAggregated summary preview:')
        print(agg_df[preview_cols].to_string(index=False))


if __name__ == '__main__':
    main()
