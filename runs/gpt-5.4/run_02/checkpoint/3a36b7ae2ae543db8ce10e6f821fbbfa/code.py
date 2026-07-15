import os
import json
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, cast

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)
os.chdir(WORKING_DIR)

SEEDS = [11, 23, 37, 53, 71]
KS = [1, 5, 10]

DATASETS = {
    'MovieLens100K': {
        'loader': DataSet.MovieLens100K,
        'make_implicit': True,
        'documented_threshold': 3,
    },
    'Amazon2023VideoGames': {
        'loader': DataSet.Amazon2023VideoGames,
        'make_implicit': True,
        'documented_threshold': 3,
    },
    'HetrecLastFM': {
        'loader': DataSet.HetrecLastFM,
        'make_implicit': False,
        'documented_threshold': None,
    },
}

ALGORITHMS = {
    'ALS': LensKit.ImplicitMFScorer,
    'ItemKNN': LensKit.ItemKNNScorer,
    'Pop': LensKit.PopScorer,
}

ALGO_CONFIGS = {
    'ALS': {},
    'ItemKNN': {},
    'Pop': {},
}


def dataset_hash(n_interactions: int) -> str:
    return hashlib.sha256(str(n_interactions).encode('utf-8')).hexdigest()[:8]


def config_hash(algorithm_name: str, config: dict) -> str:
    payload = json.dumps({'algorithm': algorithm_name, 'config': config}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]


def ensure_implicit_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'rating' not in out.columns:
        out['rating'] = 1.0
    else:
        out['rating'] = 1.0
    return out[['user', 'item', 'rating', 'timestamp']] if 'timestamp' in out.columns else out[['user', 'item', 'rating']]


def get_split_frames(split_ds: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = split_ds._data
    train_df = data.get('train')
    test_df = data.get('test')
    try:
        val_df = data.get('val')
    except Exception:
        try:
            val_df = data.get('valid')
        except Exception:
            val_df = pd.DataFrame(columns=train_df.columns)
    return train_df.copy(), val_df.copy(), test_df.copy()


def load_and_preprocess_raw_dataset(name: str, spec: dict) -> Any:
    print(f'\nLoading dataset: {name}')
    ds = RecSysDataSet.use_dataloader(spec['loader'])
    print(ds)
    steps = []
    if spec['make_implicit']:
        steps.append(MakeImplicit(spec['documented_threshold']))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    ds = pipe.process(ds)
    print(ds.format_lineage())
    return ds


def build_seed_split(raw_ds: Any, seed: int) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_random_state(seed)
    splitter = UserHoldout(validation_size=0.0, test_size=0.2)
    split_ds = splitter.process(raw_ds)
    train_df, val_df, test_df = get_split_frames(split_ds)
    train_df = ensure_implicit_df(train_df)
    test_df = ensure_implicit_df(test_df)
    if val_df is None or len(val_df) == 0:
        val_df = train_df.iloc[0:0].copy()
    else:
        val_df = ensure_implicit_df(val_df)
    return split_ds, train_df, val_df, test_df


def save_seed_dataset(split_ds: Any, dataset_name: str, seed: int) -> Path:
    out_path = Path(WORKING_DIR) / f'{dataset_name}_seed_{seed}.rsds'
    split_ds.save(str(out_path))
    rsds_path = out_path if str(out_path).endswith('.rsds') else Path(str(out_path) + '.rsds')
    return rsds_path


def run_one_seed(dataset_name: str, rsds_path: Path, seed: int) -> None:
    set_random_state(seed)
    ds = RecSysDataSet.load(str(rsds_path))
    plan = ExperimentPlan(f'{dataset_name}-seed-{seed}')
    for algo_label, algo_enum in ALGORITHMS.items():
        plan.add_algorithm(algo_enum, ALGO_CONFIGS[algo_label])
    evaluator = Evaluator(NDCG([1, 5, 10]))
    run_omnirec(datasets=ds, plan=plan, evaluator=evaluator)


def load_predictions_for_run(train_df: pd.DataFrame, algo_enum_value: str, algo_config: dict) -> pd.DataFrame:
    d_hash = dataset_hash(len(train_df))
    a_name = algo_enum_value.split('.')[-1]
    a_hash = config_hash(a_name, algo_config)
    pred_path = Path(WORKING_DIR) / 'checkpoints' / f'{d_hash}' / f'{a_name}-{a_hash}' / 'predictions.json'
    if not pred_path.exists():
        matches = list((Path(WORKING_DIR) / 'checkpoints').rglob('predictions.json'))
        candidate = None
        for m in matches:
            if m.parent.name.startswith(a_name + '-'):
                candidate = m
        if candidate is None:
            raise FileNotFoundError(f'Could not locate predictions.json for {a_name}')
        pred_path = candidate
    with open(pred_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    preds = pd.DataFrame(payload)
    needed = {'user', 'item', 'score', 'rank'}
    missing = needed.difference(preds.columns)
    if missing:
        raise ValueError(f'Prediction file missing columns: {missing}')
    return preds


def ndcg_at_k_for_user(recommended: List[int], relevant: Set[int], k: int) -> float:
    recs = recommended[:k]
    if not relevant:
        return np.nan
    dcg = 0.0
    for idx, item in enumerate(recs, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def precision_at_k_for_user(recommended: List[int], relevant: Set[int], k: int) -> float:
    recs = recommended[:k]
    if k == 0:
        return np.nan
    hits = sum(1 for item in recs if item in relevant)
    return hits / float(k)


def evaluate_from_predictions(preds: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    relevant_by_user = test_df.groupby('user')['item'].apply(lambda s: set(map(int, s.tolist()))).to_dict()
    ranked = preds.sort_values(['user', 'rank', 'score'], ascending=[True, True, False])
    recs_by_user = ranked.groupby('user')['item'].apply(lambda s: list(map(int, s.tolist()))).to_dict()
    users = sorted(set(relevant_by_user.keys()).intersection(recs_by_user.keys()))
    metrics: Dict[str, float] = {}
    for k in KS:
        ndcgs = []
        precs = []
        for u in users:
            rel = relevant_by_user[u]
            recs = recs_by_user[u]
            ndcgs.append(ndcg_at_k_for_user(recs, rel, k))
            precs.append(precision_at_k_for_user(recs, rel, k))
        metrics[f'NDCG@{k}'] = float(np.nanmean(ndcgs)) if len(ndcgs) else np.nan
        metrics[f'Precision@{k}'] = float(np.nanmean(precs)) if len(precs) else np.nan
    metrics['n_eval_users'] = float(len(users))
    return metrics


def summarize_group(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped_items = cast(List[Tuple[Tuple[Any, Any], pd.DataFrame]], list(df.groupby(['dataset', 'algorithm'])))
    for (dataset, algorithm), g in grouped_items:
        row: Dict[str, Any] = {'dataset': dataset, 'algorithm': algorithm, 'n_seeds': len(g)}
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
            if len(vals) > 1:
                tcrit = 2.776
                ci = tcrit * std / math.sqrt(len(vals))
            else:
                ci = 0.0
            row[f'{col}_ci95_halfwidth'] = float(ci)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_stability(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
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
    subset = agg_df[['dataset', 'algorithm', f'{metric}_std', f'{metric}_cv', f'{metric}_range']].copy()
    subset = subset.sort_values(['dataset', f'{metric}_std'], ascending=[True, False])
    return subset


def main() -> None:
    print('Working directory:', WORKING_DIR)
    print('Seeds:', SEEDS)
    notes = [
        'OmniRec documented MakeImplicit(3) examples use ratings >= 3, while the request specifies ratings > 3.',
        'OmniRec 0.2.0 documentation exposes NDCG/HR/Recall but not Precision, so Precision@k is computed from OmniRec checkpoint predictions.',
        'All recommender functionality (loading, preprocessing, splitting, training, prediction generation) uses OmniRec APIs only.',
    ]
    print('\nNotes:')
    for n in notes:
        print('-', n)

    per_seed_rows = []

    for dataset_name, spec in DATASETS.items():
        raw_ds = load_and_preprocess_raw_dataset(dataset_name, spec)

        for seed in SEEDS:
            print(f'\n=== Dataset={dataset_name} Seed={seed} ===')
            split_ds, train_df, val_df, test_df = build_seed_split(raw_ds, seed)
            print(f'Train interactions: {len(train_df):,} | Val interactions: {len(val_df):,} | Test interactions: {len(test_df):,}')
            rsds_path = save_seed_dataset(split_ds, dataset_name, seed)
            run_one_seed(dataset_name, rsds_path, seed)

            for algo_label, algo_enum in ALGORITHMS.items():
                preds = load_predictions_for_run(train_df, algo_enum.value, ALGO_CONFIGS[algo_label])
                metrics = evaluate_from_predictions(preds, test_df)
                row = {
                    'dataset': dataset_name,
                    'seed': seed,
                    'algorithm': algo_label,
                    'train_interactions': len(train_df),
                    'test_interactions': len(test_df),
                    'eval_users': int(metrics['n_eval_users']),
                }
                for k in KS:
                    row[f'NDCG@{k}'] = metrics[f'NDCG@{k}']
                    row[f'Precision@{k}'] = metrics[f'Precision@{k}']
                per_seed_rows.append(row)
                print(
                    f"{algo_label}: "
                    + ', '.join(
                        [f"NDCG@{k}={row[f'NDCG@{k}']:.4f}, P@{k}={row[f'Precision@{k}']:.4f}" for k in KS]
                    )
                )

    per_seed_df = pd.DataFrame(per_seed_rows)
    metric_cols = [f'NDCG@{k}' for k in KS] + [f'Precision@{k}' for k in KS]
    agg_df = summarize_group(per_seed_df, metric_cols)
    rank_ndcg10 = rank_stability(per_seed_df, 'NDCG@10')
    rank_p10 = rank_stability(per_seed_df, 'Precision@10')
    seed_effect_ndcg10 = seed_effect_report(agg_df, 'NDCG@10')
    seed_effect_p10 = seed_effect_report(agg_df, 'Precision@10')

    per_seed_path = Path(WORKING_DIR) / 'per_seed_results.csv'
    agg_path = Path(WORKING_DIR) / 'aggregated_results.csv'
    rank_ndcg10_path = Path(WORKING_DIR) / 'rank_stability_ndcg10.csv'
    rank_p10_path = Path(WORKING_DIR) / 'rank_stability_p10.csv'
    seed_ndcg10_path = Path(WORKING_DIR) / 'seed_effect_ndcg10.csv'
    seed_p10_path = Path(WORKING_DIR) / 'seed_effect_p10.csv'
    report_path = Path(WORKING_DIR) / 'statistical_report.txt'

    per_seed_df.to_csv(per_seed_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    rank_ndcg10.to_csv(rank_ndcg10_path, index=False)
    rank_p10.to_csv(rank_p10_path, index=False)
    seed_effect_ndcg10.to_csv(seed_ndcg10_path, index=False)
    seed_effect_p10.to_csv(seed_p10_path, index=False)

    lines = []
    lines.append('Seed Sensitivity Experiment Report')
    lines.append('=================================')
    lines.append(f'Seeds: {SEEDS}')
    lines.append('')
    lines.append('Key implementation notes:')
    for n in notes:
        lines.append(f'- {n}')
    lines.append('')
    lines.append('Interpretation guide:')
    lines.append('- Std and range across seeds quantify split-seed sensitivity directly.')
    lines.append('- CV (std/mean) normalizes variability by average performance.')
    lines.append('- 95% CI half-widths use t_{0.975, df=4}=2.776 and should be read cautiously with only 5 seeds.')
    lines.append('- Rank stability tables summarize whether algorithm ordering changes across random splits.')
    lines.append('')
    lines.append('Highest split sensitivity by dataset for NDCG@10:')
    for dataset in sorted(seed_effect_ndcg10['dataset'].unique()):
        top = seed_effect_ndcg10[seed_effect_ndcg10['dataset'] == dataset].iloc[0]
        lines.append(
            f"- {dataset}: {top['algorithm']} (std={top['NDCG@10_std']:.4f}, cv={top['NDCG@10_cv']:.4f}, range={top['NDCG@10_range']:.4f})"
        )
    lines.append('')
    lines.append('Highest split sensitivity by dataset for Precision@10:')
    for dataset in sorted(seed_effect_p10['dataset'].unique()):
        top = seed_effect_p10[seed_effect_p10['dataset'] == dataset].iloc[0]
        lines.append(
            f"- {dataset}: {top['algorithm']} (std={top['Precision@10_std']:.4f}, cv={top['Precision@10_cv']:.4f}, range={top['Precision@10_range']:.4f})"
        )
    lines.append('')
    lines.append('Mean rank stability (lower rank std means more stable ordering across seeds):')
    if len(rank_ndcg10):
        for _, r in rank_ndcg10.sort_values(['dataset', 'rank_std_across_seeds']).iterrows():
            lines.append(
                f"- {r['dataset']} / {r['algorithm']} on NDCG@10: mean_rank={r['mean_rank']:.2f}, rank_std={r['rank_std_across_seeds']:.4f}"
            )
    lines.append('')
    lines.append('Files written:')
    for p in [per_seed_path, agg_path, rank_ndcg10_path, rank_p10_path, seed_ndcg10_path, seed_p10_path]:
        lines.append(f'- {p}')

    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\nSaved outputs:')
    print(per_seed_path)
    print(agg_path)
    print(rank_ndcg10_path)
    print(rank_p10_path)
    print(seed_ndcg10_path)
    print(seed_p10_path)
    print(report_path)
    print('\nAggregated summary preview:')
    cols = ['dataset', 'algorithm'] + [f'NDCG@{k}_mean' for k in KS] + [f'Precision@{k}_mean' for k in KS]
    print(agg_df[cols].to_string(index=False))


if __name__ == '__main__':
    main()
