import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from omnirec import RecSysDataSet, NDCG, Recall
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

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


@dataclass
class RunRecord:
    dataset: str
    seed: int
    algorithm: str
    metric: str
    k: int
    value: float


def load_dataset(dataset_name: str) -> RecSysDataSet:
    return RecSysDataSet.use_dataloader(getattr(DataSet, dataset_name))


def preprocess_dataset(dataset_name: str) -> RecSysDataSet:
    ds = load_dataset(dataset_name)
    steps = []
    if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name='SeedSensitivity-LensKit-Baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def build_evaluator() -> Evaluator:
    return Evaluator(NDCG([1, 5, 10]), Recall([1, 5, 10]))


def extract_results(results: Dict[str, pd.DataFrame], dataset_name: str, seed: int) -> List[RunRecord]:
    out: List[RunRecord] = []
    for _, df in results.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            algo = str(row.get('algorithm', 'unknown'))
            metric = str(row.get('name', 'unknown'))
            k = row.get('k', np.nan)
            value = row.get('value', np.nan)
            if pd.notnull(value):
                out.append(RunRecord(dataset_name, seed, algo, metric, int(k) if pd.notnull(k) else -1, float(value)))
    return out


def short_statistical_analysis(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in results_df.groupby(['dataset', 'algorithm', 'metric', 'k']):
        dataset, algorithm, metric, k = tuple(keys)
        vals = grp['value'].astype(float).values
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = std / mean if mean != 0 else np.nan
        if len(vals) > 1:
            ci_low, ci_high = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=stats.sem(vals))
            p_val = float(stats.ttest_1samp(vals, 0.0).pvalue)
        else:
            ci_low, ci_high, p_val = np.nan, np.nan, np.nan
        rows.append({
            'dataset': dataset,
            'algorithm': algorithm,
            'metric': metric,
            'k': k,
            'mean': mean,
            'std': std,
            'cv': cv,
            'ci95_low': float(ci_low) if np.isfinite(ci_low) else np.nan,
            'ci95_high': float(ci_high) if np.isfinite(ci_high) else np.nan,
            'p_value_one_sample_vs_0': p_val,
        })
    return pd.DataFrame(rows)


def run_single_experiment(dataset_name: str, seed: int, plan: ExperimentPlan, evaluator: Evaluator) -> Dict[str, pd.DataFrame]:
    set_random_state(seed)
    processed = preprocess_dataset(dataset_name)
    split = UserHoldout(validation_size=0.2, test_size=0.2)
    split_ds = split.process(processed)
    run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
    return evaluator.get_results()


def main() -> None:
    seeds = [11, 22, 33, 44, 55]
    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']

    all_records: List[RunRecord] = []
    metadata = []

    for dataset_name in dataset_names:
        for seed in seeds:
            evaluator = build_evaluator()
            plan = build_plan()
            results = run_single_experiment(dataset_name, seed, plan, evaluator)
            all_records.extend(extract_results(results, dataset_name, seed))
            metadata.append({
                'dataset': dataset_name,
                'seed': seed,
                'preprocessing': 'MakeImplicit(3) for MovieLens100K/Amazon2014VideoGames; CorePruning(5)',
                'split': 'UserHoldout(validation_size=0.2, test_size=0.2)',
                'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
                'metrics': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10'],
            })

    results_df = pd.DataFrame([asdict(r) for r in all_records])
    results_df.to_csv(os.path.join(working_dir, 'all_seed_results_long.csv'), index=False)

    if not results_df.empty:
        summary = results_df.groupby(['dataset', 'algorithm', 'metric', 'k'])['value'].agg(['mean', 'std', 'count']).reset_index()
        summary.to_csv(os.path.join(working_dir, 'summary_mean_std.csv'), index=False)
        stats_df = short_statistical_analysis(results_df)
        stats_df.to_csv(os.path.join(working_dir, 'seed_variation_analysis.csv'), index=False)
        print(summary.to_string(index=False))
        print('\nShort statistical analysis:')
        print(stats_df.to_string(index=False))
    else:
        print('No results were collected. Check OmniRec evaluator results.')

    with open(os.path.join(working_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump({
            'datasets': dataset_names,
            'seeds': seeds,
            'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
            'preprocessing': {
                'MovieLens100K': ['MakeImplicit(3)', 'CorePruning(5)'],
                'Amazon2014VideoGames': ['MakeImplicit(3)', 'CorePruning(5)'],
                'HetrecLastFM': ['CorePruning(5)'],
            },
            'split': {'type': 'UserHoldout', 'validation_size': 0.2, 'test_size': 0.2},
            'metrics': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10'],
            'seed_control': 'omnirec global random state',
        }, f, indent=2)


if __name__ == '__main__':
    main()
