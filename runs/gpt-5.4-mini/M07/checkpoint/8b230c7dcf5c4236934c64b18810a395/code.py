import os
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import Precision
from omnirec.util.util import set_random_state
from omnirec.util.run import run_omnirec


def build_dataset(name: str) -> RecSysDataSet:
    if name == 'MovieLens100K':
        return RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    if name == 'Amazon2014VideoGames':
        return RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
    if name == 'HetrecLastFM':
        return RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
    raise ValueError(f'Unknown dataset: {name}')


def make_pipeline(dataset_name: str) -> Pipe:
    steps = []
    if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    # Valid 80/20 user-based holdout: 20% test, 0% validation expressed as 0 by API only if supported.
    # To avoid the invalid split path, use a tiny nonzero validation split and report the effective 80/20 test split.
    steps.append(UserHoldout(validation_size=0.0, test_size=0.2))
    return Pipe(*steps)


def summarize_seed_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.startswith('NDCG@') or c.startswith('Precision@')]
    rows = []
    grouped = df.groupby(['dataset', 'algorithm'], as_index=False)
    for g in grouped:
        dataset = g['dataset'].iloc[0]
        algorithm = g['algorithm'].iloc[0]
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors='coerce').dropna().to_numpy()
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cv = float(std / mean) if mean != 0 else np.nan
            rows.append({
                'dataset': dataset,
                'algorithm': algorithm,
                'metric': col,
                'mean': mean,
                'std': std,
                'cv': cv,
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            })
    return pd.DataFrame(rows).sort_values(['dataset', 'algorithm', 'metric']).reset_index(drop=True)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']

    all_results = []
    metadata = []

    for dataset_name in dataset_names:
        print(f'\n=== Loading {dataset_name} ===')
        raw_dataset = build_dataset(dataset_name)
        metadata.append({'dataset': dataset_name})

        for seed in seeds:
            print(f'\n--- {dataset_name} | seed={seed} ---')
            set_random_state(seed)

            dataset = build_dataset(dataset_name)
            split_ds = make_pipeline(dataset_name).process(dataset)

            plan = ExperimentPlan(plan_name=f'{dataset_name}_seed_{seed}')
            plan.add_algorithm(LensKit.ImplicitMFScorer)
            plan.add_algorithm(LensKit.ItemKNNScorer)
            plan.add_algorithm(LensKit.PopScorer)

            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )

            result = run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            if hasattr(result, 'to_df'):
                res_df = result.to_df().copy()
            elif isinstance(result, pd.DataFrame):
                res_df = result.copy()
            else:
                res_df = pd.DataFrame(result)

            res_df['dataset'] = dataset_name
            res_df['seed'] = seed
            all_results.append(res_df)
            print(res_df)

    results = pd.concat(all_results, ignore_index=True)
    results_path = os.path.join(working_dir, 'seed_sensitivity_results.csv')
    results.to_csv(results_path, index=False)

    summary = summarize_seed_sensitivity(results)
    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    summary.to_csv(summary_path, index=False)

    print('\n=== Dataset metadata ===')
    print(pd.DataFrame(metadata))
    print('\n=== Seed sensitivity summary ===')
    print(summary)

    print('\nBrief statistical analysis:')
    grouped_summary = summary.groupby(['dataset', 'algorithm'])
    for keys, g in grouped_summary:
        clean = g.replace([np.inf, -np.inf], np.nan)
        max_cv = clean['cv'].max()
        avg_std = clean['std'].mean()
        print(f'- {g["dataset"].iloc[0]} / {g["algorithm"].iloc[0]}: max CV={max_cv:.4f}, mean std={avg_std:.4f}')

    print(f'\nSaved results to: {results_path}')
    print(f'Saved summary to: {summary_path}')


if __name__ == '__main__':
    main()
