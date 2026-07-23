import os
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List

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

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

SEEDS = [7, 13, 29, 42, 101]
DATASETS = {
    'MovieLens100K': DataSet.MovieLens100K,
    'Amazon Video Games': DataSet.Amazon2014VideoGames,
    'Last.FM': DataSet.HetrecLastFM,
}


def preprocess_dataset(dataset_name, dataset_enum):
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if dataset_name in {'MovieLens100K', 'Amazon Video Games'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    return pipe.process(ds)


def clone_dataset(ds):
    return RecSysDataSet.load(ds.save(os.path.join(working_dir, '_tmp_clone')))


def to_raw_dataset(ds):
    # Prefer public APIs; if the dataset is already split/folded, use the available trainable/raw access pattern.
    if hasattr(ds, 'to_frame'):
        df = ds.to_frame().copy()
    elif hasattr(ds, '_data') and hasattr(ds._data, 'df'):
        df = ds._data.df.copy()
    else:
        raise ValueError('Cannot extract dataframe from dataset')
    if 'rating' not in df.columns:
        df['rating'] = 1
    return df[['user', 'item', 'rating']].copy()


def summarize(values):
    values = [float(v) for v in values if pd.notna(v)]
    if not values:
        return {'mean': float('nan'), 'std': float('nan'), 'cv': float('nan'), 'min': float('nan'), 'max': float('nan')}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = float(std / mean) if mean != 0 else float('nan')
    return {'mean': mean, 'std': std, 'cv': cv, 'min': float(np.min(values)), 'max': float(np.max(values))}


def build_plan():
    plan = ExperimentPlan(plan_name='Seed-Sensitivity-Study')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def main():
    all_rows = []
    evaluator = Evaluator(NDCG([1, 5, 10]))

    for dataset_name, dataset_enum in DATASETS.items():
        print(f'Loading and preprocessing {dataset_name}')
        processed = preprocess_dataset(dataset_name, dataset_enum)

        for seed in SEEDS:
            print(f'  Running seed={seed}')
            set_random_state(seed)

            # Re-load and re-process per seed to keep the same protocol and allow seed-controlled splitting.
            seed_processed = preprocess_dataset(dataset_name, dataset_enum)
            split_ds = UserHoldout(validation_size=0.2, test_size=0.2).process(seed_processed)

            plan = build_plan()
            run_omnirec(
                datasets=split_ds,
                plan=plan,
                evaluator=evaluator,
            )

            # Collect results from checkpoint CSVs if available.
            results_file = os.path.join(working_dir, 'results.csv')
            if os.path.exists(results_file):
                seed_results = pd.read_csv(results_file)
                seed_results['dataset'] = dataset_name
                seed_results['seed'] = seed
                all_rows.append(seed_results)

    if all_rows:
        results = pd.concat(all_rows, ignore_index=True)
        results.to_csv(os.path.join(working_dir, 'seed_effect_results.csv'), index=False)

        metric_cols = [c for c in results.columns if 'ndcg' in c.lower() or 'precision' in c.lower()]
        summary_rows = []
        for (dataset_name, algorithm), grp in results.groupby(['dataset', 'algorithm']):
            for metric in metric_cols:
                stats = summarize(grp[metric].tolist())
                summary_rows.append({
                    'dataset': dataset_name,
                    'algorithm': algorithm,
                    'metric': metric,
                    **stats,
                })
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(os.path.join(working_dir, 'seed_effect_summary.csv'), index=False)
        print(summary)

    print('Seeds used:', SEEDS)
    print('Short analysis: compare the per-metric standard deviation and coefficient of variation across seeds; larger values indicate greater sensitivity to the split seed.')


if __name__ == '__main__':
    main()
