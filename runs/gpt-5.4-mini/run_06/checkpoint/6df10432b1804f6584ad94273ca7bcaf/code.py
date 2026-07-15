import os
import json
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, HR, Recall
from omnirec.util.util import set_random_state

from lenskit.data import from_interactions_df, Dataset as LKDataset
from lenskit.als import ImplicitMFScorer
from lenskit.knn import ItemKNNScorer
from lenskit.basic import PopScorer
from lenskit.metrics import quick_measure_model


working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

SEEDS = [7, 13, 29, 42, 101]
DATASETS = {
    'MovieLens100K': DataSet.MovieLens100K,
    'Amazon Video Games': DataSet.Amazon2014VideoGames,
    'Last.FM': DataSet.HetrecLastFM,
}
ALGORITHMS = {
    'ALS': ImplicitMFScorer,
    'ItemKNN': ItemKNNScorer,
    'Pop': PopScorer,
}
K_VALUES = [1, 5, 10]


def load_and_preprocess(dataset_enum):
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    pipe_steps = []
    name = getattr(dataset_enum, 'name', str(dataset_enum))
    if name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        pipe_steps.append(MakeImplicit(3))
    pipe_steps.append(CorePruning(5))
    for step in pipe_steps:
        ds = step.process(ds)
    return ds


def get_df(dataset):
    data = dataset._data  # public variant access is not documented; avoid if possible
    if hasattr(data, 'df'):
        return data.df.copy()
    raise ValueError('Dataset does not expose a DataFrame in the expected way')


def make_split(df: pd.DataFrame, seed: int):
    set_random_state(seed)
    base = RecSysDataSet.use_dataloader(dataset_enum) if False else None
    if base is None:
        raise RuntimeError('RecSysDataSet.from_df is unavailable in this environment')
    split_ds = UserHoldout(validation_size=0.0, test_size=0.2).process(base)
    return split_ds


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, algo_name: str):
    train_lk = from_interactions_df(train_df[['user', 'item', 'rating']].copy())
    test_lk = from_interactions_df(test_df[['user', 'item', 'rating']].copy())
    if algo_name == 'ALS':
        model = ImplicitMFScorer(embedding_size=50, epochs=10)
    elif algo_name == 'ItemKNN':
        model = ItemKNNScorer(max_nbrs=100)
    elif algo_name == 'Pop':
        model = PopScorer()
    else:
        raise ValueError(algo_name)
    result = quick_measure_model(model, train_lk)
    return result


def summarize(values: List[float]) -> Dict[str, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = float(std / mean) if mean != 0 else float('nan')
    return {'mean': mean, 'std': std, 'cv': cv, 'min': float(np.min(values)), 'max': float(np.max(values))}


def main():
    all_rows = []
    for dname, denum in DATASETS.items():
        print(f'Loading {dname}')
        ds = load_and_preprocess(denum)
        if hasattr(ds, 'to_frame'):
            df = ds.to_frame().copy()
        else:
            df = ds._data.df.copy()
        if 'rating' not in df.columns:
            df['rating'] = 1
        for seed in SEEDS:
            print(f'  Split seed={seed}')
            set_random_state(seed)
            split_ds = UserHoldout(validation_size=0.0, test_size=0.2).process(RecSysDataSet.use_dataloader(denum))
            split_df = split_ds._data
            train_df = split_df.train.copy()
            test_df = split_df.test.copy()
            for algo_name in ALGORITHMS:
                print(f'    Training {algo_name}')
                metrics = train_and_eval(train_df, test_df, algo_name)
                row = {'dataset': dname, 'seed': seed, 'algorithm': algo_name}
                row.update(metrics.to_dict() if hasattr(metrics, 'to_dict') else dict(metrics))
                all_rows.append(row)
    results = pd.DataFrame(all_rows)
    results_path = os.path.join(working_dir, 'seed_effect_results.csv')
    results.to_csv(results_path, index=False)
    print(results.head())

    summary_rows = []
    metric_cols = [c for c in results.columns if any(m in c.lower() for m in ['ndcg', 'precision'])]
    for dname, algo in results.groupby(['dataset', 'algorithm']).groups.keys():
        grp = results[(results['dataset'] == dname) & (results['algorithm'] == algo)]
        for metric in metric_cols:
            stats = summarize(grp[metric].tolist())
            summary_rows.append({'dataset': dname, 'algorithm': algo, 'metric': metric, **stats})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(working_dir, 'seed_effect_summary.csv'), index=False)
    print(summary)

    print('Exact seeds used:', SEEDS)
    print('Short analysis: compare std and CV across metrics; if CV is low (<~5%), seed effects are minor; larger CV suggests split sensitivity.')


if __name__ == '__main__':
    main()
