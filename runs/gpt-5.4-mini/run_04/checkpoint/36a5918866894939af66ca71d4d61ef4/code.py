import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# NOTE: The implementation below is written to follow the verified OmniRec/LensKit APIs.
# It assumes the exact dataset enum names are available in the installed OmniRec build.
# If an enum alias differs, adjust only the dataset name mapping section.

WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)

SEEDS = [7, 21, 42, 87, 123]
DATASET_SPECS = [
    ("MovieLens100K", True),
    ("Amazon2014VideoGames", True),
    ("HetrecLastFM", False),
]
K_VALUES = [1, 5, 10]
ALGORITHMS = ["ALS", "ItemKNN", "Pop"]


def _get_omnirec():
    from omnirec import RecSysDataSet
    from omnirec.data_loaders.datasets import DataSet
    from omnirec.preprocess.pipe import Pipe
    from omnirec.preprocess.core_pruning import CorePruning
    from omnirec.preprocess.feedback_conversion import MakeImplicit
    from omnirec.preprocess.split import UserHoldout
    from omnirec.metrics.ranking import NDCG, Precision
    return RecSysDataSet, DataSet, Pipe, CorePruning, MakeImplicit, UserHoldout, NDCG, Precision


def _load_dataset(dataset_enum):
    RecSysDataSet, DataSet, *_ = _get_omnirec()
    return RecSysDataSet.use_dataloader(dataset_enum)


def _build_preprocess_pipeline(Pipe, CorePruning, MakeImplicit, UserHoldout, make_implicit: bool, seed: int):
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    steps.append(UserHoldout(validation_size=0.2, test_size=0.2, random_state=seed))
    return Pipe(*steps)


def _extract_split_df(dataset):
    data = dataset._data
    train = data.get('train') if hasattr(data, 'get') else data.train
    val = data.get('val') if hasattr(data, 'get') else data.val
    test = data.get('test') if hasattr(data, 'get') else data.test
    return train, val, test


def _make_lenskit_models():
    from lenskit.basic import PopScorer
    from lenskit.knn import ItemKNNScorer
    from lenskit.als import ImplicitMFScorer
    return {
        'ALS': ImplicitMFScorer(),
        'ItemKNN': ItemKNNScorer(),
        'Pop': PopScorer(),
    }


def _train_and_recommend(model, train_df, users, n=10):
    # This function is intentionally conservative and uses LensKit's pipeline/recommend API.
    from lenskit.pipeline import topn_pipeline
    from lenskit import recommend

    pipe = topn_pipeline(model)
    pipe.train(train_df)
    recs = []
    for u in users:
        out = recommend(pipe, u, n=n)
        recs.append((u, out))
    return recs


def _compute_metrics_at_k(recs_by_user, test_df, k):
    # Minimal, explicit ranking-metric computation for the requested cutoffs.
    # Assumes recommendation outputs are item lists with item identifiers in order.
    truth = test_df.groupby('user')['item'].apply(list).to_dict()
    ndcgs = []
    precisions = []
    for u, rec_items in recs_by_user:
        rel = set(truth.get(u, []))
        if not rel:
            continue
        topk = list(rec_items[:k])
        hits = [1 if i in rel else 0 for i in topk]
        precisions.append(sum(hits) / k)
        dcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(hits))
        ideal_len = min(len(rel), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else np.nan, float(np.mean(precisions)) if precisions else np.nan


def run_experiment():
    RecSysDataSet, DataSet, Pipe, CorePruning, MakeImplicit, UserHoldout, NDCG, Precision = _get_omnirec()
    results = []

    dataset_map = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': getattr(DataSet, 'Amazon2014VideoGames', getattr(DataSet, 'AmazonVideoGames', None)),
        'HetrecLastFM': getattr(DataSet, 'HetrecLastFM', getattr(DataSet, 'LastFM', None)),
    }
    if dataset_map['Amazon2014VideoGames'] is None or dataset_map['HetrecLastFM'] is None:
        raise AttributeError('Requested dataset enum names were not found in OmniRec DataSet registry.')

    for dataset_name, needs_implicit in DATASET_SPECS:
        raw = RecSysDataSet.use_dataloader(dataset_map[dataset_name])
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            pipeline = _build_preprocess_pipeline(Pipe, CorePruning, MakeImplicit, UserHoldout, needs_implicit, seed)
            split_ds = pipeline.process(raw)
            train_df, val_df, test_df = _extract_split_df(split_ds)
            users = sorted(test_df['user'].unique())

            models = _make_lenskit_models()
            for algo_name, model in models.items():
                recs = _train_and_recommend(model, train_df, users, n=max(K_VALUES))
                user_to_items = [(u, list(out['item']) if hasattr(out, '__getitem__') and 'item' in out else list(out)) for u, out in recs]
                row = {'dataset': dataset_name, 'seed': seed, 'algorithm': algo_name}
                for k in K_VALUES:
                    ndcg, prec = _compute_metrics_at_k(user_to_items, test_df, k)
                    row[f'NDCG@{k}'] = ndcg
                    row[f'Precision@{k}'] = prec
                results.append(row)
                print(f"{dataset_name} seed={seed} algo={algo_name} done")

    results_df = pd.DataFrame(results)
    results_path = os.path.join(WORKING_DIR, 'seed_sensitivity_results.csv')
    results_df.to_csv(results_path, index=False)

    summary = (
        results_df.groupby(['dataset', 'algorithm'])
        .agg(['mean', 'std', 'median', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    )
    summary_path = os.path.join(WORKING_DIR, 'seed_sensitivity_summary.csv')
    summary.to_csv(summary_path)

    print('Seeds:', SEEDS)
    print('Saved per-run results to:', results_path)
    print('Saved summary to:', summary_path)
    print(summary)

    return results_df, summary


if __name__ == '__main__':
    run_experiment()
