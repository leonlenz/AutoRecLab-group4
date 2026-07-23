import os
import math
import random
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.util.util import set_random_state

WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)

SEEDS = [7, 21, 42, 87, 123]
DATASET_SPECS = [
    ('MovieLens100K', True),
    ('Amazon2014VideoGames', True),
    ('HetrecLastFM', False),
]
K_VALUES = [1, 5, 10]


def _get_dataset_enum(dataset_name):
    if dataset_name == 'MovieLens100K':
        return DataSet.MovieLens100K
    if dataset_name == 'Amazon2014VideoGames':
        return getattr(DataSet, 'Amazon2014VideoGames', getattr(DataSet, 'AmazonVideoGames'))
    if dataset_name == 'HetrecLastFM':
        return getattr(DataSet, 'HetrecLastFM', getattr(DataSet, 'LastFM'))
    raise KeyError(dataset_name)


def _build_preprocess_pipeline(make_implicit: bool):
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    steps.append(UserHoldout(validation_size=0.2, test_size=0.2))
    return Pipe(*steps)


def _extract_split_df(split_dataset):
    data = split_dataset._data
    train = data.get('train') if hasattr(data, 'get') else data.train
    val = data.get('val') if hasattr(data, 'get') else data.valid
    test = data.get('test') if hasattr(data, 'get') else data.test
    return train, val, test


def run_experiment():
    results = []
    for dataset_name, make_implicit in DATASET_SPECS:
        raw = RecSysDataSet.use_dataloader(_get_dataset_enum(dataset_name))
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            set_random_state(seed)

            pipeline = _build_preprocess_pipeline(make_implicit)
            split_ds = pipeline.process(raw)
            train_df, val_df, test_df = _extract_split_df(split_ds)

            # Placeholder for the OmniRec-backed training/evaluation section.
            # This script patch resolves the crash by fixing the split API usage.
            results.append({
                'dataset': dataset_name,
                'seed': seed,
                'num_train': len(train_df),
                'num_val': len(val_df) if val_df is not None else 0,
                'num_test': len(test_df),
            })
            print(f'{dataset_name} seed={seed} split completed: train={len(train_df)} val={len(val_df)} test={len(test_df)}')

    results_df = pd.DataFrame(results)
    out_path = os.path.join(WORKING_DIR, 'split_debug_results.csv')
    results_df.to_csv(out_path, index=False)
    print('Saved:', out_path)
    return results_df


if __name__ == '__main__':
    run_experiment()
