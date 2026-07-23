import os
import json
import math
import statistics
from collections import defaultdict

import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def load_and_preprocess(dataset_name):
    dataset = RecSysDataSet.use_dataloader(dataset_name)
    steps = []
    if dataset_name in {DataSet.MovieLens100K, DataSet.Amazon2014VideoGames}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(dataset)


def split_with_seed(dataset, seed):
    set_random_state(seed)
    split = UserHoldout(validation_size=0.2, test_size=0.2)
    return split.process(dataset)


def build_plan():
    plan = ExperimentPlan(plan_name='seed_sensitivity_lenskit_baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def summarize(df):
    agg = df.groupby(['dataset', 'algorithm', 'metric'])['value'].agg(['mean', 'std', 'min', 'max']).reset_index()
    return agg


def seed_sensitivity_analysis(df):
    rows = []
    for (dataset, algorithm, metric), g in df.groupby(['dataset', 'algorithm', 'metric']):
        vals = g['value'].tolist()
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        cv = std / mean if mean not in (0, None) else math.nan
        rows.append({'dataset': dataset, 'algorithm': algorithm, 'metric': metric, 'mean': mean, 'std': std, 'cv': cv})
    return pd.DataFrame(rows)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    seeds = [1, 2, 3, 4, 5]
    datasets = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': DataSet.Amazon2014VideoGames,
        'HetrecLastFM': DataSet.HetrecLastFM,
    }

    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    prepared = []
    for name, ds_name in datasets.items():
        print(f'Loading and preprocessing {name}...')
        base = load_and_preprocess(ds_name)
        for seed in seeds:
            print(f'Preparing split for {name}, seed={seed}...')
            split_ds = split_with_seed(base, seed)
            prepared.append(split_ds)

    print('Running experiments...')
    run_omnirec(datasets=prepared, plan=plan, evaluator=evaluator)

    print('Please inspect OmniRec checkpoint outputs for per-run metrics, then aggregate them into a results table.')


if __name__ == '__main__':
    main()
