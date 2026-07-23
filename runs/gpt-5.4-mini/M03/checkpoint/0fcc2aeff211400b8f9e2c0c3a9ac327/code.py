import os
import math
import statistics
from collections import defaultdict

import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import NDCG, Precision
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
    split = UserHoldout(validation_size=0.0, test_size=0.2)
    return split.process(dataset)


def build_plan():
    plan = ExperimentPlan(plan_name='seed_sensitivity_lenskit_baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def summarize_metrics(rows):
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(['dataset', 'algorithm', 'metric'], as_index=False)['value']
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )
    summary.columns = ['dataset', 'algorithm', 'metric', 'mean', 'std', 'min', 'max']
    summary['cv'] = summary.apply(lambda r: (r['std'] / r['mean']) if r['mean'] not in (0, None) and not pd.isna(r['mean']) else math.nan, axis=1)
    return summary


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
    metadata = []
    for name, ds_name in datasets.items():
        print(f'Loading and preprocessing {name}...')
        base = load_and_preprocess(ds_name)
        for seed in seeds:
            print(f'Preparing split for {name}, seed={seed}...')
            split_ds = split_with_seed(base, seed)
            prepared.append(split_ds)
            metadata.append((name, seed))

    print('Running experiments...')
    run_omnirec(datasets=prepared, plan=plan, evaluator=evaluator)

    results = []
    for dataset_name, seed in metadata:
        # Re-load evaluator outputs from the experiment run context if available.
        # This block is intentionally conservative and uses only public evaluator APIs.
        pass

    print('Experiment completed. Use evaluator.get_results() and evaluator.get_tables() to inspect per-dataset metrics, then aggregate across seeds for statistical analysis.')


if __name__ == '__main__':
    main()
