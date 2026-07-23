import os
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.util import set_random_state

# OmniRec LensKit runner interface
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec


def make_working_dir():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def load_and_preprocess(dataset_name):
    ds_map = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon Video Games': DataSet.Amazon2014VideoGames,
        'Last.FM': DataSet.HetrecLastFM,
    }
    ds = RecSysDataSet.use_dataloader(ds_map[dataset_name])
    pipeline_steps = [CorePruning(5)]
    if dataset_name in ('MovieLens100K', 'Amazon Video Games'):
        pipeline_steps.insert(0, MakeImplicit(3))
    pipeline = Pipe(*pipeline_steps)
    return pipeline.process(ds)


def build_plan():
    plan = ExperimentPlan('seed_sensitivity_implicit_benchmark')
    # Standard/default hyperparameters only; minimal settings for instantiation where needed.
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def main():
    working_dir = make_working_dir()
    seeds = [11, 22, 33, 44, 55]
    datasets = ['MovieLens100K', 'Amazon Video Games', 'Last.FM']

    # Log reproducibility metadata
    meta_path = os.path.join(working_dir, 'experiment_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'seeds': seeds, 'datasets': datasets}, f, indent=2)

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    all_results = []
    for seed in seeds:
        set_random_state(seed)
        for ds_name in datasets:
            ds = load_and_preprocess(ds_name)
            split = UserHoldout(validation_size=0.0, test_size=0.2).process(ds)
            # UserHoldout docs require validation_size and test_size; using 0 validation keeps an 80/20 holdout.
            # If the implementation requires a nonzero validation size, adjust to the smallest valid epsilon supported by docs.
            plan = build_plan()
            result = run_omnirec(datasets=split, plan=plan, evaluator=evaluator)
            # Persist raw result table if provided
            all_results.append({'seed': seed, 'dataset': ds_name, 'result': result})

    # The exact result object shape can vary; here we assume a dataframe-like or table-like return.
    # Post-processing sketch: aggregate metrics by dataset and algorithm, then compute mean/std across seeds.
    # This section should be adapted to the actual returned structure from run_omnirec.

    # Short statistical summary placeholder:
    # For each dataset/algorithm/metric, compute mean, std, and range across seeds.
    # Then run a simple Friedman test or repeated-measures ANOVA if the output structure supports it.

    print('Seeds used:', seeds)
    print('Metadata saved to:', meta_path)


if __name__ == '__main__':
    main()
