import os
import json
import math
from collections import defaultdict
from statistics import mean, pstdev

import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state
from omnirec.metrics.ranking import NDCG, HR


def load_and_preprocess(dataset_name, make_implicit_threshold=None):
    dataset = RecSysDataSet.use_dataloader(dataset_name)
    steps = [CorePruning(5)]
    if make_implicit_threshold is not None:
        steps.insert(0, MakeImplicit(make_implicit_threshold))
    dataset = Pipe(*steps).process(dataset)
    return dataset


def build_plan():
    plan = ExperimentPlan(plan_name='SeedSensitivity_LensKit_StandardDefaults')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def summarize(results_df):
    rows = []
    metric_cols = [c for c in results_df.columns if any(m in c for m in ['NDCG', 'Precision'])]
    group_cols = ['dataset', 'algorithm']
    for (dataset, algorithm), grp in results_df.groupby(group_cols):
        for col in metric_cols:
            vals = grp[col].dropna().tolist()
            if not vals:
                continue
            rows.append({
                'dataset': dataset,
                'algorithm': algorithm,
                'metric': col,
                'mean': mean(vals),
                'std': pstdev(vals) if len(vals) > 1 else 0.0,
                'min': min(vals),
                'max': max(vals),
                'n_seeds': len(vals),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    seed_list = [11, 22, 33, 44, 55]
    dataset_specs = [
        ('MovieLens100K', DataSet.MovieLens100K, 3),
        ('AmazonVideoGames', DataSet.Amazon2023VideoGames, 3),
        ('HetrecLastFM', DataSet.HetrecLastFM, None),
    ]

    all_results = []
    evaluator = Evaluator(NDCG([1, 5, 10]), HR([1, 5, 10]))
    plan = build_plan()

    for dataset_label, dataset_enum, implicit_threshold in dataset_specs:
        base_ds = load_and_preprocess(dataset_enum, implicit_threshold)
        for seed in seed_list:
            set_random_state(seed)
            split_ds = UserHoldout(0.2, 0.0).process(base_ds)
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            results_path = os.path.join(working_dir, f'{dataset_label}_seed_{seed}_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    all_results.extend(json.load(f))

    if all_results:
        df = pd.DataFrame(all_results)
        summary = summarize(df)
        print(df.to_string(index=False))
        print('\nSeed-sensitivity summary:')
        print(summary.to_string(index=False))
        print('\nShort analysis: If std across seeds is large relative to the mean gap between algorithms, split choice materially affects conclusions; if std is small, rankings are stable across random holdout realizations.')
    else:
        print('Experiment completed. Check OmniRec checkpoints/logs for results.')
