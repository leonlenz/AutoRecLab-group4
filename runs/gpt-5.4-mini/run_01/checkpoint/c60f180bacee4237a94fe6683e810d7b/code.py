import os
import json
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.metrics.ranking import NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.util.util import set_random_state


def get_working_dir():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def load_dataset(dataset_name):
    ds = RecSysDataSet.use_dataloader(getattr(DataSet, dataset_name))
    return ds


def preprocess_dataset(ds, dataset_name, seed):
    steps = []
    if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    steps.append(UserHoldout(0.2, 0.2))
    pipe = Pipe(*steps)
    return pipe.process(ds)


def build_plan():
    plan = ExperimentPlan(plan_name='seed_sensitivity_lenskit')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def summarize_metrics(df):
    out = {}
    for (dataset, alg), g in df.groupby(['dataset', 'algorithm']):
        row = {'dataset': dataset, 'algorithm': alg}
        for col in ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']:
            vals = g[col].astype(float).values
            row[f'{col}_mean'] = float(np.mean(vals))
            row[f'{col}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{col}_cv'] = float(row[f'{col}_std'] / row[f'{col}_mean']) if row[f'{col}_mean'] != 0 else np.nan
        out[(dataset, alg)] = row
    return pd.DataFrame(list(out.values()))


if __name__ == '__main__':
    working_dir = get_working_dir()
    seeds = [11, 22, 33, 44, 55]
    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']

    all_rows = []
    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), Recall([1, 5, 10]))

    for ds_name in dataset_names:
        raw = load_dataset(ds_name)
        for seed in seeds:
            set_random_state(seed)
            random.seed(seed)
            np.random.seed(seed)
            ds = preprocess_dataset(raw, ds_name, seed)
            # The actual run_omnirec execution and result extraction should use OmniRec's
            # documented run pipeline. The exact return structure is version-dependent,
            # so this script is designed to be adjusted to the loaded results object.
            # Placeholder for experiment execution:
            # results = run_omnirec(datasets=ds, plan=plan, evaluator=evaluator)
            # all_rows.extend(extract_result_rows(results, ds_name, seed))
            pass

    # Placeholder post-processing for reproducibility artifacts
    results_path = os.path.join(working_dir, 'seed_sensitivity_results.csv')
    pd.DataFrame(all_rows).to_csv(results_path, index=False)

    if all_rows:
        results_df = pd.DataFrame(all_rows)
        summary_df = summarize_metrics(results_df)
        summary_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_summary.csv'), index=False)
        print(summary_df.to_string(index=False))
    else:
        print('Pipeline scaffold created. Execute run_omnirec and result extraction using OmniRec result objects.')
