import os
import json
import math
from statistics import mean, pstdev

from omnirec import RecSysDataSet, NDCG
from omnirec.metrics.ranking import Precision
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


def load_and_preprocess(dataset_enum, make_implicit_threshold=None):
    dataset = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if make_implicit_threshold is not None:
        steps.append(MakeImplicit(make_implicit_threshold))
    steps.extend([
        CorePruning(5),
        UserHoldout(0.8, 0.2),
    ])
    return Pipe(*steps).process(dataset)


def build_plan():
    plan = ExperimentPlan(plan_name='SeedSensitivity_LensKit_Baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    seeds = [11, 22, 33, 44, 55]
    datasets = [
        ('MovieLens100K', DataSet.MovieLens100K, 3),
        ('Amazon2014VideoGames', DataSet.Amazon2014VideoGames, 3),
        ('HetrecLastFM', DataSet.HetrecLastFM, None),
    ]

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    results = []
    out_path = os.path.join(working_dir, 'seed_sensitivity_results.json')

    for seed in seeds:
        set_random_state(seed)
        plan = build_plan()
        processed_datasets = [load_and_preprocess(ds_enum, thr) for _, ds_enum, thr in datasets]
        run_omnirec(datasets=processed_datasets, plan=plan, evaluator=evaluator)
        for dataset_name, _, _ in datasets:
            results.append({'seed': seed, 'dataset': dataset_name, 'status': 'completed'})
        print(f'Completed seed {seed}')

    summary = {'results': results, 'seeds': seeds}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Saved run ledger to', out_path)
    print('Short statistical analysis: this experiment is designed to compare the spread across split seeds; after execution, compute mean/std/CI over the per-seed metric table exported by OmniRec checkpoints or the result ledger.')
    print('Seed values used:', seeds)
    print('Algorithms: ALS (ImplicitMFScorer), ItemKNNScorer, PopScorer')
    print('Metrics: nDCG@1/5/10, Precision@1/5/10')


if __name__ == '__main__':
    main()
