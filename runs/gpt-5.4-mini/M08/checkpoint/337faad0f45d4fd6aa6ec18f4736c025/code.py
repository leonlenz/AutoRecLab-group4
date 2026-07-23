import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

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

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    enum: DataSet
    make_implicit: bool


DATASETS = [
    DatasetConfig('MovieLens100K', DataSet.MovieLens100K, True),
    DatasetConfig('Amazon2014VideoGames', DataSet.Amazon2014VideoGames, True),
    DatasetConfig('HetrecLastFM', DataSet.HetrecLastFM, False),
]
SEEDS = [11, 22, 33, 44, 55]
KS = [1, 5, 10]


def preprocess_dataset(dataset, make_implicit: bool):
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(dataset)


def build_plan():
    plan = ExperimentPlan(plan_name='seed_sensitivity_lenskit')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def collect_results_from_evaluator(evaluator, dataset_name, seed):
    results_obj = evaluator.get_results()
    if isinstance(results_obj, dict):
        frames = []
        for key, df in results_obj.items():
            tmp = df.copy()
            tmp['result_key'] = key
            frames.append(tmp)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        out = results_obj.copy()
    if len(out) > 0:
        out['dataset'] = dataset_name
        out['seed'] = seed
    return out


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(['dataset', 'algorithm', 'name', 'k'], as_index=False)['value']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    summary.columns = ['dataset', 'algorithm', 'name', 'k', 'mean', 'std', 'count']
    summary['std'] = summary['std'].fillna(0.0)
    summary['cv'] = np.where(summary['mean'] != 0, summary['std'] / summary['mean'], np.nan)
    return summary


def main():
    all_results = []

    for ds_cfg in DATASETS:
        for seed in SEEDS:
            set_random_state(seed)
            dataset = RecSysDataSet.use_dataloader(ds_cfg.enum)
            dataset = preprocess_dataset(dataset, make_implicit=ds_cfg.make_implicit)

            # User-based holdout; validation must be positive in this OmniRec implementation.
            dataset = UserHoldout(validation_size=0.05, test_size=0.20).process(dataset)

            plan = build_plan()
            evaluator = Evaluator(
                NDCG(KS),
                Precision(KS),
            )

            print(f'Running dataset={ds_cfg.name}, seed={seed}')
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
            res = collect_results_from_evaluator(evaluator, ds_cfg.name, seed)
            if len(res) > 0:
                all_results.append(res)
                print(res[['algorithm', 'name', 'k', 'value']].to_string(index=False))

    if not all_results:
        print('No results collected.')
        return

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize(results)
    summary = summary.sort_values(['dataset', 'algorithm', 'name', 'k'])

    print('\n=== Aggregated results across seeds ===')
    print(summary.to_string(index=False))

    seed_sensitivity = (
        summary.groupby(['dataset', 'algorithm'], as_index=False)['cv']
        .mean()
        .sort_values(['dataset', 'cv'], ascending=[True, False])
    )
    print('\n=== Seed-sensitivity summary (mean CV across metrics) ===')
    print(seed_sensitivity.to_string(index=False))

    best_by_dataset = (
        summary.groupby(['dataset', 'algorithm'], as_index=False)['mean']
        .mean()
        .sort_values(['dataset', 'mean'], ascending=[True, False])
    )
    print('\n=== Mean performance by dataset/algorithm ===')
    print(best_by_dataset.to_string(index=False))

    out_path = os.path.join(working_dir, 'seed_sensitivity_results.csv')
    summary.to_csv(out_path, index=False)
    print(f'Saved summary to {out_path}')


if __name__ == '__main__':
    main()
