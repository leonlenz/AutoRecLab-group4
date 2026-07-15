import os
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.util import set_random_state
from omnirec.util.run import run_omnirec


def build_dataset(name: str) -> RecSysDataSet:
    if name == 'MovieLens100K':
        return RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    if name == 'Amazon2014VideoGames':
        return RecSysDataSet.use_dataloader(DataSet.Amazon2014VideoGames)
    if name == 'HetrecLastFM':
        return RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
    raise ValueError(f'Unknown dataset: {name}')


def preprocess_dataset(dataset_name: str, raw_ds: RecSysDataSet) -> RecSysDataSet:
    steps = []
    if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    steps.append(UserHoldout(validation_size=0.05, test_size=0.2))
    return Pipe(*steps).process(raw_ds)


def summarize_seed_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.startswith('NDCG@') or c.startswith('Precision@')]
    rows = []
    grouped = df.melt(
        id_vars=['dataset', 'algorithm', 'seed'],
        value_vars=metric_cols,
        var_name='metric',
        value_name='value'
    ).groupby(['dataset', 'algorithm', 'metric'])
    for (dataset, algorithm, metric), g in grouped:
        vals = pd.to_numeric(g['value'], errors='coerce').dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append({
            'dataset': dataset,
            'algorithm': algorithm,
            'metric': metric,
            'mean': mean,
            'std': std,
            'cv': float(std / mean) if mean != 0 else np.nan,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
        })
    return pd.DataFrame(rows).sort_values(['dataset', 'algorithm', 'metric']).reset_index(drop=True)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']

    all_results = []

    for dataset_name in dataset_names:
        print(f'\n=== Loading {dataset_name} ===')
        raw_ds = build_dataset(dataset_name)

        for seed in seeds:
            print(f'\n--- {dataset_name} | seed={seed} ---')
            set_random_state(seed)
            split_ds = preprocess_dataset(dataset_name, raw_ds)

            plan = ExperimentPlan(plan_name=f'{dataset_name}_seed_{seed}')
            plan.add_algorithm(LensKit.ImplicitMFScorer)
            plan.add_algorithm(LensKit.ItemKNNScorer)
            plan.add_algorithm(LensKit.PopScorer)

            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10])
            )

            result = run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            if hasattr(result, 'to_df'):
                res_df = result.to_df().copy()
            elif isinstance(result, pd.DataFrame):
                res_df = result.copy()
            else:
                res_df = pd.DataFrame(result)

            res_df['dataset'] = dataset_name
            res_df['seed'] = seed
            all_results.append(res_df)
            print(res_df)

    results = pd.concat(all_results, ignore_index=True)
    results_path = os.path.join(working_dir, 'seed_sensitivity_results.csv')
    results.to_csv(results_path, index=False)

    summary = summarize_seed_sensitivity(results)
    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    summary.to_csv(summary_path, index=False)

    print('\n=== Seed sensitivity summary ===')
    print(summary)

    print('\n=== Brief statistical analysis ===')
    for (dataset, algorithm), g in summary.groupby(['dataset', 'algorithm']):
        max_cv = g['cv'].replace([np.inf, -np.inf], np.nan).max()
        avg_std = g['std'].mean()
        print(f'- {dataset} / {algorithm}: avg std={avg_std:.6f}, max CV={max_cv:.6f}')

    print(f'\nSaved results to: {results_path}')
    print(f'Saved summary to: {summary_path}')


if __name__ == '__main__':
    main()
