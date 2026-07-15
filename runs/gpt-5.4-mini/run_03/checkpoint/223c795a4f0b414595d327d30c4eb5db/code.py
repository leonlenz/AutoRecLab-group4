import os
import math
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.preprocess.pipe import Pipe
from omnirec.runner.algos import LensKit
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec


def maybe_get_dataframe(split_obj, split_name):
    if hasattr(split_obj, split_name):
        return getattr(split_obj, split_name)
    if isinstance(split_obj, dict) and split_name in split_obj:
        return split_obj[split_name]
    return None


def summarize(values):
    values = [float(v) for v in values]
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.pstdev(values))}


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    print(f'Working directory: {working_dir}')

    datasets = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': DataSet.Amazon2014VideoGames,
        'HetrecLastFM': DataSet.HetrecLastFM,
    }
    algorithms = [LensKit.PopScorer, LensKit.ItemKNNScorer, LensKit.ImplicitMFScorer]
    seeds = [11, 22, 33, 44, 55]

    all_rows = []

    for ds_name, ds_enum in datasets.items():
        print(f'\nLoading dataset: {ds_name}')
        dataset = RecSysDataSet.use_dataloader(ds_enum)

        steps = [CorePruning(5)]
        if ds_name in ('MovieLens100K', 'Amazon2014VideoGames'):
            steps.append(MakeImplicit(3))
        pipeline = Pipe(*steps)
        dataset = pipeline.process(dataset)

        for seed in seeds:
            split = UserHoldout(validation_size=0.0, test_size=0.2)
            split_dataset = split.process(dataset)

            evaluator = Evaluator(
                NDCG([1]), NDCG([5]), NDCG([10]),
                Precision([1]), Precision([5]), Precision([10])
            )

            plan = ExperimentPlan(f'{ds_name}-seed-{seed}')
            for algo in algorithms:
                plan.add_algorithm(algo, {})

            result = run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)

            if isinstance(result, pd.DataFrame):
                df = result.copy()
            elif hasattr(result, 'to_dataframe'):
                df = result.to_dataframe()
            else:
                df = pd.DataFrame(result)

            df['dataset'] = ds_name
            df['seed'] = seed
            all_rows.append(df)
            print(f'Completed {ds_name} seed={seed}')

    results = pd.concat(all_rows, ignore_index=True)
    print('\nRaw result preview:')
    print(results.head())

    metric_cols = [c for c in results.columns if any(m in c.lower() for m in ['ndcg', 'precision'])]
    group_cols = ['dataset', 'algorithm'] if 'algorithm' in results.columns else ['dataset']
    summary = results.groupby(group_cols)[metric_cols].agg(['mean', 'std'])
    print('\nAggregate results (mean/std across seeds):')
    print(summary)

    print('\nShort statistical analysis:')
    if metric_cols:
        for col in metric_cols:
            vals = results[col].dropna().tolist()
            s = summarize(vals)
            print(f'- {col}: mean={s["mean"]:.4f}, std={s["std"]:.4f}')
        print('Seed sensitivity is summarized by the across-seed standard deviation for each metric; larger std indicates greater sensitivity to the random holdout split.')
    else:
        print('No metric columns were detected in the returned evaluation table; please verify OmniRec result schema.')


if __name__ == '__main__':
    main()
