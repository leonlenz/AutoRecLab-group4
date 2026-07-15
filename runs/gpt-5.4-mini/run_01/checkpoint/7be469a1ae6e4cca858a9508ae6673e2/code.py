import os
from collections import defaultdict
from statistics import mean, pstdev

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


def load_and_preprocess(dataset_enum, implicit=False):
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if implicit:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    pipe = Pipe(*steps)
    return pipe.process(ds)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    seeds = [11, 22, 33, 44, 55]
    datasets = [
        (DataSet.MovieLens100K, True, 'MovieLens100K'),
        (DataSet.Amazon2014VideoGames, True, 'Amazon2014VideoGames'),
        (DataSet.HetrecLastFM, False, 'HetrecLastFM'),
    ]

    plan = ExperimentPlan('seed_sensitivity_lenskit')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    all_rows = []

    for ds_enum, make_implicit, ds_name in datasets:
        base_ds = load_and_preprocess(ds_enum, implicit=make_implicit)
        for seed in seeds:
            set_random_state(seed)
            split_ds = UserHoldout(validation_size=0.0, test_size=0.2).process(base_ds)
            evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            results = evaluator.get_results()
            for dataset_key, df in results.items():
                for algo, row in df.iterrows():
                    rec = {'dataset': ds_name, 'seed': seed, 'algorithm': algo}
                    rec.update(row.to_dict())
                    all_rows.append(rec)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv('seed_sensitivity_results.csv', index=False)
    print('\nPer-seed results:')
    print(results_df)

    metric_cols = [c for c in results_df.columns if c not in {'dataset', 'seed', 'algorithm'}]
    summary = (
        results_df.groupby(['dataset', 'algorithm'])[metric_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )
    summary.to_csv('seed_sensitivity_summary.csv', index=False)
    print('\nSummary across seeds:')
    print(summary)

    print('\nBrief statistical note:')
    for _, grp in results_df.groupby(['dataset', 'algorithm']):
        dataset = grp['dataset'].iloc[0]
        algo = grp['algorithm'].iloc[0]
        for metric in metric_cols:
            vals = grp[metric].dropna().tolist()
            if len(vals) > 1:
                print(f'{dataset} / {algo} / {metric}: mean={mean(vals):.4f}, std={pstdev(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}')


if __name__ == '__main__':
    main()
