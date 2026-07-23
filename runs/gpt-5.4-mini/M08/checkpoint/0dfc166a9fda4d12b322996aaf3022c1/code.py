import os
import json
import math
import statistics as stats
from collections import defaultdict

import numpy as np
import pandas as pd


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    from omnirec import RecSysDataSet
    from omnirec.data_loaders.datasets import DataSet
    from omnirec.preprocess.pipe import Pipe
    from omnirec.preprocess.feedback_conversion import MakeImplicit
    from omnirec.preprocess.core_pruning import CorePruning
    from omnirec.preprocess.split import UserHoldout
    from omnirec.runner.plan import ExperimentPlan
    from omnirec.runner.evaluation import Evaluator
    from omnirec.util.run import run_omnirec
    from omnirec.util.util import set_random_state
    from omnirec.runner.algos import LensKit
    from omnirec.metrics.ranking import NDCG, Recall

    dataset_map = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': DataSet.Amazon2014VideoGames,
        'HetrecLastFM': DataSet.HetrecLastFM,
    }

    seeds = [11, 22, 33, 44, 55]
    ks = [1, 5, 10]

    def build_pipeline(ds_name):
        if ds_name in ('MovieLens100K', 'Amazon2014VideoGames'):
            return Pipe(
                MakeImplicit(3),
                CorePruning(5),
            )
        return Pipe(
            CorePruning(5),
        )

    def build_plan():
        plan = ExperimentPlan(plan_name='seed_sensitivity_baseline')
        plan.add_algorithm(LensKit.ImplicitMFScorer)
        plan.add_algorithm(LensKit.ItemKNNScorer)
        plan.add_algorithm(LensKit.PopScorer)
        return plan

    evaluator = Evaluator(
        NDCG(ks),
        Recall(ks),
    )

    raw_rows = []
    summary_rows = []
    protocol_rows = []
    seed_result_store = defaultdict(list)

    for ds_name, ds_id in dataset_map.items():
        raw_ds = RecSysDataSet.use_dataloader(ds_id)
        processed = build_pipeline(ds_name).process(raw_ds)

        protocol_rows.append({
            'dataset': ds_name,
            'n_interactions_after_processing': len(processed._data.df),
            'preprocessing': 'MakeImplicit(3)->CorePruning(5)' if ds_name != 'HetrecLastFM' else 'CorePruning(5)',
            'split': 'UserHoldout(validation_size=0.2, test_size=0.2)',
            'algorithms': 'LensKit.ImplicitMFScorer,LensKit.ItemKNNScorer,LensKit.PopScorer',
            'metrics': 'NDCG@1,5,10; Recall@1,5,10',
        })

        for seed in seeds:
            set_random_state(seed)
            split_ds = UserHoldout(validation_size=0.2, test_size=0.2).process(processed)
            plan = build_plan()
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            results_dict = evaluator.get_results()

            if len(results_dict) == 1:
                result_df = next(iter(results_dict.values())).copy()
            else:
                result_df = pd.concat(
                    [df.assign(dataset_id=dsid) for dsid, df in results_dict.items()],
                    ignore_index=True,
                )

            result_df['dataset'] = ds_name
            result_df['seed'] = seed
            raw_rows.append(result_df)
            seed_result_store[(ds_name, seed)].append(result_df)

    results_df = pd.concat(raw_rows, ignore_index=True)
    results_df = results_df.rename(columns={
        'algorithm': 'algorithm',
        'name': 'metric',
        'k': 'k',
        'value': 'value',
    })

    summary = (
        results_df
        .groupby(['dataset', 'algorithm', 'metric', 'k'], as_index=False)['value']
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )
    summary.columns = ['dataset', 'algorithm', 'metric', 'k', 'mean', 'std', 'min', 'max']

    analysis_lines = []
    for ds_name in dataset_map.keys():
        ds = results_df[results_df['dataset'] == ds_name]
        for metric in ['NDCG', 'Recall']:
            for k in ks:
                sub = ds[(ds['metric'] == metric) & (ds['k'] == k)]
                algo_means = sub.groupby('algorithm')['value'].mean().sort_values(ascending=False)
                algo_stds = sub.groupby('algorithm')['value'].std().fillna(0.0)
                best = algo_means.index[0]
                spread = float(algo_means.iloc[0] - algo_means.iloc[-1]) if len(algo_means) > 1 else 0.0
                analysis_lines.append(
                    f'{ds_name} | {metric}@{k}: best={best}, mean={algo_means.iloc[0]:.4f}, '
                    f'sd={float(algo_stds.loc[best]):.4f}, spread={spread:.4f}'
                )

    results_path = os.path.join(working_dir, 'seed_sensitivity_raw_results.csv')
    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    protocol_path = os.path.join(working_dir, 'seed_sensitivity_protocol.csv')
    analysis_path = os.path.join(working_dir, 'seed_sensitivity_analysis.txt')

    results_df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(protocol_rows).to_csv(protocol_path, index=False)
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(analysis_lines))

    print('Preprocessing and split protocol:')
    print(pd.DataFrame(protocol_rows))
    print('\nSummary metrics by dataset/algorithm/metric/k:')
    print(summary)
    print('\nShort statistical analysis:')
    for line in analysis_lines:
        print(line)


if __name__ == '__main__':
    main()
