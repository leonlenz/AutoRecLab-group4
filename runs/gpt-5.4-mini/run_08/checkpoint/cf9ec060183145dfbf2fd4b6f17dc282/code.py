import os
import json
import math
import statistics as stats
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    # NOTE: This script is written to use OmniRec exclusively for the experiment workflow.
    # Exact class/function names are verified from documentation; if your local OmniRec
    # build exposes slightly different import paths, adjust only the import lines below.
    from omnirec import RecSysDataSet
    from omnirec.metrics.ranking import HR, NDCG
    from omnirec.data_loaders.datasets import DataSet
    from omnirec.preprocess.core_pruning import CorePruning
    from omnirec.preprocess.feedback_conversion import MakeImplicit
    from omnirec.preprocess.pipe import Pipe
    from omnirec.preprocess.split import UserHoldout
    from omnirec.runner.plan import ExperimentPlan
    from omnirec.runner.evaluation import Evaluator
    from omnirec.util.run import run_omnirec
    from omnirec.runner.algos import LensKit

    dataset_map = {
        'MovieLens100K': DataSet.MovieLens100K,
        'AmazonVideoGames': DataSet.Amazon2014VideoGames,
        'LastFM': DataSet.HetrecLastFM,
    }

    seeds = [11, 22, 33, 44, 55]
    k_values = [1, 5, 10]

    # Preprocessing pipelines per dataset
    pipelines = {
        'MovieLens100K': Pipe(
            MakeImplicit(3),
            CorePruning(5),
        ),
        'AmazonVideoGames': Pipe(
            MakeImplicit(3),
            CorePruning(5),
        ),
        'LastFM': Pipe(
            CorePruning(5),
        ),
    }

    # Evaluation metrics
    metrics = []
    for k in k_values:
        metrics.append(NDCG([k]))
        metrics.append(HR([k]))
    evaluator = Evaluator(*metrics)

    # Build a LensKit-backed OmniRec plan with default hyperparameters.
    # The docs confirm these model IDs exist; no tuning is performed.
    plan = ExperimentPlan(plan_name='seed_sensitivity_baseline')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    all_rows = []
    protocol_rows = []

    for ds_name, ds_id in dataset_map.items():
        raw_ds = RecSysDataSet.use_dataloader(ds_id)
        processed = pipelines[ds_name].process(raw_ds)

        for seed in seeds:
            # User-based 80/20 holdout.
            split = UserHoldout(0.2, 0.2)
            split_ds = split.process(processed)

            # Run the configured baseline algorithms on this split.
            # The run_omnirec call should return measured results compatible with pandas.
            results = run_omnirec(
                datasets=split_ds,
                plan=plan,
                evaluator=evaluator,
            )

            if isinstance(results, pd.DataFrame):
                res_df = results.copy()
            else:
                res_df = pd.DataFrame(results)

            res_df['dataset'] = ds_name
            res_df['seed'] = seed
            all_rows.append(res_df)

            protocol_rows.append({
                'dataset': ds_name,
                'seed': seed,
                'n_interactions_after_processing': processed.num_interactions(),
            })

    results_df = pd.concat(all_rows, ignore_index=True)
    protocol_df = pd.DataFrame(protocol_rows)

    # Expected result table columns from OmniRec/LensKit runs may vary slightly;
    # normalize common metric/model column naming conventions.
    col_candidates = {
        'model': ['model', 'algorithm', 'algo'],
        'metric': ['metric', 'measure'],
        'value': ['value', 'score', 'metric_value'],
    }

    def pick_col(df, options):
        for c in options:
            if c in df.columns:
                return c
        raise KeyError(f'Missing one of {options} in results columns: {list(df.columns)}')

    model_col = pick_col(results_df, col_candidates['model'])
    metric_col = pick_col(results_df, col_candidates['metric'])
    value_col = pick_col(results_df, col_candidates['value'])

    summary = (
        results_df
        .groupby(['dataset', model_col, metric_col], as_index=False)[value_col]
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )

    # Short statistical analysis: per-dataset rank by mean metric and variability by seed.
    analysis_lines = []
    for ds_name in dataset_map.keys():
        ds_sub = results_df[results_df['dataset'] == ds_name]
        for metric_name in sorted(ds_sub[metric_col].unique()):
            msub = ds_sub[ds_sub[metric_col] == metric_name]
            algo_means = msub.groupby(model_col)[value_col].mean().sort_values(ascending=False)
            algo_stds = msub.groupby(model_col)[value_col].std().sort_values(ascending=True)
            best_algo = algo_means.index[0]
            best_mean = float(algo_means.iloc[0])
            best_sd = float(msub[msub[model_col] == best_algo][value_col].std())
            analysis_lines.append(
                f'{ds_name} | {metric_name}: best={best_algo} mean={best_mean:.4f} sd={best_sd:.4f}; '
                f'between-algorithm spread={(float(algo_means.iloc[0]) - float(algo_means.iloc[-1])):.4f}'
            )

    # Save outputs
    results_path = os.path.join(working_dir, 'seed_sensitivity_raw_results.csv')
    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    protocol_path = os.path.join(working_dir, 'seed_sensitivity_protocol.csv')
    with open(os.path.join(working_dir, 'seed_sensitivity_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(analysis_lines))

    results_df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    protocol_df.to_csv(protocol_path, index=False)

    print('Preprocessing and split protocol:')
    print(protocol_df.groupby('dataset')['seed'].count())
    print('\nSummary metrics by dataset/algorithm/metric:')
    print(summary)
    print('\nShort statistical analysis:')
    for line in analysis_lines:
        print(line)


if __name__ == '__main__':
    main()
