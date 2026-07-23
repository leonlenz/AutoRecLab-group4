import os
import math
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
    from omnirec.preprocess.split import RandomHoldout
    from omnirec.runner.plan import ExperimentPlan
    from omnirec.runner.evaluation import Evaluator
    from omnirec.util.run import run_omnirec
    from omnirec.util.util import set_random_state
    from omnirec.runner.algos import LensKit
    from omnirec.metrics.ranking import NDCG, Precision

    dataset_map = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': DataSet.Amazon2014VideoGames,
        'HetrecLastFM': DataSet.HetrecLastFM,
    }

    seeds = [11, 22, 33, 44, 55]
    ks = [1, 5, 10]

    def build_pipeline(ds_name):
        steps = []
        if ds_name in ('MovieLens100K', 'Amazon2014VideoGames'):
            steps.append(MakeImplicit(3))
        steps.append(CorePruning(5))
        return Pipe(*steps)

    def build_plan():
        plan = ExperimentPlan(plan_name='seed_sensitivity_baseline')
        plan.add_algorithm(LensKit.ImplicitMFScorer)
        plan.add_algorithm(LensKit.ItemKNNScorer)
        plan.add_algorithm(LensKit.PopScorer)
        return plan

    raw_rows = []
    protocol_rows = []

    for ds_name, ds_id in dataset_map.items():
        raw_ds = RecSysDataSet.use_dataloader(ds_id)
        processed = build_pipeline(ds_name).process(raw_ds)

        protocol_rows.append({
            'dataset': ds_name,
            'n_interactions_after_processing': len(processed._data.df),
            'preprocessing': 'MakeImplicit(3)->CorePruning(5)' if ds_name != 'HetrecLastFM' else 'CorePruning(5)',
            'split': 'RandomHoldout(validation_size=0.0, test_size=0.2) with seed control via set_random_state(seed)',
            'algorithms': 'LensKit.ImplicitMFScorer,LensKit.ItemKNNScorer,LensKit.PopScorer',
            'metrics': 'NDCG@1,5,10; Precision@1,5,10',
        })

        for seed in seeds:
            set_random_state(seed)
            split_ds = RandomHoldout(validation_size=0.0, test_size=0.2).process(processed)
            plan = build_plan()
            evaluator = Evaluator(NDCG(ks), Precision(ks))
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            results_dict = evaluator.get_results()

            if len(results_dict) == 1:
                result_df = next(iter(results_dict.values())).copy()
            else:
                result_df = pd.concat([df.assign(dataset_id=dsid) for dsid, df in results_dict.items()], ignore_index=True)

            result_df['dataset'] = ds_name
            result_df['seed'] = seed
            raw_rows.append(result_df)

    results_df = pd.concat(raw_rows, ignore_index=True)

    summary = (
        results_df
        .groupby(['dataset', 'algorithm', 'name', 'k'], as_index=False)['value']
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )

    analysis_lines = []
    for ds_name in dataset_map.keys():
        ds = results_df[results_df['dataset'] == ds_name]
        for metric in ['NDCG', 'Precision']:
            for k in ks:
                sub = ds[(ds['name'] == metric) & (ds['k'] == k)]
                if sub.empty:
                    continue
                algo_stats = sub.groupby('algorithm')['value'].agg(['mean', 'std']).sort_values('mean', ascending=False)
                best_algo = algo_stats.index[0]
                best_mean = float(algo_stats.iloc[0]['mean'])
                best_std = float(algo_stats.iloc[0]['std']) if not math.isnan(float(algo_stats.iloc[0]['std'])) else 0.0
                worst_mean = float(algo_stats.iloc[-1]['mean'])
                spread = best_mean - worst_mean
                cv = (best_std / best_mean) if best_mean > 0 else float('inf')
                analysis_lines.append(
                    f'{ds_name} | {metric}@{k}: best={best_algo}, mean={best_mean:.4f}, sd={best_std:.4f}, cv={cv:.3f}, spread={spread:.4f}'
                )

    variability_rows = []
    for (dataset, algorithm, name, k), grp in results_df.groupby(['dataset', 'algorithm', 'name', 'k']):
        variability_rows.append({
            'dataset': dataset,
            'algorithm': algorithm,
            'metric': name,
            'k': k,
            'mean': grp['value'].mean(),
            'std': grp['value'].std(),
            'cv': (grp['value'].std() / grp['value'].mean()) if grp['value'].mean() else np.nan,
        })
    variability_df = pd.DataFrame(variability_rows)

    results_path = os.path.join(working_dir, 'seed_sensitivity_raw_results.csv')
    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    protocol_path = os.path.join(working_dir, 'seed_sensitivity_protocol.csv')
    analysis_path = os.path.join(working_dir, 'seed_sensitivity_analysis.txt')

    results_df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(protocol_rows).to_csv(protocol_path, index=False)
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write('Short statistical analysis\n')
        f.write('==========================\n')
        f.write('Across the 5 split seeds, we report mean/std per dataset, algorithm, and metric.\n')
        f.write('Lower std and coefficient of variation indicate lower sensitivity to random split seeds.\n\n')
        for line in analysis_lines:
            f.write(line + '\n')
        f.write('\nPer-group variability (top rows):\n')
        f.write(variability_df.sort_values(['dataset', 'metric', 'k', 'mean'], ascending=[True, True, True, False]).head(20).to_string(index=False))

    print('Preprocessing and split protocol:')
    print(pd.DataFrame(protocol_rows))
    print('\nSummary metrics by dataset/algorithm/metric/k:')
    print(summary)
    print('\nShort statistical analysis:')
    for line in analysis_lines:
        print(line)


if __name__ == '__main__':
    main()