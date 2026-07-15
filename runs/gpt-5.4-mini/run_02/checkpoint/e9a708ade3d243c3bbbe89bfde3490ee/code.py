import os
import json
import math
import statistics as stats
from pathlib import Path
from collections import defaultdict

import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def build_dataset(dataset_enum, implicit_threshold=None, seed=42):
    set_random_state(seed)
    ds = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.append(CorePruning(5))
    # UserHoldout expects validation_size and test_size, not train/test fractions.
    # Use a valid user-based split. Here we create 70/10/20 train/valid/test.
    steps.append(UserHoldout(validation_size=0.1, test_size=0.2))
    pipe = Pipe(*steps)
    return pipe.process(ds)


def make_plan():
    plan = ExperimentPlan(plan_name="SeedSensitivity_LensKit_Baselines")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def extract_result_frames(results_obj):
    if isinstance(results_obj, dict):
        frames = []
        for key, value in results_obj.items():
            if isinstance(value, pd.DataFrame):
                df = value.copy()
                if 'dataset_id' not in df.columns:
                    df['dataset_id'] = key
                frames.append(df)
        return frames
    if isinstance(results_obj, pd.DataFrame):
        return [results_obj.copy()]
    return []


def run_experiment():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    datasets = {
        'MovieLens100K': (DataSet.MovieLens100K, 3),
        'Amazon2014VideoGames': (DataSet.Amazon2014VideoGames, 3),
        'HetrecLastFM': (DataSet.HetrecLastFM, None),
    }
    seeds = [11, 22, 33, 44, 55]

    all_run_records = []
    metadata = []

    for ds_name, (ds_enum, thr) in datasets.items():
        print(f"\n=== Dataset: {ds_name} ===")
        for seed in seeds:
            print(f"-- Seed {seed}")
            dataset = build_dataset(ds_enum, implicit_threshold=thr, seed=seed)
            plan = make_plan()
            evaluator = Evaluator(
                NDCG([1, 5, 10]),
                Precision([1, 5, 10]),
            )
            run_omnirec(dataset, plan, evaluator)
            results = evaluator.get_results()
            frames = extract_result_frames(results)
            for df in frames:
                df = df.copy()
                df['dataset'] = ds_name
                df['seed'] = seed
                all_run_records.append(df)
            metadata.append({
                'dataset': ds_name,
                'seed': seed,
                'preprocessing': {
                    'implicit_threshold': thr,
                    'core_filter': 5,
                    'split': {'type': 'UserHoldout', 'validation_size': 0.1, 'test_size': 0.2},
                },
                'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
                'metrics': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Precision@1', 'Precision@5', 'Precision@10'],
            })
            print(results)

    out_dir = Path(working_dir)
    meta_path = out_dir / 'experiment_metadata.json'
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    if all_run_records:
        full = pd.concat(all_run_records, ignore_index=True)
        full.to_csv(out_dir / 'raw_results.csv', index=False)

        # normalize common result column names across possible evaluator outputs
        rename_map = {}
        if 'metric' in full.columns and 'name' not in full.columns:
            rename_map['metric'] = 'name'
        if 'algorithm_name' in full.columns and 'algorithm' not in full.columns:
            rename_map['algorithm_name'] = 'algorithm'
        if rename_map:
            full = full.rename(columns=rename_map)

        required_cols = [c for c in ['dataset', 'algorithm', 'name', 'k', 'value'] if c in full.columns]
        if len(required_cols) == 5:
            summary = full.groupby(['dataset', 'algorithm', 'name', 'k'], as_index=False)['value'].agg(['mean', 'std', 'count'])
            summary.to_csv(out_dir / 'summary_by_dataset_algorithm_metric.csv')
            print('\n=== Aggregated Summary ===')
            print(summary)

            print('\n=== Short Statistical Analysis ===')
            for (dataset, alg, metric, k), grp in full.groupby(['dataset', 'algorithm', 'name', 'k']):
                vals = grp['value'].tolist()
                mean_v = stats.mean(vals)
                std_v = stats.pstdev(vals) if len(vals) > 1 else 0.0
                cv = std_v / mean_v if mean_v != 0 else float('inf')
                print(f'{dataset} | {alg} | {metric}@{k}: mean={mean_v:.4f}, std={std_v:.4f}, cv={cv:.3f}')

            ndcg10 = full[(full['name'] == 'NDCG') & (full['k'] == 10)]
            if not ndcg10.empty:
                pairwise = ndcg10.groupby(['dataset', 'algorithm'])['value'].mean().unstack()
                print('\n=== Pairwise Comparison on mean NDCG@10 ===')
                print(pairwise)
        else:
            print('\nRaw results columns were not in the expected format; saved raw_results.csv for inspection.')


if __name__ == '__main__':
    run_experiment()
