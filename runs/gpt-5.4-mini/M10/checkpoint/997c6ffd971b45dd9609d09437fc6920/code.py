import os
from pathlib import Path
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
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


WORKING_DIR = os.path.join(os.getcwd(), 'working')
os.makedirs(WORKING_DIR, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
DATASET_SPECS = [
    ('MovieLens100K', DataSet.MovieLens100K, 3),
    ('Amazon Video Games', DataSet.Amazon2014VideoGames, 3),
    ('Last.FM', DataSet.HetrecLastFM, None),
]


def load_and_preprocess(dataset_id, make_implicit_threshold=None):
    ds = RecSysDataSet.use_dataloader(dataset_id)
    steps = []
    if make_implicit_threshold is not None:
        steps.append(MakeImplicit(make_implicit_threshold))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def build_plan():
    plan = ExperimentPlan('seed_sensitivity_lenskit_baselines')
    plan.add_algorithm(LensKit.ImplicitMFScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def build_split_dataset(dataset, seed):
    set_random_state(seed)
    # OmniRec's documented UserHoldout requires validation_size and test_size.
    # We keep validation tiny and use a 20% test split for the requested user-based holdout.
    splitter = UserHoldout(validation_size=0.01, test_size=0.20)
    return splitter.process(dataset)


def collect_results_df(evaluator, dataset_name, seed):
    rows = []
    for _, df in evaluator.get_results().items():
        tmp = df.copy()
        tmp['dataset'] = dataset_name
        tmp['seed'] = seed
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=pd.Index(['algorithm', 'fold', 'name', 'k', 'value', 'dataset', 'seed']))
    return pd.concat(rows, ignore_index=True)


def summarize_seed_variation(raw_df):
    rows = []
    grouped = raw_df.groupby(['dataset', 'algorithm', 'name', 'k'], dropna=False)
    for (dataset, algorithm, metric, k), grp in grouped:
        vals = grp['value'].astype(float).to_numpy()
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else np.nan
        rows.append({
            'dataset': dataset,
            'algorithm': algorithm,
            'metric': f'{metric}@{int(k)}' if pd.notna(k) else metric,
            'mean': mean,
            'std': std,
            'cv': cv,
            'n': int(len(vals)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['dataset', 'algorithm', 'metric']).reset_index(drop=True)
    return out


def short_statistical_analysis(summary_df):
    lines = []
    for (dataset, metric), grp in summary_df.groupby(['dataset', 'metric']):
        best = grp.sort_values(['cv', 'std', 'mean'], ascending=[True, True, False]).iloc[0]
        lines.append(
            f'{dataset} / {metric}: most stable was {best["algorithm"]} '
            f'(mean={best["mean"]:.4f}, std={best["std"]:.4f}, cv={best["cv"]:.4f}, n={int(best["n"])})'
        )
    return '\n'.join(lines)


def main():
    plan = build_plan()
    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    all_raw = []

    for dataset_name, dataset_id, threshold in DATASET_SPECS:
        print(f'\n=== Preparing {dataset_name} ===')
        base_dataset = load_and_preprocess(dataset_id, make_implicit_threshold=threshold)

        for seed in SEEDS:
            print(f'--- Running {dataset_name}, seed={seed} ---')
            split_dataset = build_split_dataset(base_dataset, seed)
            run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
            run_df = collect_results_df(evaluator, dataset_name, seed)
            all_raw.append(run_df)
            print(f'Finished {dataset_name} seed={seed} with {len(run_df)} metric rows')

    raw_results_df = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
    summary_df = summarize_seed_variation(raw_results_df) if not raw_results_df.empty else pd.DataFrame()

    raw_path = os.path.join(WORKING_DIR, 'seed_sensitivity_raw_results.csv')
    summary_path = os.path.join(WORKING_DIR, 'seed_sensitivity_summary.csv')
    raw_results_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print('\nSeed-variation summary:')
    if summary_df.empty:
        print('No results collected.')
    else:
        print(summary_df.to_string(index=False))

    print('\nShort statistical analysis:')
    if summary_df.empty:
        print('No statistical analysis available.')
    else:
        print(short_statistical_analysis(summary_df))

    print(f'\nRaw results saved to: {raw_path}')
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
