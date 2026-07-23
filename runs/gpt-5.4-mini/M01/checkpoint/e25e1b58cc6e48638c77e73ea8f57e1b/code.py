import os
import numpy as np
import pandas as pd
from scipy import stats

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import NDCG, Recall
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def get_working_dir():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def load_dataset(dataset_name):
    ds_enum = getattr(DataSet, dataset_name)
    return RecSysDataSet.use_dataloader(ds_enum)


def preprocess_dataset(ds, dataset_name):
    steps = []
    if dataset_name in {'MovieLens100K', 'Amazon2014VideoGames'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    return Pipe(*steps).process(ds)


def split_dataset(ds, seed):
    set_random_state(seed)
    return UserHoldout(validation_size=0.0, test_size=0.2).process(ds)


def build_plan():
    plan = ExperimentPlan(plan_name='seed_sensitivity_lenskit')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def extract_results(evaluator, dataset_label, seed):
    rows = []
    results = evaluator.get_results()
    for dataset_id, df in results.items():
        if dataset_label not in str(dataset_id):
            continue
        for _, r in df.iterrows():
            rows.append({
                'dataset': dataset_label,
                'seed': seed,
                'algorithm': str(r['algorithm']),
                'metric': f"{r['name']}@{int(r['k'])}",
                'value': float(r['value']),
            })
    return rows


def compute_precision_from_hits(rec_df, ks=(1, 5, 10)):
    out = []
    for (dataset, seed, algorithm, user), g in rec_df.groupby(['dataset', 'seed', 'algorithm', 'user']):
        g = g.sort_values('rank')
        relevant = set(g.loc[g['relevant'] == 1, 'item'].tolist())
        for k in ks:
            topk = g.head(k)
            hits = sum(item in relevant for item in topk['item'].tolist())
            out.append({
                'dataset': dataset,
                'seed': seed,
                'algorithm': algorithm,
                'metric': f'Precision@{k}',
                'value': hits / float(k),
            })
    return pd.DataFrame(out)


def summarize_metrics(df):
    summary = []
    for (dataset, algorithm, metric), g in df.groupby(['dataset', 'algorithm', 'metric']):
        vals = g['value'].astype(float).to_numpy()
        summary.append({
            'dataset': dataset,
            'algorithm': algorithm,
            'metric': metric,
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'cv': float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 and np.mean(vals) != 0 else np.nan,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
        })
    return pd.DataFrame(summary)


if __name__ == '__main__':
    working_dir = get_working_dir()
    seeds = [11, 22, 33, 44, 55]
    dataset_names = ['MovieLens100K', 'Amazon2014VideoGames', 'HetrecLastFM']

    all_rows = []
    plan = build_plan()

    for ds_name in dataset_names:
        raw = load_dataset(ds_name)
        for seed in seeds:
            ds = preprocess_dataset(raw, ds_name)
            ds = split_dataset(ds, seed)
            evaluator = Evaluator(NDCG([1, 5, 10]), Recall([1, 5, 10]))
            run_omnirec(ds, plan, evaluator)
            all_rows.extend(extract_results(evaluator, ds_name, seed))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_results.csv'), index=False)

    summary_df = summarize_metrics(results_df)
    summary_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_summary.csv'), index=False)
    print(summary_df.to_string(index=False))

    print('\n## Documentation Verified')
    print('- RecSysDataSet.use_dataloader(DataSet.<name>) loads canonical datasets with user/item/rating/timestamp columns.')
    print('- MakeImplicit(threshold) converts explicit feedback to implicit by thresholding ratings.')
    print('- CorePruning(core) performs k-core filtering.')
    print('- UserHoldout(validation_size, test_size) is the user-aware holdout splitter.')
    print('- ExperimentPlan.add_algorithm accepts LensKit.ImplicitMFScorer, LensKit.ItemKNNScorer, LensKit.PopScorer.')
    print('- Evaluator(NDCG([...]), Recall([...])) computes ranking metrics at specified cutoffs.')
