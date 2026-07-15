# Experiment Summary

## User Request

Quantify how much data-split random seeds affect recommender system accuracy using LensKit on:

- Algorithms: ALS, ItemKNN, Pop
- Datasets: MovieLens100K, Amazon Video Games, Last.FM
- Preprocessing: 5-core filtering for all datasets; convert ratings > 3 to implicit interactions for MovieLens and Amazon
- Evaluation: 5 random seeds, user-based 80/20 holdout split, standard hyperparameters
- Metrics: nDCG@1, 5, 10 and Precision@1, 5, 10
- Also provide a short statistical analysis

## What Was Run

The code used the following setup:

- **Datasets**
  - `MovieLens100K`
  - `Amazon2014VideoGames`
  - `HetrecLastFM`

- **Preprocessing**
  - MovieLens100K and Amazon2014VideoGames:
    - `MakeImplicit(3)` then `CorePruning(5)`
  - HetrecLastFM:
    - `CorePruning(5)` only

- **Split protocol**
  - `UserHoldout(validation_size=0.2, test_size=0.2)`
  - Run for seeds: `11, 22, 33, 44, 55`

- **Algorithms**
  - `LensKit.ImplicitMFScorer`
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`

- **Metrics**
  - `NDCG(1, 5, 10)`
  - `Precision(1, 5, 10)`

Note: The code used `ImplicitMFScorer`, not an algorithm explicitly named ALS. The output only shows these LensKit algorithm identifiers.

## Key Results

The output explicitly reports only the **best algorithm per dataset/metric/k**, along with mean, standard deviation, coefficient of variation, and spread across algorithms. Full per-algorithm/per-seed tables were not included in the visible output, so exact results for all three algorithms cannot be reconstructed here.

| Dataset | Metric | k | Best Algorithm | Mean | Std | CV | Spread |
|---|---:|---:|---|---:|---:|---:|---:|
| MovieLens100K | NDCG | 1 | LensKit.ItemKNNScorer-c3d66610-44 | 0.3001 | 0.0000 | 0.000 | 0.1156 |
| MovieLens100K | NDCG | 5 | LensKit.ItemKNNScorer-c3d66610-44 | 0.2442 | 0.0000 | 0.000 | 0.0884 |
| MovieLens100K | NDCG | 10 | LensKit.ItemKNNScorer-c3d66610-44 | 0.2166 | 0.0000 | 0.000 | 0.0762 |
| MovieLens100K | Precision | 1 | LensKit.ItemKNNScorer-c3d66610-44 | 0.3001 | 0.0000 | 0.000 | 0.1156 |
| MovieLens100K | Precision | 5 | LensKit.ItemKNNScorer-c3d66610-44 | 0.2284 | 0.0000 | 0.000 | 0.0846 |
| MovieLens100K | Precision | 10 | LensKit.ItemKNNScorer-c3d66610-44 | 0.1968 | 0.0000 | 0.000 | 0.0684 |
| Amazon2014VideoGames | NDCG | 1 | LensKit.ItemKNNScorer-c3d66610-55 | 0.2423 | 0.1104 | 0.456 | 0.1094 |
| Amazon2014VideoGames | NDCG | 5 | LensKit.ItemKNNScorer-c3d66610-55 | 0.1953 | 0.0889 | 0.456 | 0.0768 |
| Amazon2014VideoGames | NDCG | 10 | LensKit.ItemKNNScorer-c3d66610-55 | 0.1728 | 0.0788 | 0.456 | 0.0678 |
| Amazon2014VideoGames | Precision | 1 | LensKit.ItemKNNScorer-c3d66610-55 | 0.2423 | 0.1104 | 0.456 | 0.1094 |
| Amazon2014VideoGames | Precision | 5 | LensKit.ItemKNNScorer-c3d66610-55 | 0.1824 | 0.0831 | 0.456 | 0.0685 |
| Amazon2014VideoGames | Precision | 10 | LensKit.ImplicitMFScorer-f0dd7ecb-22 | 0.1624 | 0.0000 | 0.000 | 0.0654 |
| HetrecLastFM | NDCG | 1 | LensKit.ImplicitMFScorer-f0dd7ecb-22 | 0.2174 | 0.0000 | 0.000 | 0.0845 |
| HetrecLastFM | NDCG | 5 | LensKit.ImplicitMFScorer-f0dd7ecb-22 | 0.1863 | 0.0000 | 0.000 | 0.0678 |
| HetrecLastFM | NDCG | 10 | LensKit.ImplicitMFScorer-f0dd7ecb-22 | 0.1724 | 0.0000 | 0.000 | 0.0674 |
| HetrecLastFM | Precision | 1 | LensKit.ImplicitMFScorer-f0dd7ecb-22 | 0.2174 | 0.0000 | 0.000 | 0.0845 |
| HetrecLastFM | Precision | 5 | LensKit.ImplicitMFScorer-f0dd7ecb-22 | 0.1786 | 0.0000 | 0.000 | 0.0647 |
| HetrecLastFM | Precision | 10 | LensKit.ImplicitMFScorer-f0dd7ecb-22 | 0.1624 | 0.0000 | 0.000 | 0.0654 |

Short statistical takeaways from the reported analysis:

- **MovieLens100K**: the reported standard deviation across seeds is `0.0000` for the best algorithm lines, suggesting no visible seed sensitivity in the displayed summary.
- **Amazon2014VideoGames**: the reported coefficient of variation is `0.456` for several metrics, indicating noticeably higher seed sensitivity than the other datasets in the displayed summary.
- **HetrecLastFM**: the reported standard deviation is `0.0000` for the best algorithm lines, again suggesting no visible seed sensitivity in the displayed summary.

## Limitations

- The user asked for **ALS**, but the experiment code used **`LensKit.ImplicitMFScorer`**. The output does not explicitly confirm that this is ALS, so that equivalence should not be assumed here.
- The visible output does **not** include the full raw or summary tables for all algorithm/seed combinations, so exact per-algorithm results cannot be fully reported.
- The experiment output is truncated; only the short statistical analysis and partial summary are available.
- The output does not show separate results for validation vs. test beyond the described holdout procedure.
- The preprocessing and dataset names in code/output differ slightly from the request:
  - `Amazon2014VideoGames` vs. “Amazon Video Games”
  - `HetrecLastFM` vs. “Last.FM”

## Conclusion

This experiment did run a LensKit-based seed-sensitivity test with 5 random split seeds, user holdout splitting, 5-core filtering, and implicit conversion for MovieLens and Amazon. From the reported summary, ItemKNN was best on MovieLens100K across all shown metrics, while ImplicitMFScorer was best on HetrecLastFM and on Amazon at Precision@10. The displayed statistical summary suggests Amazon2014VideoGames showed the greatest sensitivity to split seed variation, while MovieLens100K and HetrecLastFM showed little to no visible variation in the reported best-algorithm summaries.