# Experiment Summary

## User Request

The user requested an experiment to quantify how much data split random seeds affect recommender system accuracy. The experiment was to test three algorithms (ALS/ImplicitMF, ItemKNN, Pop) on three datasets (MovieLens100K, Amazon Video Games, Last.FM) using LensKit, with 5 different random seeds for user-based 80/20 holdout splits, measuring nDCG@k and Precision@k for k=1, 5, 10, followed by a statistical analysis.

## What Was Run

The experiment code implemented the requested procedure with the following details:

- **Datasets**: MovieLens100K, Amazon2014VideoGames, HetrecLastFM
- **Preprocessing**: 5-core filtering on all datasets; MovieLens100K and Amazon2014VideoGames were converted to implicit feedback using a threshold of 4 (ratings ≥ 4 kept as positive interactions)
- **Algorithms**: PopScorer (popularity baseline), ItemKNNScorer (max_nbrs=30, min_nbrs=5), ImplicitMFScorer (features=50, epochs=100) — all with `feedback="implicit"`
- **Split**: UserHoldout with test_size=0.2 and a tiny validation_size=0.001 (to work around a library constraint)
- **Seeds**: 5 seeds: [42, 73, 123, 256, 999]
- **Metrics**: NDCG@k and Precision@k for k=1, 5, 10
- **Statistical analysis**: Mean, standard deviation, and coefficient of variation (CV) computed across the 5 seeds per metric

## Key Results

**The experiment did not complete successfully.** A critical failure occurred during the final seed (seed=999) for the Amazon2014VideoGames dataset with the ImplicitMFScorer (ALS) algorithm, which timed out after exceeding a one-hour execution limit. As a result, results for that specific combination are missing. Additionally, the output was truncated, so results for the HetrecLastFM dataset and the statistical analysis section are not available in the provided output.

Below are the results that were successfully extracted from the output before the timeout:

### MovieLens100K — All seeds completed

| Algorithm | Metric | Seed 42 | Seed 73 | Seed 123 | Seed 256 | Seed 999 |
|-----------|--------|---------|---------|----------|----------|----------|
| PopScorer | nDCG@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PopScorer | nDCG@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PopScorer | nDCG@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PopScorer | Precision@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PopScorer | Precision@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PopScorer | Precision@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ItemKNNScorer | nDCG@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ItemKNNScorer | nDCG@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ItemKNNScorer | nDCG@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ItemKNNScorer | Precision@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ItemKNNScorer | Precision@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ItemKNNScorer | Precision@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ImplicitMFScorer | nDCG@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ImplicitMFScorer | nDCG@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ImplicitMFScorer | nDCG@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ImplicitMFScorer | Precision@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ImplicitMFScorer | Precision@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ImplicitMFScorer | Precision@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

All metrics for MovieLens100K are exactly 0.0000 across all algorithms, seeds, and metrics. This is an anomalous result that likely indicates a systematic issue — possibly related to the implicit feedback conversion, the user holdout split configuration, or how the evaluation was performed.

### Amazon2014VideoGames — Partial results

The output shows that for seeds 42, 73, 123, and 256, all three algorithms completed successfully on Amazon2014VideoGames. However, the actual metric values for these runs were not printed in the truncated output before the timeout occurred on seed 999 with ImplicitMFScorer. The specific metric values for Amazon2014VideoGames are therefore unavailable from the provided output.

### HetrecLastFM — No results available

The output was truncated before any results for the HetrecLastFM dataset were printed.

## Limitations

1. **Experiment did not complete**: The ImplicitMFScorer (ALS) timed out on Amazon2014VideoGames during seed 999, and the output was truncated before HetrecLastFM results or the statistical analysis section were printed.
2. **All MovieLens100K metrics are zero**: This is highly unusual and suggests a potential bug or misconfiguration — possibly the implicit conversion threshold (ratings ≥ 4) combined with 5-core filtering left the test set with no positive predictions, or the evaluation logic produced zero-valued results. This makes the MovieLens100K results uninformative for the stated goal.
3. **Missing Amazon2014VideoGames metric values**: While the runs completed for seeds 42–256, the actual metric values were not captured in the truncated output.
4. **No statistical analysis available**: The mean, standard deviation, and CV calculations across seeds were not printed due to the truncation.
5. **No HetrecLastFM results**: The dataset was loaded and preprocessed, but no evaluation results were printed.

## Conclusion

The experiment was designed correctly to answer the user's question about the impact of random seeds on recommender accuracy, but it did not produce usable results due to two main issues:

1. **All MovieLens100K metrics are zero** across all algorithms and seeds, which is a clear anomaly that prevents any meaningful analysis of seed variability for that dataset.
2. **The experiment timed out** on the Amazon2014VideoGames dataset with ImplicitMFScorer (ALS), and the output was truncated before results for HetrecLastFM and the statistical analysis could be printed.

As a result, it is not possible to draw any conclusions about how much data split random seeds affect recommender system accuracy from this experiment output. The zero-valued results for MovieLens100K suggest a systematic issue that would need to be diagnosed (e.g., the implicit feedback threshold, the holdout split configuration, or the evaluation pipeline) before the experiment can produce meaningful findings.