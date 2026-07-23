# Experiment Summary

## User Request

The user wanted to quantify how much data split random seeds affect recommender system accuracy. The request specified testing three algorithms (ALS, ItemKNN, Pop) on three datasets (MovieLens100K, Amazon Video Games, Last.FM) with implicit feedback, 5-core filtering, 5 different random seeds for user-based 80/20 holdout splits, and evaluation using nDCG@k and Precision@k for k=1, 5, 10 with a statistical analysis.

## What Was Run

The experiment code implemented the requested design:

- **Datasets**: MovieLens100K, Amazon2014VideoGames, HetrecLastFM
- **Algorithms**: ALS (ImplicitMFScorer), ItemKNN (ItemKNNScorer), Pop (PopScorer) — all using implicit feedback
- **Preprocessing**: Ratings > 3 converted to implicit for MovieLens100K and Amazon2014VideoGames; 5-core pruning applied to all datasets; user-based holdout split (70% train, 10% validation, 20% test)
- **Random seeds**: 5 seeds (42, 123, 456, 789, 1111)
- **Metrics**: NDCG@k and Precision@k for k=1, 5, 10
- **Total conditions**: 3 datasets × 3 algorithms × 5 seeds = 45 runs

## Key Results

**The experiment did not complete successfully.** A `TimeoutError` occurred during the PopScorer evaluation on the Amazon2014VideoGames dataset (seed 42), and a `KeyError: 'rank'` occurred during evaluation result collection. As a result, **no metric values are available** for any condition.

The output shows that only the first seed (42) partially executed. The following runs completed before the timeout:

| Dataset | Algorithm | Status |
|---|---|---|
| MovieLens100K | ALS | Completed (seed 42) |
| MovieLens100K | ItemKNN | Completed (seed 42) |
| MovieLens100K | Pop | Completed (seed 42) |
| Amazon2014VideoGames | ALS | Completed (seed 42) |
| Amazon2014VideoGames | ItemKNN | Completed (seed 42) |
| Amazon2014VideoGames | Pop | **TimeoutError** (seed 42) |
| HetrecLastFM | All | Not reached |

Even for completed runs, a `KeyError: 'rank'` prevented the results from being collected into the combined dataframe, so **no metric values (nDCG or Precision) were recorded**.

## Limitations

1. **Experiment did not finish**: A timeout (1 hour limit) was hit during the PopScorer evaluation on Amazon2014VideoGames. The HetrecLastFM dataset was never processed.
2. **KeyError in result collection**: A `KeyError: 'rank'` occurred when trying to sort evaluation results, preventing any metric values from being saved even for completed runs.
3. **Only 1 of 5 seeds executed**: Only seed 42 was partially run; seeds 123, 456, 789, and 1111 were never reached.
4. **No statistical analysis possible**: Since no metric values were collected, the mean ± std across seeds cannot be computed.

## Conclusion

The experiment was designed correctly to answer the user's question but failed to produce any usable results due to two issues: (1) a `TimeoutError` during the PopScorer evaluation on the Amazon2014VideoGames dataset, and (2) a `KeyError: 'rank'` in the result aggregation logic that prevented metric values from being collected even for completed runs. No conclusions can be drawn about the impact of data split random seeds on recommender accuracy from the available output. The code and pipeline would need to be debugged (fixing the `'rank'` key issue and either increasing the timeout or optimizing the Pop algorithm on large datasets) before the experiment can produce meaningful results.