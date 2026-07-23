# Experiment Summary

## User Request

The request was to quantify how random split seeds affect recommender accuracy using LensKit on three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM. The intended setup was:

- 5-core filtering on all datasets
- Convert ratings > 3 to implicit feedback for MovieLens and Amazon
- 5 different random seeds
- User-based 80/20 holdout split
- Algorithms: ALS, ItemKNN, Pop
- Metrics: nDCG@1,5,10 and Precision@1,5,10
- Short statistical analysis across seeds

## What Was Run

The code did the following:

- Loaded three datasets:
  - `MovieLens100K` with implicit conversion
  - `Amazon2014VideoGames` with implicit conversion
  - `HetrecLastFM` without implicit conversion
- Applied:
  - `MakeImplicit(3)` to MovieLens and Amazon
  - `CorePruning(5)` to all datasets
- Used 5 seeds: `11, 22, 33, 44, 55`
- For each dataset and seed:
  - Performed `UserHoldout(validation_size=0.2, test_size=0.2)`
  - Evaluated with `NDCG([1, 5, 10])` and `Precision([1, 5, 10])`
- Algorithms added to the plan:
  - `LensKit.ImplicitMFScorer`
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`

Note: the code used `ImplicitMFScorer`, not ALS. The output also shows the run did not finish for all combinations.

## Key Results

| Dataset | Algorithm | Seeds completed | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens100K | ALS requested / ImplicitMFScorer run | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | Pop | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | ALS requested / ImplicitMFScorer run | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | Pop | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | ALS requested / ImplicitMFScorer run | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | Pop | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

The output provided does not include the actual per-seed metric values or the summary table contents. The only explicit result visible is that the experiment ran at least through parts of the MovieLens and Amazon evaluations, but it timed out during `Amazon2014VideoGames/PopScorer`.

## Limitations

- The experiment did not complete successfully; it ended with:
  - `TimeoutError: Execution exceeded the time limit of an hour`
- Because of the timeout, full results for all dataset/algorithm/seed combinations are not available.
- The output snippet does not contain the computed metric values or the saved CSV contents, so exact nDCG and Precision values cannot be reported.
- There is a mismatch between the request and the code:
  - The request asked for **ALS**
  - The code actually used **ImplicitMFScorer**
- There is also a dataset-name mismatch:
  - The request said **Last.FM**
  - The code used **HetrecLastFM**
- The run settings in the code also differ from the requested split wording:
  - `UserHoldout(validation_size=0.2, test_size=0.2)` creates validation and test splits, rather than a simple 80/20 holdout only.

## Conclusion

Based on the provided materials, the experiment was set up to test seed sensitivity for LensKit recommenders with 5-core filtering and implicit conversion, but it did not finish. As a result, no valid quantitative conclusion about how much random seeds affected nDCG@k or Precision@k can be drawn from the available output. The only factual conclusion is that the job timed out before completion, so the requested statistical analysis is unavailable from this run.