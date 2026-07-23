# Experiment Summary

## User Request

Quantify how data split random seeds affect recommender system accuracy using LensKit on three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM. The requested setup was:

- 5-core filtering on all datasets
- Convert MovieLens and Amazon ratings > 3 to implicit interactions
- 5 random split seeds
- User-based 80/20 holdout split
- Algorithms: ALS, ItemKNN, Pop
- Metrics: nDCG@1,5,10 and Precision@1,5,10
- Short statistical analysis of seed sensitivity

## What Was Run

The code attempted to:

- Load:
  - `MovieLens100K`
  - `Amazon2014VideoGames`
  - `HetrecLastFM`
- Apply preprocessing:
  - `MakeImplicit(3)` for MovieLens and Amazon
  - `CorePruning(5)` for all datasets
  - `UserHoldout(validation_size=0.0, test_size=0.2)`
- Evaluate three LensKit algorithms:
  - `LensKit.ImplicitMFScorer`
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`
- Measure:
  - `NDCG([1, 5, 10])`
  - `Precision([1, 5, 10])`
- Repeat for seeds:
  - 11, 22, 33, 44, 55

## Key Results

| Dataset | Seed | Preprocessing Completed? | Evaluation Completed? | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---:|---|---|---|---|---|---|---|---|
| MovieLens100K | 11 | Yes, until split step | No | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | All | Not reached | No | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | All | Not reached | No | N/A | N/A | N/A | N/A | N/A | N/A |

Observed preprocessing output for MovieLens100K seed 11:

- Before implicit conversion: 100,000 interactions
- After `MakeImplicit(3)`: 82,520 interactions
- After 5-core pruning: 81,697 interactions

The run then failed during the split step with:

- `InvalidParameterError: The 'test_size' parameter of train_test_split must be a float in the range (0.0, 1.0)... Got 0.0 instead.`

## Limitations

- No model training or metric computation completed.
- No results were produced for any dataset, algorithm, or seed.
- The experiment crashed before the first split finished, so seed sensitivity could not be analyzed from the provided output.
- The requested `validation_size=0.0` caused the split failure in the `UserHoldout` preprocessing step.

## Conclusion

The experiment did not complete, so there are no valid nDCG or Precision results to report. The only confirmed outcome is that MovieLens100K was successfully loaded and transformed through implicit conversion and 5-core filtering before the run failed at the holdout split stage because `test_size=0.0` was rejected by the split implementation.