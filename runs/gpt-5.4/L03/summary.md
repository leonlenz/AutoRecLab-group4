# Experiment Summary

## User Request

Quantify how much data split random seeds affect recommender accuracy using LensKit on three implicit-feedback datasets:

- MovieLens100K
- Amazon Video Games
- Last.FM

Requested setup:

- 5-core preprocessing on all datasets
- For MovieLens and Amazon, convert ratings greater than 3 to implicit interactions
- 5 random seeds for user-based 80/20 holdout splits
- Test 3 algorithms: ALS, ItemKNN, Pop
- Evaluate nDCG@k and Precision@k for k = 1, 5, 10
- Provide a short statistical analysis of seed sensitivity

## What Was Run

Based on the code, the experiment was configured as follows:

- Seeds: `11, 23, 37, 53, 71`
- Algorithms:
  - `LensKit.ImplicitMFScorer` → ALS
  - `LensKit.ItemKNNScorer` → ItemKNN
  - `LensKit.PopScorer` → Pop
- Split strategy:
  - `UserHoldout(test_size=0.2)` with a tiny validation size effectively merging validation back into train for analysis
- Metrics:
  - nDCG@1, nDCG@5, nDCG@10
  - Precision@1, Precision@5, Precision@10
- Preprocessing:
  - MovieLens100K: implicit conversion with threshold 4, then 5-core pruning
  - Amazon2014VideoGames: implicit conversion with threshold 4, then 5-core pruning
  - HetrecLastFM: 5-core pruning only

The output confirms these preprocessing results:

| Dataset | Interactions after preprocessing | Users | Items |
|---|---:|---:|---:|
| MovieLens100K | 54,413 | 938 | 1,008 |
| Amazon2014VideoGames | 132,209 | N/A | N/A |
| HetrecLastFM | N/A | N/A | N/A |

For MovieLens, the output showed:
- 100,000 interactions before implicit conversion
- 55,375 after implicit conversion
- 54,413 after 5-core pruning

For Amazon Video Games, the output showed:
- 1,324,753 interactions before implicit conversion
- 970,030 after implicit conversion
- 132,209 after 5-core pruning

## Key Results

The experiment did **not complete**. The run timed out during the Amazon2014VideoGames / PopScorer stage:

- ALS on Amazon completed
- ItemKNN on Amazon completed
- Pop on Amazon started prediction but the overall execution hit:
  - `TimeoutError: Execution exceeded the time limit of an hour`

Because the output does not include the printed per-seed metric tables or the saved CSV contents, the requested accuracy numbers and seed-sensitivity statistics are not available from the provided materials.

| Dataset | Algorithm | Seeds completed | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 | Statistical analysis |
|---|---|---:|---|---|---|---|---|---|---|
| MovieLens100K | ALS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | Pop | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | ALS | At least partially run | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | ItemKNN | At least partially run | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | Pop | Incomplete | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | ALS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | Pop | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Why values are `N/A`:
- The output was truncated and does not include the final printed results tables.
- The run terminated before completion, so the summary/statistical CSV outputs referenced in the code were not confirmed as produced.

## Limitations

- The experiment output is incomplete and truncated.
- The run exceeded the time limit before finishing all dataset/algorithm/seed combinations.
- No per-seed metric values are visible in the provided output.
- No final summary tables or saved CSV contents are shown.
- Therefore, the requested comparison of seed effects on nDCG and Precision cannot be quantified from the provided materials alone.

## Conclusion

The code correctly set up the requested LensKit seed-sensitivity experiment with 5-core preprocessing, implicit conversion for MovieLens and Amazon, 5 random user-holdout splits, and evaluation with nDCG@{1,5,10} and Precision@{1,5,10}. However, the actual run did not finish: it timed out during Amazon Video Games with PopScorer, and the provided output does not contain the final metric tables. As a result, no factual statistical conclusion about how much random seeds affected recommender accuracy can be drawn from the available results.