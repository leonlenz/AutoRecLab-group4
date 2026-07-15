# Experiment Summary

## User Request

You asked for a seed-sensitivity experiment in LensKit using three algorithms—ALS, ItemKNN, and Pop—on three implicit-feedback datasets:

- MovieLens100K
- Amazon Video Games
- Last.FM

Requested procedure:

- 5-core preprocessing on all datasets
- For MovieLens and Amazon, keep only ratings greater than 3 as implicit interactions
- 5 random seeds for user-based 80/20 holdout splits
- Standard hyperparameters
- Evaluate with nDCG@1/5/10 and Precision@1/5/10
- Provide a short statistical analysis of seed effects

## What Was Run

From the code, the experiment was set up as follows:

- Seeds intended: `7, 19, 42, 77, 123`
- Datasets intended:
  - `MovieLens100K`
  - `Amazon2014VideoGames`
  - `HetrecLastFM`
- Preprocessing:
  - `MovieLens100K` and `Amazon2014VideoGames`: ratings `> 3` were retained and converted to implicit user-item interactions
  - `HetrecLastFM`: user-item interactions only
  - `CorePruning(5)` applied to all datasets
- Split strategy:
  - user-based holdout with `test_size=0.2`
  - at least 1 test interaction per user, with the remainder in train
- Algorithms configured:
  - `LensKit.ImplicitMFScorer` → ALS
  - `LensKit.ItemKNNScorer` → ItemKNN
  - `LensKit.PopScorer` → Pop
- Metrics configured in code:
  - `NDCG([1, 5, 10])`
  - `Precision([1, 5, 10])`

However, the provided output shows the run did not finish. It timed out during:

- `Amazon2014VideoGames / ItemKNNScorer / seed=123`

No printed seed-level results, summary table, or statistical analysis were produced before timeout.

## Key Results

Only preprocessing counts and partial execution status are explicitly available from the output.

| Dataset | Seed | Preprocessing result | Algorithms completed according to output | Metrics available |
|---|---:|---|---|---|
| MovieLens100K | 7 | 55,375 interactions before 5-core; 54,413 after 5-core | ALS completed; later output truncated, so full status unknown | N/A |
| Amazon2014VideoGames | 123 | 970,030 interactions before 5-core; 132,209 after 5-core | ALS completed; ItemKNN started but timed out during evaluation | N/A |
| HetrecLastFM | N/A | N/A | N/A | N/A |

Short statistical analysis: **not available**, because the run did not complete and the output does not include the generated result files (`seed_level_results.csv`, `summary_by_dataset_algorithm.csv`, or `paired_seed_differences.csv`).

Metric interpretation, based on the experiment code and LensKit ranking metrics:

- **Precision@k** measures the fraction of the top-`k` recommended items that are relevant.
- **nDCG@k** measures ranked recommendation quality, giving more credit when relevant items appear nearer the top of the list.

## Limitations

- The experiment **did not complete**; it ended with:
  - `TimeoutError: Execution exceeded the time limit of an hour`
- Because of that, the output contains:
  - no final metric values
  - no per-seed results table
  - no aggregated mean/std/range statistics
  - no paired seed-difference analysis
- The logs are also truncated, so even completion status for many earlier runs cannot be verified from the provided text alone.
- Although the code would have written result artifacts if successful, those artifact contents were not included in the provided output.

## Conclusion

The code correctly sets up the requested seed-sensitivity experiment with:

- 5-core filtering
- implicit conversion for MovieLens and Amazon using ratings `> 3`
- 5 random seeds
- user-based 80/20 holdout
- ALS, ItemKNN, and Pop
- nDCG@1/5/10 and Precision@1/5/10
- summary statistics across seeds

But based on the provided output, the experiment **timed out before completion**, so there are **no factual accuracy results or statistical seed-effect findings to report**. The only confirmed numeric findings are preprocessing interaction counts for two dataset/seed cases shown in the logs. To answer your original research question quantitatively, the experiment would need to be rerun to completion or the generated output files would need to be provided.