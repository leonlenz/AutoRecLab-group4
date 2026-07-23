# Experiment Summary

## User Request

You asked for a LensKit experiment to quantify how random data-split seeds affect recommender accuracy for three algorithms:

- ALS
- ItemKNN
- Pop

across three implicit-feedback datasets:

- MovieLens100K
- Amazon Video Games
- Last.FM

with:

- 5-core preprocessing
- ratings > 3 converted to implicit interactions for MovieLens and Amazon
- 5 random seeds
- user-based 80/20 holdout split
- metrics: nDCG@1/5/10 and Precision@1/5/10
- short statistical analysis

## What Was Run

From the code and output, the experiment was configured as follows:

- Seeds: `11, 23, 37, 47, 59`
- Metrics: `NDCG` and `Precision` at `k = 1, 5, 10`
- Algorithms:
  - `LensKit.ImplicitMFScorer` → ALS
  - `LensKit.ItemKNNScorer` → ItemKNN
  - `LensKit.PopScorer` → Pop
- Splitting:
  - `UserHoldout(validation_size=0.2, test_size=0.2)`
  - The script reports only the `test` fold
- Preprocessing:
  - MovieLens100K: ratings filtered to `>= 4`, then made implicit, then 5-core pruning
  - Amazon2014VideoGames: intended the same as MovieLens
  - HetrecLastFM: made implicit, then 5-core pruning

The output shows preprocessing completed for MovieLens100K:

- interactions before implicit conversion: `55,375`
- interactions after implicit conversion: `55,375`
- interactions after 5-core pruning: `54,413`

However, the run crashed during the first dataset/seed combination:

- dataset: `MovieLens100K`
- seed: `11`

The failure occurred after LensKit printed an evaluation table, because the script could not extract evaluation outputs and raised:

- `RuntimeError: No evaluation outputs discovered ...`

As a result, the full 5-seed × 3-dataset experiment, summary statistics, and paired tests were not completed.

## Key Results

Only one visible evaluation result was produced before the crash: MovieLens100K, seed 11.

| Dataset | Seed | Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MovieLens100K | 11 | ALS | 0.141791 | 0.137005 | 0.122649 | 0.141791 | 0.135394 | 0.115672 |
| MovieLens100K | 11 | ItemKNN | 0.206823 | 0.169741 | 0.156337 | 0.206823 | 0.162047 | 0.146588 |
| MovieLens100K | 11 | Pop | 0.184435 | 0.135297 | 0.118941 | 0.184435 | 0.125586 | 0.107249 |
| MovieLens100K | 23 | ALS | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | 37 | ALS | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | 47 | ALS | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | 59 | ALS | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | all | all | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | all | all | N/A | N/A | N/A | N/A | N/A | N/A |

Factual observations from the only available result:

- On `MovieLens100K` with `seed 11`, `ItemKNN` had the highest score on all six reported metrics.
- `Pop` outperformed `ALS` on all six reported metrics for that same run.
- No seed-variation analysis can be computed from a single completed seed.

## Limitations

The requested experiment did not complete.

Specific limitations visible in the output:

1. The run crashed on the first dataset/seed combination:
   - `MovieLens100K`, `seed 11`

2. Although LensKit displayed one evaluation table, the script failed to collect those results programmatically and raised:
   - `RuntimeError: No evaluation outputs discovered ...`

3. Because of that crash:
   - no results were produced for seeds `23, 37, 47, 59`
   - no results were produced for `Amazon2014VideoGames`
   - no results were produced for `HetrecLastFM`
   - no summary CSVs, paired t-tests, confidence intervals, or seed-sensitivity statistics were generated

4. The requested “short statistical analysis” cannot be performed from the provided output, because statistical comparisons across seeds require multiple completed runs.

## Conclusion

Based on the provided materials, the full seed-sensitivity experiment was not successfully executed. The only confirmed result is for `MovieLens100K` with `seed 11`, where `ItemKNN` was best, followed by `Pop`, then `ALS`, across all reported `nDCG` and `Precision` cutoffs.

But the main question—how much random split seeds affect recommender accuracy across the three datasets and three algorithms—cannot be answered from this run, because only one seed produced visible metrics before the script crashed.