# Experiment Summary

## User Request

Quantify how random data split seeds affect recommender system accuracy using LensKit on three algorithms (ALS, ItemKNN, Pop) and three implicit-feedback datasets (MovieLens100K, Amazon Video Games, Last.FM), with 5-core filtering, 5 random seeds, user-based 80/20 holdout splitting, and evaluation by nDCG@1/5/10 and Precision@1/5/10.

## What Was Run

The experiment code did the following:

- Loaded three datasets:
  - `MovieLens100K` with implicit conversion threshold `> 3`
  - `Amazon2014VideoGames` with implicit conversion threshold `> 3`
  - `HetrecLastFM` with no explicit implicit conversion step
- Applied:
  - `MakeImplicit(3)` for MovieLens and Amazon Video Games
  - `CorePruning(5)` to all datasets
  - `UserHoldout(validation_size=0.1, test_size=0.2)` as the split procedure
- Ran five seeds:
  - `11, 22, 33, 44, 55`
- Trained three LensKit algorithms:
  - `LensKit.ImplicitMFScorer` (ALS-style implicit matrix factorization)
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`
- Evaluated:
  - `NDCG@1, 5, 10`
  - `Precision@1, 5, 10`

## Key Results

The output is incomplete, so exact aggregated metric values for all datasets/algorithms/seeds are not available. The experiment began successfully for MovieLens100K, but the provided output only includes partial per-run logs and one visible result fragment. Amazon Video Games failed during the first seed’s run due to a timeout, and no complete results for Last.FM are shown in the provided output.

| Dataset | Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens100K | ALS / ImplicitMFScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | ItemKNNScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | PopScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | ALS / ImplicitMFScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | ItemKNNScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | PopScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | ALS / ImplicitMFScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | ItemKNNScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| HetrecLastFM | PopScorer | N/A | N/A | N/A | N/A | N/A | N/A |

Observed factual points from the output:

- MovieLens100K preprocessing succeeded:
  - interactions before implicit conversion: `100000`
  - after implicit conversion: `82520`
  - after 5-core pruning: `81697`
- Amazon Video Games preprocessing succeeded for the first shown run:
  - interactions before implicit conversion: `1324753`
  - after implicit conversion: `1094400`
  - after 5-core pruning: `177572`
- Amazon Video Games did not complete the first shown model run:
  - `TimeoutError: Execution exceeded the time limit of an hour`
- A visible result fragment shows one PopScorer entry with very low values:
  - `NDCG@5 = 0.007069`
  - `NDCG@10 = 0.006037`
  - `Precision@1 = 0.009780`
  - `Precision@5 = 0.006379`
  - `Precision@10 = 0.005241`
  - The dataset and seed for that fragment are not fully identifiable from the truncated output, so it cannot be reliably assigned here.

## Limitations

- The output is truncated and does not include the full aggregated summary table.
- Exact results for all dataset/algorithm/seed combinations are not present.
- The experiment used `validation_size=0.1` and `test_size=0.2`, which means it was not a simple 80/20 split as requested; the code actually created a 70/10/20 train/validation/test split.
- The Last.FM dataset name in code is `HetrecLastFM`, which may not exactly match the user’s requested naming.
- Because Amazon Video Games timed out, the experiment did not finish fully.
- No statistical comparison results can be computed from the provided output alone.

## Conclusion

The experiment setup matches the user’s intent in terms of datasets, algorithms, seeds, core filtering, implicit conversion for MovieLens and Amazon, and ranking metrics. However, the provided output is incomplete and one dataset run timed out, so there are no reliable final accuracy numbers or seed-sensitivity statistics to report. The only firm conclusion from the visible output is that preprocessing succeeded for MovieLens100K and Amazon Video Games, but the Amazon run failed during model evaluation due to timeout.