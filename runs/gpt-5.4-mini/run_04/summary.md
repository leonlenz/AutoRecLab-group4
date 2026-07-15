# Experiment Summary

## User Request

Quantify how much data split random seeds affect recommender system accuracy using LensKit on three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM. The requested setup was:

- 5-core filtering for all datasets
- Convert MovieLens and Amazon ratings greater than 3 to implicit interactions
- 5 random split seeds
- User-based 80/20 holdout
- Algorithms: ALS, ItemKNN, Pop
- Metrics: nDCG@1,5,10 and Precision@1,5,10
- Short statistical analysis of seed sensitivity

## What Was Run

The code used OmniRec with LensKit scorers and performed:

- Preprocessing:
  - `MakeImplicit(3)` for MovieLens100K and Amazon2014VideoGames
  - `CorePruning(5)` for all datasets
- Datasets:
  - MovieLens100K
  - Amazon2014VideoGames
  - HetrecLastFM
- Split seeds:
  - `[7, 13, 21, 42, 87]`
- Split method:
  - `UserHoldout(validation_size=0.2, test_size=0.2)`  
  - The printed text says “user-based 80/20 holdout,” but the code actually uses both validation and test size of 0.2.
- Algorithms:
  - `LensKit.ImplicitMFScorer` (ALS/implicit MF)
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`
- Metrics:
  - `NDCG([1, 5, 10])`
  - `Precision([1, 5, 10])`

## Key Results

The output includes a partial per-seed sample and a short statistical analysis, but it does **not** provide the full per-dataset, per-algorithm, per-k summary table in the visible output. Therefore, exact values for most requested combinations are unavailable here.

| Dataset | Algorithm | Metric | k | Seed-sensitivity summary |
|---|---|---:|---:|---|
| MovieLens100K | N/A | N/A | N/A | average range = 0.0000; max range = 0.0000 |
| Amazon2014VideoGames | N/A | N/A | N/A | average range = 0.0741; max range = 0.2794 |
| HetrecLastFM | N/A | N/A | N/A | average range = 0.0741; max range = 0.2794 |

Additional factual findings from the visible output:

- MovieLens100K preprocessing:
  - Interactions before implicit conversion: 100,000
  - After implicit conversion: 82,520
  - After 5-core pruning: 81,697
- The short statistical analysis reported:
  - MovieLens100K: no seed variation observed in the summary (`range=0.0000`)
  - Amazon2014VideoGames: most sensitive result was `LensKit.ItemKNNScorer-c3d66610-21 NDCG@1` with `std=0.1353`, `range=0.2794`
  - HetrecLastFM: most sensitive result was `LensKit.ItemKNNScorer-c3d66610-21 NDCG@1` with `std=0.1397`, `range=0.2794`

## Limitations

- The visible experiment output is truncated and does not contain the full summary table for all dataset/algorithm/metric/k combinations.
- Because of that truncation, exact mean/std/min/max/range values for each requested metric and algorithm cannot be fully reported from the provided output.
- The output shows `HetrecLastFM`, while the user requested `Last.FM`. The code uses `DataSet.HetrecLastFM`, so that is the dataset actually run.
- The code’s split call uses `validation_size=0.2` and `test_size=0.2`, which is not a plain single 80/20 split. The printed message says “user-based 80/20 holdout,” but the code indicates two 20% holdouts.
- The experiment output only shows a sample of per-seed rows, not the complete evaluation results needed to reconstruct every requested number.

## Conclusion

The experiment was run on MovieLens100K, Amazon2014VideoGames, and HetrecLastFM with 5-core filtering, implicit conversion for MovieLens and Amazon, five split seeds, and LensKit ALS/ItemKNN/Pop models. The visible results indicate:

- **MovieLens100K** showed no seed sensitivity in the reported summary.
- **Amazon2014VideoGames** and **HetrecLastFM** showed noticeable seed sensitivity, with the largest reported variation at **ItemKNN nDCG@1**.
- A complete per-metric comparison for all algorithms and k values is not available from the truncated output, so the detailed accuracy table cannot be fully reconstructed here.