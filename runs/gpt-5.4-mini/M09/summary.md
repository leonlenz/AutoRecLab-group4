# Experiment Summary

## User Request

The request was to quantify how random data-splitting seeds affect recommender system accuracy in LensKit, using three algorithms—ALS, ItemKNN, and Pop—on three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM. The experiment should:

- Apply 5-core filtering to all datasets
- Convert ratings greater than 3 to implicit interactions for MovieLens and Amazon
- Run 5 random seeds
- Use a user-based 80/20 holdout split
- Evaluate nDCG@1,5,10 and Precision@1,5,10
- Provide a short statistical analysis

## What Was Run

The code used `omnirec` with LensKit-backed models and ran the following setup:

- **Datasets**
  - `MovieLens100K`
  - `Amazon2014VideoGames`
  - `HetrecLastFM`

- **Preprocessing**
  - 5-core filtering for all datasets
  - Implicit conversion with threshold 3 for MovieLens100K and Amazon Video Games
  - No implicit conversion for Last.FM

- **Algorithms**
  - `LensKit.ImplicitMFScorer`  
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`

- **Split**
  - `UserHoldout(validation_size=0.20, test_size=0.20)`

- **Seeds**
  - `[7, 13, 21, 42, 84]`

- **Metrics**
  - `NDCG([1, 5, 10])`
  - `Precision([1, 5, 10])`

The script also included a short statistical summary over repeated runs, using mean, standard deviation, coefficient of variation, min, max, and range.

## Key Results

Only partial output is available in the provided log. The visible output shows results for **MovieLens100K** and only for **LensKit.PopScorer**. Results for Amazon Video Games, Last.FM, ALS/ItemKNN, and the full statistical analysis are not visible in the output provided here.

### Visible MovieLens100K results for PopScorer

| Dataset | Algorithm | Metric | k | Runs | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens100K | PopScorer | Precision | 1 | 5 | 0.220573 | ~0.000000 | 0.220573 | 0.220573 |
| MovieLens100K | PopScorer | Precision | 5 | 5 | 0.139343 | 0.000000 | 0.139343 | 0.139343 |
| MovieLens100K | PopScorer | Precision | 10 | 5 | 0.131601 | 0.000000 | 0.131601 | 0.131601 |
| MovieLens100K | PopScorer | nDCG | 1 | 5 | 0.220573 | ~0.000000 | 0.220573 | 0.220573 |
| MovieLens100K | PopScorer | nDCG | 5 | 5 | 0.155064 | 0.000000 | 0.155064 | 0.155064 |
| MovieLens100K | PopScorer | nDCG | 10 | 5 | 0.144184 | 0.000000 | 0.144184 | 0.144184 |

### Interpretation of the visible results

- For the **visible MovieLens100K PopScorer runs**, the metric values are identical across all 5 seeds, indicating **no observed variation from seed changes** in the output shown.
- However, this does **not** establish the same conclusion for the other algorithms or datasets, because their results are not visible in the provided output.

## Limitations

- The provided execution log is **truncated**, so the full per-seed results are not available.
- The output does **not show**:
  - Amazon Video Games results
  - Last.FM results
  - ALS results
  - ItemKNN results
  - The complete summary or statistical analysis tables
- The user requested **ALS, ItemKNN, and Pop**, but the code actually ran **ImplicitMFScorer, ItemKNNScorer, and PopScorer**. The code suggests `ImplicitMFScorer` is the ALS-like model used in LensKit, but the output does not explicitly label it as ALS.
- Because of the truncation, a complete statistical analysis of seed sensitivity across all datasets and algorithms cannot be reported from the provided materials alone.

## Conclusion

Based on the available output, the experiment successfully ran a seed-sensitivity study with 5 seeds, 5-core filtering, implicit conversion for MovieLens100K, and user-based holdout splitting. The only visible results are for **MovieLens100K with PopScorer**, and those results show **no variation across seeds** for the reported nDCG and Precision values. A full answer for all datasets and algorithms is not possible from the truncated output provided.