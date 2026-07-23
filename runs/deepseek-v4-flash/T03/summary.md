# Experiment Summary

## User Request

The user wanted to run an experiment to quantify how much data split random seeds affect recommender system accuracy. The request specified:

- **Algorithms**: ALS (ImplicitMF), ItemKNN, and Pop
- **Datasets**: MovieLens100K, Amazon Video Games 2014, and HetRec LastFM
- **Preprocessing**: 5-core filtering; for MovieLens and Amazon, convert ratings > 3 to implicit feedback
- **Procedure**: 5 random seeds for data splitting, user-based 80/20 holdout, standard hyperparameters
- **Evaluation**: nDCG@k and Precision@k for k = 1, 5, 10, plus statistical analysis

## What Was Run

The experiment code implemented the full procedure as described. However, the experiment **crashed during the first dataset (MovieLens100K)** because the required data file (`data/ml-100k.zip`) was not found on disk. The error occurred at the `load_movielens_df` call inside `load_and_preprocess_movielens()`. No results were produced for any algorithm, dataset, or seed.

## Key Results

| Algorithm | Dataset | Metric | Mean | Std | Min | Max |
|-----------|---------|--------|------|-----|-----|-----|
| N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**No results were collected.** The experiment failed before any evaluation could take place. The output shows only the error traceback and a crash after approximately 4 seconds of runtime.

## Limitations

- **Missing data file**: The MovieLens 100K dataset (`data/ml-100k.zip`) was not present at the expected path. The code does not include automatic download logic, so the experiment could not proceed.
- **Cascading failure**: Since the experiment loops over datasets sequentially and MovieLens100K was the first dataset, none of the three datasets were processed. Results for Amazon Video Games and LastFM are also unavailable.
- **No partial results**: The code does not checkpoint or save intermediate results, so even if later datasets were reachable, they were never attempted.

## Conclusion

The experiment could not be executed due to a missing input file (`data/ml-100k.zip`). Consequently, there is **no data** to quantify the impact of data split random seeds on recommender system accuracy. To complete this experiment, the MovieLens 100K dataset must be downloaded and placed at the expected location (`data/ml-100k.zip`), or the code must be modified to download it automatically. Once the data is available, the experiment can be re-run to produce the desired metrics and statistical analysis.