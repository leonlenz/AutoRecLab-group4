# Experiment Summary

## User Request

The user wanted to quantify how much data split random seeds affect recommender system accuracy. The experiment was designed to test three algorithms (ALS, ItemKNN, Pop) on three datasets (MovieLens100K, Amazon Video Games, Last.FM) using 5 different random seeds for data splitting, with 5-core filtering, implicit feedback conversion (for MovieLens and Amazon), user-based 80/20 holdout splits, and evaluation using nDCG@k and Precision@k for k=1, 5, 10.

## What Was Run

The experiment script was executed but **crashed before producing any results**. The failure occurred at the very first dataset loading step:

- **Seed**: 42
- **Dataset**: MovieLens100K
- **Error**: `FileNotFoundError` — the MovieLens 100K dataset file (`ml-100k.zip`) was not found at the expected path (`/home/prv_tristan/AutoRecLab-group4/out/run_02/workspace/data/ml-100k.zip`).
- The program terminated after approximately 18 seconds with no results collected for any algorithm, dataset, or seed.

No training, splitting, recommendation, or evaluation was performed. No metrics (nDCG or Precision) were computed.

## Key Results

| Dataset | Algorithm | Metric | k | Mean | Std |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | N/A | N/A |

**No results are available.** The experiment failed entirely due to a missing input file. The code logic for all subsequent steps (implicit conversion, 5-core filtering, splitting, training, recommendation, evaluation, and statistical analysis) was never reached.

## Limitations

- **Missing data file**: The MovieLens 100K dataset (`ml-100k.zip`) was not present in the expected directory. This is a prerequisite that was not satisfied.
- **No fallback or download logic**: The code uses LensKit's `load_movielens()` function, which expects the file to already exist locally. There is no automatic download or alternative path handling.
- **Cascading failure**: Because the script crashes on the first dataset, none of the other datasets (Amazon Video Games, Last.FM) or any seeds beyond 42 were attempted.
- **No partial results**: The output contains only the error traceback; no metrics, statistics, or analysis were produced.

## Conclusion

The experiment could not be executed due to a missing input file (`ml-100k.zip`). As a result, **no conclusions can be drawn** about the impact of data split random seeds on recommender system accuracy. To proceed, the MovieLens 100K dataset must be placed at the expected path, or the code must be modified to download it automatically or point to the correct location. Once the data is available, the experiment can be re-run to generate the intended results.