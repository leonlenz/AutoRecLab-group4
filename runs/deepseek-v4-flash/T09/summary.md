# Experiment Summary

## User Request

The user wanted to quantify how much data split random seeds affect recommender system accuracy. The request specified:

- **Algorithms**: ALS (ImplicitMF), ItemKNN, and Pop (Popularity)
- **Datasets**: MovieLens100K, Amazon Video Games, Last.FM (HetrecLastFM)
- **Preprocessing**: 5-core filtering; for MovieLens and Amazon, convert ratings > 3 to implicit interactions
- **Procedure**: 5 different random seeds for data splitting, user-based 80/20 holdout, standard hyperparameters
- **Metrics**: nDCG@k and Precision@k for k = 1, 5, 10, with statistical analysis

## What Was Run

The code implemented the requested experiment using **OmniRec** (not LensKit directly, but via the `LensKit` runner within OmniRec). The experiment looped over 5 seeds (42, 123, 456, 789, 1111) and for each seed:

1. Preprocessed MovieLens100K with `MakeImplicit(4)` (ratings ≥ 4 → implicit), `CorePruning(5)`, and `UserHoldout(test_size=0.2)`.
2. Preprocessed Amazon2014VideoGames with the same pipeline.
3. Preprocessed HetrecLastFM with `CorePruning(5)` and `UserHoldout(test_size=0.2)` (no implicit conversion, as it is already implicit).
4. Trained Pop, ItemKNN, and ImplicitMF (ALS) with `feedback="implicit"`.
5. Evaluated with NDCG and Precision at k = 1, 5, 10.

## Key Results

**The experiment crashed on the first seed (seed = 42) during MovieLens100K preprocessing.** The error occurred at the `UserHoldout` step with `validation_size=0.0`. The `train_test_split` function from scikit-learn rejected `test_size=0.0` as invalid — it requires a float strictly in (0.0, 1.0) or an integer ≥ 1. Since `validation_size=0.0` was passed, the split step failed before any training or evaluation could take place.

**No results were produced.** The following table reflects the available data:

| Dataset | Algorithm | Metric | Mean | Std | N |
|---|---|---|---|---|---|
| MovieLens100K | — | — | N/A | N/A | 0 |
| Amazon2014VideoGames | — | — | N/A | N/A | 0 |
| HetrecLastFM | — | — | N/A | N/A | 0 |

All values are N/A because the experiment terminated before any evaluation could be performed.

## Limitations

- **Experiment did not complete.** The crash on the first seed prevented all training and evaluation across all datasets and algorithms.
- **Root cause:** The `UserHoldout` step was configured with `validation_size=0.0`, which is not a valid parameter for scikit-learn's `train_test_split`. The intended 80/20 split (no validation set) should use `validation_size=None` or omit the validation split entirely, rather than setting it to 0.0.
- **No statistical analysis possible.** Since no metric values were recorded, there is no data to analyze the effect of random seeds on accuracy.
- **OmniRec vs. LensKit.** The code used OmniRec's LensKit runner rather than LensKit directly. This is a valid approach but may introduce differences from a pure LensKit implementation.

## Conclusion

The experiment could not be completed due to a parameter validation error in the data splitting step. Specifically, `UserHoldout(validation_size=0.0, test_size=0.2)` passed `0.0` as the validation size, which scikit-learn's `train_test_split` does not accept. To fix this, the `validation_size` parameter should be set to `None` (or the split step should be configured to produce only a train/test split without a validation partition). Once corrected, the experiment would need to be re-run from scratch to obtain the desired results.