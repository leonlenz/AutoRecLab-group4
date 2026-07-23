# Experiment Summary

## User Request

The request was to run a LensKit experiment comparing three algorithms — ALS, ItemKNN, and Pop — on three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM. The experiment should:

- apply 5-core filtering to all datasets,
- convert ratings greater than 3 to implicit interactions for MovieLens100K and Amazon Video Games,
- use 5 random split seeds,
- perform user-based 80/20 holdout splitting,
- train with standard hyperparameters,
- report nDCG@1/5/10 and Precision@1/5/10,
- and provide a short statistical analysis of seed sensitivity.

## What Was Run

The code did the following:

- Loaded datasets via OmniRec/LensKit.
- For MovieLens100K and Amazon Video Games, applied `MakeImplicit(3)` before pruning.
- Applied `CorePruning(5)` to all datasets.
- Built an experiment plan with three LensKit algorithms:
  - `LensKit.ImplicitMFScorer` (used in the code as the ALS-style model),
  - `LensKit.ItemKNNScorer`,
  - `LensKit.PopScorer`.
- Used five seeds: `11, 22, 33, 44, 55`.
- For each dataset and seed:
  - set the random state,
  - performed `UserHoldout(validation_size=0.01, test_size=0.20)`,
  - ran the experiment,
  - evaluated with `NDCG([1, 5, 10])` and `Precision([1, 5, 10])`.
- Summarized results across seeds using mean, standard deviation, and coefficient of variation.

## Key Results

The run did not complete successfully, so no final metric table is available from the output. The only concrete preprocessing results shown were for MovieLens100K. The experiment crashed later while processing Amazon Video Games.

| Dataset | Algorithm | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| MovieLens100K | ALS / ImplicitMFScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | Pop | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | ALS / ImplicitMFScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | Pop | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | ALS / ImplicitMFScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | Pop | N/A | N/A | N/A | N/A | N/A | N/A |

Observed preprocessing output for MovieLens100K:

- Interactions before implicit conversion: 100,000
- After `MakeImplicit(3)`: 82,520
- After 5-core pruning: 81,697

The log also shows that Amazon Video Games had begun running, and the crash occurred during the PopScorer run for Amazon2014VideoGames.

The short statistical analysis could not be produced from the provided output because the run crashed before any summary results were printed.

## Limitations

- The experiment output is incomplete and ends with: `Program crashed with exception (see above) after 53 minutes!`
- No final metric values (`nDCG@k`, `Precision@k`) are present in the provided output.
- No seed-level summary table is shown.
- No statistical analysis results are available.
- Although the code defines the procedure for all three datasets, the visible output only confirms preprocessing for MovieLens100K and partial execution for Amazon Video Games; there is no visible completion for Last.FM.

## Conclusion

The experiment was set up correctly to test seed sensitivity of LensKit baselines under 5-core filtering and user-based 80/20 holdout, with implicit conversion for MovieLens100K and Amazon Video Games. However, the run crashed before completion, so the provided output does not contain the requested accuracy metrics or a completed statistical analysis.