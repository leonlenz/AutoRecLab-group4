# Experiment Summary

## User Request

Run a LensKit experiment to compare how random data-split seeds affect recommender accuracy for three algorithms (ALS, ItemKNN, Pop) on three implicit-feedback datasets (MovieLens100K, Amazon Video Games, Last.FM), using 5-core filtering, 5 random seeds, user-based 80/20 holdout splits, standard hyperparameters, and reporting nDCG@k and Precision@k for k = 1, 5, 10 with a short statistical analysis.

## What Was Run

The provided code indicates the following planned procedure:

- Datasets:
  - MovieLens100K
  - Amazon2014VideoGames
  - HetrecLastFM
- Preprocessing:
  - 5-core filtering on all datasets
  - For MovieLens100K and Amazon2014VideoGames, ratings greater than 3 were to be converted to implicit interactions
- Splitting:
  - User-based 80/20 holdout split
  - Repeated for 5 distinct random seeds
- Algorithms:
  - `PopScorer`
  - `ItemKNNScorer`
  - `ImplicitMFScorer` (ALS)
- Evaluation:
  - nDCG@1, nDCG@5, nDCG@10
  - Precision@1, Precision@5, Precision@10
- Aggregation:
  - Mean and standard deviation across seeds

The output only confirms that the script started and lists the planned datasets and algorithms. It does not include any model training logs, metric values, or statistical summaries.

## Key Results

| Dataset | Algorithm | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 | Seed sensitivity / stats |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MovieLens100K | ALS | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | Pop | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | ALS | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | Pop | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | ALS | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | Pop | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

No metric values were present in the provided output, so exact results cannot be reported.

## Limitations

- The execution output is incomplete: it only shows the working directory, planned datasets, planned algorithms, and that execution took “a moment.”
- No evidence of actual training, splitting, evaluation, or aggregation results is included.
- Because no numeric outputs were provided, no statistical analysis of seed sensitivity can be performed from the available materials.
- The documentation confirms that Precision and NDCG are valid LensKit ranking metrics, but it does not supply any experiment-specific results.

## Conclusion

The experiment was configured to run the requested LensKit comparison with 5-core filtering, 5 random seeds, 80/20 user holdout, and evaluation using nDCG@1/5/10 and Precision@1/5/10. However, the provided output does not contain any measured results, so the effect of random split seeds on accuracy cannot be quantified from this run alone.