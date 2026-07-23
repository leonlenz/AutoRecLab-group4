# Experiment Summary

## User Request

You asked for a LensKit-based seed-sensitivity experiment on three implicit-feedback datasets:

- MovieLens100K
- Amazon Video Games
- Last.FM

with:

- 5-core preprocessing
- ratings converted to implicit interactions for MovieLens and Amazon using ratings greater than 3
- 5 random split seeds: `7, 19, 42, 77, 123`
- user-based 80/20 holdout per seed
- algorithms: ALS, ItemKNN, Pop
- metrics: nDCG@k and Precision@k for `k = 1, 5, 10`
- a short statistical analysis

## What Was Run

Based on the code, the experiment implemented the following:

- Datasets loaded through OmniRec’s LensKit runner integration.
- Algorithms actually run:
  - `LensKit.ImplicitMFScorer` (used in the code as the ALS-style implicit matrix factorization model)
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`
- Preprocessing:
  - MovieLens100K: `RatingFilter(lower=4)` then 5-core pruning
  - Amazon2014VideoGames: `RatingFilter(lower=4)` then 5-core pruning
  - HetrecLastFM: 5-core pruning only
  - After filtering, rating columns were dropped to make the data implicit.
- Splitting:
  - custom user-based 80/20 holdout for each seed
- Evaluation:
  - nDCG@1, nDCG@5, nDCG@10 via OmniRec evaluator
  - Precision@1, @5, @10 was intended to be reconstructed from saved recommendation files if present

Observed preprocessing counts from the output:

- MovieLens100K: 55,375 interactions before 5-core pruning, 54,413 after
- Amazon2014VideoGames: 970,030 before 5-core pruning, 132,209 after
- HetrecLastFM: 71,064 before 5-core pruning, 52,551 after

## Key Results

The experiment did **not complete**. The run timed out during:

- `HetrecLastFM`, seed `42`
- specifically while running/evaluating `PopScorer`

Because of that, no final aggregated seed-sensitivity analysis, pairwise comparisons, or statistical analysis tables were produced in the provided output.

The only exact evaluation results visible in the output are for **HetrecLastFM, seed 19**, and only for **nDCG**. No Precision values are shown in the provided output.

| Dataset | Seed | Algorithm | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| HetrecLastFM | 19 | ImplicitMFScorer | 0.11834862385321102 | 0.08962090472663986 | 0.07968688131973249 | N/A | N/A | N/A |
| HetrecLastFM | 19 | ItemKNNScorer | 0.10366972477064221 | 0.08482044605189627 | 0.07494235485215657 | N/A | N/A | N/A |
| HetrecLastFM | 19 | PopScorer | 0.03577981651376147 | 0.03806353703915404 | 0.03609123515597756 | N/A | N/A | N/A |

Factual interpretation from the available results only:

- For **HetrecLastFM with seed 19**, `ImplicitMFScorer` had the highest nDCG at all reported cutoffs.
- `ItemKNNScorer` was second.
- `PopScorer` was clearly lowest on the shown nDCG values.
- No seed-effect quantification can be completed from the provided output because only one seed’s exact metric values are visible and the full experiment terminated early.

## Limitations

- The experiment output is incomplete due to a **timeout after one hour**.
- The run stopped at `HetrecLastFM` seed `42`, so the full 5-seed study was not completed.
- The requested **Precision@k** results are not present in the provided output.
- The requested **statistical analysis** cannot be reported from the provided materials because the experiment did not finish and no final analysis tables are shown.
- Results for MovieLens100K and Amazon2014VideoGames are not visible in the provided output excerpt, so their accuracy and seed sensitivity cannot be summarized factually.

## Conclusion

The code sets up the requested seed-sensitivity experiment correctly in broad terms, using OmniRec’s LensKit integration with:

- 5 seeds
- user-based 80/20 holdout
- `ImplicitMFScorer` / `ItemKNNScorer` / `PopScorer`
- nDCG evaluation, with an attempted Precision reconstruction

However, the provided run **did not finish**, so the requested cross-seed comparison and short statistical analysis cannot be answered from the available output.

From the only exact metrics shown, for **HetrecLastFM at seed 19**, the ranking was:

1. `ImplicitMFScorer`
2. `ItemKNNScorer`
3. `PopScorer`

but this is **not enough** to quantify how random split seeds affect recommender accuracy across datasets and algorithms.