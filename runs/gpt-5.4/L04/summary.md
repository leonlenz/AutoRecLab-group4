# Experiment Summary

## User Request

You asked for a LensKit experiment to quantify how much random data-split seeds affect recommender accuracy for three algorithms—ALS, ItemKNN, and Pop—on three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM.

Requested setup:

- 5-core preprocessing on all datasets
- Convert ratings greater than 3 to implicit interactions for MovieLens and Amazon
- 5 random seeds for user-based 80/20 holdout splitting
- Standard hyperparameters
- Evaluate with nDCG@1/5/10 and Precision@1/5/10
- Provide a short statistical analysis

## What Was Run

From the code and output, the experiment did the following:

- Algorithms:
  - `LensKit.ImplicitMFScorer` → reported as **ALS**
  - `LensKit.ItemKNNScorer` → **ItemKNN**
  - `LensKit.PopScorer` → **Pop**
- Seeds: `11, 22, 33, 44, 55`
- Split:
  - Exact **user-based 80/20 holdout**
  - At least 1 test interaction per eligible user
  - Empty validation set
- Metrics:
  - **NDCG** at `k = 1, 5, 10`
  - **Precision** at `k = 1, 5, 10`
- Preprocessing:
  - MovieLens100K: made implicit with threshold 3, then 5-core pruning
  - Amazon2014VideoGames: intended to be made implicit with threshold 3, then 5-core pruning
  - HetrecLastFM: 5-core pruning only
- Statistical summary computed in code:
  - mean across seeds
  - standard deviation
  - coefficient of variation
  - min/max/range
  - 95% confidence interval
  - simple consecutive-seed delta analysis
  - per-seed cross-algorithm ranking

Observed preprocessing details from output for MovieLens100K:

- Interactions before implicit conversion: `100000`
- After implicit conversion: `82520`
- After 5-core pruning: `81697`

## Key Results

The provided output is incomplete. It contains clear aggregated results for **MovieLens100K**, and only partial/ambiguous evidence for **HetrecLastFM**. No usable aggregated table for **Amazon2014VideoGames** is visible in the provided output.

### Compact results table

| Dataset | Algorithm | Metric@k | Mean | Std | Range | Notes |
|---|---:|---:|---:|---:|---:|---|
| MovieLens100K | ItemKNN | NDCG@1 | 0.382397 | 0.015298 | 0.037116 | Best shown for NDCG@1 |
| MovieLens100K | Pop | NDCG@1 | 0.254436 | 0.026588 | 0.111347 | Most seed-sensitive on MovieLens100K |
| MovieLens100K | ALS | NDCG@1 | 0.215058 | 0.013252 | 0.028632 | Lowest of the three at NDCG@1 |
| MovieLens100K | ItemKNN | NDCG@5 | 0.293755 | 0.012882 | 0.032567 | Best shown for NDCG@5 |
| MovieLens100K | ALS | NDCG@5 | 0.204354 | 0.003299 | 0.007331 | Lower variability than ItemKNN/Pop |
| MovieLens100K | Pop | NDCG@5 | 0.182637 | 0.009561 | 0.034165 |  |
| MovieLens100K | ItemKNN | NDCG@10 | 0.254925 | 0.012709 | 0.030943 | Best shown for NDCG@10 |
| MovieLens100K | ALS | NDCG@10 | 0.191167 | 0.001271 | 0.003590 | Very stable across seeds |
| MovieLens100K | Pop | NDCG@10 | 0.167987 | 0.009334 | 0.031926 |  |
| MovieLens100K | ItemKNN | Precision@1 | 0.382397 | 0.015298 | 0.037116 | Same numeric value as NDCG@1 in output |
| MovieLens100K | Pop | Precision@1 | 0.254436 | 0.026588 | 0.111347 | Most seed-sensitive metric on MovieLens100K |
| MovieLens100K | ALS | Precision@1 | 0.215058 | 0.013252 | 0.028632 |  |
| MovieLens100K | ItemKNN | Precision@5 | 0.271417 | 0.014439 | 0.033934 | Best shown for Precision@5 |
| MovieLens100K | ALS | Precision@5 | 0.200848 | 0.005768 | 0.014422 |  |
| MovieLens100K | Pop | Precision@5 | 0.165048 | 0.008489 | 0.024390 |  |
| MovieLens100K | ItemKNN | Precision@10 | 0.226744 | 0.013640 | 0.034358 | Best shown for Precision@10 |
| MovieLens100K | ALS | Precision@10 | 0.183521 | 0.002615 | 0.007317 |  |
| MovieLens100K | Pop | Precision@10 | 0.152810 | 0.008403 | 0.020148 |  |
| HetrecLastFM | ItemKNN | NDCG@1 | N/A | 0.171644 | 0.363542 | Output explicitly says this was the most seed-sensitive on HetrecLastFM |
| Amazon2014VideoGames | N/A | N/A | N/A | N/A | N/A | No readable aggregated results in provided output |

### Short statistical analysis

Based strictly on the visible output:

- **MovieLens100K**
  - **ItemKNN** is best on every reported metric and cutoff shown.
  - **ALS** is generally more stable across seeds than ItemKNN and Pop, especially at larger cutoffs:
    - NDCG@10 std = `0.001271`
    - Precision@10 std = `0.002615`
  - **Pop** shows the greatest seed sensitivity on MovieLens100K:
    - most seed-sensitive metric reported: **Pop NDCG@1**
    - std = `0.026588`
    - range = `0.111347`
- **HetrecLastFM**
  - The output explicitly reports the most seed-sensitive case:
    - **ItemKNN NDCG@1**
    - std = `0.171644`
    - range = `0.363542`
  - However, the full HetrecLastFM aggregate table is not present, so broader comparison is not possible from the supplied text.
- **Amazon2014VideoGames**
  - No visible aggregate results were provided, so no factual statistical interpretation can be made.

## Limitations

- The experiment output is **truncated**, so the results are not complete.
- Full aggregated results are only clearly available for **MovieLens100K**.
- For **HetrecLastFM**, only a partial snippet is visible; one explicit seed-sensitivity statement can be extracted, but not the full metric table.
- For **Amazon2014VideoGames**, no readable results are present in the provided output.
- Because of these missing results, I cannot give a complete three-dataset comparison without guessing, so I have not done so.

## Conclusion

The code matches your requested design closely: LensKit ALS, ItemKNN, and Pop were evaluated under five random user-based 80/20 holdout seeds after the requested preprocessing, using nDCG and Precision at 1, 5, and 10.

From the visible results:

- On **MovieLens100K**, **ItemKNN** achieved the best accuracy across all shown metrics.
- **Seed choice does affect measured accuracy**, with the strongest visible MovieLens effect occurring for **Pop at NDCG@1** (std `0.026588`, range `0.111347`).
- **ALS** appears comparatively robust to split-seed variation on MovieLens100K, especially at larger cutoffs.
- On **HetrecLastFM**, the visible output indicates much larger seed sensitivity for **ItemKNN NDCG@1** (std `0.171644`, range `0.363542`) than anything shown for MovieLens100K.
- A complete conclusion across all three datasets is not possible because the provided output does not include the full results for **Amazon2014VideoGames** and only partial results for **HetrecLastFM**.