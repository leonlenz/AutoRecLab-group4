# Experiment Summary

## User Request

Quantify how much random data-split seeds affect recommender accuracy in LensKit for three algorithms:

- ALS
- ItemKNN
- Pop

on three implicit-feedback datasets:

- MovieLens100K
- Amazon Video Games
- Last.FM

with:

- 5-core preprocessing on all datasets
- ratings converted to implicit for MovieLens and Amazon using the rule “ratings greater than 3”
- 5 random seeds for user-based 80/20 holdout splits
- evaluation with nDCG@1/5/10 and Precision@1/5/10
- a short statistical analysis of seed sensitivity

## What Was Run

From the code, the experiment did the following:

- Datasets used:
  - `MovieLens100K`
  - `Amazon2014VideoGames`
  - `HetrecLastFM`
- Preprocessing:
  - MovieLens100K and Amazon Video Games:
    - `MakeImplicit(4)` → keeps ratings `>= 4`, matching “ratings > 3”
    - then `CorePruning(5)`
  - HetrecLastFM:
    - `MakeImplicit(1)` and `CorePruning(5)`
- Split seeds:
  - `2027, 3109, 4513, 7127, 9901`
- Split method:
  - user-based random 80/20 holdout
  - per user, test size was rounded to 20%, with at least 1 test interaction and at least 1 train interaction
- Algorithms:
  - Pop (`PopScorer`)
  - ItemKNN (`ItemKNNScorer`)
  - ALS (`ImplicitMFScorer`)
- Evaluation metrics:
  - nDCG@1, nDCG@5, nDCG@10
  - Precision@1, Precision@5, Precision@10
- Analysis:
  - for each dataset and metric, the code computed:
    - mean standard deviation across seeds (averaged over algorithms)
    - mean gap between algorithms
    - ratio: `seed std / algorithm gap`
  - then it summarized each dataset as low/moderate/high seed sensitivity

Higher nDCG and Precision indicate better top-k ranking quality.

## Key Results

The provided output clearly reports the descriptive seed-sensitivity conclusion, but it does **not** include the full per-algorithm accuracy table because the log excerpt is truncated.

| Dataset | Visible preprocessing evidence | Exact nDCG/Precision values | Avg. seed-std / algo-gap ratio | Reported interpretation |
|---|---|---:|---:|---|
| Amazon2014VideoGames | 1,324,753 interactions before implicit; 970,030 after implicit; 132,209 after 5-core | N/A | 0.030 | Low seed sensitivity relative to algorithm gaps |
| HetrecLastFM | 71,064 interactions before implicit; 71,064 after implicit | N/A | 0.093 | Low seed sensitivity relative to algorithm gaps |
| MovieLens100K | Preprocessing defined in code, but counts not visible in excerpt | N/A | 0.095 | Low seed sensitivity relative to algorithm gaps |

Additional factual observations from the output:

- Runtime was about **51 minutes**.
- Output files were saved to:
  - `per_run_results_dataset_algorithm_seed.csv`
  - `seed_variation_summary_mean_std.csv`
  - `seed_sensitivity_vs_algorithm_gap.csv`
  - `split_seeds.json`

These filenames indicate that the exact per-seed and per-algorithm metric values were written to disk, even though they are not visible in the provided log excerpt.

## Limitations

- The experiment log is explicitly **truncated**, so the exact nDCG@k and Precision@k scores for each dataset/algorithm/seed are **not visible** here.
- Because of that truncation, I cannot truthfully report:
  - which algorithm had the best mean accuracy on each dataset
  - the exact mean/std values for ALS, ItemKNN, and Pop
  - the full per-metric table for all conditions
- The statistical analysis implemented in the code is **descriptive**, not inferential:
  - it compares seed variation to between-algorithm gaps
  - it does **not** report p-values, confidence intervals, or formal significance tests
- The visible excerpt does not explicitly show the final “all 45 conditions completed” line, so I should not claim that completion status with certainty from the excerpt alone.

## Conclusion

Based on the experiment output that is visible, random split seeds had a **small effect on measured recommender accuracy relative to the differences between ALS, ItemKNN, and Pop** on all three datasets.

Reported average seed-sensitivity ratios were:

- **Amazon2014VideoGames:** 0.030
- **HetrecLastFM:** 0.093
- **MovieLens100K:** 0.095

Using the experiment’s own interpretation thresholds, all three datasets show **low seed sensitivity relative to algorithm gaps**. In practical terms, within this setup, changing the holdout random seed affected results much less than changing the recommendation algorithm.

However, the exact nDCG@1/5/10 and Precision@1/5/10 values are not recoverable from the provided truncated output, so a complete accuracy ranking cannot be reported from this excerpt alone.