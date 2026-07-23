# Experiment Summary

## User Request

The user wanted to quantify how much data split random seeds affect recommender system accuracy. The request specified:

- **Libraries**: LensKit
- **Algorithms**: ALS (ImplicitMFScorer), ItemKNN, and Pop (PopScorer)
- **Datasets**: MovieLens100K, Amazon Video Games, Last.FM
- **Preprocessing**: 5-core filtering; for MovieLens and Amazon, convert ratings > 3 to implicit interactions
- **Procedure**: 5 random seeds (0–4), user-based 80/20 holdout split, standard hyperparameters
- **Metrics**: nDCG@k and Precision@k for k = 1, 5, 10, plus statistical analysis

## What Was Run

The experiment was executed exactly as specified. Key details from the code:

- **Datasets**: MovieLens100K, Amazon2014VideoGames, HetrecLastFM
- **Algorithms**: `PopScorer`, `ItemKNNScorer` (feedback="implicit"), `ImplicitMFScorer` (ALS)
- **Seeds**: 0, 1, 2, 3, 4
- **Split**: `sample_users` with `SampleFrac(0.2)` (user-based 80/20 holdout)
- **Preprocessing**: Ratings > 3.0 converted to implicit (set to 1) for MovieLens and Amazon; 5-core filtering applied to all datasets
- **Metrics**: NDCG@1,5,10 and Precision@1,5,10, computed via LensKit's `RunAnalysis`
- **Results**: Mean ± standard deviation (with ddof=1) across the 5 seeds for each (dataset, algorithm, metric, k) combination

**Note**: The Last.FM dataset (HetrecLastFM) does not appear in the final results output. The output shows results only for MovieLens100K and Amazon2014VideoGames. The Last.FM results were either not collected or failed silently during execution.

## Key Results

### Amazon2014VideoGames

| Algorithm | Metric | Mean ± Std (n=5 seeds) |
|---|---|---|
| **ImplicitMFScorer** | NDCG@1 | 0.044604 ± 0.008328 |
| | NDCG@5 | 0.050172 ± 0.006665 |
| | NDCG@10 | 0.058560 ± 0.006202 |
| | Precision@1 | 0.044604 ± 0.008328 |
| | Precision@5 | 0.030647 ± 0.003198 |
| | Precision@10 | 0.024504 ± 0.001364 |
| **ItemKNNScorer** | NDCG@1 | 0.032230 ± 0.004969 |
| | NDCG@5 | 0.038353 ± 0.002752 |
| | NDCG@10 | 0.043734 ± 0.002175 |
| | Precision@1 | 0.032230 ± 0.004969 |
| | Precision@5 | 0.023079 ± 0.001140 |
| | Precision@10 | 0.018216 ± 0.000658 |
| **PopScorer** | NDCG@1 | 0.018129 ± 0.003152 |
| | NDCG@5 | 0.017461 ± 0.002154 |
| | NDCG@10 | 0.020046 ± 0.001991 |
| | Precision@1 | 0.018129 ± 0.003152 |
| | Precision@5 | 0.010590 ± 0.000752 |
| | Precision@10 | 0.008201 ± 0.000719 |

### MovieLens100K

| Algorithm | Metric | Mean ± Std (n=5 seeds) |
|---|---|---|
| **ImplicitMFScorer** | NDCG@1 | 0.165775 ± 0.017328 |
| | NDCG@5 | 0.177744 ± 0.007972 |
| | NDCG@10 | 0.194821 ± 0.007883 |
| | Precision@1 | 0.165775 ± 0.017328 |
| | Precision@5 | 0.158289 ± 0.008948 |
| | Precision@10 | 0.143102 ± 0.008824 |
| **ItemKNNScorer** | NDCG@1 | 0.295187 ± 0.030813 |
| | NDCG@5 | 0.249975 ± 0.025074 |
| | NDCG@10 | 0.246947 ± 0.019492 |
| | Precision@1 | 0.295187 ± 0.030813 |
| | Precision@5 | 0.219465 ± 0.019033 |
| | Precision@10 | 0.177112 ± 0.012886 |
| **PopScorer** | NDCG@1 | 0.206417 ± 0.028946 |
| | NDCG@5 | 0.159509 ± 0.020684 |
| | NDCG@10 | 0.155021 ± 0.014116 |
| | Precision@1 | 0.206417 ± 0.028946 |
| | Precision@5 | 0.141818 ± 0.019438 |
| | Precision@10 | 0.117968 ± 0.010605 |

### Statistical Observations

- **Standard deviations** (the "±" values) quantify the variability due to different data split random seeds. Across all metrics and algorithms, standard deviations range from roughly **0.0007 to 0.031**.
- **Relative variability** (std / mean) is generally higher on the Amazon dataset (e.g., PopScorer NDCG@1: 0.003152 / 0.018129 ≈ 17%) compared to MovieLens100K (e.g., PopScorer NDCG@1: 0.028946 / 0.206417 ≈ 14%), suggesting seed effects are proportionally larger on the sparser/noisier Amazon dataset.
- **ItemKNNScorer** on MovieLens100K shows the largest absolute variability (e.g., NDCG@1 std = 0.030813), indicating this algorithm-dataset combination is most sensitive to the split seed.
- **PopScorer** on Amazon shows the smallest absolute variability (e.g., Precision@10 std = 0.000719), but this is partly because its absolute performance is very low.

## Limitations

1. **Last.FM results are missing**: The final output contains results only for MovieLens100K and Amazon2014VideoGames. The HetrecLastFM dataset was loaded and preprocessed in the per-seed runs, but its results do not appear in the aggregated analysis. This may be due to a silent failure or an issue with how the Last.FM data was handled during evaluation.

2. **No formal significance test**: The analysis reports mean and standard deviation across seeds but does not include a formal statistical test (e.g., ANOVA, paired t-test) to determine whether seed effects are statistically significant.

3. **Standard hyperparameters**: The code uses default parameters for all algorithms (e.g., `ImplicitMFScorer()` with no explicit configuration). The results reflect these defaults, not tuned hyperparameters.

4. **Single train/test split per seed**: Each seed produces one specific split. The variability measured is across different random splits, not across multiple runs of the same split.

## Conclusion

The experiment successfully quantified the effect of data split random seeds on recommender accuracy for two of the three requested datasets (MovieLens100K and Amazon2014VideoGames). Across 5 random seeds, standard deviations ranged from approximately 0.0007 to 0.031 depending on the algorithm, dataset, and metric. ItemKNN on MovieLens100K showed the highest sensitivity to the split seed (NDCG@1 std ≈ 0.031), while PopScorer on Amazon showed the lowest absolute variability. The relative impact of seed choice was larger on the Amazon dataset, likely due to its sparser interaction patterns. Results for the Last.FM dataset were not captured in the final output and would need to be re-run to complete the full analysis.