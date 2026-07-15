import os

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Research sketch and implementation plan:
# 1. Load MovieLens100K, Amazon2014VideoGames, and HetrecLastFM via OmniRec built-in loaders.
# 2. Apply preprocessing in a fixed order: 5-core filtering; for MovieLens100K and Amazon2014VideoGames, convert ratings > 3 to implicit; then user-based 80/20 holdout split.
# 3. Repeat the split with five distinct random seeds, varying only the seed.
# 4. Evaluate three LensKit algorithms exposed through OmniRec: PopScorer, ItemKNNScorer, and ImplicitMFScorer (ALS).
# 5. Use default hyperparameters only; no tuning or EDA.
# 6. Compute NDCG@1,5,10 and Precision@1,5,10 on the held-out test set only.
# 7. Aggregate across seeds and report mean, std, and a short seed-sensitivity summary.

print(f'Working directory: {working_dir}')
print('Planned datasets: MovieLens100K, Amazon2014VideoGames, HetrecLastFM')
print('Planned algorithms: LensKit.PopScorer, LensKit.ItemKNNScorer, LensKit.ImplicitMFScorer')
