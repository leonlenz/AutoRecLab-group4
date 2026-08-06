# -*- coding: utf-8 -*-
"""Independent control computation for one AutoRecLab condition.

This script deliberately shares no code with AutoRecLab, OmniRec or LensKit.
It re-implements, from the raw MovieLens 100K archive:

  1. the implicit-feedback conversion (rating > 3 -> 1),
  2. iterative 5-core filtering,
  3. a user-based 80/20 holdout (20% of users are test users, 20% of each test
     user's interactions are held out),
  4. a most-popular recommender and Precision@10 / nDCG@10 over the held-out
     items,

and compares the results with the values the generated experiments reported.

Relevance definition: an item is relevant for a test user iff it is in that
user's held-out 20%. Users without held-out items cannot occur after 5-core
filtering (5 * 0.2 >= 1). Both metrics are normalised by the cutoff k, matching
the reported runs (their nDCG@1 equals their Precision@1 for every condition,
which only holds under that normalisation).

Usage:  python scripts/independent_metric_check.py
"""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
CACHE = Path(__file__).resolve().parent / "_ml100k_cache.zip"
OUT = Path(__file__).resolve().parent / "independent_check.json"

# Values the generated experiments logged for MovieLens 100K.
REPORTED_PREPROCESSING = {
    "raw_interactions": 100000,
    "after_implicit_conversion": 55375,
    "after_5_core": 54413,
    "users": 938,
    "items": 1008,
    "test_users": 187,
    "train_interactions": 52085,   # T01, node 6637738e
}
# nDCG@10 / Precision@10 for PopScorer on MovieLens 100K.
REPORTED_POP = {
    "T01 (own LensKit evaluation, 5 seeds)":
        dict(ndcg10=[0.152484, 0.144608, 0.179765, 0.148680, 0.149565],
             prec10=[0.122995, 0.108556, 0.134225, 0.112834, 0.111230],
             note="seeds 0-4, node 6637738e"),
    "C01 (OmniRec evaluator, 3 seeds)":
        dict(ndcg10=[0.130967, 0.129257, 0.134180],
             prec10=[0.112580, 0.113859, 0.118443],
             note="seeds 2027, 3109, 4513"),
}


def load_raw():
    if not CACHE.exists():
        with urllib.request.urlopen(URL, timeout=60) as r:
            CACHE.write_bytes(r.read())
    with zipfile.ZipFile(CACHE) as z:
        with z.open("ml-100k/u.data") as f:
            return pd.read_csv(io.BytesIO(f.read()), sep="\t",
                               names=["user", "item", "rating", "ts"])


def to_implicit(df, threshold=3.0):
    return df[df["rating"] > threshold].copy()


def k_core(df, k=5):
    while True:
        n = len(df)
        df = df[df["user"].map(df["user"].value_counts()) >= k]
        df = df[df["item"].map(df["item"].value_counts()) >= k]
        if len(df) == n:
            return df


def holdout(df, rng, user_frac=0.2, item_frac=0.2):
    users = np.sort(df["user"].unique())
    n_test = max(1, int(user_frac * len(users)))
    test_users = rng.choice(users, size=n_test, replace=False)
    test_rows, train_mask = [], np.ones(len(df), dtype=bool)
    pos = {u: i for i, u in enumerate(df.index)}
    for u in test_users:
        idx = df.index[df["user"].values == u].to_numpy()
        n_hold = max(1, int(round(item_frac * len(idx))))
        held = rng.choice(idx, size=n_hold, replace=False)
        test_rows.append((u, set(df.loc[held, "item"])))
        for h in held:
            train_mask[pos[h]] = False
    return df[train_mask], dict(test_rows), len(test_users)


def evaluate_pop(train, test, k=10):
    popularity = train["item"].value_counts()
    ranked = popularity.index.to_numpy()
    seen = train.groupby("user")["item"].apply(set).to_dict()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = discounts.cumsum()
    ndcgs, precs = [], []
    for user, relevant in test.items():
        known = seen.get(user, set())
        recs, i = [], 0
        while len(recs) < k and i < len(ranked):
            if ranked[i] not in known:
                recs.append(ranked[i])
            i += 1
        hits = np.array([1.0 if r in relevant else 0.0 for r in recs])
        dcg = float((hits * discounts[:len(hits)]).sum())
        idcg = ideal[min(len(relevant), k) - 1]
        ndcgs.append(dcg / idcg)
        precs.append(hits.sum() / k)
    return float(np.mean(ndcgs)), float(np.mean(precs))


def main():
    raw = load_raw()
    implicit = to_implicit(raw)
    core = k_core(implicit)
    pre = {
        "raw_interactions": len(raw),
        "after_implicit_conversion": len(implicit),
        "after_5_core": len(core),
        "users": core["user"].nunique(),
        "items": core["item"].nunique(),
        "test_users": max(1, int(0.2 * core["user"].nunique())),
    }
    print("Preprocessing (our control vs. the values the runs logged)")
    ok = True
    for key, reported in REPORTED_PREPROCESSING.items():
        if key not in pre:
            continue
        match = pre[key] == reported
        ok &= match
        print(f"  {key:28} control={pre[key]:>7}  reported={reported:>7}  "
              f"{'match' if match else 'MISMATCH'}")

    ndcgs, precs, trains = [], [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        train, test, n_test = holdout(core, rng)
        n, p = evaluate_pop(train, test)
        ndcgs.append(n)
        precs.append(p)
        trains.append(len(train))
    res = {
        "preprocessing_control": pre,
        "preprocessing_matches_logs": bool(ok),
        "train_interactions_control": [min(trains), max(trains)],
        "pop_ndcg10": {"mean": float(np.mean(ndcgs)), "std": float(np.std(ndcgs, ddof=1)),
                       "min": float(min(ndcgs)), "max": float(max(ndcgs))},
        "pop_precision10": {"mean": float(np.mean(precs)), "std": float(np.std(precs, ddof=1)),
                            "min": float(min(precs)), "max": float(max(precs))},
        "reported": REPORTED_POP,
    }
    print("\nIndependent most-popular baseline on MovieLens 100K "
          "(10 independently drawn 80/20 user holdouts)")
    print(f"  train interactions      {min(trains)}-{max(trains)} "
          f"(T01 logged {REPORTED_PREPROCESSING['train_interactions']})")
    print(f"  nDCG@10      {np.mean(ndcgs):.4f} +- {np.std(ndcgs, ddof=1):.4f} "
          f"[{min(ndcgs):.4f}, {max(ndcgs):.4f}]")
    print(f"  Precision@10 {np.mean(precs):.4f} +- {np.std(precs, ddof=1):.4f} "
          f"[{min(precs):.4f}, {max(precs):.4f}]")
    for label, rep in REPORTED_POP.items():
        vals = rep["ndcg10"]
        print(f"  reported nDCG@10 {label}: "
              f"{np.mean(vals):.4f} [{min(vals):.4f}, {max(vals):.4f}]")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwritten to {OUT.name}")


if __name__ == "__main__":
    main()
