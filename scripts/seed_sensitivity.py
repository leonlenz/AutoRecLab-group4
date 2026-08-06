# -*- coding: utf-8 -*-
"""Seed-sensitivity numbers reported in Section 'Scientific output'.

Two things are computed here:

1. A re-implementation of the ratio that run C01 reports, to document what its
   number actually is:

       ratio(dataset) = mean over the six metrics of
                          mean_a  std_s  m(d, a, s)
                        ------------------------------------
                        mean_{a<a'} | mean_s m(d,a,s) - mean_s m(d,a',s) |

   std_s is the sample standard deviation (ddof=1) over the seeds that are
   present for that (dataset, algorithm); missing conditions are dropped, they
   are not imputed. This reproduces C01's printed 0.030 / 0.093 / 0.095.

2. A relative seed-induced variation that *is* comparable to the human-led
   baseline study, which reports accuracy shifts of up to ~6.3%:

       relative range = (max_s m - min_s m) / mean_s m   per (dataset, algorithm)

Inputs are parsed from the archived checkpoint logs; nothing is hard-coded.
"""
import itertools
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C01_NODE = next((ROOT / "codexRuns/run_01/checkpoint").glob("7376cfbe*")) / "out.log"
T01_NODE = next((ROOT / "tristanRuns/run_01/checkpoint").glob("6637738e*")) / "out.log"
METRICS = ["ndcg@1", "ndcg@5", "ndcg@10", "precision@1", "precision@5", "precision@10"]


def parse_c01():
    """Rows of the '===== Per-Run Results =====' block of C01's reported node."""
    text = C01_NODE.read_text(errors="replace")
    block = text.split("===== Per-Run Results")[1].split("=====")[1]
    out = {}
    for line in block.splitlines():
        p = line.split()
        if len(p) == 9 and re.match(r"^\d+$", p[1]):
            ds, seed, algo = p[0], int(p[1]), p[2]
            for name, val in zip(METRICS, map(float, p[3:])):
                out[(ds, algo, seed, name)] = val
    return out


def parse_t01():
    text = T01_NODE.read_text(errors="replace")
    cur = algo = None
    out = {}
    for line in text.splitlines():
        m = re.search(r"--- Processing (\w+) with seed (\d+) ---", line)
        if m:
            cur, algo = (m.group(1), int(m.group(2))), None
            continue
        m = re.search(r"--- Algorithm: (\w+) ---", line)
        if m:
            algo = m.group(1).replace("Scorer", "")
            continue
        m = re.search(r"(NDCG|Precision)@(\d+)=([\d.]+)", line)
        if m and cur and algo:
            out[(cur[0], algo, cur[1],
                 f"{m.group(1).lower()}@{m.group(2)}")] = float(m.group(3))
    return out


def ratio_per_dataset(rows):
    datasets = sorted({d for d, _, _, _ in rows})
    algos = sorted({a for _, a, _, _ in rows})
    result = {}
    for d in datasets:
        per_metric = []
        for metric in METRICS:
            stds, means = [], {}
            for a in algos:
                vals = [v for (dd, aa, _, mm), v in rows.items()
                        if dd == d and aa == a and mm == metric]
                if len(vals) >= 2:
                    stds.append(statistics.stdev(vals))
                if vals:
                    means[a] = statistics.fmean(vals)
            gaps = [abs(means[x] - means[y])
                    for x, y in itertools.combinations(sorted(means), 2)]
            if stds and gaps:
                per_metric.append(statistics.fmean(stds) / statistics.fmean(gaps))
        result[d] = statistics.fmean(per_metric) if per_metric else float("nan")
    return result


def relative_variation(rows, metric="ndcg@10"):
    out = {}
    for d in sorted({d for d, _, _, _ in rows}):
        for a in sorted({a for _, aa, _, _ in rows for a in [aa]}):
            vals = [v for (dd, aa, _, mm), v in rows.items()
                    if dd == d and aa == a and mm == metric]
            if len(vals) >= 2:
                mean = statistics.fmean(vals)
                out[(d, a)] = dict(n=len(vals),
                                   rel_range=(max(vals) - min(vals)) / mean,
                                   cv=statistics.stdev(vals) / mean)
    return out


def report(name, rows):
    print(f"\n=== {name} ===")
    print("seeds:", sorted({s for _, _, s, _ in rows}),
          "| conditions:", len({(d, a, s) for d, a, s, _ in rows}))
    print("seed-std / algorithm-gap ratio per dataset:")
    for d, r in ratio_per_dataset(rows).items():
        print(f"  {d:22} {r:.3f}")
    print("relative seed-induced variation of nDCG@10:")
    rv = relative_variation(rows)
    for (d, a), v in sorted(rv.items()):
        print(f"  {d:22} {a:8} n={v['n']}  range/mean={100*v['rel_range']:5.1f}%  "
              f"cv={100*v['cv']:5.1f}%")
    rr = [v["rel_range"] for v in rv.values()]
    cv = [v["cv"] for v in rv.values()]
    print(f"  -> range/mean {100*min(rr):.1f}%-{100*max(rr):.1f}% "
          f"(median {100*statistics.median(rr):.1f}%), "
          f"cv median {100*statistics.median(cv):.1f}%")


if __name__ == "__main__":
    report("C01 (cross-model Codex, reported node position 3)", parse_c01())
    report("T01 (DeepSeek, reported node position 6)", parse_t01())
