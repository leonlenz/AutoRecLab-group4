# -*- coding: utf-8 -*-
"""Build the machine-readable run overview (run_status.csv, node_status.csv).

Every field that can be derived from the archived artefacts is derived here, so
that Table 1 and the outcome figures are generated from one single source of
truth. Fields that rest on a manual code audit are kept in AUDIT below, each
with the evidence string that justifies it.

Status dimensions (see paper, Section "Defining run status"):
  C1 tree_search_completed  all 7 checkpoints present
  C2 evaluation_output      >=1 node completed >=1 recommender evaluation
  C3 statistical_summary    >=1 node ended without error and printed numeric
                            aggregate seed statistics
  C4 grid coverage          datasets_evaluated (0-3) and conditions_completed
                            (n/45) of the reported node
  C5 extraction_audited     result-extraction logic read and judged
  C6 scientific_status      usable / partially usable / not usable
"""
import csv, json, re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

ARMS = {
    "L": (ROOT / "allRuns" / "gpt-5.4", "gpt-5.4", "single-model"),
    "M": (ROOT / "allRuns" / "gpt-5.4-mini", "gpt-5.4-mini", "single-model"),
    "T": (ROOT / "tristanRuns", "deepseek-v4-flash", "single-model"),
    "C": (ROOT / "codexRuns", "gpt-5.3-codex + gpt-5.4", "cross-model"),
}
BRANCH = {"L": "develop", "M": "develop", "T": "feat/deepseek-compatibility",
          "C": "feat/enable-codex"}
# All arms build on the same develop commit; the two feature branches add a
# backend on top of it.
BASE_COMMIT = "5a1343d"

# Framework-emitted line: one per completed (dataset, algorithm) evaluation.
FRAMEWORK_DONE = re.compile(r"INFO\s+([A-Za-z0-9_]+)/([A-Za-z]+)Scorer done!")
# Generated code that evaluates without OmniRec's evaluator (T01 pattern).
OWN_COND = re.compile(r"--- Processing (\w+) with seed (\d+) ---")
OWN_ALGO = re.compile(r"--- Algorithm: (\w+?)(?:Scorer)? ---")
ERROR_TAIL = re.compile(r"Traceback|TimeoutError|Program crashed|MemoryError")
# Metric values actually printed (an evaluation table or a metric line with a
# number) -- completing an evaluation phase is not the same as reporting it.
METRICS_PRINTED = re.compile(r"(?i)evaluation results|ndcg@?\s*\d[^\n]{0,60}\d\.\d{3}")
# A numeric seed-aggregate: a std/sd column or a mean +- std line with numbers.
NUMERIC_STAT = re.compile(
    r"(?im)^(?!.*Traceback)(?:"
    r".*\b(?:std|sd)\s*=\s*\d"
    r"|.*±\s*\d\.\d+"
    r"|.*\bmean\b.*\b(?:std|sd)\b"
    r"|.*seed-induced range="
    r"|.*seed_std_over_algo_gap_ratio"
    r")")

DATASETS = {"MovieLens100K": "ML", "Amazon2014VideoGames": "AMZ",
            "HetrecLastFM": "LFM"}

# ---------------------------------------------------------------------------
# Manual-audit results. Each entry records what was read and what it showed.
# reported_node: the node whose output the paper reports for this run.
# ---------------------------------------------------------------------------
AUDIT = {
    "L04": dict(reported_node=4, extraction="incorrect", evidence_level=3,
                scientific="not usable",
                failure="cumulative-results mislabelling",
                audit_note="code.py stamps get_results() rows with the current "
                           "dataset/seed; Last.FM never evaluated yet reported"),
    "M02": dict(reported_node=None, extraction="incorrect", evidence_level=3,
                scientific="not usable",
                failure="cumulative-results mislabelling (degenerate std)",
                audit_note="split seed folded into the algorithm identifier, so "
                           "every group has n=1 and std==0; all three dataset "
                           "labels carry identical values"),
    "M04": dict(reported_node=3, extraction="incorrect", evidence_level=3,
                scientific="not usable",
                failure="cumulative-results mislabelling",
                audit_note="fresh Evaluator per seed, but get_results() still "
                           "restores rows from disk; Amazon and Last.FM report "
                           "identical range 0.2794, Last.FM never evaluated"),
    "M06": dict(reported_node=5, extraction="incorrect", evidence_level=3,
                scientific="not usable",
                failure="cumulative-results mislabelling",
                audit_note="MovieLens rows reprinted under 'Amazon Video Games' "
                           "and 'Last.FM' labels; Last.FM never evaluated"),
    "M08": dict(reported_node=3, extraction="incorrect", evidence_level=3,
                scientific="not usable",
                failure="cumulative-results mislabelling",
                audit_note="one Evaluator reused across all seeds; std==0.0000 "
                           "for MovieLens and Last.FM"),
    "M09": dict(reported_node=None, extraction="incorrect", evidence_level=3,
                scientific="not usable",
                failure="degenerate statistics (seed folded into algorithm id)",
                audit_note="std==0 for every group; only MovieLens evaluated"),
    "T09": dict(reported_node=5, extraction="incorrect", evidence_level=3,
                scientific="not usable",
                failure="degenerate statistics; earlier node had an invalid split",
                audit_note="summary reports std==0.0000 for every group because "
                           "the seed is part of the algorithm identifier; only "
                           "MovieLens evaluated"),
    "T01": dict(reported_node=6, extraction="correct", evidence_level=3,
                scientific="partially usable",
                failure="Last.FM download returned HTTP 404",
                audit_note="does not use evaluator.get_results(); computes "
                           "metrics itself via LensKit RunAnalysis per split"),
    "T06": dict(reported_node=2, extraction="incorrect", evidence_level=1,
                scientific="not usable",
                failure="post-processing indexes cumulative results and misses",
                audit_note="framework tables hold real values, run summary "
                           "reports every MovieLens metric as 0.0000"),
    "C01": dict(reported_node=3, extraction="correct", evidence_level=3,
                scientific="partially usable",
                failure="runtime budget expired at 25 of 45 conditions",
                audit_note="one row per (dataset, seed, algorithm), explicit "
                           "de-duplication on those keys, checkpointed resume"),
}
DEFAULT_AUDIT = dict(reported_node=None, extraction="not audited",
                     evidence_level=1, scientific="not usable",
                     failure="", audit_note="")

FAILURE = {  # decisive failure class, from the checkpoint logs
    "L00": "result-collection crash (RuntimeError)",
    "L01": "per-node execution timeout", "L02": "per-node execution timeout",
    "L03": "per-node execution timeout",
    "M01": "per-node execution timeout", "M03": "no final tables",
    "M05": "no evaluation output", "M07": "extraction crash (KeyError)",
    "M10": "per-node execution timeout",
    "T02": "per-node execution timeout (KeyError: 'rank')",
    "T03": "missing input file", "T04": "host freeze (memory exhaustion)",
    "T05": "per-node execution timeout",
    "T07": "host freeze (memory exhaustion)",
    "T08": "host freeze (memory exhaustion)",
    "T09": "degenerate split configuration",
    "T10": "missing input file (reused workspace)",
}


def parse_node_txt(p):
    d = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def scan_log(text):
    """Return (conditions, datasets, ended_with_error, numeric_stats, metrics,
    combinations)."""
    conds = set()
    n_fw = 0
    for ds, algo in FRAMEWORK_DONE.findall(text):
        # the framework prints one line per (dataset, algorithm) execution;
        # each execution corresponds to one seed of that condition
        n_fw += 1
        conds.add((ds, algo, f"exec{n_fw}"))
    if not conds:  # generated code with its own evaluation loop (T01 pattern)
        cur, algo = None, None
        for line in text.splitlines():
            m = OWN_COND.search(line)
            if m:
                cur, algo = (m.group(1), m.group(2)), None
                continue
            m = OWN_ALGO.search(line)
            if m:
                algo = m.group(1)
                continue
            if cur and algo and re.search(r"NDCG@1\s*=\s*\d", line):
                conds.add((cur[0], algo, cur[1]))
    datasets = {d for d, _, _ in conds}
    combos = {(d, a) for d, a, _ in conds}
    tail = "\n".join(text.splitlines()[-6:])
    return (len(conds), datasets, bool(ERROR_TAIL.search(tail)),
            bool(NUMERIC_STAT.search(text)), bool(METRICS_PRINTED.search(text)),
            combos)


def main():
    runs, node_rows = [], []
    for arm, (base, backend, roles) in ARMS.items():
        for rd in sorted(base.glob("run_*")):
            idx = int(rd.name.split("_")[1])
            label = f"{arm}{idx:02d}"

            cfg = (rd / "config.toml").read_text()
            allrows = list(csv.DictReader((rd / "costs_log.csv").open()))
            rows = [x for x in allrows
                    if re.match(r"^[\d.]+$", (x["Total USD"] or ""))
                    and re.match(r"^\d+$", (x["Position"] or ""))]
            ts = [datetime.strptime(x["Timestamp"], "%Y-%m-%d %H:%M:%S")
                  for x in rows if re.match(r"\d{4}-\d{2}-\d{2}", x["Timestamp"] or "")]
            models = sorted({x["Model"] for x in rows})
            host = "unknown"
            dbg = rd / "debug.log"
            if dbg.exists():
                m = re.search(r"/home/([a-z_0-9]+)/", dbg.read_text(errors="replace"))
                if m:
                    host = m.group(1)

            nodes = {}
            for f in sorted((rd / "statistics").glob("[0-9]*_*.txt")):
                d = parse_node_txt(f)
                nodes[d["ID"]] = dict(
                    pos=int(d["Position"]), score=float(d["Score"]),
                    buggy=d["Is Buggy"] == "True",
                    satisfactory=d["Is Satisfactory"] == "True",
                    exec_time=float(d["Execution Time"]))

            per_node = {}
            for ck in sorted((rd / "checkpoint").glob("*")):
                log = ck / "out.log"
                if not log.exists():
                    continue
                n, ds, err, stats, met, combos = scan_log(
                    log.read_text(errors="replace"))
                info = nodes.get(ck.name, dict(pos=-1, score=float("nan"),
                                               buggy=None, satisfactory=None,
                                               exec_time=float("nan")))
                per_node[info["pos"]] = dict(hash=ck.name, conditions=n,
                                             datasets=ds, error_end=err,
                                             numeric_stats=stats, metrics=met,
                                             combos=combos, **info)
                node_rows.append(dict(
                    run=label, position=info["pos"], node_hash=ck.name,
                    reviewer_score=round(info["score"], 4),
                    is_buggy=info["buggy"], is_satisfactory=info["satisfactory"],
                    conditions_completed=n,
                    datasets_evaluated=len(ds),
                    metric_values_printed=met,
                    ended_with_error=err, numeric_seed_statistics=stats))

            aud = dict(DEFAULT_AUDIT, **AUDIT.get(label, {}))
            # C3: a node that terminated cleanly and printed numeric statistics
            stat_nodes = [p for p, v in per_node.items()
                          if v["numeric_stats"] and not v["error_end"]
                          and v["conditions"] > 0 and v["metrics"]]
            rep = aud["reported_node"]
            if rep is None:
                rep = (max(stat_nodes, key=lambda p: per_node[p]["conditions"])
                       if stat_nodes else
                       (max(per_node, key=lambda p: per_node[p]["conditions"])
                        if per_node else None))
            rn = per_node.get(rep, dict(conditions=0, datasets=set(), hash="",
                                        score=float("nan"), combos=set(),
                                        metrics=False))
            best = max((v["score"] for v in nodes.values()), default=float("nan"))

            runs.append(dict(
                run=label, arm=arm, backend=backend, roles=roles,
                backend_effective="+".join(models) or "not logged",
                branch=BRANCH[arm], base_commit=BASE_COMMIT, host=host,
                started=min(ts).isoformat() if ts else "",
                wall_clock_min=round((max(ts) - min(ts)).total_seconds() / 60) if ts else "",
                cost_usd=round(sum(float(x["Total USD"]) for x in rows), 2),
                tokens=sum(int(x["Total Tokens"]) for x in rows),
                n_checkpoints=len(per_node),
                tree_search_completed=len(per_node) == 7,
                evaluation_output=any(v["metrics"] for v in per_node.values()),
                statistical_summary=bool(stat_nodes),
                reported_node=rep if rep is not None else "",
                reported_node_hash=rn["hash"][:8],
                datasets_evaluated=len(rn["datasets"]),
                conditions_completed=rn["conditions"],
                conditions_requested=45,
                # algorithm-dataset-run combinations, the preprint's unit
                combinations_with_output=len(rn["combos"]) if rn["metrics"] else 0,
                combinations_valid=(len(rn["combos"])
                                    if aud["extraction"] == "correct" else 0),
                combinations_requested=9,
                best_reviewer_score=round(best, 4),
                satisfactory_node=any(v["satisfactory"] for v in nodes.values()),
                extraction_logic_audited=aud["extraction"] != "not audited",
                extraction_verdict=aud["extraction"],
                metrics_independently_recomputed=False,
                evidence_level=aud["evidence_level"],
                primary_failure=aud["failure"] or FAILURE.get(label, ""),
                scientific_status=aud["scientific"],
                audit_note=aud["audit_note"],
            ))

    with (OUT / "run_status.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    with (OUT / "node_status.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(node_rows[0]))
        w.writeheader()
        w.writerows(node_rows)

    n = len(runs)
    print(f"{n} runs written to run_status.csv, {len(node_rows)} nodes")
    for k in ["tree_search_completed", "evaluation_output", "statistical_summary"]:
        print(f"  {k}: {sum(bool(r[k]) for r in runs)}/{n}")
    print("  all 3 datasets evaluated:",
          sum(r["datasets_evaluated"] == 3 for r in runs))
    print("  45/45 conditions:", sum(r["conditions_completed"] == 45 for r in runs))
    print("  extraction correct:",
          [r["run"] for r in runs if r["extraction_verdict"] == "correct"])
    print("  statistical summary runs:",
          [r["run"] for r in runs if r["statistical_summary"]])
    print("  algorithm-dataset-run combinations: "
          f"{sum(r['combinations_with_output'] for r in runs)} with output, "
          f"{sum(r['combinations_valid'] for r in runs)} valid, of {9 * n}")
    for arm in ARMS:
        a = [r for r in runs if r["arm"] == arm]
        print(f"  arm {arm} (n={len(a)}): "
              f"TS {sum(r['tree_search_completed'] for r in a)}, "
              f"E {sum(r['evaluation_output'] for r in a)}, "
              f"S {sum(r['statistical_summary'] for r in a)}, "
              f"3ds {sum(r['datasets_evaluated'] == 3 for r in a)}, "
              f"correct {sum(r['extraction_verdict'] == 'correct' for r in a)}")


if __name__ == "__main__":
    main()
