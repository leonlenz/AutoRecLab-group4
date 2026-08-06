# -*- coding: utf-8 -*-
"""Analyze all AutoRecLab runs. Structure: allRuns/<model>/run_XX/"""
import csv, json, re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# group label -> directory of run_XX folders
GROUPS = {
    "flo/gpt-5.4": ROOT / "allRuns" / "gpt-5.4",
    "flo/mini": ROOT / "allRuns" / "gpt-5.4-mini",
    "tristan/mini": ROOT / "tristanRuns",
    "leon/codex": ROOT / "codexRuns",
}

def parse_node_txt(p):
    d = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d

runs = []
for group, base in GROUPS.items():
  for rd in sorted(base.glob("run_*")):
    r = {"run": rd.name, "group": group, "id": f"{group}/{rd.name}"}
    cfg = (rd / "config.toml").read_text()
    m = re.search(r'model = "([^"]+)"', cfg)
    r["model"] = m.group(1) if m else "?"
    r["keep_only"] = "keep_only_relevant_files = true" in cfg

    # costs (exclude SUMMARIZED/aggregate rows)
    allrows = list(csv.DictReader((rd / "costs_log.csv").open()))
    rows = [x for x in allrows
            if re.match(r"^[\d.]+$", (x["Total USD"] or ""))
            and re.match(r"^\d+$", (x["Position"] or ""))]
    # effective model = what the API was actually called with
    r["model_logged"] = rows[0]["Model"] if rows else "?"
    r["n_llm_calls"] = len(rows)
    r["total_usd"] = sum(float(x["Total USD"]) for x in rows)
    r["total_tokens"] = sum(int(x["Total Tokens"]) for x in rows)
    ts = [datetime.strptime(x["Timestamp"], "%Y-%m-%d %H:%M:%S")
          for x in rows if re.match(r"\d{4}-\d{2}-\d{2}", x["Timestamp"] or "")]
    r["wall_min"] = (max(ts) - min(ts)).total_seconds() / 60.0
    r["start"] = min(ts).isoformat()

    # per-node stats
    nodes = []
    for f in sorted((rd / "statistics").glob("[0-9]*_*.txt")):
        d = parse_node_txt(f)
        nodes.append({
            "pos": int(d["Position"]),
            "id": d["ID"],
            "score": float(d["Score"]),
            "buggy": d["Is Buggy"] == "True",
            "satisfactory": d["Is Satisfactory"] == "True",
            "exec_time": float(d["Execution Time"]),
            "loc": int(d["Lines of Code"]),
            "ins": int(d["Insertions"]),
            "dels": int(d["Deletions"]),
        })
    r["nodes"] = nodes
    r["n_nodes"] = len(nodes)
    r["n_buggy"] = sum(n["buggy"] for n in nodes)
    r["n_satisfactory"] = sum(n["satisfactory"] for n in nodes)
    best = max(nodes, key=lambda n: n["score"])
    r["best_score"] = best["score"]
    r["best_pos"] = best["pos"]
    r["best_loc"] = best["loc"]
    r["best_node_id"] = best["id"]
    ch = [n["ins"] + n["dels"] for n in nodes if n["pos"] > 0]
    r["mean_change"] = sum(ch) / len(ch) if ch else 0
    r["mean_loc_all"] = sum(n["loc"] for n in nodes) / len(nodes)
    r["mean_exec_time_s"] = sum(n["exec_time"] for n in nodes) / len(nodes)
    runs.append(r)

hdr = ["id", "model_logged", "wall_min", "total_usd", "total_tokens", "n_nodes",
       "n_buggy", "n_satisfactory", "best_score", "best_pos", "best_loc", "mean_change"]
print("\t".join(hdr))
for r in runs:
    print("\t".join(f"{r.get(h):.2f}" if isinstance(r.get(h), float) else str(r.get(h)) for h in hdr))

import statistics as st
def stats_block(name, sel):
    xs = [r for r in runs if sel(r)]
    if not xs:
        return
    def m(key): return st.mean([r[key] for r in xs])
    def med(key): return st.median([r[key] for r in xs])
    print(f"--- {name} (n={len(xs)}) ---")
    print(f" wall_min mean={m('wall_min'):.1f} median={med('wall_min'):.1f} "
          f"min={min(r['wall_min'] for r in xs):.1f} max={max(r['wall_min'] for r in xs):.1f}")
    print(f" cost mean=${m('total_usd'):.2f} median=${med('total_usd'):.2f} total=${sum(r['total_usd'] for r in xs):.2f}")
    print(f" tokens mean={m('total_tokens'):,.0f}")
    print(f" best_score mean={m('best_score'):.3f}")
    print(f" best_loc mean={m('best_loc'):.0f}; loc_all_nodes mean={m('mean_loc_all'):.0f}")
    print(f" change/iter mean={m('mean_change'):.0f}")

print()
for g in GROUPS:
    stats_block(g, lambda r, g=g: r["group"] == g)
stats_block("ALL", lambda r: True)

print()
print("per-run node scores:")
for r in runs:
    print(r["id"], [(n["pos"], round(n["score"], 2), "B" if n["buggy"] else "-") for n in sorted(r["nodes"], key=lambda n: n["pos"])])

out = Path(__file__).parent / "runs_summary.json"
out.write_text(json.dumps(runs, indent=1, default=str))
print("saved", out)
