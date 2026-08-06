# -*- coding: utf-8 -*-
"""Paper figures for the three-arm comparison (gpt-5.4 / gpt-5.4-mini / deepseek).

Reads scripts/runs_summary.json (produced by analyze_runs.py) for the per-run
cost/score/runtime figures, and scripts/run_status.csv plus node_status.csv
(produced by build_run_status.py) for every outcome-status figure, so that the
status definitions live in exactly one place.
"""
import csv, json, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

runs = json.loads((HERE / "runs_summary.json").read_text())
status = list(csv.DictReader((HERE / "run_status.csv").open()))
nodes_csv = list(csv.DictReader((HERE / "node_status.csv").open()))
def yes(row, key):
    return row[key] == "True"
g54 = [r for r in runs if r["group"] == "flo/gpt-5.4"]
gmini = [r for r in runs if r["group"] == "flo/mini"]
gdeep = [r for r in runs if r["group"] == "tristan/mini"]
gcodex = [r for r in runs if r["group"] == "leon/codex"]
ordered = g54 + gmini + gdeep + gcodex

# ---- palette (dataviz default, validated; slots 1-3) ----
BLUE = "#2a78d6"    # gpt-5.4
AQUA = "#1baf7a"    # gpt-5.4-mini
YELLOW = "#eda100"  # deepseek-v4-flash
GREEN = "#008300"   # codex (cross-model)
INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#ffffff"; GRID = "#e5e4e0"
RED = "#d03b3b"

GROUPS = [("gpt-5.4", g54, BLUE), ("gpt-5.4-mini", gmini, AQUA),
          ("deepseek-v4-flash", gdeep, YELLOW), ("codex", gcodex, GREEN)]

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 5.4, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.bbox": "tight", "savefig.dpi": 300,
    "font.family": "sans-serif",
})

def style_ax(ax, rot=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if rot:
        ax.tick_params(axis="x", rotation=90)

def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png")
    plt.close(fig)

PREFIX = {"flo/gpt-5.4": "L", "flo/mini": "M", "tristan/mini": "T", "leon/codex": "C"}
def lab(r):
    return PREFIX[r["group"]] + r["run"].replace("run_", "")

labels = [lab(r) for r in ordered]
cmap = {"flo/gpt-5.4": BLUE, "flo/mini": AQUA, "tristan/mini": YELLOW, "leon/codex": GREEN}
colors = [cmap[r["group"]] for r in ordered]
LEG = [Patch(color=c, label=n) for n, _, c in GROUPS]

def group_ranges():
    i = 0
    for name, grp, col in GROUPS:
        yield name, grp, col, range(i, i + len(grp))
        i += len(grp)

# ================= Fig: best reviewer score per run =================
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.bar(labels, [r["best_score"] for r in ordered], color=colors, width=0.72)
for name, grp, col, xs in group_ranges():
    m = st.mean([r["best_score"] for r in grp])
    ax.plot([min(xs) - 0.4, max(xs) + 0.4], [m, m], color=INK2, linewidth=0.9,
            linestyle="--")
    ax.text(min(xs) - 0.3, m + 0.015, f"{m:.2f}", fontsize=6.2, color=INK2)
ax.set_ylabel("Best reviewer score (0–1)")
ax.set_ylim(0, 1.06)
style_ax(ax, rot=True)
ax.legend(handles=LEG, frameon=False, loc="upper left", ncols=3,
          handlelength=0.9, handleheight=0.9, columnspacing=0.8,
          bbox_to_anchor=(-0.02, 1.14))
save(fig, "fig_scores_per_run")

# ================= Fig: runtime and tokens per run =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.1), sharex=True,
                               gridspec_kw={"hspace": 0.15})
ax1.bar(labels, [r["wall_min"] for r in ordered], color=colors, width=0.72)
ax1.set_ylabel("Wall-clock time (min)")
style_ax(ax1)
ax2.bar(labels, [r["total_tokens"] / 1e6 for r in ordered], color=colors, width=0.72)
ax2.set_ylabel("LLM tokens (millions)")
style_ax(ax2, rot=True)
ax1.legend(handles=LEG, frameon=False, loc="upper left", ncols=3,
           handlelength=0.9, handleheight=0.9, columnspacing=0.8,
           bbox_to_anchor=(-0.02, 1.17))
save(fig, "fig_runtime_cost")

# ================= Fig: outcome funnel (grouped, 3 arms) =================
# Every stage is a separate status criterion, counted from run_status.csv.
# The criteria are deliberately not nested: T08 printed metric values without
# completing its tree search.
CRIT = [
    ("Runs started", lambda r: True),
    ("Tree search\ncompleted", lambda r: yes(r, "tree_search_completed")),
    ("Metric values\nin $\\geq$1 node", lambda r: yes(r, "evaluation_output")),
    ("Numeric seed\nstatistics emitted", lambda r: yes(r, "statistical_summary")),
    ("All 3 datasets\nevaluated", lambda r: int(r["datasets_evaluated"]) == 3),
    ("Extraction logic\naudited correct",
     lambda r: r["extraction_verdict"] == "correct"),
    ("All 45 conditions\ncompleted", lambda r: int(r["conditions_completed"]) == 45),
    ("Reviewer-satisfactory\nnode", lambda r: yes(r, "satisfactory_node")),
]
stages = [(label, *[sum(pred(r) for r in status if r["arm"] == a)
                    for a in "LMTC"]) for label, pred in CRIT]
fig, ax = plt.subplots(figsize=(3.4, 3.4))
names = [s[0] for s in stages][::-1]
vals = [[s[i] for s in stages][::-1] for i in (1, 2, 3, 4)]
y = np.arange(len(names))
h = 0.2
for off, v, (name, _, col) in zip((1.5 * h, 0.5 * h, -0.5 * h, -1.5 * h), vals, GROUPS):
    ax.barh(y + off, v, height=h * 0.92, color=col, label=name)
    for yi, vv in zip(y + off, v):
        if vv:  # zero bars would stack four labels on the axis
            ax.text(vv + 0.12, yi, str(vv), va="center", fontsize=6, color=INK)
ax.set_yticks(y, names, fontsize=6.5)
ax.set_xlim(0, 11)
ax.set_xlabel("Number of runs")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", handlelength=0.9, handleheight=0.9)
save(fig, "fig_outcome_funnel")

# ========= Fig: checkpoint progress, self-assessed vs. observable =========
# Top: the Reviewer's own score. Bottom: two measures that do not depend on the
# Reviewer at all -- how many nodes printed metric values, and how much of the
# 45-condition grid they covered.
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True,
                              gridspec_kw={"hspace": 0.18})
positions = list(range(7))
for name, grp, col in GROUPS:
    per_pos = {p: [] for p in positions}
    for r in grp:
        for n in r["nodes"]:
            per_pos[n["pos"]].append(n["score"])
    means = [st.mean(per_pos[p]) for p in positions if per_pos[p]]
    ax.plot(positions[:len(means)], means, color=col, linewidth=2, marker="o",
            markersize=3.5, label=name)
ax.set_ylabel("Mean reviewer score (0–1)")
ax.set_ylim(0, 1.0)
style_ax(ax)
ax.legend(frameon=False, loc="upper left", ncols=1)

by_pos = {p: [n for n in nodes_csv if int(n["position"]) == p] for p in positions}
share = [100 * sum(yes(n, "metric_values_printed") for n in by_pos[p]) / len(by_pos[p])
         for p in positions]
cond = [st.mean([int(n["conditions_completed"]) for n in by_pos[p]]) for p in positions]
ax2.bar([p - 0.19 for p in positions], share, width=0.36, color=INK2,
        label="nodes printing metric values (%)")
ax2.bar([p + 0.19 for p in positions], [c / 45 * 100 for c in cond], width=0.36,
        color=RED, label="mean grid coverage (% of 45 conditions)")
ax2.set_xlabel("Checkpoint position in tree search")
ax2.set_ylabel("Percent of nodes / of grid")
ax2.set_xticks(positions)
ax2.set_ylim(0, 62)
style_ax(ax2)
ax2.legend(frameon=False, loc="upper left", handlelength=0.9, handleheight=0.9)
save(fig, "fig_checkpoint_evolution")

# ================= aggregates =================
for name, grp, _ in GROUPS:
    def m(k): return st.mean([r[k] for r in grp])
    print(f"{name}: n={len(grp)} wall_mean={m('wall_min'):.0f} "
          f"wall_med={st.median([r['wall_min'] for r in grp]):.0f} "
          f"cost_mean={m('total_usd'):.2f} tokens_mean={m('total_tokens')/1e6:.2f}M "
          f"score_mean={m('best_score'):.3f} loc={m('mean_loc_all'):.0f} chg={m('mean_change'):.0f}")
print("total cost:", round(sum(r["total_usd"] for r in runs), 2))
print("total nodes:", sum(r["n_nodes"] for r in runs),
      "buggy:", sum(r["n_buggy"] for r in runs))
print("figures written to", FIGDIR)
