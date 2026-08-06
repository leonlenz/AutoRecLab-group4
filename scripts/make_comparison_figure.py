# -*- coding: utf-8 -*-
"""Comparison figure: our reproduction vs. the AutoRecLab preprint.

The two studies report success at two different levels of aggregation, so the
figure keeps them apart. The left panel counts whole AutoRecLab executions
(our 26 runs; the preprint's 4 runs). The right panel counts
algorithm-dataset-run combinations, the unit behind the preprint's 15/27, and
compares it with the same unit in our study (26 runs x 3 datasets x 3
algorithms = 234 combinations). A preprint reference line only ever appears in
the panel whose unit matches it.

Counts come from scripts/run_status.csv (build_run_status.py); nothing is
hard-coded except the two figures quoted from the preprint.

Writes figures/fig_vs_preprint.{pdf,png}.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---- palette (shared with make_figures.py) ----
BLUE = "#2a78d6"    # operational success criteria
RED = "#d03b3b"     # scientific-validity criteria
INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#ffffff"; GRID = "#e5e4e0"
GREEN = "#008300"; YELLOW = "#eda100"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.4, "axes.labelsize": 7.5,
    "xtick.labelsize": 5.7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.bbox": "tight", "savefig.dpi": 300,
    "font.family": "sans-serif",
})

status = list(csv.DictReader((HERE / "run_status.csv").open()))
N = len(status)
NC = 9 * N  # algorithm-dataset-run combinations available in our study


def n_runs(pred):
    return sum(bool(pred(r)) for r in status)


# ---- panel 1: whole executions ------------------------------------------
RUN_CRITERIA = [
    ("Tree search\ncompleted",
     n_runs(lambda r: r["tree_search_completed"] == "True"), True),
    ("Metric values\n($\\geq$1 node)",
     n_runs(lambda r: r["evaluation_output"] == "True"), True),
    ("Numeric seed\nstatistics",
     n_runs(lambda r: r["statistical_summary"] == "True"), False),
    ("Correct extr.\n$+$ 3 datasets",
     n_runs(lambda r: r["extraction_verdict"] == "correct"
            and int(r["datasets_evaluated"]) == 3), False),
    ("All 45\nconditions",
     n_runs(lambda r: int(r["conditions_completed"]) == 45), False),
    ("Reviewer-\nsatisfactory",
     n_runs(lambda r: r["satisfactory_node"] == "True"), False),
]
# ---- panel 2: algorithm-dataset-run combinations -------------------------
COMBO_CRITERIA = [
    ("Produced\nmetric values",
     sum(int(r["combinations_with_output"]) for r in status), True),
    ("From audited-correct\nextraction",
     sum(int(r["combinations_valid"]) for r in status), False),
]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.9, 2.6),
                               gridspec_kw={"width_ratios": [3, 1.15],
                                            "wspace": 0.28})


def draw(ax, criteria, total, title, unit):
    rates = [100.0 * c[1] / total for c in criteria]
    ax.bar(range(len(criteria)), rates, width=0.68,
           color=[BLUE if c[2] else RED for c in criteria], zorder=3)
    for xi, (c, r) in enumerate(zip(criteria, rates)):
        ax.text(xi, r + 1.5, "{:.0f}%\n({}/{})".format(r, c[1], total),
                ha="center", va="bottom", fontsize=5.8, color=INK)
    ax.set_xticks(range(len(criteria)))
    ax.set_xticklabels([c[0] for c in criteria], fontsize=5.4)
    ax.set_ylabel(unit)
    ax.set_title(title, color=INK2)
    ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


draw(axl, RUN_CRITERIA, N, "(a) whole AutoRecLab executions",
     "Share of our {} runs (%)".format(N))
draw(axr, COMBO_CRITERIA, NC, "(b) algorithm–dataset–run combinations",
     "Share of {} combinations (%)".format(NC))

# The preprint's 3/4 completed runs belong in panel (a), its 15/27 valid
# sub-experiments in panel (b) -- never both in the same panel.
axl.axhline(75.0, color=GREEN, linewidth=1.0, linestyle="--", zorder=2)
axl.text(len(RUN_CRITERIA) - 0.45, 76.0,
         "preprint: 3/4 runs completed (75%)", ha="right", va="bottom",
         color=GREEN, fontsize=5.9)
axr.axhline(56.0, color=YELLOW, linewidth=1.0, linestyle="--", zorder=2)
axr.text(-0.45, 90.0, "preprint:\n15/27 valid (56%)",
         ha="left", va="bottom", color=YELLOW, fontsize=5.9)

LEG = [Patch(color=BLUE, label="operational criterion"),
       Patch(color=RED, label="scientific-validity criterion")]
axl.legend(handles=LEG, frameon=False, loc="upper right",
           handlelength=0.9, handleheight=0.9, bbox_to_anchor=(1.0, 1.02))

fig.savefig(FIGDIR / "fig_vs_preprint.pdf")
fig.savefig(FIGDIR / "fig_vs_preprint.png")
plt.close(fig)
print("wrote", FIGDIR / "fig_vs_preprint.pdf")
print("run-level:", [(c[0].replace("\n", " "), c[1]) for c in RUN_CRITERIA])
print("combination-level:", [(c[0].replace("\n", " "), c[1]) for c in COMBO_CRITERIA],
      "of", NC)
