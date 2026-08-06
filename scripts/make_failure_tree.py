# -*- coding: utf-8 -*-
"""Classification tree of all 26 runs -> failure / usefulness leaves.
Standalone: writes figures/fig_failure_tree.{pdf,png} only.
Every run appears in exactly one leaf; leaf counts sum to 26."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# palette
GREEN = "#1baf7a"; AMBER = "#eda100"; RED = "#d03b3b"; DARKRED = "#9c2b2b"
BLUE = "#2a78d6"; SLATE = "#5b6b7a"
INK = "#0b0b0b"; SURF = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

def leaf(title, runs, color, textcolor="white"):
    return {"title": title, "runs": runs, "color": color, "tc": textcolor,
            "children": []}

def node(title, color, textcolor, children):
    return {"title": title, "runs": "", "color": color, "tc": textcolor,
            "children": children}

TREE = node("All runs\n(26)", SLATE, "white", [
    node("Numeric seed\nstatistics (9)", BLUE, "white", [
        leaf("Audited correct, partial grid", "T01  C01  (2)", GREEN),
        leaf("Cross-dataset mislabelling", "L04  M02  M04  M06  M08  (5)", AMBER, INK),
        leaf("Degenerate std (seed in algo id)", "M09  T09  (2)", AMBER, INK),
    ]),
    node("Metric values,\nno statistics (13)", BLUE, "white", [
        leaf("Timeout (Amazon / ItemKNN)", "L01-L03  M01  M10  T02  T05  (7)", RED),
        leaf("Missing dataset file", "T10  (1)", RED),
        leaf("Result-extraction crash", "L00  M07  (2)", RED),
        leaf("Mis-indexed post-processing", "T06  (1)", RED),
        leaf("No final tables", "M03  (1)", RED),
        leaf("Partial output, then freeze", "T08  (1)", AMBER, INK),
    ]),
    node("No metric\nvalues (4)", BLUE, "white", [
        leaf("No evaluation output", "M05  T03  (2)", RED),
        leaf("Aborted - host freeze", "T04  T07  (2)", DARKRED),
    ]),
])

# ---- tidy layout: leaves get evenly spaced y slots, parents = mean(children) ----
DX = {0: 0.06, 1: 0.29, 2: 0.66}    # x-centre per depth
BW = {0: 0.055, 1: 0.125, 2: 0.195} # box half-width per depth
_slot = [0.0]
def assign(nd, depth):
    nd["x"] = DX[depth]; nd["hw"] = BW[depth]
    if not nd["children"]:
        nd["y"] = _slot[0]; _slot[0] += 1.0
        nd["depth"] = depth
        return
    for c in nd["children"]:
        assign(c, depth + 1)
    nd["y"] = sum(c["y"] for c in nd["children"]) / len(nd["children"])
    nd["depth"] = depth

assign(TREE, 0)
n_leaf = _slot[0]

# normalise y into [0,1] (invert so first leaf on top)
def norm_y(nd):
    nd["y"] = 1.0 - (nd["y"] + 0.5) / n_leaf
    for c in nd["children"]:
        norm_y(c)
norm_y(TREE)

fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

BH = 0.5 / n_leaf * 0.72  # box half-height (< slot/2 so leaves have gaps)

def draw(nd):
    x, y, hw = nd["x"], nd["y"], nd["hw"]
    # connectors to children (elbow)
    for c in nd["children"]:
        x0 = x + hw
        x1 = c["x"] - c["hw"]
        xm = (x0 + x1) / 2
        ax.plot([x0, xm, xm, x1], [y, y, c["y"], c["y"]],
                color="#9aa4ad", linewidth=0.8, zorder=1)
    # box
    box = FancyBboxPatch((x - hw, y - BH), 2 * hw, 2 * BH,
                         boxstyle="round,pad=0.004,rounding_size=0.012",
                         linewidth=0, facecolor=nd["color"], zorder=2)
    ax.add_patch(box)
    if nd["runs"]:
        ax.text(x, y + 0.014, nd["title"], ha="center", va="center",
                color=nd["tc"], fontsize=6.4, fontweight="bold", zorder=3)
        ax.text(x, y - 0.015, nd["runs"], ha="center", va="center",
                color=nd["tc"], fontsize=5.8, family="monospace", zorder=3)
    else:
        ax.text(x, y, nd["title"], ha="center", va="center",
                color=nd["tc"], fontsize=6.9, fontweight="bold", zorder=3)
    for c in nd["children"]:
        draw(c)

draw(TREE)

# legend
from matplotlib.patches import Patch
leg = [Patch(color=GREEN, label="usable / correct"),
       Patch(color=AMBER, label="produced results, not trustworthy"),
       Patch(color=RED, label="crashed / incomplete"),
       Patch(color=DARKRED, label="aborted (infrastructure)")]
ax.legend(handles=leg, loc="lower center", ncols=4, frameon=False,
          fontsize=6.0, handlelength=1.0, handleheight=1.0,
          bbox_to_anchor=(0.5, -0.06), columnspacing=1.2)

fig.savefig(FIGDIR / "fig_failure_tree.pdf")
fig.savefig(FIGDIR / "fig_failure_tree.png")
plt.close(fig)
print("wrote", FIGDIR / "fig_failure_tree.pdf", "| leaves:", int(n_leaf))
