#!/usr/bin/env python3
"""Figure 3 -- cross-site degradation (OBSERVED data only).

Deliberately uses only the two real sites (MIMIC-CLIF internal holdout and Rush
external), so the paper's central empirical claim -- that a model strong on
public data degrades at a hospital that did not build it -- is unambiguously
grounded in observed results, with no synthetic content.

Left panel: a slope chart, one connected line per task from MIMIC-internal AUROC
to Rush-external AUROC, with the drop annotated. A shaded band below AUROC 0.5
marks worse-than-chance discrimination (reached by sepsis at Rush).
Right panel: the internal->external AUROC drop per task as a bar, ordered by
magnitude, making sepsis and readmission the visual headline.

Run with a Python that has matplotlib + numpy (repo .venv).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flair_paper_data as D

FIG = Path(__file__).resolve().parent / "figures"
OUT = FIG / "figure3_degradation.png"
C = D.COLOR

# distinct hues per task, harmonized with the FLAIR palette
TASK_COLOR = {
    "T1": "#5b86a6", "T2": "#6f9e78", "T3": "#c9a24b",
    "T4": "#b9744e", "T5": "#8a6ea6",
}


def main():
    tasks = D.TASK_IDS
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=150,
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.patch.set_facecolor("white")

    # ---- Left: slope chart ------------------------------------------------
    axL.set_facecolor(C["panel"])
    axL.axhspan(0.30, 0.5, color=C["band"], alpha=0.55, zorder=0)
    axL.axhline(0.5, color=C["diag"], lw=1.2, ls=(0, (2, 3)), zorder=1)
    axL.text(1.5, 0.5, "chance", fontsize=8.5, color=C["muted"],
             family="monospace", va="bottom", ha="right")

    for tid in tasks:
        mi = D.REAL["mimic"]["auroc"][tid]
        ru = D.REAL["rush"]["auroc"][tid]
        col = TASK_COLOR[tid]
        axL.plot([0, 1], [mi, ru], color=col, lw=2.4, marker="o", ms=8,
                 mfc=col, mec="white", zorder=4)
        axL.annotate(f"{tid}", (0, mi), xytext=(-8, 0), textcoords="offset points",
                     fontsize=10, color=col, family="serif", va="center", ha="right", weight="bold")
        drop = mi - ru
        axL.annotate(f"-{drop:.3f}", (1, ru), xytext=(10, 0), textcoords="offset points",
                     fontsize=9.5, color=col, family="monospace", va="center", ha="left")

    axL.set_xlim(-0.28, 1.42)
    axL.set_ylim(0.36, 0.90)
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["MIMIC-CLIF\n(internal holdout)", "Rush\n(external)"],
                        fontsize=10, family="monospace", color=C["ink"])
    axL.set_ylabel("AUROC", fontsize=10.5, color=C["muted"], family="monospace")
    axL.tick_params(labelsize=9, colors=C["muted"])
    for lab in axL.get_yticklabels():
        lab.set_family("monospace")
    axL.grid(True, axis="y", color=C["grid"], linewidth=1)
    axL.set_axisbelow(True)
    axL.set_title("Internal vs external discrimination", fontsize=12.5,
                  color=C["ink"], family="serif")
    for sp in axL.spines.values():
        sp.set_edgecolor(C["grid"])

    # ---- Right: degradation bars, ordered by drop -------------------------
    axR.set_facecolor(C["panel"])
    drops = [(tid, D.REAL["mimic"]["auroc"][tid] - D.REAL["rush"]["auroc"][tid]) for tid in tasks]
    drops.sort(key=lambda x: x[1])
    labels = [f"{tid} {dict((t[0], t[1]) for t in D.TASKS)[tid]}" for tid, _ in drops]
    ypos = range(len(drops))
    axR.barh(list(ypos), [d for _, d in drops],
             color=[TASK_COLOR[tid] for tid, _ in drops], alpha=0.9, zorder=3)
    for i, (tid, d) in enumerate(drops):
        axR.text(d + 0.004, i, f"{d:.3f}", va="center", ha="left",
                 fontsize=9.5, color=C["ink"], family="monospace")
    axR.set_yticks(list(ypos))
    axR.set_yticklabels(labels, fontsize=9.5, family="monospace", color=C["ink"])
    axR.set_xlim(0, max(d for _, d in drops) * 1.28)
    axR.set_xlabel("AUROC drop (internal - external)", fontsize=10, color=C["muted"], family="monospace")
    axR.tick_params(labelsize=9, colors=C["muted"])
    for lab in axR.get_xticklabels():
        lab.set_family("monospace")
    axR.grid(True, axis="x", color=C["grid"], linewidth=1)
    axR.set_axisbelow(True)
    axR.set_title("Cross-site AUROC drop by task", fontsize=12.5, color=C["ink"], family="serif")
    for sp in axR.spines.values():
        sp.set_edgecolor(C["grid"])

    fig.suptitle("Observed cross-site degradation of the frozen baseline (MIMIC-CLIF → Rush)",
                 fontsize=13.5, color=C["ink"], family="serif", y=1.01)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
