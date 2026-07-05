#!/usr/bin/env python3
"""Figure 3 -- cross-site degradation across the four observed sites.

Uses the four real reporting sites (MIMIC-CLIF internal holdout + three external:
Rush, Northwestern, University of Chicago Medicine). The paper's central empirical
claim -- a model strong on the public development set degrades, and for sepsis
falls below chance, at hospitals that did not build it -- is grounded entirely in
observed results.

Left panel: a slope chart. For each task, a line runs from the MIMIC-internal
AUROC to EACH external site's AUROC, so sepsis (T4) shows three lines crossing the
chance line. A shaded band below AUROC 0.5 marks worse-than-chance discrimination.
Right panel: the internal->external AUROC drop per task (median across the three
external sites), ordered by magnitude.

Landmark tasks (T1, T2) use the pooled dynamic AUC. All numbers come from
flair_paper_data.py. Run with a Python that has matplotlib (repo .venv).
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

TASK_COLOR = {
    "T1": "#5b86a6", "T2": "#6f9e78", "T3": "#c9a24b",
    "T4": "#b9744e", "T5": "#8a6ea6",
}


def main():
    tasks = D.TASK_IDS
    externals = D.external_sites()  # rush, nu, ucmc
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 5.4), dpi=150,
                                   gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.patch.set_facecolor("white")

    # ---- Left: slope chart (MIMIC internal -> each external) --------------
    axL.set_facecolor(C["panel"])
    axL.axhspan(0.30, 0.5, color=C["band"], alpha=0.55, zorder=0)
    axL.axhline(0.5, color=C["diag"], lw=1.2, ls=(0, (2, 3)), zorder=1)
    axL.text(1.02, 0.5, "chance", fontsize=8.5, color=C["muted"],
             family="monospace", va="bottom", ha="left")

    for tid in tasks:
        mi = D.REAL["mimic"]["auroc"][tid]
        col = TASK_COLOR[tid]
        # one line MIMIC -> each external site
        for s in externals:
            ex = D.REAL[s]["auroc"][tid]
            axL.plot([0, 1], [mi, ex], color=col, lw=1.6, alpha=0.55, zorder=3)
            axL.plot([1], [ex], marker="o", ms=5.5, mfc=col, mec="white", zorder=4)
        axL.plot([0], [mi], marker="D", ms=8, mfc=col, mec="white", zorder=5)
        axL.annotate(f"{tid}", (0, mi), xytext=(-9, 0), textcoords="offset points",
                     fontsize=10, color=col, family="serif", va="center", ha="right",
                     weight="bold")

    axL.set_xlim(-0.30, 1.30)
    axL.set_ylim(0.36, 0.90)
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["MIMIC-CLIF\n(internal holdout)", "External sites\n(Rush, NU, UChicago)"],
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

    # ---- Right: median internal->external drop per task, ordered ----------
    axR.set_facecolor(C["panel"])
    drops = [(tid, round(D.REAL["mimic"]["auroc"][tid] - D.external_median(tid), 3))
             for tid in tasks]
    drops.sort(key=lambda x: x[1])
    short = dict((t[0], t[1]) for t in D.TASKS)
    labels = [f"{tid} {short[tid]}" for tid, _ in drops]
    ypos = range(len(drops))
    axR.barh(list(ypos), [d for _, d in drops],
             color=[TASK_COLOR[tid] for tid, _ in drops], alpha=0.9, zorder=3)
    for i, (tid, d) in enumerate(drops):
        axR.text(d + 0.004, i, f"{d:.3f}", va="center", ha="left",
                 fontsize=9.5, color=C["ink"], family="monospace")
    axR.set_yticks(list(ypos))
    axR.set_yticklabels(labels, fontsize=9.5, family="monospace", color=C["ink"])
    axR.set_xlim(0, max(d for _, d in drops) * 1.30)
    axR.set_xlabel("Median AUROC drop (internal - external)", fontsize=10,
                   color=C["muted"], family="monospace")
    axR.tick_params(labelsize=9, colors=C["muted"])
    for lab in axR.get_xticklabels():
        lab.set_family("monospace")
    axR.grid(True, axis="x", color=C["grid"], linewidth=1)
    axR.set_axisbelow(True)
    axR.set_title("Cross-site AUROC drop by task", fontsize=12.5, color=C["ink"], family="serif")
    for sp in axR.spines.values():
        sp.set_edgecolor(C["grid"])

    fig.suptitle("Observed cross-site degradation of the frozen baseline (MIMIC-CLIF vs three external sites)",
                 fontsize=13.0, color=C["ink"], family="serif", y=1.01)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
