#!/usr/bin/env python3
"""Per-task ROC plots overlaying every site (MIMIC + Rush), FLAIR-website style.

One axes per task -> figures/roc_overlay_t{1..5}.png.
- episodic/peak (T3,T4,T5): raw curves.roc.fpr/tpr; AUROC = metrics.auroc.
- landmark (T1,T2): POOL all lead-time landmarks into one curve = unweighted mean
  of each landmark's curves.roc.tpr_at_grid on the shared grid_fpr; pooled AUROC
  = trapz(mean_tpr, grid_fpr).  (the codebase's grid pooling primitive)

Reads only the non-PHI discrimination.json per site. Run with a venv that has
matplotlib + numpy (flair_baseline/.venv).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "flair_baseline"
FIG = Path(__file__).resolve().parent / "figures"

# website palette (static/js/report.js, static/css/site.css)
PANEL = "#f7f7f6"; GRID = "#e7e7e4"; DIAG = "#c9c9c4"; MUTED = "#6b6b6b"; INK = "#1a1a1a"
SITE_COLOR = {"mimic": "#5b86a6", "rush": "#b9744e"}   # blue, terra
SITE_LABEL = {"mimic": "MIMIC", "rush": "Rush"}

TASKS = [
    ("task1_icu_daily_mortality", "T1  ICU daily mortality", True),
    ("task2_icu_daily_ltach",     "T2  LTAC discharge",      True),
    ("task3_extubation_failure_24h", "T3  Extubation failure 24 h", False),
    ("task4_sepsis_abx_6h",       "T4  Sepsis 6 h",          False),
    ("task5_icu_readmission",     "T5  Unplanned ICU readmission", False),
]
SITES = ["mimic", "rush"]


def site_roc(site: str, task: str, landmark: bool):
    """Return (fpr, tpr, auroc) for one site/task; pooled if landmark."""
    p = ROOT / f"{site}_baseline_non_phi_for_upload" / task / "report" / "discrimination.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if landmark:
        lms = d["landmarks"]
        grid = np.asarray(lms[0]["curves"]["roc"]["grid_fpr"], float)
        mat = np.vstack([np.asarray(lm["curves"]["roc"]["tpr_at_grid"], float) for lm in lms])
        tpr = mat.mean(axis=0)                       # POOL: mean TPR over landmarks
        tpr[0], tpr[-1] = 0.0, 1.0
        auroc = float(np.trapezoid(tpr, grid)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, grid))
        return grid, tpr, auroc
    roc = d["curves"]["roc"]
    return np.asarray(roc["fpr"], float), np.asarray(roc["tpr"], float), float(d["metrics"]["auroc"])


def style_ax(ax, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=DIAG, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("false positive rate", fontsize=10, color=MUTED, family="monospace")
    ax.set_ylabel("true positive rate", fontsize=10, color=MUTED, family="monospace")
    ax.set_title(title, fontsize=12.5, color=INK, family="serif")
    ax.tick_params(labelsize=9, colors=MUTED)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family("monospace"); lab.set_fontsize(9)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)


def main():
    for task, title, landmark in TASKS:
        fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=150)
        style_ax(ax, title + ("  (pooled lead times)" if landmark else ""))
        for site in SITES:
            r = site_roc(site, task, landmark)
            if r is None:
                continue
            fpr, tpr, auroc = r
            ax.plot(fpr, tpr, color=SITE_COLOR[site], lw=2, zorder=2,
                    label=f"{SITE_LABEL[site]} (AUROC {auroc:.3f})")
        leg = ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white",
                        edgecolor=GRID)
        for t in leg.get_texts():
            t.set_family("monospace"); t.set_color(MUTED)
        n = task[4]
        out = FIG / f"roc_overlay_t{n}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
