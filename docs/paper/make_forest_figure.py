#!/usr/bin/env python3
"""Figure 2 -- multi-site AUROC forest plot (the headline visual).

Five task panels (T1-T5). Within each, one marker per consortium site with a
95% CI whisker:
  * MIMIC-CLIF  -- solid blue diamond (internal holdout; also drawn as the
                   per-panel reference line).
  * Rush        -- solid terracotta circle (observed external).
  * Sites C-H   -- open grey circles (ILLUSTRATIVE, pending federated runs).

A faint diagonal "ILLUSTRATIVE" watermark and an explicit legend entry keep the
observed/projected distinction unambiguous. Reads all numbers from
flair_paper_data.py so the figure can never disagree with the tables.

Run with a Python that has matplotlib + numpy (repo .venv).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flair_paper_data as D

FIG = Path(__file__).resolve().parent / "figures"
OUT = FIG / "figure2_forest_auroc.png"

C = D.COLOR


def site_style(site: str):
    info = D.ALL_SITES[site]
    if site == "mimic":
        return dict(color=C["mimic"], marker="D", ms=8, mfc=C["mimic"], mec=C["mimic"], zorder=5)
    if site == "rush":
        return dict(color=C["rush"], marker="o", ms=8, mfc=C["rush"], mec=C["rush"], zorder=5)
    # illustrative sites: open grey markers
    return dict(color=C["synth"], marker="o", ms=7, mfc="white", mec=C["synth"], zorder=4)


def main():
    sites = D.SITE_ORDER
    y = {s: len(sites) - i for i, s in enumerate(sites)}  # top-to-bottom

    fig, axes = plt.subplots(1, 5, figsize=(14.5, 5.3), dpi=150, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (tid, short, mode, thr, direction, prev, grid) in zip(axes, D.TASKS):
        ax.set_facecolor(C["panel"])
        ax.grid(True, axis="x", color=C["grid"], linewidth=1)
        ax.set_axisbelow(True)

        mimic_auroc = D.REAL["mimic"]["auroc"][tid]
        # reference line at MIMIC internal AUROC
        ax.axvline(mimic_auroc, color=C["mimic"], lw=1.4, ls=(0, (5, 3)), zorder=2, alpha=0.9)
        # chance line
        ax.axvline(0.5, color=C["diag"], lw=1.2, ls=(0, (2, 3)), zorder=1)
        # below-useful band (< 0.5)
        ax.axvspan(0.30, 0.5, color=C["band"], alpha=0.55, zorder=0)

        for s in sites:
            info = D.ALL_SITES[s]
            a = info["auroc"][tid]
            lo, hi = info["ci"][tid]
            st = site_style(s)
            ax.plot([lo, hi], [y[s], y[s]], color=st["color"], lw=1.6,
                    alpha=0.9 if info["observed"] else 0.7, zorder=st["zorder"] - 1)
            ax.plot([a], [y[s]], ls="none", **st)

        ax.set_xlim(0.30, 0.92)
        ax.set_ylim(0.4, len(sites) + 0.6)
        ax.set_title(f"{tid}  {short}\n({mode})", fontsize=11, color=C["ink"], family="serif")
        ax.set_xlabel("AUROC", fontsize=9.5, color=C["muted"], family="monospace")
        ax.set_xticks([0.4, 0.6, 0.8])
        ax.tick_params(labelsize=8.5, colors=C["muted"])
        for lab in ax.get_xticklabels():
            lab.set_family("monospace")
        for sp in ax.spines.values():
            sp.set_edgecolor(C["grid"])

    # y tick labels = site names (only on the left axis)
    axes[0].set_yticks([y[s] for s in sites])
    axes[0].set_yticklabels([D.ALL_SITES[s]["label"] for s in sites],
                            fontsize=9.5, family="monospace", color=C["ink"])
    for lab, s in zip(axes[0].get_yticklabels(), sites):
        if not D.ALL_SITES[s]["observed"]:
            lab.set_color(C["muted"])

    # faint diagonal watermark over the illustrative region
    fig.text(0.60, 0.45, "ILLUSTRATIVE", fontsize=44, color=C["synth"],
             alpha=0.10, rotation=24, ha="center", va="center",
             family="serif", weight="bold", zorder=0)

    # legend
    handles = [
        plt.Line2D([], [], color=C["mimic"], marker="D", ls="none", ms=8, label="MIMIC-CLIF (observed, internal)"),
        plt.Line2D([], [], color=C["rush"], marker="o", ls="none", ms=8, label="Rush (observed, external)"),
        plt.Line2D([], [], color=C["synth"], marker="o", mfc="white", ls="none", ms=7,
                   label="Sites C-H (illustrative, pending federated runs)"),
        plt.Line2D([], [], color=C["mimic"], lw=1.4, ls=(0, (5, 3)), label="MIMIC internal AUROC (reference)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5,
                     frameon=True, facecolor="white", edgecolor=C["grid"],
                     bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_family("monospace")
        t.set_color(C["muted"])

    fig.suptitle("Per-site discrimination of the frozen XGBoost baseline across the CLIF consortium",
                 fontsize=13.5, color=C["ink"], family="serif", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
