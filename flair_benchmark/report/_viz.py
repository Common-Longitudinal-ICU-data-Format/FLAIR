"""Local site-validation PNGs. NEVER exported — only the JSON leaves a site.

Every plot reads the SAME dicts that were written to disk, so a figure can never
show a number the JSON does not contain. If a panel looks wrong, the JSON is
wrong.

  continuous  landmark_auroc / _auprc / _brier / _slope   each pooled number as
                  a function of hospitalization time, CI ribbon, pooled value as
                  a horizontal reference line
              threshold_sweep                             the decision figure

  episodic    roc, pr, calibration, dca, fairness
              threshold_sweep                             the same decision figure

matplotlib is an optional extra; the caller catches ImportError and skips.
"""

from __future__ import annotations

from pathlib import Path

# The decision figure: which metric moves, and how much, as the threshold moves.
# Sensitivity/specificity/PPV/NPV are the four clinical panels; NNE is the
# alarms-per-true-case number a rare-event screener lives or dies by; flag_rate
# is the workload the site has to staff. (bounded: y-axis pinned to [0, 1])
_SWEEP_PANELS = (
    ("sensitivity", "Sensitivity", True),
    ("specificity", "Specificity", True),
    ("ppv", "PPV", True),
    ("npv", "NPV", True),
    ("nne", "NNE (alarms per true case)", False),
    ("flag_rate", "Flag rate (alert workload)", True),
)

_LANDMARK_PANELS = {
    "landmark_auroc": ("auroc", "AUROC", None),
    "landmark_auprc": ("auprc", "AUPRC", None),
    "landmark_brier": ("brier", "Brier score", None),
    "landmark_slope": ("calibration_slope", "Calibration slope", 1.0),
}


def render_pngs(payloads: dict[str, dict], out_dir: Path,
                mode: str) -> list[Path]:
    """Render the mode's figure set. Returns the paths written."""
    plt = _import_plt()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    written: list[Path] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    hosp = payloads.get("hospitalization_level", {})
    if mode == "continuous":
        lm = payloads.get("landmark", {})
        for name, (key, title, reference) in _LANDMARK_PANELS.items():
            save(_landmark_fig(lm, key, title, reference, plt), f"{name}.png")
    else:
        overall = payloads.get("overall", {})
        save(_roc_fig(overall, plt), "roc.png")
        save(_pr_fig(overall, plt), "pr.png")
        save(_calibration_fig(overall, plt), "calibration.png")
        save(_dca_fig(hosp, plt), "dca.png")
        save(_fairness_fig(hosp, plt), "fairness.png")
    save(_sweep_fig(hosp, plt), "threshold_sweep.png")
    return written


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")            # headless: no display on a cluster node
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------- landmark


def _landmark_fig(lm: dict, key: str, title: str, reference: float | None, plt):
    """One pooled metric as a function of hospitalization time."""
    entries = [e for e in lm.get("by_landmark", []) if e.get(key) is not None]
    unit = (lm.get("metadata") or {}).get("window_unit") or "window"
    fig, ax = plt.subplots(figsize=(7, 4))

    if entries:
        ax.plot([e["hospitalization_time"] for e in entries],
                [e[key] for e in entries],
                marker="o", ms=3, lw=1.6, color="#1f77b4", label=title)
        band = [(e["hospitalization_time"], e[f"{key}_ci"]) for e in entries
                if e.get(f"{key}_ci")]
        if band:
            ax.fill_between([b[0] for b in band], [b[1][0] for b in band],
                            [b[1][1] for b in band], alpha=0.18,
                            color="#1f77b4", lw=0)
        pooled = (lm.get("pooled") or {}).get(key)
        if pooled is not None:
            ci = (lm.get("pooled") or {}).get(f"{key}_ci")
            label = f"pooled = {pooled:.3f}"
            if ci:
                label += f" ({ci[0]:.3f}–{ci[1]:.3f})"
            ax.axhline(pooled, ls="--", lw=1.2, color="#d62728", label=label)
    else:
        ax.text(0.5, 0.5, "no scored landmarks", ha="center", va="center",
                transform=ax.transAxes, color="grey")

    if reference is not None:
        ax.axhline(reference, ls=":", lw=1.0, color="grey",
                   label=f"perfect = {reference:g}")
    ax.set_xlabel(f"Hospitalization time ({unit})")
    ax.set_ylabel(title)
    ax.set_title(f"{title} by hospitalization time")
    ax.grid(alpha=0.25)
    if entries:
        ax.legend(fontsize=8, frameon=False)
    return fig


# ------------------------------------------------------------ episodic curves


def _grid():
    from flair_benchmark.report.overall import GRID
    return GRID


def _roc_fig(overall: dict, plt):
    d = overall.get("discrimination") or {}
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    if d.get("tpr_at_grid"):
        ax.plot(_grid(), d["tpr_at_grid"], lw=1.8, color="#1f77b4")
    ax.plot([0, 1], [0, 1], ls=":", lw=1.0, color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {_ci_label('AUROC', d.get('auroc'), d.get('auroc_ci'))}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    return fig


def _pr_fig(overall: dict, plt):
    d = overall.get("discrimination") or {}
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    if d.get("precision_at_grid"):
        ax.plot(_grid(), d["precision_at_grid"], lw=1.8, color="#2ca02c")
    base = d.get("auprc_baseline")
    if base is not None:
        ax.axhline(base, ls=":", lw=1.0, color="grey",
                   label=f"prevalence = {base:.3f}")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — {_ci_label('AUPRC', d.get('auprc'), d.get('auprc_ci'))}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    return fig


def _calibration_fig(overall: dict, plt):
    """Equal-COUNT reliability points — ten bins spanning the range in use."""
    cal = overall.get("calibration") or {}
    bins = cal.get("quantile_bins") or {}
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    pred = bins.get("bin_pred_prob") or []
    true = bins.get("bin_true_prob") or []
    pts = [(p, t) for p, t in zip(pred, true) if p is not None and t is not None]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", ms=5, lw=1.6, color="#9467bd")
        hi = max(max(p[0] for p in pts), max(p[1] for p in pts)) * 1.1
    else:
        hi = 1.0
        ax.text(0.5, 0.5, "no bin cleared the cell-count floor", ha="center",
                va="center", transform=ax.transAxes, color="grey", fontsize=8)
    ax.plot([0, hi], [0, hi], ls=":", lw=1.0, color="grey")
    bits = []
    if cal.get("slope") is not None:
        bits.append(f"slope {cal['slope']:.2f}")
    if cal.get("intercept") is not None:
        bits.append(f"intercept {cal['intercept']:+.2f}")
    if cal.get("brier") is not None:
        bits.append(f"Brier {cal['brier']:.4f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title("Calibration — " + ", ".join(bits) if bits
                 else "Calibration")
    ax.grid(alpha=0.25)
    return fig


def _dca_fig(hosp: dict, plt):
    """The decision curve, read straight off the sweep rows."""
    rows = [r for r in hosp.get("by_threshold", [])
            if r.get("net_benefit") is not None]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    if rows:
        x = [r["threshold"] for r in rows]
        ax.plot(x, [r["net_benefit"] for r in rows], lw=1.8,
                color="#1f77b4", label="model")
        band = [(r["threshold"], r["net_benefit_ci"]) for r in rows
                if r.get("net_benefit_ci")]
        if band:
            ax.fill_between([b[0] for b in band], [b[1][0] for b in band],
                            [b[1][1] for b in band], alpha=0.18,
                            color="#1f77b4", lw=0)
        ax.plot(x, [r.get("net_benefit_treat_all") for r in rows], lw=1.2,
                ls="--", color="#d62728", label="treat all")
        ax.axhline(0.0, lw=1.2, ls=":", color="grey", label="treat none")
        # keep the y-axis on the model's scale: treat-all dives toward -inf as
        # t -> 1 and would otherwise flatten the curve that matters.
        ys = [r["net_benefit"] for r in rows]
        ax.set_ylim(min(min(ys), 0.0) - 0.02, max(max(ys), 0.0) + 0.02)
        ax.legend(fontsize=8, frameon=False)
    _mark_clinical(ax, hosp)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve")
    ax.grid(alpha=0.25)
    return fig


def _fairness_fig(hosp: dict, plt):
    """Subgroup AUROC with CIs — does the model rank every group as well?"""
    fair = hosp.get("fairness") or {}
    rows = []
    for col, levels in (fair.get("subgroups") or {}).items():
        for level, block in levels.items():
            if block.get("auroc") is not None:
                rows.append((f"{col.replace('_category', '')}: {level}",
                             block["auroc"], block.get("auroc_ci"),
                             block.get("n")))
    fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.42 * len(rows) + 1.4)))
    if not rows:
        ax.text(0.5, 0.5, "no demographics on the predictions frame",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_axis_off()
        return fig

    ys = list(range(len(rows)))
    ax.errorbar(
        [r[1] for r in rows], ys,
        xerr=[[max(0.0, r[1] - r[2][0]) if r[2] else 0.0 for r in rows],
              [max(0.0, r[2][1] - r[1]) if r[2] else 0.0 for r in rows]],
        fmt="o", ms=5, lw=1.2, color="#1f77b4", ecolor="#8fb8d8", capsize=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}  (n={r[3]})" for r in rows], fontsize=8)
    ax.axvline(0.5, ls=":", lw=1.0, color="grey")
    ax.set_xlabel("AUROC (95% CI)")
    ax.set_title(f"Subgroup discrimination at threshold "
                 f"{fair.get('at_threshold')}")
    ax.grid(alpha=0.25, axis="x")
    ax.invert_yaxis()
    return fig


# ------------------------------------------------------------ threshold sweep


def _sweep_fig(hosp: dict, plt):
    """THE decision figure: which metric moves, and how much, with threshold."""
    rows = hosp.get("by_threshold", [])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    surface = (hosp.get("metadata") or {}).get("surface", "")
    fig.suptitle(f"Metrics across the threshold sweep ({surface} surface)",
                 fontsize=12)

    for ax, (key, title, bounded) in zip(axes.ravel(), _SWEEP_PANELS):
        pts = [(r["threshold"], r[key]) for r in rows if r.get(key) is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.6,
                    color="#1f77b4")
            # Both endpoints must be numeric. A CI can legitimately be half-open
            # — NNE is 1/PPV, so a PPV bound at zero gives nne_ci = [4.8, None] —
            # and a truthiness test alone lets that through to fill_between,
            # where matplotlib's isfinite raises TypeError on the None.
            band = [(r["threshold"], r[f"{key}_ci"]) for r in rows
                    if r.get(f"{key}_ci")
                    and all(isinstance(b, (int, float)) for b in r[f"{key}_ci"])]
            if band:
                ax.fill_between([b[0] for b in band], [b[1][0] for b in band],
                                [b[1][1] for b in band], alpha=0.18,
                                color="#1f77b4", lw=0)
            if bounded:
                ax.set_ylim(0, 1.02)
            else:
                ax.set_yscale("log")     # NNE spans orders of magnitude
        else:
            ax.text(0.5, 0.5, "not estimable", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=8)
        _mark_clinical(ax, hosp)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)

    for ax in axes[1]:
        ax.set_xlabel("Threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Value")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _mark_clinical(ax, hosp: dict) -> None:
    """Vertical markers: the shipped policy (red) and where Youden lands (green)."""
    t0 = (hosp.get("metadata") or {}).get("clinical_threshold")
    if t0 is not None:
        ax.axvline(t0, color="#d62728", lw=1.1, ls="--", alpha=0.8)
    youden = next((s for s in hosp.get("selected_thresholds", [])
                   if s.get("rule") == "youden_j"), None)
    if youden and youden.get("threshold") is not None:
        ax.axvline(youden["threshold"], color="#2ca02c", lw=1.1, ls=":",
                   alpha=0.8)


def _ci_label(name: str, value, ci) -> str:
    if value is None:
        return f"{name} n/a"
    if ci:
        return f"{name} {value:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"
    return f"{name} {value:.3f}"
