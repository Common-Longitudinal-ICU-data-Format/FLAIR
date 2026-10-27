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


# --------------------------------------------------------------------- STROBE
#
# Written by `flair load-task` next to table1.json (non-PHI folder), drawn only from that
# dict: its `attrition` steps plus by_split / by_label (suppressed cells print as
# "<10" and count as 0 for geometry).

_RETAINED, _EXCLUDED = "#2a78d6", "#eb6834"      # one validated adjacent pair
_INK, _INK_2, _SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def render_strobe_pngs(table1: dict, out_dir: Path) -> list[Path]:
    """Write out_dir/strobe.png (flow diagram) + out_dir/strobe_attrition.png."""
    plt = _import_plt()
    written = []
    for name, fig in (("strobe.png", _strobe_flow_fig(table1, plt)),
                      ("strobe_attrition.png", _strobe_attrition_fig(table1, plt))):
        path = Path(out_dir) / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
        plt.close(fig)
        written.append(path)
    return written


def _fmt(v) -> str:
    return f"{v:,}" if isinstance(v, int) else str(v)


def _num(v) -> int:
    return v if isinstance(v, int) else 0


def _unit(step: dict) -> str:
    return "encounters" if "n_encounters" in step else "hospitalizations"


# Plot wording only: the attrition keys stay n_encounters / n_hospitalizations,
# but a stitched encounter is labelled by its column name so the figure matches
# the cohort file a reader will open.
JOIN_LABEL = "hospitalization_join_id"
_UNIT_LABEL = {"encounters": JOIN_LABEL, "hospitalizations": "hospitalizations"}


def _strobe_box(ax, x, y, w, h, text, edge, *, muted=False, bold_first=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
        fc=edge + "14", ec=edge, lw=1.2 if not muted else 0.8))
    lines = text.split("\n")
    color = _INK_2 if muted else _INK
    if bold_first:
        rest = [ln for ln in lines[1:] if ln]
        line_h = 0.17                            # inches at 8pt, linespacing 1.4
        top = y - line_h * len(rest) / 2         # centre title + details as a block
        ax.text(x, top, lines[0], ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=color)
        ax.text(x, top + line_h * (len(rest) + 1) / 2, "\n".join(rest),
                ha="center", va="center", fontsize=8, color=color,
                linespacing=1.4)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=color,
                linespacing=1.4)


def _arrow(ax, xy_from, xy_to, color=_INK_2):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=10))


def _strobe_flow_fig(table1: dict, plt):
    """Classic STROBE flow: retained boxes down the centre, exclusions right."""
    import textwrap

    steps = table1.get("attrition", [])
    n_of = lambda strata, k: ((table1.get(strata) or {}).get(k) or {}).get("n")
    box_h, gap, main_x, main_w, side_x, side_w = 0.95, 0.75, 3.0, 4.6, 7.9, 3.4
    pitch = box_h + gap
    height = 1.0 + pitch * len(steps) + 1.2
    fig = plt.figure(figsize=(10.5, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10.5)
    ax.set_ylim(height, 0)                      # 1 unit = 1 inch, top-down
    ax.axis("off")
    fig.patch.set_facecolor(_SURFACE)
    task = (table1.get("metadata") or {}).get("task", "")
    ax.text(0.3, 0.45, f"STROBE cohort flow — {task}", fontsize=12,
            fontweight="bold", color=_INK, va="center")

    for i, s in enumerate(steps):
        y = 1.0 + box_h / 2 + i * pitch
        unit = _unit(s)
        detail = f"n = {_fmt(s.get(f'n_{unit}'))} {_UNIT_LABEL[unit]}"
        extra = []
        if "n_patients" in s:
            extra.append(f"{_fmt(s['n_patients'])} patients")
        if s.get("n_rows") != s.get(f"n_{unit}"):
            extra.append(f"{_fmt(s['n_rows'])} rows")
        title = textwrap.shorten(s["step"], 58, placeholder="…")
        _strobe_box(ax, main_x, y, main_w, box_h,
                    "\n".join([title, detail, " · ".join(extra)]).rstrip(),
                    _RETAINED, bold_first=True)
        if i == 0:
            continue
        top = y - box_h / 2 - gap
        _arrow(ax, (main_x, top), (main_x, y - box_h / 2))
        mid = top + gap / 2
        excluded = s.get(f"n_excluded_{unit}")
        if excluded is None:                    # unit switched at this step
            ax.text(main_x + 0.15, mid, f"hospitalizations merged into {JOIN_LABEL}",
                    fontsize=7.5, color=_INK_2, style="italic", va="center")
            continue
        muted = excluded == 0
        pts = s.get("n_excluded_patients")
        text = f"Excluded: {_fmt(excluded)} {_UNIT_LABEL[unit]}"
        if pts is not None:
            text += f"\n{_fmt(pts)} patients"
        ax.plot([main_x, side_x - side_w / 2], [mid, mid], color=_INK_2, lw=0.8,
                alpha=0.5 if muted else 1)
        _strobe_box(ax, side_x, mid, side_w, 0.55, text,
                    _INK_2 if muted else _EXCLUDED, muted=muted)

    # Final cohort -> train / test, with the label split in the parent box.
    y_last = 1.0 + box_h / 2 + (len(steps) - 1) * pitch
    if table1.get("by_label"):
        ax.text(main_x, y_last + box_h / 2 + 0.18,
                f"label 1: {_fmt(n_of('by_label', '1'))}   ·   label 0: "
                f"{_fmt(n_of('by_label', '0'))}  ({JOIN_LABEL})",
                ha="center", va="center", fontsize=8, color=_INK_2)
    y_split = y_last + box_h / 2 + 1.0
    for x, name in ((main_x - 1.3, "train"), (main_x + 1.3, "test")):
        _arrow(ax, (main_x, y_last + box_h / 2 + 0.32), (x, y_split - 0.3))
        _strobe_box(ax, x, y_split, 2.2, 0.6,
                    f"{name}\n{_fmt(n_of('by_split', name))} {JOIN_LABEL}",
                    _RETAINED, bold_first=True)
    return fig


def _strobe_attrition_fig(table1: dict, plt):
    """Retained (blue) + excluded-at-this-step (orange) per step, top-down."""
    steps = table1.get("attrition", [])
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(steps) + 1.4))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ys = list(range(len(steps)))
    kept = [_num(s.get(f"n_{_unit(s)}")) for s in steps]
    lost = [_num(s.get(f"n_excluded_{_unit(s)}")) for s in steps]
    ax.barh(ys, kept, height=0.6, color=_RETAINED, label="retained",
            edgecolor=_SURFACE, linewidth=2)
    ax.barh(ys, lost, left=kept, height=0.6, color=_EXCLUDED,
            label="excluded at this step", edgecolor=_SURFACE, linewidth=2)
    span = max((k + l for k, l in zip(kept, lost)), default=1) or 1
    for y, s, k, l in zip(ys, steps, kept, lost):
        label = _fmt(s.get(f"n_{_unit(s)}"))
        if l and isinstance(s.get(f"n_excluded_{_unit(s)}"), int):
            label += f"   (−{l:,}, −{100 * l / (k + l):.1f}%)"
        ax.text(k + l + span * 0.01, y, label, va="center", fontsize=8,
                color=_INK)
    switch = next((i for i, s in enumerate(steps) if "n_encounters" in s), None)
    if switch:
        ax.axhline(switch - 0.5, color=_INK_2, ls="--", lw=0.8)
        ax.text(span * 1.3, switch - 0.58, f"hospitalizations ↑   {JOIN_LABEL} ↓",
                ha="right", va="bottom", fontsize=7.5, color=_INK_2)
    ax.set_yticks(ys, [s["step"] for s in steps], fontsize=8, color=_INK)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.3)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="x", labelsize=8, colors=_INK_2)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    task = (table1.get("metadata") or {}).get("task", "")
    ax.set_title(f"Cohort attrition by step — {task}", fontsize=11,
                 color=_INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig
