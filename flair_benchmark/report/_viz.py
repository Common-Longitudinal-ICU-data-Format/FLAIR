"""Sanity-check PNGs rendered FROM the report JSON dicts.

Deliberately consumes the already-built dicts (never the parquet) so a
rendered PNG proves the JSON alone is sufficient for the FLAIR website's
interactive plots. Requires the optional extra:

    pip install flair-benchmark[viz]

Two entry points, one per report mode:
  render_sanity_pngs     : episodic bundle (roc_pr / calibration / dca / fairness).
  render_continuous_pngs : continuous bundle — peak section (peak_roc_pr,
                           peak_fairness_auroc), per-landmark curves on a
                           unit-aware lead-time axis (days for tasks 1/2, hours
                           for task 4), leadtime sensitivity with the red
                           mean±SD intervention line, and the operating-points
                           section (confusion matrix + optional sub-analyses).
"""

from __future__ import annotations

from pathlib import Path


def _import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for --viz. "
            "Install it with: pip install 'flair-benchmark[viz]'"
        ) from exc


def _make_saver(out_dir: Path, plt):
    """Fresh out_dir saver; clears stale PNGs so dropped sections leave no orphans."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    written: list[Path] = []

    def _save(fig, name: str, tight: bool = True) -> None:
        path = out_dir / name
        # tight=False for figures whose layout is already pinned by a gridspec and
        # holds an axes tight_layout cannot measure (the scorecard's value table),
        # which would otherwise warn and reflow the panels.
        if tight:
            fig.tight_layout()
        # bbox_inches="tight" keeps below-axis artists (the KM "No. at risk"
        # row) inside the saved image.
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return _save, written


def _ci_label(name: str, val, ci) -> str:
    """'AUROC=0.82 [0.79–0.85]' — value with its 95% CI when available."""
    if val is None:
        return f"{name}=NA"
    if ci and ci[0] is not None and ci[1] is not None:
        return f"{name}={val:g} [{ci[0]:g}–{ci[1]:g}]"
    return f"{name}={val:g}"


def render_sanity_pngs(reports: dict[str, dict], out_dir: Path) -> list[Path]:
    """Render one PNG per applicable section. Returns the written paths."""
    plt = _import_plt()
    _save, written = _make_saver(out_dir, plt)

    ops = reports.get("operating_points")
    _render_roc_pr(reports.get("discrimination", {}), _save, plt, ops=ops)
    _render_scorecard(ops, _save, plt)

    cal = reports.get("calibration", {})
    cal_curves, cal_metrics = cal.get("curves", {}), cal.get("metrics", {})
    bins = cal_curves.get("reliability")
    if bins:
        fig, ax = plt.subplots(figsize=(5, 5))
        pairs = [(p, t) for p, t in zip(bins["bin_pred_prob"], bins["bin_true_prob"])
                 if p is not None and t is not None]
        if pairs:
            ax.scatter(*zip(*pairs), label="bins (n>=10)")
        smoothed = cal_curves.get("smoothed")
        if smoothed:
            ax.plot(smoothed["pred_grid"], smoothed["obs_smoothed"],
                    label="logistic calibration fit")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
        ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
               title="Calibration (reliability)")
        # All scalars (slope/intercept/Brier with 95% CIs, ECE, O:E) collected into a
        # single boxed annotation at lower-right — keeps them off the points/curve and
        # avoids overlapping the recalibration warning. Calibration is threshold-free,
        # so no operating-threshold is shown here.
        stats = "\n".join([
            _ci_label("slope", cal_metrics.get("slope"), cal_metrics.get("slope_ci")),
            _ci_label("intercept", cal_metrics.get("intercept"),
                      cal_metrics.get("intercept_ci")),
            _ci_label("Brier", cal_metrics.get("brier"), cal_metrics.get("brier_ci")),
            f"ECE={cal_metrics.get('ece')}",
            f"O:E={cal_metrics.get('o_e_ratio')}",
        ])
        ax.text(0.97, 0.03, stats, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        slope = cal_metrics.get("slope")
        if slope is not None and abs(slope - 1) > 0.3:
            verdict = ("over-confident (scores too extreme)" if slope < 1
                       else "under-confident (scores too compressed)")
            # right-centre is the only empty quadrant (points sit low-left, legend
            # upper-left, stats box lower-right, title on top) — no collisions.
            ax.text(0.97, 0.62, f"probabilities uncalibrated\n{verdict}",
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=8, color="firebrick")
        ax.legend(loc="upper left", fontsize=8)
        _save(fig, "calibration.png")

    _render_dca(reports.get("dca", {}), _save, plt)

    _render_fairness_auroc(reports.get("fairness", {}), _save, plt)

    return written


def _render_roc_pr(disc: dict, _save, plt, prefix: str = "",
                   title_note: str = "", ops: dict | None = None) -> None:
    """ROC + PR panel from a discrimination payload (episodic or peak section).

    The operating point comes from whichever place the mode keeps it: the
    continuous peak section still carries `metrics.operating_point` inline, while
    the episodic report has hoisted it into operating_points.json — so `ops` (the
    scorecard payload) marks the WHOLE grid on the ROC, not just one point.
    """
    curves, metrics = disc.get("curves", {}), disc.get("metrics", {})
    if "roc" not in curves:
        return
    roc, pr = curves["roc"], curves["pr"]
    op = metrics.get("operating_point", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(roc["fpr"], roc["tpr"])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    if not op and ops:
        _mark_grid_on_roc(axes[0], ops)
    # operating point at the clinical threshold, with sens/spec 95% CIs as
    # error bars (x = FPR = 1-specificity, y = TPR = sensitivity).
    sens, spec = op.get("sensitivity"), op.get("specificity")
    if sens is not None and spec is not None:
        sci, spci = op.get("sensitivity_ci"), op.get("specificity_ci")
        yerr = [[sens - sci[0]], [sci[1] - sens]] if sci else None
        # FPR = 1 - specificity, so its CI bounds flip.
        xerr = [[spci[1] - spec], [spec - spci[0]]] if spci else None
        axes[0].errorbar([1 - spec], [sens], yerr=yerr, xerr=xerr,
                         fmt="o", color="red", capsize=3, ms=5,
                         label=f"op @ {op.get('threshold')}")
        axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set(xlabel="FPR", ylabel="TPR",
                title=f"ROC{title_note} "
                      f"({_ci_label('AUROC', metrics.get('auroc'), metrics.get('auroc_ci'))})")
    axes[1].plot(pr["recall"], pr["precision"])
    base = metrics.get("auprc_baseline")
    if base is not None:
        axes[1].axhline(base, color="k", ls="--", lw=0.8,
                        label=f"prevalence {base:.4f}")
        axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title=f"PR{title_note} "
                      f"({_ci_label('AUPRC', metrics.get('auprc'), metrics.get('auprc_ci'))}, "
                      f"lift×{metrics.get('auprc_lift')})")
    _save(fig, f"{prefix}roc_pr.png")


def _render_dca(dca: dict, _save, plt) -> None:
    """dca.png — the net-benefit curve, ON ITS OWN AXIS, with its verdict stated.

    Everything here comes from dca.json and nothing from the scorecard. That is the
    whole point: a "below" task's curve lives on the COMPLEMENT axis (task3's x is
    the probability of successful extubation, so its 0.15 policy sits at x=0.85),
    and borrowing the scorecard's alarm-basis number would plot a different quantity
    at the wrong place — a red dot floating off its own curve.
    """
    curves, metrics = dca.get("curves", {}), dca.get("metrics", {})
    if not curves.get("thresholds"):
        return
    md = dca.get("metadata", {})
    action = md.get("action", "treat")
    thr = curves["thresholds"]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(thr, [v if v is not None else float("nan")
                  for v in curves["net_benefit_model"]], label="model")
    ax.plot(thr, curves["net_benefit_all"], label=f"'{action}' on all")
    ax.axhline(0.0, color="k", lw=0.8, label=f"'{action}' on none")
    if metrics.get("useful_range"):
        lo, hi = metrics["useful_range"]["low"], metrics["useful_range"]["high"]
        ax.axvspan(lo, hi, alpha=0.15, color="green",
                   label=f"model wins ({lo:g}–{hi:g})")

    ct = metrics.get("clinical_threshold")           # already on THIS axis
    verdict = metrics.get("verdict") or {}
    if ct is not None:
        t_label = md.get("clinical_threshold_on_label_scale")
        lbl = f"shipped policy ({ct:g})"
        if t_label is not None and t_label != ct:
            lbl += f"  [= p(event) {t_label:g}]"
        ax.axvline(ct, color="red", lw=1.2, ls=":", label=lbl)
        nb, ci = verdict.get("model"), verdict.get("model_ci")
        if nb is not None:
            yerr = ([[nb - ci[0]], [ci[1] - nb]]
                    if ci and ci[0] is not None else None)
            ax.errorbar([ct], [nb], yerr=yerr, fmt="o", color="red",
                        capsize=3, ms=5)

    if verdict.get("beats"):
        # state the conclusion instead of leaving three lines to be compared by eye.
        ok = verdict.get("clinically_useful")
        ax.text(0.5, -0.30, f"At the shipped policy: {verdict['beats']}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                color="#1f6f3f" if ok else "firebrick", wrap=True)

    prev = dca.get("prevalence") or 0.0
    # sentence-case the first letter ONLY — .capitalize() would lowercase the rest
    # and turn "unplanned ICU readmission" into "unplanned icu readmission".
    xlabel = md.get("x_axis", "threshold probability")
    ax.set(xlabel=xlabel[:1].upper() + xlabel[1:],
           ylabel="Net benefit", title="Decision curve analysis",
           xlim=(0.0, 1.0),
           # y is clamped so the treat-all line diving to large negatives at high
           # thresholds doesn't crush the detail.
           ylim=(min(-0.01, -prev * 0.5), max(0.01, prev * 1.2)))
    ax.legend(fontsize=8)
    _save(fig, "dca.png")


def _mark_grid_on_roc(ax, ops: dict) -> None:
    """Plot every scorecard threshold on the ROC; the clinical one carries its CIs.

    Showing the whole grid is the point of the episodic scorecard: it makes the
    cost of moving the policy legible (how much sensitivity you buy per point of
    specificity you give up) instead of asserting one operating point.
    """
    pts = [(e["metrics"].get("fpr"), e["metrics"].get("sensitivity"),
            e.get("threshold"), e.get("is_clinical"), e["metrics"])
           for e in ops.get("by_threshold", []) if e.get("metrics")]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if not pts:
        return
    off = [(x, y) for x, y, _t, clin, _m in pts if not clin]
    if off:
        ax.plot(*zip(*off), "o", color="#5b86a6", ms=4, alpha=0.8,
                label="threshold grid (±0.05, ±0.10)")
    for x, y, t, clin, m in pts:
        if not clin:
            ax.annotate(f"{t:g}", (x, y), textcoords="offset points",
                        xytext=(4, -7), fontsize=6.5, color="#33566b")
            continue
        sci, spci = m.get("sensitivity_ci"), m.get("fpr_ci")
        yerr = [[y - sci[0]], [sci[1] - y]] if sci else None
        xerr = [[x - spci[0]], [spci[1] - x]] if spci else None
        ax.errorbar([x], [y], yerr=yerr, xerr=xerr, fmt="o", color="red",
                    capsize=3, ms=5, label=f"clinical @ {t:g}")
    ax.legend(loc="lower right", fontsize=8)


# KPI panels, in the order a reader asks about them: can we catch the event, what
# does it cost, and is acting worth it. All ALARM basis except the last, which is
# the DECISION basis (see operating_points.py) — hence the separate lookup.
_SCORECARD_PANELS = [
    ("sensitivity", "Sensitivity", "sensitivity_ci"),
    ("specificity", "Specificity", "specificity_ci"),
    ("ppv", "PPV", "ppv_ci"),
    ("npv", "NPV", "npv_ci"),
    ("f1", "F1", "f1_ci"),
    ("flag_rate", "Alert rate", None),
    ("nne", "NNE (alerts per TP)", "nne_ci"),
]
_SCORECARD_TABLE = [("sensitivity", "sens"), ("specificity", "spec"),
                    ("ppv", "PPV"), ("npv", "NPV"), ("f1", "F1"),
                    ("fpr", "FPR"), ("nne", "NNE"), ("flag_rate", "alert")]
_DECISION_TABLE = [("net_benefit", "net ben"), ("net_benefit_treat_all", "vs all"),
                   ("clinically_useful", "useful?")]


def _render_scorecard(ops: dict | None, _save, plt) -> None:
    """operating_points.png — every KPI against the threshold grid, + a value table.

    One panel per KPI with its 95% CI band, the clinical threshold marked, and the
    exact numbers tabulated underneath. This is the picture the JSON is for: what
    the decision costs at, and either side of, the shipped policy.
    """
    rows = [e for e in (ops or {}).get("by_threshold", []) if e.get("metrics")]
    if not rows:
        return
    thr = [e["threshold"] for e in rows]
    clinical = next((e["threshold"] for e in rows if e.get("is_clinical")), None)

    n_panels = len(_SCORECARD_PANELS) + 1        # + the decision (net-benefit) panel
    table_rows = _SCORECARD_TABLE + _DECISION_TABLE
    ncol = 4
    nrow = -(-n_panels // ncol)
    fig = plt.figure(figsize=(3.2 * ncol, 2.5 * nrow + 0.3 * len(table_rows) + 1))
    # the value table is a real subplot spanning the bottom row, not a floating
    # add_axes — _make_saver runs tight_layout(), which would shove a free axes.
    gs = fig.add_gridspec(nrow + 1, ncol, hspace=0.5, wspace=0.3,
                          left=0.06, right=0.98, top=0.90, bottom=0.04,
                          height_ratios=[*([2.4] * nrow), 0.3 * len(table_rows)])
    for i, (key, title, ci_key) in enumerate(_SCORECARD_PANELS):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        _panel(ax, thr, [e["metrics"] for e in rows], key, ci_key, clinical)
        ax.set_title(title, fontsize=9)

    # The decision panel is the one that answers "should we act here at all", so it
    # carries the treat-all line: without it a rising net-benefit curve looks like
    # good news even when doing it to everyone would score higher.
    i = len(_SCORECARD_PANELS)
    ax = fig.add_subplot(gs[i // ncol, i % ncol])
    dec = [e.get("decision", {}) for e in rows]
    _panel(ax, thr, dec, "net_benefit", "net_benefit_ci", clinical)
    ref = [(t, d.get("net_benefit_treat_all")) for t, d in zip(thr, dec)]
    ref = [(t, v) for t, v in ref if v is not None]
    if ref:
        ax.plot(*zip(*ref), "--", color="#c25b5b", lw=1.0, label="act on all")
        ax.legend(fontsize=6.5, loc="best")
    ax.set_title("Net benefit (decision basis)", fontsize=9)

    tax = fig.add_subplot(gs[nrow, :])
    tax.axis("off")
    cells = ([[_fmt(e["metrics"].get(k)) for e in rows]
              for k, _lbl in _SCORECARD_TABLE]
             + [[_fmt(e.get("decision", {}).get(k)) for e in rows]
                for k, _lbl in _DECISION_TABLE])
    table = tax.table(
        cellText=cells,
        rowLabels=[lbl for _k, lbl in table_rows],
        colLabels=[f"[{t:g}]" if t == clinical else f"{t:g}" for t in thr],
        cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    if clinical in thr:
        c = thr.index(clinical)
        for r in range(len(table_rows) + 1):
            table[r, c].set_facecolor("#d6e6f2")   # the shipped policy

    md = ops.get("metadata") or {}
    note = (f"clinical threshold {clinical:g}" if clinical is not None
            else "no clinical threshold on the grid")
    fig.suptitle(f"Operating-points scorecard — {note} "
                 f"(direction: {md.get('clinical_direction', 'above')}); "
                 f"highlighted column is the shipped policy", fontsize=10)
    _save(fig, "operating_points.png", tight=False)


def _panel(ax, thr: list, blocks: list[dict], key: str, ci_key: str | None,
           clinical) -> None:
    """One KPI against the threshold grid: point line + CI band + policy marker."""
    pairs = [(t, b.get(key)) for t, b in zip(thr, blocks)]
    pairs = [(t, v) for t, v in pairs if v is not None]
    if pairs:
        ax.plot(*zip(*pairs), "o-", color="#5b86a6", ms=4)
        if ci_key:
            band = [(t, b.get(ci_key)) for t, b in zip(thr, blocks)]
            band = [(t, c) for t, c in band
                    if c and c[0] is not None and c[1] is not None]
            if band:
                bt, bc = zip(*band)
                ax.fill_between(bt, [c[0] for c in bc], [c[1] for c in bc],
                                color="#5b86a6", alpha=0.18, lw=0)
    if clinical is not None:
        ax.axvline(clinical, color="red", lw=1.0, ls="--")
    ax.tick_params(labelsize=7)
    ax.set_xticks(thr)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:g}"


def _render_fairness_auroc(fairness: dict, _save, plt, prefix: str = "",
                           title_note: str = "") -> None:
    """Horizontal subgroup-AUROC bars from a fairness payload."""
    rows = [(f"{col}:{lvl}", m["auroc"])
            for col, levels in fairness.get("subgroups", {}).items()
            for lvl, m in levels.items() if m.get("auroc") is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(rows))))
    names, aucs = zip(*rows)
    ax.barh(names, aucs)
    ax.set(xlabel="AUROC", title=f"Subgroup AUROC{title_note}", xlim=(0.4, 1.0))
    _save(fig, f"{prefix}fairness_auroc.png")


def render_continuous_pngs(payloads: dict[str, dict], out_dir: Path) -> list[Path]:
    """Local site-validation PNGs for the continuous report (never exported).

    Peak section: peak_roc_pr / peak_fairness_auroc (stay-peak screening view).
    FORWARD landmark curves on a Kaplan-Meier-style axis — hospitalization
    day/hour ascending 1..horizon (days for T1/T2, hours for T4), each with a
    "No. at risk" row beneath the axis (a true KM count: stays still producing
    predictions, i.e. still at risk and event-free):
      landmark_auc_by_hospital_time : AUROC +/- 95% CI per hospitalization time
                                      (the time-varying / dynamic AUC curve)
      sensitivity_by_hospital_time  : detection sensitivity +/- 95% CI, with
                                      the red mean±SD intervention line (from the
                                      config `display`) when the task declares
                                      an intervention/treatment column
      landmark_calibration          : calibration slope +/- 95% CI (drift)
      landmark_reliability          : reliability curves at 3 hospitalization
                                      times (earliest / mid / latest scored)
      landmark_risk_trajectory      : mean predicted risk by eventual outcome
    Operating points:
      operating_points_confusion    : per-stay ALERT × LABEL 2×2 at the three
                                      thresholds (clinical ± 0.05), TP emphasized
      operating_points_subanalyses  : per-threshold sub-analyses (only for tasks
                                      with an intervention/treatment column) —
                                      rows = thresholds, cols = label-positive
                                      hours-before-treatment / treated-negative Δ
                                      / untreated misfire rate
    Suppressed (n<10) or undefined landmarks are dropped from the curves
    (gaps) but still show as "<10" in the at-risk row.
    """
    plt = _import_plt()
    _save, written = _make_saver(out_dir, plt)

    disc = payloads.get("discrimination", {})
    fair = payloads.get("fairness", {})
    _render_roc_pr(disc.get("peak") or {}, _save, plt,
                   prefix="peak_", title_note=" — stay peak")
    _render_fairness_auroc(fair.get("peak") or {}, _save, plt,
                           prefix="peak_", title_note=" — stay peak")

    pillars = {k: payloads.get(k, {}).get("landmarks") or []
               for k in ("discrimination", "calibration")}
    leadtime = payloads.get("leadtime", {})
    op = payloads.get("operating_points", {})

    # unit-aware hospitalization-time axis: the landmark grid is window-ordinal,
    # but the physical unit (day/hour) comes from META, stamped on every tag.
    unit = (leadtime.get("metadata", {}).get("window_unit")
            or next((it.get("window_unit")
                     for it in pillars["discrimination"] if it.get("window_unit")),
                    None))
    xlabel = f"hospitalization {unit}" if unit else "hospitalization window"

    def _at_risk_row(ax, items, n_key="n"):
        """KM-style 'No. at risk' row beneath the axis (≤8 sampled ticks).

        A true Kaplan-Meier count here: n at hospitalization time t = stays
        still producing predictions at t (at risk, event-free). Suppressed
        landmarks print their "<10" stub, so the risk table stays complete
        even where the curve has gaps.
        """
        from matplotlib.transforms import blended_transform_factory
        pts = [(it["hospitalization_time"], it.get(n_key))
               for it in items if it.get("hospitalization_time") is not None]
        if not pts:
            return
        step = max(1, round(len(pts) / 7))
        ticks = pts[::step]
        if ticks[-1][0] != pts[-1][0]:
            # include the horizon endpoint, replacing a too-close neighbour so
            # adjacent at-risk labels never overlap.
            if pts[-1][0] - ticks[-1][0] < step:
                ticks[-1] = pts[-1]
            else:
                ticks.append(pts[-1])
        ax.set_xticks([x for x, _ in ticks])
        ax.set_xlim(pts[0][0] - 0.5, pts[-1][0] + 0.5)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for x, n in ticks:
            ax.text(x, -0.24, "—" if n is None else str(n), transform=trans,
                    ha="center", va="top", fontsize=7, color="0.35",
                    clip_on=False)
        ax.text(-0.02, -0.24, "No. at risk", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="0.35",
                style="italic", clip_on=False)

    def _xyci(items, val_at, ci_at):
        """(xs, ys, lo, hi) over non-suppressed landmarks with a defined value."""
        xs, ys, lo, hi = [], [], [], []
        for it in items:
            if it.get("suppressed"):
                continue
            v, c = val_at(it), ci_at(it)
            if v is None:
                continue
            xs.append(it.get("hospitalization_time"))
            ys.append(v)
            if c and c[0] is not None and c[1] is not None:
                lo.append(c[0])
                hi.append(c[1])
            else:
                lo.append(v)
                hi.append(v)
        return xs, ys, lo, hi

    def _lead_png(items, val_at, ci_at, name, title, ylabel, ylim=None,
                  hline=None, intervention=None):
        xs, ys, lo, hi = _xyci(items, val_at, ci_at)
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(6, 4.4))
        ax.fill_between(xs, lo, hi, alpha=0.2, color="#5b86a6", label="95% CI")
        ax.plot(xs, ys, "-o", color="#5b86a6", ms=4)
        if hline is not None:
            ax.axhline(hline, color="k", ls="--", lw=0.8)
        if intervention and intervention.get("mean_hospitalization_time") is not None:
            m = intervention["mean_hospitalization_time"]
            sd = intervention.get("sd")
            ax.axvline(m, color="red", lw=1.4,
                       label=f"mean intervention ({unit or 'window'} {m:g})")
            if sd:
                ax.axvspan(m - sd, m + sd, color="red", alpha=0.10,
                           label="±1 SD")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if ylim:
            ax.set_ylim(*ylim)
        _at_risk_row(ax, items)
        ax.legend(loc="best", fontsize=8)
        _save(fig, name)

    _m = lambda key: (lambda it: (it.get("metrics") or {}).get(key))

    _lead_png(pillars["discrimination"], _m("auroc"), _m("auroc_ci"),
              "landmark_auc_by_hospital_time.png",
              f"Time-varying AUC by hospitalization {unit or 'window'}",
              "AUROC", ylim=(0.4, 1.0), hline=0.5)
    _lead_png(leadtime.get("curve", []),
              lambda it: it.get("sensitivity"), lambda it: it.get("sensitivity_ci"),
              "sensitivity_by_hospital_time.png",
              f"Detection sensitivity by hospitalization {unit or 'window'}",
              "sensitivity", ylim=(0.0, 1.0),
              intervention=op.get("intervention_offsets"))
    _lead_png(pillars["calibration"], _m("slope"), _m("slope_ci"),
              "landmark_calibration.png",
              f"Calibration slope by hospitalization {unit or 'window'}",
              "calibration slope", hline=1.0)

    _render_landmark_reliability(pillars["calibration"], _save, plt, unit)

    # predicted risk trajectory by outcome: mean predicted prob for positive vs
    # negative stays at each time-to-event (two lines) + shared at-risk row.
    traj = leadtime.get("trajectory", [])
    pts = [it for it in traj if not it.get("suppressed")]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        for key, color, lbl in (("mean_prob_positive", "#1f77b4", "positive (label=1)"),
                                ("mean_prob_negative", "#ff7f0e", "negative (label=0)")):
            xs = [it["hospitalization_time"] for it in pts if it.get(key) is not None]
            ys = [it[key] for it in pts if it.get(key) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, "-o", color=color, ms=4, label=lbl)
        ax.set(xlabel=xlabel, ylabel="Mean predicted probability",
               title="Predicted risk trajectory by outcome", ylim=(0.0, 1.0))
        _at_risk_row(ax, traj)
        ax.legend(loc="best", fontsize=8)
        _save(fig, "landmark_risk_trajectory.png")

    _render_operating_points(op, _save, plt)
    _render_operating_points_sub(op, _save, plt)

    return written


def _render_landmark_reliability(cal_items: list[dict], _save, plt,
                                 unit: str | None) -> None:
    """Reliability curves at 3 hospitalization times (landmark_reliability.png).

    Overlays the earliest, mid-horizon, and latest non-suppressed landmark's
    reliability data — bin scatter + smoothed fit + ideal diagonal — so
    continuous-mode calibration is visible as actual reliability curves, not
    only the slope-drift summary. Render-only: the bins/fits come straight from
    the per-landmark calibration payloads.
    """
    scored = [it for it in cal_items
              if not it.get("suppressed") and not it.get("undefined")
              and (it.get("curves") or {}).get("reliability")]
    if not scored:
        return
    picks = {0, len(scored) // 2, len(scored) - 1}   # earliest / mid / latest
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for color, idx in zip(colors, sorted(picks)):
        it = scored[idx]
        m = it.get("metrics") or {}
        label = (f"{unit or 'window'} {it['hospitalization_time']} "
                 f"(slope {m.get('slope')}, ECE {m.get('ece')})")
        bins = it["curves"]["reliability"]
        pairs = [(p, t) for p, t in zip(bins["bin_pred_prob"],
                                        bins["bin_true_prob"])
                 if p is not None and t is not None]
        if pairs:
            ax.scatter(*zip(*pairs), color=color, s=18, label=label)
        smoothed = (it.get("curves") or {}).get("smoothed")
        if smoothed:
            ax.plot(smoothed["pred_grid"], smoothed["obs_smoothed"],
                    color=color, lw=1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
           title=f"Reliability by hospitalization {unit or 'window'} "
                 f"(selected slices)",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=7.5)
    _save(fig, "landmark_reliability.png")


def _render_operating_points(op: dict, _save, plt) -> None:
    """Per-stay ALERT × LABEL 2×2 per threshold (operating_points_confusion.png).

    One panel per threshold (clinical ± 0.05); the TP (fired & label-positive)
    cell is emphasized, with sensitivity / specificity / FPR / PPV / NNE beneath.
    Task-agnostic — no intervention config needed.

    A task near the edge of (0, 1) gets FEWER than three panels: task1 sits at 0.95,
    so its +0.05 arm is dropped rather than thresholded at an unusable 1.0. Say so on
    the figure — otherwise a reader comparing task1's two panels against task2's three
    is left to guess whether a panel failed to render.
    """
    by_thr = op.get("by_threshold") or []
    if not by_thr:
        return
    n = len(by_thr)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, e in zip(axes, by_thr):
        c = e.get("confusion") or {}
        # each quadrant is annotated with its confusion type AND count.
        cells = [[f"TP\n{c.get('tp')}", f"FN\n{c.get('fn')}"],
                 [f"FP\n{c.get('fp')}", f"TN\n{c.get('tn')}"]]
        tbl = ax.table(cellText=cells,
                       rowLabels=["label == 1", "label == 0"],
                       colLabels=["alert == 1\n(fired)", "alert == 0\n(silent)"],
                       cellLoc="center", loc="center",
                       bbox=[0.24, 0.30, 0.72, 0.54])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        for (rr, cc), col in {(1, 0): "#cfe3d4", (1, 1): "#f2d9d6",
                              (2, 0): "#f2d9d6", (2, 1): "#cfe3d4"}.items():
            tbl[rr, cc].set_facecolor(col)   # green = correct (TP/TN), red = error (FP/FN)
        for cell in tbl.get_celld().values():   # taller rows so the 2-line cells fit
            cell.set_height(0.20)
        ax.axis("off")
        stat = (f"sens {c.get('sensitivity')}  spec {c.get('specificity')}\n"
                f"FPR {c.get('fpr')}  PPV {c.get('ppv')}  NNE {c.get('nne')}")
        ax.text(0.5, 0.05, stat, transform=ax.transAxes, ha="center",
                va="center", fontsize=8, family="monospace")
        ax.set_title(f"t = {e.get('threshold')}   (n = {c.get('n')})", fontsize=10)
    fig.suptitle("Per-stay operating points  (alert == 1  ×  label == 1)",
                 fontsize=11)
    dropped = (op.get("metadata") or {}).get("dropped_offsets") or []
    if dropped:
        note = "; ".join(
            f"{d.get('offset'):+g} dropped (threshold {d.get('threshold')}: "
            f"{d.get('reason')})" for d in dropped)
        fig.text(0.5, 0.02, note, ha="center", fontsize=8, color="#8a6d3b")
    _save(fig, "operating_points_confusion.png")


def _render_operating_points_sub(op: dict, _save, plt) -> None:
    """Per-threshold intervention sub-analyses (operating_points_subanalyses.png).

    Rows = the thresholds; cols = label-positive alarm lead time / label-negative
    treated alarm lead time / label-negative untreated misfires. Both lead-time
    panels share ONE axis ("hours before treatment", positive = alarm earlier).
    Rendered only when the task declares an intervention/treatment column (each
    entry has `sub_analyses`). All labels use the config `display` — no task
    specifics.
    """
    by_thr = [e for e in (op.get("by_threshold") or []) if e.get("sub_analyses")]
    if not by_thr:
        return
    iv = (op.get("metadata") or {}).get("intervention") or {}
    display = iv.get("display") or iv.get("column") or "treatment"
    nrows = len(by_thr)
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 3.4 * nrows), squeeze=False)
    for r, e in enumerate(by_thr):
        t = e.get("threshold")
        sub = e["sub_analyses"]
        _lead_hist(axes[r][0],
                   (sub.get("label_positive") or {}).get("hours_before_treatment"),
                   f"label == 1 & alert == 1   (t = {t})", display,
                   (sub.get("label_positive") or {}))
        _lead_hist(axes[r][1],
                   (sub.get("label_negative_treated") or {}).get("hours_before_treatment"),
                   f"label == 0 & {display} == 1 & alert == 1   (t = {t})", display,
                   (sub.get("label_negative_treated") or {}))
        _misfire_split(axes[r][2], sub.get("label_negative_untreated") or {},
                       f"label == 0 & {display} == 0   (t = {t})")
    fig.suptitle(f"Operating-point sub-analyses  (anchor: {display} start time)",
                 fontsize=11)
    _save(fig, "operating_points_subanalyses.png")


def _lead_hist(ax, summary: dict | None, title: str, display: str,
               arm: dict | None = None) -> None:
    """Alarm lead-time histogram: hours BEFORE treatment (positive = earlier).

    Mass sits on the left (near t=0), so the stats are annotated at the upper
    RIGHT and the treatment line is labelled inline — no legend box to collide
    with the bars.
    """
    bins = (summary or {}).get("histogram") or []
    if not summary or summary.get("suppressed") or not bins:
        ax.text(0.5, 0.5, "suppressed / no data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set(title=title, xticks=[], yticks=[])
        return
    ax.bar([b["lo"] for b in bins], [b["count"] for b in bins],
           width=[b["hi"] - b["lo"] for b in bins], align="edge",
           color="#5b86a6", edgecolor="white")
    ax.axvline(0.0, color="red", lw=1.4)
    # inline label on the treatment line (rotated), instead of a legend.
    ax.annotate(f"{display} start (t = 0)", xy=(0, 1), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points", rotation=90,
                va="top", ha="left", color="red", fontsize=7.5)
    arm = arm or {}
    pct = arm.get("pct_fired")
    fired = f"fired {round(pct * 100, 1)}%" if pct is not None else ""
    stat = (f"median {summary.get('median')} h before {display}\n"
            f"IQR [{summary.get('p25')}, {summary.get('p75')}] h   n = {summary.get('n')}"
            + (f"\n{fired} of arm" if fired else ""))
    ax.text(0.97, 0.97, stat, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set(ylabel="stays that fired",
           xlabel=f"hours before {display} (positive = alarm earlier)")
    ax.set_title(title, fontsize=9)


def _misfire_split(ax, arm: dict, title: str) -> None:
    """Untreated-negative outcome split: misfired (false alarm) vs silent (correct).

    Two side-by-side fraction bars summing to 1 (not a lone bar / stack); 95% CI
    error bar on the misfired bar.
    """
    rate = arm.get("misfire_rate")
    if rate is None:
        ax.text(0.5, 0.5, "suppressed / no data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set(title=title, xticks=[], yticks=[])
        return
    ci = arm.get("misfire_rate_ci")
    yerr = [[rate - ci[0]], [ci[1] - rate]] if ci and ci[0] is not None else None
    labels = ["alert == 1\n(misfired)", "alert == 0\n(silent)"]
    heights = [rate, 1 - rate]
    bars = ax.bar(labels, heights, color=["#c25b5b", "#8bbf9f"], width=0.6)
    ax.errorbar([labels[0]], [rate], yerr=yerr, fmt="none", color="k", capsize=4)
    # annotate each bar with its fraction (and count where known).
    n_mis = arm.get("n_misfired")
    notes = [f"{rate:.4f}" + (f"\n(n={n_mis})" if isinstance(n_mis, int) else ""),
             f"{1 - rate:.4f}"]
    for b, note in zip(bars, notes):
        ax.text(b.get_x() + b.get_width() / 2, min(b.get_height() + 0.03, 0.95),
                note, ha="center", va="bottom", fontsize=7.5, family="monospace")
    ax.set(ylim=(0, 1.08), ylabel="fraction of untreated negatives")
    ax.set_title(f"{title}   n = {arm.get('n')}", fontsize=9)
