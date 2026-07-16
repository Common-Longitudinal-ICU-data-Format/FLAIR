"""Sanity-check PNGs rendered FROM the report JSON dicts.

Deliberately consumes the already-built dicts (never the parquet) so a
rendered PNG proves the JSON alone is sufficient for the FLAIR website's
interactive plots. Requires the optional extra:

    pip install flair-benchmark[viz]

Two entry points, one per report mode:
  render_sanity_pngs     : episodic bundle (roc_pr / calibration / dca / fairness).
  render_continuous_pngs : continuous bundle — peak section (peak_roc_pr,
                           peak_fairness_auroc), per-landmark AUC/calibration
                           curves on a unit-aware axis (days for tasks 1/2, hours
                           for task 4), and the operating-points section (99-point
                           alert-burden twin axis, data-driven locator strip,
                           named-threshold confusion + optional treatment sub-rates).
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


# The universal sweep publishes 99 thresholds; the figures cannot mark all of
# them (dots retrace the ROC, 99 confusion panels / table columns are unreadable).
# So the picked SET of data-driven locators is what gets drawn — the rest live only
# in operating_points.json. Keep this small (design doc §6, Q6).
_NAMED_LOCATORS = ("youden_j", "f_beta_max", "closest_01")
_LOC_LABEL = {"youden_j": "Youden", "f_beta_max": "F1", "closest_01": "closest01",
              "concordance_cz": "CZ", "index_of_union": "IU",
              "sensitivity_target": "sens*", "specificity_target": "spec*",
              "cost_based": "cost", "ppv_target": "PPV*",
              "alert_rate_target": "rate*", "max_accuracy": "maxAcc",
              "clinical": "clinical"}
_VLINE_COLORS = {"youden_j": "#6a51a3", "f_beta_max": "#2ca25f",
                 "closest_01": "#d95f0e"}
_XTICKS = [i / 10 for i in range(11)]


def _row_nearest(rows: list[dict], t: float) -> dict:
    """The by_threshold row whose threshold is closest to t."""
    return min(rows, key=lambda e: abs(e["threshold"] - t))


def _locator_vlines(locators: list[dict]) -> list[tuple]:
    """(label, threshold, color) for the small named locator set drawn as vlines."""
    out = []
    for loc in locators:
        if loc.get("rule") in _NAMED_LOCATORS and loc.get("threshold") is not None:
            out.append((_LOC_LABEL.get(loc["rule"], loc["rule"]), loc["threshold"],
                        _VLINE_COLORS.get(loc["rule"], "#888888")))
    return out


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
    """dca.png — the net-benefit curve, ON ITS OWN AXIS (verdict text lives in dca.json).

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
    """Mark the clinical operating point + the named data-driven locators on the ROC.

    The ROC line already traces the full 99-point sweep, so plotting all 99 as dots
    just retraces it. Only the NAMED points are marked — where each rule STANDS on
    the curve — with the clinical point carrying its sens / FPR 95% CI error bars.
    """
    locs = [loc for loc in ops.get("data_driven_thresholds", [])
            if loc.get("rule") in _NAMED_LOCATORS and loc.get("threshold") is not None]
    plotted = False
    for loc in locs:
        ma = loc.get("metrics_at") or {}
        x, y = ma.get("fpr"), ma.get("sensitivity")
        if x is None or y is None:
            continue
        ax.plot([x], [y], "o", color="#5b86a6", ms=5, alpha=0.85,
                label="data-driven thresholds" if not plotted else None)
        plotted = True
        ax.annotate(f"{_LOC_LABEL.get(loc['rule'], loc['rule'])} {loc['threshold']:g}",
                    (x, y), textcoords="offset points", xytext=(4, -7),
                    fontsize=6.5, color="#33566b")
    clin = next((e for e in ops.get("by_threshold", []) if e.get("is_clinical")), None)
    if clin:
        m = clin.get("metrics") or {}
        x, y = m.get("fpr"), m.get("sensitivity")
        if x is not None and y is not None:
            sci, spci = m.get("sensitivity_ci"), m.get("fpr_ci")
            yerr = [[y - sci[0]], [sci[1] - y]] if sci and sci[0] is not None else None
            xerr = [[x - spci[0]], [spci[1] - x]] if spci and spci[0] is not None else None
            ax.errorbar([x], [y], yerr=yerr, xerr=xerr, fmt="o", color="red",
                        capsize=3, ms=5, label=f"clinical @ {clin.get('threshold'):g}")
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
_DECISION_TABLE = [("net_benefit", "net ben"), ("net_benefit_treat_all", "vs all")]


def _named_columns(rows: list[dict], clinical, locators: list[dict]) -> list[tuple]:
    """(column label, by_threshold row) for clinical + each located cutpoint.

    A 99-column value table is unreadable, so the table shows only the named
    thresholds — the shipped policy plus where each data-driven rule lands (§6.1).
    """
    cols: list[tuple] = []
    seen: set[float] = set()
    if clinical is not None:
        cols.append((f"clin\n{clinical:g}", _row_nearest(rows, clinical)))
        seen.add(round(clinical, 2))
    for loc in locators:
        if loc.get("rule") == "clinical" or loc.get("threshold") is None:
            continue
        t = round(float(loc["threshold"]), 2)
        if t in seen:
            continue
        seen.add(t)
        cols.append((f"{_LOC_LABEL.get(loc['rule'], loc['rule'])}\n{t:g}",
                     _row_nearest(rows, t)))
    return cols


def _render_scorecard(ops: dict | None, _save, plt) -> None:
    """operating_points.png — every KPI as a trend over the 99-point sweep + a table.

    One trend panel per KPI (line + 95% CI band) with the clinical threshold and the
    named data-driven locators marked; the exact numbers at the named thresholds are
    tabulated underneath. This is the picture the JSON is for: the whole shape of the
    trade-off, and where each rule would stand on it.
    """
    rows = [e for e in (ops or {}).get("by_threshold", []) if e.get("metrics")]
    if not rows:
        return
    thr = [e["threshold"] for e in rows]
    clinical = next((e["threshold"] for e in rows if e.get("is_clinical")), None)
    locators = (ops or {}).get("data_driven_thresholds", [])
    vlines = _locator_vlines(locators)
    named = _named_columns(rows, clinical, locators)

    n_panels = len(_SCORECARD_PANELS) + 1        # + the decision (net-benefit) panel
    table_rows = _SCORECARD_TABLE + _DECISION_TABLE
    ncol = 4
    nrow = -(-n_panels // ncol)
    fig = plt.figure(figsize=(3.2 * ncol, 2.5 * nrow + 0.32 * len(table_rows) + 1))
    # the value table is a real subplot spanning the bottom row, not a floating
    # add_axes — _make_saver runs tight_layout(), which would shove a free axes.
    gs = fig.add_gridspec(nrow + 1, ncol, hspace=0.5, wspace=0.3,
                          left=0.06, right=0.98, top=0.90, bottom=0.04,
                          height_ratios=[*([2.4] * nrow), 0.32 * len(table_rows)])
    for i, (key, title, ci_key) in enumerate(_SCORECARD_PANELS):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        _panel(ax, thr, [e["metrics"] for e in rows], key, ci_key, clinical, vlines)
        ax.set_title(title, fontsize=9)

    # The decision panel answers "should we act here at all", so it carries the
    # treat-all line: without it a rising net-benefit curve looks like good news even
    # when doing it to everyone would score higher.
    i = len(_SCORECARD_PANELS)
    ax = fig.add_subplot(gs[i // ncol, i % ncol])
    dec = [e.get("decision", {}) for e in rows]
    _panel(ax, thr, dec, "net_benefit", "net_benefit_ci", clinical, vlines)
    ref = [(t, d.get("net_benefit_treat_all")) for t, d in zip(thr, dec)]
    ref = [(t, v) for t, v in ref if v is not None]
    if ref:
        ax.plot(*zip(*ref), "--", color="#c25b5b", lw=1.0, label="act on all")
        ax.legend(fontsize=6.5, loc="best")
    ax.set_title("Net benefit (decision basis)", fontsize=9)

    tax = fig.add_subplot(gs[nrow, :])
    tax.axis("off")
    cells = ([[_fmt((r["metrics"] or {}).get(k)) for _lbl, r in named]
              for k, _lbl in _SCORECARD_TABLE]
             + [[_fmt((r.get("decision") or {}).get(k)) for _lbl, r in named]
                for k, _lbl in _DECISION_TABLE])
    table = tax.table(
        cellText=cells,
        rowLabels=[lbl for _k, lbl in table_rows],
        colLabels=[lbl for lbl, _r in named],
        cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for c, (_lbl, r) in enumerate(named):
        if r.get("is_clinical"):
            for rr in range(len(table_rows) + 1):
                table[rr, c].set_facecolor("#d6e6f2")   # the shipped policy

    md = ops.get("metadata") or {}
    note = (f"clinical {clinical:g}" if clinical is not None
            else "no clinical threshold on the grid")
    fig.suptitle(f"Operating-points scorecard — trend over 99 thresholds ({note}, "
                 f"direction: {md.get('clinical_direction', 'above')}); "
                 f"table columns = clinical + data-driven locators", fontsize=10)
    _save(fig, "operating_points.png", tight=False)


def _panel(ax, thr: list, blocks: list[dict], key: str, ci_key: str | None,
           clinical, vlines: list[tuple] = ()) -> None:
    """One KPI as a trend line over the sweep: line + CI band + policy/locator marks."""
    pairs = [(t, b.get(key)) for t, b in zip(thr, blocks)]
    pairs = [(t, v) for t, v in pairs if v is not None]
    if pairs:
        ax.plot(*zip(*pairs), "-", color="#5b86a6", lw=1.3)
        if ci_key:
            band = [(t, b.get(ci_key)) for t, b in zip(thr, blocks)]
            band = [(t, c) for t, c in band
                    if c and c[0] is not None and c[1] is not None]
            if band:
                bt, bc = zip(*band)
                ax.fill_between(bt, [c[0] for c in bc], [c[1] for c in bc],
                                color="#5b86a6", alpha=0.18, lw=0)
    for _lbl, t, color in vlines:
        ax.axvline(t, color=color, lw=0.8, ls=":", alpha=0.85)
    if clinical is not None:
        ax.axvline(clinical, color="red", lw=1.0, ls="--")
    ax.tick_params(labelsize=7)
    ax.set_xticks(_XTICKS)                # sparse: 11 ticks, not all 99 thresholds
    ax.set_xlim(0, 1)


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
      landmark_calibration          : calibration slope +/- 95% CI (drift)
      landmark_reliability          : reliability curves at 3 hospitalization
                                      times (earliest / mid / latest scored)
    Operating points (all on the peak surface, 99-point sweep):
      alert_burden                  : encounter-level twin axis — detection
                                      (sens/PPV) vs fraction of admissions alerting,
                                      alarm-fatigue budget shaded
      data_driven_thresholds        : where each selection rule lands vs the policy
      operating_points_confusion    : ALERT × LABEL 2×2 at named thresholds
                                      (clinical / Youden / F1), full 4-cell matrix
      operating_points_subanalyses  : treatment fire-rate curves + named-threshold
                                      median lead-time (only for a task with an
                                      intervention/treatment column)
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
    _lead_png(pillars["calibration"], _m("slope"), _m("slope_ci"),
              "landmark_calibration.png",
              f"Calibration slope by hospitalization {unit or 'window'}",
              "calibration slope", hline=1.0)

    _render_landmark_reliability(pillars["calibration"], _save, plt, unit)

    # (the day/hour-level sensitivity curve and risk trajectory were removed — the
    # operating-point analysis is encounter-level only, design doc §9.)
    _render_alert_burden(op, _save, plt)
    _render_data_driven(op, _save, plt)
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


def _confusion_named_rows(op: dict) -> list[tuple]:
    """(label, by_threshold row) for clinical + Youden + F1 — the confusion panels."""
    rows = op.get("by_threshold") or []
    locs = op.get("data_driven_thresholds", [])
    picks: list[tuple] = []
    clin = next((e for e in rows if e.get("is_clinical")), None)
    seen: set[float] = set()
    if clin:
        picks.append(("clinical", clin))
        seen.add(round(clin["threshold"], 2))
    for rule in ("youden_j", "f_beta_max"):
        loc = next((x for x in locs
                    if x.get("rule") == rule and x.get("threshold") is not None), None)
        if loc:
            t = round(float(loc["threshold"]), 2)
            if t in seen:
                continue
            seen.add(t)
            picks.append((_LOC_LABEL.get(rule, rule), _row_nearest(rows, t)))
    return picks


def _render_operating_points(op: dict, _save, plt) -> None:
    """ALERT × LABEL 2×2 at the named thresholds (operating_points_confusion.png).

    99 panels would be absurd, so only the named set is drawn — clinical, Youden, F1
    — each a full 2×2 (all four cells) with sensitivity / specificity / FPR / PPV /
    NNE beneath. Task-agnostic — no intervention config needed.
    """
    named = _confusion_named_rows(op)
    if not named:
        return
    n = len(named)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, (label, e) in zip(axes, named):
        c = e.get("confusion") or {}
        m = e.get("metrics") or {}
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
        stat = (f"sens {m.get('sensitivity')}  spec {m.get('specificity')}\n"
                f"FPR {m.get('fpr')}  PPV {m.get('ppv')}  NNE {m.get('nne')}")
        ax.text(0.5, 0.05, stat, transform=ax.transAxes, ha="center",
                va="center", fontsize=8, family="monospace")
        ax.set_title(f"{label}: t = {e.get('threshold')}   (n = {c.get('n')})",
                     fontsize=10)
    fig.suptitle("Operating points at named thresholds  (full alert × label 2×2)",
                 fontsize=11)
    _save(fig, "operating_points_confusion.png")


def _render_operating_points_sub(op: dict, _save, plt) -> None:
    """Treatment sub-analysis (operating_points_subanalyses.png).

    Fire-rate curves across the sweep — positive-arm fired, treated-negative fired,
    untreated misfire — which decompose the false-alarm axis into "fired on a stay
    that got treated anyway" vs "pure misfire on a never-treated stay". A single
    median lead-time is annotated at the named thresholds (no distribution, no
    histogram — design doc §4.4). Rendered only for a task with an intervention.
    """
    rows = [e for e in (op.get("by_threshold") or []) if e.get("sub_rates")]
    if not rows:
        return
    iv = (op.get("metadata") or {}).get("intervention") or {}
    display = iv.get("display") or iv.get("column") or "treatment"
    clinical = next((e["threshold"] for e in rows if e.get("is_clinical")), None)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # each line = one patient subgroup (arm); y = that arm's SHARE of every stay that
    # alarms at t (one shared denominator = all firing stays). The three arms partition
    # every stay — event-positive, or event-negative split by whether the treatment was
    # ever given — so the three shares sum to 1 (a "what is my alarm made of" read).
    series = [("positive_share", "#1f77b4", "label=1 (event): share of alarms"),
              ("treated_negative_share", "#2ca02c",
               f"label=0 & treated ({display}): share of alarms"),
              ("untreated_misfire_share", "#c25b5b",
               "label=0 & untreated: share of alarms (misfire)")]
    for key, color, lbl in series:
        pts = [(e["threshold"], (e["sub_rates"] or {}).get(key)) for e in rows]
        pts = [(t, v) for t, v in pts if v is not None]
        if pts:
            ax.plot(*zip(*pts), "-", color=color, lw=1.5, label=lbl)
    if clinical is not None:
        ax.axvline(clinical, color="red", ls="--", lw=1.0,
                   label=f"clinical threshold {clinical:g}")

    notes = []
    for x in op.get("intervention_timing") or []:
        mp = x.get("median_hours_before_treatment_positive")
        if mp is not None:
            notes.append(f"t={x['threshold']:g}: {mp:g} h"
                         + ("  (clinical)" if x.get("is_clinical") else ""))
    if notes:
        # lower-left: all three arms fire near 1.0 on the left, so low-y is the empty
        # quadrant — keeps this box clear of the upper-right legend.
        ax.text(0.02, 0.02, f"median lead time before {display}\n" + "\n".join(notes[:6]),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
                family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set(xlabel="threshold  (peak risk ≥ threshold ⇒ the stay alerts)",
           ylabel="share of alarms  (of all firing stays; 3 arms sum to 1)",
           xlim=(0, 1), ylim=(0, 1.02),
           title=f"Treatment sub-analysis — alarm composition by arm (anchor: {display})")
    ax.set_xticks(_XTICKS)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.text(0.5, -0.04,
             "Each line is a patient subgroup (arm); y = share of ALL firing stays that "
             "fall in this arm (the three arms sum to 1).\nGood model: blue high (most "
             "alarms are real events) while red stays low (few pure misfires).",
             ha="center", va="top", fontsize=7, color="0.35")
    _save(fig, "operating_points_subanalyses.png")


def _render_alert_burden(op: dict, _save, plt) -> None:
    """alert_burden.png — the two coupled axes, encounter-level (peak surface, §6.3).

    Left: detection (sensitivity, PPV) on the peak surface. Right: the encounter
    alert-rate (fraction of admissions ever alerting) with the alarm-fatigue budget
    shaded. NNE at the clinical threshold is annotated (its scale dwarfs a 0–1 axis,
    so it is a number, not a co-plotted curve). No patient-day denominator.
    """
    rows = [e for e in (op.get("by_threshold") or []) if e.get("alert_burden")]
    if not rows:
        return
    clinical = next((e["threshold"] for e in rows if e.get("is_clinical")), None)

    def series(key, src):
        pts = [(e["threshold"], (e.get(src) or {}).get(key)) for e in rows]
        return [(t, v) for t, v in pts if v is not None]

    # collect legend handles EXPLICITLY — ax.get_lines() would also sweep up the
    # unlabeled axvline/axhline reference lines as matplotlib "_child" entries.
    handles = []
    fig, axl = plt.subplots(figsize=(7.6, 4.6))
    for key, color, lbl in (("sensitivity", "#1f77b4", "sensitivity"),
                            ("ppv", "#2ca02c", "PPV")):
        p = series(key, "metrics")
        if p:
            handles += axl.plot(*zip(*p), "-", color=color, lw=1.5, label=lbl)
    axl.set(xlabel="threshold", ylabel="detection (sensitivity / PPV)",
            xlim=(0, 1), ylim=(0, 1.02))
    axl.set_xticks(_XTICKS)

    axr = axl.twinx()
    fa = series("frac_stays_alerting", "alert_burden")
    if fa:
        handles += axr.plot(*zip(*fa), "-", color="#c25b5b", lw=1.5,
                            label="frac stays alerting")
    axr.set(ylabel="fraction of admissions alerting", ylim=(0, 1.02))

    targets = ((op.get("metadata") or {}).get("selection_params") or {}).get(
        "alert_rate_targets") or []
    if targets:
        budget = max(targets)
        handles.append(axr.axhline(budget, color="#c25b5b", ls="--", lw=0.8,
                                   label=f"alert budget {budget:g}"))
        axr.axhspan(budget, 1.02, color="#c25b5b", alpha=0.06)   # alarm-fatigue zone
    clin_row = next((e for e in rows if e.get("is_clinical")), None)
    if clin_row:
        nne = (clin_row.get("alert_burden") or {}).get("nne")
        if nne is not None:
            axr.text(0.02, 0.02, f"NNE @ clinical = {nne:g} alerts / TP",
                     transform=axr.transAxes, fontsize=7.5, family="monospace",
                     bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    if clinical is not None:
        handles.append(axl.axvline(clinical, color="red", ls="--", lw=1.0,
                                   label=f"clinical {clinical:g}"))
    axl.legend(handles, [h.get_label() for h in handles],
               loc="upper center", fontsize=7.5, ncol=3)
    axl.set_title("Alert burden — encounter-level (peak surface)", fontsize=10)
    _save(fig, "alert_burden.png")


def _render_data_driven(op: dict, _save, plt) -> None:
    """data_driven_thresholds.png — where every rule lands vs the shipped policy (§6.5)."""
    locs = [loc for loc in op.get("data_driven_thresholds", [])
            if loc.get("threshold") is not None and loc.get("rule") != "clinical"]
    if not locs:
        return
    clinical = (op.get("metadata") or {}).get("clinical_threshold")
    locs = sorted(locs, key=lambda loc: loc["threshold"])
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.42 * len(locs))))
    ys = list(range(len(locs)))
    for y, loc in zip(ys, locs):
        t = loc["threshold"]
        ma = loc.get("metrics_at") or {}
        ax.plot([0, t], [y, y], "-", color="#cccccc", lw=1.0, zorder=1)
        tci = loc.get("threshold_ci")           # selection-aware cutpoint stability
        if tci and tci[0] is not None and tci[1] is not None:
            ax.plot(tci, [y, y], "-", color="#5b86a6", lw=3.0, alpha=0.35, zorder=2,
                    solid_capstyle="round")
        ax.plot([t], [y], "o", color="#5b86a6", ms=7, zorder=3)
        deg = "  (degenerate)" if loc.get("degenerate") else ""
        ax.annotate(f"{t:g}   sens {ma.get('sensitivity')}  ppv {ma.get('ppv')}{deg}",
                    (t, y), textcoords="offset points", xytext=(9, 0),
                    va="center", fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels([_LOC_LABEL.get(loc["rule"], loc["rule"]) for loc in locs],
                       fontsize=8)
    if clinical is not None:
        ax.axvline(clinical, color="red", ls="--", lw=1.2,
                   label=f"clinical {clinical:g}")
        ax.legend(loc="lower right", fontsize=8)
    ax.set(xlabel="threshold", xlim=(0, 1),
           title="Where each rule lands vs the shipped policy")
    _save(fig, "data_driven_thresholds.png")
