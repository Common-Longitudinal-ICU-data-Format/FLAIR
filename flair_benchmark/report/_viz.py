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
                           mean±SD intervention line, and the three-arm
                           alert-vs-intervention timing panels.
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

    def _save(fig, name: str) -> None:
        path = out_dir / name
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

    _render_roc_pr(reports.get("discrimination", {}), _save, plt)

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

    dca = reports.get("dca", {})
    dca_curves, dca_metrics = dca.get("curves", {}), dca.get("metrics", {})
    if dca_curves.get("thresholds"):
        thr = dca_curves["thresholds"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thr, [v if v is not None else float("nan")
                      for v in dca_curves["net_benefit_model"]], label="model")
        ax.plot(thr, dca_curves["net_benefit_all"], label="treat all")
        ax.axhline(0.0, color="k", lw=0.8, label="treat none")
        if dca_metrics.get("useful_range"):
            ax.axvspan(dca_metrics["useful_range"]["low"],
                       dca_metrics["useful_range"]["high"],
                       alpha=0.15, color="green", label="useful range")
        ct = dca_metrics.get("clinical_threshold")
        if ct is not None:
            ax.axvline(ct, color="red", lw=1.2, ls=":",
                       label=f"clinical threshold ({ct})")
            # net benefit at the clinical threshold with its 95% bootstrap CI.
            nb = dca_metrics.get("net_benefit_at_clinical_threshold", {})
            m_nb, m_ci = nb.get("model"), nb.get("model_ci")
            if m_nb is not None:
                yerr = ([[m_nb - m_ci[0]], [m_ci[1] - m_nb]]
                        if m_ci and m_ci[0] is not None else None)
                ax.errorbar([ct], [m_nb], yerr=yerr, fmt="o", color="red",
                            capsize=3, ms=5)
        # full threshold range (no zoom); y is still clamped so the treat-all line
        # diving to large negatives at high thresholds doesn't crush the detail.
        ax.set(xlabel="Threshold probability", ylabel="Net benefit",
               title="Decision curve analysis",
               xlim=(0.0, 1.0),
               ylim=(min(-0.01, -dca["prevalence"] * 0.5),
                     max(0.01, dca["prevalence"] * 1.2)))
        ax.legend()
        _save(fig, "dca.png")

    _render_fairness_auroc(reports.get("fairness", {}), _save, plt)

    return written


def _render_roc_pr(disc: dict, _save, plt, prefix: str = "",
                   title_note: str = "") -> None:
    """ROC + PR panel from a discrimination payload (episodic or peak section)."""
    curves, metrics = disc.get("curves", {}), disc.get("metrics", {})
    if "roc" not in curves:
        return
    roc, pr = curves["roc"], curves["pr"]
    op = metrics.get("operating_point", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(roc["fpr"], roc["tpr"])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
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
                                      the red mean±SD intervention (e.g.
                                      antibiotics) line when the task declares
                                      one
      landmark_calibration          : calibration slope +/- 95% CI (drift)
      landmark_reliability          : reliability curves at 3 hospitalization
                                      times (earliest / mid / latest scored)
      landmark_risk_trajectory      : mean predicted risk by eventual outcome
    Alert timing: alert_timing.png — the three alert-vs-intervention arms
    (only for tasks with an intervention config).
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
    avi = leadtime.get("alert_vs_intervention") or {}

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
              intervention=avi.get("intervention_offsets"))
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

    _render_alert_timing(avi, _save, plt)

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


def _render_alert_timing(avi: dict, _save, plt) -> None:
    """Three-arm alert-vs-intervention panels (alert_timing.png).

    A: label-positive stays — (t_first_alert - t_intervention) histogram, red
       line at 0 = intervention; left of the line = model fired first.
    B: label-negative stays that received the intervention — same layout.
    C: label-negative, never treated — did the model fire at all (false-alarm
       rate with its 95% CI).
    Skipped entirely when the task has no intervention config or every arm is
    suppressed.
    """
    iv = (avi.get("metadata") or {}).get("intervention")
    if not iv:
        return
    display = iv.get("display") or iv["column"]

    def _delta_panel(ax, arm: dict | None, title: str) -> bool:
        w = (arm or {}).get("with_intervention") or {}
        d = w.get("delta_hours") or {}
        bins = d.get("histogram") or []
        if w.get("suppressed") or not bins:
            ax.text(0.5, 0.5, "suppressed / no data", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            ax.set(title=title, xticks=[], yticks=[])
            return False
        ax.bar([b["lo"] for b in bins],
               [b["count"] for b in bins],
               width=[b["hi"] - b["lo"] for b in bins],
               align="edge", color="#5b86a6", edgecolor="white")
        ax.axvline(0.0, color="red", lw=1.4, label=f"{display} (t = 0)")
        note = "\n".join([
            f"fired before {display}: "
            f"{_pct(w.get('fired_before_intervention_rate'))} (n={w.get('n')})",
            f"median Δ = {d.get('median')} h",
        ])
        ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.set(xlabel=f"t_first_alert − t_{display} (hours; negative = fired first)",
               ylabel="stays", title=title)
        ax.legend(loc="upper right", fontsize=7.5)
        return True

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    drew_a = _delta_panel(axes[0], avi.get("label_positive"),
                          "Label-positive stays")
    drew_b = _delta_panel(axes[1], avi.get("label_negative_with_intervention"),
                          f"Label-negative, {display} given")

    arm_c = avi.get("label_negative_without_intervention") or {}
    ax = axes[2]
    rate = arm_c.get("false_alarm_rate")
    drew_c = rate is not None
    if drew_c:
        ci = arm_c.get("false_alarm_rate_ci")
        yerr = ([[rate - ci[0]], [ci[1] - rate]]
                if ci and ci[0] is not None else None)
        ax.bar(["fired"], [rate], color="#c25b5b", width=0.5)
        ax.errorbar(["fired"], [rate], yerr=yerr, fmt="none", color="k",
                    capsize=4)
        ax.set(ylabel="false-alarm rate", ylim=(0, 1),
               title=f"Label-negative, no {display} (n={arm_c.get('n')})")
    else:
        ax.text(0.5, 0.5, "suppressed / no data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set(title=f"Label-negative, no {display}", xticks=[], yticks=[])

    if drew_a or drew_b or drew_c:
        fig.suptitle(f"Alert vs {display} timing", fontsize=11)
        _save(fig, "alert_timing.png")
    else:
        plt.close(fig)


def _pct(rate) -> str:
    return "NA" if rate is None else f"{100 * rate:.0f}%"
