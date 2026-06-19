"""Sanity-check PNGs rendered FROM the report JSON dicts.

Deliberately consumes the already-built dicts (never the parquet) so a
rendered PNG proves the JSON alone is sufficient for the FLAIR website's
interactive plots. Requires the optional extra:

    pip install flair-benchmark[viz]
"""

from __future__ import annotations

from pathlib import Path


def render_sanity_pngs(reports: dict[str, dict], out_dir: Path) -> list[Path]:
    """Render one PNG per applicable section. Returns the written paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for --viz. "
            "Install it with: pip install 'flair-benchmark[viz]'"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    # clear stale PNGs so a section that no longer renders (e.g. by_year skipped on a
    # date-shifted site) does not leave an orphaned image from a previous run.
    for old in out_dir.glob("*.png"):
        old.unlink()
    written: list[Path] = []

    def _save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    def _ci_label(name: str, val, ci) -> str:
        """'AUROC=0.82 [0.79–0.85]' — value with its 95% CI when available."""
        if val is None:
            return f"{name}=NA"
        if ci and ci[0] is not None and ci[1] is not None:
            return f"{name}={val:g} [{ci[0]:g}–{ci[1]:g}]"
        return f"{name}={val:g}"

    disc = reports.get("discrimination", {})
    curves, metrics = disc.get("curves", {}), disc.get("metrics", {})
    if "roc" in curves:
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
                    title=f"ROC ({_ci_label('AUROC', metrics.get('auroc'), metrics.get('auroc_ci'))})")
        axes[1].plot(pr["recall"], pr["precision"])
        base = metrics.get("auprc_baseline")
        if base is not None:
            axes[1].axhline(base, color="k", ls="--", lw=0.8,
                            label=f"prevalence {base:.4f}")
            axes[1].legend(loc="upper right", fontsize=8)
        axes[1].set(xlabel="Recall", ylabel="Precision",
                    title=f"PR ({_ci_label('AUPRC', metrics.get('auprc'), metrics.get('auprc_ci'))}, "
                          f"lift×{metrics.get('auprc_lift')})")
        _save(fig, "roc_pr.png")

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

    fairness = reports.get("fairness", {})
    rows = [(f"{col}:{lvl}", m["auroc"])
            for col, levels in fairness.get("subgroups", {}).items()
            for lvl, m in levels.items() if m.get("auroc") is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(rows))))
        names, aucs = zip(*rows)
        ax.barh(names, aucs)
        ax.set(xlabel="AUROC", title="Subgroup AUROC", xlim=(0.4, 1.0))
        _save(fig, "fairness_auroc.png")

    return written


def render_landmark_pngs(pillars: dict[str, list[dict]], leadtime: dict,
                         out_dir: Path) -> list[Path]:
    """Local site-validation PNGs for the landmark report (never exported).

    Three curves against lead time (windows before each stay's last window):
      landmark_skill_over_leadtime : AUROC +/- 95% CI (skill as the event nears)
      leadtime_sensitivity         : detection sensitivity +/- 95% CI
      landmark_calibration         : calibration slope +/- 95% CI (drift)
    Suppressed (n<10) or undefined landmarks are dropped (gaps).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for --viz. "
            "Install it with: pip install 'flair-benchmark[viz]'"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    written: list[Path] = []

    def _save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    def _annotate_n(ax, xs, ys, ns):
        """Small grey 'n=…' above each plotted landmark point."""
        for x, y, n in zip(xs, ys, ns):
            if n is None:
                continue
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=6.5, color="0.45")

    def _xyci(items, val_at, ci_at):
        """(xs, ys, lo, hi, ns) over non-suppressed landmarks with a defined value."""
        xs, ys, lo, hi, ns = [], [], [], [], []
        for it in items:
            if it.get("suppressed"):
                continue
            v, c = val_at(it), ci_at(it)
            if v is None:
                continue
            xs.append(it.get("windows_before_end"))
            ys.append(v)
            ns.append(it.get("n"))
            if c and c[0] is not None and c[1] is not None:
                lo.append(c[0])
                hi.append(c[1])
            else:
                lo.append(v)
                hi.append(v)
        return xs, ys, lo, hi, ns

    def _lead_png(items, val_at, ci_at, name, title, ylabel, ylim=None,
                  hline=None):
        xs, ys, lo, hi, ns = _xyci(items, val_at, ci_at)
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.fill_between(xs, lo, hi, alpha=0.2, color="#5b86a6", label="95% CI")
        ax.plot(xs, ys, "-o", color="#5b86a6", ms=4)
        _annotate_n(ax, xs, ys, ns)
        if hline is not None:
            ax.axhline(hline, color="k", ls="--", lw=0.8)
        ax.invert_xaxis()  # landmark 0 (the endpoint) on the right; the event nears
        ax.set(xlabel="windows before stay end (0 = last window)",
               ylabel=ylabel, title=title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc="best", fontsize=8)
        _save(fig, name)

    _m = lambda key: (lambda it: (it.get("metrics") or {}).get(key))

    _lead_png(pillars.get("discrimination", []), _m("auroc"), _m("auroc_ci"),
              "landmark_skill_over_leadtime.png",
              "Discrimination over lead time", "AUROC", ylim=(0.4, 1.0), hline=0.5)
    _lead_png(leadtime.get("curve", []),
              lambda it: it.get("sensitivity"), lambda it: it.get("sensitivity_ci"),
              "leadtime_sensitivity.png",
              "Detection sensitivity over lead time", "sensitivity", ylim=(0.0, 1.0))
    _lead_png(pillars.get("calibration", []), _m("slope"), _m("slope_ci"),
              "landmark_calibration.png",
              "Calibration slope over lead time", "calibration slope", hline=1.0)

    # predicted risk trajectory by outcome: mean predicted prob for positive vs
    # negative stays at each landmark (two lines), with per-point n.
    traj = leadtime.get("trajectory", [])
    pts = [it for it in traj if not it.get("suppressed")]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, color, lbl in (("mean_prob_positive", "#1f77b4", "positive (label=1)"),
                                ("mean_prob_negative", "#ff7f0e", "negative (label=0)")):
            xs = [it["windows_before_end"] for it in pts if it.get(key) is not None]
            ys = [it[key] for it in pts if it.get(key) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, "-o", color=color, ms=4, label=lbl)
            ns = [it.get("n_positive" if "positive" in key else "n_negative")
                  for it in pts if it.get(key) is not None]
            _annotate_n(ax, xs, ys, ns)
        ax.invert_xaxis()
        ax.set(xlabel="windows before stay end (0 = last window)",
               ylabel="Mean predicted probability",
               title="Predicted risk trajectory by outcome", ylim=(0.0, 1.0))
        ax.legend(loc="best", fontsize=8)
        _save(fig, "landmark_risk_trajectory.png")

    return written
