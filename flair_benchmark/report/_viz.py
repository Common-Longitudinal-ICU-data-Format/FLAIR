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
                    label="linear calibration fit")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
        ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
               title=f"Calibration  "
                     f"{_ci_label('slope', cal_metrics.get('slope'), cal_metrics.get('slope_ci'))}  "
                     f"O:E={cal_metrics.get('o_e_ratio')}")
        # Brier + intercept with their 95% CIs as an in-axes annotation.
        ax.text(0.98, 0.02,
                _ci_label("Brier", cal_metrics.get("brier"), cal_metrics.get("brier_ci"))
                + "\n"
                + _ci_label("intercept", cal_metrics.get("intercept"),
                            cal_metrics.get("intercept_ci")),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        slope = cal_metrics.get("slope")
        if slope is not None and abs(slope - 1) > 0.3:
            ax.text(0.5, 0.02, "probabilities uncalibrated (ranking only) — "
                    "scale_pos_weight inflates risk", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7, color="firebrick")
        ax.legend(loc="upper left")
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
        # zoom to the rare-event decision region so the curve isn't a flat line
        fr = dca_curves.get("focus_range") or [0.0, 1.0]
        ax.set(xlabel="Threshold probability", ylabel="Net benefit",
               title=f"Decision curve analysis (zoom 0–{fr[1]:g})",
               xlim=(fr[0], fr[1]),
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
