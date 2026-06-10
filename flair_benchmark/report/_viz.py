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
    written: list[Path] = []

    def _save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    curves = reports.get("curves", {})
    if "roc" in curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(curves["roc"]["fpr"], curves["roc"]["tpr"])
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
        axes[1].plot(curves["pr"]["recall"], curves["pr"]["precision"])
        axes[1].set(xlabel="Recall", ylabel="Precision", title="PR")
        _save(fig, "roc_pr.png")

    cal = curves.get("calibration")
    ext = curves.get("calibration_extended") or {}
    if cal:
        fig, ax = plt.subplots(figsize=(5, 5))
        pairs = [(p, t) for p, t in zip(cal["bin_pred_prob"], cal["bin_true_prob"])
                 if p is not None and t is not None]
        if pairs:
            ax.scatter(*zip(*pairs), label="bins (n>=10)")
        smoothed = ext.get("smoothed")
        if smoothed:
            ax.plot(smoothed["pred_grid"], smoothed["obs_smoothed"],
                    label="logistic recalibration")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
        ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
               title=f"Calibration (ECE={cal['ece']}, slope={ext.get('slope')})")
        ax.legend()
        _save(fig, "calibration.png")

    dca = reports.get("dca", {})
    if dca.get("thresholds"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(dca["thresholds"], [v if v is not None else float("nan")
                                    for v in dca["net_benefit_model"]], label="model")
        ax.plot(dca["thresholds"], dca["net_benefit_all"], label="treat all")
        ax.axhline(0.0, color="k", lw=0.8, label="treat none")
        if dca.get("useful_range"):
            ax.axvspan(dca["useful_range"]["low"], dca["useful_range"]["high"],
                       alpha=0.15, color="green", label="useful range")
        if dca.get("clinical_threshold") is not None:
            ax.axvline(dca["clinical_threshold"], color="red", lw=1.2, ls=":",
                       label=f"clinical threshold ({dca['clinical_threshold']})")
        ax.set(xlabel="Threshold probability", ylabel="Net benefit",
               title="Decision curve analysis",
               ylim=(-0.05, max(0.05, dca["prevalence"] * 1.5)))
        ax.legend()
        _save(fig, "dca.png")

    temporal = reports.get("temporal", {})
    for section, fname, xlabel in (("by_window", "auroc_by_window.png", "Day-in-stay window"),
                                   ("by_year", "auroc_by_year.png", "Year of admission")):
        block = temporal.get(section)
        bins = (block or {}).get("bins") or []
        pts = [(str(b["bin"]), b["auroc"]) for b in bins if b["auroc"] is not None]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o")
        ax.set(xlabel=xlabel, ylabel="AUROC", title=f"AUROC {section}", ylim=(0.4, 1.0))
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        _save(fig, fname)

    traj = temporal.get("trajectory_by_outcome")
    if traj and traj.get("series"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in traj["series"]:
            ax.plot(traj["windows"],
                    [v if v is not None else float("nan") for v in s["mean_prob"]],
                    marker="o", label=f"{s['outcome']} (label={s['label_value']})")
        ax.set(xlabel="Day-in-stay window", ylabel="Mean predicted probability",
               title="Predicted risk trajectory by outcome")
        ax.legend()
        _save(fig, "trajectory_by_outcome.png")

    ta = reports.get("threshold_analysis", {})
    lm = ta.get("label_metrics")
    if lm and "roc" in curves:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(curves["roc"]["fpr"], curves["roc"]["tpr"], lw=1)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        op_fpr, op_tpr = 1 - lm["specificity"], lm["sensitivity"]
        ax.scatter([op_fpr], [op_tpr], color="red", zorder=5,
                   label=f"t={ta['clinical_threshold']} ({ta.get('clinical_action')})")
        pauc = ta.get("partial_auroc") or {}
        if pauc.get("fpr_range"):
            ax.axvspan(*pauc["fpr_range"], alpha=0.15, color="orange",
                       label=f"pAUC band (std={pauc.get('pauc_standardized')})")
        ax.set(xlabel="FPR", ylabel="TPR",
               title=f"Operating point: sens={lm['sensitivity']}, "
                     f"spec={lm['specificity']}")
        ax.legend(loc="lower right")
        _save(fig, "threshold_analysis.png")

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
