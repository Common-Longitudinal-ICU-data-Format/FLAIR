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

    curves = reports.get("curves", {})
    if "roc" in curves:
        roc, pr = curves["roc"], curves["pr"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(roc["fpr"], roc["tpr"])
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC (AUROC={roc.get('auroc')})")
        axes[1].plot(pr["recall"], pr["precision"])
        base = pr.get("baseline")
        if base is not None:
            axes[1].axhline(base, color="k", ls="--", lw=0.8,
                            label=f"prevalence {base:.4f}")
            axes[1].legend(loc="upper right", fontsize=8)
        axes[1].set(xlabel="Recall", ylabel="Precision",
                    title=f"PR (AUPRC={pr.get('auprc')}, lift×{pr.get('auprc_lift')})")
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
               title=f"Calibration  slope={ext.get('slope')} "
                     f"intercept={ext.get('intercept')}  O:E={ext.get('o_e_ratio')}")
        slope = ext.get("slope")
        if slope is not None and abs(slope - 1) > 0.3:
            ax.text(0.5, 0.02, "probabilities uncalibrated (ranking only) — "
                    "scale_pos_weight inflates risk", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7, color="firebrick")
        ax.legend(loc="upper left")
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
        # zoom to the rare-event decision region so the curve isn't a flat line
        fr = dca.get("focus_range") or [0.0, 1.0]
        ax.set(xlabel="Threshold probability", ylabel="Net benefit",
               title=f"Decision curve analysis (zoom 0–{fr[1]:g})",
               xlim=(fr[0], fr[1]),
               ylim=(min(-0.01, -dca["prevalence"] * 0.5),
                     max(0.01, dca["prevalence"] * 1.2)))
        ax.legend()
        _save(fig, "dca.png")

    temporal = reports.get("temporal", {})
    unit = ((temporal.get("by_window") or {}).get("unit") or "").capitalize()
    win_xlabel = f"{unit}-in-stay window" if unit else "In-stay window"
    # by_year is skipped automatically on date-shifted sites (reason → no bins).
    for section, fname, xlabel in (("by_window", "auroc_by_window.png", win_xlabel),
                                   ("by_year", "auroc_by_year.png", "Year of admission")):
        block = temporal.get(section)
        bins = (block or {}).get("bins") or []
        pts = [b for b in bins if b["auroc"] is not None]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [str(b["bin"]) for b in pts]
        ys = [b["auroc"] for b in pts]
        ax.plot(xs, ys, marker="o")
        if len(pts) <= 10:               # annotate only when uncluttered
            for x, b in zip(xs, pts):
                ax.annotate(f"n={b['n']}\npos={b['n_pos']}", (x, b["auroc"]),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7, color="0.4")
        else:
            ax.tick_params(axis="x", rotation=90, labelsize=6)
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
        ax.set(xlabel=win_xlabel, ylabel="Mean predicted probability",
               title="Predicted risk trajectory by outcome")
        ax.legend()
        _save(fig, "trajectory_by_outcome.png")

    enc = reports.get("encounter", {})
    hist = enc.get("lead_time_hist") or {}
    labels = hist.get("labels")
    if labels and hist.get("counts"):
        # suppressed bins ("<10") render as 0 height — below the cell floor.
        counts = [c if isinstance(c, (int, float)) else 0 for c in hist["counts"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, counts, color="#3b7dd8", edgecolor="white", linewidth=0.4)
        lt = (enc.get("at_clinical_threshold") or {}).get("lead_time_hours") or {}
        med = f"  median {lt['median']} h" if lt.get("median") is not None else ""
        ax.set(xlabel="Lead time: first alarm → event (fixed clinical bins)",
               ylabel="Event stays",
               title=f"Lead-time distribution (caught event stays){med}")
        _save(fig, "lead_time.png")

    oc = enc.get("operating_curve") or {}
    if oc.get("alarms_per_patient_day") and any(s is not None for s in oc.get("stay_sensitivity", [])):
        xs = [x for x, y in zip(oc["alarms_per_patient_day"], oc["stay_sensitivity"])
              if x is not None and y is not None]
        ys = [y for x, y in zip(oc["alarms_per_patient_day"], oc["stay_sensitivity"])
              if x is not None and y is not None]
        if xs:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, ys, marker="o", ms=3)
            ct = oc.get("clinical_threshold")
            if ct in oc["thresholds"]:
                i = oc["thresholds"].index(ct)
                if oc["alarms_per_patient_day"][i] is not None and oc["stay_sensitivity"][i] is not None:
                    ax.scatter([oc["alarms_per_patient_day"][i]], [oc["stay_sensitivity"][i]],
                               color="red", zorder=5, label=f"clinical threshold ({ct})")
                    ax.legend(loc="lower right")
            ax.set(xlabel="Alarms per patient-day", ylabel="Stay-level sensitivity",
                   title="Encounter operating curve", ylim=(0, 1.02))
            _save(fig, "encounter_operating_curve.png")

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
