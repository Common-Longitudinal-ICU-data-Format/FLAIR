"""Operating points → operating_points.json.

The one self-contained decision-point view: everything that depends on WHERE the
decision threshold is put. Two entry points, one per report mode.

``build_episodic_operating_points`` (episodic; one prediction per stay)
---------------------------------------------------------------------
THE scorecard. Every threshold-dependent number the episodic report produces
lives here and nowhere else — discrimination.json, dca.json and fairness.json are
threshold-FREE by construction (they pass operating_point=False /
clinical_marker=False / threshold_metrics=False). Swept over FIVE thresholds, the
clinical threshold from META and ± 0.05 / ± 0.10 around it, so a site can read how
the decision degrades if the policy moves. Per threshold:

  confusion  : {TP,FP,FN,TN}                                    ALARM basis
  metrics    : sensitivity / specificity / PPV / NPV / F1 / FPR / NNE /
               flag-rate, each with a 95% stay bootstrap CI     ALARM basis
  decision   : the act taken (who, and the event rate in each arm) + the decision
               curve's value there vs treat-all / treat-none    DECISION basis
  fairness   : per-subgroup positive rate / TPR / FPR + parity gaps, at THIS
               threshold                                        ALARM basis

TWO BASES, TWO BLOCKS. They are separated because they answer different questions,
and conflating them is how a decision curve gets read backwards.

  ALARM basis (``y_prob >= t``, in BOTH directions) — "how well do we DETECT the
  event". For a ``clinical_direction: "below"`` task like task3 (extubate when
  failure risk < t) the *action* fires on LOW risk, but the *alarm* — the thing the
  model warns about — is still high risk. Scoring the confusion matrix on the action
  would make TP = "extubated AND failed": a clinical disaster booked as a true
  positive, with a "sensitivity" that RISES as the model gets worse. So sens / spec
  / PPV stay in risk-classifier convention regardless of direction, which is also
  what keeps them comparable across tasks.

  DECISION basis (``dca.to_decision_basis``) — "is the ACT worth taking here". This
  is the frame the clinician decides in: for task3 the event is successful
  extubation, the score is 1-p, and the threshold sits at 1-t
  (``decision.dca_threshold``: grid t=0.15 is x=0.85, "extubate when P(success) >=
  85%"). ``decision.net_benefit`` is therefore THE SAME NUMBER as dca.json's curve
  at that x — it is not a second opinion, and it is deliberately NOT derivable from
  this entry's alarm-basis TP/FP.

``build_operating_points`` (continuous; many predictions per stay)
-----------------------------------------------------------------
The stay-level decision view for a continuous task. Evaluated at the task's clinical
threshold from META and clinical ± 0.05 — three points, unless an offset lands outside
the open interval (0, 1), in which case it is dropped and recorded in
``metadata.dropped_offsets`` (see ``threshold_grid``). At EACH threshold it produces:

  confusion    : model ALERT × LABEL per stay (fired = the stay ever crosses the
                 threshold, direction-aware; label = ever-positive) → {TP,FP,FN,TN}
                 + sensitivity / specificity / FPR / PPV / NPV / NNE / F1 /
                 flag-rate, each with a 500-rep stay bootstrap CI. MANDATORY for
                 every continuous task — no config needed.
  sub_analyses : OPTIONAL, present only when the task declares an
                 ``intervention`` config ``{"column", "display"}`` naming a
                 per-stay treatment-time column in the cohort. Three arms keyed on
                 (label, treatment, alert):
                   label_positive          — how many positive stays fired, and
                                             the *hours before treatment* the
                                             alarm first fired (positive = early).
                   label_negative_treated  — of treated non-positive stays, how
                                             many fired and the *hours before
                                             treatment* the alarm fired (same
                                             positive = early convention as above).
                   label_negative_untreated— of never-treated non-positive stays,
                                             the misfire (false-alarm) count/rate.
                                             No timing.

This module is TASK-AGNOSTIC: it never assumes sepsis / antibiotics. The anchor
is whatever timestamp column the task's ``intervention`` config points at, and
every user-facing label is the config's ``display`` string. No intervention
config → the confusion matrix is still produced; ``sub_analyses`` is omitted with
a reason and ``intervention_offsets`` is null.

Privacy: only rates, suppressed counts, quantile summaries and >=MIN_CELL_COUNT
histogram bins are emitted. Confusion + sub-analysis rates carry stay-bootstrap
CIs (`_ci.bootstrap_ci`); the one-row-per-stay collapse makes that an honest
patient-level bootstrap.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._ci import BOOT_REPS, BOOT_SEED, bootstrap_ci
from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress
from flair_benchmark.report.dca import (
    _net_benefit,
    axis_threshold,
    net_benefit_all,
    to_decision_basis,
)
from flair_benchmark.report.discrimination import _confusion_metric, _operating_point
from flair_benchmark.report.fairness import MIN_SUBGROUP_N, SUBGROUP_COLS, _bin_age

_BLOCK = "hospitalization_join_id"
_DELTAS = (-0.05, 0.0, 0.05)   # continuous: clinical threshold and ±0.05

# Episodic sweeps two steps either side of the clinical threshold, so a site can
# see how the decision degrades if the policy moves — the question clinicians
# actually argue about ("what if we set the bar at 0.10 instead of 0.15?").
_EPISODIC_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)


def alert_expr(threshold: float, direction: str) -> pl.Expr:
    """Direction-aware alert: p < t when the action fires on LOW risk."""
    if direction == "below":
        return pl.col("y_prob") < threshold
    return pl.col("y_prob") >= threshold


def build_operating_points(preds: pl.DataFrame, meta: dict) -> dict:
    """Per-threshold confusion matrix + optional per-threshold sub-analyses."""
    t0 = meta.get("clinical_threshold")
    direction = meta.get("clinical_direction", "above")
    iv = meta.get("intervention")
    window_unit = meta.get("window_unit")
    label_col = meta["label_column"]
    md: dict = {
        "clinical_threshold": t0, "clinical_direction": direction,
        "intervention": iv, "window_unit": window_unit,
        "min_cell_count": MIN_CELL_COUNT,
    }
    out: dict = {"metadata": md}
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or t0 is None:
        md["reason"] = ("operating points require a binary task with y_prob and a "
                        "clinical_threshold")
        out["by_threshold"] = []
        out["intervention_offsets"] = None
        return out

    # Same grid primitive the episodic scorecard uses: rounded, and offsets landing
    # outside the OPEN interval (0, 1) are DROPPED rather than clamped. task1 sits at
    # 0.95, so its +0.05 arm would otherwise be pinned to a threshold of 1.0 — which
    # flags nobody, has no PPV, and makes DCA's t/(1-t) odds diverge.
    grid, dropped = threshold_grid(t0, _DELTAS)
    t0r = round(float(t0), 4)
    md["thresholds"] = [t for t, _o in grid]
    md["offsets"] = [o for _t, o in grid]
    md["dropped_offsets"] = dropped
    iv_col = iv["column"] if iv else None
    has_dttm = "prediction_dttm" in preds.columns
    if not iv_col:
        md["sub_analyses_reason"] = (
            "no intervention/treatment column configured; sub-analyses omitted")

    by_threshold, clinical_agg = [], None
    for t, off in grid:
        agg = _agg_at(preds, t, direction, iv_col, has_dttm, label_col)
        entry: dict = {
            "threshold": t,
            "offset": off,
            "is_clinical": t == t0r,
            "confusion": _confusion_at(agg["label"].to_numpy(),
                                       agg["fired"].to_numpy(), t),
        }
        if iv_col:
            entry["sub_analyses"] = {
                "label_positive": _positive_arm(agg, has_dttm),
                "label_negative_treated": _treated_negative_arm(agg, has_dttm),
                "label_negative_untreated": _untreated_negative_arm(agg),
            }
        by_threshold.append(entry)
        if t == t0r:
            clinical_agg = agg

    out["by_threshold"] = by_threshold
    out["intervention_offsets"] = (
        _intervention_offsets(clinical_agg, window_unit, has_dttm)
        if iv_col and clinical_agg is not None else None)
    return out


def _agg_at(preds: pl.DataFrame, t: float, direction: str, iv_col: str | None,
            has_dttm: bool, label_col: str) -> pl.DataFrame:
    """One row per stay at threshold t: label, fired, first-alert + treatment ts."""
    alert = alert_expr(t, direction)
    aggs = [(pl.col(label_col).max() > 0).cast(pl.Int8).alias("label"),
            alert.any().alias("fired")]
    if has_dttm:
        aggs += [pl.col("prediction_dttm").filter(alert).min().alias("first_alert_dttm"),
                 pl.col("prediction_dttm").min().alias("_stay_start_dttm")]
    if iv_col is not None:
        aggs.append(pl.col(iv_col).min().alias("_treat_dttm"))
    return preds.group_by(_BLOCK).agg(aggs)


# --------------------------------------------------------------- confusion matrix


def _confusion_at(label: np.ndarray, fired: np.ndarray, t: float) -> dict:
    """Alert × label 2×2 + rates (with stay-bootstrap CIs) at threshold t."""
    lab = label.astype(int)
    fire = fired.astype(bool)
    n = len(lab)
    tp = int((fire & (lab == 1)).sum())
    fp = int((fire & (lab == 0)).sum())
    fn = int((~fire & (lab == 1)).sum())
    tn = int((~fire & (lab == 0)).sum())

    fns = {k: _confusion_stat(k)
           for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "fpr")}
    ci = bootstrap_ci(lab, fire.astype(float), fns)
    point = {k: _confusion_metric(tp, fp, fn, tn, k)
             for k in ("sensitivity", "specificity", "ppv", "npv", "f1")}
    fpr = fp / (fp + tn) if (fp + tn) else None
    nne, nne_ci = _nne_from_ppv(point["ppv"], ci.get("ppv"))
    return {
        "threshold": round(float(t), 4),
        "n": n,
        "tp": suppress(tp), "fp": suppress(fp),
        "fn": suppress(fn), "tn": suppress(tn),
        **{k: _r1(v) for k, v in point.items()},
        **{f"{k}_ci": ci[k] for k in point},
        "fpr": _r1(fpr), "fpr_ci": ci.get("fpr"),
        "nne": nne, "nne_ci": nne_ci,
        "flag_rate": round((tp + fp) / n, 4) if n else None,
    }


def _confusion_stat(name: str):
    """Bootstrap stat_fn over (label, fired-as-0/1): one confusion rate."""
    def _fn(yt: np.ndarray, yp: np.ndarray) -> float | None:
        pred = yp >= 0.5
        tp = int((pred & (yt == 1)).sum())
        fp = int((pred & (yt == 0)).sum())
        fn = int((~pred & (yt == 1)).sum())
        tn = int((~pred & (yt == 0)).sum())
        if name == "fpr":
            return fp / (fp + tn) if (fp + tn) else None
        return _confusion_metric(tp, fp, fn, tn, name)
    return _fn


def _nne_from_ppv(ppv: float | None,
                  ci: list[float] | None) -> tuple[float | None, list | None]:
    """NNE = 1/PPV (alarms per true positive); CI is the inverted PPV CI."""
    if not ppv or ppv <= 0:
        return None, None
    nne = round(1.0 / ppv, 1)
    if ci and ci[0] is not None and ci[1] is not None:
        hi = round(1.0 / ci[0], 1) if ci[0] > 0 else None
        lo = round(1.0 / ci[1], 1) if ci[1] > 0 else None
        return nne, [lo, hi]
    return nne, None


# ------------------------------------------------------------------ sub-analyses


def _positive_arm(agg: pl.DataFrame, has_dttm: bool) -> dict:
    """Sub-1 (label==1): fire rate + hours-before-treatment the alarm fired."""
    pos = agg.filter(pl.col("label") == 1)
    n = pos.height
    if n < MIN_CELL_COUNT:
        return {"n_stays": suppress(n), "suppressed": True}
    n_fired = int(pos["fired"].sum())
    arm: dict = {"n_stays": n, "n_fired": suppress(n_fired),
                 "pct_fired": round(n_fired / n, 4)}
    if has_dttm:
        fp = pos.filter(pl.col("fired") & pl.col("_treat_dttm").is_not_null()
                        & pl.col("first_alert_dttm").is_not_null())
        # positive = alarm fired BEFORE the treatment time.
        hb = ((pl.col("_treat_dttm") - pl.col("first_alert_dttm"))
              .dt.total_seconds() / 3600)
        arm["hours_before_treatment"] = _delta_summary(
            fp.select(hb.alias("d"))["d"].to_numpy().astype(float))
    return arm


def _treated_negative_arm(agg: pl.DataFrame, has_dttm: bool) -> dict:
    """Sub-2 (label==0 & treated): fire rate + hours before treatment.

    Same convention as Sub-1 (positive = the alarm fired BEFORE the treatment),
    so the two lead-time panels read on one consistent axis.
    """
    neg = agg.filter((pl.col("label") == 0) & pl.col("_treat_dttm").is_not_null())
    n = neg.height
    if n < MIN_CELL_COUNT:
        return {"n": suppress(n), "suppressed": True}
    n_fired = int(neg["fired"].sum())
    arm: dict = {"n": n, "n_fired": suppress(n_fired),
                 "pct_fired": round(n_fired / n, 4)}
    if has_dttm:
        fr = neg.filter(pl.col("fired") & pl.col("first_alert_dttm").is_not_null())
        hb = ((pl.col("_treat_dttm") - pl.col("first_alert_dttm"))
              .dt.total_seconds() / 3600)
        arm["hours_before_treatment"] = _delta_summary(
            fr.select(hb.alias("d"))["d"].to_numpy().astype(float))
    return arm


def _untreated_negative_arm(agg: pl.DataFrame) -> dict:
    """Sub-3 (label==0 & never treated): misfire (false-alarm) count/rate only."""
    neg = agg.filter((pl.col("label") == 0) & pl.col("_treat_dttm").is_null())
    n = neg.height
    if n < MIN_CELL_COUNT:
        return {"n": suppress(n), "suppressed": True}
    fired = neg["fired"].to_numpy().astype(float)
    n_mis = int(fired.sum())
    return {"n": n, "n_misfired": suppress(n_mis),
            "misfire_rate": round(float(fired.mean()), 4),
            "misfire_rate_ci": _rate_ci(fired)}


# ----------------------------------------------------------------------- helpers


def _delta_summary(deltas: np.ndarray) -> dict | None:
    """Quantile + binned-histogram summary of an hours distribution.

    Histogram bins below MIN_CELL_COUNT are dropped (their total is reported) so
    no small cell leaves the site. Bin width scales with spread so the histogram
    stays plottable at hourly and daily granularity alike.
    """
    if len(deltas) < MIN_CELL_COUNT:
        return {"n": f"<{MIN_CELL_COUNT}", "suppressed": True}
    p10, p25, p75, p90 = np.percentile(deltas, [10, 25, 75, 90])
    out = {
        "n": len(deltas),
        "median": round(float(np.median(deltas)), 2),
        "median_ci": bootstrap_ci(
            deltas, deltas,
            {"median": lambda a, _b: float(np.median(a))})["median"],
        "mean": round(float(deltas.mean()), 2),
        "p10": round(float(p10), 2), "p25": round(float(p25), 2),
        "p75": round(float(p75), 2), "p90": round(float(p90), 2),
    }
    span = float(deltas.max() - deltas.min())
    width = max(1.0, round(span / 24)) if span else 1.0
    lo = np.floor(deltas.min() / width) * width
    edges = np.arange(lo, deltas.max() + width, width)
    counts, edges = np.histogram(deltas, bins=edges if len(edges) > 1 else 2)
    bins, n_suppressed = [], 0
    for c, e_lo, e_hi in zip(counts, edges[:-1], edges[1:]):
        if int(c) >= MIN_CELL_COUNT:
            bins.append({"lo": round(float(e_lo), 2),
                         "hi": round(float(e_hi), 2), "count": int(c)})
        elif c:
            n_suppressed += int(c)
    out["histogram"] = bins
    out["histogram_bin_hours"] = float(width)
    out["n_in_suppressed_bins"] = suppress(n_suppressed)
    return out


def _intervention_offsets(agg: pl.DataFrame, window_unit: str | None,
                          has_dttm: bool) -> dict | None:
    """Mean ± SD treatment hospitalization time (for the sensitivity red line).

    Places the treatment on the forward landmark axis (hospitalization day/hour;
    the stay's first prediction window sits at 1). Anchored on the stay START —
    never on the event or outcome.
    """
    if not has_dttm:
        return None
    treated = agg.filter(pl.col("_treat_dttm").is_not_null())
    if treated.height < MIN_CELL_COUNT:
        return None
    hours_per_window = 24.0 if window_unit == "day" else 1.0
    off = (treated.select(
        ((pl.col("_treat_dttm") - pl.col("_stay_start_dttm")).dt.total_seconds()
         / 3600 / hours_per_window + 1).alias("o"))["o"].to_numpy().astype(float))
    return {
        "n": treated.height,
        "window_unit": window_unit,
        "mean_hospitalization_time": round(float(off.mean()), 2),
        "sd": round(float(off.std(ddof=1)), 2) if len(off) > 1 else None,
    }


def _rate_ci(indicator: np.ndarray) -> list[float] | None:
    """Stay-bootstrap 95% CI of a proportion (mean of a 0/1 indicator)."""
    return bootstrap_ci(
        indicator, indicator,
        {"rate": lambda a, _b: float(a.mean())})["rate"]


def _r1(x) -> float | None:
    return None if x is None else round(float(x), 4)


# ============================================================ EPISODIC SCORECARD


def threshold_grid(t0: float,
                   offsets: tuple[float, ...] = _EPISODIC_OFFSETS
                   ) -> tuple[list[tuple[float, float]], list[dict]]:
    """(grid, dropped) for a ± sweep around t0: [(threshold, offset), ...].

    THE grid primitive for both report modes — episodic passes _EPISODIC_OFFSETS,
    continuous passes _DELTAS. One rule, so the two cannot drift.

    Rounding is not cosmetic — it is required. ``0.15 - 0.10`` is
    0.049999999999999996, ``0.2 + 0.1`` is 0.30000000000000004, and ``0.95 - 0.05``
    is 0.8999999999999999, so an unrounded grid would threshold at one value while
    publishing another, and would never compare equal to the clinical threshold.

    Offsets landing outside the OPEN interval (0, 1) are DROPPED, not clamped: at
    t=0 every stay is flagged, at t=1 none is, and DCA's t/(1-t) odds diverge at 1.
    A threshold of 1.0 is not a threshold — ``tasks._validate_taxonomy`` already
    enforces 0 < clinical_threshold < 1 on the policy itself, and this keeps the
    DERIVED grid honest to the same invariant. Clamping would also silently collapse
    two offsets onto the same threshold for a task near the edge (e.g. t0=0.95 →
    both +0.05 and +0.10 become 1.0). Whatever is dropped is reported, never
    silently swallowed.
    """
    grid: list[tuple[float, float]] = []
    dropped: list[dict] = []
    for o in offsets:
        t = round(t0 + o, 4)
        off = round(o, 4)
        if not 0.0 < t < 1.0:
            dropped.append({"offset": off, "threshold": t,
                            "reason": "outside the open interval (0, 1)"})
        elif any(t == g for g, _ in grid):
            dropped.append({"offset": off, "threshold": t,
                            "reason": "duplicate threshold after rounding"})
        else:
            grid.append((t, off))
    return grid, dropped


def build_episodic_operating_points(preds: pl.DataFrame, meta: dict) -> dict:
    """Every threshold-dependent episodic metric, swept over the 5-point grid."""
    t0 = meta.get("clinical_threshold")
    direction = meta.get("clinical_direction", "above")
    label_col = meta["label_column"]
    md: dict = {"clinical_threshold": t0, "clinical_direction": direction,
                "min_cell_count": MIN_CELL_COUNT,
                "flag_rule": "y_prob >= threshold (alarm convention)",
                "net_benefit_basis": (
                    "decision basis — identical to dca.json's curve. Each entry's "
                    "decision.net_benefit is the net benefit of the ACT at that "
                    "threshold, read at decision.dca_threshold on the decision "
                    "curve's axis (1-t for a 'below' task). It is NOT derivable "
                    "from this entry's alarm-basis TP/FP.")}
    out: dict = {"task_type": meta["task_type"], "metadata": md}

    if (meta["task_type"] != "binary" or "y_prob" not in preds.columns
            or t0 is None):
        md["reason"] = ("the operating-points scorecard requires a binary task "
                        "with y_prob and a clinical_threshold")
        out["by_threshold"] = []
        return out

    grid, dropped = threshold_grid(t0)
    md["offsets"] = [o for _t, o in grid]
    md["thresholds"] = [t for t, _o in grid]
    md["dropped_offsets"] = dropped

    y_true = preds[label_col].to_numpy().astype(int)
    y_prob = preds["y_prob"].to_numpy().astype(float)
    n = len(y_true)
    prevalence = round(float(y_true.mean()), 6) if n else None
    out["prevalence"] = prevalence

    t0r = round(float(t0), 4)
    md["clinical_index"] = next(
        (i for i, (t, _o) in enumerate(grid) if t == t0r), None)

    fair = _fairness_by_threshold(preds, meta, [t for t, _o in grid])

    by_threshold = []
    for i, (t, off) in enumerate(grid):
        entry = _episodic_entry(y_true, y_prob, meta, t, off, prevalence)
        entry["is_clinical"] = t == t0r
        if fair is not None:
            entry["fairness"] = fair[i]
        by_threshold.append(entry)
    out["by_threshold"] = by_threshold
    return out


def _episodic_entry(y_true: np.ndarray, y_prob: np.ndarray, meta: dict,
                    t: float, offset: float, prevalence: float | None) -> dict:
    """One grid point. TWO bases, kept in two blocks so neither can be misread.

    `confusion` + `metrics` are the ALARM basis (y_prob >= t): how well do we
    DETECT the event. Delegated to discrimination._operating_point, so the clinical
    entry is bit-identical to the operating point the report used to publish inside
    discrimination.json — the sweep adds thresholds, it does not restate the
    existing numbers.

    `decision` is the DECISION basis: the act actually taken. For a "below" task
    the act fires on LOW risk and its net benefit is computed on the complement
    (dca.to_decision_basis), which is the frame the clinician decides in — so the
    number here IS dca.json's curve at dca_threshold, not a second opinion.
    """
    op = _operating_point(y_true, y_prob, meta, t=t)
    n = len(y_true)
    pred = y_prob >= t
    n_flagged = int(pred.sum())
    tp = int((pred & (y_true == 1)).sum())
    fp = n_flagged - tp
    fn = int((~pred & (y_true == 1)).sum())
    tn = n - n_flagged - fn

    rates = {k: op.get(k) for k in
             ("sensitivity", "specificity", "ppv", "npv", "f1")}
    cis = {f"{k}_ci": op.get(f"{k}_ci") for k in rates}
    fpr = fp / (fp + tn) if (fp + tn) else None
    fpr_ci = bootstrap_ci(y_true, y_prob, {"fpr": _fpr_stat(t)})["fpr"]
    nne, nne_ci = _nne_from_ppv(rates["ppv"], cis["ppv_ci"])

    metrics: dict = {
        **{k: _r1(v) for k, v in rates.items()}, **cis,
        "fpr": _r1(fpr), "fpr_ci": fpr_ci,
        "nne": nne, "nne_ci": nne_ci,
        "flag_rate": round(n_flagged / n, 4) if n else None,
    }
    entry: dict = {
        "threshold": t,
        "offset": offset,
        "confusion": {"n": n, "n_flagged": n_flagged,
                      "tp": suppress(tp), "fp": suppress(fp),
                      "fn": suppress(fn), "tn": suppress(tn)},
        "metrics": metrics,
        "decision": _decision_at(y_true, y_prob, meta, t, op),
    }

    # A threshold that flags almost nobody yields a PPV derived from a handful of
    # patients: unstable, and disclosive at these cell counts. Null it and say so —
    # the same rule dca.py applies to its curve. (Net benefit is suppressed on its
    # OWN arm, n_acted, inside _decision_at: for a "below" task the acted arm is the
    # complement of the flagged one, so the two floors are not the same floor.)
    if n_flagged < MIN_CELL_COUNT:
        for k in ("ppv", "ppv_ci", "nne", "nne_ci"):
            metrics[k] = None
        entry["suppressed"] = True
        entry["suppressed_reason"] = (
            f"fewer than {MIN_CELL_COUNT} stays flagged at this threshold")
    return entry


def _decision_at(y_true: np.ndarray, y_prob: np.ndarray, meta: dict,
                 t: float, op: dict) -> dict:
    """The act taken at t, and what it is worth — all on the DECISION basis.

    Absorbs discrimination._action_block (who gets acted on, and the observed event
    rate in each arm) and adds the decision curve's verdict at this threshold. Net
    benefit is computed through dca's own primitives on dca's own frame, so
    `net_benefit` here equals dca.json's curve at `dca_threshold` by construction.
    """
    direction = meta.get("clinical_direction", "above")
    action = dict(op.get("action") or {})
    t_axis = axis_threshold(t, direction)
    yt, yp = to_decision_basis(y_true, y_prob, direction)

    nb = _net_benefit(yt, yp, t_axis)
    nb_ci = bootstrap_ci(
        yt, yp, {"nb": lambda a, b: _net_benefit(a, b, t_axis)})["nb"]
    prev = float(yt.mean()) if len(yt) else None
    nb_all = None if prev is None else net_benefit_all(prev, t_axis)

    out = {
        **action,
        # where this grid point sits on the decision curve's x-axis: for task3,
        # grid t=0.15 (P failure) is 0.85 (P successful extubation).
        "dca_threshold": t_axis,
        "net_benefit": None if nb is None else round(float(nb), 6),
        "net_benefit_ci": nb_ci,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": 0.0,
        "clinically_useful": (None if nb is None or nb_all is None
                              else bool(nb > max(nb_all, 0.0))),
    }
    # the acted arm is what the net benefit is derived from — suppress on IT.
    n_acted = action.get("n_acted")
    if isinstance(n_acted, str) or (isinstance(n_acted, int)
                                    and n_acted < MIN_CELL_COUNT):
        for k in ("net_benefit", "net_benefit_ci", "clinically_useful"):
            out[k] = None
        out["suppressed_reason"] = (
            f"fewer than {MIN_CELL_COUNT} stays acted on at this threshold")
    return out


def _fpr_stat(t: float):
    """Bootstrap stat_fn: false-positive rate at threshold t (alarm convention)."""
    def _fn(yt: np.ndarray, yp: np.ndarray) -> float | None:
        pred = yp >= t
        fp = int((pred & (yt == 0)).sum())
        tn = int((~pred & (yt == 0)).sum())
        return fp / (fp + tn) if (fp + tn) else None
    return _fn


# ------------------------------------------------- per-threshold subgroup equity


def _fairness_by_threshold(preds: pl.DataFrame, meta: dict,
                           thresholds: list[float]) -> list[dict] | None:
    """Subgroup positive-rate / TPR / FPR + parity gaps, at EVERY grid threshold.

    Returns one dict per threshold (aligned with ``thresholds``), or None when the
    frame carries no demographics.

    Gaps are max-minus-min across subgroup levels, as in fairness.py. The CI keeps
    that module's pinned-levels guard: the level set is fixed from the FULL data
    (n >= MIN_SUBGROUP_N there) and a resample that loses one of those levels is
    discarded whole, rather than quietly taking the gap over a smaller group set —
    which would shrink max-minus-min toward zero and report a fairness the model
    does not have.

    One resample index is drawn per replicate and reused across every threshold and
    every subgroup column, so the gaps are mutually coherent (same bootstrap
    cohort). The arithmetic is numpy rather than the polars filters fairness.py
    uses, because this runs 5 thresholds deep and the semantics — not the
    implementation — are what must match.
    """
    df = _bin_age(preds)
    cols = [c for c in SUBGROUP_COLS if c in df.columns]
    if not cols:
        return None

    label = df[meta["label_column"]].to_numpy().astype(int)
    prob = df["y_prob"].to_numpy().astype(float)
    preds_at = [prob >= t for t in thresholds]

    # Pin each column's eligible levels from the full data, and encode them as
    # integer codes so the bootstrap can index with numpy instead of re-filtering.
    pinned: dict[str, tuple[np.ndarray, list[str]]] = {}
    for col in cols:
        values = df[col].to_numpy()
        levels = [lv for lv in dict.fromkeys(v for v in values if v is not None)
                  if int((values == lv).sum()) >= MIN_SUBGROUP_N]
        if len(levels) < 2:
            continue
        codes = np.full(len(values), -1, dtype=int)
        for k, lv in enumerate(levels):
            codes[values == lv] = k
        pinned[col] = (codes, [str(lv) for lv in levels])
    if not pinned:
        return None

    # point estimates
    per_threshold: list[dict] = []
    for pred in preds_at:
        subgroups, parity = {}, {}
        for col, (codes, levels) in pinned.items():
            subgroups[col] = {
                lv: _level_rates(label[codes == k], pred[codes == k])
                for k, lv in enumerate(levels)
            }
            dp, tg, fg = _gaps(codes, label, pred, len(levels))
            parity[col] = {
                "reference_group": max(
                    levels, key=lambda lv: (int((codes == levels.index(lv)).sum()), lv)),
                "demographic_parity_gap": _r1(dp),
                "equalized_odds_tpr_gap": _r1(tg),
                "equalized_odds_fpr_gap": _r1(fg),
            }
        per_threshold.append({"subgroups": subgroups, "parity": parity})

    # CIs: one shared resample index per replicate, reused across every
    # (threshold, column) pair.
    rng = np.random.default_rng(BOOT_SEED)
    n = len(label)
    samples: dict[tuple[int, str, str], list[float]] = {}
    for _ in range(BOOT_REPS):
        idx = rng.integers(0, n, n)
        lab_b = label[idx]
        for ti, pred in enumerate(preds_at):
            pred_b = pred[idx]
            for col, (codes, levels) in pinned.items():
                dp, tg, fg = _gaps(codes[idx], lab_b, pred_b, len(levels))
                for name, val in (("demographic_parity_gap", dp),
                                  ("equalized_odds_tpr_gap", tg),
                                  ("equalized_odds_fpr_gap", fg)):
                    if val is not None:
                        samples.setdefault((ti, col, name), []).append(val)

    for ti, block in enumerate(per_threshold):
        for col in block["parity"]:
            for name in ("demographic_parity_gap", "equalized_odds_tpr_gap",
                         "equalized_odds_fpr_gap"):
                vals = samples.get((ti, col, name), [])
                block["parity"][col][f"{name}_ci"] = (
                    _pct(vals) if len(vals) >= BOOT_REPS // 2 else None)
    return per_threshold


def _level_rates(label: np.ndarray, pred: np.ndarray) -> dict:
    """One subgroup level at one threshold: who gets flagged, at what error rates."""
    pos = label == 1
    neg = label == 0
    return {
        "n": int(len(label)),
        "positive_rate": _r1(float(pred.mean())) if len(pred) else None,
        "tpr": _r1(float(pred[pos].mean())) if pos.any() else None,
        "fpr": _r1(float(pred[neg].mean())) if neg.any() else None,
    }


def _gaps(codes: np.ndarray, label: np.ndarray, pred: np.ndarray,
          n_levels: int) -> tuple[float | None, float | None, float | None]:
    """(demographic parity, equalized-odds TPR, equalized-odds FPR) gaps.

    Returns all-None if a pinned level is empty in this (re)sample — the caller
    discards the replicate rather than comparing a shrunken set of groups.
    """
    pos_rates, tprs, fprs = [], [], []
    for k in range(n_levels):
        m = codes == k
        if not m.any():
            return None, None, None
        lab_k, pred_k = label[m], pred[m]
        pos_rates.append(float(pred_k.mean()))
        if (lab_k == 1).any():
            tprs.append(float(pred_k[lab_k == 1].mean()))
        if (lab_k == 0).any():
            fprs.append(float(pred_k[lab_k == 0].mean()))
    if len(pos_rates) < 2:
        return None, None, None
    dp = max(pos_rates) - min(pos_rates)
    tg = max(tprs) - min(tprs) if len(tprs) >= 2 else None
    fg = max(fprs) - min(fprs) if len(fprs) >= 2 else None
    return dp, tg, fg


def _pct(vals: list[float]) -> list[float]:
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]
