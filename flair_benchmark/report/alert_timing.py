"""Alert-vs-intervention timing: did the model fire before the clinician acted?

Embedded into ``leadtime.json`` as the ``alert_vs_intervention`` section (see
report/continuous.py). Generalizes the sepsis question "did the model fire
before antibiotics were administered, and by how much?" to any continuous task
whose META declares an ``intervention`` config::

    META["intervention"] = {"column": "first_abx_dttm", "display": "antibiotics"}

The column is generated at task-BUILD time and stored in the cohort parquet
(task4 emits the first CMS-qualifying antibiotic ``admin_dttm`` per stay); the
report only consumes it — it never touches raw CLIF data. When the config is
set, the orchestrator REQUIRES the column on preds (directly or joined via
--cohort).

Three arms, per stay (first alert = earliest window where the model crosses the
clinical threshold, direction-aware):

  label_positive                     : e.g. sepsis patients — did the model fire,
                                       did it fire BEFORE the intervention, and
                                       the (t_first_alert - t_intervention)
                                       distribution in real hours (negative =
                                       fired before antibiotics).
  label_negative_with_intervention   : treated but never labeled positive —
                                       same fire-before-treatment view.
  label_negative_without_intervention: never treated — did the model fire at
                                       all (the false-alarm rate).

Without an intervention config (tasks 1/2 today) the section degrades to the
generic fire/no-fire rates by label, with a visible reason.

Privacy: only rates, quantile summaries and >=MIN_CELL_COUNT histogram bins are
emitted — never per-stay timestamps. CIs are stay-level percentile bootstraps
via the shared `_ci.bootstrap_ci`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._ci import bootstrap_ci
from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress


def alert_expr(threshold: float, direction: str) -> pl.Expr:
    """Direction-aware alert: p < t when the action fires on LOW risk."""
    if direction == "below":
        return pl.col("y_prob") < threshold
    return pl.col("y_prob") >= threshold


def build_alert_timing(preds: pl.DataFrame, meta: dict) -> dict:
    """Three-arm alert-vs-intervention timing (see module docstring)."""
    t = meta.get("clinical_threshold")
    direction = meta.get("clinical_direction", "above")
    iv = meta.get("intervention")
    out: dict = {"metadata": {
        "clinical_threshold": t, "clinical_direction": direction,
        "intervention": iv, "delta_unit": "hour",
        "window_unit": meta.get("window_unit"),
        "min_cell_count": MIN_CELL_COUNT,
    }}
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or t is None:
        out["metadata"]["reason"] = (
            "alert timing requires a binary task with y_prob and a "
            "clinical_threshold")
        return out

    block = "hospitalization_join_id"
    label_col = meta["label_column"]
    alert = alert_expr(t, direction)
    has_dttm = "prediction_dttm" in preds.columns

    aggs = [(pl.col(label_col).max() > 0).alias("_outcome"),
            alert.any().alias("_fired")]
    if has_dttm:
        aggs += [pl.col("prediction_dttm").filter(alert).min()
                 .alias("_first_alert_dttm"),
                 pl.col("prediction_dttm").min().alias("_stay_start_dttm")]
    iv_col = iv["column"] if iv else None
    if iv_col is not None:
        # constant per stay in the cohort; min() is null-safe if a row slipped.
        aggs.append(pl.col(iv_col).min().alias("_intervention_dttm"))
    stay = preds.group_by(block).agg(aggs)

    pos = stay.filter(pl.col("_outcome"))
    neg = stay.filter(~pl.col("_outcome"))

    if iv_col is None:
        out["metadata"]["reason"] = "no intervention configured for this task"
        out["label_positive"] = _fire_arm(pos)
        out["label_negative"] = _fire_arm(neg, rate_key="false_alarm_rate")
        out["intervention_offsets"] = None
        return out

    treated = pl.col("_intervention_dttm").is_not_null()
    out["label_positive"] = _fire_arm(pos)
    out["label_positive"]["with_intervention"] = _timing_arm(
        pos.filter(treated), has_dttm)
    out["label_negative_with_intervention"] = {
        **_fire_arm(neg.filter(treated)),
        "with_intervention": _timing_arm(neg.filter(treated), has_dttm)}
    out["label_negative_without_intervention"] = _fire_arm(
        neg.filter(~treated), rate_key="false_alarm_rate")
    out["intervention_offsets"] = _intervention_offsets(
        stay.filter(treated), meta.get("window_unit"), has_dttm)
    return out


def _fire_arm(stays: pl.DataFrame, rate_key: str = "fire_rate") -> dict:
    """n / n_fired / rate (+ stay-bootstrap CI) for one arm; n<10 suppressed."""
    n = stays.height
    if n < MIN_CELL_COUNT:
        return {"n": f"<{MIN_CELL_COUNT}", "suppressed": True}
    fired = stays["_fired"].to_numpy().astype(float)
    return {
        "n": n,
        "n_fired": suppress(int(fired.sum())),
        rate_key: round(float(fired.mean()), 4),
        f"{rate_key}_ci": _rate_ci(fired),
    }


def _timing_arm(stays: pl.DataFrame, has_dttm: bool) -> dict:
    """Fired-before-intervention rate + (t_alert - t_intervention) summary.

    `stays` is already restricted to intervention_dttm non-null. The rate's
    denominator is ALL treated stays in the arm (a stay that never fired did
    not fire before treatment); the delta distribution is over stays that both
    fired and were treated.
    """
    n = stays.height
    if n < MIN_CELL_COUNT:
        return {"n": f"<{MIN_CELL_COUNT}", "suppressed": True}
    out: dict = {"n": n}
    if not has_dttm:
        out["reason"] = ("prediction_dttm not available; cannot compare alert "
                         "and intervention times")
        return out
    before = (
        stays.select((pl.col("_first_alert_dttm").is_not_null()
                      & (pl.col("_first_alert_dttm")
                         < pl.col("_intervention_dttm")))
                     .alias("b"))["b"].to_numpy().astype(float))
    out["n_fired_before_intervention"] = suppress(int(before.sum()))
    out["fired_before_intervention_rate"] = round(float(before.mean()), 4)
    out["fired_before_intervention_rate_ci"] = _rate_ci(before)
    deltas = (
        stays.filter(pl.col("_first_alert_dttm").is_not_null())
        .select(((pl.col("_first_alert_dttm") - pl.col("_intervention_dttm"))
                 .dt.total_seconds() / 3600).alias("d"))["d"]
        .to_numpy().astype(float))
    out["delta_hours"] = _delta_summary(deltas)
    return out


def _delta_summary(deltas: np.ndarray) -> dict | None:
    """Quantile + binned-histogram summary of (t_alert - t_intervention) hours.

    Negative = the model fired BEFORE the intervention. Histogram bins below
    MIN_CELL_COUNT are dropped (their total is reported), so no small cell
    leaves the site.
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
    # bin width scales with spread so the histogram stays plottable at both
    # hourly (task4) and daily (tasks 1/2) granularity.
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


def _intervention_offsets(treated: pl.DataFrame, window_unit: str | None,
                          has_dttm: bool) -> dict | None:
    """Mean +/- SD of the intervention's hospitalization time, in window units.

    Places the intervention on the forward landmark axis (hospitalization
    day/hour; the stay's first prediction window sits at 1) so the viz can
    draw the red mean +/- SD line on the leadtime sensitivity curve. Anchored
    on the stay START — never on the event or outcome.
    """
    if not has_dttm or treated.height < MIN_CELL_COUNT:
        return None
    hours_per_window = 24.0 if window_unit == "day" else 1.0
    off = (
        treated.select(((pl.col("_intervention_dttm") - pl.col("_stay_start_dttm"))
                        .dt.total_seconds() / 3600 / hours_per_window + 1)
                       .alias("o"))["o"].to_numpy().astype(float))
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
