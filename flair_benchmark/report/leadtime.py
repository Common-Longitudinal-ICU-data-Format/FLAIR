"""Lead-time extraction → leadtime.json (companion to the landmark report).

Early-warning views on the same FORWARD landmark grid as landmark.py
(hospitalization day/hour t = the slice ``window_number == t``; risk set =
stays still at risk at t — nothing anchors on the event or outcome):

  curve  : per-landmark detection sensitivity, ALARM basis (p >= t in BOTH
           directions — how much of the high-risk signal is visible at time t;
           see report/_basis.py for why detection is never scored on the
           action side), each with a 95% CI.
  median_first_alert_hospitalization_time : per caught ever-positive stay, the
           hospitalization time (window_number) of the model's FIRST alert on
           the ACTION basis (direction-aware — for a "below" task the model
           "fires" when risk drops under t) — how early in the stay the model
           first fires; reported as a median (+ stay-bootstrap CI) and a
           percent-of-positives-caught.
  trajectory : mean predicted risk by eventual outcome at each time.

Sensitivity reuses _confusion.mk_stat + _ci.bootstrap_ci so the CI method
matches the rest of the report. The median first-alert CI is a stay-level
percentile bootstrap (resample caught stays).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._basis import action_expr
from flair_benchmark.report._ci import bootstrap_ci, scalar_ci
from flair_benchmark.report._confusion import mk_stat
from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress
from flair_benchmark.report.landmark import partition_windows, resolve_n_landmarks


def build_leadtime(preds: pl.DataFrame, meta: dict,
                   n_landmarks: int | None = None,
                   parts: dict[int, pl.DataFrame] | None = None) -> dict:
    """Per-landmark sensitivity curve + median first-alert time, with CIs.

    ``n_landmarks`` defaults to the task's ``META['landmark_horizon']`` (same
    resolution as the landmark report, so the two stay on one grid).
    ``parts`` (from :func:`landmark.partition_windows`) shares the one-pass
    window partition with the landmark report.
    """
    out: dict = {"task_type": meta["task_type"], "metadata": {}}
    t = meta.get("clinical_threshold")
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or t is None:
        out["metadata"]["reason"] = (
            "lead-time requires a binary task with y_prob and a clinical_threshold")
        return out
    n_landmarks = resolve_n_landmarks(meta, n_landmarks)

    block = "hospitalization_join_id"
    label_col = meta["label_column"]
    window_unit = meta.get("window_unit")
    direction = meta.get("clinical_direction", "above")
    if parts is None:
        parts = partition_windows(preds)
    empty = preds.clear()

    curve = _sensitivity_curve(parts, empty, label_col, t, n_landmarks,
                               window_unit)
    lead = _median_first_alert(preds, block, label_col, t, direction)
    trajectory = _risk_trajectory(preds, label_col, block, n_landmarks,
                                  window_unit)

    out["prevalence"] = round(float((preds[label_col] > 0).mean()), 6)
    out["curve"] = curve
    out["metrics"] = lead
    out["trajectory"] = trajectory
    out["metadata"] = {"clinical_threshold": t, "window_unit": window_unit,
                       "min_cell_count": MIN_CELL_COUNT,
                       # the two panels deliberately use different bases —
                       # detection is scored on the alarm, timing on the action.
                       "curve_basis": "alarm (y_prob >= threshold, both directions)",
                       "first_alert_basis": "action (direction-aware)"}
    return out


def _sensitivity_curve(parts: dict[int, pl.DataFrame], empty: pl.DataFrame,
                       label_col: str, t: float,
                       n_landmarks: int, window_unit: str | None) -> list[dict]:
    """Detection sensitivity (+ 95% CI) at each hospitalization time, n<10 suppressed."""
    sens_fn = mk_stat(t, "sensitivity")   # ALARM basis (p >= t), by design
    points: list[dict] = []
    for lm in range(n_landmarks):
        sl = parts.get(lm + 1, empty)
        base = {"hospitalization_time": lm + 1, "window_unit": window_unit,
                "n": sl.height}
        if sl.height < MIN_CELL_COUNT:
            points.append({**base, "n": f"<{MIN_CELL_COUNT}",
                           "suppressed": True, "sensitivity": None,
                           "sensitivity_ci": None})
            continue
        y_true = sl[label_col].to_numpy().astype(int)
        y_prob = sl["y_prob"].to_numpy().astype(float)
        sens = sens_fn(y_true, y_prob)
        ci = bootstrap_ci(y_true, y_prob, {"sensitivity": sens_fn})["sensitivity"]
        points.append({**base,
                       "sensitivity": None if sens is None else round(float(sens), 4),
                       "sensitivity_ci": ci,
                       "n_positive": int((y_true == 1).sum())})
    return points


def _risk_trajectory(preds: pl.DataFrame, label_col: str, block: str,
                     n_landmarks: int, window_unit: str | None) -> list[dict]:
    """Mean predicted risk by eventual outcome at each hospitalization time.

    Each stay is tagged with its eventual outcome (ever-positive). At time t
    the one-row-per-stay slice is split into positive- vs negative-outcome
    stays and their mean y_prob reported — the "predicted risk trajectory by
    outcome". An arm's mean is nulled when its count < MIN_CELL_COUNT; the
    whole landmark is suppressed when the slice itself is < MIN_CELL_COUNT.
    """
    w = preds.with_columns(pl.col(label_col).max().over(block).alias("_outcome"))
    w_parts = partition_windows(w)     # one pass, not one filter per landmark
    w_empty = w.clear()
    points: list[dict] = []
    for lm in range(n_landmarks):
        sl = w_parts.get(lm + 1, w_empty)
        base = {"hospitalization_time": lm + 1, "window_unit": window_unit,
                "n": sl.height}
        if sl.height < MIN_CELL_COUNT:
            points.append({**base, "n": f"<{MIN_CELL_COUNT}", "suppressed": True,
                           "mean_prob_positive": None, "mean_prob_negative": None})
            continue
        pos = sl.filter(pl.col("_outcome") > 0)["y_prob"]
        neg = sl.filter(pl.col("_outcome") == 0)["y_prob"]
        points.append({
            **base,
            "n_positive": suppress(pos.len()),
            "n_negative": suppress(neg.len()),
            "mean_prob_positive": (round(float(pos.mean()), 4)
                                   if pos.len() >= MIN_CELL_COUNT else None),
            "mean_prob_negative": (round(float(neg.mean()), 4)
                                   if neg.len() >= MIN_CELL_COUNT else None),
        })
    return points


def _median_first_alert(preds: pl.DataFrame, block: str, label_col: str,
                        t: float, direction: str) -> dict:
    """Median hospitalization time of the FIRST alert over caught positive stays.

    Forward view: among ever-positive stays, the first window_number whose
    prediction crosses the threshold (direction-aware, via the shared
    `_basis.action_expr`) — "how early in the stay does the model first
    fire". All of the stay's windows count, not only the labeled-positive ones.
    """
    per_stay = (
        preds.with_columns(pl.col(label_col).max().over(block).alias("_outcome"))
        .filter(pl.col("_outcome") > 0)
        .group_by(block)
        .agg(
            pl.col("window_number").filter(action_expr(t, direction))
            .min().alias("first_alert"),
        )
        # .sort is load-bearing: group_by's row order is nondeterministic, and the
        # seeded bootstrap resamples by row INDEX (see _ci's caller contract).
        .sort(block)
    )
    n_pos = per_stay.height
    caught = per_stay.filter(pl.col("first_alert").is_not_null())
    firsts = caught["first_alert"].to_numpy().astype(float)

    out: dict = {
        "n_positive_stays": n_pos,
        "n_caught": int(caught.height),
        "pct_caught": round(caught.height / n_pos, 4) if n_pos else None,
    }
    if len(firsts) < MIN_CELL_COUNT:
        out["median_first_alert_hospitalization_time"] = None
        out["median_first_alert_hospitalization_time_ci"] = None
        return out
    out["median_first_alert_hospitalization_time"] = round(float(np.median(firsts)), 2)
    out["median_first_alert_hospitalization_time_ci"] = _median_ci(firsts)
    return out


def _median_ci(values: np.ndarray) -> list[float] | None:
    """Percentile bootstrap CI for the median, resampling stays (shared engine)."""
    return scalar_ci(values)
