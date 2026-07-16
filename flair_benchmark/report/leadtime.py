"""Lead-time extraction → leadtime.json (companion to the landmark report).

One early-warning number on the FORWARD landmark grid (nothing anchors on the
event or outcome):

  median_first_alert_hospitalization_time : per caught ever-positive stay, the
           hospitalization time (window_number) of the model's FIRST alert on
           the ACTION basis (direction-aware — for a "below" task the model
           "fires" when risk drops under t) — how early in the stay the model
           first fires; reported as a median (+ stay-bootstrap CI) and a
           percent-of-positives-caught.

The per-hospitalization-time sensitivity CURVE and the mean-risk TRAJECTORY (both
day/hour-level breakdowns) were removed: the operating-point analysis is
encounter-level only, so no day-granular breakdown is published (design doc §9;
report/operating_points.py). The median first-alert CI is a stay-level percentile
bootstrap (resample caught stays).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._basis import action_expr
from flair_benchmark.report._ci import scalar_ci
from flair_benchmark.report._suppress import MIN_CELL_COUNT


def build_leadtime(preds: pl.DataFrame, meta: dict,
                   n_landmarks: int | None = None,
                   parts: dict[int, pl.DataFrame] | None = None) -> dict:
    """Median first-alert hospitalization time (+ CI, % caught).

    ``n_landmarks`` / ``parts`` are accepted for a stable call signature with the
    landmark report but no longer used — the day-level curve and trajectory that
    consumed them are gone.
    """
    out: dict = {"task_type": meta["task_type"], "metadata": {}}
    t = meta.get("clinical_threshold")
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or t is None:
        out["metadata"]["reason"] = (
            "lead-time requires a binary task with y_prob and a clinical_threshold")
        return out

    block = "hospitalization_join_id"
    label_col = meta["label_column"]
    window_unit = meta.get("window_unit")
    direction = meta.get("clinical_direction", "above")

    lead = _median_first_alert(preds, block, label_col, t, direction)

    out["prevalence"] = round(float((preds[label_col] > 0).mean()), 6)
    out["metrics"] = lead
    out["metadata"] = {"clinical_threshold": t, "window_unit": window_unit,
                       "min_cell_count": MIN_CELL_COUNT,
                       "first_alert_basis": "action (direction-aware)"}
    return out


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
