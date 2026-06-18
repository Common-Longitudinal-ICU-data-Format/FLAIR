"""Lead-time extraction → leadtime.json (companion to the landmark report).

Two early-warning views, both anchored on each stay's last window (= the event
anchor; see landmark.py):

  curve  : per-landmark detection sensitivity. At landmark L (= L windows before
           the end), among at-risk rows that are positive, the fraction already
           flagged (y_prob >= clinical_threshold), each with a 95% CI. This is
           the "how much of the signal is visible L windows out" curve.
  median_lead : per caught positive stay, how many windows before its endpoint
           the model FIRST fired; reported as a median (+ stay-bootstrap CI) and
           a percent-of-positives-caught.

Sensitivity reuses discrimination._mk_stat + _ci.bootstrap_ci so the CI method
matches the rest of the report. The median lead-time CI is a stay-level
percentile bootstrap (resample caught stays).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._ci import BOOT_REPS, BOOT_SEED, bootstrap_ci
from flair_benchmark.report._suppress import MIN_CELL_COUNT
from flair_benchmark.report.discrimination import _mk_stat
from flair_benchmark.report.landmark import N_LANDMARKS


def build_leadtime(preds: pl.DataFrame, meta: dict,
                   n_landmarks: int = N_LANDMARKS) -> dict:
    """Per-landmark sensitivity curve + median lead-time, with CIs."""
    out: dict = {"task_type": meta["task_type"], "metadata": {}}
    t = meta.get("clinical_threshold")
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or t is None:
        out["metadata"]["reason"] = (
            "lead-time requires a binary task with y_prob and a clinical_threshold")
        return out

    block = ("hospitalization_join_id"
             if "hospitalization_join_id" in preds.columns
             else "hospitalization_id")
    label_col = meta["label_column"]
    window_unit = meta.get("window_unit")
    w = preds.with_columns(
        pl.col("window_number").max().over(block).alias("_w_last"))

    curve = _sensitivity_curve(w, label_col, t, n_landmarks, window_unit)
    lead = _median_lead(w, block, label_col, t)

    out["prevalence"] = round(float((preds[label_col] > 0).mean()), 6)
    out["curve"] = curve
    out["metrics"] = lead
    out["metadata"] = {"clinical_threshold": t, "window_unit": window_unit,
                       "min_cell_count": MIN_CELL_COUNT}
    return out


def _sensitivity_curve(w: pl.DataFrame, label_col: str, t: float,
                       n_landmarks: int, window_unit: str | None) -> list[dict]:
    """Detection sensitivity (+ 95% CI) at each landmark, suppressing n<10."""
    sens_fn = _mk_stat(t, "sensitivity")
    points: list[dict] = []
    for lm in range(n_landmarks):
        sl = w.filter(pl.col("window_number") == pl.col("_w_last") - lm)
        base = {"windows_before_end": lm, "window_unit": window_unit, "n": sl.height}
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


def _median_lead(w: pl.DataFrame, block: str, label_col: str, t: float) -> dict:
    """Median windows of warning over caught positive stays (+ stay bootstrap CI)."""
    # positive stays = ever-positive; first alert = earliest window at/above t;
    # lead = windows between first alert and the stay's last window.
    per_stay = (
        w.filter(pl.col(label_col) > 0)
        .group_by(block)
        .agg(
            pl.col("_w_last").first().alias("w_last"),
            pl.col("window_number").filter(pl.col("y_prob") >= t)
            .min().alias("first_alert"),
        )
    )
    n_pos = per_stay.height
    caught = per_stay.filter(pl.col("first_alert").is_not_null())
    leads = (caught.select((pl.col("w_last") - pl.col("first_alert")))
             .to_series().to_numpy().astype(float))

    out: dict = {
        "n_positive_stays": n_pos,
        "n_caught": int(caught.height),
        "pct_caught": round(caught.height / n_pos, 4) if n_pos else None,
    }
    if len(leads) < MIN_CELL_COUNT:
        out["median_lead_windows"] = None
        out["median_lead_ci"] = None
        return out
    out["median_lead_windows"] = round(float(np.median(leads)), 2)
    out["median_lead_ci"] = _median_ci(leads)
    return out


def _median_ci(leads: np.ndarray) -> list[float] | None:
    """Percentile bootstrap CI for the median, resampling stays."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(leads)
    meds = [float(np.median(leads[rng.integers(0, n, n)])) for _ in range(BOOT_REPS)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]
