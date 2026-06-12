"""Encounter-level (stay-level) operating metrics → encounter.json.

Row-level discrimination (curves.py / temporal.py) answers "does a higher score
mean higher risk?" but for a continuous / per-window model that fires every
hour (or day) it says nothing about real-world alarm burden or how early the
alert arrives. Each stay contributes many correlated rows, so row-level AUROC
overstates deployed utility and ignores alarm fatigue. This module reports the
clinically meaningful layer, one summary per encounter:

  counts                 stays / event stays / control stays
  at_clinical_threshold  one-alarm-per-stay sensitivity & specificity,
                         control-alarm fraction, alarms per patient-day
                         (raw + silenced), number-needed-to-alert, and the
                         lead-time distribution (first event vs first alarm)
  lead_time_hist         histogram of lead times — the TREWS Fig 2 analog
  operating_curve        stay-sensitivity vs alarm burden across thresholds —
                         the deployment trade-off that replaces row-ROC

References: JMIR 2026 (evaluation-strategy / alarm fatigue), Henry 2022 Nat Med
(TREWS lead-time figure), SepsisAI 2024 (false-alarms + silencing window).

Continuous (one-per-window) binary tasks only; everything else returns a
``metadata.reason`` stub, mirroring temporal.py. HIPAA suppression (n<10) is
applied to event-stay counts and lead-time histogram bins.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress

# operating-curve threshold grid (coarser than dca.py's 0.01 step — each point
# recomputes a per-stay reduction, and 49 points trace the trade-off cleanly).
_CURVE_GRID = [round(t, 2) for t in np.arange(0.02, 1.0, 0.02)]


def build_encounter(preds: pl.DataFrame, meta: dict) -> dict:
    """Stay-level alarm/lead-time metrics on the test split. Continuous binary only."""
    out: dict = {"level": "encounter", "metadata": {"min_cell_count": MIN_CELL_COUNT}}

    if meta["task_type"] != "binary" or "y_prob" not in preds.columns:
        out["metadata"]["reason"] = "requires binary task with y_prob"
        return out
    if meta.get("temporal_pattern") != "continuous":
        out["metadata"]["reason"] = "one_per_stay_task"
        return out
    if "prediction_dttm" not in preds.columns:
        out["metadata"]["reason"] = "requires prediction_dttm"
        return out

    label_col = meta["label_column"]
    thr = float(meta["clinical_threshold"])
    direction = meta.get("clinical_direction", "above")
    unit = meta.get("window_unit")
    stay_key = ("hospitalization_join_id" if "hospitalization_join_id" in preds.columns
                else "hospitalization_id")
    # default silencing window: one grid step (a day-grid alarm is "the same alarm"
    # for 24h; an hour-grid alarm for 6h, the TREWS post-alarm window).
    silence_h = float(meta.get("alarm_silence_hours", 24.0 if unit == "day" else 6.0))

    test = (
        preds.filter(pl.col("split") == "test")
        .select(stay_key, "prediction_dttm",
                pl.col(label_col).cast(pl.Int8).alias("y"),
                pl.col("y_prob").cast(pl.Float64))
        .drop_nulls(["prediction_dttm"])
    )
    if test.height == 0:
        out["metadata"]["reason"] = "no test rows with prediction_dttm"
        return out

    alarm = (pl.col("y_prob") >= thr) if direction == "above" else (pl.col("y_prob") <= thr)
    test = test.with_columns(alarm.alias("alarm"))

    # one row per stay
    g = test.group_by(stay_key).agg(
        pl.col("y").max().alias("is_event"),
        pl.col("y_prob").max().alias("peak_all"),
        pl.col("prediction_dttm").min().alias("t_min"),
        pl.col("prediction_dttm").max().alias("t_max"),
        pl.col("prediction_dttm").filter(pl.col("y") == 1).min().alias("first_event_dttm"),
        pl.col("prediction_dttm").filter(pl.col("alarm")).min().alias("first_alarm_dttm"),
        pl.col("alarm").sum().alias("n_alarm_raw"),
    )

    # silenced alarm count = distinct silence-windows (anchored at stay start) that
    # contain >=1 alarm — a deterministic approximation of "suppress repeats for W h".
    sil = (
        test.filter(pl.col("alarm"))
        .join(g.select(stay_key, "t_min"), on=stay_key)
        .with_columns(
            (((pl.col("prediction_dttm") - pl.col("t_min")).dt.total_seconds()
              / 3600.0 / silence_h).floor()).alias("bucket"))
        .group_by(stay_key).agg(pl.col("bucket").n_unique().alias("n_alarm_silenced"))
    )
    g = (
        g.join(sil, on=stay_key, how="left")
        .with_columns(pl.col("n_alarm_silenced").fill_null(0))
        .with_columns(
            ((pl.col("t_max") - pl.col("t_min")).dt.total_seconds() / 86400.0)
            .alias("monitored_days"),
            (pl.col("is_event") == 1).alias("event"),
            pl.col("first_alarm_dttm").is_not_null().alias("alarmed"),
        )
        .with_columns(
            (pl.col("event") & pl.col("alarmed")
             & (pl.col("first_alarm_dttm") <= pl.col("first_event_dttm"))).alias("caught"))
        .with_columns(
            pl.when(pl.col("caught"))
            .then((pl.col("first_event_dttm") - pl.col("first_alarm_dttm"))
                  .dt.total_seconds() / 3600.0)
            .otherwise(None).alias("lead_h"))
    )

    n_stays = g.height
    n_event = int(g["event"].sum())
    n_control = n_stays - n_event
    out["stay_key"] = stay_key
    out["time_unit"] = unit
    out["counts"] = {
        "n_stays": n_stays,
        "n_event_stays": suppress(n_event),
        "n_control_stays": n_control,
    }

    out["at_clinical_threshold"] = _at_threshold(g, n_event, n_control, thr,
                                                 direction, silence_h)
    out["lead_time_hist"] = _lead_hist(g, n_event, unit)
    out["operating_curve"] = _operating_curve(test, g, stay_key, n_event, n_control,
                                              direction, thr)
    out["metadata"].update({"stay_key": stay_key, "clinical_threshold": thr,
                            "clinical_direction": direction,
                            "silence_window_hours": silence_h})
    return out


def _at_threshold(g: pl.DataFrame, n_event: int, n_control: int, thr: float,
                  direction: str, silence_h: float) -> dict:
    """Operating metrics at the task's clinical_threshold."""
    total_days = float(g["monitored_days"].sum())
    n_alarmed = int(g["alarmed"].sum())
    block: dict = {
        "threshold": thr,
        "direction": direction,
        "silence_window_hours": silence_h,
        "alarms_per_patient_day_raw":
            _ratio(float(g["n_alarm_raw"].sum()), total_days),
        "alarms_per_patient_day_silenced":
            _ratio(float(g["n_alarm_silenced"].sum()), total_days),
    }

    control = g.filter(~pl.col("event"))
    n_ctrl_alarmed = int(control["alarmed"].sum()) if n_control else 0
    control_frac = _ratio(n_ctrl_alarmed, n_control)
    block["control_alarm_fraction"] = control_frac
    block["stay_specificity"] = (round(1.0 - control_frac, 4)
                                 if control_frac is not None else None)

    if n_event < MIN_CELL_COUNT:
        block["reason_event_metrics"] = "insufficient event stays"
        block["stay_sensitivity"] = None
        block["n_caught"] = suppress(int(g["caught"].sum()))
        return block

    n_caught = int(g["caught"].sum())
    block["stay_sensitivity"] = round(n_caught / n_event, 4)
    block["n_caught"] = suppress(n_caught)
    block["n_missed"] = suppress(n_event - n_caught)
    block["number_needed_to_alert"] = (round(n_alarmed / n_caught, 2)
                                       if n_caught else None)
    lead = g.filter(pl.col("caught"))["lead_h"].to_numpy()
    block["lead_time_hours"] = _quantiles(lead)
    return block


# Fixed clinical lead-time bins (hours) — SAME at every site so a coordinator can
# sum counts across sites to pool the distribution. Adaptive per-site bins would
# not align. Edges in hours; the open-ended last bin holds everything beyond 7 d.
_LEAD_EDGES = [0.0, 6.0, 12.0, 24.0, 48.0, 168.0, float("inf")]
_LEAD_LABELS = ["0-6h", "6-12h", "12-24h", "1-2d", "2-7d", ">7d"]


def _lead_hist(g: pl.DataFrame, n_event: int, unit: str | None) -> dict:
    """Caught lead times bucketed into FIXED clinical bins — the TREWS Fig 2 analog.

    Fixed, site-invariant bins make the distribution poolable across sites (sum the
    per-bin counts). Per-bin counts < 10 are PHI-suppressed.
    """
    if n_event < MIN_CELL_COUNT:
        return {"reason": "insufficient event stays"}
    lead = g.filter(pl.col("caught"))["lead_h"].to_numpy()
    lead = lead[~np.isnan(lead)]
    if lead.size == 0:
        return {"reason": "no caught event stays"}
    counts, _ = np.histogram(lead, bins=_LEAD_EDGES)
    return {
        "bins": "fixed_clinical",
        "labels": _LEAD_LABELS,
        "bin_edges_hours": _LEAD_EDGES[:-1] + ["inf"],
        "counts": [suppress(int(c)) for c in counts],
    }


def _operating_curve(test: pl.DataFrame, g: pl.DataFrame, stay_key: str,
                     n_event: int, n_control: int, direction: str,
                     thr: float) -> dict:
    """Stay-sensitivity vs alarm burden across thresholds (deployment trade-off)."""
    grid = _CURVE_GRID
    above = direction == "above"

    # alarm-row count per threshold from the sorted score vector (no per-t groupby).
    scores = np.sort(test["y_prob"].to_numpy())
    total_days = float(g["monitored_days"].sum())

    def n_alarm_rows(t: float) -> int:
        # above: rows with score >= t ; below: rows with score <= t
        return (len(scores) - int(np.searchsorted(scores, t, "left")) if above
                else int(np.searchsorted(scores, t, "right")))

    # control-alarm fraction per threshold from per-stay peak.
    ctrl_peak = g.filter(~pl.col("event"))["peak_all"].to_numpy() if n_control else np.empty(0)

    def control_frac(t: float) -> float | None:
        if not n_control:
            return None
        hit = (ctrl_peak >= t).sum() if above else (ctrl_peak <= t).sum()
        return round(float(hit) / n_control, 4)

    # sensitivity + median lead per threshold from the (small) pre-event rows only.
    sens_curve: list = []
    lead_curve: list = []
    if n_event >= MIN_CELL_COUNT:
        ev = g.filter(pl.col("event")).select(stay_key, "first_event_dttm")
        pre = (test.join(ev, on=stay_key)
               .filter(pl.col("prediction_dttm") <= pl.col("first_event_dttm"))
               .select(stay_key, "prediction_dttm", "y_prob", "first_event_dttm"))
        for t in grid:
            crossed = (pre.filter(pl.col("y_prob") >= t) if above
                       else pre.filter(pl.col("y_prob") <= t))
            fa = crossed.group_by(stay_key).agg(
                pl.col("prediction_dttm").min().alias("fa"),
                pl.col("first_event_dttm").first().alias("ev"))
            n_caught = fa.height
            sens_curve.append(round(n_caught / n_event, 4))
            if n_caught:
                lead = ((fa["ev"] - fa["fa"]).dt.total_seconds() / 3600.0).to_numpy()
                lead_curve.append(round(float(np.median(lead)), 2))
            else:
                lead_curve.append(None)
    else:
        sens_curve = [None] * len(grid)
        lead_curve = [None] * len(grid)

    return {
        "thresholds": grid,
        "clinical_threshold": thr,
        "stay_sensitivity": sens_curve,
        "alarms_per_patient_day": [_ratio(float(n_alarm_rows(t)), total_days) for t in grid],
        "control_alarm_fraction": [control_frac(t) for t in grid],
        "median_lead_time_hours": lead_curve,
    }


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _quantiles(arr: np.ndarray) -> dict:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {
        "median": round(float(np.median(arr)), 2),
        "q25": round(float(np.quantile(arr, 0.25)), 2),
        "q75": round(float(np.quantile(arr, 0.75)), 2),
        "n": suppress(int(arr.size)),
    }
