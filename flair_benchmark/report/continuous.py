"""Continuous report — ONE bundle for multi-row-per-stay tasks.

A continuous task predicts repeatedly during a stay, so there is no single honest
way to score it. This module composes the two honest views (see
docs/eval_continuous_three_ways.md) plus the timing analyses into a single run;
there is no peak-vs-landmark mode choice to make.

  Part 1  build_continuous_report()  the composition, and what each piece is for
  Part 2  build_peak_report()        the stay-peak view (Way 2)
  Part 3  build_leadtime()           how early the model first fires

Peak and leadtime live here rather than in files of their own because
`build_continuous_report` is their only caller — they are sections of this
bundle, not independent reports. The landmark view (Way 3) and the KPI
scoreboard are big enough to keep their own files.

The sections of the emitted bundle:

  peak section     : stay-peak screening (Way 2) — discrimination (+NNE) and
                     fairness on the max-risk collapse. Calibration is omitted
                     here (the peak-calibration trap) and lives only in the
                     landmark section. DCA is not reported in continuous mode at
                     all: the same trap makes net benefit on max-of-many risk
                     misleading, so it stays an episodic-only view.
  landmark section : FORWARD landmarks (Way 3): skill/calibration/fairness per
                     hospitalization day or hour over META['landmark_horizon']
                     times — the time-varying / dynamic AUC view. The risk set is
                     every stay still at risk at that time. See report/landmark.py.
  leadtime         : median first-alert hospitalization time (how early in the
                     stay the model first fires) — a single number, no day-level
                     curve or trajectory.
  operating_points : the one decision-point section — a fixed 99-point universal
                     sweep (0.01..0.99) on the PEAK surface with the clinical
                     policy flagged, per-threshold ALERT x LABEL confusion +
                     metrics, encounter-level alert burden, data-driven locators,
                     an operating-budget view, and — when the task declares an
                     intervention column — per-threshold treatment sub-rates plus
                     a named-threshold median lead-time. See
                     report/operating_points.py.
  kpi              : headline scoreboard — Heagerty-style pooled dynamic AUC,
                     peak NNE, calibration at hospitalization time 1, lead-time
                     and operating-point KPIs. See report/kpi.py.

All landmark-grain entries are indexed by `hospitalization_time` (1-based window
count since the stay's first prediction; unit = META['window_unit']). Nothing
anchors on the event or the outcome. The peak view is an additive "peak" sibling
of "landmarks" inside each pillar's payload.

Depends on: report/discrimination.py, fairness.py, landmark.py, kpi.py,
            operating_points.py, _basis.py, _stats.py.
Used by: report/__init__.py when mode == "continuous".
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._basis import action_expr
from flair_benchmark.report._stats import MIN_CELL_COUNT, nne_from_ppv, scalar_ci
from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.fairness import build_fairness
from flair_benchmark.report.kpi import build_kpi
from flair_benchmark.report.landmark import build_landmark_report, partition_windows
from flair_benchmark.report.operating_points import build_operating_points

_PEAK_OMITTED = ("peak calibration omitted: max-of-many risk is upward biased "
                 "(peak-calibration trap); see the landmark section")


# ===========================================================================
# Part 1 — Composition
# ===========================================================================


def build_continuous_report(preds: pl.DataFrame, meta: dict,
                            n_landmarks: int | None = None) -> dict[str, dict]:
    """Peak + landmark + leadtime + operating_points in one payload dict."""
    peak = build_peak_report(preds, meta)
    parts = partition_windows(preds)   # ONE pass shared by landmark + leadtime
    lm = build_landmark_report(preds, meta, n_landmarks, parts=parts)
    leadtime = build_leadtime(preds, meta, n_landmarks, parts=parts)
    operating_points = build_operating_points(preds, meta)
    return {
        "discrimination": {"peak": peak["discrimination"],
                           "landmarks": lm["discrimination"]},
        "calibration": {"landmarks": lm["calibration"],
                        "metadata": {"peak_omitted_reason": _PEAK_OMITTED}},
        "fairness": {"peak": peak["fairness"], "landmarks": lm["fairness"]},
        "leadtime": leadtime,
        "operating_points": operating_points,
        "kpi": build_kpi(peak, lm, leadtime, operating_points,
                         preds=preds, meta=meta),
    }


# ===========================================================================
# Part 2 — Peak section (Way 2 of docs/eval_continuous_three_ways.md)
#
# Collapse the task to ONE row per stay using the peak predicted risk over the
# stay — y_prob = max, label = "ever positive" (also max) — then score it exactly
# like an episodic report. This answers the stay-level screening question: "did
# this stay ever alarm vs ever have the event?", the number sepsis screening
# papers (Wong 2021) headline.
#
# Only two pillars are emitted:
#   discrimination : ranking + operating point (+ NNE, number-needed-to-evaluate)
#   fairness       : subgroup ranking + parity
#
# Calibration and DCA are DELIBERATELY omitted: max-of-many is upward biased (the
# "peak-calibration trap", §5 of the doc), so a reliability curve or net-benefit
# computed on peak risk is misleading. Honest calibration lives in the landmark
# section instead.
#
# After the collapse there is one row per stay, so the row bootstrap in
# report/_stats.py is again an honest patient-level bootstrap.
# ===========================================================================

# Demographic / context columns carried through the collapse so fairness still
# has something to subgroup on. Taken as the stay's first value — constant per
# stay for demographics.
_CARRY_FIRST = ["age_at_admission", "sex_category", "race_category",
                "ethnicity_category"]


def build_peak_report(preds: pl.DataFrame, meta: dict) -> dict[str, dict]:
    """Collapse to stay-peak risk, then emit discrimination + fairness only."""
    stay = _collapse_to_peak(preds, meta)

    discrimination = build_discrimination(stay, meta)
    _inject_nne(discrimination)
    fairness = build_fairness(stay, meta)

    note = {"section": "peak", "n_stays": stay.height,
            "peak_calibration_omitted": True}
    for payload in (discrimination, fairness):
        payload.setdefault("metadata", {}).update(note)
    return {"discrimination": discrimination, "fairness": fairness}


def _collapse_to_peak(preds: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """One row per stay: peak y_prob, ever-positive label, first demographics."""
    block = "hospitalization_join_id"
    label_col = meta["label_column"]
    aggs = [pl.col("y_prob").max().alias("y_prob"),
            pl.col(label_col).max().alias(label_col),
            pl.len().alias("n_windows")]
    aggs += [pl.col(c).first().alias(c)
             for c in _CARRY_FIRST if c in preds.columns]
    # .sort is load-bearing: group_by's row order is nondeterministic, and the
    # seeded bootstrap resamples by row INDEX (see _stats' caller contract).
    return preds.group_by(block).agg(aggs).sort(block)


def _inject_nne(discrimination: dict) -> None:
    """Add number-needed-to-evaluate (1/PPV) at the operating point, in place.

    NNE is the headline operating number for rare-event screeners (Wong 2021's
    "NNE ~= 109" for the Epic sepsis model): how many flagged stays per true
    positive. Derived from the PPV already computed by build_discrimination, so
    its CI is the inverted PPV CI (and the bounds swap order under 1/x).
    """
    op = discrimination.get("metrics", {}).get("operating_point")
    if not op:
        return
    op["nne"], op["nne_ci"] = nne_from_ppv(op.get("ppv"), op.get("ppv_ci"))


# ===========================================================================
# Part 3 — Lead time
#
# One early-warning number on the FORWARD landmark grid. Nothing anchors on the
# event or the outcome:
#
#   median_first_alert_hospitalization_time : per caught ever-positive stay, the
#     hospitalization time (window_number) of the model's FIRST alert on the
#     ACTION basis (direction-aware — for a "below" task the model "fires" when
#     risk drops under t). Reported as a median with a stay-bootstrap CI, plus a
#     percent-of-positives-caught.
#
# The per-hospitalization-time sensitivity CURVE and the mean-risk TRAJECTORY
# (both day/hour-level breakdowns) were removed: the operating-point analysis is
# encounter-level only, so no day-granular breakdown is published (design doc §9,
# report/operating_points.py).
# ===========================================================================


def build_leadtime(preds: pl.DataFrame, meta: dict,
                   n_landmarks: int | None = None,
                   parts: dict[int, pl.DataFrame] | None = None) -> dict:
    """Median first-alert hospitalization time (+ CI, % caught).

    `n_landmarks` / `parts` are accepted for a stable call signature with the
    landmark report but are no longer used — the day-level curve and trajectory
    that consumed them are gone.
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
    `_basis.action_expr`) — "how early in the stay does the model first fire".
    All of the stay's windows count, not only the labeled-positive ones.
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
        # seeded bootstrap resamples by row INDEX (see _stats' caller contract).
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
