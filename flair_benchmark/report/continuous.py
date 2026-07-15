"""Continuous report: ONE bundle for multi-row-per-stay (continuous) tasks.

Composes the two honest views of a continuous task (docs/eval_continuous_three_
ways.md) plus the timing analyses into a single run — there is no longer a
peak-vs-landmark mode choice:

  peak section     : stay-peak screening (Way 2) — discrimination (+NNE) and
                     fairness on the max-risk collapse. Calibration/DCA are
                     omitted here (peak-calibration trap) and live only in the
                     landmark section.
  landmark section : FORWARD landmarks (Way 3): skill/calibration/net-benefit/
                     fairness per hospitalization day/hour over
                     META['landmark_horizon'] times (30 days for tasks 1/2,
                     168 hours for task 4) — the time-varying / dynamic AUC
                     view; risk set = stays still at risk at that time.
  leadtime         : sensitivity curve, median first-alert hospitalization
                     time, and mean risk trajectory by eventual outcome.
  operating_points : the one decision-point section (operating_points.json) —
                     per-threshold (clinical ± 0.05) ALERT × LABEL confusion
                     matrix (MANDATORY) plus, when the task declares an
                     intervention/treatment column, the three per-threshold
                     sub-analyses. See report/operating_points.py.
  kpi              : headline scoreboard (kpi.json) — Heagerty-style pooled
                     dynamic AUC, peak NNE, calibration at hospitalization
                     time 1, lead-time and operating-point KPIs. See report/kpi.py.

All landmark-grain entries are indexed by `hospitalization_time` (1-based
window count since the stay's first prediction; unit = META['window_unit']) —
nothing anchors on the event or outcome. Output keeps per-pillar payloads
(discrimination / calibration / dca / fairness / leadtime, + operating_points +
kpi): the peak view is an additive `"peak"` sibling of `"landmarks"`.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report.kpi import build_kpi
from flair_benchmark.report.landmark import build_landmark_report, partition_windows
from flair_benchmark.report.leadtime import build_leadtime
from flair_benchmark.report.operating_points import build_operating_points
from flair_benchmark.report.peak import build_peak_report

_PEAK_OMITTED = ("peak calibration/DCA omitted: max-of-many risk is upward "
                 "biased (peak-calibration trap); see the landmark section")


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
        "dca": {"landmarks": lm["dca"],
                "metadata": {"peak_omitted_reason": _PEAK_OMITTED}},
        "fairness": {"peak": peak["fairness"], "landmarks": lm["fairness"]},
        "leadtime": leadtime,
        "operating_points": operating_points,
        "kpi": build_kpi(peak, lm, leadtime, operating_points,
                         preds=preds, meta=meta),
    }
