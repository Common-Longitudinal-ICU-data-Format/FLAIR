"""Landmark report: Way 3 of docs/eval_continuous_three_ways.md.

FORWARD landmarking (van Houwelingen; the time-dependent-AUC idiom of the
US-CRS eFigure-2): landmark ``t`` is the slice ``window_number == t`` for
``t = 1..n_landmarks`` — **hospitalization day t** for the daily tasks,
**hospitalization hour t** for hourly task 4 (unit from META's
``window_unit``). ``n_landmarks`` comes from ``META['landmark_horizon']``
(30 daily landmarks for tasks 1/2, 168 hourly for task 4).

The risk set at landmark t is every stay still producing predictions at t.
Task builders only generate windows inside ``[pred_start, pred_stop)`` (stop =
event onset / discharge / death), so each slice is automatically restricted to
stays still at risk and event-free — the per-landmark ``n`` is a true
Kaplan-Meier "No. at risk" count that falls as events occur and stays end.
Nothing anchors on the event or outcome: the slice at day t is exactly the
cohort a deployed model would score on day t.

Each slice has (at most) one row per stay — the day-t prediction — so it is an
"episodic" cohort and runs through the unchanged four-pillar
`build_episodic_report`. Stacking the slices gives skill, calibration, net
benefit and fairness AS A FUNCTION OF HOSPITALIZATION TIME (the time-varying /
dynamic AUC view).

Why this is honest: one row per stay per landmark means report/_ci.py's row
bootstrap is again a patient-level bootstrap (unlike row-pooling all windows).

Notes:
  - Early landmarks of windowed tasks (e.g. task4's 6h label) can be
    single-class when no event falls within the horizon yet; those slices are
    undefined (no contrast) and emit a compact marker.
  - Deep landmarks shrink as stays end; a risk set below MIN_LANDMARK_N has its
    ENTIRE entry suppressed (not just per-cell), matching the house n<10 rule.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report.episodic import build_episodic_report
from flair_benchmark.report._suppress import MIN_CELL_COUNT

MIN_LANDMARK_N = MIN_CELL_COUNT
_PILLARS = ("discrimination", "calibration", "dca", "fairness")


def resolve_n_landmarks(meta: dict, n_landmarks: int | None) -> int:
    """Explicit override wins; otherwise the task's META['landmark_horizon'].

    The horizon is a landmark COUNT in the task's window_unit (30 days for the
    daily tasks, 168 hours for hourly sepsis) — it lives in TASK_POLICY /
    META, never hardcoded here.
    """
    if n_landmarks is not None:
        return n_landmarks
    h = meta.get("landmark_horizon")
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError(
            "META['landmark_horizon'] (positive int, landmarks in the task's "
            "window_unit) is required for the continuous report; got "
            f"{h!r} for task {meta.get('name')!r}.")
    return h


def build_landmark_report(preds: pl.DataFrame, meta: dict,
                          n_landmarks: int | None = None) -> dict[str, list[dict]]:
    """Per-landmark episodic bundle, stacked into one list per pillar."""
    n_landmarks = resolve_n_landmarks(meta, n_landmarks)
    window_unit = meta.get("window_unit")

    label_col = meta["label_column"]
    out: dict[str, list[dict]] = {p: [] for p in _PILLARS}
    for lm in range(n_landmarks):
        # forward slice: everyone still at risk at hospitalization time lm+1
        # (windows only exist pre-event/pre-discharge, so this is the honest
        # deployed-model risk set — never anchored on the outcome).
        sl = preds.filter(pl.col("window_number") == lm + 1)
        tag = {"landmark_index": lm, "hospitalization_time": lm + 1,
               "window_unit": window_unit, "n": sl.height}
        if sl.height < MIN_LANDMARK_N:
            stub = {**tag, "n": f"<{MIN_LANDMARK_N}", "suppressed": True}
            for p in _PILLARS:
                out[p].append(dict(stub))
            continue
        # A single-class risk set (e.g. early hours of a windowed task where no
        # event falls inside the label horizon yet) has no contrast: AUROC /
        # calibration / DCA are all undefined. Skip the builders (avoids wasted
        # work + sklearn single-class warnings) and emit a compact marker.
        if sl[label_col].n_unique() < 2:
            stub = {**tag, "undefined": True,
                    "reason": "single label class at this hospitalization time"}
            for p in _PILLARS:
                out[p].append(dict(stub))
            continue
        rep = build_episodic_report(sl, meta)
        for p in _PILLARS:
            payload = rep[p]
            payload.setdefault("metadata", {}).update(tag)
            out[p].append({**tag, **payload})
    return out
