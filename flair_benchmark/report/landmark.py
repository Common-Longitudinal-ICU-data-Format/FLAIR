"""Landmark report: Way 3 of docs/eval_continuous_three_ways.md.

Anchor at each stay's LAST prediction window and count back: landmark ``L`` is
the slice ``window_number == W_last - L`` for ``L = 0..n_landmarks-1``. Each
slice has exactly one row per stay — the at-risk prediction that many windows
before the stay's endpoint — so it is an "episodic" cohort and runs through the
unchanged four-pillar `build_episodic_report`. Stacking the slices gives skill,
calibration, net benefit and fairness AS A FUNCTION OF LEAD TIME before the
event (van Houwelingen landmarking; the Kamran/PhysioNet early-warning view).

Why this is honest: one row per stay per landmark means report/_ci.py's row
bootstrap is again a patient-level bootstrap (unlike row-pooling all windows).

Anchoring notes:
  - End-of-stay outcomes (task1 mortality, task2 LTACH): W_last IS the event, so
    "L windows before the last window" = L windows before the outcome.
  - task4 sepsis: pred_stop = min(t_zero, ...), so a case's last window is the
    pre-onset / pre-treatment hour — the last-window anchor already lands on the
    Kamran onset-anchored, before-treatment view. Controls anchor to discharge.
  - For windowed tasks only ~`label horizon` landmarks carry positives; deeper
    ones have no events (AUROC undefined) and either return None metrics or hit
    the n<10 suppression below.

Safety: a landmark whose risk set has < MIN_LANDMARK_N patients has its ENTIRE
entry suppressed (not just per-cell), matching the house n<10 cell rule.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report.episodic import build_episodic_report
from flair_benchmark.report._suppress import MIN_CELL_COUNT

N_LANDMARKS = 12
MIN_LANDMARK_N = MIN_CELL_COUNT
_PILLARS = ("discrimination", "calibration", "dca", "fairness")


def build_landmark_report(preds: pl.DataFrame, meta: dict,
                          n_landmarks: int = N_LANDMARKS) -> dict[str, list[dict]]:
    """Per-landmark episodic bundle, stacked into one list per pillar."""
    block = ("hospitalization_join_id"
             if "hospitalization_join_id" in preds.columns
             else "hospitalization_id")
    window_unit = meta.get("window_unit")
    w = preds.with_columns(
        pl.col("window_number").max().over(block).alias("_w_last"))

    label_col = meta["label_column"]
    out: dict[str, list[dict]] = {p: [] for p in _PILLARS}
    for lm in range(n_landmarks):
        sl = w.filter(pl.col("window_number") == pl.col("_w_last") - lm)
        tag = {"landmark_index": lm, "windows_before_end": lm,
               "window_unit": window_unit, "n": sl.height}
        if sl.height < MIN_LANDMARK_N:
            stub = {**tag, "n": f"<{MIN_LANDMARK_N}", "suppressed": True}
            for p in _PILLARS:
                out[p].append(dict(stub))
            continue
        # A single-class risk set (common for deep landmarks of windowed tasks,
        # where nobody is within the label horizon yet) has no contrast: AUROC /
        # calibration / DCA are all undefined. Skip the builders (avoids wasted
        # work + sklearn single-class warnings) and emit a compact marker.
        if sl[label_col].n_unique() < 2:
            stub = {**tag, "undefined": True,
                    "reason": "single label class at this lead time"}
            for p in _PILLARS:
                out[p].append(dict(stub))
            continue
        rep = build_episodic_report(sl.drop("_w_last"), meta)
        for p in _PILLARS:
            payload = rep[p]
            payload.setdefault("metadata", {}).update(tag)
            out[p].append({**tag, **payload})
    return out
