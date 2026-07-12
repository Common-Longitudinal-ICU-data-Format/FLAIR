"""KPI scoreboard for the continuous report -> kpi.json.

One compact block of headline numbers, aggregated from the ALREADY-BUILT
section dicts (peak / landmark / leadtime) — pure aggregation, no access to the
predictions, so the scoreboard can never disagree with the per-section JSONs.

  pooled_dynamic_auroc : Heagerty & Zheng (2005) style time-varying AUC — the
        per-landmark (incident/dynamic) AUCs pooled into one global concordance
        with Mann-Whitney pair weights. Within landmark t the identity
        concordant pairs = AUC(t) * cases_t * controls_t gives

            AUC_pooled = sum_t AUC(t) * w_t / sum_t w_t ,  w_t = cases_t * controls_t

        with cases_t = n_t * prevalence_t. The CI is the pair-weighted
        combination of the per-landmark bootstrap CIs — an interim method,
        pending a stay-level pooled bootstrap. (Same formula the FLAIR paper
        pipeline uses; single-sourced here so papers can read kpi.json.)
  peak         : stay-peak screening headline (AUROC/AUPRC, operating point,
                 NNE = 1/PPV — alarms per true case).
  calibration  : calibration at hospitalization time 1 (the first prediction
                 window) + the slope range across landmarks (drift).
  leadtime     : pct of event stays ever caught + median first-alert
                 hospitalization time (how early in the stay the model fires).
  alert_timing : fired-before-intervention rate, median delta hours, and the
                 never-treated false-alarm rate (only when the task declares an
                 intervention, e.g. task4 antibiotics).
"""

from __future__ import annotations

_METHOD_NOTE = ("pooled dynamic AUC per Heagerty & Zheng 2005 (incident/dynamic"
                " concordance, pair-weighted across FORWARD landmark risk sets"
                " — patients still at risk at each hospitalization day/hour);"
                " landmarking per van Houwelingen. Pooled CI is the"
                " pair-weighted combination of per-landmark bootstrap CIs"
                " (interim).")


def build_kpi(peak: dict[str, dict], landmarks: dict[str, list[dict]],
              leadtime: dict) -> dict:
    """Headline scoreboard from the built continuous sections."""
    disc_lms = landmarks.get("discrimination", [])
    out: dict = {"metadata": {
        "horizon": len(disc_lms),
        "window_unit": leadtime.get("metadata", {}).get("window_unit"),
        "method": _METHOD_NOTE,
    }}

    pooled = _pooled_dynamic_auroc(disc_lms)
    out["pooled_dynamic_auroc"] = pooled[0]
    out["pooled_dynamic_auroc_ci"] = pooled[1]
    out["peak"] = _peak_block(peak.get("discrimination", {}))
    out["calibration"] = _calibration_block(landmarks.get("calibration", []))
    out["leadtime"] = _leadtime_block(leadtime)
    out["alert_timing"] = _alert_block(leadtime.get("alert_vs_intervention"))
    return out


def _scored(entries: list[dict]) -> list[dict]:
    """Landmark entries that carry metrics (skip suppressed/undefined stubs)."""
    return [e for e in entries
            if not e.get("suppressed") and not e.get("undefined")
            and e.get("metrics")]


def _pooled_dynamic_auroc(entries: list[dict]) -> tuple[float | None,
                                                        list[float] | None]:
    num = den = lo_num = hi_num = 0.0
    for e in _scored(entries):
        auc = e["metrics"].get("auroc")
        prev = e.get("prevalence")
        n = e.get("n")
        if auc is None or prev is None or not isinstance(n, int):
            continue
        cases = n * prev
        w = cases * (n - cases)      # Mann-Whitney comparable pairs at t
        if w <= 0:
            continue
        ci = e["metrics"].get("auroc_ci") or [auc, auc]
        num += auc * w
        den += w
        lo_num += ci[0] * w
        hi_num += ci[1] * w
    if den == 0:
        return None, None
    return (round(num / den, 4),
            [round(lo_num / den, 4), round(hi_num / den, 4)])


def _peak_block(disc: dict) -> dict | None:
    m = disc.get("metrics")
    if not m:
        return None
    op = m.get("operating_point") or {}
    return {
        "auroc": m.get("auroc"), "auroc_ci": m.get("auroc_ci"),
        "auprc": m.get("auprc"), "auprc_ci": m.get("auprc_ci"),
        "sensitivity": op.get("sensitivity"),
        "specificity": op.get("specificity"),
        "ppv": op.get("ppv"),
        "flag_rate": op.get("flag_rate"),
        "nne": op.get("nne"), "nne_ci": op.get("nne_ci"),
    }


def _calibration_block(entries: list[dict]) -> dict | None:
    scored = _scored(entries)
    if not scored:
        return None
    nearest = min(scored, key=lambda e: e.get("hospitalization_time", 0))
    m = nearest["metrics"]
    slopes = [e["metrics"]["slope"] for e in scored
              if e["metrics"].get("slope") is not None]
    return {
        "at_hospitalization_time": nearest.get("hospitalization_time"),
        "slope": m.get("slope"), "slope_ci": m.get("slope_ci"),
        "intercept": m.get("intercept"), "intercept_ci": m.get("intercept_ci"),
        "ece": m.get("ece"),
        "brier": m.get("brier"), "brier_ci": m.get("brier_ci"),
        "o_e_ratio": m.get("o_e_ratio"),
        "slope_range": ([round(min(slopes), 4), round(max(slopes), 4)]
                        if slopes else None),
    }


def _leadtime_block(leadtime: dict) -> dict | None:
    m = leadtime.get("metrics")
    if not m:
        return None
    return {
        "pct_caught": m.get("pct_caught"),
        "median_first_alert_hospitalization_time":
            m.get("median_first_alert_hospitalization_time"),
        "median_first_alert_hospitalization_time_ci":
            m.get("median_first_alert_hospitalization_time_ci"),
    }


def _alert_block(avi: dict | None) -> dict | None:
    if not avi or not (avi.get("metadata") or {}).get("intervention"):
        return None
    w = (avi.get("label_positive") or {}).get("with_intervention") or {}
    without = avi.get("label_negative_without_intervention") or {}
    return {
        "fired_before_intervention_rate":
            w.get("fired_before_intervention_rate"),
        "fired_before_intervention_rate_ci":
            w.get("fired_before_intervention_rate_ci"),
        "median_delta_hours": (w.get("delta_hours") or {}).get("median"),
        "false_alarm_rate_untreated": without.get("false_alarm_rate"),
        "false_alarm_rate_untreated_ci": without.get("false_alarm_rate_ci"),
    }
