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
  operating_points : the clinical-threshold confusion headline (sensitivity /
                 specificity / FPR / PPV / NNE) + FPR across the three thresholds,
                 plus the intervention sub-analysis highlights (positive fire
                 rate, median hours-before-treatment, treated-negative median Δ,
                 untreated misfire rate) when the task declares an intervention.
"""

from __future__ import annotations

_METHOD_NOTE = ("pooled dynamic AUC per Heagerty & Zheng 2005 (incident/dynamic"
                " concordance, pair-weighted across FORWARD landmark risk sets"
                " — patients still at risk at each hospitalization day/hour);"
                " landmarking per van Houwelingen. Pooled CI is the"
                " pair-weighted combination of per-landmark bootstrap CIs"
                " (interim).")


def build_kpi(peak: dict[str, dict], landmarks: dict[str, list[dict]],
              leadtime: dict, operating_points: dict | None = None) -> dict:
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
    out["operating_points"] = _operating_points_block(operating_points or {})
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


def _operating_points_block(op: dict) -> dict | None:
    """Clinical-threshold confusion headline + FPR sweep + sub-analysis highlights.

    Reads the built operating_points section. `sub_analyses` is None for tasks
    without an intervention/treatment column.
    """
    by_thr = op.get("by_threshold") or []
    if not by_thr:
        return None
    t0 = (op.get("metadata") or {}).get("clinical_threshold")
    # The clinical row is flagged, never positional: a ± offset outside (0, 1) is
    # dropped from the grid (task1 at 0.95 loses its +0.05 arm), so "the middle row"
    # stops meaning "the clinical row". The 0.0 offset itself can never be dropped —
    # tasks._validate_taxonomy guarantees 0 < clinical_threshold < 1.
    clinical = next((e for e in by_thr if e.get("is_clinical")),
                    next((e for e in by_thr
                          if t0 is not None and e.get("threshold") == round(t0, 4)),
                         by_thr[len(by_thr) // 2]))
    conf = clinical.get("confusion") or {}
    block = {
        "clinical_threshold": t0,
        "sensitivity": conf.get("sensitivity"),
        "specificity": conf.get("specificity"),
        "fpr": conf.get("fpr"),
        "ppv": conf.get("ppv"),
        "nne": conf.get("nne"), "nne_ci": conf.get("nne_ci"),
        "fpr_by_threshold": [{"threshold": e.get("threshold"),
                              "fpr": (e.get("confusion") or {}).get("fpr")}
                             for e in by_thr],
        "sub_analyses": None,
    }
    sub = clinical.get("sub_analyses")
    if sub:
        pos = sub.get("label_positive") or {}
        negt = sub.get("label_negative_treated") or {}
        negu = sub.get("label_negative_untreated") or {}
        block["sub_analyses"] = {
            "pct_positive_fired": pos.get("pct_fired"),
            "median_hours_before_treatment_positive":
                (pos.get("hours_before_treatment") or {}).get("median"),
            "median_hours_before_treatment_treated_negative":
                (negt.get("hours_before_treatment") or {}).get("median"),
            "misfire_rate_untreated": negu.get("misfire_rate"),
        }
    return block
