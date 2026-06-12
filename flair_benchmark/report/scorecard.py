"""Flat, cross-site headline scorecard → scorecard.json.

Every site emits this single flat dict with an identical schema per task. A
coordinating center concatenates scorecards across sites → one row per (site,
task) → tables + forest plots, without parsing the larger curve JSONs. It is
built purely by reading the already-computed report dicts (curves, threshold,
encounter) — no recomputation, so it can never disagree with them.

The encounter block is null for one-per-stay tasks (row == stay there); the
row-level discrimination block is always present.
"""

from __future__ import annotations


def build_scorecard(meta: dict, curves: dict, threshold: dict,
                    encounter: dict, site: str | None = None) -> dict:
    cal = curves.get("calibration_extended") or {}
    roc = curves.get("roc") or {}
    pr = curves.get("pr") or {}
    lm = (threshold or {}).get("label_metrics") or {}

    card: dict = {
        "task": meta["name"],
        "task_type": meta["task_type"],
        "prediction_unit": meta.get("prediction_unit"),
        "site": site,
        "n_test": (curves.get("metadata") or {}).get("n_test"),
        "prevalence": curves.get("prevalence"),
        "discrimination": {
            "auroc": roc.get("auroc"),
            "auroc_ci": roc.get("auroc_ci"),
            "auprc": pr.get("auprc"),
            "auprc_lift": pr.get("auprc_lift"),   # AUPRC / prevalence (rare-event safe)
        },
        "calibration": {
            "slope": cal.get("slope"),
            "intercept": cal.get("intercept"),
            "ece": (curves.get("calibration") or {}).get("ece"),
            "o_e_ratio": cal.get("o_e_ratio"),
            "brier": cal.get("brier"),
        },
        "row_operating_at_clinical_threshold": {
            "threshold": (threshold or {}).get("clinical_threshold"),
            "sensitivity": lm.get("sensitivity"),
            "specificity": lm.get("specificity"),
            "ppv": lm.get("ppv"),
            "npv": lm.get("npv"),
        },
        "encounter": _encounter_block(encounter),
        "metadata": {},
    }
    return card


def _encounter_block(encounter: dict) -> dict | None:
    """Headline stay-level operating metrics (continuous tasks only)."""
    at = (encounter or {}).get("at_clinical_threshold")
    if not at:
        return None  # one-per-stay task, or insufficient data → row metrics only
    lead = at.get("lead_time_hours") or {}
    return {
        "stay_sensitivity": at.get("stay_sensitivity"),
        "stay_specificity": at.get("stay_specificity"),
        "control_alarm_fraction": at.get("control_alarm_fraction"),
        "alarms_per_patient_day_silenced": at.get("alarms_per_patient_day_silenced"),
        "number_needed_to_alert": at.get("number_needed_to_alert"),
        "median_lead_hours": lead.get("median"),
        "n_event_stays": (encounter.get("counts") or {}).get("n_event_stays"),
    }
