"""Decision Curve Analysis → dca.json (binary tasks).

The frame is split-filtered upstream (episodic: one row per stay).

Brute-force sweep over threshold probabilities t:

  nb_model = TP/n - FP/n * t/(1-t)
  nb_all   = prevalence - (1-prevalence) * t/(1-t)   (treat everyone)
  nb_none  = 0                                        (treat no one)

THE DECISION BASIS. A DCA describes the ACT, not the alarm — so the curve is
always built on the frame the clinician is actually deciding in. For a
``clinical_direction: "above"`` task that frame IS the label: the event is the
positive class, the score is p, and the clinical threshold t sits directly on the
x-axis — e.g. icu_readmission "keep in ICU when readmission risk >= 0.20" puts the
policy at x = 0.20 and `prevalence` = the readmission rate.

For a ``clinical_direction: "below"`` task the frame is the COMPLEMENT of the
label. extubation_failure_24h is the live one: it predicts extubation FAILURE but
its decision is whether to EXTUBATE, which fires on LOW risk, so the DCA's event
is "successful extubation" (= the label's complement) and the x-axis is the
probability of success a clinician would require before extubating:

    x = 0    not worried about failure at all, worried about the harms of
             continued intubation  →  extubate everyone
    x = 1    terrified of extubation failure  →  keep everyone intubated

The clinical threshold t (on the FAILURE scale) then sits at 1-t on this axis (the
shipped 0.10 → x = 0.90, "extubate when P(success) >= 90%"), and because the whole
frame is complemented `prevalence` is P(success), NOT the failure rate —
`metadata.prevalence_of` says which out loud. Both defaults are named too:
`dca_axis.inaction` supplies the treat-none arm, so a two-sided decision reads as
"keep the patient intubated" rather than the double negative "'extubate' on no one".

``to_decision_basis`` / ``axis_threshold`` / ``label_threshold`` are the single
source of that mapping; report/operating_points.py calls them too, so the
scorecard's net benefit and this curve are the same quantity and cannot drift
apart. Note which half complements: the LABELS do, the SCORES do not. The acted
arm is cut on the label scale by ``action_mask`` in both files — comparing ``1-p``
is not exact at a tie (report/_basis.py's boundary convention).

`useful_range` is the widest contiguous threshold interval where the model has
strictly higher net benefit than both default strategies; `all_useful_intervals`
lists every such interval. `metrics.verdict` reports the model against both
defaults AT the shipped policy — the DCA's own conclusion, kept with its own curve.

The net benefit at the clinical threshold carries a 95% bootstrap CI
(report/_ci.py). The full-curve CI band is left out by cost (one resample per
threshold × grid); add later if a band is needed.

Suppression: net benefit at a threshold flagging < 10 patients positive is
derived from too few patients — nulled and listed in suppressed_thresholds.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._basis import (
    action_mask,
    axis_threshold,
    label_threshold,
    net_benefit,
    net_benefit_all,
    to_decision_basis,
)
from flair_benchmark.report._ci import bootstrap_ci
from flair_benchmark.report._suppress import MIN_CELL_COUNT

__all__ = ["build_dca", "axis_threshold", "net_benefit_all", "to_decision_basis"]

# Uniform 0.01..0.99 body, plus a dense tail at BOTH ends (step .002).
#   low  (0.001..0.05): the rare-event decision region, where an "above" task with
#                       0.15% prevalence actually operates — otherwise it collapses
#                       into one suppressed point near 0.
#   high (0.90..0.999): the mirror image, which is where a "below" task's decision
#                       lives once the frame is complemented (extubation_failure_24h
#                       ships at 0.90 on this axis). At step .01 the useful-range
#                       boundary would snap to a gridline.
_LOW = np.round(np.arange(0.001, 0.05, 0.002), 3)
_HIGH = np.round(np.arange(0.90, 1.0, 0.002), 3)
THRESHOLDS = np.unique(np.concatenate([
    _LOW, np.round(np.arange(0.01, 1.00, 0.01), 2), _HIGH]))
_TOL = 1e-9


def build_dca(preds: pl.DataFrame, meta: dict) -> dict:
    """Net-benefit curve on the decision basis, + the verdict at the shipped policy."""
    out: dict = {"task_type": meta["task_type"], "metadata": {}}
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns:
        out["metadata"]["reason"] = "requires binary task with y_prob"
        return out

    y_true = preds[meta["label_column"]].to_numpy().astype(int)
    y_prob = preds["y_prob"].to_numpy().astype(float)
    n = len(y_true)
    if n == 0:
        out["metadata"]["reason"] = "empty split"
        return out

    # Put the frame in the basis the clinician decides in (see module docstring):
    # for an act-when-LOW task the event is the label's complement and the score is
    # P(that event), so the curve describes the act (extubate) rather than reading
    # backwards. Shared with operating_points.py so the two cannot drift.
    direction = meta.get("clinical_direction", "above")
    below = direction == "below"
    t_clin = meta.get("clinical_threshold")
    # Labels move to the decision basis; the SCORES stay on the label scale, because
    # that is where the acted arm is cut (report/_basis.py boundary convention).
    y_true, _ = to_decision_basis(y_true, y_prob, direction)
    t_eval = None if t_clin is None else axis_threshold(t_clin, direction)
    prev = float(y_true.mean())

    nb_model: list[float | None] = []
    nb_all: list[float] = []
    suppressed: list[float] = []
    useful_mask: list[bool] = []
    for t_np in THRESHOLDS:
        # python float, not np.float64 — see net_benefit_all() on why the dtype
        # decides the 6th decimal, and why operating_points.py must agree with it.
        t = float(t_np)
        all_nb = net_benefit_all(prev, t)
        nb_all.append(all_nb)
        # Acted arm cut on the LABEL scale — literally the shipped action
        # (p_fail < t for a "below" task) — while t weighs the odds on the
        # decision axis. Same two-threshold split as _basis.net_benefit, which is
        # what makes the curve and the scorecard agree even at p == t.
        t_lab = label_threshold(t, direction)
        acted = action_mask(y_prob, t_lab, direction)
        n_flagged = int(acted.sum())
        if n_flagged < MIN_CELL_COUNT:
            nb_model.append(None)
            suppressed.append(t)
            useful_mask.append(False)
            continue
        model_nb = net_benefit(y_true, y_prob, t_lab, t, direction)
        nb_model.append(round(float(model_nb), 6))
        useful_mask.append(model_nb > max(all_nb, 0.0) + _TOL)

    intervals = _contiguous_intervals(THRESHOLDS, useful_mask)
    axis = _axis_framing(meta, direction, t_clin, t_eval)
    clinical = _clinical_marker(t_eval, t_clin, direction, nb_model, nb_all,
                                intervals, y_true, y_prob,
                                axis["action"], axis["inaction"])
    focus = _focus_range(prev, t_eval or 0.0, below)
    useful_range = (max(intervals, key=lambda iv: iv["high"] - iv["low"])
                    if intervals else None)
    return {
        "task_type": "binary",
        "direction": direction,
        "basis": ("complement: predict non-event (1-p), act when y_prob < "
                  "clinical_threshold" if below else "standard: act when y_prob >= "
                  "clinical_threshold"),
        "clinical_threshold_original": t_clin,
        # NB: on a "below" task this is P(the complement event) — for
        # extubation_failure_24h, P(successful extubation) ~ 0.95, NOT the ~5%
        # failure rate. `metadata.prevalence_of` names it so that cannot be
        # misread as a bug.
        "prevalence": round(prev, 4),
        # scalars (the verdict at the shipped policy, useful range) first, curves below.
        "metrics": {
            **clinical,
            "useful_range": useful_range,
            "all_useful_intervals": [[iv["low"], iv["high"]] for iv in intervals],
        },
        "curves": {
            "thresholds": [float(t) for t in THRESHOLDS],
            "net_benefit_model": nb_model,
            "net_benefit_all": nb_all,
            "net_benefit_none": 0.0,
            "focus_range": focus,
            "suppressed_thresholds": suppressed,
        },
        "metadata": {
            "grid": "0.001-0.05 step .002 + 0.01-0.99 step .01 + 0.90-0.999 step .002",
            "min_flagged_n": MIN_CELL_COUNT, "n_test": n, **axis,
        },
    }


def _focus_range(prev: float, clin: float, below: bool) -> list[float]:
    """Where to zoom the plot. The full grid always stays in the JSON.

    "Above": rare-event decisions live near 0 — treating at ~10x prevalence already
    implies a high alarm rate — so the window is [0, max(0.05, 10*prev, clin+0.05)],
    always wide enough to contain the shipped policy.

    "Below" is the SAME rule seen through the complement, so it is written as a
    literal mirror rather than restated: on that basis the rare event is 1-prev and
    the decisions live up near 1, which is what dca's dense _HIGH grid is for.
    Restating the formula with the signs flipped would let the two drift; recursing
    makes ``_focus_range(p, c, True) == [1-b, 1-a]`` exact by construction.

    Computing it from the complemented `prev` directly is what used to break: at
    prev ~= 0.95, ``10 * prev`` clamps to 1.0 and the zoom hint silently dies.
    """
    if below:
        lo, hi = _focus_range(1.0 - prev, 1.0 - clin, False)
        return [round(1.0 - hi, 6), round(1.0 - lo, 6)]
    return [0.0, min(1.0, max(0.05, round(10.0 * prev, 3), round(clin + 0.05, 3)))]


def _axis_framing(meta: dict, direction: str, t_clin: float | None,
                  t_eval: float | None) -> dict:
    """Say, in clinical language, what this curve's x-axis IS.

    Without this a reader sees `prevalence: 0.954` on a task whose event rate is
    4.6% and reasonably concludes the report is broken. The event here is the
    label's COMPLEMENT (for a "below" task, e.g. successful extubation), declared
    in the task's META `dca_axis` because no name for it is derivable from the
    label.

    `inaction` names the OTHER arm. A decision curve's two reference lines are two
    real policies, and for a two-sided decision the treat-none line carries an
    actual alternative — "keep the patient intubated", not "do nothing". Tasks that
    do not set it fall back to "not <action>", which is what every task meant when
    the only shipped direction was "above".
    """
    axis = meta.get("dca_axis") or {}
    action = axis.get("action") or meta.get("clinical_action") or "treat"
    if direction == "below":
        event = axis.get("event") or f"NOT {meta.get('positive_class', 'the event')}"
    else:
        event = axis.get("event") or meta.get("positive_class") or "the event"
    return {
        "event": event,
        "action": action,
        "inaction": axis.get("inaction") or f"not {action}",
        "x_axis": (f"threshold probability of {event} required to {action}"),
        "prevalence_of": event,
        # where the shipped policy lands on THIS axis: for a below task a 0.10
        # (P failure) policy is 0.90 (P successful extubation); for an above task
        # like icu_readmission it is t itself (0.20).
        "clinical_threshold_on_axis": t_eval,
        "clinical_threshold_on_label_scale": t_clin,
    }


def _clinical_marker(t_axis: float | None, t_label: float | None, direction: str,
                     nb_model: list, nb_all: list, intervals: list[dict],
                     y_true: np.ndarray, y_prob: np.ndarray,
                     action: str = "treat", inaction: str = "") -> dict:
    """The DCA's own conclusion AT the shipped policy: does the model earn its keep?

    `t_axis` is the threshold on the evaluated (decision-basis) curve — the META
    clinical_threshold for an "above" task, or its complement for a "below" one.
    `t_label` is the META threshold itself, which is what cuts the acted arm.

    Kept here, with the curve, rather than hoisted into operating_points.json with
    the other threshold-dependent metrics: for a task like extubation_failure_24h
    the decision curve IS the primary analysis, and a conclusion living elsewhere
    invites the conclusion to be missed. The scorecard reports the same quantity at
    all five grid thresholds, computed through the same primitives, so the two
    cannot disagree.
    """
    if t_axis is None:
        return {}
    idx = int(np.argmin(np.abs(THRESHOLDS - t_axis)))
    ci = bootstrap_ci(
        y_true, y_prob,
        {"model": lambda yt, yp: net_benefit(yt, yp, t_label, t_axis,
                                             direction)})["model"]
    model, treat_all = nb_model[idx], nb_all[idx]
    useful = any(iv["low"] <= t_axis <= iv["high"] for iv in intervals)
    return {
        "clinical_threshold": t_axis,
        "net_benefit_at_clinical_threshold": {
            "model": model,
            "model_ci": ci,
            "treat_all": treat_all,
            "treat_none": 0.0,
        },
        "clinically_useful": useful,
        "verdict": {
            "clinically_useful": useful,
            "model": model,
            "model_ci": ci,
            "treat_all": treat_all,
            "treat_none": 0.0,
            "beats": _beats(model, treat_all, useful, action, inaction),
        },
    }


def _beats(model: float | None, treat_all: float | None, useful: bool,
           action: str, inaction: str = "") -> str:
    """One line a reader can act on, instead of three numbers to compare by eye.

    Both defaults are named. "'extubate' on NO ONE" is a double negative the
    reader has to decode into "keep everyone intubated" — which for a two-sided
    decision is the other real policy, not the absence of one.
    """
    if model is None or treat_all is None:
        return "net benefit is not estimable at the shipped policy"
    none_arm = f"'{inaction}' on everyone" if inaction else f"'{action}' on no one"
    if useful:
        return (f"the model beats both defaults — net benefit {model:.3f}, vs "
                f"{treat_all:.3f} for '{action}' on everyone and 0 for {none_arm}")
    if treat_all >= 0.0:
        return (f"'{action}' on EVERYONE beats the model "
                f"(net benefit {treat_all:.3f} vs {model:.3f}) — the model does not "
                f"earn its keep at the shipped policy")
    return (f"{none_arm[0].upper()}{none_arm[1:]} beats the model "
            f"(net benefit 0 vs {model:.3f}) — the model does not "
            f"earn its keep at the shipped policy")


def _contiguous_intervals(thresholds: np.ndarray, mask: list[bool]) -> list[dict]:
    intervals: list[dict] = []
    start_idx = None
    for i, ok in enumerate(mask):
        if ok and start_idx is None:
            start_idx = i
        elif not ok and start_idx is not None:
            intervals.append({"low": float(thresholds[start_idx]),
                              "high": float(thresholds[i - 1])})
            start_idx = None
    if start_idx is not None:
        intervals.append({"low": float(thresholds[start_idx]),
                          "high": float(thresholds[-1])})
    return intervals
