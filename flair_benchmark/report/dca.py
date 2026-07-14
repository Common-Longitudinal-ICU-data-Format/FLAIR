"""Decision Curve Analysis → dca.json (binary tasks).

The frame is split-filtered upstream (episodic: one row per stay).

Brute-force sweep over threshold probabilities t:

  nb_model = TP/n - FP/n * t/(1-t)
  nb_all   = prevalence - (1-prevalence) * t/(1-t)   (treat everyone)
  nb_none  = 0                                        (treat no one)

THE DECISION BASIS. A DCA describes the ACT, not the alarm — so the curve is
always built on the frame the clinician is actually deciding in, which is the
COMPLEMENT of the label for a ``clinical_direction: "below"`` task.

Task3 is the case in point. The model predicts extubation FAILURE, but the
decision is whether to EXTUBATE, and it fires on LOW risk. So the DCA's event is
"successful extubation" (= the label's complement), the predicted probability is
1-p, and the x-axis is the probability of success a clinician would require
before extubating:

    x = 0    not worried about failure at all, worried about the harms of
             continued intubation  →  extubate everyone
    x = 1    terrified of extubation failure  →  extubate nobody

The task's clinical threshold t (on the FAILURE scale) therefore sits at 1-t on
this axis: task3's 0.15 is x = 0.85 ("extubate when P(success) >= 85%"). Because
the whole frame is complemented, `prevalence` here is P(success) = 0.954, NOT the
failure rate — `metadata.prevalence_of` says so out loud.

``to_decision_basis`` / ``axis_threshold`` are the single source of that mapping;
report/operating_points.py calls them too, so the scorecard's net benefit and this
curve are the same quantity and cannot drift apart.

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

from flair_benchmark.report._ci import bootstrap_ci
from flair_benchmark.report._suppress import MIN_CELL_COUNT

# Uniform 0.01..0.99 body, plus a dense tail at BOTH ends (step .002).
#   low  (0.001..0.05): the rare-event decision region, where an "above" task with
#                       0.15% prevalence actually operates — otherwise it collapses
#                       into one suppressed point near 0.
#   high (0.90..0.999): the mirror image, which is where a "below" task's decision
#                       lives once the frame is complemented (task3 ships at 0.85
#                       and its useful range starts around 0.93). At step .01 the
#                       useful-range boundary would snap to a gridline.
_LOW = np.round(np.arange(0.001, 0.05, 0.002), 3)
_HIGH = np.round(np.arange(0.90, 1.0, 0.002), 3)
THRESHOLDS = np.unique(np.concatenate([
    _LOW, np.round(np.arange(0.01, 1.00, 0.01), 2), _HIGH]))
_TOL = 1e-9


def axis_threshold(t: float, direction: str) -> float:
    """Where a clinical threshold t (label scale) sits on the DECISION-CURVE axis.

    An "act on LOW risk" task decides on the complement, so its threshold maps to
    1-t: task3's 0.15 (P failure) is 0.85 (P successful extubation).
    """
    return round(1.0 - t, 6) if direction == "below" else t


def net_benefit_all(prevalence: float, t: float) -> float | None:
    """Net benefit of acting on EVERYONE at t — the treat-all default strategy.

    Shared with operating_points.py. Note the ``float()`` before rounding: iterating
    the numpy THRESHOLDS grid yields np.float64, whose round() is
    ``rint(x*1e6)/1e6`` and can land 1 ULP away from Python's correctly-rounded
    round() on the identical value. Two files computing "the same" net benefit would
    then disagree in the 6th decimal. Round in one place, in one dtype.
    """
    if t >= 1:
        return None
    odds = t / (1 - t)
    return round(float(prevalence - (1 - prevalence) * odds), 6)


def to_decision_basis(y_true: np.ndarray, y_prob: np.ndarray,
                      direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Put (labels, probabilities) into the frame the clinician decides in.

    For "below", complement both: the event becomes the label's complement (task3:
    successful extubation) and the score becomes P(that event). Net benefit then
    describes treating-the-low-risk — extubating — instead of reading backwards.
    """
    if direction == "below":
        return 1 - y_true, 1.0 - y_prob
    return y_true, y_prob


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
    y_true, y_prob = to_decision_basis(y_true, y_prob, direction)
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
        pred_pos = y_prob >= t
        n_flagged = int(pred_pos.sum())
        if n_flagged < MIN_CELL_COUNT:
            nb_model.append(None)
            suppressed.append(t)
            useful_mask.append(False)
            continue
        model_nb = _net_benefit(y_true, y_prob, t)
        nb_model.append(round(float(model_nb), 6))
        useful_mask.append(model_nb > max(all_nb, 0.0) + _TOL)

    intervals = _contiguous_intervals(THRESHOLDS, useful_mask)
    axis = _axis_framing(meta, direction, t_clin, t_eval)
    clinical = _clinical_marker(t_eval, nb_model, nb_all, intervals, y_true, y_prob,
                                axis["action"])
    # zoom the plot to where rare-event decisions live (treating at ~10x prevalence
    # already implies a high alarm rate), but always include the clinical threshold;
    # capped at 1.0. Keeps the full grid in JSON.
    clin = t_eval or 0.0
    focus_hi = min(1.0, max(0.05, round(10 * prev, 3), round(clin + 0.05, 3)))
    useful_range = (max(intervals, key=lambda iv: iv["high"] - iv["low"])
                    if intervals else None)
    return {
        "task_type": "binary",
        "direction": direction,
        "basis": ("complement: predict non-event (1-p), act when y_prob < "
                  "clinical_threshold" if below else "standard: act when y_prob >= "
                  "clinical_threshold"),
        "clinical_threshold_original": t_clin,
        # NB: on a "below" task this is P(the complement event) — P(successful
        # extubation) for task3, not the failure rate. `metadata.prevalence_of`
        # names it so the 0.95 cannot be misread as a bug.
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
            "focus_range": [0.0, focus_hi],
            "suppressed_thresholds": suppressed,
        },
        "metadata": {
            "grid": "0.001-0.05 step .002 + 0.01-0.99 step .01 + 0.90-0.999 step .002",
            "min_flagged_n": MIN_CELL_COUNT, "n_test": n, **axis,
        },
    }


def _axis_framing(meta: dict, direction: str, t_clin: float | None,
                  t_eval: float | None) -> dict:
    """Say, in clinical language, what this curve's x-axis IS.

    Without this a reader sees `prevalence: 0.954` on a task whose event rate is
    4.6% and reasonably concludes the report is broken. The event here is the
    label's COMPLEMENT (task3: successful extubation), declared in TASK_POLICY's
    `dca_axis` because no name for it is derivable from the label.
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
        "x_axis": (f"threshold probability of {event} required to {action}"),
        "prevalence_of": event,
        # where the shipped policy lands on THIS axis: task3's 0.15 (P failure) is
        # 0.85 (P successful extubation).
        "clinical_threshold_on_axis": t_eval,
        "clinical_threshold_on_label_scale": t_clin,
    }


def _clinical_marker(t: float | None, nb_model: list, nb_all: list,
                     intervals: list[dict], y_true: np.ndarray,
                     y_prob: np.ndarray, action: str = "treat") -> dict:
    """The DCA's own conclusion AT the shipped policy: does the model earn its keep?

    `t` is the threshold on the evaluated (decision-basis) curve — the META
    clinical_threshold for an "above" task, or its complement for a "below" one.

    Kept here, with the curve, rather than hoisted into operating_points.json with
    the other threshold-dependent metrics: for a task like task3 the decision curve
    IS the primary analysis, and a curve whose conclusion lives in another file
    invites the conclusion to be missed. The scorecard reports the same quantity at
    all five grid thresholds, computed through the same primitives, so the two
    cannot disagree.
    """
    if t is None:
        return {}
    idx = int(np.argmin(np.abs(THRESHOLDS - t)))
    ci = bootstrap_ci(y_true, y_prob,
                      {"model": lambda yt, yp: _net_benefit(yt, yp, t)})["model"]
    model, treat_all = nb_model[idx], nb_all[idx]
    useful = any(iv["low"] <= t <= iv["high"] for iv in intervals)
    return {
        "clinical_threshold": t,
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
            "beats": _beats(model, treat_all, useful, action),
        },
    }


def _beats(model: float | None, treat_all: float | None, useful: bool,
           action: str) -> str:
    """One line a reader can act on, instead of three numbers to compare by eye."""
    if model is None or treat_all is None:
        return "net benefit is not estimable at the shipped policy"
    if useful:
        return (f"the model beats both defaults — net benefit {model:.3f}, vs "
                f"{treat_all:.3f} for '{action}' on everyone and 0 on no one")
    if treat_all >= 0.0:
        return (f"'{action}' on EVERYONE beats the model "
                f"(net benefit {treat_all:.3f} vs {model:.3f}) — the model does not "
                f"earn its keep at the shipped policy")
    return (f"'{action}' on NO ONE beats the model (net benefit 0 vs {model:.3f}) — "
            f"the model does not earn its keep at the shipped policy")


def _net_benefit(y_true: np.ndarray, y_prob: np.ndarray, t: float) -> float | None:
    """Model net benefit at threshold t: TP/n - FP/n * t/(1-t)."""
    n = len(y_true)
    if n == 0 or t >= 1:
        return None
    pred_pos = y_prob >= t
    tp = int((pred_pos & (y_true == 1)).sum())
    fp = int(pred_pos.sum()) - tp
    return tp / n - (fp / n) * (t / (1 - t))


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
