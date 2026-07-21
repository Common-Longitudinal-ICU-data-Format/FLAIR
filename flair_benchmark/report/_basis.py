"""Shared decision-basis primitives: alarm vs action, and the DCA frame.

ONE place decides what "the model fired" means, so every consumer is correct
by construction instead of by remembering to thread ``clinical_direction``:

  ALARM basis   : ``y_prob >= t`` in BOTH directions. "How well do we DETECT
                  the event." Confusion matrices, sensitivity/specificity/PPV
                  and detection curves are ALWAYS scored here — scoring them on
                  the action would book "acted AND the event happened" as a
                  true positive, a metric that RISES as the model gets worse
                  for an act-on-LOW-risk task.
  ACTION basis  : the decision actually taken. ``y_prob < t`` when
                  ``clinical_direction == "below"`` (act on LOW risk, e.g.
                  extubate when failure risk < t), else ``y_prob >= t``. Timing
                  panels ("did the model fire before treatment", first-alert
                  time) and the decision/net-benefit blocks live here.
  DECISION basis: the frame a decision curve is drawn in. For a "below" task
                  the event is the label's COMPLEMENT and the score is 1-p, so
                  net benefit describes the act (extubate) rather than reading
                  backwards; the x-axis threshold maps to 1-t.

BOUNDARY CONVENTION (p exactly == t): the action for a "below" task is STRICT
(``p < t`` — e.g. "extubate when P(failure) < t"), so on the complement basis
the acted arm is ``P(success) > 1-t``, also strict. Above tasks act non-strictly
(``p >= t``), matching the alarm convention and the standard DCA rule — this is
what all live tasks use, including task3 ("perform a tracheostomy when
P(failure) >= 0.80"). ``net_benefit(..., strict=True)`` is how the complement
basis expresses that; the curve and the scorecard both route through it, so
they cannot disagree at the boundary.

Both a polars-expression and a numpy-mask form are provided for each mask so
frame-level and array-level code share one definition.
"""

from __future__ import annotations

import numpy as np
import polars as pl


# --------------------------------------------------------------- alarm basis


def alarm_expr(threshold: float, prob_col: str = "y_prob") -> pl.Expr:
    """ALARM: the model warns of the (high-risk) event — ``p >= t``, always."""
    return pl.col(prob_col) >= threshold


def alarm_mask(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Numpy form of :func:`alarm_expr`."""
    return y_prob >= threshold


# -------------------------------------------------------------- action basis


def action_expr(threshold: float, direction: str,
                prob_col: str = "y_prob") -> pl.Expr:
    """ACTION: p < t when the action fires on LOW risk, else p >= t."""
    if direction == "below":
        return pl.col(prob_col) < threshold
    return pl.col(prob_col) >= threshold


def action_mask(y_prob: np.ndarray, threshold: float,
                direction: str) -> np.ndarray:
    """Numpy form of :func:`action_expr`."""
    if direction == "below":
        return y_prob < threshold
    return y_prob >= threshold


# ------------------------------------------------------------ decision basis


def axis_threshold(t: float, direction: str) -> float:
    """Where a clinical threshold t (label scale) sits on the DECISION-CURVE axis.

    An "act on LOW risk" task decides on the complement, so its threshold maps to
    1-t: e.g. an extubate-on-low-failure-risk task's 0.15 (P failure) is 0.85
    (P successful extubation).
    """
    return round(1.0 - t, 6) if direction == "below" else t


def to_decision_basis(y_true: np.ndarray, y_prob: np.ndarray,
                      direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Put (labels, probabilities) into the frame the clinician decides in.

    For "below", complement both: the event becomes the label's complement (e.g.
    successful extubation for an extubate-on-low-risk task) and the score becomes
    P(that event). Net benefit then describes treating-the-low-risk — extubating —
    instead of reading backwards.
    """
    if direction == "below":
        return 1 - y_true, 1.0 - y_prob
    return y_true, y_prob


def net_benefit_all(prevalence: float, t: float) -> float | None:
    """Net benefit of acting on EVERYONE at t — the treat-all default strategy.

    Note the ``float()`` before rounding: iterating a numpy grid yields
    np.float64, whose round() is ``rint(x*1e6)/1e6`` and can land 1 ULP away
    from Python's correctly-rounded round() on the identical value. Two files
    computing "the same" net benefit would then disagree in the 6th decimal.
    Round in one place, in one dtype.
    """
    if t >= 1:
        return None
    odds = t / (1 - t)
    return round(float(prevalence - (1 - prevalence) * odds), 6)


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, t: float,
                strict: bool = False) -> float | None:
    """Model net benefit at threshold t: TP/n - FP/n * t/(1-t).

    ``strict=True`` counts the acted arm as ``y_prob > t`` instead of ``>= t``
    — pass it on the COMPLEMENT basis of a "below" task so the acted arm is
    exactly the shipped action (``P(fail) < t``; see module docstring). Above
    tasks keep the standard non-strict rule.
    """
    n = len(y_true)
    if n == 0 or t >= 1:
        return None
    pred_pos = (y_prob > t) if strict else (y_prob >= t)
    tp = int((pred_pos & (y_true == 1)).sum())
    fp = int(pred_pos.sum()) - tp
    return tp / n - (fp / n) * (t / (1 - t))
