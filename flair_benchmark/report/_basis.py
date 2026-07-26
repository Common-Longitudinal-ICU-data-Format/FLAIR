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

BOUNDARY CONVENTION (p exactly == t). The acted arm is ALWAYS
``action_mask(p, t_label, direction)`` — ``p < t`` for a "below" task (its action
is STRICT: "extubate when P(failure) < t"), ``p >= t`` otherwise, matching the
alarm convention and the standard DCA rule. Every consumer computes it that way,
on the LABEL scale, in the operands the task author actually wrote; the decision
axis only rescales the odds weight inside the net-benefit formula.

NEVER derive the arm from ``1-p``. ``1.0 - x`` is exact in float64 only for
x in [0.5, 1] (Sterbenz), so a patient sitting exactly on t lands on the wrong
side of ``(1-p) > (1-t)`` at 20 of the 99 sweep thresholds — which is how dca.py
and operating_points.py used to publish two different numbers for the same
quantity. ``to_decision_basis`` still complements the probabilities, but only for
display; arm membership never reads them.

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
    1-t: e.g. an extubate-on-low-failure-risk task's 0.10 (P failure) is 0.90
    (P successful extubation).
    """
    return round(1.0 - t, 6) if direction == "below" else t


def label_threshold(x: float, direction: str) -> float:
    """Inverse of :func:`axis_threshold`: the LABEL-scale threshold behind axis x.

    Exact on the 0.01 sweep grid in both directions, so a consumer walking the
    decision axis can recover the operand the task author wrote and compare
    against it directly. This is what lets net benefit be evaluated at an axis
    threshold while its acted arm is still cut on the label scale.
    """
    return round(1.0 - x, 6) if direction == "below" else x


def to_decision_basis(y_true: np.ndarray, y_prob: np.ndarray,
                      direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Put (labels, probabilities) into the frame the clinician decides in.

    For "below", complement both: the event becomes the label's complement (e.g.
    successful extubation for an extubate-on-low-risk task) and the score becomes
    P(that event). Net benefit then describes treating-the-low-risk — extubating —
    instead of reading backwards.

    The returned LABELS are the decision-basis event and are used in every
    downstream count. The returned PROBABILITIES are for display only — never
    derive the acted arm from them (see the boundary convention in the module
    docstring); use :func:`action_mask` on the original label-scale scores.
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


def net_benefit(y_true_dec: np.ndarray, y_prob_label: np.ndarray,
                t_label: float, t_axis: float, direction: str) -> float | None:
    """Model net benefit on the decision axis: TP/n - FP/n * t_axis/(1-t_axis).

    Two thresholds, two jobs, and keeping them separate is the whole point:

      ``t_label``  cuts the acted arm, via :func:`action_mask`, on the LABEL scale
                   — the operand the task author wrote (``p < t`` / ``p >= t``).
      ``t_axis``   weighs a false action against a missed one, and is the only
                   place the complement enters for a "below" task.

    ``y_true_dec`` is on the decision basis (:func:`to_decision_basis`), so TP is
    "acted, and the decision-basis event happened" — for an extubate task, "we
    extubated and they stayed extubated".
    """
    n = len(y_true_dec)
    if n == 0 or t_axis >= 1:
        return None
    acted = action_mask(y_prob_label, t_label, direction)
    tp = int((acted & (y_true_dec == 1)).sum())
    fp = int(acted.sum()) - tp
    return tp / n - (fp / n) * (t_axis / (1 - t_axis))
