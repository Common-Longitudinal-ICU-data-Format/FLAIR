"""Episodic report: one prediction per stay.

Five analyses. The frame is split-filtered and verified one-row-per-stay upstream
(report.build_report); builders here do no filtering.

The split is by WHETHER A DECISION THRESHOLD IS INVOLVED. Four pillars answer
questions about the model that hold no matter where the threshold is put, and are
therefore threshold-FREE:

  discrimination   : ranking — AUROC / AUPRC / lift + ROC & PR curves
  calibration      : do the probabilities mean what they say — ECE, Brier,
                     slope/intercept, reliability bins
  fairness         : does the model RANK and CALIBRATE equally well across
                     subgroups — per-group AUROC / AUPRC / Brier / slope

The fifth owns every number that moves when the threshold moves:

  operating_points : THE scorecard. Confusion matrix, sensitivity / specificity /
                     PPV / NPV / F1 / FPR / NNE / flag-rate, net benefit, the
                     direction-aware decision taken, and subgroup positive-rate /
                     TPR / FPR + parity gaps — all of it at FIVE thresholds (the
                     task's clinical threshold and ± 0.05 / ± 0.10), each scalar
                     with a 95% bootstrap CI.

Hence the two ``False`` flags below: the threshold-dependent blocks discrimination
and fairness used to carry inline now live in operating_points.json, at five
thresholds instead of one.

  dca              : the ONE deliberate exception. It keeps its net-benefit curve
                     AND its verdict at the shipped policy, because for a task like
                     extubation_failure_24h it is the primary analysis and a curve
                     whose conclusion lives in a different file invites the
                     conclusion to be missed. The scorecard reports the same net
                     benefit at all five thresholds, through the same primitives
                     (dca.to_decision_basis / dca.axis_threshold), so the two
                     cannot disagree.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report.calibration import build_calibration
from flair_benchmark.report.dca import build_dca
from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.fairness import build_fairness
from flair_benchmark.report.operating_points import build_episodic_operating_points


def build_episodic_report(preds: pl.DataFrame, meta: dict) -> dict[str, dict]:
    return {
        "discrimination": build_discrimination(preds, meta, operating_point=False),
        "calibration": build_calibration(preds, meta),
        "dca": build_dca(preds, meta),
        "fairness": build_fairness(preds, meta, threshold_metrics=False),
        "operating_points": build_episodic_operating_points(preds, meta),
    }
