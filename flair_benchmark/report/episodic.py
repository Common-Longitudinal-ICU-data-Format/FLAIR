"""Episodic report: one prediction per stay.

Four analyses, all at the task's fixed clinical threshold, every scalar carrying
a 95% bootstrap CI where feasible. The frame is split-filtered and verified
one-row-per-stay upstream (report.build_report); builders here do no filtering.

  discrimination : ranking (AUROC/AUPRC + curves + operating point)
  calibration    : reliability bins, ECE, slope/intercept, Brier
  dca            : net-benefit curve + clinical operating point
  fairness       : subgroup metrics + parity gaps at the clinical threshold
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report.calibration import build_calibration
from flair_benchmark.report.dca import build_dca
from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.fairness import build_fairness


def build_episodic_report(preds: pl.DataFrame, meta: dict) -> dict[str, dict]:
    return {
        "discrimination": build_discrimination(preds, meta),
        "calibration": build_calibration(preds, meta),
        "dca": build_dca(preds, meta),
        "fairness": build_fairness(preds, meta),
    }
