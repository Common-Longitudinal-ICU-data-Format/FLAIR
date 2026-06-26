"""Peak (stay-peak) report: Way 2 of docs/eval_continuous_three_ways.md.

Collapse a continuous (multi-row-per-stay) task to ONE row per stay using the
peak predicted risk over the stay — `y_prob = max`, label = "ever positive"
(`max`) — then score it exactly like an episodic report. This answers the
stay-level screening question: "did this stay ever alarm vs ever have the
event?", the number sepsis papers (Wong 2021, Kamran 2024) headline.

Only two pillars are emitted:
  discrimination : ranking + operating point (+ NNE, the number-needed-to-evaluate)
  fairness       : subgroup ranking + parity

Calibration and DCA are DELIBERATELY omitted: max-of-many is upward biased (the
"peak-calibration trap", §5 of the doc), so a reliability curve / net-benefit on
peak risk is misleading. Honest calibration lives in the landmark report instead.

After the collapse there is one row per stay, so report/_ci.py's row bootstrap is
again an honest patient-level bootstrap.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.fairness import build_fairness

# demographic / context columns carried through the collapse so fairness still
# has something to subgroup on (taken as the stay's first value — constant per
# stay for demographics).
_CARRY_FIRST = ["age_at_admission", "sex_category", "race_category",
                "ethnicity_category"]


def build_peak_report(preds: pl.DataFrame, meta: dict) -> dict[str, dict]:
    """Collapse to stay-peak risk, then emit discrimination + fairness only."""
    stay = _collapse_to_peak(preds, meta)

    discrimination = build_discrimination(stay, meta)
    _inject_nne(discrimination)
    fairness = build_fairness(stay, meta)

    note = {"report_mode": "peak", "n_stays": stay.height,
            "peak_calibration_omitted": True}
    for payload in (discrimination, fairness):
        payload.setdefault("metadata", {}).update(note)
    return {"discrimination": discrimination, "fairness": fairness}


def _collapse_to_peak(preds: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """One row per stay: peak y_prob, ever-positive label, first demographics."""
    block = ("hospitalization_join_id"
             if "hospitalization_join_id" in preds.columns
             else "hospitalization_id")
    label_col = meta["label_column"]
    aggs = [pl.col("y_prob").max().alias("y_prob"),
            pl.col(label_col).max().alias(label_col),
            pl.len().alias("n_windows")]
    aggs += [pl.col(c).first().alias(c)
             for c in _CARRY_FIRST if c in preds.columns]
    return preds.group_by(block).agg(aggs)


def _inject_nne(discrimination: dict) -> None:
    """Add number-needed-to-evaluate (1/PPV) at the operating point, in place.

    NNE is the headline operating number for rare-event screeners (Wong 2021's
    "NNE ~= 109" for the Epic sepsis model): how many flagged stays per true
    positive. Derived from the PPV already computed by build_discrimination, so
    its CI is the inverted PPV CI (and the bounds swap order under 1/x).
    """
    op = discrimination.get("metrics", {}).get("operating_point")
    if not op:
        return
    ppv = op.get("ppv")
    if not ppv or ppv <= 0:
        op["nne"] = None
        op["nne_ci"] = None
        return
    op["nne"] = round(1.0 / ppv, 1)
    ci = op.get("ppv_ci")
    # 1/x is decreasing, so [ppv_lo, ppv_hi] -> [1/ppv_hi, 1/ppv_lo]; a zero
    # lower PPV bound means an unbounded NNE upper bound (null).
    if ci and ci[0] is not None and ci[1] is not None:
        hi = round(1.0 / ci[0], 1) if ci[0] > 0 else None
        lo = round(1.0 / ci[1], 1) if ci[1] > 0 else None
        op["nne_ci"] = [lo, hi]
    else:
        op["nne_ci"] = None
