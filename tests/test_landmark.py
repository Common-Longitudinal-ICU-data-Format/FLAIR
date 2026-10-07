"""Formula correctness for landmark.json.

Landmark t is the slice ``window_number == t`` — the cohort a deployed model
would actually score on hospitalization day/hour t. Everything is checked
against sklearn computed directly on that slice, never against another part of
the report.

  Part 1  the slices      is landmark t really window_number == t
  Part 2  per-landmark    do AUROC / AUPRC / Brier / slope match sklearn
  Part 3  pooling         are the four pooled numbers weighted as documented
  Part 4  suppression     small and single-class risk sets
  Part 5  what is absent  no fairness, no thresholds, no curves
  Part 6  determinism     same data -> same numbers, any row order
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from flair_benchmark.report.landmark import build_landmark

META = {
    "name": "unit_task",
    "task_type": "binary",
    "label_column": "y",
    "window_unit": "day",
    "landmark_horizon": 6,
    "clinical_threshold": 0.30,
    "clinical_direction": "above",
}


def make_preds(n_stays: int = 300, max_w: int = 6, seed: int = 3) -> pl.DataFrame:
    """Several windows per stay; risk sets shrink with hospitalization time."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        # stays end at different times, so deep landmarks have smaller risk sets
        last = int(rng.integers(2, max_w + 1))
        for w in range(1, last + 1):
            y = int(rng.binomial(1, 0.2))
            rows.append({
                "hospitalization_join_id": f"H{i:04d}",
                "window_number": w,
                "y_prob": float(np.clip(rng.beta(2, 6) + 0.3 * y, 0.001, 0.999)),
                "y": y,
            })
    return pl.DataFrame(rows)


def slice_of(df: pl.DataFrame, t: int) -> pl.DataFrame:
    return df.filter(pl.col("window_number") == t)


def scored(payload: dict) -> list[dict]:
    return [e for e in payload["by_landmark"]
            if not e.get("suppressed") and not e.get("undefined")]


@pytest.fixture(scope="module")
def preds():
    return make_preds()


@pytest.fixture(scope="module")
def payload(preds):
    return build_landmark(preds, META)


# ============================================================== Part 1 — slices


def test_one_entry_per_landmark_in_the_horizon(payload):
    times = [e["hospitalization_time"] for e in payload["by_landmark"]]
    assert times == list(range(1, META["landmark_horizon"] + 1))


def test_landmark_n_is_the_size_of_that_window_slice(preds, payload):
    for e in payload["by_landmark"]:
        t = e["hospitalization_time"]
        if isinstance(e["n"], str):
            continue
        assert e["n"] == slice_of(preds, t).height


def test_each_landmark_holds_one_row_per_stay(preds, payload):
    """A stay-level bootstrap is only honest if the slice has no duplicates."""
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        assert sl["hospitalization_join_id"].n_unique() == sl.height


def test_risk_set_shrinks_with_hospitalization_time(preds, payload):
    ns = [e["n"] for e in payload["by_landmark"] if isinstance(e["n"], int)]
    assert all(a >= b for a, b in zip(ns, ns[1:])), "risk set grew over time"


# ======================================================== Part 2 — per-landmark


def test_landmark_auroc_matches_sklearn_on_that_slice(preds, payload):
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        assert e["auroc"] == pytest.approx(
            roc_auc_score(sl["y"].to_numpy(), sl["y_prob"].to_numpy()), abs=1e-4)


def test_landmark_auprc_matches_sklearn_on_that_slice(preds, payload):
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        assert e["auprc"] == pytest.approx(
            average_precision_score(sl["y"].to_numpy(),
                                    sl["y_prob"].to_numpy()), abs=1e-4)


def test_landmark_brier_matches_sklearn_on_that_slice(preds, payload):
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        assert e["brier"] == pytest.approx(
            brier_score_loss(sl["y"].to_numpy(), sl["y_prob"].to_numpy()),
            abs=1e-4)


def test_landmark_prevalence_matches_the_slice(preds, payload):
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        assert e["prevalence"] == pytest.approx(float(sl["y"].mean()), abs=1e-4)


def test_every_landmark_metric_carries_a_ci(payload):
    for e in scored(payload):
        for k in ("auroc", "auprc", "brier", "calibration_slope"):
            assert f"{k}_ci" in e, f"{k} has no CI at t={e['hospitalization_time']}"


# ============================================================= Part 3 — pooling


def test_pooled_auroc_is_pair_weighted_across_landmarks(preds, payload):
    """Heagerty & Zheng: weight each landmark by its comparable-pair count."""
    num = den = 0.0
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        y = sl["y"].to_numpy()
        cases, controls = int(y.sum()), int((y == 0).sum())
        w = cases * controls
        if w <= 0:
            continue
        num += roc_auc_score(y, sl["y_prob"].to_numpy()) * w
        den += w
    assert payload["pooled"]["auroc"] == pytest.approx(num / den, abs=1e-3)


def test_pooled_auprc_is_case_weighted(preds, payload):
    """Average precision is anchored on the positives, so cases are the weight."""
    num = den = 0.0
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        y = sl["y"].to_numpy()
        cases = int(y.sum())
        if cases == 0:
            continue
        num += average_precision_score(y, sl["y_prob"].to_numpy()) * cases
        den += cases
    assert payload["pooled"]["auprc"] == pytest.approx(num / den, abs=1e-3)


def test_pooled_brier_equals_the_plain_brier_over_all_landmark_rows(preds, payload):
    """An n-weighted mean of per-slice Brier IS the pooled mean squared error."""
    ys, ps = [], []
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        ys.append(sl["y"].to_numpy())
        ps.append(sl["y_prob"].to_numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    assert payload["pooled"]["brier"] == pytest.approx(
        brier_score_loss(y, p), abs=1e-4)


def test_pooled_slope_is_one_logistic_fit_not_an_average_of_slopes(preds, payload):
    """Averaging regression coefficients is a different estimator entirely."""
    ys, ps = [], []
    for e in scored(payload):
        sl = slice_of(preds, e["hospitalization_time"])
        ys.append(sl["y"].to_numpy())
        ps.append(sl["y_prob"].to_numpy())
    y = np.concatenate(ys)
    p = np.clip(np.concatenate(ps), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    expect = LogisticRegression(C=1e6).fit(logit, y).coef_[0, 0]
    assert payload["pooled"]["calibration_slope"] == pytest.approx(expect, abs=1e-3)

    per_landmark = [e["calibration_slope"] for e in scored(payload)
                    if e["calibration_slope"] is not None]
    mean_of_slopes = float(np.mean(per_landmark))
    # the two estimators must not be silently interchangeable
    assert payload["pooled"]["calibration_slope"] != pytest.approx(
        mean_of_slopes, abs=1e-9)


def test_a_perfectly_calibrated_model_pools_to_slope_one():
    """The headline sanity check: slope 1 means probabilities mean what they say."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4000):
        for w in (1, 2, 3):
            p = float(rng.uniform(0.05, 0.95))
            rows.append({"hospitalization_join_id": f"H{i}", "window_number": w,
                         "y_prob": p, "y": int(rng.binomial(1, p))})
    payload = build_landmark(pl.DataFrame(rows),
                             {**META, "landmark_horizon": 3})
    assert payload["pooled"]["calibration_slope"] == pytest.approx(1.0, abs=0.12)


def test_an_overconfident_model_pools_to_a_slope_below_one():
    """Doubling the logit halves the slope — the classic overfit signature."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(4000):
        for w in (1, 2, 3):
            true_p = float(rng.uniform(0.05, 0.95))
            logit = np.log(true_p / (1 - true_p))
            shown = 1.0 / (1.0 + np.exp(-2.0 * logit))   # twice as confident
            rows.append({"hospitalization_join_id": f"H{i}", "window_number": w,
                         "y_prob": float(shown),
                         "y": int(rng.binomial(1, true_p))})
    payload = build_landmark(pl.DataFrame(rows),
                             {**META, "landmark_horizon": 3})
    assert payload["pooled"]["calibration_slope"] == pytest.approx(0.5, abs=0.08)


def test_pooled_point_estimates_lie_inside_their_cis(payload):
    for k in ("auroc", "auprc", "brier", "calibration_slope"):
        v = payload["pooled"][k]
        ci = payload["pooled"][f"{k}_ci"]
        if v is None or ci is None:
            continue
        assert ci[0] <= v <= ci[1], f"{k}={v} outside {ci}"


def test_pooled_auroc_ignores_landmarks_with_no_contrast():
    """A single-class risk set has zero comparable pairs and must not shift it."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(200):
        for w in (1, 2):
            rows.append({"hospitalization_join_id": f"H{i}", "window_number": w,
                         "y_prob": float(rng.uniform(0.01, 0.99)),
                         "y": 0 if w == 2 else int(rng.binomial(1, 0.3))})
    payload = build_landmark(pl.DataFrame(rows), {**META, "landmark_horizon": 2})
    lm2 = payload["by_landmark"][1]
    assert lm2.get("undefined") is True
    # the pool must equal landmark 1 alone
    assert payload["pooled"]["auroc"] == pytest.approx(
        payload["by_landmark"][0]["auroc"], abs=1e-3)


# ======================================================== Part 4 — suppression


def test_a_risk_set_below_ten_is_suppressed_whole():
    rows = []
    for i in range(40):
        rows.append({"hospitalization_join_id": f"H{i}", "window_number": 1,
                     "y_prob": 0.1 + 0.02 * (i % 10), "y": i % 2})
    for i in range(5):                       # only 5 stays reach window 2
        rows.append({"hospitalization_join_id": f"H{i}", "window_number": 2,
                     "y_prob": 0.4, "y": i % 2})
    payload = build_landmark(pl.DataFrame(rows), {**META, "landmark_horizon": 2})
    lm2 = payload["by_landmark"][1]
    assert lm2["suppressed"] is True
    assert lm2["n"] == "<10"
    assert "auroc" not in lm2


def test_a_single_class_risk_set_is_marked_undefined():
    rows = []
    for i in range(60):
        rows.append({"hospitalization_join_id": f"H{i}", "window_number": 1,
                     "y_prob": 0.1 + 0.01 * (i % 20), "y": i % 2})
        rows.append({"hospitalization_join_id": f"H{i}", "window_number": 2,
                     "y_prob": 0.2, "y": 0})           # no events at all
    payload = build_landmark(pl.DataFrame(rows), {**META, "landmark_horizon": 2})
    lm2 = payload["by_landmark"][1]
    assert lm2["undefined"] is True
    assert "single label class" in lm2["reason"]


def test_landmarks_past_the_data_are_still_emitted_as_stubs():
    rows = [{"hospitalization_join_id": f"H{i}", "window_number": 1,
             "y_prob": 0.1 + 0.01 * (i % 20), "y": i % 2} for i in range(60)]
    payload = build_landmark(pl.DataFrame(rows), {**META, "landmark_horizon": 4})
    assert len(payload["by_landmark"]) == 4
    for e in payload["by_landmark"][1:]:
        assert e["suppressed"] is True


# ====================================================== Part 5 — what is absent


def test_landmark_carries_no_fairness(payload):
    dumped = str(payload)
    for banned in ("sex_category", "race_category", "ethnicity_category",
                   "parity", "subgroups", "age_bin"):
        assert banned not in dumped, f"{banned} leaked into landmark.json"


def test_landmark_carries_no_threshold_metrics(payload):
    dumped = str(payload)
    for banned in ("sensitivity", "specificity", "ppv", "npv",
                   "operating_point", "threshold", "net_benefit"):
        assert banned not in dumped, f"{banned} leaked into landmark.json"


def test_landmark_carries_no_curves(payload):
    """Per-landmark ROC/PR curves were the bulk of the old bundle."""
    for e in scored(payload):
        assert "curves" not in e
        for k, v in e.items():
            assert not isinstance(v, list) or k.endswith("_ci"), \
                f"unexpected array {k} at t={e['hospitalization_time']}"


def test_landmark_entry_has_exactly_the_documented_keys(payload):
    expect = {"hospitalization_time", "n", "prevalence",
              "auroc", "auroc_ci", "auprc", "auprc_ci",
              "brier", "brier_ci", "calibration_slope", "calibration_slope_ci"}
    for e in scored(payload):
        assert set(e) == expect, f"key drift: {set(e) ^ expect}"


def test_metadata_names_the_window_unit_and_horizon(payload):
    md = payload["metadata"]
    assert md["window_unit"] == "day"
    assert md["horizon"] == META["landmark_horizon"]
    assert md["n_landmarks_scored"] == len(scored(payload))


def test_missing_landmark_horizon_is_an_error():
    with pytest.raises(ValueError, match="landmark_horizon"):
        build_landmark(make_preds(n_stays=20),
                       {k: v for k, v in META.items() if k != "landmark_horizon"})


# ======================================================== Part 6 — determinism


def test_row_order_does_not_change_any_number(preds):
    a = build_landmark(preds, META)
    b = build_landmark(preds.reverse(), META)
    assert a == b


def test_repeated_builds_are_identical(preds):
    assert build_landmark(preds, META) == build_landmark(preds, META)
