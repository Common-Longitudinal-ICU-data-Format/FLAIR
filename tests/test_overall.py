"""Formula correctness for overall.json (episodic, threshold-free).

Two blocks only: how well the model RANKS, and whether its probabilities mean
what they say. Everything threshold-dependent lives in
hospitalization_level.json.

  Part 1  discrimination  AUROC / AUPRC against sklearn
  Part 2  fixed grids     the curves as one array each, poolable across sites
  Part 3  calibration     Brier, slope, intercept, quantile bins
  Part 4  what is absent  no ECE, no bin counts, no DCA block
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from flair_benchmark.report.overall import (
    GRID,
    build_overall,
)

META = {
    "name": "unit_task",
    "task_type": "binary",
    "label_column": "y",
    "clinical_threshold": 0.30,
    "clinical_direction": "above",
}


def make_preds(n: int = 800, seed: int = 13) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.2, n)
    p = np.clip(rng.beta(2, 6, n) + 0.35 * y, 0.001, 0.999)
    return pl.DataFrame({
        "hospitalization_join_id": [f"H{i:04d}" for i in range(n)],
        "y_prob": p,
        "y": y,
    })


def calibrated(n: int = 6000, seed: int = 2) -> pl.DataFrame:
    """A model whose stated probabilities are exactly right."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    return pl.DataFrame({
        "hospitalization_join_id": [f"H{i:05d}" for i in range(n)],
        "y_prob": p,
        "y": rng.binomial(1, p),
    })


@pytest.fixture(scope="module")
def preds():
    return make_preds()


@pytest.fixture(scope="module")
def payload(preds):
    return build_overall(preds, META)


# ====================================================== Part 1 — discrimination


def test_auroc_matches_sklearn(preds, payload):
    assert payload["discrimination"]["auroc"] == pytest.approx(
        roc_auc_score(preds["y"].to_numpy(), preds["y_prob"].to_numpy()),
        abs=1e-4)


def test_auprc_matches_sklearn(preds, payload):
    assert payload["discrimination"]["auprc"] == pytest.approx(
        average_precision_score(preds["y"].to_numpy(),
                                preds["y_prob"].to_numpy()), abs=1e-4)


def test_auprc_baseline_is_the_prevalence(preds, payload):
    prev = float(preds["y"].mean())
    assert payload["prevalence"] == pytest.approx(prev, abs=1e-6)
    assert payload["discrimination"]["auprc_baseline"] == pytest.approx(
        prev, abs=1e-6)


def test_auroc_and_auprc_carry_cis(payload):
    d = payload["discrimination"]
    for k in ("auroc", "auprc"):
        ci = d[f"{k}_ci"]
        assert ci is not None and ci[0] <= d[k] <= ci[1]


def test_n_is_the_cohort_size(preds, payload):
    assert payload["n"] == preds.height


# ========================================================= Part 2 — fixed grids


def test_grid_is_101_points_from_zero_to_one():
    assert len(GRID) == 101
    assert GRID[0] == 0.0
    assert GRID[-1] == 1.0


def test_curves_are_one_array_each_on_that_grid(payload):
    d = payload["discrimination"]
    assert len(d["tpr_at_grid"]) == 101
    assert len(d["precision_at_grid"]) == 101
    # the x-axes are known constants and must NOT be stored
    assert "fpr" not in d
    assert "recall" not in d
    assert "thresholds" not in d


def envelope(x, y, grid):
    """Independent reference: one y per x (the max), then linear interpolation.

    Written as a plain dict loop on purpose — same math as the production
    lexsort, deliberately different code.
    """
    best: dict[float, float] = {}
    for xi, yi in zip(x, y):
        best[float(xi)] = max(best.get(float(xi), -np.inf), float(yi))
    xs = sorted(best)
    return np.interp(grid, np.array(xs), np.array([best[v] for v in xs]))


def test_tpr_at_grid_matches_an_interpolation_of_sklearns_roc(preds, payload):
    fpr, tpr, _ = roc_curve(preds["y"].to_numpy(), preds["y_prob"].to_numpy())
    expect = envelope(fpr, tpr, np.array(GRID))
    got = np.array(payload["discrimination"]["tpr_at_grid"])
    assert np.max(np.abs(got - expect)) < 1e-4


def test_tpr_at_grid_is_a_proper_roc(payload):
    tpr = payload["discrimination"]["tpr_at_grid"]
    # NOT 0 at fpr=0: every positive ranked above every negative contributes a
    # vertical rise there, which is a real property of the model, not an error.
    assert 0.0 <= tpr[0] < 1.0
    assert tpr[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(a <= b + 1e-9 for a, b in zip(tpr, tpr[1:])), "ROC must not dip"


def test_tpr_at_fpr_zero_is_the_best_tpr_with_no_false_alarms(preds, payload):
    """The vertical rise at FPR 0: how many events outrank every non-event."""
    y = preds["y"].to_numpy()
    p = preds["y_prob"].to_numpy()
    cutoff = p[y == 0].max()          # highest-scoring non-event
    expect = float((p[y == 1] > cutoff).sum() / (y == 1).sum())
    assert payload["discrimination"]["tpr_at_grid"][0] == pytest.approx(
        expect, abs=1e-3)


def test_auroc_is_recoverable_from_the_stored_roc_grid(preds, payload):
    """The stored array must describe the same ranking sklearn scored."""
    area = float(np.trapezoid(payload["discrimination"]["tpr_at_grid"],
                              np.array(GRID)))
    assert area == pytest.approx(
        roc_auc_score(preds["y"].to_numpy(), preds["y_prob"].to_numpy()),
        abs=5e-3)


def test_precision_at_full_recall_is_the_best_precision_that_catches_everything(
        preds, payload):
    """Recall 1.0 is reached at MANY thresholds, not only at 'flag everyone'.

    The lowest of them gives precision == prevalence; the published value is the
    highest — the tightest threshold that still misses no event.
    """
    y = preds["y"].to_numpy()
    p = preds["y_prob"].to_numpy()
    cutoff = p[y == 1].min()          # lowest-scoring event
    flagged = p >= cutoff
    expect = float(y[flagged].mean())
    got = payload["discrimination"]["precision_at_grid"][-1]
    assert got == pytest.approx(expect, abs=1e-3)
    assert got >= float(y.mean()), "precision can never fall below prevalence"


def test_precision_at_grid_matches_sklearns_pr_curve(preds, payload):
    prec, rec, _ = precision_recall_curve(preds["y"].to_numpy(),
                                          preds["y_prob"].to_numpy())
    expect = envelope(rec, prec, np.array(GRID))
    got = np.array(payload["discrimination"]["precision_at_grid"])
    assert np.max(np.abs(got - expect)) < 1e-4


# ========================================================= Part 3 — calibration


def test_brier_matches_sklearn(preds, payload):
    assert payload["calibration"]["brier"] == pytest.approx(
        brier_score_loss(preds["y"].to_numpy(), preds["y_prob"].to_numpy()),
        abs=1e-4)


def test_a_perfectly_calibrated_model_scores_slope_one_intercept_zero():
    payload = build_overall(calibrated(), META)
    assert payload["calibration"]["slope"] == pytest.approx(1.0, abs=0.1)
    assert payload["calibration"]["intercept"] == pytest.approx(0.0, abs=0.1)


def test_an_overconfident_model_scores_a_slope_below_one():
    rng = np.random.default_rng(4)
    n = 6000
    true_p = rng.uniform(0.02, 0.98, n)
    logit = np.log(true_p / (1 - true_p))
    shown = 1.0 / (1.0 + np.exp(-2.0 * logit))     # twice as confident
    df = pl.DataFrame({
        "hospitalization_join_id": [f"H{i:05d}" for i in range(n)],
        "y_prob": shown,
        "y": rng.binomial(1, true_p),
    })
    assert build_overall(df, META)["calibration"]["slope"] == pytest.approx(
        0.5, abs=0.08)


def test_calibration_scalars_carry_cis(payload):
    cal = payload["calibration"]
    for k in ("brier", "slope", "intercept"):
        ci = cal[f"{k}_ci"]
        assert ci is not None and ci[0] <= cal[k] <= ci[1], f"{k} outside {ci}"


def test_quantile_bins_hold_roughly_equal_numbers_of_stays(preds, payload):
    """Equal-COUNT binning is what makes ~10 points span a rare event's range."""
    bins = payload["calibration"]["quantile_bins"]
    pred = bins["bin_pred_prob"]
    assert len(pred) == 10
    assert len(bins["bin_true_prob"]) == 10
    # each bin's mean predicted probability must be increasing
    vals = [v for v in pred if v is not None]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_quantile_bin_means_match_a_hand_computation(preds, payload):
    p = preds["y_prob"].to_numpy()
    y = preds["y"].to_numpy()
    edges = np.quantile(p, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    bins = payload["calibration"]["quantile_bins"]
    for b in range(10):
        mask = idx == b
        if not mask.any():
            continue
        assert bins["bin_pred_prob"][b] == pytest.approx(
            float(p[mask].mean()), abs=1e-3)
        if int(mask.sum()) >= 10:
            assert bins["bin_true_prob"][b] == pytest.approx(
                float(y[mask].mean()), abs=1e-3)


def test_a_bin_with_fewer_than_ten_stays_has_its_observed_rate_nulled():
    n = 55                          # 10 quantile bins over 55 rows -> ~5 each
    rng = np.random.default_rng(9)
    df = pl.DataFrame({
        "hospitalization_join_id": [f"H{i}" for i in range(n)],
        "y_prob": np.linspace(0.01, 0.99, n),
        "y": rng.binomial(1, 0.4, n),
    })
    bins = build_overall(df, META)["calibration"]["quantile_bins"]
    assert all(v is None for v in bins["bin_true_prob"])
    assert any(v is not None for v in bins["bin_pred_prob"])


# ====================================================== Part 4 — what is absent


def test_no_ece_anywhere(payload):
    assert "ece" not in str(payload)


def test_no_bin_counts_or_edges_are_published(payload):
    bins = payload["calibration"]["quantile_bins"]
    assert set(bins) == {"bin_pred_prob", "bin_true_prob"}


def test_no_equal_width_or_smoothed_curve(payload):
    dumped = str(payload)
    for banned in ("smoothed", "reliability", "o_e_ratio", "bin_edges"):
        assert banned not in dumped, f"{banned} survived"


def test_no_dca_block(payload):
    """Net benefit lives in hospitalization_level.json, published once."""
    assert "dca" not in payload
    assert "net_benefit" not in str(payload)


def test_no_threshold_dependent_metrics(payload):
    dumped = str(payload)
    for banned in ("sensitivity", "specificity", "ppv", "npv",
                   "operating_point", "flag_rate"):
        assert banned not in dumped, f"{banned} leaked into overall.json"


def test_single_class_cohort_degrades_to_a_marker():
    df = pl.DataFrame({
        "hospitalization_join_id": [f"H{i}" for i in range(50)],
        "y_prob": np.linspace(0.01, 0.5, 50),
        "y": np.zeros(50, dtype=int),
    })
    payload = build_overall(df, META)
    assert payload["discrimination"]["auroc"] is None
    assert "reason" in payload["metadata"]


def test_repeated_builds_are_identical(preds):
    assert build_overall(preds, META) == build_overall(preds, META)
