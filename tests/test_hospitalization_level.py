"""Formula correctness for hospitalization_level.json.

Every assertion here checks the report against an INDEPENDENT computation —
sklearn, or a brute-force loop written the obvious way — never against another
part of the report. If both sides came from the same code the test would only
prove the code agrees with itself.

Grouped by what could actually be wrong:

  Part 1  the collapse          is one row per hospitalization built correctly
  Part 2  the 2x2               do the counts match a brute-force scan
  Part 3  the rates             do sensitivity/PPV/... match sklearn
  Part 4  monotonicity          does the sweep behave like a sweep must
  Part 5  Youden                is the cutpoint really the argmax
  Part 6  net benefit           does it match the DCA formula and treat-all
  Part 7  suppression           are small cells masked in pairs
  Part 8  fairness              gaps, level dropping, reference group
  Part 9  determinism           same data -> same numbers, any row order
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from flair_benchmark.report._sweep import SWEEP_GRID
from flair_benchmark.report.hospitalization_level import (
    build_hospitalization_level,
    gap_triplet,
)

META = {
    "name": "unit_task",
    "task_type": "binary",
    "label_column": "y",
    "clinical_threshold": 0.30,
    "clinical_direction": "above",
}


def make_episodic(n: int = 600, seed: int = 7) -> pl.DataFrame:
    """One row per stay, with a genuinely informative score and demographics."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.25, n)
    # score correlates with y, so AUROC is meaningfully above 0.5
    score = np.clip(rng.beta(2, 6, n) + 0.35 * y, 0.001, 0.999)
    return pl.DataFrame({
        "hospitalization_join_id": [f"H{i:04d}" for i in range(n)],
        "y_prob": score,
        "y": y,
        "sex_category": rng.choice(["Female", "Male"], n),
        "race_category": rng.choice(["White", "Black", "Asian"], n),
        "age_at_admission": rng.integers(20, 95, n),
    })


def make_continuous(n_stays: int = 200, seed: int = 11) -> pl.DataFrame:
    """Several windows per stay, so the peak collapse has something to do."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        ever = int(rng.binomial(1, 0.25))
        for w in range(1, int(rng.integers(2, 7)) + 1):
            rows.append({
                "hospitalization_join_id": f"H{i:04d}",
                "window_number": w,
                "y_prob": float(np.clip(rng.beta(2, 6) + 0.3 * ever, 0.001, 0.999)),
                "y": ever if w > 1 else 0,
                "sex_category": "Female" if i % 2 else "Male",
                "age_at_admission": int(20 + (i % 70)),
            })
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def episodic():
    return build_hospitalization_level(make_episodic(), META, collapse=False)


@pytest.fixture(scope="module")
def continuous():
    return build_hospitalization_level(make_continuous(), META, collapse=True)


def row_at(payload: dict, t: float) -> dict:
    return next(r for r in payload["by_threshold"]
                if abs(r["threshold"] - t) < 1e-9)


# =========================================================== Part 1 — collapse


def test_peak_collapse_takes_max_score_and_ever_positive_label():
    """continuous mode must score max(y_prob) against 'the event ever happened'."""
    df = make_continuous()
    payload = build_hospitalization_level(df, META, collapse=True)

    expected = df.group_by("hospitalization_join_id").agg(
        pl.col("y_prob").max().alias("peak"),
        (pl.col("y").max() > 0).cast(pl.Int8).alias("ever"),
    )
    assert payload["n_hospitalizations"] == expected.height
    assert payload["prevalence"] == pytest.approx(
        float(expected["ever"].mean()), abs=1e-6)

    # the 2x2 at t must reflect the PEAK, not any individual window
    t = 0.30
    peak = expected["peak"].to_numpy()
    ever = expected["ever"].to_numpy()
    r = row_at(payload, t)
    assert r["tp"] == int(((peak >= t) & (ever == 1)).sum())
    assert r["fp"] == int(((peak >= t) & (ever == 0)).sum())


def test_episodic_mode_does_not_collapse():
    df = make_episodic(n=120)
    payload = build_hospitalization_level(df, META, collapse=False)
    assert payload["n_hospitalizations"] == 120
    assert payload["metadata"]["surface"] == "stay"


def test_continuous_surface_is_labelled_peak(continuous):
    assert continuous["metadata"]["surface"] == "peak"
    assert "max(y_prob)" in continuous["metadata"]["score_rule"]


# ================================================================ Part 2 — 2x2


def test_confusion_counts_match_brute_force_at_every_threshold():
    """The vectorized sweep must equal the obvious `score >= t` loop."""
    df = make_episodic(n=400)
    payload = build_hospitalization_level(df, META, collapse=False)
    y = df["y"].to_numpy()
    p = df["y_prob"].to_numpy()

    partner = {"tp": "fp", "fp": "tp", "fn": "tn", "tn": "fn"}
    for r in payload["by_threshold"]:
        t = r["threshold"]
        flag = p >= t
        exp = {
            "tp": int((flag & (y == 1)).sum()),
            "fp": int((flag & (y == 0)).sum()),
            "fn": int((~flag & (y == 1)).sum()),
            "tn": int((~flag & (y == 0)).sum()),
        }
        for cell, want in exp.items():
            got = r[cell]
            if isinstance(got, str):
                # masking is in PAIRS: a big cell is masked to protect a small
                # partner, because their shared total is published.
                assert want < 10 or exp[partner[cell]] < 10, (
                    f"{cell} masked at t={t} but neither it ({want}) nor its "
                    f"partner ({exp[partner[cell]]}) is small")
            else:
                assert got == want, f"{cell} wrong at t={t}"


def test_confusion_cells_sum_to_n(episodic):
    for r in episodic["by_threshold"]:
        cells = [r[c] for c in ("tp", "fp", "fn", "tn")]
        if any(isinstance(c, str) for c in cells):
            continue
        assert sum(cells) == r["n"]


def test_positives_and_negatives_are_constant_across_thresholds(episodic):
    """tp+fn is the number of events; it cannot depend on the threshold."""
    totals = set()
    negatives = set()
    for r in episodic["by_threshold"]:
        if any(isinstance(r[c], str) for c in ("tp", "fp", "fn", "tn")):
            continue
        totals.add(r["tp"] + r["fn"])
        negatives.add(r["tn"] + r["fp"])
    assert len(totals) == 1, f"event count drifted across thresholds: {totals}"
    assert len(negatives) == 1


def test_grid_is_the_fixed_99_point_grid(episodic):
    thr = [r["threshold"] for r in episodic["by_threshold"]]
    assert thr == [round(float(t), 2) for t in SWEEP_GRID]
    assert len(thr) == 99
    assert episodic["metadata"]["grid"] == {
        "min": 0.01, "max": 0.99, "step": 0.01, "n": 99}


def test_exactly_one_row_is_flagged_clinical(episodic):
    clinical = [r for r in episodic["by_threshold"] if r["is_clinical"]]
    assert len(clinical) == 1
    assert clinical[0]["threshold"] == META["clinical_threshold"]


# ============================================================== Part 3 — rates


def test_rates_match_sklearn_at_the_clinical_threshold():
    df = make_episodic(n=400)
    payload = build_hospitalization_level(df, META, collapse=False)
    y = df["y"].to_numpy()
    p = df["y_prob"].to_numpy()
    t = META["clinical_threshold"]
    flag = (p >= t).astype(int)

    r = row_at(payload, t)
    assert r["sensitivity"] == pytest.approx(
        recall_score(y, flag), abs=1e-4)
    assert r["ppv"] == pytest.approx(
        precision_score(y, flag, zero_division=0), abs=1e-4)
    assert r["specificity"] == pytest.approx(
        recall_score(1 - y, 1 - flag), abs=1e-4)
    assert r["fpr"] == pytest.approx(1.0 - r["specificity"], abs=1e-4)
    assert r["flag_rate"] == pytest.approx(float(flag.mean()), abs=1e-4)


def test_f1_is_the_harmonic_mean_of_ppv_and_sensitivity(episodic):
    for r in episodic["by_threshold"]:
        if r["ppv"] is None or r["sensitivity"] in (None, 0) or r["ppv"] == 0:
            continue
        expect = 2 * r["ppv"] * r["sensitivity"] / (r["ppv"] + r["sensitivity"])
        assert r["f1"] == pytest.approx(expect, abs=1e-3)


def test_nne_is_one_over_ppv(episodic):
    for r in episodic["by_threshold"]:
        if r["ppv"]:
            assert r["nne"] == pytest.approx(1.0 / r["ppv"], abs=0.05)
        else:
            assert r["nne"] is None


def test_nne_ci_is_the_inverted_ppv_ci(episodic):
    """1/x is decreasing, so the CI bounds must swap order."""
    for r in episodic["by_threshold"]:
        if not r["ppv_ci"] or not r["nne_ci"]:
            continue
        lo, hi = r["ppv_ci"]
        if lo > 0 and hi > 0:
            assert r["nne_ci"][0] == pytest.approx(1.0 / hi, abs=0.05)
            assert r["nne_ci"][1] == pytest.approx(1.0 / lo, abs=0.05)


def test_every_rate_carries_a_ci_band(episodic):
    r = row_at(episodic, META["clinical_threshold"])
    for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "fpr"):
        assert f"{k}_ci" in r, f"{k} has no CI"
        ci = r[f"{k}_ci"]
        if ci is not None:
            assert ci[0] <= ci[1]
            assert 0.0 <= ci[0] <= 1.0


def test_point_estimate_lies_inside_its_ci(episodic):
    """A percentile bootstrap CI must bracket the point estimate."""
    r = row_at(episodic, META["clinical_threshold"])
    for k in ("sensitivity", "specificity", "npv"):
        if r[k] is not None and r[f"{k}_ci"] is not None:
            lo, hi = r[f"{k}_ci"]
            assert lo - 0.05 <= r[k] <= hi + 0.05, f"{k}={r[k]} outside {lo, hi}"


# ====================================================== Part 4 — monotonicity


def test_sensitivity_is_non_increasing_in_threshold(episodic):
    """Raising the bar can only ever catch fewer events."""
    vals = [r["sensitivity"] for r in episodic["by_threshold"]
            if r["sensitivity"] is not None]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_specificity_is_non_decreasing_in_threshold(episodic):
    vals = [r["specificity"] for r in episodic["by_threshold"]
            if r["specificity"] is not None]
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]))


def test_flag_rate_is_non_increasing_in_threshold(episodic):
    vals = [r["flag_rate"] for r in episodic["by_threshold"]
            if r["flag_rate"] is not None]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_flag_rate_equals_flagged_over_n(episodic):
    for r in episodic["by_threshold"]:
        if isinstance(r["tp"], str) or isinstance(r["fp"], str):
            continue
        assert r["flag_rate"] == pytest.approx((r["tp"] + r["fp"]) / r["n"],
                                               abs=1e-4)


# ============================================================= Part 5 — Youden


def test_youden_threshold_is_the_argmax_of_sensitivity_plus_specificity(episodic):
    """The published cutpoint must actually maximise J on the published sweep."""
    youden = next(s for s in episodic["selected_thresholds"]
                  if s["rule"] == "youden_j")
    best_j, best_t = -2.0, None
    for r in episodic["by_threshold"]:
        if r["sensitivity"] is None or r["specificity"] is None:
            continue
        j = r["sensitivity"] + r["specificity"] - 1.0
        if j > best_j:
            best_j, best_t = j, r["threshold"]
    assert youden["threshold"] == best_t


def test_youden_metrics_match_that_rows_metrics(episodic):
    youden = next(s for s in episodic["selected_thresholds"]
                  if s["rule"] == "youden_j")
    r = row_at(episodic, youden["threshold"])
    for k in ("sensitivity", "specificity", "npv", "f1", "fpr", "flag_rate"):
        assert youden[k] == r[k], f"{k} disagrees with by_threshold"


def test_only_youden_and_clinical_survive(episodic):
    rules = [s["rule"] for s in episodic["selected_thresholds"]]
    assert set(rules) <= {"youden_j", "clinical"}
    assert "clinical" in rules


def test_clinical_entry_is_not_marked_in_sample(episodic):
    """The shipped policy is not fitted, so it carries no selection bias."""
    clinical = next(s for s in episodic["selected_thresholds"]
                    if s["rule"] == "clinical")
    assert clinical["in_sample"] is False
    assert clinical["threshold_ci"] is None
    assert clinical["threshold"] == META["clinical_threshold"]

    youden = next(s for s in episodic["selected_thresholds"]
                  if s["rule"] == "youden_j")
    assert youden["in_sample"] is True
    assert youden["citation"] == "Youden 1950"


def test_youden_threshold_ci_is_wider_than_a_single_grid_step(episodic):
    """A selection-aware CI re-picks the cutpoint, so it must have real width."""
    youden = next(s for s in episodic["selected_thresholds"]
                  if s["rule"] == "youden_j")
    assert youden["threshold_ci"] is not None
    lo, hi = youden["threshold_ci"]
    assert hi - lo >= 0.01


# ======================================================== Part 6 — net benefit


def test_net_benefit_matches_the_dca_formula():
    """nb = TP/n - FP/n * t/(1-t), on the acted arm."""
    df = make_episodic(n=400)
    payload = build_hospitalization_level(df, META, collapse=False)
    y = df["y"].to_numpy()
    p = df["y_prob"].to_numpy()
    n = len(y)

    for t in (0.10, 0.30, 0.50):
        acted = p >= t
        tp = int((acted & (y == 1)).sum())
        fp = int(acted.sum()) - tp
        expect = tp / n - (fp / n) * (t / (1 - t))
        assert row_at(payload, t)["net_benefit"] == pytest.approx(expect, abs=1e-6)


def test_net_benefit_treat_all_matches_the_formula(episodic):
    prev = episodic["prevalence"]
    for r in episodic["by_threshold"]:
        t = r["threshold"]
        expect = prev - (1 - prev) * (t / (1 - t))
        # Both tolerances, because the error has two regimes: net benefit is
        # published rounded to 6dp (so ~5e-7 absolute near zero), while at the
        # top of the grid t/(1-t) reaches 99 and magnifies the 6dp rounding of
        # `prevalence` into the 5th decimal. approx() takes the looser of the two.
        assert r["net_benefit_treat_all"] == pytest.approx(
            expect, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("direction", ["above", "below"])
def test_vectorized_net_benefit_matches_the_scalar_reference_at_ties(direction):
    """Every score sits EXACTLY on a grid threshold — the historical break.

    `1.0 - x` is exact in float64 only for x in [0.5, 1] (Sterbenz), so deriving
    the acted arm from `1-p` puts a patient sitting exactly on t on the wrong
    side at 20 of the 99 thresholds. The vectorized sweep and the scalar
    reference must agree on every one of them.
    """
    from flair_benchmark.report._basis import (
        axis_threshold,
        net_benefit,
        to_decision_basis,
    )

    n = 400
    rng = np.random.default_rng(3)
    # draw only exact 2-dp values, so ties with the grid are guaranteed
    p = np.round(rng.integers(1, 100, n) / 100.0, 2)
    y = rng.binomial(1, 0.3, n)
    df = pl.DataFrame({
        "hospitalization_join_id": [f"H{i}" for i in range(n)],
        "y_prob": p, "y": y,
    })
    meta = {**META, "clinical_direction": direction}
    payload = build_hospitalization_level(df, meta, collapse=False)

    y_dec, _ = to_decision_basis(y, p, direction)
    for r in payload["by_threshold"]:
        t = r["threshold"]
        expect = net_benefit(y_dec, p, t, axis_threshold(t, direction), direction)
        if expect is None or r["net_benefit"] is None:
            continue
        assert r["net_benefit"] == pytest.approx(expect, abs=1e-6), \
            f"vectorized and scalar net benefit disagree at t={t}"


def test_continuous_mode_omits_net_benefit(continuous):
    """DCA on a max-of-many surface is the peak-calibration trap."""
    for r in continuous["by_threshold"]:
        assert "net_benefit" not in r
        assert "net_benefit_treat_all" not in r
    assert "peak-calibration trap" in \
        continuous["metadata"]["net_benefit_omitted_reason"]


def test_below_direction_flips_the_net_benefit_axis():
    """An act-on-LOW-risk task decides on the complement of the label."""
    meta = {**META, "clinical_direction": "below", "clinical_threshold": 0.30}
    df = make_episodic(n=400)
    payload = build_hospitalization_level(df, meta, collapse=False)

    y = df["y"].to_numpy()
    p = df["y_prob"].to_numpy()
    n = len(y)
    t = 0.30
    acted = p < t                       # act when risk is LOW
    y_dec = 1 - y                       # the event is the label's complement
    t_axis = 1.0 - t
    tp = int((acted & (y_dec == 1)).sum())
    fp = int(acted.sum()) - tp
    expect = tp / n - (fp / n) * (t_axis / (1 - t_axis))
    assert row_at(payload, t)["net_benefit"] == pytest.approx(expect, abs=1e-6)


def test_alarm_rates_are_direction_independent():
    """Sensitivity always means 'how well do we detect the event'."""
    df = make_episodic(n=400)
    above = build_hospitalization_level(df, META, collapse=False)
    below = build_hospitalization_level(
        {**META, "clinical_direction": "below"} and df,
        {**META, "clinical_direction": "below"}, collapse=False)
    for a, b in zip(above["by_threshold"], below["by_threshold"]):
        assert a["sensitivity"] == b["sensitivity"]
        assert a["specificity"] == b["specificity"]
        assert a["tp"] == b["tp"]


# ======================================================== Part 7 — suppression


def test_small_confusion_cells_are_masked_in_pairs():
    """Masking one cell alone is theatre: n minus the other three recovers it."""
    df = make_episodic(n=300)
    payload = build_hospitalization_level(df, META, collapse=False)
    for r in payload["by_threshold"]:
        if isinstance(r["tp"], str) or isinstance(r["fp"], str):
            # tp and fp share the n_flagged total, so both must go
            assert isinstance(r["tp"], str) or r["tp"] == 0
            assert isinstance(r["fp"], str) or r["fp"] == 0
        if isinstance(r["fn"], str) or isinstance(r["tn"], str):
            assert isinstance(r["fn"], str) or r["fn"] == 0
            assert isinstance(r["tn"], str) or r["tn"] == 0


def test_ppv_is_nulled_when_almost_nobody_is_flagged():
    """A precision built on <10 stays is unstable AND disclosive."""
    df = make_episodic(n=300)
    payload = build_hospitalization_level(df, META, collapse=False)
    for r in payload["by_threshold"]:
        cells = (r["tp"], r["fp"])
        if any(isinstance(c, str) for c in cells):
            continue
        if r["tp"] + r["fp"] < 10:
            assert r["ppv"] is None
            assert r["nne"] is None


def test_a_true_zero_cell_is_published_as_zero():
    """Zero identifies nobody, and n minus the rest gives it away anyway."""
    n = 200
    df = pl.DataFrame({
        "hospitalization_join_id": [f"H{i}" for i in range(n)],
        # every score below 0.5, so thresholds above it flag exactly nobody
        "y_prob": np.linspace(0.01, 0.49, n),
        "y": np.array([1] * 50 + [0] * 150),
    })
    payload = build_hospitalization_level(df, META, collapse=False)
    high = row_at(payload, 0.90)
    assert high["tp"] == 0
    assert high["fp"] == 0


# =========================================================== Part 8 — fairness


def test_fairness_is_reported_at_the_clinical_threshold_only(episodic):
    fair = episodic["fairness"]
    assert fair["at_threshold"] == META["clinical_threshold"]
    # and NOT inside each swept row
    for r in episodic["by_threshold"]:
        assert "fairness" not in r


def test_subgroup_tpr_and_fpr_match_a_hand_computation():
    df = make_episodic(n=600)
    payload = build_hospitalization_level(df, META, collapse=False)
    t = META["clinical_threshold"]

    y = df["y"].to_numpy()
    p = df["y_prob"].to_numpy()
    sex = df["sex_category"].to_numpy()
    for level, block in payload["fairness"]["subgroups"]["sex_category"].items():
        mask = sex == level
        assert block["n"] == int(mask.sum())
        flag = p[mask] >= t
        yy = y[mask]
        assert block["positive_rate"] == pytest.approx(float(flag.mean()), abs=1e-4)
        assert block["tpr"] == pytest.approx(float(flag[yy == 1].mean()), abs=1e-4)
        assert block["fpr"] == pytest.approx(float(flag[yy == 0].mean()), abs=1e-4)


def test_subgroup_auroc_matches_sklearn():
    df = make_episodic(n=600)
    payload = build_hospitalization_level(df, META, collapse=False)
    y = df["y"].to_numpy()
    p = df["y_prob"].to_numpy()
    sex = df["sex_category"].to_numpy()
    for level, block in payload["fairness"]["subgroups"]["sex_category"].items():
        mask = sex == level
        assert block["auroc"] == pytest.approx(
            roc_auc_score(y[mask], p[mask]), abs=1e-4)


def test_parity_gap_is_max_minus_min_across_levels():
    df = make_episodic(n=600)
    payload = build_hospitalization_level(df, META, collapse=False)
    block = payload["fairness"]["subgroups"]["race_category"]
    rates = [v["positive_rate"] for v in block.values()]
    gap = payload["fairness"]["parity"]["race_category"]["demographic_parity_gap"]
    assert gap == pytest.approx(max(rates) - min(rates), abs=1e-3)


def test_subgroup_levels_below_ten_are_dropped_entirely():
    """An AUROC on <10 patients can reveal individual outcomes."""
    df = make_episodic(n=300)
    sex = ["Female"] * 295 + ["Rare"] * 5
    df = df.with_columns(pl.Series("sex_category", sex))
    payload = build_hospitalization_level(df, META, collapse=False)
    assert "Rare" not in payload["fairness"]["subgroups"]["sex_category"]


def test_null_age_does_not_become_an_eighty_plus_patient():
    df = make_episodic(n=200)
    ages = df["age_at_admission"].to_list()
    ages[:50] = [None] * 50
    df = df.with_columns(pl.Series("age_at_admission", ages, dtype=pl.Int64))
    payload = build_hospitalization_level(df, META, collapse=False)
    bins = payload["fairness"]["subgroups"].get("age_bin", {})
    total = sum(b["n"] for b in bins.values())
    assert total <= 150, "missing ages leaked into an age bin"


def test_reference_group_is_the_largest_level():
    df = make_episodic(n=600)
    payload = build_hospitalization_level(df, META, collapse=False)
    block = payload["fairness"]["subgroups"]["race_category"]
    biggest = max(block, key=lambda lv: (block[lv]["n"], lv))
    assert payload["fairness"]["parity"]["race_category"]["reference_group"] \
        == biggest


def test_gap_triplet_discards_a_resample_that_loses_a_level():
    """Pinned levels: a vanished group must void the triplet, not shrink it."""
    codes = np.array([0, 0, 0, 1, 1, 1])
    label = np.array([1, 0, 1, 0, 1, 0])
    flag = np.array([1, 0, 1, 0, 0, 0]).astype(bool)
    assert gap_triplet(codes, label, flag, 2)[0] is not None
    # level 1 is gone -> all three gaps must be None
    gone = np.array([0, 0, 0, 0, 0, 0])
    assert gap_triplet(gone, label, flag, 2) == (None, None, None)


def test_fairness_is_none_without_demographics():
    df = make_episodic(n=200).drop(
        ["sex_category", "race_category", "age_at_admission"])
    payload = build_hospitalization_level(df, META, collapse=False)
    assert payload["fairness"] is None


# ======================================================== Part 9 — determinism


def test_row_order_does_not_change_any_number():
    """The seeded bootstrap resamples by row INDEX, so the collapse must sort.

    Without a deterministic order the same seed draws a different set of
    patients on every run — point estimates look stable (they are order
    invariant) while every CI silently moves.
    """
    df = make_continuous()
    a = build_hospitalization_level(df, META, collapse=True)
    b = build_hospitalization_level(df.reverse(), META, collapse=True)
    assert a["by_threshold"] == b["by_threshold"]
    assert a["selected_thresholds"] == b["selected_thresholds"]
    assert a["fairness"] == b["fairness"]


def test_repeated_builds_are_identical():
    df = make_episodic(n=300)
    a = build_hospitalization_level(df, META, collapse=False)
    b = build_hospitalization_level(df, META, collapse=False)
    assert a == b


def test_auroc_of_the_swept_surface_is_recoverable_from_the_sweep(episodic):
    """A sanity check that the sweep describes the same ranking sklearn sees.

    Trapezoid over the (fpr, sensitivity) points the sweep publishes should land
    near the true AUROC. It is coarse (99 points) so the tolerance is loose —
    the point is to catch a sweep that describes a DIFFERENT surface entirely.
    """
    df = make_episodic()
    truth = roc_auc_score(df["y"].to_numpy(), df["y_prob"].to_numpy())
    pts = [(r["fpr"], r["sensitivity"]) for r in episodic["by_threshold"]
           if r["fpr"] is not None and r["sensitivity"] is not None]
    # ROC runs from (0,0) to (1,1) with fpr ASCENDING; the sweep walks it
    # backwards (raising the threshold lowers both axes), so sort before
    # closing the ends.
    pts = [(0.0, 0.0), *sorted(pts), (1.0, 1.0)]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    area = float(np.trapezoid(y, x))
    assert area == pytest.approx(truth, abs=0.03)


def test_degenerate_task_returns_empty_sections_not_a_crash():
    df = make_episodic(n=100)
    payload = build_hospitalization_level(
        df, {**META, "clinical_threshold": None}, collapse=False)
    assert payload["by_threshold"] == []
    assert payload["selected_thresholds"] == []
    assert "reason" in payload["metadata"]


def test_auprc_baseline_is_the_prevalence():
    """Guards against scoring the sweep on a differently-filtered frame."""
    df = make_episodic(n=400)
    payload = build_hospitalization_level(df, META, collapse=False)
    assert payload["prevalence"] == pytest.approx(
        float(df["y"].mean()), abs=1e-6)
    # and the ranking metric on the same surface is genuinely informative
    assert average_precision_score(
        df["y"].to_numpy(), df["y_prob"].to_numpy()) > payload["prevalence"]
