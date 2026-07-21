"""Synthetic-data tests for the episodic report bundle (no CLIF data needed).

Episodic = one prediction per stay. Five builders, each receiving an
already-split-filtered, one-row-per-stay frame and emitting JSON with raw curve
arrays + 95% bootstrap CIs.

Four are THRESHOLD-FREE (discrimination, calibration, dca, fairness); the fifth,
operating_points, owns every number that moves when the decision threshold moves
and reports it at five thresholds (clinical ± 0.05 / ± 0.10).

NOTE: the pillar builders keep their threshold blocks by DEFAULT — the continuous
report's peak and landmark sections rely on that. It is build_episodic_report that
turns them off. So the direct-call tests below still see an `operating_point` on
build_discrimination; the end-to-end episodic bundle does not.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from flair_benchmark.tasks import get_task, list_tasks
from flair_benchmark.report._calib import calibration_extended
from flair_benchmark.report._ci import bootstrap_ci
from flair_benchmark.report.calibration import build_calibration
from flair_benchmark.report.dca import build_dca
from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.fairness import build_fairness
from flair_benchmark.report._sweep import SWEEP_GRID
from flair_benchmark.report._locators import _CI_KEYS, _attach_ci, locate_band
from flair_benchmark.report.operating_points import build_episodic_operating_points
from flair_benchmark.tasks._finalize import finalize_cohort

META = {
    "name": "synthetic_episodic_task",
    "display_name": "Synthetic Episodic Task",
    "task_class": "outcome",
    "temporal_pattern": "episode",
    "prediction_unit": "one-per-stay",
    "task_type": "binary",
    "label_column": "label",
    "clinical_threshold": 0.20,
    "clinical_action": "intervene",
    "clinical_direction": "above",
}


def make_synth(n_stays: int = 600, seed: int = 0,
               calibrated: bool = True) -> pl.DataFrame:
    """One row per stay: demographics + y_prob correlated with the label."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        adm = datetime(int(rng.integers(2017, 2024)), int(rng.integers(2, 12)),
                       int(rng.integers(1, 28)), 7, 0)
        latent = rng.normal()
        logit = -2.2 + 1.6 * latent + rng.normal(scale=0.3)
        prob = 1 / (1 + np.exp(-logit))
        label = int(rng.random() < prob) if calibrated else int(latent > 0.8)
        rows.append({
            "patient_id": f"p{i}",
            "hospitalization_join_id": f"h{i}",
            "hospitalization_id": f"h{i}",
            "prediction_dttm": adm,
            "label": label,
            "split": "test",
            "age_at_admission": int(rng.integers(18, 95)),
            "sex_category": "Female" if rng.random() < 0.5 else "Male",
            "race_category": "White",
            "ethnicity_category": "Non-Hispanic",
            "y_prob": float(prob),
            "y_pred": int(prob >= 0.5),
        })
    df = pl.DataFrame(rows)
    output_cols = ["hospitalization_id", "hospitalization_join_id",
                   "prediction_dttm", "label", "split",
                   "age_at_admission", "sex_category", "race_category",
                   "ethnicity_category", "y_prob", "y_pred"]
    return finalize_cohort(df, output_cols, META)


# ---------------------------------------------------------------- finalize


def test_prediction_id_one_per_stay():
    df = make_synth(n_stays=20)
    assert df.height == df["hospitalization_join_id"].n_unique() == 20
    one = df.filter(pl.col("hospitalization_join_id") == "h0")
    assert one["window_number"][0] == 1
    assert one["prediction_id"][0] == "h0-h0-p0-1"


# ---------------------------------------------------------------- bootstrap CI


def test_bootstrap_ci_orders_and_handles_degenerate():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    p = rng.random(500)
    out = bootstrap_ci(y, p, {"mean_prob": lambda yt, yp: float(yp.mean())})
    assert out["mean_prob"] is not None
    lo, hi = out["mean_prob"]
    assert lo <= hi
    # a statistic that is always undefined yields a None CI, never a crash
    none_out = bootstrap_ci(y, p, {"always_none": lambda yt, yp: None})
    assert none_out["always_none"] is None


# ---------------------------------------------------------------- discrimination


def test_discrimination_curves_and_ci():
    d = build_discrimination(make_synth(n_stays=1000), META)
    assert d["task_type"] == "binary"
    m, roc = d["metrics"], d["curves"]["roc"]
    assert len(roc["fpr"]) == len(roc["tpr"]) <= 500
    assert len(roc["grid_fpr"]) == len(roc["tpr_at_grid"]) == 101
    assert 0.6 < m["auroc"] <= 1.0
    assert m["auroc_ci"][0] <= m["auroc"] <= m["auroc_ci"][1]
    pr = d["curves"]["pr"]
    assert len(pr["recall"]) == len(pr["precision"]) <= 500
    assert m["auprc"] is not None and m["auprc_ci"][0] <= m["auprc_ci"][1]
    assert m["auprc_lift"] > 1.0


def test_discrimination_operating_point_at_clinical_threshold():
    d = build_discrimination(make_synth(n_stays=1000), META)
    op = d["metrics"]["operating_point"]
    assert op["threshold"] == 0.20 and op["direction"] == "above"
    for k in ("sensitivity", "specificity", "ppv", "npv", "f1"):
        assert op[k] is not None
        assert op[f"{k}_ci"][0] <= op[k] <= op[f"{k}_ci"][1]
    # confusion counts reconstruct the cohort size
    assert op["tp"] + op["fp"] + op["fn"] + op["tn"] == 1000
    assert 0 <= op["flag_rate"] <= 1


# ---------------------------------------------------------------- calibration


def test_calibration_well_calibrated_with_ci():
    c = build_calibration(make_synth(n_stays=3000, calibrated=True), META)
    m = c["metrics"]
    assert abs(m["slope"] - 1.0) < 0.2
    assert abs(m["intercept"]) < 0.2
    assert abs(m["o_e_ratio"] - 1.0) < 0.1
    assert 0 < m["brier"] < 0.15
    assert 0 <= m["ece"] < 0.05
    # CIs present and ordered
    for k in ("ece", "brier", "slope", "intercept"):
        lo, hi = m[f"{k}_ci"]
        assert lo <= m[k] <= hi
    assert len(c["curves"]["reliability"]["bin_counts"]) == 10
    assert c["curves"]["smoothed"]["method"] == "linear_fit"


def test_calibration_non_binary_reason():
    meta = {**META, "task_type": "regression"}
    c = build_calibration(make_synth(n_stays=50), meta)
    assert "reason" in c["metadata"]


def test_calibration_extended_still_works():
    df = make_synth(n_stays=2000, calibrated=True)
    ext = calibration_extended(df["label"].to_numpy().astype(float),
                               df["y_prob"].to_numpy())
    assert abs(ext["slope"] - 1.0) < 0.2
    assert ext["smoothed"]["method"] == "linear_fit"
    assert len(ext["smoothed"]["pred_grid"]) == 50


# ---------------------------------------------------------------- DCA


def test_dca_good_model_useful_range_and_clinical_ci():
    dca = build_dca(make_synth(n_stays=2000), META)
    m, curves = dca["metrics"], dca["curves"]
    assert len(curves["thresholds"]) == 164  # dense tail at BOTH ends + 0.01-0.99
    assert m["useful_range"] is not None
    assert 0 < m["useful_range"]["low"] < m["useful_range"]["high"] < 1
    nb = m["net_benefit_at_clinical_threshold"]
    assert nb["model"] is not None
    assert nb["model_ci"][0] <= nb["model"] <= nb["model_ci"][1]
    assert m["clinical_threshold"] == 0.20


def test_dca_suppresses_sparse_thresholds():
    dca = build_dca(make_synth(n_stays=120), META)
    curves = dca["curves"]
    for t, nb in zip(curves["thresholds"], curves["net_benefit_model"]):
        if t in curves["suppressed_thresholds"]:
            assert nb is None


# ---------------------------------------------------------------- fairness


def test_fairness_at_clinical_threshold_with_ci():
    f = build_fairness(make_synth(n_stays=1500), META)
    assert f["metadata"]["operating_threshold"] == 0.20
    sex = f["subgroups"]["sex_category"]
    assert set(sex) == {"Male", "Female"}
    for m in sex.values():
        assert m["n"] >= 10 and "brier" in m
        assert m["auroc_ci"][0] <= m["auroc"] <= m["auroc_ci"][1]
    parity = f["parity"]["sex_category"]
    assert parity["reference_group"] in ("Male", "Female")
    for gap in ("demographic_parity_gap", "equalized_odds_tpr_gap",
                "equalized_odds_fpr_gap"):
        assert 0 <= parity[gap] <= 1
        ci = parity[f"{gap}_ci"]
        assert ci is None or ci[0] <= ci[1]


def test_fairness_reference_group_tiebreak_deterministic():
    """An exact n-tie between subgroups must resolve the SAME way every run.

    reference_group = max(levels, key=n); with equal n, max() returns whichever
    key comes first in dict order — and polars group_by is hash-ordered (varies
    across process runs). The (n, label) tie-break makes the pick order-invariant.
    Regression for the serial-vs-parallel reference_group flip.
    """
    df = make_synth(n_stays=400, seed=1)
    # force an exact Male/Female count tie (200 each).
    sexes = ["Male"] * 200 + ["Female"] * 200
    df = df.with_columns(pl.Series("sex_category", sexes))
    refs = set()
    for seed in range(6):                      # shuffle rows -> different group order
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
        f = build_fairness(shuffled, META)
        refs.add(f["parity"]["sex_category"]["reference_group"])
    assert refs == {"Male"}                    # lexicographic max wins the tie, always


def test_fairness_overall_block_and_threshold():
    f = build_fairness(make_synth(n_stays=1500), META)
    # threshold recorded in metadata
    assert f["metadata"]["threshold"] == 0.20
    assert f["metadata"]["flag_rule"] == "y_prob >= threshold"
    # cohort-wide operating point at the clinical threshold
    ov = f["overall"]
    assert ov["threshold"] == 0.20 and ov["n"] == 1500
    for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy",
              "flag_rate", "prevalence"):
        assert k in ov
    assert ov["sensitivity_ci"][0] <= ov["sensitivity"] <= ov["sensitivity_ci"][1]


def test_fairness_overall_without_demographics():
    # drop demographics -> no subgroups, but the overall block must still populate
    df = make_synth(n_stays=400).drop(
        ["age_at_admission", "sex_category", "race_category", "ethnicity_category"])
    f = build_fairness(df, META)
    assert f["subgroups"] == {}
    assert f["overall"]["n"] == 400 and "f1" in f["overall"]


# ------------------------------------------------- operating-points scorecard


def test_sweep_grid_is_fixed_and_open_interval():
    """The universal sweep is a fixed 0.01-step grid in the OPEN interval (0, 1).

    At t=0, `y_prob >= 0` flags everyone; at t=1 it flags nobody, PPV is undefined,
    and DCA's t/(1-t) odds diverge. The fixed 0.01..0.99 grid never emits either.
    """
    assert len(SWEEP_GRID) == 99
    assert SWEEP_GRID[0] == 0.01 and SWEEP_GRID[-1] == 0.99
    assert all(0.0 < float(t) < 1.0 for t in SWEEP_GRID)
    # consecutive step is exactly 0.01 (rounded, no float drift)
    assert all(round(float(b - a), 2) == 0.01
               for a, b in zip(SWEEP_GRID[:-1], SWEEP_GRID[1:]))


@pytest.mark.parametrize("task_name", sorted(list_tasks()))
def test_every_clinical_threshold_lands_on_the_sweep_grid(task_name):
    """The shipped policy must be exactly one row of the fixed grid (is_clinical).

    A threshold off the 0.01 grid would have no is_clinical row; every shipped
    threshold is a multiple of 0.01, and tasks._validate_taxonomy now enforces it.
    """
    t0 = get_task(task_name).META["clinical_threshold"]
    on_grid = [t for t in SWEEP_GRID if abs(float(t) - round(t0, 2)) < 1e-9]
    assert len(on_grid) == 1, f"{task_name}: clinical threshold {t0} is off the grid"


def test_scorecard_clinical_entry_matches_the_old_operating_point():
    """THE regression test for the move.

    The clinical row of the scorecard must reproduce, exactly, the operating point
    the report used to publish inside discrimination.json. If this drifts, sites'
    already-published sensitivity/PPV silently changed meaning.
    """
    df = make_synth(n_stays=900, seed=3)
    op = build_discrimination(df, META)["metrics"]["operating_point"]
    card = build_episodic_operating_points(df, META)
    clinical = next(e for e in card["by_threshold"] if e["is_clinical"])

    assert clinical["threshold"] == op["threshold"] == 0.20
    for k in ("tp", "fp", "fn", "tn"):
        assert clinical["confusion"][k] == op[k]
    for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "flag_rate"):
        assert clinical["metrics"][k] == op[k]
        if f"{k}_ci" in op:
            assert clinical["metrics"][f"{k}_ci"] == op[f"{k}_ci"]
    # the decision block absorbs the old action block verbatim (and adds the DCA half)
    for k, v in op["action"].items():
        assert clinical["decision"][k] == v


def test_data_driven_thresholds_carry_selection_aware_ci():
    """Every locator gains a CI; the fitted rules' is selection-aware, clinical's is fixed.

    A data-driven cutpoint is chosen ON this data, so its honest interval must include
    the wobble of the cutpoint itself (threshold_ci) and the re-selection variance in
    its rates. The shipped clinical policy is not fitted, so it reuses the plain
    fixed-threshold by_threshold band and carries no threshold_ci.
    """
    df = make_synth(n_stays=1200, seed=7)
    card = build_episodic_operating_points(df, META)
    ddt = card["data_driven_thresholds"]
    assert "data_driven_ci" in card["metadata"]

    fitted = [loc for loc in ddt
              if loc["rule"] != "clinical" and loc.get("threshold") is not None]
    assert fitted, "expected at least one fitted rule"
    for loc in fitted:
        tci = loc["threshold_ci"]
        if tci is not None:                       # None only if the rule was too unstable
            lo, hi = tci
            assert lo <= hi
            assert lo <= loc["threshold"] <= hi   # the CI brackets the point cutpoint
        ma = loc["metrics_at"]
        for k in _CI_KEYS:
            assert f"{k}_ci" in ma
        assert "nne_ci" in ma
        assert "flag_rate_ci" not in ma           # flag_rate stays point-only

    # clinical entry: fixed band == the by_threshold row at the clinical index, no threshold_ci
    clinical = next(loc for loc in ddt if loc["rule"] == "clinical")
    assert clinical["threshold_ci"] is None
    ci_idx = card["metadata"]["clinical_index"]
    band_row = card["by_threshold"][ci_idx]["metrics"]
    for k in _CI_KEYS:
        assert clinical["metrics_at"][f"{k}_ci"] == band_row[f"{k}_ci"]

    # selection-aware width >= the fixed band at the same threshold (honest: wider or equal)
    yj = next(loc for loc in fitted if loc["rule"] == "youden_j")
    idx = int(round(yj["threshold"] * 100)) - 1
    fixed = card["by_threshold"][idx]["metrics"]["sensitivity_ci"]
    sel = yj["metrics_at"]["sensitivity_ci"]
    if fixed is not None and sel is not None:
        assert (sel[1] - sel[0]) >= (fixed[1] - fixed[0]) - 1e-9


def test_locate_band_is_deterministic_and_covers_every_rule():
    """Shared BOOT_SEED -> byte-identical bands on rerun, one entry per rule."""
    df = make_synth(n_stays=800, seed=2)
    label = (df["label"].to_numpy() > 0).astype(int)
    prob = df["y_prob"].to_numpy().astype(float)
    from flair_benchmark.report.discrimination import _auroc
    auc = _auroc(label, prob)
    a = locate_band(label, prob, SWEEP_GRID, auc, {})
    b = locate_band(label, prob, SWEEP_GRID, auc, {})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    for rule in ("youden_j", "f_beta_max", "ppv_target", "alert_rate_target"):
        assert rule in a
        assert set(a[rule]) == {"threshold", *_CI_KEYS}
    # empty input -> all-None band, never a crash
    empty = locate_band(np.array([]), np.array([]), SWEEP_GRID, None, {})
    assert empty["youden_j"] == {"threshold": None, **{c: None for c in _CI_KEYS}}


def test_locator_ci_suppressed_below_min_cell_count():
    """A cutpoint flagging < MIN_CELL_COUNT stays yields a disclosive precision -> null."""
    entry = {"metrics_at": {"ppv": 0.5, "nne": 2.0}}
    ci_map = {c: [0.1, 0.9] for c in _CI_KEYS}
    tp = np.zeros(10, dtype=int); fp = np.zeros(10, dtype=int)
    tp[5] = 3                                     # tp+fp at row 5 = 3 < 10
    _attach_ci(entry, 5, ci_map, [0.3, 0.4], (tp, fp))
    ma = entry["metrics_at"]
    assert entry["threshold_ci"] == [0.3, 0.4]
    assert ma["sensitivity_ci"] == [0.1, 0.9]     # non-precision rates keep their CI
    assert ma["ppv_ci"] is None and ma["nne_ci"] is None

    entry2 = {"metrics_at": {"ppv": 0.5, "nne": 2.0}}
    tp[5] = 40                                     # now well above the floor
    _attach_ci(entry2, 5, ci_map, [0.3, 0.4], (tp, fp))
    assert entry2["metrics_at"]["ppv_ci"] == [0.1, 0.9]


def test_scorecard_uses_alarm_convention_for_a_below_task():
    """A 'below' task acts on LOW risk, but the ALARM is still high risk.

    Scoring the confusion matrix on the action would make TP = "acted and the event
    happened" -- for task3, "extubated AND failed", i.e. a clinical disaster booked
    as a true positive, with a 'sensitivity' that RISES as the model gets worse. So
    sens/spec/PPV stay in risk-classifier convention (y_prob >= t) in both
    directions; only the `decision` block flips.
    """
    below = {**META, "clinical_direction": "below"}
    df = make_synth(n_stays=900, seed=3)
    card = build_episodic_operating_points(df, below)
    clinical = next(e for e in card["by_threshold"] if e["is_clinical"])

    # identical confusion to the "above" task: the alarm did not move.
    above = next(e for e in build_episodic_operating_points(df, META)["by_threshold"]
                 if e["is_clinical"])
    assert clinical["confusion"] == above["confusion"]
    assert clinical["metrics"]["sensitivity"] == above["metrics"]["sensitivity"]

    # only the decision flips: we now act on everyone BELOW the threshold...
    dec = clinical["decision"]
    assert dec["acts_on"] == "y_prob < threshold"
    n = clinical["confusion"]["n"]
    n_fired = clinical["confusion"]["tp"] + clinical["confusion"]["fp"]
    assert dec["n_acted"] == n - n_fired
    assert above["decision"]["acts_on"] == "y_prob >= threshold"
    # ...and its net benefit is read on the COMPLEMENT axis: 0.20 -> 0.80.
    assert dec["dca_threshold"] == 0.80
    assert above["decision"]["dca_threshold"] == 0.20
    assert dec["net_benefit"] != above["decision"]["net_benefit"]


def test_scorecard_net_benefit_matches_the_dca_curve():
    """The scorecard is not a second opinion on net benefit — it is the same curve.

    Both go through dca.to_decision_basis / dca.axis_threshold, so a drift here means
    one of the two files has started describing a different decision.
    """
    for direction in ("above", "below"):
        meta = {**META, "clinical_direction": direction}
        df = make_synth(n_stays=1200, seed=11)
        dca = build_dca(df, meta)
        thr = dca["curves"]["thresholds"]
        nb_model, nb_all = (dca["curves"]["net_benefit_model"],
                            dca["curves"]["net_benefit_all"])
        for e in build_episodic_operating_points(df, meta)["by_threshold"]:
            dec = e["decision"]
            i = min(range(len(thr)), key=lambda j: abs(thr[j] - dec["dca_threshold"]))
            # sparse tails null net benefit on their own arm (n_acted < 10); where it
            # IS defined it must be the SAME number as dca.json's curve.
            if dec["net_benefit"] is not None and nb_model[i] is not None:
                assert dec["net_benefit"] == nb_model[i], (direction, e["threshold"])
            assert dec["net_benefit_treat_all"] == nb_all[i], direction
            if e["is_clinical"]:
                v = dca["metrics"]["verdict"]
                assert dec["net_benefit"] == v["model"]
                assert dec["clinically_useful"] == v["clinically_useful"]


def test_scorecard_suppresses_a_threshold_that_flags_almost_nobody():
    df = make_synth(n_stays=400, seed=5)
    # the far-right tail of the full sweep flags almost nobody (t -> 0.99).
    card = build_episodic_operating_points(df, {**META, "clinical_threshold": 0.85})
    sparse = [e for e in card["by_threshold"] if e.get("suppressed")]
    assert sparse, "expected at least one near-empty threshold on the sweep tail"
    for e in sparse:
        # a PPV from <10 flagged stays is unstable and disclosive
        assert e["metrics"]["ppv"] is None
        assert isinstance(e["confusion"]["tp"], str)   # "<10"
        # net benefit is suppressed on its OWN arm (n_acted), which for an "above"
        # task is the flagged arm — so it is nulled here too.
        assert e["decision"]["net_benefit"] is None


def test_scorecard_non_binary_task_degrades_to_a_reason():
    df = make_synth(n_stays=200)
    card = build_episodic_operating_points(df, {**META, "task_type": "regression"})
    assert card["by_threshold"] == []
    assert "reason" in card["metadata"]


# ---------------------------------------------------------------- end-to-end


def _fake_task():
    return SimpleNamespace(META=META, __file__=__file__)


def test_build_report_episodic_end_to_end(tmp_path):
    df = make_synth()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "report"),
                         mode="episodic")
    jsons = {k: p for k, p in paths.items() if not k.endswith("_png")}
    assert set(jsons) == {"discrimination", "calibration", "dca", "fairness",
                          "operating_points"}
    for name, path in jsons.items():
        payload = json.loads(path.read_text())
        md = payload["metadata"]
        assert md["report_mode"] == "episodic"
        assert md["split"] == "test" and md["n_stays"] == df.height


def test_build_report_always_scores_test_only(tmp_path):
    # mark half the stays "train"; the report must ALWAYS analyze only "test"
    # (no split override exists) — train rows are never scored.
    df = make_synth(n_stays=400)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 200).then(pl.lit("train"))
        .otherwise(pl.lit("test")).alias("split"))
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "r"),
                         mode="episodic")
    disc = json.loads(paths["discrimination"].read_text())
    assert disc["metadata"]["n_stays"] == 200
    assert disc["metadata"]["split"] == "test"

    # A preds file with no 'test' rows is rejected (can't silently report on train).
    no_test = df.with_columns(pl.lit("train").alias("split"))
    no_test_path = tmp_path / "no_test.parquet"
    no_test.write_parquet(no_test_path)
    with pytest.raises(ValueError, match="test"):
        build_report(str(no_test_path), _fake_task(), str(tmp_path / "r2"),
                     mode="episodic")


def test_build_report_requires_prediction_id(tmp_path):
    # prediction_id is part of the always-required preds contract.
    df = make_synth(n_stays=50).drop("prediction_id")
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)
    from flair_benchmark.report import build_report
    with pytest.raises(ValueError, match="prediction_id"):
        build_report(str(preds_path), _fake_task(), str(tmp_path / "r"),
                     mode="episodic")


def test_build_report_requires_hospitalization_join_id(tmp_path):
    # hospitalization_join_id is the required report/evaluation unit; source
    # hospitalization_id is audit-only and cannot substitute for it.
    df = make_synth(n_stays=50).drop("hospitalization_join_id")
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)
    from flair_benchmark.report import build_report
    with pytest.raises(ValueError, match="hospitalization_join_id"):
        build_report(str(preds_path), _fake_task(), str(tmp_path / "r"),
                     mode="episodic")


def test_build_report_rejects_same_join_id_in_train_and_test(tmp_path):
    # The dangerous leakage case: one stitched encounter has rows in both train
    # and test. This must fail before the report filters to test rows.
    df = make_synth(n_stays=50)
    train_row = df.head(1).with_columns(pl.lit("train").alias("split"))
    bad = pl.concat([df, train_row])
    preds_path = tmp_path / "preds.parquet"
    bad.write_parquet(preds_path)
    from flair_benchmark.report import build_report
    with pytest.raises(ValueError, match="multiple splits"):
        build_report(str(preds_path), _fake_task(), str(tmp_path / "r"),
                     mode="episodic")


def test_build_report_continuous_mode_inferred(tmp_path):
    # A continuous task with no --mode now infers mode='continuous' (peak +
    # landmark merged); the horizon must come from META['landmark_horizon'].
    df = make_synth()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)
    from flair_benchmark.report import build_report

    cont_meta = {**META, "temporal_pattern": "continuous",
                 "prediction_unit": "one-per-window"}
    # without a landmark_horizon the continuous report cannot size its grid
    with pytest.raises(ValueError, match="landmark_horizon"):
        build_report(str(preds_path),
                     SimpleNamespace(META=cont_meta, __file__=__file__),
                     str(tmp_path / "r0"), viz=False)

    cont_meta = {**cont_meta, "landmark_horizon": 2}
    paths = build_report(str(preds_path),
                         SimpleNamespace(META=cont_meta, __file__=__file__),
                         str(tmp_path / "r1"), viz=False)
    disc = json.loads(paths["discrimination"].read_text())
    assert disc["metadata"]["report_mode"] == "continuous"
    assert "peak" in disc and len(disc["landmarks"]) == 2


def test_build_report_rejects_multi_row_per_stay(tmp_path):
    # duplicate every stay -> two rows per stay -> episodic guard must fire
    df = make_synth(n_stays=100)
    df = pl.concat([df, df])
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    with pytest.raises(ValueError, match="one row per stay"):
        build_report(str(preds_path), _fake_task(), str(tmp_path / "r"),
                     mode="episodic")


def test_build_report_cohort_join(tmp_path):
    df = make_synth()
    cohort_path = tmp_path / "cohort.parquet"
    df.write_parquet(cohort_path)
    # preds stripped of demographics, carrying prediction_id for the join
    preds = df.drop(["age_at_admission", "sex_category", "race_category",
                     "ethnicity_category"])
    preds_path = tmp_path / "preds.parquet"
    preds.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "report"),
                         cohort_path=str(cohort_path), mode="episodic")
    fairness = json.loads(paths["fairness"].read_text())
    assert "sex_category" in fairness["subgroups"]


def test_viz_renders_episodic_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    df = make_synth()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "report"),
                         mode="episodic", viz=True)
    pngs = {p.name for p in paths.values() if p.suffix == ".png"}
    assert pngs == {"roc_pr.png", "calibration.png", "dca.png", "fairness_auroc.png",
                    "operating_points.png"}
    assert all(p.exists() for p in paths.values() if p.suffix == ".png")
