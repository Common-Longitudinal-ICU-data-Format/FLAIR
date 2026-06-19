"""Synthetic-data tests for the episodic report bundle (no CLIF data needed).

Episodic = one prediction per stay. The four builders (discrimination,
calibration, dca, fairness) each receive an already-split-filtered, one-row-per-
stay frame and emit JSON with raw curve arrays + 95% bootstrap CIs.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from flair_benchmark.report._calib import calibration_extended
from flair_benchmark.report._ci import bootstrap_ci
from flair_benchmark.report.calibration import build_calibration
from flair_benchmark.report.dca import build_dca
from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.fairness import build_fairness
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
    output_cols = ["hospitalization_id", "prediction_dttm", "label", "split",
                   "age_at_admission", "sex_category", "race_category",
                   "ethnicity_category", "y_prob", "y_pred"]
    return finalize_cohort(df, output_cols, META)


# ---------------------------------------------------------------- finalize


def test_prediction_id_one_per_stay():
    df = make_synth(n_stays=20)
    assert df.height == df["hospitalization_id"].n_unique() == 20
    one = df.filter(pl.col("hospitalization_id") == "h0")
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
    assert len(curves["thresholds"]) == 124  # dense low tail + 0.01-0.99 grid
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


# ---------------------------------------------------------------- end-to-end


def _fake_task():
    return SimpleNamespace(META=META, __file__=__file__)


def test_build_report_episodic_end_to_end(tmp_path):
    df = make_synth()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "report"),
                         episodic=True)
    jsons = {k: p for k, p in paths.items() if not k.endswith("_png")}
    assert set(jsons) == {"discrimination", "calibration", "dca", "fairness"}
    for name, path in jsons.items():
        payload = json.loads(path.read_text())
        md = payload["metadata"]
        assert md["report_mode"] == "episodic"
        assert md["split"] == "test" and md["n_stays"] == df.height


def test_build_report_split_selection(tmp_path):
    # mark half the stays "train"; only "test" should be analyzed
    df = make_synth(n_stays=400)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 200).then(pl.lit("train"))
        .otherwise(pl.lit("test")).alias("split"))
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "r"),
                         split="test", episodic=True)
    disc = json.loads(paths["discrimination"].read_text())
    assert disc["metadata"]["n_stays"] == 200

    with pytest.raises(ValueError, match="not found"):
        build_report(str(preds_path), _fake_task(), str(tmp_path / "r2"),
                     split="val", episodic=True)


def test_build_report_continuous_requires_mode(tmp_path):
    # A continuous task with no --mode must tell the caller to pick peak/landmark
    # (it used to raise NotImplementedError).
    df = make_synth()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)
    cont_task = SimpleNamespace(
        META={**META, "temporal_pattern": "continuous",
              "prediction_unit": "one-per-window"},
        __file__=__file__)

    from flair_benchmark.report import build_report
    with pytest.raises(ValueError, match="Continuous task"):
        build_report(str(preds_path), cont_task, str(tmp_path / "r"),
                     episodic=False)


def test_build_report_rejects_multi_row_per_stay(tmp_path):
    # duplicate every stay -> two rows per stay -> episodic guard must fire
    df = make_synth(n_stays=100)
    df = pl.concat([df, df])
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    with pytest.raises(ValueError, match="one row per stay"):
        build_report(str(preds_path), _fake_task(), str(tmp_path / "r"),
                     episodic=True)


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
                         cohort_path=str(cohort_path), episodic=True)
    fairness = json.loads(paths["fairness"].read_text())
    assert "sex_category" in fairness["subgroups"]


def test_viz_renders_four_episodic_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    df = make_synth()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "report"),
                         episodic=True, viz=True)
    pngs = {p.name for p in paths.values() if p.suffix == ".png"}
    assert pngs == {"roc_pr.png", "calibration.png", "dca.png", "fairness_auroc.png"}
    assert all(p.exists() for p in paths.values() if p.suffix == ".png")
