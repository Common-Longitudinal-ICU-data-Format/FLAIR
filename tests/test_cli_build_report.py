"""CLI-level regression test for `flair build-report`.

Runs the real typer command end-to-end on deterministic synthetic
predictions (seeded — same numbers every run) and checks the structure
and key values of all 6 JSONs plus the --viz PNGs. If a code change
alters any reported metric, the value assertions here will catch it.

The full generated bundle is also copied to tests/artifacts/example_report/
so you can open the JSONs and PNGs and see exactly what a site produces:

    pytest tests/test_cli_build_report.py
    open tests/artifacts/example_report/viz/*.png
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from test_report_extensions import make_synth
from typer.testing import CliRunner

from flair_benchmark.cli import app

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "example_report"

EXPECTED_JSONS = {"table1", "tripod_ai", "fairness", "curves", "temporal", "dca",
                  "threshold_analysis"}
EXPECTED_PNGS = {"roc_pr.png", "calibration.png", "dca.png", "auroc_by_window.png",
                 "auroc_by_year.png", "trajectory_by_outcome.png", "fairness_auroc.png"}
# task1's clinical threshold is 0.95 (withdraw life support); the synthetic
# model rarely predicts >0.95, so threshold_analysis.png is correctly absent
# (label_metrics suppressed) — the suppression path is itself under test.

runner = CliRunner()


@pytest.fixture(scope="module")
def report_dir(tmp_path_factory) -> Path:
    """Run the CLI once for the whole module; return the report directory."""
    tmp = tmp_path_factory.mktemp("cli")
    preds_path = tmp / "preds.parquet"
    out_dir = tmp / "report"

    # seeded generator -> byte-stable inputs -> stable metrics across runs
    df = make_synth(n_stays=600, seed=42).rename({"label": "label_mortality"})
    # keep only the strict preds contract + passthrough columns a site would ship
    df.write_parquet(preds_path)

    result = runner.invoke(app, [
        "build-report", str(preds_path),
        "--task", "task1_icu_daily_mortality",
        "--out", str(out_dir),
        "--viz",
    ])
    assert result.exit_code == 0, result.output
    # echo lists every artifact: 6 JSONs + 7 PNGs
    for kind in EXPECTED_JSONS:
        assert kind in result.output

    # keep a browsable copy for humans
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    shutil.copytree(out_dir, ARTIFACT_DIR)
    return out_dir


def _load(report_dir: Path, name: str) -> dict:
    return json.loads((report_dir / f"{name}.json").read_text())


def test_all_outputs_exist(report_dir):
    assert {p.stem for p in report_dir.glob("*.json")} == EXPECTED_JSONS
    assert {p.name for p in (report_dir / "viz").glob("*.png")} == EXPECTED_PNGS
    for png in (report_dir / "viz").glob("*.png"):
        assert png.stat().st_size > 5_000  # a real rendered figure, not a stub


def test_every_json_has_audit_metadata(report_dir):
    for name in EXPECTED_JSONS:
        md = _load(report_dir, name)["metadata"]
        assert md["task"] == "task1_icu_daily_mortality"
        assert "task_script_sha256" in md and "flair_version" in md


def test_tripod_ai_values(report_dir):
    res = _load(report_dir, "tripod_ai")["results"]
    perf, cal = res["performance"], res["calibration"]
    # synthetic generator is seeded; these only move if the code changes
    assert perf["auroc"] == pytest.approx(0.8214, abs=0.001)
    assert perf["auprc"] > 0.4
    assert perf["tp"] + perf["tn"] + perf["fp"] + perf["fn"] == 3220
    # generator is calibrated by construction
    assert cal["slope"] == pytest.approx(1.0, abs=0.1)
    assert cal["intercept"] == pytest.approx(0.0, abs=0.1)
    assert cal["o_e_ratio"] == pytest.approx(1.0, abs=0.05)
    assert 0 < cal["brier"] < 0.15
    assert 0 <= cal["ece"] < 0.05


def test_curves_structure(report_dir):
    curves = _load(report_dir, "curves")
    assert len(curves["roc"]["fpr"]) == len(curves["roc"]["tpr"]) <= 500
    assert curves["calibration"]["n_bins"] == 10
    smoothed = curves["calibration_extended"]["smoothed"]
    assert smoothed["method"] == "logistic_recalibration"
    assert len(smoothed["pred_grid"]) == len(smoothed["obs_smoothed"]) == 50
    # the smoothed curve is monotone increasing
    assert smoothed["obs_smoothed"] == sorted(smoothed["obs_smoothed"])


def test_temporal_windows_and_years(report_dir):
    t = _load(report_dir, "temporal")
    windows = [b["bin"] for b in t["by_window"]["bins"]]
    assert windows == ["1", "2", "3", "4", "5", "6", "7+"]
    for b in t["by_window"]["bins"]:
        assert b["n"] >= 10
        assert b["auroc"] is None or 0.5 < b["auroc"] <= 1.0
    assert t["by_window"]["unit"] == "day"  # task1 grid is local-07:00 daily
    years = [b["bin"] for b in t["by_year"]["bins"]]
    assert years == sorted(years) and set(years) <= set(range(2017, 2024))
    assert t["by_year"]["year_source"] == "derived_from_min_prediction_dttm"
    pos, neg = t["trajectory_by_outcome"]["series"]
    assert pos["outcome"] == "positive" and neg["outcome"] == "negative"
    # risk separates outcomes in every reportable window
    for p, q in zip(pos["mean_prob"], neg["mean_prob"]):
        if p is not None and q is not None:
            assert p > q


def test_dca_values(report_dir):
    dca = _load(report_dir, "dca")
    assert len(dca["thresholds"]) == 99
    assert dca["net_benefit_none"] == 0.0
    assert dca["prevalence"] == pytest.approx(0.16, abs=0.05)
    # a discriminative calibrated model is useful over a wide range
    ur = dca["useful_range"]
    assert ur is not None and ur["low"] <= 0.05 and ur["high"] >= 0.5
    # model net benefit at low threshold approaches prevalence
    assert dca["net_benefit_model"][0] == pytest.approx(dca["prevalence"], abs=0.02)


def test_threshold_analysis_task1(report_dir):
    """task1 clinical threshold = 0.95 (withdraw life support if risk > 95%)."""
    ta = _load(report_dir, "threshold_analysis")
    assert ta["clinical_threshold"] == 0.95
    assert ta["clinical_action"] == "withdraw life support"
    assert ta["clinical_direction"] == "above"
    # synthetic model rarely exceeds 0.95 -> suppressed, never fabricated
    assert ta["label_metrics"] is None
    assert ta["action_metrics"] is None
    assert ta["metadata"]["pauc_band"] == 0.05
    # dca carries the same marker
    dca = _load(report_dir, "dca")
    assert dca["clinical_threshold"] == 0.95
    assert "model" in dca["net_benefit_at_clinical_threshold"]
    assert isinstance(dca["clinically_useful"], bool)


def test_fairness_subgroups_and_parity(report_dir):
    f = _load(report_dir, "fairness")
    sex = f["subgroups"]["sex_category"]
    assert set(sex) == {"Male", "Female"}
    for m in sex.values():
        assert m["n"] >= 10 and "brier" in m and "auroc" in m
    parity = f["parity"]["sex_category"]
    assert parity["reference_group"] in ("Male", "Female")
    for gap in ("demographic_parity_gap", "equalized_odds_tpr_gap",
                "equalized_odds_fpr_gap"):
        assert 0 <= parity[gap] <= 1


def test_table1_suppression_format(report_dir):
    payload = json.dumps(_load(report_dir, "table1"))
    # no raw small counts: any literal count below 10 must appear as "<10"
    assert '"<10"' in payload or all(
        f'": {k},' not in payload for k in range(1, 10))


def test_y_prob_only_derives_y_pred(tmp_path):
    """A site ships the cohort file + y_prob only (no y_pred) -> y_pred is
    derived at 0.5, and confusion counts match a run that supplied y_pred."""
    base = make_synth(n_stays=600, seed=42).rename({"label": "label_mortality"})

    # run A: y_prob only, y_pred dropped
    a_path = tmp_path / "a.parquet"
    base.drop("y_pred").write_parquet(a_path)
    out_a = tmp_path / "report_a"
    res_a = runner.invoke(app, [
        "build-report", str(a_path),
        "--task", "task1_icu_daily_mortality", "--out", str(out_a)])
    assert res_a.exit_code == 0, res_a.output

    # run B: explicit y_pred = (y_prob >= 0.5), same data
    b_path = tmp_path / "b.parquet"
    base.with_columns((pl.col("y_prob") >= 0.5).cast(pl.Int8).alias("y_pred")
                      ).write_parquet(b_path)
    out_b = tmp_path / "report_b"
    res_b = runner.invoke(app, [
        "build-report", str(b_path),
        "--task", "task1_icu_daily_mortality", "--out", str(out_b)])
    assert res_b.exit_code == 0, res_b.output

    perf_a = json.loads((out_a / "tripod_ai.json").read_text())["results"]["performance"]
    perf_b = json.loads((out_b / "tripod_ai.json").read_text())["results"]["performance"]
    for k in ("tp", "tn", "fp", "fn"):
        assert perf_a[k] == perf_b[k]


def test_binary_without_y_pred_or_y_prob_rejected(tmp_path):
    """Neither y_pred nor y_prob -> clear schema error."""
    df = (make_synth(n_stays=100, seed=1).rename({"label": "label_mortality"})
          .drop(["y_pred", "y_prob"]))
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)
    result = runner.invoke(app, [
        "build-report", str(preds_path),
        "--task", "task1_icu_daily_mortality", "--out", str(tmp_path / "r")])
    assert result.exit_code != 0
    assert "y_prob" in str(result.exception)


def test_cli_without_viz_or_prob_columns(tmp_path):
    """One-per-stay-shaped preds (no window cols) still produce all 6 JSONs."""
    df = (make_synth(n_stays=300, seed=7)
          .rename({"label": "label_mortality"})
          .unique(subset="hospitalization_id")
          .drop(["window_number", "prediction_id", "prediction_dttm"]))
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)
    result = runner.invoke(app, [
        "build-report", str(preds_path),
        "--task", "task5_icu_readmission",
        "--out", str(tmp_path / "report"),
    ])
    # task5 has a different label column -> strict schema must reject this
    assert result.exit_code != 0
    assert "label_icu_readmission" in str(result.exception)

    result = runner.invoke(app, [
        "build-report", str(preds_path),
        "--task", "task1_icu_daily_mortality",
        "--out", str(tmp_path / "report"),
    ])
    assert result.exit_code == 0, result.output
    t = json.loads((tmp_path / "report" / "temporal.json").read_text())
    # no temporal context in preds -> sections degrade, never crash
    assert t["by_year"] is None


def _write_preds(tmp_path, df) -> Path:
    preds_path = tmp_path / "preds.parquet"
    df.rename({"label": "label_mortality"}).write_parquet(preds_path)
    return preds_path


@pytest.mark.parametrize("bad_prob", [1.5, -0.2])
def test_y_prob_out_of_range_rejected(tmp_path, bad_prob):
    """y_prob outside [0,1] is a hard error (would silently corrupt cal/DCA/Brier)."""
    df = make_synth(n_stays=200, seed=11)
    # poison a handful of probabilities
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 5)
        .then(pl.lit(bad_prob))
        .otherwise(pl.col("y_prob"))
        .alias("y_prob")
    )
    result = runner.invoke(app, [
        "build-report", str(_write_preds(tmp_path, df)),
        "--task", "task1_icu_daily_mortality",
        "--out", str(tmp_path / "report"),
    ])
    assert result.exit_code != 0
    msg = str(result.exception)
    assert "y_prob" in msg and "[0, 1]" in msg


def test_y_prob_null_rejected(tmp_path):
    """Null y_prob is a hard error too (becomes NaN downstream)."""
    df = make_synth(n_stays=200, seed=12)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 3)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("y_prob"))
        .alias("y_prob")
    )
    result = runner.invoke(app, [
        "build-report", str(_write_preds(tmp_path, df)),
        "--task", "task1_icu_daily_mortality",
        "--out", str(tmp_path / "report"),
    ])
    assert result.exit_code != 0
    assert "y_prob" in str(result.exception) and "null" in str(result.exception)
