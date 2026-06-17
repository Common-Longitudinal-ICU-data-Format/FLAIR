"""CLI-level regression test for `flair build-report` (episodic path).

Runs the real typer command end-to-end on deterministic synthetic, one-row-per-
stay predictions and checks the structure + key values of the four episodic
JSONs (discrimination, calibration, dca, fairness) plus the --viz PNGs.

The generated bundle is copied to tests/artifacts/example_report/ for browsing:

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

EXPECTED_JSONS = {"discrimination", "calibration", "dca", "fairness"}
EXPECTED_PNGS = {"roc_pr.png", "calibration.png", "dca.png", "fairness_auroc.png"}

TASK = "task5_icu_readmission"      # episodic: one prediction per stay
LABEL = "label_icu_readmission"

runner = CliRunner()


@pytest.fixture(scope="module")
def report_dir(tmp_path_factory) -> Path:
    """Run the CLI once for the whole module; return the report directory."""
    tmp = tmp_path_factory.mktemp("cli")
    preds_path = tmp / "preds.parquet"
    out_dir = tmp / "report"

    df = make_synth(n_stays=800, seed=42).rename({"label": LABEL})
    df.write_parquet(preds_path)

    result = runner.invoke(app, [
        "build-report", str(preds_path),
        "--task", TASK, "--out", str(out_dir),
        "--split", "test", "--episodic", "--viz",
    ])
    assert result.exit_code == 0, result.output
    for kind in EXPECTED_JSONS:
        assert kind in result.output

    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    shutil.copytree(out_dir, ARTIFACT_DIR)
    return out_dir


def _load(report_dir: Path, name: str) -> dict:
    return json.loads((report_dir / f"{name}.json").read_text())


def test_all_outputs_exist(report_dir):
    assert {p.stem for p in report_dir.glob("*.json")} == EXPECTED_JSONS
    rendered = {p.name for p in (report_dir / "viz").glob("*.png")}
    assert rendered == EXPECTED_PNGS
    for png in (report_dir / "viz").glob("*.png"):
        assert png.stat().st_size > 5_000  # a real rendered figure, not a stub


def test_every_json_has_audit_metadata(report_dir):
    for name in EXPECTED_JSONS:
        md = _load(report_dir, name)["metadata"]
        assert md["task"] == TASK
        assert "task_script_sha256" in md and "flair_version" in md
        assert md["report_mode"] == "episodic"
        assert md["split"] == "test"
        assert md["n_stays"] == 800


def test_discrimination_values(report_dir):
    d = _load(report_dir, "discrimination")
    m, roc = d["metrics"], d["curves"]["roc"]
    assert m["auroc"] is not None
    lo, hi = m["auroc_ci"]
    assert lo <= m["auroc"] <= hi
    assert len(roc["fpr"]) == len(roc["tpr"]) <= 500
    op = m["operating_point"]
    assert op["threshold"] == 0.20   # task5 clinical threshold
    assert op["sensitivity_ci"][0] <= op["sensitivity"] <= op["sensitivity_ci"][1]


def test_calibration_values(report_dir):
    c = _load(report_dir, "calibration")
    m, curves = c["metrics"], c["curves"]
    assert curves["reliability"]["n_bins"] == 10
    assert curves["smoothed"]["method"] == "linear_fit"
    assert len(curves["smoothed"]["pred_grid"]) == 50
    assert m["brier_ci"][0] <= m["brier"] <= m["brier_ci"][1]


def test_dca_values(report_dir):
    dca = _load(report_dir, "dca")
    m, curves = dca["metrics"], dca["curves"]
    assert len(curves["thresholds"]) == 124
    assert curves["net_benefit_none"] == 0.0
    assert m["clinical_threshold"] == 0.20
    nb = m["net_benefit_at_clinical_threshold"]
    assert "model" in nb and "model_ci" in nb


def test_fairness_subgroups_and_parity(report_dir):
    f = _load(report_dir, "fairness")
    assert f["metadata"]["operating_threshold"] == 0.20
    sex = f["subgroups"]["sex_category"]
    assert set(sex) == {"Male", "Female"}
    for m in sex.values():
        assert m["n"] >= 10 and "brier" in m and "auroc" in m
    parity = f["parity"]["sex_category"]
    assert parity["reference_group"] in ("Male", "Female")
    assert "demographic_parity_gap_ci" in parity


def test_y_prob_only_derives_y_pred(tmp_path):
    """A site ships y_prob only (no y_pred) — the episodic report still builds.

    Discrimination/calibration/DCA are probability-based and the operating point
    is at the clinical threshold, so dropping the 0.5-derived y_pred column must
    not change the discrimination output.
    """
    base = make_synth(n_stays=600, seed=42).rename({"label": LABEL})

    a_path = tmp_path / "a.parquet"
    base.drop("y_pred").write_parquet(a_path)
    out_a = tmp_path / "report_a"
    res_a = runner.invoke(app, ["build-report", str(a_path), "--task", TASK,
                                "--out", str(out_a), "--episodic"])
    assert res_a.exit_code == 0, res_a.output

    b_path = tmp_path / "b.parquet"
    base.write_parquet(b_path)
    out_b = tmp_path / "report_b"
    res_b = runner.invoke(app, ["build-report", str(b_path), "--task", TASK,
                                "--out", str(out_b), "--episodic"])
    assert res_b.exit_code == 0, res_b.output

    disc_a = json.loads((out_a / "discrimination.json").read_text())["curves"]["roc"]
    disc_b = json.loads((out_b / "discrimination.json").read_text())["curves"]["roc"]
    assert disc_a == disc_b


def test_binary_without_y_pred_or_y_prob_rejected(tmp_path):
    df = (make_synth(n_stays=100, seed=1).rename({"label": LABEL})
          .drop(["y_pred", "y_prob"]))
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)
    result = runner.invoke(app, ["build-report", str(preds_path), "--task", TASK,
                                 "--out", str(tmp_path / "r"), "--episodic"])
    assert result.exit_code != 0
    assert "y_prob" in str(result.exception)


def _write_preds(tmp_path, df) -> Path:
    preds_path = tmp_path / "preds.parquet"
    df.rename({"label": LABEL}).write_parquet(preds_path)
    return preds_path


@pytest.mark.parametrize("bad_prob", [1.5, -0.2])
def test_y_prob_out_of_range_rejected(tmp_path, bad_prob):
    df = make_synth(n_stays=200, seed=11)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 5).then(pl.lit(bad_prob))
        .otherwise(pl.col("y_prob")).alias("y_prob"))
    result = runner.invoke(app, ["build-report", str(_write_preds(tmp_path, df)),
                                 "--task", TASK, "--out", str(tmp_path / "r"),
                                 "--episodic"])
    assert result.exit_code != 0
    msg = str(result.exception)
    assert "y_prob" in msg and "[0, 1]" in msg


def test_y_prob_null_rejected(tmp_path):
    df = make_synth(n_stays=200, seed=12)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 3).then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("y_prob")).alias("y_prob"))
    result = runner.invoke(app, ["build-report", str(_write_preds(tmp_path, df)),
                                 "--task", TASK, "--out", str(tmp_path / "r"),
                                 "--episodic"])
    assert result.exit_code != 0
    assert "y_prob" in str(result.exception) and "null" in str(result.exception)
