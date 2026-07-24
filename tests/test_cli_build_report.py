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
from test_report_continuous import make_synth_continuous
from test_report_extensions import make_synth
from typer.testing import CliRunner

from flair_benchmark.cli import app

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "example_report"

EXPECTED_JSONS = {"discrimination", "calibration", "dca", "fairness",
                  "operating_points"}
EXPECTED_PNGS = {"roc_pr.png", "calibration.png", "dca.png", "fairness_auroc.png",
                 "operating_points.png"}

TASK = "icu_readmission"      # episodic: one prediction per stay
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
        "--mode", "episodic", "--viz",
    ])
    assert result.exit_code == 0, result.output
    for kind in EXPECTED_JSONS:
        assert kind in result.output

    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    shutil.copytree(out_dir, ARTIFACT_DIR)
    return out_dir


def test_cli_continuous_mode(tmp_path):
    """`--mode continuous` on a continuous task emits the merged 5-JSON bundle
    (peak section + landmark pillars + leadtime with alert timing)."""
    preds_path = tmp_path / "preds.parquet"
    df = make_synth_continuous(n_stays=240, seed=7).rename({"label": "label_mortality"})
    df.write_parquet(preds_path)
    out_dir = tmp_path / "ct"

    result = runner.invoke(app, [
        "build-report", str(preds_path),
        "--task", "icu_daily_mortality", "--out", str(out_dir),
        "--mode", "continuous",
    ])
    assert result.exit_code == 0, result.output
    json_stems = {p.stem for p in out_dir.glob("*.json")}
    assert json_stems >= {
        "discrimination", "calibration", "fairness", "leadtime",
        "operating_points", "kpi"}
    assert "dca" not in json_stems   # DCA is episodic-only; no dca.json in continuous
    disc = json.loads((out_dir / "discrimination.json").read_text())
    assert disc["metadata"]["report_mode"] == "continuous"
    assert "peak" in disc
    # task1's META landmark_horizon = 30 (days); deep landmarks beyond the
    # synthetic stay lengths are suppressed stubs but still emitted.
    assert len(disc["landmarks"]) == 30
    assert disc["landmarks"][0]["hospitalization_time"] == 1
    # operating points: task1's clinical threshold is 0.95, which lands on exactly
    # one row of the fixed 99-point sweep (is_clinical). No offset arithmetic, no
    # dropped offsets. Confusion + burden mandatory; no sub-rates (no intervention).
    op = json.loads((out_dir / "operating_points.json").read_text())
    assert "dropped_offsets" not in op["metadata"] and "thresholds" not in op["metadata"]
    assert len(op["by_threshold"]) == 99
    assert "sub_rates" not in op["by_threshold"][0]
    clinical = [e for e in op["by_threshold"] if e["is_clinical"]]
    assert len(clinical) == 1 and clinical[0]["threshold"] == 0.95
    kpi = json.loads((out_dir / "kpi.json").read_text())
    assert kpi["pooled_dynamic_auroc"] is not None
    assert len(kpi["operating_points"]["fpr_by_threshold"]) == 99
    assert kpi["operating_points"]["clinical_threshold"] == 0.95


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
    # threshold-free by construction: the operating point moved to the scorecard.
    assert "operating_point" not in m


def test_operating_points_scorecard(report_dir):
    op = _load(report_dir, "operating_points")
    md = op["metadata"]
    # task5: clinical 0.20 lands on grid row 19 of the fixed 99-point sweep.
    assert md["grid"] == {"min": 0.01, "max": 0.99, "step": 0.01, "n": 99}
    assert md["clinical_index"] == 19
    assert "dropped_offsets" not in md and "offsets" not in md

    rows = op["by_threshold"]
    assert len(rows) == 99
    clinical = [e for e in rows if e["is_clinical"]]
    assert len(clinical) == 1 and clinical[0]["threshold"] == 0.20

    for e in rows:
        c, m = e["confusion"], e["metrics"]
        # counts can be suppressed to "<10" at the sparse tails; n is always the
        # full cohort size.
        assert c["n"] == 800
        assert 0.0 <= m["flag_rate"] <= 1.0
        for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "fpr"):
            v, ci = m[k], m.get(f"{k}_ci")
            if v is not None and ci:
                assert ci[0] <= v <= ci[1], f"{k} at t={e['threshold']} outside its CI"
        # net benefit is the DECISION basis and lives in its own block
        assert "net_benefit" not in m
        assert e["decision"]["net_benefit_treat_none"] == 0.0

    # Raising the bar flags fewer stays: sensitivity falls, specificity rises.
    sens = [e["metrics"]["sensitivity"] for e in rows]
    spec = [e["metrics"]["specificity"] for e in rows]
    flag = [e["metrics"]["flag_rate"] for e in rows]
    assert sens == sorted(sens, reverse=True)
    assert spec == sorted(spec)
    assert flag == sorted(flag, reverse=True)


def test_operating_points_decision_is_direction_aware(report_dir):
    """task5 is direction 'above', so the two bases collapse into one."""
    op = _load(report_dir, "operating_points")
    assert op["metadata"]["clinical_direction"] == "above"
    for e in op["by_threshold"]:
        dec = e["decision"]
        assert dec["acts_on"] == "y_prob >= threshold"
        # the acted arm IS the flagged arm (tp+fp) when the action fires on HIGH risk
        c = e["confusion"]
        if all(isinstance(c[k], int) for k in ("tp", "fp")) and isinstance(dec["n_acted"], int):
            assert dec["n_acted"] == c["tp"] + c["fp"]
        # ...and the decision curve's axis is the label axis, untransformed
        assert dec["dca_threshold"] == e["threshold"]


def test_operating_points_carries_subgroup_equity(report_dir):
    """Fairness at a threshold lives here now — at every threshold, not just one."""
    op = _load(report_dir, "operating_points")
    for e in op["by_threshold"]:
        sex = e["fairness"]["subgroups"]["sex_category"]
        assert set(sex) == {"Male", "Female"}
        for lvl in sex.values():
            assert lvl["n"] >= 10
            assert 0.0 <= lvl["positive_rate"] <= 1.0
        parity = e["fairness"]["parity"]["sex_category"]
        assert parity["reference_group"] in ("Male", "Female")
        assert parity["demographic_parity_gap"] >= 0
        assert "demographic_parity_gap_ci" in parity


def test_calibration_values(report_dir):
    c = _load(report_dir, "calibration")
    m, curves = c["metrics"], c["curves"]
    assert curves["reliability"]["n_bins"] == 10
    assert curves["smoothed"]["method"] == "linear_fit"
    assert len(curves["smoothed"]["pred_grid"]) == 50
    assert m["brier_ci"][0] <= m["brier"] <= m["brier_ci"][1]


def test_dca_values(report_dir):
    """dca.json keeps its curve AND its verdict — the decision curve is the analysis."""
    dca = _load(report_dir, "dca")
    m, curves = dca["metrics"], dca["curves"]
    assert len(curves["thresholds"]) == 164   # dense tail at BOTH ends + 0.01-0.99
    assert curves["net_benefit_none"] == 0.0
    assert "useful_range" in m and "all_useful_intervals" in m
    v = m["verdict"]
    assert {"clinically_useful", "model", "model_ci", "treat_all", "beats"} <= set(v)
    # task5 is "above": the decision axis IS the label axis
    assert dca["metadata"]["clinical_threshold_on_axis"] == 0.20
    assert m["clinical_threshold"] == 0.20


def test_net_benefit_has_one_definition_across_the_bundle(report_dir):
    """THE cross-check. The scorecard and dca.json must be the same quantity.

    They are computed from the same primitives (dca.to_decision_basis /
    axis_threshold), so at every grid threshold the scorecard's decision.net_benefit
    must equal dca.json's curve at decision.dca_threshold, and at the clinical
    threshold both must equal dca.json's verdict. If these ever drift apart, one of
    the two files is quietly reporting a different decision.
    """
    dca = _load(report_dir, "dca")
    thr = dca["curves"]["thresholds"]
    nb_model = dca["curves"]["net_benefit_model"]
    nb_all = dca["curves"]["net_benefit_all"]

    for e in _load(report_dir, "operating_points")["by_threshold"]:
        dec = e["decision"]
        x = dec["dca_threshold"]
        i = min(range(len(thr)), key=lambda j: abs(thr[j] - x))
        # sparse tails null net benefit on their own arm; where BOTH are defined the
        # scorecard and dca.json must be the SAME number.
        if dec["net_benefit"] is not None and nb_model[i] is not None:
            assert dec["net_benefit"] == nb_model[i], f"NB drifted at t={e['threshold']}"
        assert dec["net_benefit_treat_all"] == nb_all[i]
        if e["is_clinical"]:
            v = dca["metrics"]["verdict"]
            assert dec["net_benefit"] == v["model"]
            assert dec["net_benefit_ci"] == v["model_ci"]
            assert dec["net_benefit_treat_all"] == v["treat_all"]
            assert dec["clinically_useful"] == v["clinically_useful"]


def test_fairness_subgroups_and_parity(report_dir):
    """fairness.json keeps the threshold-FREE view: subgroup ranking + calibration."""
    f = _load(report_dir, "fairness")
    assert f["metadata"]["threshold_free"] is True
    sex = f["subgroups"]["sex_category"]
    assert set(sex) == {"Male", "Female"}
    for m in sex.values():
        assert m["n"] >= 10 and "brier" in m and "auroc" in m
        # the decision-threshold view is reported per threshold, not here
        assert "positive_rate" not in m and "tpr" not in m and "fpr" not in m
    assert "parity" not in f
    assert "overall" not in f


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
                                "--out", str(out_a), "--mode", "episodic"])
    assert res_a.exit_code == 0, res_a.output

    b_path = tmp_path / "b.parquet"
    base.write_parquet(b_path)
    out_b = tmp_path / "report_b"
    res_b = runner.invoke(app, ["build-report", str(b_path), "--task", TASK,
                                "--out", str(out_b), "--mode", "episodic"])
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
                                 "--out", str(tmp_path / "r"), "--mode", "episodic"])
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
                                 "--mode", "episodic"])
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
                                 "--mode", "episodic"])
    assert result.exit_code != 0
    assert "y_prob" in str(result.exception) and "null" in str(result.exception)
