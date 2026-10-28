"""End-to-end: `flair build-report` must emit exactly TWO JSONs per mode.

The whole point of the simplification is the file list, so it is pinned here
rather than left implicit in the builders:

    continuous  ->  landmark.json  +  hospitalization_level.json
    episodic    ->  overall.json   +  hospitalization_level.json

`hospitalization_level.json` carries the SAME schema in both modes, apart from
the net-benefit keys, which are episodic-only. That is what lets the FLAIR
website use one reader and one aggregator, so it is asserted directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from flair_benchmark.cli import app

runner = CliRunner()

CONTINUOUS_JSONS = {"landmark", "hospitalization_level"}
EPISODIC_JSONS = {"overall", "hospitalization_level"}

CONTINUOUS_PNGS = {"landmark_auroc.png", "landmark_auprc.png",
                   "landmark_brier.png", "landmark_slope.png",
                   "threshold_sweep.png"}
EPISODIC_PNGS = {"roc.png", "pr.png", "calibration.png", "dca.png",
                 "threshold_sweep.png", "fairness.png"}

# gone for good — a stale file in an output directory would be uploaded
RETIRED_JSONS = {"kpi", "leadtime", "dca", "calibration", "discrimination",
                 "fairness", "operating_points"}


def synth_episodic(n: int = 700, seed: int = 21) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.2, n)
    return pl.DataFrame({
        "hospitalization_join_id": [f"H{i:04d}" for i in range(n)],
        "prediction_id": [f"P{i:04d}" for i in range(n)],
        "split": ["test"] * n,
        "y_prob": np.clip(rng.beta(2, 6, n) + 0.35 * y, 0.001, 0.999),
        "label_icu_readmission": y,
        "sex_category": rng.choice(["Female", "Male"], n),
        "race_category": rng.choice(["White", "Black"], n),
        "age_at_admission": rng.integers(20, 95, n),
    })


def synth_continuous(n_stays: int = 300, seed: int = 22) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        ever = int(rng.binomial(1, 0.25))
        for w in range(1, int(rng.integers(3, 9))):
            rows.append({
                "hospitalization_join_id": f"H{i:04d}",
                "prediction_id": f"P{i:04d}_{w}",
                "split": "test",
                "window_number": w,
                "y_prob": float(np.clip(rng.beta(2, 6) + 0.3 * ever, 0.001, 0.999)),
                "label_mortality": ever if w > 1 else 0,
                "sex_category": "Female" if i % 2 else "Male",
                "age_at_admission": int(20 + (i % 70)),
            })
    return pl.DataFrame(rows)


def run(tmp_path: Path, df: pl.DataFrame, task: str, mode: str) -> Path:
    preds = tmp_path / "preds.parquet"
    out = tmp_path / "report"
    df.write_parquet(preds)
    result = runner.invoke(app, [
        "build-report", str(preds), "--task", task,
        "--out", str(out), "--mode", mode,
    ])
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture(scope="module")
def episodic_dir(tmp_path_factory):
    return run(tmp_path_factory.mktemp("epi"), synth_episodic(),
               "icu_readmission", "episodic")


@pytest.fixture(scope="module")
def continuous_dir(tmp_path_factory):
    return run(tmp_path_factory.mktemp("cont"), synth_continuous(),
               "icu_daily_mortality", "continuous")


def load(d: Path, name: str) -> dict:
    return json.loads((d / f"{name}.json").read_text())


# ------------------------------------------------------------- the file list


def test_continuous_emits_exactly_two_jsons(continuous_dir):
    got = {p.stem for p in continuous_dir.glob("*.json")}
    assert got == CONTINUOUS_JSONS


def test_episodic_emits_exactly_two_jsons(episodic_dir):
    got = {p.stem for p in episodic_dir.glob("*.json")}
    assert got == EPISODIC_JSONS


def test_no_retired_file_is_written(continuous_dir, episodic_dir):
    for d in (continuous_dir, episodic_dir):
        for stem in {p.stem for p in d.glob("*.json")}:
            assert stem not in RETIRED_JSONS, f"{stem}.json came back"


def test_continuous_renders_five_pngs(continuous_dir):
    assert {p.name for p in (continuous_dir / "viz").glob("*.png")} \
        == CONTINUOUS_PNGS


def test_episodic_renders_its_pngs(episodic_dir):
    assert {p.name for p in (episodic_dir / "viz").glob("*.png")} \
        == EPISODIC_PNGS


# ----------------------------------------------------------- the shared schema


def test_hospitalization_level_has_the_same_keys_in_both_modes(
        continuous_dir, episodic_dir):
    cont = load(continuous_dir, "hospitalization_level")
    epi = load(episodic_dir, "hospitalization_level")
    assert set(cont) == set(epi)
    assert set(cont["fairness"]) == set(epi["fairness"])
    assert set(cont["selected_thresholds"][0]) == \
        set(epi["selected_thresholds"][0])


def test_only_net_benefit_keys_differ_between_modes(continuous_dir, episodic_dir):
    cont = load(continuous_dir, "hospitalization_level")["by_threshold"][0]
    epi = load(episodic_dir, "hospitalization_level")["by_threshold"][0]
    assert set(epi) - set(cont) == {
        "net_benefit", "net_benefit_ci", "net_benefit_treat_all"}
    assert set(cont) - set(epi) == set()


# ------------------------------------------------------------------- metadata


def test_every_payload_carries_the_audit_block(continuous_dir, episodic_dir):
    for d, names in ((continuous_dir, CONTINUOUS_JSONS),
                     (episodic_dir, EPISODIC_JSONS)):
        for name in names:
            md = load(d, name)["metadata"]
            for key in ("task", "flair_version", "generated_at", "report_mode",
                        "split", "report_schema"):
                assert key in md, f"{name}.json missing metadata.{key}"


def test_report_schema_is_bumped_to_four(episodic_dir):
    assert load(episodic_dir, "overall")["metadata"]["report_schema"] == 4


def test_reports_are_scored_on_the_test_split(episodic_dir):
    assert load(episodic_dir, "overall")["metadata"]["split"] == "test"


# --------------------------------------------------------------- the contents


def test_continuous_landmark_holds_the_four_pooled_numbers(continuous_dir):
    pooled = load(continuous_dir, "landmark")["pooled"]
    assert set(pooled) == {
        "auroc", "auroc_ci", "auprc", "auprc_ci",
        "brier", "brier_ci", "calibration_slope", "calibration_slope_ci"}


def test_continuous_bundle_has_no_landmark_fairness(continuous_dir):
    assert "sex_category" not in json.dumps(load(continuous_dir, "landmark"))


def test_continuous_bundle_still_reports_fairness_at_the_decision(continuous_dir):
    fair = load(continuous_dir, "hospitalization_level")["fairness"]
    assert "sex_category" in fair["subgroups"]


def test_episodic_bundle_has_no_separate_dca_file(episodic_dir):
    assert not (episodic_dir / "dca.json").exists()
    row = load(episodic_dir, "hospitalization_level")["by_threshold"][0]
    assert row["net_benefit_treat_all"] is not None


def test_below_direction_task_runs_end_to_end(tmp_path):
    """extubation_failure_24h acts on LOW risk — the direction-aware path."""
    df = synth_episodic().rename(
        {"label_icu_readmission": "label_extubation_failure_24h"})
    out = run(tmp_path, df, "extubation_failure_24h", "episodic")
    payload = load(out, "hospitalization_level")
    assert payload["metadata"]["clinical_direction"] == "below"
    assert payload["by_threshold"][0]["net_benefit"] is not None
