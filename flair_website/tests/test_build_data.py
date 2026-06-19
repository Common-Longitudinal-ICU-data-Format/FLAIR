"""The runs_data → data/ parser (scripts/build_data.py)."""
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_data", Path(__file__).resolve().parent.parent / "scripts" / "build_data.py")
build_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_data)


def _meta(mode):
    return {"report_mode": mode, "n_stays": 100, "flair_version": "9.9.9",
            "generated_at": "2026-01-01", "split": "test", "evaled_at": 0.5}


def _disc_flat(mode, auroc):
    return {"task_type": "binary", "metadata": _meta(mode), "prevalence": 0.2,
            "metrics": {"auroc": auroc, "auroc_ci": [auroc - 0.02, auroc + 0.02],
                        "auprc": 0.5, "operating_point": {"threshold": 0.5}},
            "curves": {"roc": {"fpr": [0, 1], "tpr": [0, 1]},
                       "pr": {"recall": [0, 1], "precision": [1, 0]}}}


def _calib_flat(slope):
    return {"metadata": _meta("episodic"),
            "metrics": {"slope": slope, "intercept": 0.1, "brier": 0.12},
            "curves": {"reliability": {}, "smoothed": {}}}


def _disc_landmark(auroc0):
    lms = [{"landmark_index": i, "windows_before_end": i, "window_unit": "day", "n": 50 - i,
            "prevalence": 0.2, "metrics": {"auroc": auroc0 - 0.01 * i,
            "auroc_ci": [0.7, 0.9], "auprc": 0.5}, "curves": {"roc": {}, "pr": {}}}
           for i in range(3)]
    return {"metadata": _meta("landmark"), "landmarks": lms}


def _write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _fixture(runs: Path):
    base = runs / "sitex_baseline_non_phi_for_upload"
    # episodic task
    e = base / "taskE" / "report"
    _write(e / "discrimination.json", _disc_flat("episodic", 0.80))
    _write(e / "calibration.json", _calib_flat(1.05))
    _write(e / "fairness.json", {"subgroups": {}, "parity": {}})
    _write(base / "taskE" / "table1.json", {"n_total": 100})
    # peak task (no calibration)
    p = base / "taskP" / "report"
    _write(p / "discrimination.json", _disc_flat("peak", 0.70))
    _write(p / "fairness.json", {"subgroups": {}, "parity": {}})
    _write(base / "taskP" / "table1.json", {"n_total": 100})
    # landmark task
    lm = base / "taskL" / "report"
    _write(lm / "discrimination.json", _disc_landmark(0.86))
    _write(lm / "calibration.json", {"metadata": _meta("landmark"),
           "landmarks": [{"landmark_index": 0, "metrics": {"slope": 1.2, "intercept": 0.0, "brier": 0.1}}]})
    _write(lm / "leadtime.json", {"curve": [], "metrics": {"pct_caught": 0.5}, "trajectory": []})
    _write(base / "taskL" / "table1.json", {"n_total": 100})


def test_parser_builds_leaderboard_and_reports(tmp_path):
    runs, data = tmp_path / "runs", tmp_path / "data"
    runs.mkdir()
    _fixture(runs)
    board = build_data.build(runs=runs, data_dir=data)

    assert [s["key"] for s in board["sites"]] == ["sitex"]
    scores = board["models"][0]["scores"]

    # episodic: headline auroc + calibration slope flow through
    assert scores["taskE"]["sitex"]["auroc"] == 0.80
    assert scores["taskE"]["sitex"]["cal_slope"] == 1.05

    # peak: no calibration → cal_slope is None
    assert scores["taskP"]["sitex"]["cal_slope"] is None
    assert scores["taskP"]["sitex"]["auroc"] == 0.70

    # landmark: headline taken from landmark_index 0
    assert scores["taskL"]["sitex"]["auroc"] == 0.86
    assert scores["taskL"]["sitex"]["cal_slope"] == 1.2

    # report bundles written with the right mode + pillars
    rep = json.loads((data / "reports" / "flair-xgboost-baseline" / "sitex" /
                      "taskL" / "report.json").read_text())
    assert rep["mode"] == "landmark" and rep["landmark"] is True and rep["leadtime"]
    peak = json.loads((data / "reports" / "flair-xgboost-baseline" / "sitex" /
                       "taskP" / "report.json").read_text())
    assert peak["calibration"] is None and peak["dca"] is None
