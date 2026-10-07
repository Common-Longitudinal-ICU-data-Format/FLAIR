"""Config templates shipped as package data, and the clif config contract.

`flair init-config` scaffolds from flair_benchmark/templates/; the scaffolded clif
config must parse; timezone is required; the cache never lands in the data dir.
All synthetic — no CLIF data is read.
"""

import json
from importlib.resources import files
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from flair_benchmark.cli import app
from flair_benchmark.cohort.clif import read_clif_config
from flair_benchmark.cohort.stitch import _cache_dir
from flair_benchmark.constants import TEMPLATE_FILES

runner = CliRunner()


def test_templates_are_package_data():
    templates = files("flair_benchmark") / "templates"
    for name in TEMPLATE_FILES:
        assert (templates / name).is_file(), name
    json.loads((templates / "clif_config.json").read_text())
    feat = yaml.safe_load((templates / "feature_config.yaml").read_text())
    assert feat["clif_config"] == "clif_config.json"


def test_init_config_writes_templates(tmp_path):
    res = runner.invoke(app, ["init-config", "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output
    templates = files("flair_benchmark") / "templates"
    for name in TEMPLATE_FILES:
        assert (tmp_path / name).read_text() == (templates / name).read_text()


def test_init_config_skips_existing_unless_force(tmp_path):
    target = tmp_path / "clif_config.json"
    target.write_text("{}")
    res = runner.invoke(app, ["init-config", "--out", str(tmp_path)])
    assert res.exit_code == 0 and "skip" in res.output
    assert target.read_text() == "{}"

    res = runner.invoke(app, ["init-config", "--out", str(tmp_path), "--force"])
    assert res.exit_code == 0
    assert target.read_text() != "{}"


def test_scaffolded_clif_config_parses(tmp_path):
    runner.invoke(app, ["init-config", "--out", str(tmp_path)])
    cfg = read_clif_config(str(tmp_path / "clif_config.json"))
    assert set(cfg) >= {"site", "data_directory", "filetype", "timezone",
                        "stitch_time_interval_hours"}


def test_read_clif_config_requires_timezone(tmp_path):
    path = tmp_path / "clif_config.json"
    path.write_text(json.dumps({"site": "s", "data_directory": "/tmp/x",
                                "filetype": "parquet"}))
    with pytest.raises(ValueError, match="timezone"):
        read_clif_config(str(path))


def test_cache_dir_never_inside_data_directory():
    cfg = {"data_directory": "/data/clif"}
    assert _cache_dir(cfg) == Path("output") / "_flair_cache"
    assert _cache_dir({**cfg, "cache_directory": "/tmp/c"}) == Path("/tmp/c")
