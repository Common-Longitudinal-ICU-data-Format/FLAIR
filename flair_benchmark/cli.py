"""flair CLI — the only supported public API.

Commands:
  flair load-task    <task_name>  --clif-config ... [--train-end ...] [--test-start ...] --out cohort.parquet
  flair build-report <preds.pq>   --task <task_name>  --out ./report/
  flair list-tasks

The predictions parquet for `build-report` MUST contain (strict schema):
  - hospitalization_id, split, <label_column>
  - binary tasks:      y_pred, y_prob (recommended)
  - multiclass tasks:  y_pred, class_probs_<class>
  - regression tasks:  y_pred_value

Demographic columns flow through from `load-task` (age_at_admission,
sex_category, race_category, ethnicity_category) and are required for
fairness analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="FLAIR: CLIF-based federated ICU benchmark CLI")


@app.command("init-config")
def init_config_cmd(
    out: str = typer.Option(".", "--out", help="Directory to write config templates into"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
) -> None:
    """Scaffold clif_config.json + feature_config.yaml templates.

    Lets a fresh `pip install flair-benchmark` user start entirely from the CLI.
    Edit clif_config.json with your data path/timezone, then run load-task / fe-*.
    """
    from flair_benchmark._templates import TEMPLATES

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in TEMPLATES.items():
        path = out_dir / name
        if path.exists() and not force:
            typer.echo(f"  skip (exists): {path}  — pass --force to overwrite")
            continue
        path.write_text(content)
        typer.echo(f"  wrote {path}")
    typer.echo("Edit clif_config.json (data_directory, timezone), then: flair load-task ...")


@app.command("load-task")
def load_task(
    task: str = typer.Argument(..., help="Task name, e.g. task1_icu_daily_mortality"),
    clif_config: str = typer.Option(..., "--clif-config",
                                    help="Path to clif_config.json"),
    out: str = typer.Option(..., "--out", help="Output parquet path"),
    train_end: Optional[str] = typer.Option(None, "--train-end",
                                            help="YYYY-MM-DD (omit for MIMIC 70/30 split)"),
    test_start: Optional[str] = typer.Option(None, "--test-start",
                                             help="YYYY-MM-DD (omit for MIMIC 70/30 split)"),
) -> None:
    """Run a task's build() function and write the cohort parquet."""
    from flair_benchmark.tasks import get_task

    task_module = get_task(task)
    df = task_module.build(clif_config=clif_config,
                           train_end=train_end,
                           test_start=test_start)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    typer.echo(f"Wrote {df.height:,} rows → {out_path}")


@app.command("build-report")
def build_report_cmd(
    preds: str = typer.Argument(..., help="Predictions parquet path"),
    task: str = typer.Option(..., "--task", help="Task name"),
    out: str = typer.Option(..., "--out", help="Output directory for the 6 JSONs"),
    cohort: Optional[str] = typer.Option(
        None, "--cohort",
        help="Task cohort parquet (from load-task). Enables window/year binning "
             "when window_number/prediction_dttm don't flow through the preds."),
    viz: bool = typer.Option(
        False, "--viz",
        help="Also write sanity-check PNGs rendered from the JSONs "
             "(requires `pip install flair-benchmark[viz]`)."),
) -> None:
    """Generate table1, tripod_ai, fairness, curves, temporal and dca JSONs."""
    from flair_benchmark.report import build_report
    from flair_benchmark.tasks import get_task

    task_module = get_task(task)
    paths = build_report(preds, task_module, out, cohort_path=cohort, viz=viz)
    for kind, path in paths.items():
        typer.echo(f"  {kind:24s} → {path}")


@app.command("extract-metadata")
def extract_metadata_cmd(
    clif_config: str = typer.Option(..., "--clif-config",
                                    help="Path to clif_config.json"),
    out: str = typer.Option(..., "--out",
                            help="Output directory for per-task JSONs"),
    tasks: Optional[list[str]] = typer.Option(
        None, "--task",
        help="Restrict to one or more tasks (repeat the flag). Default: all 5."),
    skip_sofa: bool = typer.Option(False, "--skip-sofa",
                                   help="Skip SOFA (clifpy ClifOrchestrator is heavy)."),
    skip_comorb: bool = typer.Option(False, "--skip-comorb",
                                     help="Skip Charlson/Elixhauser if HospitalDiagnosis "
                                          "is unavailable at the site."),
) -> None:
    """Per-site cohort metadata for the FLAIR leaderboard. Writes one
    <task_name>.json into the output directory."""
    import json

    from flair_benchmark.metadata import extract_site_metadata

    per_task = extract_site_metadata(clif_config, tasks, skip_sofa, skip_comorb)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_name, payload in per_task.items():
        path = out_dir / f"{task_name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        typer.echo(f"  {task_name:32s} → {path}")


def _run_fe(extractor, feature_config: str, task: str, out: str) -> None:
    """Shared body for fe-wide / fe-meds: load config + task, extract, write."""
    import polars as pl

    from flair_benchmark.features._config import load_feature_config

    feature_cfg, clif_cfg = load_feature_config(feature_config)
    task_df = pl.read_parquet(task)
    df = extractor(task_df, feature_cfg, clif_cfg)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    typer.echo(f"Wrote {df.height:,} ELF events → {out_path}")


@app.command("fe-wide")
def fe_wide_cmd(
    feature_config: str = typer.Option(..., "--feature-config",
                                       help="Path to feature_config.yaml"),
    task: str = typer.Option(..., "--task", help="Task cohort parquet (from load-task)"),
    out: str = typer.Option(..., "--out", help="Output ELF parquet path"),
) -> None:
    """Wide time-series features → MEDS/ELF long, truncated at the prediction point."""
    from flair_benchmark.features import get_wide_feature_dataset
    _run_fe(get_wide_feature_dataset, feature_config, task, out)


@app.command("fe-meds")
def fe_meds_cmd(
    feature_config: str = typer.Option(..., "--feature-config",
                                       help="Path to feature_config.yaml"),
    task: str = typer.Option(..., "--task", help="Task cohort parquet (from load-task)"),
    out: str = typer.Option(..., "--out", help="Output ELF parquet path"),
) -> None:
    """Lightweight med event stream (raw dose + raw unit) → MEDS/ELF long."""
    from flair_benchmark.features import get_meds_stream
    _run_fe(get_meds_stream, feature_config, task, out)


@app.command("list-tasks")
def list_tasks_cmd() -> None:
    """List discovered task names with taxonomy columns."""
    from flair_benchmark.tasks import get_task, list_tasks

    header = (f"{'name':32s}  {'task_class':14s}  "
              f"{'temporal_pattern':16s}  {'prediction_unit':24s}  task_type")
    typer.echo(header)
    typer.echo("-" * len(header))
    for name in list_tasks():
        m = get_task(name).META
        typer.echo(
            f"{name:32s}  {m['task_class']:14s}  "
            f"{m['temporal_pattern']:16s}  {m['prediction_unit']:24s}  "
            f"{m['task_type']}"
        )


def main() -> None:
    """Entry point for `flair` script (see pyproject [project.scripts])."""
    app()


if __name__ == "__main__":
    main()
