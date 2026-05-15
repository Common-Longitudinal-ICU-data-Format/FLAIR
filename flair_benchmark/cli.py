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


@app.command("load-task")
def load_task(
    task: str = typer.Argument(..., help="Task name, e.g. task6_hospital_mortality"),
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
    out: str = typer.Option(..., "--out", help="Output directory for the 4 JSONs"),
) -> None:
    """Generate table1.json, tripod_ai.json, fairness.json, curves.json."""
    from flair_benchmark.report import build_report
    from flair_benchmark.tasks import get_task

    task_module = get_task(task)
    paths = build_report(preds, task_module, out)
    for kind, path in paths.items():
        typer.echo(f"  {kind:10s} → {path}")


@app.command("extract-metadata")
def extract_metadata_cmd(
    clif_config: str = typer.Option(..., "--clif-config",
                                    help="Path to clif_config.json"),
    out: str = typer.Option(..., "--out",
                            help="Output directory for per-task JSONs"),
    tasks: Optional[list[str]] = typer.Option(
        None, "--task",
        help="Restrict to one or more tasks (repeat the flag). Default: all 10."),
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
