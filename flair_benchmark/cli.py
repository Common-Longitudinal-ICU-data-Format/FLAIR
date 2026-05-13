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


@app.command("list-tasks")
def list_tasks_cmd() -> None:
    """List discovered task names."""
    from flair_benchmark.tasks import list_tasks

    for name in list_tasks():
        typer.echo(name)


def main() -> None:
    """Entry point for `flair` script (see pyproject [project.scripts])."""
    app()


if __name__ == "__main__":
    main()
