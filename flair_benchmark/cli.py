"""flair CLI — the only supported public API.

Commands:
  flair init-config  [--out DIR] [--force]
  flair load-task    <task_name>  --clif-config ... --out DIR   (always a 75/25 encounter split)
                     writes DIR/<site>_flair_phi/<task>/cohort.parquet           (stays at site)
                        and DIR/<site>_flair_non_phi_for_upload/<task>/table1.json (+ STROBE PNGs)
  flair build-report <preds.pq>   --task <task_name>  --out ./report/
  flair fe-meds      --feature-config feature_config.yaml --task cohort.parquet --out meds.parquet
  flair list-tasks

The predictions parquet for `build-report` MUST contain (strict schema, enforced
by report._validate_preds_schema):
  - hospitalization_join_id, prediction_id, split, <label_column>
  - binary tasks:      y_prob (REQUIRED; y_pred derived at 0.5 when absent)
  - multiclass tasks:  y_pred (+ class_probs_<class> for per-class curves)
  - regression tasks:  y_pred_value
hospitalization_id may be carried for audit traceability but is NOT the
report/evaluation unit.

Cohorts from `load-task` carry hospitalization_join_id (the stitched-encounter
key — one block can span several hospitalizations, see flair_benchmark.cohort.stitch)
and prediction_id, the auditable per-row key
  prediction_id = {hospitalization_join_id}-{hospitalization_id}-{patient_id}-{window_number}
(always 4 parts; window_number = 1 when the task has no window). hospitalization_join_id
is the stay-level key; prediction_id is the prediction-level key. Carry both
through your predictions parquet so the report groups by the same logical
encounter; they flow in from --cohort when present.

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
    from importlib.resources import files

    from flair_benchmark.constants import TEMPLATE_FILES

    templates = files("flair_benchmark") / "templates"
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TEMPLATE_FILES:
        content = (templates / name).read_text()
        path = out_dir / name
        if path.exists() and not force:
            typer.echo(f"  skip (exists): {path}  — pass --force to overwrite")
            continue
        path.write_text(content)
        typer.echo(f"  wrote {path}")
    typer.echo("Edit clif_config.json (data_directory, timezone), then: flair load-task ...")


@app.command("load-task")
def load_task(
    task: str = typer.Argument(..., help="Task name, e.g. icu_daily_mortality"),
    clif_config: str = typer.Option(..., "--clif-config",
                                    help="Path to clif_config.json"),
    out: str = typer.Option(..., "--out",
                            help="Output root; gets <site>_flair_phi/<task>/ and "
                                 "<site>_flair_non_phi_for_upload/<task>/"),
) -> None:
    """Run a task's build(); write the cohort (PHI folder) + Table 1 and STROBE plots (non-PHI folder)."""
    import json
    import re

    from flair_benchmark.cohort import strobe
    from flair_benchmark.cohort.clif import read_clif_config
    from flair_benchmark.cohort.table1 import generate_table1
    from flair_benchmark.report._viz import render_strobe_pngs
    from flair_benchmark.tasks import get_task

    task_module = get_task(task)
    cfg = read_clif_config(clif_config)
    # Same folder convention as flair_baseline: site-prefixed, split by sensitivity.
    # Only the non-PHI folder is meant to leave the site.
    site = re.sub(r"[^a-z0-9]+", "_", cfg["site"].strip().lower()).strip("_")
    phi = Path(out) / f"{site}_flair_phi" / task
    nonphi = Path(out) / f"{site}_flair_non_phi_for_upload" / task
    phi.mkdir(parents=True, exist_ok=True)
    nonphi.mkdir(parents=True, exist_ok=True)

    strobe.reset()
    df = task_module.build(clif_config=clif_config)
    out_path = phi / "cohort.parquet"
    df.write_parquet(out_path)
    typer.echo(f"Wrote {df.height:,} rows → {out_path}")

    # Table 1 (cohort characteristics, overall + by positive/negative label, plus
    # cohort hashes) is generated alongside every cohort for the website.
    table1 = generate_table1(cfg, df, task_module)
    # STROBE attrition: counts after every inclusion/exclusion step build() recorded.
    table1["attrition"] = strobe.steps()
    t1_path = nonphi / "table1.json"
    t1_path.write_text(json.dumps(table1, indent=2, default=str))
    typer.echo(f"Wrote Table 1 → {t1_path}")
    for png in render_strobe_pngs(table1, nonphi):
        typer.echo(f"Wrote STROBE plot → {png}")


@app.command("build-report")
def build_report_cmd(
    preds: str = typer.Argument(..., help="Predictions parquet path"),
    task: str = typer.Option(..., "--task", help="Task name"),
    out: str = typer.Option(..., "--out",
                            help="Output directory for the report JSONs"),
    mode: Optional[str] = typer.Option(
        None, "--mode",
        help="How to score the task: 'episodic' (one-per-stay) -> overall.json "
             "+ hospitalization_level.json, or 'continuous' "
             "(multi-row-per-stay) -> landmark.json + "
             "hospitalization_level.json. Inferred from META when omitted."),
    cohort: Optional[str] = typer.Option(
        None, "--cohort",
        help="Task cohort parquet (from load-task). Joins demographics for "
             "fairness when they don't flow through the preds."),
) -> None:
    """Generate the report bundle (--mode episodic | continuous) + local PNGs."""
    from flair_benchmark.report import build_report
    from flair_benchmark.tasks import get_task

    task_module = get_task(task)
    paths = build_report(preds, task_module, out,
                         mode=mode, cohort_path=cohort)
    for kind, path in paths.items():
        typer.echo(f"  {kind:24s} → {path}")


def _run_fe(extractor, feature_config: str, task: str, out: str) -> None:
    """Body for fe-meds: load config + task, extract, write."""
    import polars as pl

    from flair_benchmark.features._config import load_feature_config

    feature_cfg, clif_cfg = load_feature_config(feature_config)
    task_df = pl.read_parquet(task)
    df = extractor(task_df, feature_cfg, clif_cfg)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    typer.echo(f"Wrote {df.height:,} ELF events → {out_path}")


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
