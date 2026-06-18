"""flair-baseline CLI.

  flair-baseline run   --clif-config clif.json --elf-config flair_elf_config.yaml --out out
  flair-baseline run   --task task1 ...                       # single task
  flair-baseline score --task task1 --clif-config other_site.json --model out/<task>/model.json ...

`run` trains on the configured site (75/25 join-id split on MIMIC). `score` applies a
saved model to another site's full cohort (no split → all 'test').
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import typer

from flair_baseline.config import DEFAULT_ELF_CONFIG, resolve_task
from flair_baseline.featurize import count_features
from flair_baseline.train import train_and_score

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="XGBoost count-feature baseline for FLAIR tasks")


def _auroc(preds: pl.DataFrame, label_col: str, split: str) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    d = preds.filter(pl.col("split") == split)
    if d.height == 0:
        return None
    y = d[label_col].to_numpy()
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, d["y_prob"].to_numpy()))


def _build_cohort(task_module, clif_config, train_end, test_start) -> pl.DataFrame:
    return task_module.build(clif_config=clif_config, train_end=train_end, test_start=test_start)


def _run_one(task_name: str, clif_config: str, elf_config: str, out_root: str,
             train_end: Optional[str], test_start: Optional[str], report: bool,
             viz: bool, reuse: bool) -> None:
    from flair_benchmark.features.fe_meds import build_meds_tables
    from flair_benchmark.tasks import get_task

    task_module = get_task(task_name)
    label_col = task_module.META["label_column"]
    out = Path(out_root) / task_name
    out.mkdir(parents=True, exist_ok=True)
    cohort_path = out / "cohort.parquet"
    meds_path = out / "data" / "MEDS.parquet"

    table1_path = out / "table1.json"

    if reuse and cohort_path.exists() and meds_path.exists():
        typer.echo(f"[{task_name}] reusing existing cohort + MEDS table")
        cohort = pl.read_parquet(cohort_path)
    else:
        import json

        from flair_benchmark._clif import read_clif_config
        from flair_benchmark._table1 import generate_table1
        typer.echo(f"[{task_name}] building cohort …")
        cohort = _build_cohort(task_module, clif_config, train_end, test_start)
        cohort.write_parquet(cohort_path)
        # Table 1 (cohort characteristics, overall + by positive/negative label).
        table1 = generate_table1(read_clif_config(clif_config), cohort, task_module)
        table1_path.write_text(json.dumps(table1, indent=2, default=str))
        typer.echo(f"[{task_name}] Table 1 → {table1_path}")
        typer.echo(f"[{task_name}] FE-meds …")
        build_meds_tables(cohort, read_clif_config(clif_config), elf_config, str(out))

    n_train = cohort.filter(pl.col("split") == "train").height
    n_test = cohort.filter(pl.col("split") == "test").height
    typer.echo(f"[{task_name}] cohort: {cohort.height:,} rows ({n_train:,} train / {n_test:,} test)")

    typer.echo(f"[{task_name}] count featurization …")
    events_lf = pl.scan_parquet(meds_path)
    X, ids, vocab = count_features(events_lf, cohort, label_col)
    typer.echo(f"[{task_name}] feature matrix: {X.shape[0]:,} × {X.shape[1]:,} ({X.nnz:,} nnz)")

    typer.echo(f"[{task_name}] training XGBoost …")
    preds = train_and_score(X, ids, vocab, cohort, label_col,
                            model_out=str(out / "model.json"),
                            vocab_out=str(out / "vocab.json"))
    preds_path = out / "preds.parquet"
    preds.write_parquet(preds_path)

    auc_tr, auc_te = _auroc(preds, label_col, "train"), _auroc(preds, label_col, "test")
    typer.echo(f"[{task_name}] AUROC  train={auc_tr}  test={auc_te}")

    if report:
        from flair_benchmark._clif import read_clif_config
        from flair_benchmark.report import build_report
        site = read_clif_config(clif_config).get("site")
        build_report(str(preds_path), task_module, str(out / "report"),
                     cohort_path=str(cohort_path), viz=viz, site=site)
        typer.echo(f"[{task_name}] report{' + viz' if viz else ''} → {out / 'report'}")


@app.command("run")
def run_cmd(
    clif_config: str = typer.Option(..., "--clif-config", help="clif_config.json (site=mimic for 75/25 split)"),
    elf_config: str = typer.Option(DEFAULT_ELF_CONFIG, "--elf-config", help="flair_elf_config.yaml"),
    out: str = typer.Option("out", "--out", help="Output root directory"),
    task: Optional[str] = typer.Option(None, "--task", help="Task name/prefix; default = all 5"),
    train_end: Optional[str] = typer.Option(None, "--train-end"),
    test_start: Optional[str] = typer.Option(None, "--test-start"),
    report: bool = typer.Option(True, "--report/--no-report", help="Also build the FLAIR report bundle"),
    viz: bool = typer.Option(False, "--viz/--no-viz", help="Render sanity-check PNGs from the report JSONs"),
    reuse: bool = typer.Option(False, "--reuse/--no-reuse",
                               help="Reuse existing cohort.parquet + MEDS.parquet (skip ETL) when present"),
) -> None:
    """Train + score the baseline for one or all tasks."""
    from flair_benchmark._clif import read_clif_config
    from flair_benchmark._stitch import load_or_build_encounter_index
    from flair_benchmark.tasks import list_tasks

    # Stitch encounters FIRST: build/load the encounter index once up front so the
    # hospitalization_join_id mapping exists before any task builds its cohort.
    # Cached to parquet, so every task reuses this single stitched mapping.
    cfg = read_clif_config(clif_config)
    idx = load_or_build_encounter_index(cfg, cfg.get("stitch_time_interval_hours", 6))
    n_blocks = idx["hospitalization_join_id"].n_unique()
    typer.echo(f"[stitch] encounter index ready → {n_blocks:,} encounter blocks "
               f"from {idx.height:,} hospitalizations")

    tasks = [resolve_task(task)] if task else list_tasks()
    for t in tasks:
        _run_one(t, clif_config, elf_config, out, train_end, test_start, report, viz, reuse)


@app.command("score")
def score_cmd(
    task: str = typer.Option(..., "--task", help="Task name/prefix"),
    clif_config: str = typer.Option(..., "--clif-config", help="External site clif_config.json"),
    model: str = typer.Option(..., "--model", help="Saved model.json from a prior run"),
    vocab: str = typer.Option(..., "--vocab", help="Saved vocab.json from a prior run"),
    elf_config: str = typer.Option(DEFAULT_ELF_CONFIG, "--elf-config"),
    out: str = typer.Option("out_external", "--out"),
) -> None:
    """Apply a saved model to another site's full cohort (all rows scored as 'test')."""
    import json

    from flair_benchmark._clif import read_clif_config
    from flair_benchmark.features.fe_meds import build_meds_tables
    from flair_benchmark.tasks import get_task

    task_name = resolve_task(task)
    task_module = get_task(task_name)
    label_col = task_module.META["label_column"]
    out_dir = Path(out) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    clif_cfg = read_clif_config(clif_config)
    cohort = _build_cohort(task_module, clif_config, None, None)  # non-mimic → all 'test'
    cohort.write_parquet(out_dir / "cohort.parquet")
    build_meds_tables(cohort, clif_cfg, elf_config, str(out_dir))

    vocab_list = json.loads(Path(vocab).read_text())["vocab"]
    events_lf = pl.scan_parquet(out_dir / "data" / "MEDS.parquet")
    X, ids, _ = count_features(events_lf, cohort, label_col, vocab=vocab_list)
    preds = train_and_score(X, ids, vocab_list, cohort, label_col, model_in=model)
    preds.write_parquet(out_dir / "preds.parquet")
    auc = _auroc(preds, label_col, "test")
    typer.echo(f"[{task_name}] external AUROC test={auc} → {out_dir / 'preds.parquet'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
