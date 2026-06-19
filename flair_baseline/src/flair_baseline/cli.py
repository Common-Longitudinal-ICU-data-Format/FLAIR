"""flair-baseline CLI — two subcommands, site-prefixed 3-folder output.

  flair-baseline train --clif-config clif.json --elf-config flair_elf_config.yaml --out .
  flair-baseline train --task task1 ...                       # single task
  flair-baseline infer --task task1 --clif-config site.json --models-dir mimic_baseline_models

`train` fits XGBoost on the configured site (75/25 join-id split on MIMIC) and writes the
model + vocab into `<site>_baseline_models/`. That folder (plus this code) is all that ships
to another site. `infer` reads those models back in and re-runs the baseline against the new
site's own data, evaluating on a deterministic 25% holdout — regenerating that site's
`<site>_baseline_phi/` (stays local) and `<site>_baseline_non_phi_for_upload/` (shared back).

Every run partitions artifacts by sensitivity (see layout.py):
  <site>_baseline_phi/<task>/            cohort.parquet, MEDS.parquet, preds.parquet
  <site>_baseline_non_phi_for_upload/…   codes.parquet, table1.json, report/*.json (+ viz)
  <site>_baseline_models/<task>/         model.json, vocab.json
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import typer

from flair_baseline.config import DEFAULT_ELF_CONFIG, resolve_task
from flair_baseline.featurize import count_features
from flair_baseline.layout import TaskPaths, report_mode
from flair_baseline.train import train_and_score

DEFAULT_CLIF_CONFIG = "config/clif_config.template.json"

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


def _relocate(src: Path, dst: Path) -> None:
    """Move a freshly-written file to its final (cross-folder) home."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.replace(dst)


def _stitch(clif_config: str):
    """Build/load the encounter index once so hospitalization_join_id exists."""
    from flair_benchmark._clif import read_clif_config
    from flair_benchmark._stitch import load_or_build_encounter_index
    cfg = read_clif_config(clif_config)
    idx = load_or_build_encounter_index(cfg, cfg.get("stitch_time_interval_hours", 6))
    n_blocks = idx["hospitalization_join_id"].n_unique()
    typer.echo(f"[stitch] encounter index ready → {n_blocks:,} encounter blocks "
               f"from {idx.height:,} hospitalizations")
    return cfg


def _pipeline(task_name: str, clif_config: str, elf_config: str, paths: TaskPaths,
              *, train_end: Optional[str] = None, test_start: Optional[str] = None,
              model_in: Optional[str] = None, vocab_list: Optional[list] = None,
              force_holdout: bool = False, report: bool = True,
              viz: bool = False, reuse: bool = False) -> None:
    """Cohort → Table 1 → MEDS/codes → featurize → train|score → report.

    Writes land in the three site-prefixed folders via `paths`. When `model_in`
    is set the model is scored (external validation) instead of fit; when
    `force_holdout` is set the cohort is re-split 75/25 at the block grain so the
    reported 'test' split is a deterministic 25% of the site's data.
    """
    import json

    from flair_benchmark._clif import read_clif_config
    from flair_benchmark.features.fe_meds import build_meds_tables
    from flair_benchmark.tasks import get_task

    task_module = get_task(task_name)
    label_col = task_module.META["label_column"]
    paths.mkdirs()

    if reuse and paths.cohort.exists() and paths.meds.exists():
        typer.echo(f"[{task_name}] reusing existing cohort + MEDS table")
        cohort = pl.read_parquet(paths.cohort)
    else:
        from flair_benchmark._table1 import generate_table1
        typer.echo(f"[{task_name}] building cohort …")
        cohort = task_module.build(clif_config=clif_config,
                                   train_end=train_end, test_start=test_start)
        if force_holdout:
            # External site: drop the all-'test' split and force the same
            # deterministic 75/25 block-grain split MIMIC uses, so the reported
            # 'test' set is 25% of this site's data (ordered by prediction_dttm,
            # which the cohort always carries; join_id is the block grain).
            from flair_benchmark._split import assign_split
            cohort = assign_split(cohort.drop("split"), site="mimic",
                                  train_end=None, test_start=None,
                                  admission_col="prediction_dttm")
        cohort.write_parquet(paths.cohort)

        table1 = generate_table1(read_clif_config(clif_config), cohort, task_module)
        paths.table1.write_text(json.dumps(table1, indent=2, default=str))
        typer.echo(f"[{task_name}] Table 1 → {paths.table1}")

        typer.echo(f"[{task_name}] FE-meds …")
        # MEDS (PHI) + codes (non-PHI) are written under one dir by the lib;
        # write into the PHI root, then relocate codes.parquet to the upload folder.
        build_meds_tables(cohort, read_clif_config(clif_config), elf_config,
                          str(paths.phi_root))
        _relocate(paths.phi_root / "metadata" / "codes.parquet", paths.codes)

    n_train = cohort.filter(pl.col("split") == "train").height
    n_test = cohort.filter(pl.col("split") == "test").height
    typer.echo(f"[{task_name}] cohort: {cohort.height:,} rows "
               f"({n_train:,} train / {n_test:,} test)")

    typer.echo(f"[{task_name}] count featurization …")
    events_lf = pl.scan_parquet(paths.meds)
    if model_in:
        X, ids, _ = count_features(events_lf, cohort, label_col, vocab=vocab_list)
        typer.echo(f"[{task_name}] feature matrix: {X.shape[0]:,} × {X.shape[1]:,} "
                   f"({X.nnz:,} nnz)")
        typer.echo(f"[{task_name}] scoring with shipped model …")
        preds = train_and_score(X, ids, vocab_list, cohort, label_col, model_in=model_in)
    else:
        X, ids, vocab = count_features(events_lf, cohort, label_col)
        typer.echo(f"[{task_name}] feature matrix: {X.shape[0]:,} × {X.shape[1]:,} "
                   f"({X.nnz:,} nnz)")
        typer.echo(f"[{task_name}] training XGBoost …")
        preds = train_and_score(X, ids, vocab, cohort, label_col,
                                model_out=str(paths.model), vocab_out=str(paths.vocab))
    preds.write_parquet(paths.preds)

    auc_tr, auc_te = _auroc(preds, label_col, "train"), _auroc(preds, label_col, "test")
    typer.echo(f"[{task_name}] AUROC  train={auc_tr}  test={auc_te}")

    if report:
        from flair_benchmark.report import build_report
        site = read_clif_config(clif_config).get("site")
        mode = report_mode(task_name)
        build_report(str(paths.preds), task_module, str(paths.report_dir),
                     cohort_path=str(paths.cohort), viz=viz, site=site, mode=mode)
        typer.echo(f"[{task_name}] report ({mode}){' + viz' if viz else ''} "
                   f"→ {paths.report_dir}")


@app.command("train")
def train_cmd(
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config",
                                    help="clif_config.json (site=mimic for 75/25 split)"),
    elf_config: str = typer.Option(DEFAULT_ELF_CONFIG, "--elf-config",
                                   help="flair_elf_config.yaml"),
    out: str = typer.Option(".", "--out", help="Root dir for the three site-prefixed folders"),
    task: Optional[str] = typer.Option(None, "--task", help="Task name/prefix; default = all 5"),
    train_end: Optional[str] = typer.Option(None, "--train-end"),
    test_start: Optional[str] = typer.Option(None, "--test-start"),
    report: bool = typer.Option(True, "--report/--no-report", help="Also build the report bundle"),
    viz: bool = typer.Option(False, "--viz/--no-viz", help="Render sanity-check PNGs"),
    reuse: bool = typer.Option(False, "--reuse/--no-reuse",
                               help="Reuse existing cohort.parquet + MEDS.parquet when present"),
) -> None:
    """Train the baseline on the configured site; models land in <site>_baseline_models/."""
    from flair_benchmark.tasks import list_tasks

    cfg = _stitch(clif_config)
    site = cfg.get("site")
    tasks = [resolve_task(task)] if task else list_tasks()
    for t in tasks:
        paths = TaskPaths.make(out, site, t)
        _pipeline(t, clif_config, elf_config, paths, train_end=train_end,
                  test_start=test_start, report=report, viz=viz, reuse=reuse)


@app.command("infer")
def infer_cmd(
    models_dir: str = typer.Option(..., "--models-dir",
                                   help="A shipped <site>_baseline_models dir (per-task model.json/vocab.json)"),
    clif_config: str = typer.Option(DEFAULT_CLIF_CONFIG, "--clif-config",
                                    help="This site's clif_config.json (sets the folder prefix)"),
    elf_config: str = typer.Option(DEFAULT_ELF_CONFIG, "--elf-config"),
    out: str = typer.Option(".", "--out", help="Root dir for the three site-prefixed folders"),
    task: Optional[str] = typer.Option(None, "--task", help="Task name/prefix; default = all 5"),
    report: bool = typer.Option(True, "--report/--no-report"),
    viz: bool = typer.Option(False, "--viz/--no-viz"),
) -> None:
    """Score a shipped model on this site's data; report on a deterministic 25% test split."""
    import json

    from flair_benchmark.tasks import list_tasks

    cfg = _stitch(clif_config)
    site = cfg.get("site")
    models_root = Path(models_dir)
    tasks = [resolve_task(task)] if task else list_tasks()
    for t in tasks:
        model_path = models_root / t / "model.json"
        vocab_path = models_root / t / "vocab.json"
        if not (model_path.exists() and vocab_path.exists()):
            typer.echo(f"[{t}] no model under {models_root / t} — skipping")
            continue
        vocab_list = json.loads(vocab_path.read_text())["vocab"]
        paths = TaskPaths.make(out, site, t)
        _pipeline(t, clif_config, elf_config, paths, model_in=str(model_path),
                  vocab_list=vocab_list, force_holdout=True, report=report, viz=viz)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
