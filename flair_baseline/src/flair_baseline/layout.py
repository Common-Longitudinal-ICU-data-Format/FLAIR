"""Site-prefixed, sensitivity-partitioned output layout.

Every run writes into three sibling folders under one ``--out`` root, each
prefixed with the site name (from the clif config ``site`` field):

  <site>_baseline_phi/<task>/                 cohort.parquet, MEDS.parquet, preds.parquet
  <site>_baseline_non_phi_for_upload/<task>/  codes.parquet, table1.json, report/…
  <site>_baseline_models/<task>/              model.json, vocab.json

Only the non-PHI folder is meant to leave the site. Models are what `train`
ships to other sites; `infer` reads them back in and regenerates the PHI +
non-PHI folders locally against the new site's data.

`TaskPaths` is the single owner of the path mapping so `train` and `infer`
route writes identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from flair_benchmark._constants import TASK_POLICY


def report_mode(task_name: str) -> str:
    """Report mode for a task — single-sourced from flair_benchmark/_constants.py.

    task1/task2 = landmark (per lead-time), task4 = peak (stay-peak screening),
    episode tasks (3, 5) = episodic. Anything unlisted defaults to episodic.
    """
    return TASK_POLICY.get(task_name, {}).get("report_mode", "episodic")


def slug(site: str | None) -> str:
    """Folder-safe site token. Raises when empty so folders are never un-prefixed."""
    s = re.sub(r"[^a-z0-9]+", "_", (site or "").strip().lower()).strip("_")
    if not s:
        raise ValueError(
            "clif config has no usable 'site' name; set \"site\" in the config "
            "(it prefixes every output folder).")
    return s


@dataclass(frozen=True)
class TaskPaths:
    """Absolute paths for one task's artifacts across the three folders."""

    out_root: Path
    site: str
    task: str

    @classmethod
    def make(cls, out_root: str | Path, site: str, task: str) -> "TaskPaths":
        return cls(Path(out_root), slug(site), task)

    # --- roots (site-prefixed) ---------------------------------------------
    @property
    def phi_root(self) -> Path:
        return self.out_root / f"{self.site}_baseline_phi" / self.task

    @property
    def nonphi_root(self) -> Path:
        return self.out_root / f"{self.site}_baseline_non_phi_for_upload" / self.task

    @property
    def models_root(self) -> Path:
        return self.out_root / f"{self.site}_baseline_models" / self.task

    # --- PHI (stays local) -------------------------------------------------
    @property
    def cohort(self) -> Path:
        return self.phi_root / "cohort.parquet"

    @property
    def meds(self) -> Path:
        return self.phi_root / "data" / "MEDS.parquet"

    @property
    def preds(self) -> Path:
        return self.phi_root / "preds.parquet"

    # --- non-PHI (uploaded) ------------------------------------------------
    @property
    def codes(self) -> Path:
        return self.nonphi_root / "codes.parquet"

    @property
    def table1(self) -> Path:
        return self.nonphi_root / "table1.json"

    @property
    def report_dir(self) -> Path:
        return self.nonphi_root / "report"

    # --- models (shipped to other sites) -----------------------------------
    @property
    def model(self) -> Path:
        return self.models_root / "model.json"

    @property
    def vocab(self) -> Path:
        return self.models_root / "vocab.json"

    def mkdirs(self) -> None:
        """Create every parent directory this task writes into."""
        for p in (self.meds.parent, self.preds.parent, self.codes.parent,
                  self.report_dir, self.models_root):
            p.mkdir(parents=True, exist_ok=True)
