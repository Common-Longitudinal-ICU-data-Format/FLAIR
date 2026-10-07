"""FLAIR: Federated Learning Assessment for ICU Research.

The supported public surface is the `flair` CLI (see cli.py). Importing the
package exposes only the version and the task-discovery helpers; everything else
is reached through the CLI or by importing a submodule directly.

The package is organized around the two halves of a benchmark run:

    cohort/     raw CLIF tables  -> a task cohort parquet  (`flair load-task`)
    features/   a cohort         -> MEDS/ELF events        (`flair fe-meds`)
    report/     predictions      -> the report JSON bundle (`flair build-report`)

    tasks/      one module per benchmark task; drop in a .py file to add one
    meds_etl/   the vendored CLIF -> MEDS ETL that features/ drives
    constants.py  version, shared constants, config templates, the audit hash

Nothing in cohort/ reads predictions, and nothing in report/ reads raw CLIF data.
"""

from flair_benchmark.constants import __version__
from flair_benchmark.tasks import get_task, list_tasks

__all__ = ["__version__", "list_tasks", "get_task"]
