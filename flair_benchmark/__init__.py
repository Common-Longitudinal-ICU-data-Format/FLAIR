"""FLAIR: Federated Learning Assessment for ICU Research.

Public surface is the `flair` CLI (see `flair_benchmark.cli`). Importing
the package only exposes the version + task discovery helpers; everything
else lives behind the CLI.
"""

from flair_benchmark._constants import __version__

from flair_benchmark.tasks import get_task, list_tasks

__all__ = ["__version__", "list_tasks", "get_task"]
