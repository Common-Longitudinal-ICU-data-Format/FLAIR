"""Small helpers: task-name resolution and the default ELF config path."""
from __future__ import annotations

from flair_benchmark.tasks import list_tasks

DEFAULT_ELF_CONFIG = "flair_elf_config.yaml"


def resolve_task(name: str) -> str:
    """Accept a full task name or a short prefix (e.g. 'task1') → full task name."""
    tasks = list_tasks()
    if name in tasks:
        return name
    matches = [t for t in tasks if t == name or t.split("_")[0] == name or t.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Unknown task {name!r}. Available: {tasks}")
    raise ValueError(f"Ambiguous task {name!r} → {matches}. Use the full name.")
