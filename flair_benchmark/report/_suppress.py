"""HIPAA cell-count suppression: any count < 10 → '<10'."""

from __future__ import annotations

MIN_CELL_COUNT = 10


def suppress(count: int, threshold: int = MIN_CELL_COUNT) -> int | str:
    """Return the count, or the string '<{threshold}' if below threshold."""
    return count if count >= threshold else f"<{threshold}"


def suppress_dict(counts: dict, threshold: int = MIN_CELL_COUNT) -> dict:
    """Vectorized version for {category: count} dicts."""
    return {k: suppress(int(v), threshold) for k, v in counts.items()}
