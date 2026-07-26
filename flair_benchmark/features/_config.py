"""Feature-config loading + light validation.

feature_config.yaml shape::

    clif_config: clif_config_mimic.json     # path to existing JSON clif config
    output_directory: ./output/features     # optional default out dir

    meds_stream:                             # lightweight ELF pull (no conversion)
      medication_admin_continuous: {categories: [norepinephrine]}
      medication_admin_intermittent: {categories: [vancomycin, cefepime]}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from flair_benchmark._clif import read_clif_config

# CLIF 2.1 tables FLAIR v1 supports.
V1_TABLES = {
    "adt", "hospital_diagnosis", "hospitalization", "labs",
    "medication_admin_continuous", "medication_admin_intermittent",
    "microbiology_culture", "patient", "patient_assessments",
    "respiratory_support", "vitals",
}

# Category column per configurable table (used for permissible-value validation).
_CATEGORY_COLUMN = {
    "medication_admin_continuous": "med_category",
    "medication_admin_intermittent": "med_category",
}


def load_feature_config(path: str) -> tuple[dict, dict]:
    """Load feature_config.yaml; return (feature_cfg, clif_cfg).

    The ``clif_config`` path is resolved relative to the feature-config file's
    directory when not absolute.
    """
    fpath = Path(path)
    with open(fpath) as f:
        feature_cfg = yaml.safe_load(f) or {}

    clif_ref = feature_cfg.get("clif_config")
    if not clif_ref:
        raise ValueError(f"{path}: missing required key 'clif_config'")
    clif_path = Path(clif_ref)
    if not clif_path.is_absolute():
        clif_path = (fpath.parent / clif_path).resolve()
    clif_cfg = read_clif_config(str(clif_path))

    _validate(feature_cfg)
    return feature_cfg, clif_cfg


def _permissible_values(table: str) -> set[str] | None:
    """Permissible category values for a configurable table, from clifpy schema."""
    col = _CATEGORY_COLUMN.get(table)
    if col is None:
        return None
    import clifpy

    schema = Path(clifpy.__file__).parent / "schemas" / f"{table}_schema.yaml"
    try:
        with open(schema) as f:
            doc = yaml.safe_load(f)
    except FileNotFoundError:
        return None
    for field in doc.get("columns", []):
        if field.get("name") == col:
            vals = field.get("permissible_values")
            return {str(v) for v in vals} if vals else None
    return None


def _validate(feature_cfg: dict) -> None:
    """Warn (don't fail) on unsupported tables / unknown mCIDE categories."""
    for block in ("meds_stream",):
        spec = feature_cfg.get(block) or {}
        for table, tspec in spec.items():
            if table not in V1_TABLES:
                warnings.warn(
                    f"feature_config[{block}]: table '{table}' is not in the "
                    f"FLAIR v1 supported set; skipping.", stacklevel=2)
                continue
            cats = (tspec or {}).get("categories")
            if not cats:
                continue
            permissible = _permissible_values(table)
            if permissible is None:
                continue
            unknown = [c for c in cats if c not in permissible]
            if unknown:
                warnings.warn(
                    f"feature_config[{block}][{table}]: categories not in mCIDE "
                    f"permissible values {unknown}", stacklevel=2)
