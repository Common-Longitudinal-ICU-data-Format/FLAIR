"""Per-category unit lookup.

FE does no unit conversion, so every numeric ELF event must chart the unit it
was recorded in (so a later step can convert). Units are deterministic per
mCIDE category:

  - labs:   read ``lab_reference_units`` from clifpy's labs_schema.yaml.
  - vitals: mCIDE standard units (fixed, 9 categories).
  - respiratory_support: fixed units for the numeric setting columns.
  - meds (wide path): ambiguous per-category; left None here — use get_meds_stream
    for meds, which charts the real per-row med_dose_unit.

A feature_config may override any of these via a top-level ``units:`` block.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

# mCIDE standard vital units.
VITAL_UNITS: dict[str, str] = {
    "heart_rate": "bpm",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "map": "mmHg",
    "spo2": "%",
    "respiratory_rate": "breaths/min",
    "temp_c": "celsius",
    "height_cm": "cm",
    "weight_kg": "kg",
}

# respiratory_support numeric setting columns -> unit. Categorical columns
# (device_category, mode_category, tracheostomy) carry no unit (text events).
RESP_UNITS: dict[str, str] = {
    "fio2_set": "fraction",
    "lpm_set": "L/min",
    "peep_set": "cmH2O",
    "tidal_volume_set": "mL",
    "resp_rate_set": "breaths/min",
    "pressure_support_set": "cmH2O",
    "peak_inspiratory_pressure_set": "cmH2O",
    "minute_vent_obs": "L/min",
}
RESP_CATEGORICAL = {"device_category", "mode_category", "tracheostomy"}


@functools.lru_cache(maxsize=1)
def _lab_units() -> dict[str, str]:
    """lab_category -> reference unit, from clifpy labs_schema.yaml.

    '(no units)' and blanks map to no entry (None unit).
    """
    import clifpy

    schema = Path(clifpy.__file__).parent / "schemas" / "labs_schema.yaml"
    try:
        with open(schema) as f:
            doc = yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    raw = doc.get("lab_reference_units", {}) or {}
    out: dict[str, str] = {}
    for cat, unit in raw.items():
        if unit is None:
            continue
        u = str(unit).strip()
        if not u or u.lower() == "(no units)":
            continue
        out[cat] = u
    return out


def unit_for(table: str, category: str, overrides: dict | None = None) -> str | None:
    """Standard unit for a (table, category), or None.

    ``overrides`` is the optional feature_config ``units`` block, shaped
    ``{table: {category: unit}}``; it wins over the built-in maps.
    """
    if overrides:
        tbl = overrides.get(table) or {}
        if category in tbl:
            return tbl[category]
    if table == "labs":
        return _lab_units().get(category)
    if table == "vitals":
        return VITAL_UNITS.get(category)
    if table == "respiratory_support":
        return RESP_UNITS.get(category)
    return None
