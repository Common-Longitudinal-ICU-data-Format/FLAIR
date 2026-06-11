"""Config templates shipped inside the package.

Stored as string constants (not data files) so they are guaranteed present in
the PyPI wheel and a fresh `pip install flair-benchmark` user can scaffold a
working config entirely through the CLI: `flair init-config`.

Keep CLIF_CONFIG_TEMPLATE in sync with clif_config_template.json and
FEATURE_CONFIG_TEMPLATE in sync with feature_config.yaml at the repo root.
"""

CLIF_CONFIG_TEMPLATE = """\
{
    "site": "sitename",
    "data_directory": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central",
    "stitch_time_interval_hours": 6,
    "cache_directory": "./output"
}
"""

FEATURE_CONFIG_TEMPLATE = """\
# FLAIR feature-engineering config (drives `flair fe-wide` / `flair fe-meds`).
# Categories are CLIF 2.1 mCIDE values; unknown ones warn but don't fail.
clif_config: clif_config.json              # the JSON clif config (site/data_dir/tz)
output_directory: ./output/features        # convention; --out overrides

# Wide time-series features -> clifpy create_wide_dataset -> melted to ELF long.
wide:
  vitals:
    categories: [heart_rate, sbp, dbp, map, spo2, respiratory_rate, temp_c]
  labs:
    categories: [creatinine, lactate, wbc, hemoglobin, platelet_count, sodium, potassium]
  medication_admin_continuous:
    categories: [norepinephrine, epinephrine, vasopressin, propofol]
  patient_assessments:
    categories: [gcs_total, RASS]
  respiratory_support:                     # clifpy 'wide' table -> column names
    columns: [device_category, mode_category, fio2_set, peep_set]

# Lightweight med event stream: raw dose + raw unit, NO unit-conversion ETL.
meds_stream:
  medication_admin_continuous:
    categories: [norepinephrine, propofol]
  medication_admin_intermittent:
    categories: [vancomycin, cefepime, piperacillin_tazobactam]

# Optional per-category unit overrides (table -> category -> unit).
# units:
#   labs:
#     creatinine: mg/dl
"""

# Filename -> template, written by `flair init-config`.
TEMPLATES = {
    "clif_config.json": CLIF_CONFIG_TEMPLATE,
    "feature_config.yaml": FEATURE_CONFIG_TEMPLATE,
}
