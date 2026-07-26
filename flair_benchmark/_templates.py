"""Config templates shipped inside the package.

Stored as string constants (not data files) so they are guaranteed present in
the PyPI wheel and a fresh `pip install flair-benchmark` user can scaffold a
working config entirely through the CLI: `flair init-config`.

These string constants are the single source of truth for the config scaffold
(`flair init-config` writes them out). Keep FEATURE_CONFIG_TEMPLATE in sync with
feature_config.yaml at the repo root.
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
# FLAIR feature-engineering config (drives `flair fe-meds`).
# Categories are CLIF 2.1 mCIDE values; unknown ones warn but don't fail.
#
# For wide time-series data (vitals, labs, assessments, respiratory support) use
# clifpy's ClifOrchestrator.create_wide_dataset directly — and filter its output
# to `time <= feature_cutoff_dttm` yourself; clifpy applies no leakage guard.
clif_config: clif_config.json              # the JSON clif config (site/data_dir/tz)
output_directory: ./output/features        # convention; --out overrides

# Lightweight med event stream: raw dose + raw unit, NO unit-conversion ETL.
meds_stream:
  medication_admin_continuous:
    categories: [norepinephrine, propofol]
  medication_admin_intermittent:
    categories: [vancomycin, cefepime, piperacillin_tazobactam]
"""

# Filename -> template, written by `flair init-config`.
TEMPLATES = {
    "clif_config.json": CLIF_CONFIG_TEMPLATE,
    "feature_config.yaml": FEATURE_CONFIG_TEMPLATE,
}
