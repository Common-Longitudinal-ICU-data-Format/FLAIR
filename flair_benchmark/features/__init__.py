"""FLAIR feature-engineering (FE) module.

One config-driven extractor that turns a task cohort parquet into a MEDS/ELF
long event table (``subject_id, time, code, numeric_value, unit, text_value,
hospitalization_id``), loaded via clifpy and written as parquet:

    get_meds_stream(task_df, feature_cfg, clif_cfg)
        Lightweight ELF pull of selected med categories — NO unit conversion /
        weight join / dose normalization. Raw med_dose + raw med_dose_unit so a
        downstream step can convert later.

For wide time-series data (vitals, labs, assessments, respiratory support), use
clifpy's ``ClifOrchestrator.create_wide_dataset`` directly — FLAIR no longer
wraps it. Note that clifpy applies no leakage guard: filter the result to
``time <= feature_cutoff_dttm`` from the cohort yourself.

Emitted events always carry BOTH subject_id and hospitalization_id. See
flair_benchmark/features/_elf.py:ELF_SCHEMA.
"""

from flair_benchmark.features.meds_stream import get_meds_stream

__all__ = ["get_meds_stream"]
