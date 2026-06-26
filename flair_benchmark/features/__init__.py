"""FLAIR feature-engineering (FE) module.

Two config-driven extractors that turn a task cohort parquet into MEDS/ELF
long event tables (``subject_id, time, code, numeric_value, unit, text_value,
hospitalization_id``), loaded via clifpy and written as parquet:

    get_wide_feature_dataset(task_df, feature_cfg, clif_cfg)
        All configured time-series features via clifpy create_wide_dataset,
        truncated at each stay's prediction horizon, melted to ELF long.

    get_meds_stream(task_df, feature_cfg, clif_cfg)
        Lightweight ELF pull of selected med categories — NO unit conversion /
        weight join / dose normalization. Raw med_dose + raw med_dose_unit so a
        downstream step can convert later.

Both emit the same ELF schema and always carry BOTH subject_id and
hospitalization_id. See flair_benchmark/features/_elf.py:ELF_SCHEMA.
"""

from flair_benchmark.features.meds_stream import get_meds_stream
from flair_benchmark.features.wide import get_wide_feature_dataset

__all__ = ["get_wide_feature_dataset", "get_meds_stream"]
