"""Feature engineering: task cohort -> MEDS/ELF long event table.

One config-driven extractor is exposed here:

    get_meds_stream(task_df, feature_cfg, clif_cfg)
        Lightweight ELF pull of selected med categories. NO unit conversion,
        weight join, or dose normalization — raw med_dose + raw med_dose_unit,
        so a downstream step can convert later.

The heavier full CLIF->MEDS path (`build_meds_tables`, `build_shared_meds`) lives
in extractors.py alongside it; import it from there directly.

For WIDE time-series data (vitals, labs, assessments, respiratory support), use
clifpy's `ClifOrchestrator.create_wide_dataset` directly — FLAIR does not wrap
it. Note that clifpy applies NO leakage guard: filter its result to
`event_time <= feature_cutoff_dttm` from the cohort yourself.

Emitted events always carry subject_id, hospitalization_id AND
hospitalization_join_id. See features/_config.py:ELF_SCHEMA.
"""

from flair_benchmark.features.extractors import get_meds_stream

__all__ = ["get_meds_stream"]
