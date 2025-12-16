"""
FLAIR Cohort Builder Module.

Provides functionality to build the ICU cohort directly from CLIF data
using clifpy, eliminating the dependency on tokenETL.
"""

from flair.cohort.builder import FLAIRCohortBuilder, build_cohort

__all__ = ["FLAIRCohortBuilder", "build_cohort"]
