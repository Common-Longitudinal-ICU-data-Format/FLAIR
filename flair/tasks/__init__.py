"""Task definitions for FLAIR benchmark.

All 7 tasks share the same cohort (ICU hospitalizations with at least 1 ICU stay).
"""

from flair.tasks.base import BaseTask, TaskConfig, TaskType
from flair.tasks.task1_discharged_home import Task1DischargedHome
from flair.tasks.task2_discharged_ltach import Task2DischargedLTACH
from flair.tasks.task3_outcome_72hr import Task3Outcome72hr
from flair.tasks.task4_hypoxic_proportion import Task4HypoxicProportion
from flair.tasks.task5_icu_los import Task5ICULoS
from flair.tasks.task6_hospital_mortality import Task6HospitalMortality
from flair.tasks.task7_icu_readmission import Task7ICUReadmission

TASK_REGISTRY = {
    "task1_discharged_home": Task1DischargedHome,
    "task2_discharged_ltach": Task2DischargedLTACH,
    "task3_outcome_72hr": Task3Outcome72hr,
    "task4_hypoxic_proportion": Task4HypoxicProportion,
    "task5_icu_los": Task5ICULoS,
    "task6_hospital_mortality": Task6HospitalMortality,
    "task7_icu_readmission": Task7ICUReadmission,
}


def get_task(task_name: str, config: dict = None) -> BaseTask:
    """Get a task instance by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    task_class = TASK_REGISTRY[task_name]
    return task_class(config)


__all__ = [
    "BaseTask",
    "TaskConfig",
    "TaskType",
    "Task1DischargedHome",
    "Task2DischargedLTACH",
    "Task3Outcome72hr",
    "Task4HypoxicProportion",
    "Task5ICULoS",
    "Task6HospitalMortality",
    "Task7ICUReadmission",
    "TASK_REGISTRY",
    "get_task",
]
