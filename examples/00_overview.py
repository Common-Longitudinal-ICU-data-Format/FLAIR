import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # FLAIR: Federated Learning Assessment for ICU Research

        Interactive overview of all 7 FLAIR benchmark tasks.

        FLAIR is a privacy-first benchmark framework for evaluating ML/AI methods
        on ICU prediction tasks across 17+ hospital sites using the
        **CLIF (Common Longitudinal ICU Format)** data standard.

        Each task takes a standardized cohort of ICU hospitalizations and produces
        a 13-column dataset with task-specific labels, time windows, and a train/test split.
        """
    )
    return


@app.cell
def _():
    from flair_benchmark.tasks import TASK_REGISTRY, get_task
    from flair_benchmark.tasks.base import TaskType
    return TASK_REGISTRY, TaskType, get_task


@app.cell
def _(TASK_REGISTRY, get_task, mo):
    # Build summary table of all tasks
    rows = []
    for task_name, task_cls in TASK_REGISTRY.items():
        task = task_cls()
        cfg = task.config
        rows.append({
            "Task": cfg.display_name,
            "Key": task_name,
            "Type": cfg.task_type.value,
            "Label Column": cfg.label_column,
            "Input Window (hr)": str(cfg.input_window_hours) if cfg.input_window_hours else "Variable",
            "Prediction Window": f"{cfg.prediction_window[0]}-{cfg.prediction_window[1]}hr" if cfg.prediction_window else "At discharge",
            "Metrics": ", ".join(cfg.evaluation_metrics[:3]) + "...",
        })

    mo.md("## Task Registry")
    return (rows,)


@app.cell
def _(mo, rows):
    mo.ui.table(rows, label="All 7 FLAIR Benchmark Tasks")
    return


@app.cell
def _(TASK_REGISTRY, mo):
    task_selector = mo.ui.dropdown(
        options=list(TASK_REGISTRY.keys()),
        value="task1_discharged_home",
        label="Select a task to inspect:",
    )
    task_selector
    return (task_selector,)


@app.cell
def _(get_task, mo, task_selector):
    selected_task = get_task(task_selector.value)
    cfg = selected_task.config

    mo.md(
        f"""
        ## {cfg.display_name}

        | Property | Value |
        |----------|-------|
        | **Name** | `{cfg.name}` |
        | **Type** | {cfg.task_type.value} |
        | **Description** | {cfg.description} |
        | **Input Window** | {cfg.input_window_hours} hours |
        | **Prediction Window** | {f"{cfg.prediction_window[0]}-{cfg.prediction_window[1]} hours" if cfg.prediction_window else "At discharge"} |
        | **Cohort Filter** | {cfg.cohort_filter or "None (all ICU patients)"} |
        | **Label Column** | `{cfg.label_column}` |
        | **Positive Class** | {cfg.positive_class or "N/A"} |
        | **Class Names** | {", ".join(cfg.class_names) if cfg.class_names else "N/A"} |
        | **Evaluation Metrics** | {", ".join(cfg.evaluation_metrics)} |
        """
    )
    return cfg, selected_task


@app.cell
def _(mo):
    mo.md(
        """
        ## Output Schema

        All tasks produce a standardized 13-column parquet:

        | Column | Type | Description |
        |--------|------|-------------|
        | `hospitalization_id` | str | Unique patient encounter ID |
        | `admission_dttm` | datetime | Hospital admission time |
        | `discharge_dttm` | datetime | Hospital discharge time |
        | `window_start` | datetime | Data collection start (ICU admit) |
        | `window_end` | datetime | Prediction time (data cutoff) |
        | `{task_label}` | int/float | Task-specific label |
        | `split` | str | "train" or "test" |
        | `age_at_admission` | int | Patient age |
        | `sex_category` | str | Patient sex |
        | `race_category` | str | Patient race |
        | `ethnicity_category` | str | Patient ethnicity |
        | `hospital_id` | str | Hospital identifier |
        | `hospital_type` | str | Hospital type |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Getting Started

        ```python
        from flair_benchmark import generate_task_dataset

        # For MIMIC (de-identified dates, 70/30 temporal split):
        df = generate_task_dataset(
            config_path="clif_config.json",
            task_name="task5_icu_los",
        )

        # For other sites (date-based split):
        df = generate_task_dataset(
            config_path="clif_config.json",
            task_name="task5_icu_los",
            train_start="2020-01-01",
            train_end="2022-12-31",
            test_start="2023-01-01",
            test_end="2023-12-31",
        )
        ```

        Open individual task notebooks to see full walkthroughs with
        data exploration, baseline models, and evaluation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
