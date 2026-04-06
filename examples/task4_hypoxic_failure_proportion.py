import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    return mo, np, pl


@app.cell
def _(mo):
    mo.md(
        """
        # Task 4: Hypoxic Failure Proportion Prediction

        **Type:** Regression (bounded 0-1)

        **Clinical Motivation:** Predicting the fraction of time a ventilated patient
        will experience hypoxic respiratory failure helps guide ventilator settings,
        oxygen management, and escalation decisions.

        **Details:**
        - **Input:** First 24 hours of ICU data
        - **Prediction Window:** 24-72 hours post-admission
        - **Cohort:** Only patients on IMV at the 24-hour mark (same as Task 3)
        - **Label:** Proportion of hours with SpO2 < 90% or PaO2/FiO2 < 200
        - **Range:** 0.0 (no hypoxic hours) to 1.0 (all hours hypoxic)
        - **Key Metrics:** MSE, RMSE, MAE, R2
        """
    )
    return


@app.cell
def _(mo):
    config_path = mo.ui.text(
        value="clif_config.json",
        label="Path to clif_config.json",
        full_width=True,
    )
    config_path
    return (config_path,)


@app.cell
def _(mo):
    mo.md("## Task Configuration")
    return


@app.cell
def _(mo):
    from flair_benchmark.tasks import get_task

    task = get_task("task4_hypoxic_proportion")
    cfg = task.config

    mo.md(
        f"""
        | Property | Value |
        |----------|-------|
        | Name | `{cfg.name}` |
        | Type | {cfg.task_type.value} |
        | Input Window | {cfg.input_window_hours} hours |
        | Prediction Window | {cfg.prediction_window[0]}-{cfg.prediction_window[1]} hours |
        | Cohort Filter | {cfg.cohort_filter} |
        | Label Column | `{cfg.label_column}` |
        | Metrics | {", ".join(cfg.evaluation_metrics)} |
        """
    )
    return cfg, get_task, task


@app.cell
def _(mo):
    mo.md("## Generate Dataset")
    return


@app.cell
def _(config_path, mo):
    mo.stop(not config_path.value, mo.md("*Enter config path above to generate dataset.*"))

    from flair_benchmark import generate_task_dataset

    df = generate_task_dataset(
        config_path=config_path.value,
        task_name="task4_hypoxic_proportion",
    )

    mo.md(f"**Dataset shape:** {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df, generate_task_dataset


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(mo):
    mo.md("## Label Distribution")
    return


@app.cell
def _(df, mo, pl):
    train = df.filter(pl.col("split") == "train")

    stats = train.select([
        pl.col("task4_proportion").mean().alias("Mean"),
        pl.col("task4_proportion").median().alias("Median"),
        pl.col("task4_proportion").std().alias("Std"),
        pl.col("task4_proportion").min().alias("Min"),
        pl.col("task4_proportion").max().alias("Max"),
    ]).row(0, named=True)

    zero_pct = train.filter(pl.col("task4_proportion") == 0.0).height / train.height * 100

    mo.md(
        f"""
        ### Training Set Proportion Statistics

        | Statistic | Value |
        |-----------|-------|
        | Mean | {stats['Mean']:.4f} |
        | Median | {stats['Median']:.4f} |
        | Std | {stats['Std']:.4f} |
        | Min | {stats['Min']:.4f} |
        | Max | {stats['Max']:.4f} |
        | Zero-proportion | {zero_pct:.1f}% |

        The distribution is often **zero-inflated** -- many ventilated patients
        have no hypoxic hours in the 24-72hr window.
        """
    )
    return stats, train, zero_pct


@app.cell
def _(mo):
    mo.md("## Baseline: Ridge Regression")
    return


@app.cell
def _(df, mo, np, pl):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder

    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") == "test")

    cat_cols = ["sex_category", "race_category", "ethnicity_category"]

    X_train_raw = train.select(["age_at_admission"] + cat_cols).to_pandas()
    X_test_raw = test.select(["age_at_admission"] + cat_cols).to_pandas()

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_cat = enc.fit_transform(X_train_raw[cat_cols])
    X_test_cat = enc.transform(X_test_raw[cat_cols])

    X_train = np.hstack([X_train_raw[["age_at_admission"]].values, X_train_cat])
    X_test = np.hstack([X_test_raw[["age_at_admission"]].values, X_test_cat])

    y_train = train["task4_proportion"].to_numpy()
    y_test = test["task4_proportion"].to_numpy()

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    # Clip predictions to valid [0, 1] range
    y_pred = np.clip(model.predict(X_test), 0.0, 1.0)

    mo.md("Ridge regression on demographics, predictions clipped to [0, 1].")
    return (
        OneHotEncoder, Ridge, X_test, X_train, cat_cols,
        enc, model, y_pred, y_test, y_train,
    )


@app.cell
def _(mo):
    mo.md("## Evaluation")
    return


@app.cell
def _(mo, y_pred, y_test):
    from flair_benchmark.helpers.metrics import compute_regression_metrics

    metrics = compute_regression_metrics(y_test, y_pred)

    mo.md(
        f"""
        | Metric | Value |
        |--------|-------|
        | MSE | {metrics.mse:.6f} |
        | RMSE | {metrics.rmse:.4f} |
        | MAE | {metrics.mae:.4f} |
        | R2 | {metrics.r2:.4f} |
        | Explained Variance | {metrics.explained_variance:.4f} |
        """
    )
    return compute_regression_metrics, metrics


@app.cell
def _(mo):
    mo.md(
        """
        ## Insights

        - The target is a **bounded proportion** (0 to 1). Standard linear regression
          can predict outside this range, so predictions must be clipped.
        - The distribution is often **zero-inflated**: many patients have 0% hypoxic
          hours. A two-stage model (predict zero vs. non-zero, then predict proportion)
          may outperform a single regression.
        - This task shares the same **IMV-at-24hr cohort** as Task 3, so the two tasks
          can be jointly evaluated on the same patient population.
        - Hypoxic failure is defined as **SpO2 < 90%** or **PaO2/FiO2 < 200** --
          two complementary measures of oxygenation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
