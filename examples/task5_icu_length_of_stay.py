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
        # Task 5: ICU Length of Stay Prediction

        **Type:** Regression (continuous, hours)

        **Clinical Motivation:** Predicting remaining ICU length of stay after the
        first 24 hours aids bed management, staffing, and resource allocation.
        "Remaining LOS" is more actionable than total LOS because it answers
        "how much longer?" rather than "how long in total?"

        **Details:**
        - **Input:** First 24 hours of ICU data
        - **Cohort:** Patients with >= 24 hours first ICU stay
        - **Label:** Remaining ICU LOS in hours (total - 24)
        - **Outlier bounds:** 2 < hours < 1000
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

    task = get_task("task5_icu_los")
    cfg = task.config

    mo.md(
        f"""
        | Property | Value |
        |----------|-------|
        | Name | `{cfg.name}` |
        | Type | {cfg.task_type.value} |
        | Input Window | {cfg.input_window_hours} hours |
        | Cohort Filter | {cfg.cohort_filter} |
        | Label Column | `{cfg.label_column}` |
        | Metrics | {", ".join(cfg.evaluation_metrics)} |

        **Outlier bounds:** Patients with remaining LOS <= 2 hours or >= 1000 hours are excluded.
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
        task_name="task5_icu_los",
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
        pl.col("icu_los_hours").mean().alias("Mean"),
        pl.col("icu_los_hours").median().alias("Median"),
        pl.col("icu_los_hours").std().alias("Std"),
        pl.col("icu_los_hours").min().alias("Min"),
        pl.col("icu_los_hours").max().alias("Max"),
        pl.col("icu_los_hours").quantile(0.25).alias("Q25"),
        pl.col("icu_los_hours").quantile(0.75).alias("Q75"),
    ]).row(0, named=True)

    mo.md(
        f"""
        ### Training Set Remaining LOS Statistics (hours)

        | Statistic | Value |
        |-----------|-------|
        | Mean | {stats['Mean']:.1f} |
        | Median | {stats['Median']:.1f} |
        | Std | {stats['Std']:.1f} |
        | Q25 | {stats['Q25']:.1f} |
        | Q75 | {stats['Q75']:.1f} |
        | Min | {stats['Min']:.1f} |
        | Max | {stats['Max']:.1f} |

        The distribution is **right-skewed**: most stays are short, but a long tail
        of patients with extended ICU stays pulls the mean above the median.
        """
    )
    return stats, train


@app.cell
def _(df, mo, pl):
    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") == "test")

    mo.md(
        f"""
        ### Train/Test Split

        | Split | N |
        |-------|---|
        | Train | {train.height:,} |
        | Test | {test.height:,} |
        | Total | {df.height:,} |
        """
    )
    return test, train


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

    y_train = train["icu_los_hours"].to_numpy()
    y_test = test["icu_los_hours"].to_numpy()

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mo.md("Ridge regression on demographic features (age, sex, race, ethnicity).")
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
        | MSE | {metrics.mse:.2f} |
        | RMSE | {metrics.rmse:.2f} hours |
        | MAE | {metrics.mae:.2f} hours |
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

        - The distribution is **right-skewed**, so a log-transform of the target
          (`log(LOS)`) often improves regression performance by reducing the influence
          of long-stay outliers.
        - FLAIR already excludes extreme outliers (remaining LOS <= 2hr or >= 1000hr),
          but the distribution still has substantial variance.
        - A demographics-only model provides a weak baseline. Real models leverage
          the full 24-hour clinical time series (vitals trends, lab trajectories,
          ventilator settings) which carry much more predictive information.
        - **MAE in hours** is the most interpretable metric: "on average, predictions
          are off by X hours."
        """
    )
    return


if __name__ == "__main__":
    app.run()
