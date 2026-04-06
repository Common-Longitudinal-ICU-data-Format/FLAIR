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
        # Task 3: 72-Hour Respiratory Outcome Prediction

        **Type:** Multiclass Classification (3 classes)

        **Clinical Motivation:** For patients on invasive mechanical ventilation (IMV),
        predicting their respiratory status at 72 hours informs ventilator management,
        weaning protocols, and family communication.

        **Details:**
        - **Input:** First 24 hours of ICU data
        - **Prediction Window:** 24-72 hours post-admission
        - **Cohort:** Only patients on IMV at the 24-hour mark
        - **Classes:**
          - `imv_on` (0): Still on ventilator at 72 hours
          - `imv_off` (1): Successfully weaned from ventilation
          - `expired` (2): Deceased by 72 hours
        - **Key Metrics:** Macro F1, Weighted F1, per-class precision/recall
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

    task = get_task("task3_outcome_72hr")
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
        | Classes | {", ".join(cfg.class_names)} |
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
        task_name="task3_outcome_72hr",
    )

    mo.md(f"**Dataset shape:** {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df, generate_task_dataset


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(mo):
    mo.md("## Class Distribution")
    return


@app.cell
def _(cfg, df, mo, pl):
    class_names = cfg.class_names  # ["imv_on", "imv_off", "expired"]

    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") == "test")

    def class_stats(subset, name):
        total = subset.height
        row = {"Split": name, "Total": f"{total:,}"}
        for i, cname in enumerate(class_names):
            count = subset.filter(pl.col("task3_label") == i).height
            row[f"{cname} ({i})"] = f"{count:,} ({count/total*100:.1f}%)"
        return row

    stats = [class_stats(train, "Train"), class_stats(test, "Test")]
    mo.ui.table(stats, label="Class Distribution by Split")
    return class_names, class_stats, stats, test, train


@app.cell
def _(mo):
    mo.md("## Baseline: Multinomial Logistic Regression")
    return


@app.cell
def _(df, mo, np, pl):
    from sklearn.linear_model import LogisticRegression
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

    y_train = train["task3_label"].to_numpy()
    y_test = test["task3_label"].to_numpy()

    model = LogisticRegression(
        multi_class="multinomial", max_iter=1000, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mo.md("Multinomial logistic regression on demographic features.")
    return (
        LogisticRegression, OneHotEncoder, X_test, X_train, cat_cols,
        enc, model, y_pred, y_test, y_train,
    )


@app.cell
def _(mo):
    mo.md("## Evaluation")
    return


@app.cell
def _(cfg, mo, y_pred, y_test):
    from flair_benchmark.helpers.metrics import compute_multiclass_metrics

    metrics = compute_multiclass_metrics(y_test, y_pred, cfg.class_names)

    mo.md(
        f"""
        ### Overall Metrics

        | Metric | Value |
        |--------|-------|
        | Accuracy | {metrics.accuracy:.4f} |
        | Macro F1 | {metrics.macro_f1:.4f} |
        | Weighted F1 | {metrics.weighted_f1:.4f} |

        ### Per-Class Metrics

        | Class | Precision | Recall | F1 |
        |-------|-----------|--------|-----|
        """
        + "\n".join(
            f"| {cname} | {metrics.per_class_precision[cname]:.4f} | {metrics.per_class_recall[cname]:.4f} | {metrics.per_class_f1[cname]:.4f} |"
            for cname in cfg.class_names
        )
        + f"""

        ### Confusion Matrix
        ```
        {metrics.confusion_matrix}
        ```
        """
    )
    return compute_multiclass_metrics, metrics


@app.cell
def _(mo):
    mo.md(
        """
        ## Insights

        - This is the only **multiclass** task in FLAIR. The three outcomes
          (still ventilated, weaned, expired) have very different clinical implications.
        - **Macro F1** weights all classes equally, which matters because the
          "expired" class is typically the smallest but clinically most critical.
        - Misclassifying "expired" as "imv_off" (weaned) is more harmful than the
          reverse -- it creates false optimism about patient trajectory.
        - The cohort is restricted to **IMV patients at 24 hours**, which is a
          much smaller, sicker subset of all ICU patients.
        """
    )
    return


if __name__ == "__main__":
    app.run()
