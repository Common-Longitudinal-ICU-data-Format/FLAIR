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
        # Task 2: Discharged to LTACH Prediction

        **Type:** Binary Classification (Highly Imbalanced)

        **Clinical Motivation:** Early identification of patients likely needing
        Long-Term Acute Care Hospital (LTACH) discharge enables proactive
        care coordination, insurance authorization, and bed management.

        **Details:**
        - **Input:** First 24 hours of ICU data
        - **Label:** 1 = discharged to LTACH, 0 = other disposition
        - **Prevalence:** ~3.6% positive -- **highly imbalanced**
        - **Primary Metric:** AUPRC (more informative than AUROC for rare events)
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

    task = get_task("task2_discharged_ltach")
    cfg = task.config

    mo.md(
        f"""
        | Property | Value |
        |----------|-------|
        | Name | `{cfg.name}` |
        | Type | {cfg.task_type.value} |
        | Input Window | {cfg.input_window_hours} hours |
        | Label Column | `{cfg.label_column}` |
        | Positive Class | {cfg.positive_class} |
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
        task_name="task2_discharged_ltach",
    )

    mo.md(f"**Dataset shape:** {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df, generate_task_dataset


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(mo):
    mo.md("## Label Distribution (Imbalanced)")
    return


@app.cell
def _(df, mo, pl):
    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") == "test")

    def label_stats(subset, name):
        total = subset.height
        pos = subset.filter(pl.col("label_ltach") == 1).height
        neg = total - pos
        return {
            "Split": name,
            "Total": f"{total:,}",
            "LTACH (1)": f"{pos:,} ({pos/total*100:.1f}%)",
            "Other (0)": f"{neg:,} ({neg/total*100:.1f}%)",
        }

    stats = [label_stats(train, "Train"), label_stats(test, "Test")]
    mo.ui.table(stats, label="Label Distribution by Split")
    return stats, test, train


@app.cell
def _(df, mo, pl):
    total = df.height
    pos = df.filter(pl.col("label_ltach") == 1).height

    mo.md(
        f"""
        **Imbalance ratio:** 1:{(total - pos) // max(pos, 1)} (positive:negative)

        A naive classifier that always predicts "not LTACH" achieves
        **{(total - pos)/total*100:.1f}% accuracy** but **0% recall**.
        This is why AUPRC is the primary metric for this task.
        """
    )
    return pos, total


@app.cell
def _(mo):
    mo.md("## Baseline: Weighted Logistic Regression")
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

    y_train = train["label_ltach"].to_numpy()
    y_test = test["label_ltach"].to_numpy()

    # class_weight='balanced' upweights the rare LTACH class
    model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mo.md(
        """
        Using `class_weight='balanced'` to handle the ~3.6% prevalence.
        This inversely weights classes by frequency, preventing the model
        from simply predicting the majority class.
        """
    )
    return (
        LogisticRegression, OneHotEncoder, X_test, X_train, cat_cols,
        enc, model, y_pred, y_prob, y_test, y_train,
    )


@app.cell
def _(mo):
    mo.md("## Evaluation")
    return


@app.cell
def _(mo, y_pred, y_prob, y_test):
    from flair_benchmark.helpers.metrics import compute_binary_metrics

    metrics = compute_binary_metrics(y_test, y_pred, y_prob)

    mo.md(
        f"""
        | Metric | Value |
        |--------|-------|
        | **AUPRC** | **{metrics.auprc:.4f}** (primary metric) |
        | AUROC | {metrics.auroc:.4f} |
        | Accuracy | {metrics.accuracy:.4f} |
        | Precision | {metrics.precision:.4f} |
        | Recall | {metrics.recall:.4f} |
        | F1 | {metrics.f1:.4f} |
        | Specificity | {metrics.specificity:.4f} |
        | NPV | {metrics.npv:.4f} |

        **Confusion Matrix:** TP={metrics.tp}, TN={metrics.tn}, FP={metrics.fp}, FN={metrics.fn}
        """
    )
    return compute_binary_metrics, metrics


@app.cell
def _(mo):
    mo.md(
        """
        ## Insights

        - With ~3.6% prevalence, **AUPRC** is the primary evaluation metric.
          AUROC can appear misleadingly high even for poor models on imbalanced data.
        - `class_weight='balanced'` trades off specificity for recall -- the model
          generates more false positives but catches more true LTACH discharges.
        - Clinical impact: early LTACH identification triggers insurance pre-authorization
          and facility placement, which can take days.
        """
    )
    return


if __name__ == "__main__":
    app.run()
