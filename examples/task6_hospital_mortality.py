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
        # Task 6: In-Hospital Mortality Prediction

        **Type:** Binary Classification

        **Clinical Motivation:** Predicting in-hospital mortality from the first 24 hours
        of ICU data is one of the most studied clinical prediction tasks. Accurate mortality
        prediction supports goals-of-care conversations, triage decisions, and benchmarking
        ICU quality.

        **Details:**
        - **Input:** First 24 hours of ICU data
        - **Cohort:** Patients with >= 24 hours first ICU stay
        - **Label:** 1 = expired or hospice, 0 = survived
        - **Prevalence:** ~5-15% (moderately imbalanced)
        - **Key Metrics:** AUROC, AUPRC, calibration (ECE)
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

    task = get_task("task6_hospital_mortality")
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
        task_name="task6_hospital_mortality",
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
    test = df.filter(pl.col("split") == "test")

    def label_stats(subset, name):
        total = subset.height
        pos = subset.filter(pl.col("label_mortality") == 1).height
        neg = total - pos
        return {
            "Split": name,
            "Total": f"{total:,}",
            "Expired (1)": f"{pos:,} ({pos/total*100:.1f}%)",
            "Survived (0)": f"{neg:,} ({neg/total*100:.1f}%)",
        }

    stats = [label_stats(train, "Train"), label_stats(test, "Test")]
    mo.ui.table(stats, label="Mortality Distribution by Split")
    return stats, test, train


@app.cell
def _(mo):
    mo.md("## Baseline: Logistic Regression")
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

    y_train = train["label_mortality"].to_numpy()
    y_test = test["label_mortality"].to_numpy()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mo.md("Logistic regression on demographic features.")
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
        | AUROC | {metrics.auroc:.4f} |
        | AUPRC | {metrics.auprc:.4f} |
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
    mo.md("## Calibration")
    return


@app.cell
def _(mo, y_prob, y_test):
    from flair_benchmark.helpers.metrics import compute_calibration_metrics

    cal = compute_calibration_metrics(y_test, y_prob, n_bins=10)

    # Build calibration table
    cal_rows = []
    for i in range(cal["n_bins"]):
        if cal["bin_counts"][i] > 0 and cal["bin_pred_prob"][i] is not None:
            cal_rows.append({
                "Bin": f"{i/10:.1f}-{(i+1)/10:.1f}",
                "Predicted": f"{cal['bin_pred_prob'][i]:.3f}",
                "Observed": f"{cal['bin_true_prob'][i]:.3f}",
                "Count": cal["bin_counts"][i],
            })

    mo.md(f"**Expected Calibration Error (ECE):** {cal['ece']:.4f}")
    return cal, cal_rows, compute_calibration_metrics


@app.cell
def _(cal_rows, mo):
    mo.ui.table(cal_rows, label="Calibration by Probability Bin")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Insights

        - **Calibration** is especially important for mortality prediction. A model that
          says "30% chance of death" should be correct ~30% of the time in that group.
          ECE (Expected Calibration Error) measures this; lower is better.
        - The positive class includes **hospice** discharges alongside in-hospital deaths.
          This is a clinical design choice -- hospice represents expected mortality.
        - Shares the same **>= 24hr ICU stay** cohort filter as Task 5. Patients
          discharged or who die within 24 hours are excluded.
        - Age is a strong predictor of mortality, so even the demographics-only baseline
          typically achieves a reasonable AUROC (~0.60-0.65).
        """
    )
    return


if __name__ == "__main__":
    app.run()
