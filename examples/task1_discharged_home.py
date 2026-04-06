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
        # Task 1: Discharged Home Prediction

        **Type:** Binary Classification

        **Clinical Motivation:** Predicting whether an ICU patient will be discharged
        directly home helps with resource planning, post-discharge care coordination,
        and identifying patients who may need rehabilitation or skilled nursing facilities.

        **Details:**
        - **Input:** First 24 hours of ICU data
        - **Label:** 1 = discharged home, 0 = other disposition
        - **Prevalence:** ~61.8% positive (home) -- relatively balanced
        - **Key Metrics:** AUROC, AUPRC, F1, Specificity, NPV
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

    task = get_task("task1_discharged_home")
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
def _(config_path, mo, pl):
    mo.stop(not config_path.value, mo.md("*Enter config path above to generate dataset.*"))

    from flair_benchmark import generate_task_dataset

    df = generate_task_dataset(
        config_path=config_path.value,
        task_name="task1_discharged_home",
    )

    mo.md(f"**Dataset shape:** {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df, generate_task_dataset


@app.cell
def _(df, mo):
    mo.md("### Sample Rows")
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df, mo, pl):
    mo.md("## Label Distribution")
    return


@app.cell
def _(df, mo, pl):
    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") == "test")

    def label_stats(subset, name):
        total = subset.height
        pos = subset.filter(pl.col("label_home") == 1).height
        neg = total - pos
        return {
            "Split": name,
            "Total": f"{total:,}",
            "Home (1)": f"{pos:,} ({pos/total*100:.1f}%)",
            "Other (0)": f"{neg:,} ({neg/total*100:.1f}%)",
        }

    stats = [label_stats(train, "Train"), label_stats(test, "Test")]
    mo.ui.table(stats, label="Label Distribution by Split")
    return stats, test, train


@app.cell
def _(df, mo, pl):
    mo.md("## Train/Test Split")
    return


@app.cell
def _(df, mo, pl):
    train_n = df.filter(pl.col("split") == "train").height
    test_n = df.filter(pl.col("split") == "test").height
    total = df.height

    mo.md(
        f"""
        | Split | N | Percentage |
        |-------|---|------------|
        | Train | {train_n:,} | {train_n/total*100:.1f}% |
        | Test | {test_n:,} | {test_n/total*100:.1f}% |
        | **Total** | **{total:,}** | **100%** |
        """
    )
    return test_n, total, train_n


@app.cell
def _(mo):
    mo.md("## Baseline Model: Logistic Regression on Demographics")
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

    y_train = train["label_home"].to_numpy()
    y_test = test["label_home"].to_numpy()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mo.md("Logistic regression trained on demographic features (age, sex, race, ethnicity).")
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
    mo.md(
        """
        ## Insights

        - Task 1 is the most **balanced** binary task (~62% positive), making standard
          classification approaches work well without class rebalancing.
        - The demographics-only baseline gives a lower bound -- real models use the full
          24-hour clinical time series (vitals, labs, medications) for much better performance.
        - **NPV** (negative predictive value) is clinically important here: when the model
          predicts "not going home," how often is it correct? This drives care coordination.
        """
    )
    return


if __name__ == "__main__":
    app.run()
