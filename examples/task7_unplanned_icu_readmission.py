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
        # Task 7: Unplanned ICU Readmission Prediction

        **Type:** Binary Classification (Complex Logic)

        **Clinical Motivation:** Unplanned ICU readmission is a quality indicator.
        Predicting which patients will return to the ICU after their first stay
        can guide discharge decisions, step-down monitoring, and follow-up intensity.

        **Details:**
        - **Input:** Entire first ICU stay (variable length, not fixed 24 hours)
        - **Label:** 1 = unplanned ICU readmission during same hospitalization, 0 = no readmission
        - **Cohort:** Complex exclusion logic (see below)
        - **Key Metrics:** AUROC, AUPRC, F1

        **Exclusion Criteria:**
        1. Deaths during first ICU stay
        2. Direct ICU discharges (last location is ICU)
        3. Short ICU stays (< 1 hour)
        4. Intermediate ER visits between ICU stays

        **Readmission Logic:**
        - ICU -> Procedural -> ICU = **PLANNED** (merged into single stay)
        - ICU -> Ward/Other -> ICU = **UNPLANNED** (label = 1)
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

    task = get_task("task7_icu_readmission")
    cfg = task.config

    mo.md(
        f"""
        | Property | Value |
        |----------|-------|
        | Name | `{cfg.name}` |
        | Type | {cfg.task_type.value} |
        | Input Window | {"Variable (entire first ICU stay)" if cfg.input_window_hours is None else f"{cfg.input_window_hours} hours"} |
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
        task_name="task7_icu_readmission",
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
        pos = subset.filter(pl.col("label_icu_readmission") == 1).height
        neg = total - pos
        return {
            "Split": name,
            "Total": f"{total:,}",
            "Readmitted (1)": f"{pos:,} ({pos/total*100:.1f}%)",
            "Not readmitted (0)": f"{neg:,} ({neg/total*100:.1f}%)",
        }

    stats = [label_stats(train, "Train"), label_stats(test, "Test")]
    mo.ui.table(stats, label="Readmission Distribution by Split")
    return stats, test, train


@app.cell
def _(mo):
    mo.md("## Variable Input Window")
    return


@app.cell
def _(df, mo, pl):
    # Unlike other tasks with fixed 24hr windows, Task 7 uses the entire first ICU stay
    window_hours = df.with_columns(
        ((pl.col("window_end") - pl.col("window_start")).dt.total_seconds() / 3600)
        .alias("window_hours")
    )

    stats = window_hours.select([
        pl.col("window_hours").mean().alias("Mean"),
        pl.col("window_hours").median().alias("Median"),
        pl.col("window_hours").std().alias("Std"),
        pl.col("window_hours").min().alias("Min"),
        pl.col("window_hours").max().alias("Max"),
    ]).row(0, named=True)

    mo.md(
        f"""
        Task 7 is unique: the input window covers the **entire first ICU stay**,
        not a fixed 24-hour window.

        ### Input Window Duration (hours)

        | Statistic | Value |
        |-----------|-------|
        | Mean | {stats['Mean']:.1f} |
        | Median | {stats['Median']:.1f} |
        | Std | {stats['Std']:.1f} |
        | Min | {stats['Min']:.1f} |
        | Max | {stats['Max']:.1f} |

        Models for this task must handle **variable-length** input sequences.
        """
    )
    return stats, window_hours


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

    # Add ICU stay duration as a feature (unique to Task 7)
    train_with_dur = train.with_columns(
        ((pl.col("window_end") - pl.col("window_start")).dt.total_seconds() / 3600)
        .alias("icu_stay_hours")
    )
    test_with_dur = test.with_columns(
        ((pl.col("window_end") - pl.col("window_start")).dt.total_seconds() / 3600)
        .alias("icu_stay_hours")
    )

    num_cols = ["age_at_admission", "icu_stay_hours"]

    X_train_raw = train_with_dur.select(num_cols + cat_cols).to_pandas()
    X_test_raw = test_with_dur.select(num_cols + cat_cols).to_pandas()

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_cat = enc.fit_transform(X_train_raw[cat_cols])
    X_test_cat = enc.transform(X_test_raw[cat_cols])

    X_train = np.hstack([X_train_raw[num_cols].values, X_train_cat])
    X_test = np.hstack([X_test_raw[num_cols].values, X_test_cat])

    y_train = train["label_icu_readmission"].to_numpy()
    y_test = test["label_icu_readmission"].to_numpy()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mo.md(
        """
        Logistic regression on demographics **plus ICU stay duration**
        (since Task 7 has variable-length input, the duration itself is a feature).
        """
    )
    return (
        LogisticRegression, OneHotEncoder, X_test, X_train,
        cat_cols, enc, model, num_cols,
        test_with_dur, train_with_dur,
        y_pred, y_prob, y_test, y_train,
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

        - Task 7 is the **most complex** FLAIR task. It overrides `build_task_dataset()`
          entirely with custom ICU timing logic that handles location merging and
          procedural sandwich removal.
        - The **variable input window** (entire first ICU stay) distinguishes this from
          all other tasks, which use a fixed 24-hour window. Models need to handle
          sequences of different lengths.
        - The **exclusion cascade** (deaths, direct discharges, short stays, ER visits)
          substantially reduces the cohort -- only patients who leave the ICU alive
          and go to a non-ICU location are eligible.
        - **Planned vs. unplanned readmissions**: ICU -> Procedural -> ICU patterns are
          considered planned (e.g., patient went for surgery) and are merged into a single
          ICU stay. Only ICU -> Ward/Other -> ICU patterns count as unplanned readmissions.
        """
    )
    return


if __name__ == "__main__":
    app.run()
