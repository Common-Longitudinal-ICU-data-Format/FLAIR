"""
Metrics computation for FLAIR benchmark.

All metrics are aggregated - no individual-level predictions are stored or returned.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BinaryMetrics:
    """Metrics for binary classification tasks."""

    auroc: float
    auprc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    npv: float
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MulticlassMetrics:
    """Metrics for multi-class classification tasks."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionMetrics:
    """Metrics for regression tasks."""

    mse: float
    rmse: float
    mae: float
    r2: float
    explained_variance: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> BinaryMetrics:
    """
    Compute metrics for binary classification.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for positive class (optional)

    Returns:
        BinaryMetrics dataclass
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            average_precision_score,
            confusion_matrix,
        )

        # Convert to numpy if needed
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Compute specificity and NPV
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Compute AUROC and AUPRC if probabilities provided
        if y_prob is not None:
            y_prob = np.asarray(y_prob).ravel()
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
        else:
            auroc = 0.0
            auprc = 0.0

        return BinaryMetrics(
            auroc=float(auroc),
            auprc=float(auprc),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            specificity=float(specificity),
            npv=float(npv),
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
        )

    except ImportError:
        logger.warning("sklearn not available, using basic implementations")
        return _compute_binary_metrics_basic(y_true, y_pred, y_prob)


def _compute_binary_metrics_basic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> BinaryMetrics:
    """Basic implementation without sklearn."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return BinaryMetrics(
        auroc=0.0,  # Cannot compute without sklearn
        auprc=0.0,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        specificity=float(specificity),
        npv=float(npv),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
    )


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> MulticlassMetrics:
    """
    Compute metrics for multi-class classification.

    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        class_names: Names for each class

    Returns:
        MulticlassMetrics dataclass
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )

        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Get unique classes
        classes = sorted(set(y_true) | set(y_pred))
        if class_names is None:
            class_names = [str(c) for c in classes]

        # Compute per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        per_class_precision = {class_names[i]: float(precision[i]) for i in range(len(classes))}
        per_class_recall = {class_names[i]: float(recall[i]) for i in range(len(classes))}
        per_class_f1 = {class_names[i]: float(f1_per_class[i]) for i in range(len(classes))}

        return MulticlassMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        )

    except ImportError:
        logger.warning("sklearn not available for multiclass metrics")
        raise ImportError("sklearn required for multiclass metrics")


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """
    Compute metrics for regression tasks.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RegressionMetrics dataclass
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Mean Squared Error
    mse = float(np.mean((y_true - y_pred) ** 2))

    # Root Mean Squared Error
    rmse = float(np.sqrt(mse))

    # Mean Absolute Error
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Explained Variance
    var_res = np.var(y_true - y_pred)
    var_true = np.var(y_true)
    explained_variance = float(1 - var_res / var_true) if var_true > 0 else 0.0

    return RegressionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        explained_variance=explained_variance,
    )


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute calibration metrics.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with calibration metrics
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute calibration curve
    bin_true_prob = []
    bin_pred_prob = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_true_prob.append(float(np.mean(y_true[mask])))
            bin_pred_prob.append(float(np.mean(y_prob[mask])))
            bin_counts.append(int(np.sum(mask)))
        else:
            bin_true_prob.append(None)
            bin_pred_prob.append(None)
            bin_counts.append(0)

    # Expected Calibration Error (ECE)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        if bin_counts[i] > 0 and bin_true_prob[i] is not None:
            ece += bin_counts[i] / total * abs(bin_true_prob[i] - bin_pred_prob[i])

    return {
        "ece": float(ece),
        "bin_true_prob": bin_true_prob,
        "bin_pred_prob": bin_pred_prob,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }
