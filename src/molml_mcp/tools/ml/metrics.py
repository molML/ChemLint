"""
Machine learning evaluation metrics for model performance assessment.
"""

import numpy as np
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix
)
from molml_mcp.infrastructure.resources import _load_resource


# ============================================================================
# Internal metric computation functions
# ============================================================================

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Accuracy is the fraction of predictions that are correct.
    Best for balanced datasets. Range: [0, 1], higher is better.
    """
    return float(accuracy_score(y_true, y_pred))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate balanced accuracy for binary classification.
    
    Balanced accuracy is the average of recall for each class.
    Better than accuracy for imbalanced datasets. Range: [0, 1], higher is better.
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def _precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Calculate precision for binary classification.
    
    Precision is the fraction of positive predictions that are correct (TP / (TP + FP)).
    Answers: "Of all predicted positives, how many were actually positive?"
    Range: [0, 1], higher is better.
    """
    return float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))


def _recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Calculate recall (sensitivity) for binary classification.
    
    Recall is the fraction of actual positives that were correctly predicted (TP / (TP + FN)).
    Answers: "Of all actual positives, how many did we find?"
    Range: [0, 1], higher is better.
    """
    return float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Calculate F1 score for binary classification.
    
    F1 score is the harmonic mean of precision and recall (2 * P * R / (P + R)).
    Balances precision and recall. Range: [0, 1], higher is better.
    """
    return float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculate ROC AUC for binary classification.
    
    ROC AUC measures the model's ability to distinguish between classes.
    Requires predicted probabilities (not class labels).
    Range: [0, 1], 0.5 = random, 1.0 = perfect, higher is better.
    """
    return float(roc_auc_score(y_true, y_score))


def _matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC) for binary classification.
    
    MCC is a balanced measure for binary classification, even with imbalanced classes.
    Takes into account TP, TN, FP, FN.
    Range: [-1, 1], -1 = total disagreement, 0 = random, 1 = perfect, higher is better.
    """
    return float(matthews_corrcoef(y_true, y_pred))


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE) for regression.
    
    MSE is the average squared difference between predictions and true values.
    Penalizes large errors heavily. Range: [0, ∞), lower is better.
    """
    return float(mean_squared_error(y_true, y_pred))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) for regression.
    
    RMSE is the square root of MSE, in the same units as the target variable.
    More interpretable than MSE. Range: [0, ∞), lower is better.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE) for regression.
    
    MAE is the average absolute difference between predictions and true values.
    Less sensitive to outliers than MSE/RMSE. Range: [0, ∞), lower is better.
    """
    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination) for regression.
    
    R² represents the proportion of variance in the target explained by the model.
    Range: (-∞, 1], 1 = perfect, 0 = baseline (mean), negative = worse than baseline.
    Higher is better.
    """
    return float(r2_score(y_true, y_pred))


def _confusion_matrix_values(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> tuple[int, int, int, int]:
    """
    Calculate True Positives, False Positives, True Negatives, False Negatives.
    
    Returns confusion matrix values for binary classification.
    
    Returns:
        Tuple of (TP, FP, TN, FN)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, 1 - pos_label if pos_label == 1 else 0])
    TP = int(cm[0, 0])
    FP = int(cm[1, 0])
    TN = int(cm[1, 1])
    FN = int(cm[0, 1])
    return TP, FP, TN, FN


def _tp(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> int:
    """
    Calculate True Positives (TP).
    
    Count of samples correctly predicted as positive.
    """
    TP, _, _, _ = _confusion_matrix_values(y_true, y_pred, pos_label)
    return TP


def _fp(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> int:
    """
    Calculate False Positives (FP).
    
    Count of samples incorrectly predicted as positive (Type I error).
    """
    _, FP, _, _ = _confusion_matrix_values(y_true, y_pred, pos_label)
    return FP


def _tn(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> int:
    """
    Calculate True Negatives (TN).
    
    Count of samples correctly predicted as negative.
    """
    _, _, TN, _ = _confusion_matrix_values(y_true, y_pred, pos_label)
    return TN


def _fn(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> int:
    """
    Calculate False Negatives (FN).
    
    Count of samples incorrectly predicted as negative (Type II error).
    """
    _, _, _, FN = _confusion_matrix_values(y_true, y_pred, pos_label)
    return FN


# ============================================================================
# Metric registry
# ============================================================================

METRIC_REGISTRY = {
    # Classification metrics
    "accuracy": _accuracy,
    "balanced_accuracy": _balanced_accuracy,
    "precision": _precision,
    "recall": _recall,
    "f1_score": _f1_score,
    "roc_auc": _roc_auc,
    "matthews_corrcoef": _matthews_corrcoef,
    "tp": _tp,
    "fp": _fp,
    "tn": _tn,
    "fn": _fn,
    # Regression metrics
    "mse": _mse,
    "rmse": _rmse,
    "mae": _mae,
    "r2": _r2,
}


def get_metric_function(metric_name: str):
    """
    Get a metric function by name.
    
    Args:
        metric_name: Name of the metric (e.g., "accuracy", "mse")
    
    Returns:
        Metric function that takes (y_true, y_pred) and returns a number
    
    Raises:
        ValueError: If metric name is not recognized
    """
    if metric_name not in METRIC_REGISTRY:
        available = list(METRIC_REGISTRY.keys())
        raise ValueError(f"Unknown metric '{metric_name}'. Available: {available}")
    return METRIC_REGISTRY[metric_name]


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[str],
    pos_label: int = 1
) -> dict:
    """
    Compute multiple metrics on true and predicted values.
    
    This is a helper function that computes metrics without any dataset loading or
    column validation. It takes raw numpy arrays and returns computed metric values.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values or probabilities
        metrics: List of metric names to calculate
        pos_label: Positive class label for binary classification (default: 1)
    
    Returns:
        Dictionary mapping metric names to computed values
        
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> compute_metrics(y_true, y_pred, ['accuracy', 'precision', 'recall'])
        {'accuracy': 0.8, 'precision': 1.0, 'recall': 0.6667}
    """
    # Metrics that require pos_label parameter
    METRICS_WITH_POS_LABEL = {'precision', 'recall', 'f1_score', 'tp', 'fp', 'tn', 'fn'}
    
    results = {}
    for metric_name in metrics:
        metric_name_lower = metric_name.lower()
        
        # Handle "mcc" alias for matthews_corrcoef
        if metric_name_lower == "mcc":
            metric_name_lower = "matthews_corrcoef"
        
        # Get metric function from registry
        metric_func = get_metric_function(metric_name_lower)
        
        # Call with pos_label if required
        if metric_name_lower in METRICS_WITH_POS_LABEL:
            results[metric_name] = metric_func(y_true, y_pred, pos_label=pos_label)
        else:
            results[metric_name] = metric_func(y_true, y_pred)
    
    return results


# ============================================================================
# Dataset-level metric calculation function
# ============================================================================

def calculate_metrics(
    input_filename: str,
    project_manifest_path: str,
    true_label_column: str,
    predicted_column: str,
    metrics: list[str],
    pos_label: int = 1
) -> dict:
    """
    Calculate multiple metrics on a dataset with predictions.
    
    This function computes the requested metrics by comparing true labels/values
    with predicted labels/values or probabilities in a dataset.
    
    Available Binary Classification Metrics (require predicted labels):
    - accuracy: Overall correctness
    - balanced_accuracy: Better for imbalanced datasets
    - precision: Of predicted positives, how many were correct?
    - recall: Of actual positives, how many did we find?
    - f1_score: Harmonic mean of precision and recall
    - matthews_corrcoef: Balanced measure using TP/TN/FP/FN
    - TP: True Positives count
    - FP: False Positives count
    - TN: True Negatives count
    - FN: False Negatives count
    
    Available Binary Classification Metrics (require predicted probabilities):
    - roc_auc: Area under ROC curve
    
    Available Regression Metrics:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - r2: Coefficient of determination
    
    Args:
        input_filename: Input dataset filename
        project_manifest_path: Path to manifest.json
        true_label_column: Column containing true labels/values
        predicted_column: Column containing predicted labels/values or probabilities
        metrics: List of metric names to calculate (e.g., ["accuracy", "f1_score", "roc_auc"])
        pos_label: Positive class label for binary classification (default: 1)
    
    Returns:
        dict with computed metrics and metadata
        
    Example:
        >>> calculate_metrics(
        ...     "predictions_A1B2C3D4.csv",
        ...     "/path/to/manifest.json",
        ...     "true_label",
        ...     "pred_label",
        ...     ["accuracy", "precision", "recall", "f1_score"]
        ... )
        {
            "metrics": {
                "accuracy": 0.8542,
                "precision": 0.8234,
                "recall": 0.8891,
                "f1_score": 0.8549
            },
            "n_samples": 1000,
            "note": "Computed 4 metrics on 1000 samples"
        }
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    if true_label_column not in df.columns:
        raise ValueError(f"Column '{true_label_column}' not found in dataset")
    if predicted_column not in df.columns:
        raise ValueError(f"Column '{predicted_column}' not found in dataset")
    
    # Extract arrays
    y_true = df[true_label_column].values
    y_pred = df[predicted_column].values
    n_samples = len(y_true)
    
    # Compute metrics using helper function
    results = _compute_metrics(y_true, y_pred, metrics, pos_label=pos_label)
    
    # Build summary
    metric_summary = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in results.items()])
    
    return {
        "metrics": results,
        "n_samples": n_samples,
        "parameters": {
            "pos_label": pos_label
        },
        "note": f"Computed {len(results)} metric(s) on {n_samples} samples. {metric_summary}"
    }



