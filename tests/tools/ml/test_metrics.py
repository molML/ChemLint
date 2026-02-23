"""Tests for metrics.py - Machine learning evaluation metrics."""

import pandas as pd
import numpy as np
import pytest
from chemlint.tools.ml.metrics import (
    _accuracy,
    _balanced_accuracy,
    _precision,
    _recall,
    _f1_score,
    _roc_auc,
    _matthews_corrcoef,
    _mse,
    _rmse,
    _mae,
    _r2,
    _tp,
    _fp,
    _tn,
    _fn,
    list_all_supported_metrics,
    calculate_metrics,
)
from chemlint.infrastructure.resources import create_project_manifest, _store_resource


# ============================================================================
# Tests for classification metrics
# ============================================================================

def test_accuracy():
    """Test accuracy metric."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    acc = _accuracy(y_true, y_pred)
    assert acc == 0.8  # 4 out of 5 correct
    
    # Perfect prediction
    assert _accuracy(y_true, y_true) == 1.0
    
    # All wrong
    y_wrong = np.array([1, 0, 0, 1, 0])
    assert _accuracy(y_true, y_wrong) == 0.0


def test_balanced_accuracy():
    """Test balanced accuracy metric."""
    # Balanced case
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert _balanced_accuracy(y_true, y_pred) == 1.0
    
    # Imbalanced case - gets recall for each class
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])  # Class 0: 3/3, Class 1: 2/3
    bal_acc = _balanced_accuracy(y_true, y_pred)
    expected = (3/3 + 2/3) / 2  # Average of recalls
    assert bal_acc == pytest.approx(expected)


def test_precision():
    """Test precision metric."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 1])  # 3 TP, 1 FP
    
    prec = _precision(y_true, y_pred)
    assert prec == 0.75  # 3 / (3 + 1)
    
    # Perfect precision
    y_pred_perfect = np.array([0, 1, 1, 0, 1])
    assert _precision(y_true, y_pred_perfect) == 1.0
    
    # No positives predicted (edge case)
    y_pred_none = np.array([0, 0, 0, 0, 0])
    assert _precision(y_true, y_pred_none) == 0.0


def test_recall():
    """Test recall metric."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])  # 2 TP, 1 FN
    
    rec = _recall(y_true, y_pred)
    assert rec == pytest.approx(2/3)  # 2 / (2 + 1)
    
    # Perfect recall
    y_pred_perfect = np.array([0, 1, 1, 0, 1])
    assert _recall(y_true, y_pred_perfect) == 1.0
    
    # No positives found
    y_pred_none = np.array([0, 0, 0, 0, 0])
    assert _recall(y_true, y_pred_none) == 0.0


def test_f1_score():
    """Test F1 score metric."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 1])
    
    # Precision = 0.75, Recall = 1.0
    # F1 = 2 * 0.75 * 1.0 / (0.75 + 1.0) = 0.857...
    f1 = _f1_score(y_true, y_pred)
    expected = 2 * 0.75 * 1.0 / (0.75 + 1.0)
    assert f1 == pytest.approx(expected)
    
    # Perfect F1
    assert _f1_score(y_true, y_true) == 1.0


def test_roc_auc():
    """Test ROC AUC metric."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])  # Probabilities
    
    auc = _roc_auc(y_true, y_score)
    assert 0.0 <= auc <= 1.0
    assert auc > 0.5  # Should be better than random
    
    # Perfect AUC
    y_score_perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert _roc_auc(y_true, y_score_perfect) == 1.0


def test_matthews_corrcoef():
    """Test Matthews Correlation Coefficient."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    # Perfect prediction should give MCC = 1.0
    assert _matthews_corrcoef(y_true, y_pred) == 1.0
    
    # Completely wrong should give MCC = -1.0
    y_pred_wrong = np.array([1, 1, 0, 0])
    assert _matthews_corrcoef(y_true, y_pred_wrong) == -1.0
    
    # Mixed case
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    mcc = _matthews_corrcoef(y_true, y_pred)
    assert -1.0 <= mcc <= 1.0


def test_confusion_matrix_values():
    """Test TP, FP, TN, FN calculations."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    # TP=2 (indices 2,3), FP=1 (index 1), TN=1 (index 0), FN=1 (index 4)
    
    assert _tp(y_true, y_pred) == 2
    assert _fp(y_true, y_pred) == 1
    assert _tn(y_true, y_pred) == 1
    assert _fn(y_true, y_pred) == 1
    
    # Perfect prediction
    assert _tp(y_true, y_true) == 3
    assert _fp(y_true, y_true) == 0
    assert _tn(y_true, y_true) == 2
    assert _fn(y_true, y_true) == 0


# ============================================================================
# Tests for regression metrics
# ============================================================================

def test_mse():
    """Test Mean Squared Error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])
    
    mse = _mse(y_true, y_pred)
    # Errors: 0.1, 0.1, -0.1, 0.2
    # Squared: 0.01, 0.01, 0.01, 0.04
    # Mean: 0.0175
    expected = (0.01 + 0.01 + 0.01 + 0.04) / 4
    assert mse == pytest.approx(expected)
    
    # Perfect prediction
    assert _mse(y_true, y_true) == 0.0


def test_rmse():
    """Test Root Mean Squared Error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Perfect prediction
    assert _rmse(y_true, y_pred) == 0.0
    
    # Known case
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])
    assert _rmse(y_true, y_pred) == 1.0  # sqrt(1.0)


def test_mae():
    """Test Mean Absolute Error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])
    
    mae = _mae(y_true, y_pred)
    # Absolute errors: 0.1, 0.1, 0.1, 0.2
    # Mean: 0.125
    expected = (0.1 + 0.1 + 0.1 + 0.2) / 4
    assert mae == pytest.approx(expected)
    
    # Perfect prediction
    assert _mae(y_true, y_true) == 0.0


def test_r2():
    """Test R² (coefficient of determination)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Perfect prediction
    assert _r2(y_true, y_pred) == 1.0
    
    # Mean prediction (R² = 0)
    y_pred_mean = np.array([2.5, 2.5, 2.5, 2.5])
    assert _r2(y_true, y_pred_mean) == pytest.approx(0.0)
    
    # Good prediction
    y_pred_good = np.array([1.1, 2.0, 2.9, 4.0])
    r2 = _r2(y_true, y_pred_good)
    assert 0.9 < r2 < 1.0


# ============================================================================
# Tests for utility functions
# ============================================================================

def test_list_all_supported_metrics():
    """Test listing all supported metrics."""
    metrics = list_all_supported_metrics()
    
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    
    # Check for key metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "matthews_corrcoef" in metrics
    
    # Check confusion matrix values
    assert "tp" in metrics
    assert "fp" in metrics
    assert "tn" in metrics
    assert "fn" in metrics


# ============================================================================
# Tests for dataset-level metric calculation
# ============================================================================

def test_calculate_metrics_classification(session_workdir, request):
    """Test calculating classification metrics on dataset."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create predictions dataset
    df = pd.DataFrame({
        "true_label": [0, 1, 1, 0, 1, 1, 0, 0],
        "pred_label": [0, 1, 1, 0, 1, 0, 0, 1],
        "id": range(8)
    })
    input_file = _store_resource(df, manifest_path, "predictions", "test predictions", "csv")
    
    # Calculate multiple classification metrics
    result = calculate_metrics(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        true_label_column="true_label",
        predicted_column="pred_label",
        metrics=["accuracy", "precision", "recall", "f1_score"]
    )
    
    # Check return structure
    assert "metrics" in result
    assert "n_samples" in result
    assert result["n_samples"] == 8
    
    # Check all requested metrics are present
    assert "accuracy" in result["metrics"]
    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"]
    assert "f1_score" in result["metrics"]
    
    # Verify reasonable values
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= result["metrics"]["precision"] <= 1.0
    assert 0.0 <= result["metrics"]["recall"] <= 1.0
    assert 0.0 <= result["metrics"]["f1_score"] <= 1.0


def test_calculate_metrics_regression(session_workdir, request):
    """Test calculating regression metrics on dataset."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create regression predictions dataset
    df = pd.DataFrame({
        "true_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        "pred_value": [1.1, 2.0, 2.9, 4.2, 4.8],
    })
    input_file = _store_resource(df, manifest_path, "predictions", "test predictions", "csv")
    
    # Calculate regression metrics
    result = calculate_metrics(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        true_label_column="true_value",
        predicted_column="pred_value",
        metrics=["mse", "rmse", "mae", "r2"]
    )
    
    # Check return structure
    assert "metrics" in result
    assert result["n_samples"] == 5
    
    # Check all metrics present
    assert "mse" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "mae" in result["metrics"]
    assert "r2" in result["metrics"]
    
    # Verify reasonable values
    assert result["metrics"]["mse"] >= 0
    assert result["metrics"]["rmse"] >= 0
    assert result["metrics"]["mae"] >= 0
    assert result["metrics"]["r2"] <= 1.0


def test_calculate_metrics_confusion_matrix_values(session_workdir, request):
    """Test calculating TP, FP, TN, FN from dataset."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create predictions with known confusion matrix
    df = pd.DataFrame({
        "true_label": [0, 0, 1, 1, 1],
        "pred_label": [0, 1, 1, 1, 0],
    })
    input_file = _store_resource(df, manifest_path, "predictions", "test predictions", "csv")
    
    # Calculate confusion matrix values
    result = calculate_metrics(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        true_label_column="true_label",
        predicted_column="pred_label",
        metrics=["tp", "fp", "tn", "fn"]
    )
    
    # Verify counts
    assert result["metrics"]["tp"] == 2  # Indices 2, 3
    assert result["metrics"]["fp"] == 1  # Index 1
    assert result["metrics"]["tn"] == 1  # Index 0
    assert result["metrics"]["fn"] == 1  # Index 4


def test_calculate_metrics_roc_auc(session_workdir, request):
    """Test calculating ROC AUC with probabilities."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create predictions with probabilities
    df = pd.DataFrame({
        "true_label": [0, 0, 0, 1, 1, 1],
        "pred_prob": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],  # Good separation
    })
    input_file = _store_resource(df, manifest_path, "predictions", "test predictions", "csv")
    
    # Calculate ROC AUC
    result = calculate_metrics(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        true_label_column="true_label",
        predicted_column="pred_prob",
        metrics=["roc_auc"]
    )
    
    assert "roc_auc" in result["metrics"]
    assert result["metrics"]["roc_auc"] > 0.5  # Should be better than random


def test_calculate_metrics_invalid_column(session_workdir, request):
    """Test error handling for invalid column names."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({
        "true_label": [0, 1],
        "pred_label": [0, 1]
    })
    input_file = _store_resource(df, manifest_path, "predictions", "test predictions", "csv")
    
    # Test invalid true label column
    with pytest.raises(ValueError, match="not found in dataset"):
        calculate_metrics(
            input_filename=input_file,
            project_manifest_path=manifest_path,
            true_label_column="nonexistent",
            predicted_column="pred_label",
            metrics=["accuracy"]
        )
    
    # Test invalid prediction column
    with pytest.raises(ValueError, match="not found in dataset"):
        calculate_metrics(
            input_filename=input_file,
            project_manifest_path=manifest_path,
            true_label_column="true_label",
            predicted_column="nonexistent",
            metrics=["accuracy"]
        )


def test_calculate_metrics_matthews_corrcoef(session_workdir, request):
    """Test calculating Matthews Correlation Coefficient."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create balanced predictions
    df = pd.DataFrame({
        "true_label": [0, 0, 1, 1],
        "pred_label": [0, 0, 1, 1],
    })
    input_file = _store_resource(df, manifest_path, "predictions", "test predictions", "csv")
    
    # Calculate MCC
    result = calculate_metrics(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        true_label_column="true_label",
        predicted_column="pred_label",
        metrics=["matthews_corrcoef"]
    )
    
    assert "matthews_corrcoef" in result["metrics"]
    assert result["metrics"]["matthews_corrcoef"] == 1.0  # Perfect prediction


def test_calculate_metrics_mixed(session_workdir, request):
    """Test calculating both classification and regression metrics together."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Note: This tests the API, though mixing classification/regression metrics
    # on the same data may not make practical sense
    df = pd.DataFrame({
        "true_label": [0, 1, 1, 0, 1],
        "pred_label": [0, 1, 0, 0, 1],
    })
    input_file = _store_resource(df, manifest_path, "predictions", "test predictions", "csv")
    
    # Request various metrics
    result = calculate_metrics(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        true_label_column="true_label",
        predicted_column="pred_label",
        metrics=["accuracy", "precision", "recall", "f1_score", "matthews_corrcoef", "tp", "fp", "tn", "fn"]
    )
    
    # All metrics should be computed
    assert len(result["metrics"]) == 9
    for metric in ["accuracy", "precision", "recall", "f1_score", "matthews_corrcoef", "tp", "fp", "tn", "fn"]:
        assert metric in result["metrics"]
