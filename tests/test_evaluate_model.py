"""
Test evaluate_model function for single model evaluation.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molml_mcp.tools.ml.training import train_single_ml_model
from molml_mcp.tools.ml.evaluation import evaluate_models
from molml_mcp.infrastructure.resources import _store_resource, _load_resource

TEST_MANIFEST = Path(__file__).parent / "data" / "test_manifest.json"


def test_evaluate_single_model_classification():
    """Test evaluation of a single classification model."""
    np.random.seed(42)
    
    # Create training data
    train_df = pd.DataFrame({
        'smiles': [f'C{i}' for i in range(50)],
        'label': np.random.randint(0, 2, 50)
    })
    train_features = {smi: np.random.randn(5).tolist() for smi in train_df['smiles']}
    
    train_file = _store_resource(train_df, str(TEST_MANIFEST), 'eval_single_train', 'Train', 'csv')
    train_feat = _store_resource(train_features, str(TEST_MANIFEST), 'eval_single_train_feat', 'Features', 'json')
    
    # Create test data
    test_df = pd.DataFrame({
        'smiles': [f'CC{i}' for i in range(20)],
        'label': np.random.randint(0, 2, 20)
    })
    test_features = {smi: np.random.randn(5).tolist() for smi in test_df['smiles']}
    
    test_file = _store_resource(test_df, str(TEST_MANIFEST), 'eval_single_test', 'Test', 'csv')
    test_feat = _store_resource(test_features, str(TEST_MANIFEST), 'eval_single_test_feat', 'Features', 'json')
    
    # Train model
    train_result = train_single_ml_model(
        train_input_filename=train_file,
        train_feature_vectors_filename=train_feat,
        train_smiles_column='smiles',
        train_label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='single_model_eval',
        explanation='Single model for evaluation',
        model_algorithm='random_forest_classifier',
        random_state=42
    )
    
    # Evaluate model
    eval_result = evaluate_models(
        model_filename=train_result['output_filename'],
        test_input_filename=test_file,
        feature_vectors_filename=test_feat,
        test_smiles_column='smiles',
        test_label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['accuracy', 'f1_score', 'precision', 'recall'],
        output_filename='single_eval_results',
        explanation='Evaluation of single model'
    )
    
    # Verify results
    assert 'output_filename' in eval_result
    assert 'metrics_computed' in eval_result
    assert 'n_samples' in eval_result
    assert eval_result['n_samples'] == 20
    
    # Check metrics
    metrics = eval_result['metrics_computed']
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    
    # Metrics should be floats or None
    for metric_name, value in metrics.items():
        assert value is None or isinstance(value, (float, int))
        if value is not None:
            assert 0 <= value <= 1  # All metrics should be in [0, 1]
    
    # Load report
    report = _load_resource(str(TEST_MANIFEST), eval_result['output_filename'])
    assert report['evaluation_type'] == 'single_model_evaluation'
    assert report['test_dataset']['n_samples'] == 20
    assert 'metrics_computed' in report


def test_evaluate_single_model_regression():
    """Test evaluation of a single regression model."""
    np.random.seed(42)
    
    # Create data
    train_df = pd.DataFrame({
        'smiles': [f'C{i}' for i in range(30)],
        'label': np.random.randn(30)
    })
    train_features = {smi: np.random.randn(3).tolist() for smi in train_df['smiles']}
    
    test_df = pd.DataFrame({
        'smiles': [f'CC{i}' for i in range(10)],
        'label': np.random.randn(10)
    })
    test_features = {smi: np.random.randn(3).tolist() for smi in test_df['smiles']}
    
    train_file = _store_resource(train_df, str(TEST_MANIFEST), 'eval_reg_train', 'Train', 'csv')
    train_feat = _store_resource(train_features, str(TEST_MANIFEST), 'eval_reg_train_feat', 'Features', 'json')
    test_file = _store_resource(test_df, str(TEST_MANIFEST), 'eval_reg_test', 'Test', 'csv')
    test_feat = _store_resource(test_features, str(TEST_MANIFEST), 'eval_reg_test_feat', 'Features', 'json')
    
    # Train
    train_result = train_single_ml_model(
        train_input_filename=train_file,
        train_feature_vectors_filename=train_feat,
        train_smiles_column='smiles',
        train_label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='reg_model_eval',
        explanation='Regression model',
        model_algorithm='random_forest_regressor',
        random_state=42
    )
    
    # Evaluate
    eval_result = evaluate_models(
        model_filename=train_result['output_filename'],
        test_input_filename=test_file,
        feature_vectors_filename=test_feat,
        test_smiles_column='smiles',
        test_label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['mse', 'mae', 'r2'],
        output_filename='reg_eval_results',
        explanation='Regression evaluation'
    )
    
    # Verify
    assert eval_result['n_samples'] == 10
    metrics = eval_result['metrics_computed']
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    
    # MSE and MAE should be >= 0
    if metrics['mse'] is not None:
        assert metrics['mse'] >= 0
    if metrics['mae'] is not None:
        assert metrics['mae'] >= 0


def test_evaluate_model_error_handling():
    """Test error handling in evaluate_model."""
    df = pd.DataFrame({'smiles': ['CCO'], 'label': [1]})
    features = {'CCO': [1.0]}
    
    data_file = _store_resource(df, str(TEST_MANIFEST), 'eval_error_data', 'Data', 'csv')
    feat_file = _store_resource(features, str(TEST_MANIFEST), 'eval_error_feat', 'Features', 'json')
    
    train_result = train_single_ml_model(
        train_input_filename=data_file,
        train_feature_vectors_filename=feat_file,
        train_smiles_column='smiles',
        train_label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='error_model',
        explanation='Error test',
        model_algorithm='random_forest_classifier',
        random_state=42
    )
    
    # Missing columns error
    with pytest.raises(ValueError, match="Required columns not found"):
        evaluate_models(
            model_filename=train_result['output_filename'],
            test_input_filename=data_file,
            feature_vectors_filename=feat_file,
            test_smiles_column='wrong_column',
            test_label_column='label',
            project_manifest_path=str(TEST_MANIFEST),
            metrics=['accuracy'],
            output_filename='should_fail',
        )


if __name__ == "__main__":
    test_evaluate_single_model_classification()
    print("✓ Classification evaluation test passed")
    
    test_evaluate_single_model_regression()
    print("✓ Regression evaluation test passed")
    
    test_evaluate_model_error_handling()
    print("✓ Error handling test passed")
    
    print("\n✓ All tests passed!")
