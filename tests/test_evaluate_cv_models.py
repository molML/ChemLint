"""Test evaluate_cv_models function."""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molml_mcp.tools.ml.training import train_ml_models_cross_validation
from molml_mcp.tools.ml.evaluation import evaluate_models
from molml_mcp.infrastructure.resources import _store_resource, _load_resource

# Use existing test manifest
TEST_MANIFEST = Path(__file__).parent / "data" / "test_manifest.json"


@pytest.fixture
def classification_data():
    """Create classification test dataset and feature vectors."""
    # Create test data for binary classification
    smiles_list = [
        'CCO', 'CC(C)O', 'CCCO', 'CCCCO', 'CC(C)CO',
        'CCCCCO', 'CC(C)CCO', 'CCCCCCO', 'CC(C)CCCO', 'CCCCCCCO',
        'CCC', 'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC',
        'c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1'
    ]
    
    # Binary classification: alcohols (1) vs non-alcohols (0)
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # alcohols
              0, 0, 0, 0, 0,                  # alkanes
              0, 0, 0, 1, 1]                  # aromatics (some with OH/NH2)
    
    df = pd.DataFrame({
        'smiles': smiles_list,
        'label': labels
    })
    
    # Create simple feature vectors (carbon count, oxygen count, has aromatic ring)
    feature_vectors = {}
    for smi in smiles_list:
        n_carbons = smi.count('C')
        n_oxygens = smi.count('O')
        has_aromatic = int('c' in smi.lower() and 'cc' in smi.lower())
        feature_vectors[smi] = [float(n_carbons), float(n_oxygens), float(has_aromatic)]
    
    return df, feature_vectors


@pytest.fixture
def regression_data():
    """Create regression test dataset and feature vectors."""
    # Create test data for regression
    smiles_list = ['C'*i + 'O' for i in range(1, 21)]  # C1O, C2O, ..., C20O
    
    # Regression target: carbon count (simple relationship)
    labels = [float(smi.count('C')) for smi in smiles_list]
    
    df = pd.DataFrame({
        'smiles': smiles_list,
        'target': labels
    })
    
    # Simple features
    feature_vectors = {smi: [float(smi.count('C')), 1.0, float(smi.count('C'))**2] 
                      for smi in smiles_list}
    
    return df, feature_vectors


def test_evaluate_cv_models_validation_mode(classification_data):
    """Test CV evaluation using validation sets (default mode)."""
    df, feature_vectors = classification_data
    
    # Store test data
    data_file = _store_resource(df, str(TEST_MANIFEST), 'eval_cv_test_data', 'Test data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'eval_cv_test_features', 'Test features', 'json')
    
    # Train CV models
    train_result = train_ml_models_cross_validation(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='eval_cv_models',
        explanation='Test CV models for evaluation',
        model_algorithm='random_forest_classifier',
        cv_strategy='montecarlo',
        n_folds=5,
        val_size=0.3,
        random_state=42
    )
    
    # Evaluate CV models on validation sets
    result = evaluate_models(
        model_filename=train_result['output_filename'],
        feature_vectors_filename=features_file,
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['accuracy', 'f1_score', 'precision', 'recall'],
        output_filename='cv_evaluation',
        use_cv_validation_sets=True
    )
    
    # Verify result structure
    assert 'output_filename' in result
    assert result['n_models'] == 5
    assert result['cv_strategy'] == 'montecarlo'
    assert result['evaluation_mode'] == 'cv_validation'
    assert 'metrics_summary' in result
    
    # Verify metrics were computed
    metrics_summary = result['metrics_summary']
    assert 'accuracy' in metrics_summary
    assert 'f1_score' in metrics_summary
    
    # Check that each metric has statistics
    for metric_name in ['accuracy', 'f1_score', 'precision', 'recall']:
        assert metric_name in metrics_summary
        assert 'mean' in metrics_summary[metric_name]
        assert 'std' in metrics_summary[metric_name]
        assert 'min' in metrics_summary[metric_name]
        assert 'max' in metrics_summary[metric_name]
        assert metrics_summary[metric_name]['n_folds'] > 0
    
    # Load detailed report
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    assert report['evaluation_mode'] == 'cv_validation'
    assert report['n_models'] == 5
    assert len(report['per_fold_metrics']) == 5
    
    # Check per-fold metrics
    for fold_metrics in report['per_fold_metrics']:
        assert 'fold' in fold_metrics
        assert 'n_samples' in fold_metrics
        assert 'metrics' in fold_metrics
        assert 'accuracy' in fold_metrics['metrics']


def test_evaluate_cv_models_test_mode(classification_data):
    """Test CV evaluation using independent test set."""
    df, feature_vectors = classification_data
    
    # Split data into train and test
    train_df = df.iloc[:15].copy()
    test_df = df.iloc[15:].copy()
    
    # Store train data
    train_data_file = _store_resource(train_df, str(TEST_MANIFEST), 'eval_cv_train_data', 'Train data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'eval_cv_features_test', 'Features', 'json')
    
    # Train CV models on train set
    train_result = train_ml_models_cross_validation(
        input_filename=train_data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='eval_cv_models_test',
        explanation='Test CV models',
        model_algorithm='random_forest_classifier',
        cv_strategy='kfold',
        n_folds=3,
        random_state=42
    )
    
    # Store test data
    test_data_file = _store_resource(test_df, str(TEST_MANIFEST), 'eval_cv_test_data_independent', 'Test data', 'csv')
    
    # Evaluate CV models on test set
    result = evaluate_models(
        model_filename=train_result['output_filename'],
        feature_vectors_filename=features_file,
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['accuracy', 'f1_score'],
        output_filename='cv_evaluation_test',
        use_cv_validation_sets=False,
        test_input_filename=test_data_file,
        test_smiles_column='smiles',
        test_label_column='label'
    )
    
    # Verify test mode
    assert result['evaluation_mode'] == 'test'
    assert result['n_models'] == 3
    
    # Load report and verify test set info
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    assert 'test_dataset' in report
    assert report['test_dataset']['filename'] == test_data_file
    assert report['test_dataset']['n_samples'] == len(test_df)
    
    # All models evaluated on same test set
    for fold_metrics in report['per_fold_metrics']:
        assert fold_metrics['n_samples'] == len(test_df)


def test_evaluate_cv_models_regression(regression_data):
    """Test CV evaluation with regression metrics."""
    df, feature_vectors = regression_data
    
    # Store data
    data_file = _store_resource(df, str(TEST_MANIFEST), 'eval_cv_regression_data', 'Regression data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'eval_cv_regression_features', 'Features', 'json')
    
    # Train CV models
    train_result = train_ml_models_cross_validation(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='target',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='eval_cv_regression_models',
        explanation='Regression CV models',
        model_algorithm='random_forest_regressor',
        cv_strategy='kfold',
        n_folds=5,
        random_state=42
    )
    
    # Evaluate with regression metrics
    result = evaluate_models(
        model_filename=train_result['output_filename'],
        feature_vectors_filename=features_file,
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['mse', 'mae', 'r2'],
        output_filename='cv_evaluation_regression',
        use_cv_validation_sets=True
    )
    
    # Verify regression metrics
    metrics_summary = result['metrics_summary']
    assert 'mse' in metrics_summary
    assert 'mae' in metrics_summary
    assert 'r2' in metrics_summary
    
    # MSE and MAE should be computed
    assert metrics_summary['mse']['n_folds'] == 5
    assert metrics_summary['mae']['n_folds'] == 5
    assert metrics_summary['r2']['n_folds'] == 5
    
    # Check reasonable values (model should learn the simple relationship)
    assert metrics_summary['r2']['mean'] > 0.5  # Should have decent R²


def test_evaluate_cv_models_stratified(classification_data):
    """Test CV evaluation with stratified CV models."""
    df, feature_vectors = classification_data
    
    # Store data
    data_file = _store_resource(df, str(TEST_MANIFEST), 'eval_cv_stratified_data', 'Stratified data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'eval_cv_stratified_features', 'Features', 'json')
    
    # Train with stratified CV
    train_result = train_ml_models_cross_validation(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='eval_cv_stratified_models',
        explanation='Stratified CV models',
        model_algorithm='random_forest_classifier',
        cv_strategy='stratified',
        n_folds=5,
        random_state=42
    )
    
    # Evaluate
    result = evaluate_models(
        model_filename=train_result['output_filename'],
        feature_vectors_filename=features_file,
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['accuracy', 'balanced_accuracy'],
        output_filename='cv_evaluation_stratified',
        use_cv_validation_sets=True
    )
    
    # Verify CV strategy preserved
    assert result['cv_strategy'] == 'stratified'
    
    # Load report
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    assert report['cv_strategy'] == 'stratified'


def test_evaluate_cv_models_metrics_consistency():
    """Test that metrics are consistent across folds."""
    # Create simple synthetic data where model should perform well
    smiles_list = ['C'*i for i in range(1, 21)]
    labels = [0]*10 + [1]*10  # Perfect split by length
    
    df = pd.DataFrame({
        'smiles': smiles_list,
        'label': labels
    })
    
    # Features that perfectly separate classes
    feature_vectors = {smi: [float(len(smi)), float(len(smi))**2] for smi in smiles_list}
    
    # Store data
    data_file = _store_resource(df, str(TEST_MANIFEST), 'eval_cv_perfect_data', 'Perfect data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'eval_cv_perfect_features', 'Features', 'json')
    
    # Train CV models
    train_result = train_ml_models_cross_validation(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='eval_cv_perfect_models',
        explanation='Perfect separation models',
        model_algorithm='random_forest_classifier',
        cv_strategy='kfold',
        n_folds=5,
        random_state=42
    )
    
    # Evaluate
    result = evaluate_models(
        model_filename=train_result['output_filename'],
        feature_vectors_filename=features_file,
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['accuracy'],
        output_filename='cv_evaluation_perfect',
        use_cv_validation_sets=True
    )
    
    # With perfect separation, accuracy should be very high and consistent
    metrics_summary = result['metrics_summary']
    assert metrics_summary['accuracy']['mean'] > 0.8  # High accuracy
    assert metrics_summary['accuracy']['std'] < 0.3   # Low variance


def test_evaluate_cv_models_error_handling():
    """Test error handling for invalid inputs."""
    df = pd.DataFrame({'smiles': ['CCO', 'CCC'], 'label': [1, 0]})
    feature_vectors = {'CCO': [1.0], 'CCC': [2.0]}
    
    data_file = _store_resource(df, str(TEST_MANIFEST), 'eval_cv_error_data', 'Error data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'eval_cv_error_features', 'Features', 'json')
    
    # Train a simple model
    train_result = train_ml_models_cross_validation(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='eval_cv_error_models',
        explanation='Error test models',
        model_algorithm='random_forest_classifier',
        cv_strategy='kfold',
        n_folds=2,
        random_state=42
    )
    
    # Test: missing test file when use_cv_validation_sets=False
    with pytest.raises(ValueError, match="test_input_filename.*required"):
        evaluate_models(
            model_filename=train_result['output_filename'],
            feature_vectors_filename=features_file,
            project_manifest_path=str(TEST_MANIFEST),
            metrics=['accuracy'],
            output_filename='should_fail',
            use_cv_validation_sets=False
        )
    
    # Test: missing smiles column
    with pytest.raises(ValueError, match="test_smiles_column.*required"):
        evaluate_models(
            model_filename=train_result['output_filename'],
            feature_vectors_filename=features_file,
            project_manifest_path=str(TEST_MANIFEST),
            metrics=['accuracy'],
            output_filename='should_fail',
            use_cv_validation_sets=False,
            test_input_filename=data_file
        )


def test_evaluate_cv_models_summary_statistics():
    """Test that summary statistics are correctly computed."""
    df = pd.DataFrame({
        'smiles': ['C'*i for i in range(1, 21)],
        'label': [0]*10 + [1]*10
    })
    
    feature_vectors = {smi: [float(len(smi))] for smi in df['smiles']}
    
    data_file = _store_resource(df, str(TEST_MANIFEST), 'eval_cv_stats_data', 'Stats data', 'csv')
    features_file = _store_resource(feature_vectors, str(TEST_MANIFEST), 'eval_cv_stats_features', 'Features', 'json')
    
    # Train models
    train_result = train_ml_models_cross_validation(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='eval_cv_stats_models',
        explanation='Stats test models',
        model_algorithm='random_forest_classifier',
        cv_strategy='kfold',
        n_folds=4,
        random_state=42
    )
    
    # Evaluate
    result = evaluate_models(
        model_filename=train_result['output_filename'],
        feature_vectors_filename=features_file,
        project_manifest_path=str(TEST_MANIFEST),
        metrics=['accuracy'],
        output_filename='cv_evaluation_stats',
        use_cv_validation_sets=True
    )
    
    # Verify statistics
    report = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    
    accuracy_stats = report['metrics_summary']['accuracy']
    
    # Check all statistics are present and make sense
    assert 'mean' in accuracy_stats
    assert 'std' in accuracy_stats
    assert 'min' in accuracy_stats
    assert 'max' in accuracy_stats
    assert 'values' in accuracy_stats
    
    # Min <= mean <= max
    assert accuracy_stats['min'] <= accuracy_stats['mean'] <= accuracy_stats['max']
    
    # Number of values matches number of folds
    assert len(accuracy_stats['values']) == 4
    assert accuracy_stats['n_folds'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
