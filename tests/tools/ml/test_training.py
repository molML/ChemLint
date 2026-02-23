"""
Tests for training.py module.

Tests cover:
- train_single_ml_model: single model training
- train_ml_models_cross_validation: CV-based training
- _train_ml_model: internal training function
- Error handling and validation
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
from chemlint.tools.ml.training import (
    train_single_ml_model,
    train_ml_models_cross_validation,
    _train_ml_model
)
from chemlint.infrastructure.resources import _store_resource


# Fixtures
@pytest.fixture
def sample_train_data(session_workdir):
    """Create sample training dataset and feature vectors."""
    # Create training data
    train_df = pd.DataFrame({
        'smiles': ['CCO', 'CC(O)C', 'c1ccccc1', 'CC(=O)O', 'CCCC', 
                   'CCC(C)C', 'c1ccc(O)cc1', 'CCN', 'CCCO', 'c1ccncc1'],
        'activity': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    # Create feature vectors (10 samples x 5 features)
    feature_dict = {
        'CCO': [1.0, 2.0, 3.0, 4.0, 5.0],
        'CC(O)C': [1.5, 2.5, 3.5, 4.5, 5.5],
        'c1ccccc1': [2.0, 3.0, 4.0, 5.0, 6.0],
        'CC(=O)O': [2.5, 3.5, 4.5, 5.5, 6.5],
        'CCCC': [3.0, 4.0, 5.0, 6.0, 7.0],
        'CCC(C)C': [3.5, 4.5, 5.5, 6.5, 7.5],
        'c1ccc(O)cc1': [4.0, 5.0, 6.0, 7.0, 8.0],
        'CCN': [4.5, 5.5, 6.5, 7.5, 8.5],
        'CCCO': [5.0, 6.0, 7.0, 8.0, 9.0],
        'c1ccncc1': [5.5, 6.5, 7.5, 8.5, 9.5]
    }
    
    # Create manifest path
    manifest_path = session_workdir / "manifest.json"
    manifest_data = {"resources": []}
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)
    
    # Store resources
    train_filename = _store_resource(
        train_df,
        str(manifest_path),
        "train_data",
        "Training dataset",
        "csv"
    )
    
    features_filename = _store_resource(
        feature_dict,
        str(manifest_path),
        "feature_vectors",
        "Feature vectors",
        "json"
    )
    
    return {
        'manifest_path': str(manifest_path),
        'train_filename': train_filename,
        'features_filename': features_filename,
        'train_df': train_df,
        'feature_dict': feature_dict
    }


@pytest.fixture
def sample_regression_data(session_workdir):
    """Create sample regression dataset."""
    # Create regression data
    train_df = pd.DataFrame({
        'smiles': ['CCO', 'CC(O)C', 'c1ccccc1', 'CC(=O)O', 'CCCC'],
        'pIC50': [5.2, 6.3, 4.8, 7.1, 5.9]
    })
    
    feature_dict = {
        'CCO': [1.0, 2.0, 3.0],
        'CC(O)C': [1.5, 2.5, 3.5],
        'c1ccccc1': [2.0, 3.0, 4.0],
        'CC(=O)O': [2.5, 3.5, 4.5],
        'CCCC': [3.0, 4.0, 5.0]
    }
    
    manifest_path = session_workdir / "manifest.json"
    manifest_data = {"resources": []}
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)
    
    train_filename = _store_resource(
        train_df,
        str(manifest_path),
        "train_data_reg",
        "Regression training dataset",
        "csv"
    )
    
    features_filename = _store_resource(
        feature_dict,
        str(manifest_path),
        "feature_vectors_reg",
        "Regression feature vectors",
        "json"
    )
    
    return {
        'manifest_path': str(manifest_path),
        'train_filename': train_filename,
        'features_filename': features_filename
    }


@pytest.fixture
def sample_cv_data(session_workdir):
    """Create sample data with cluster and scaffold columns for CV testing."""
    train_df = pd.DataFrame({
        'smiles': ['CCO', 'CC(O)C', 'c1ccccc1', 'CC(=O)O', 'CCCC', 
                   'CCC(C)C', 'c1ccc(O)cc1', 'CCN', 'CCCO', 'c1ccncc1',
                   'c1cccnc1', 'CCCCC'],
        'activity': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'cluster': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'scaffold': ['C', 'C', 'c1ccccc1', 'c1ccccc1', 'C', 'C', 
                     'c1ccccc1', 'C', 'C', 'c1ccncc1', 'c1ccncc1', 'C']
    })
    
    feature_dict = {smi: [float(i)] * 3 for i, smi in enumerate(train_df['smiles'])}
    
    manifest_path = session_workdir / "manifest.json"
    manifest_data = {"resources": []}
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)
    
    train_filename = _store_resource(
        train_df,
        str(manifest_path),
        "train_data_cv",
        "CV training dataset",
        "csv"
    )
    
    features_filename = _store_resource(
        feature_dict,
        str(manifest_path),
        "feature_vectors_cv",
        "CV feature vectors",
        "json"
    )
    
    return {
        'manifest_path': str(manifest_path),
        'train_filename': train_filename,
        'features_filename': features_filename
    }


# ========== _train_ml_model Tests ==========

def test_train_ml_model_classifier():
    """Test internal training function with classifier."""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=50, n_features=10, random_state=42)
    
    model = _train_ml_model(
        X=X,
        y=y,
        model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 10},
        random_state=42
    )
    
    # Check model was trained
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    # Check predictions work
    predictions = model.predict(X[:5])
    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)


def test_train_ml_model_regressor():
    """Test internal training function with regressor."""
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=50, n_features=10, random_state=42)
    
    model = _train_ml_model(
        X=X,
        y=y,
        model_algorithm='ridge',
        hyperparameters={'alpha': 1.0},
        random_state=42
    )
    
    # Check model was trained
    assert hasattr(model, 'predict')
    
    # Check predictions work
    predictions = model.predict(X[:5])
    assert len(predictions) == 5
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)


def test_train_ml_model_no_hyperparameters():
    """Test training with None hyperparameters."""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=50, n_features=10, random_state=42)
    
    model = _train_ml_model(
        X=X,
        y=y,
        model_algorithm='logistic_regression',
        hyperparameters=None,
        random_state=42
    )
    
    assert hasattr(model, 'predict')


def test_train_ml_model_invalid_algorithm():
    """Test that invalid algorithm raises ValueError."""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=50, n_features=10, random_state=42)
    
    with pytest.raises(ValueError, match="Invalid model_algorithm"):
        _train_ml_model(
            X=X,
            y=y,
            model_algorithm='invalid_model',
            hyperparameters={},
            random_state=42
        )


def test_train_ml_model_deterministic():
    """Test that training is deterministic with same random_state."""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=50, n_features=10, random_state=42)
    
    model1 = _train_ml_model(
        X=X, y=y, model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 10}, random_state=42
    )
    
    model2 = _train_ml_model(
        X=X, y=y, model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 10}, random_state=42
    )
    
    # Predictions should be identical
    pred1 = model1.predict(X[:10])
    pred2 = model2.predict(X[:10])
    np.testing.assert_array_equal(pred1, pred2)


# ========== train_single_ml_model Tests ==========

def test_train_single_ml_model_basic(sample_train_data):
    """Test basic single model training."""
    result = train_single_ml_model(
        train_input_filename=sample_train_data['train_filename'],
        train_feature_vectors_filename=sample_train_data['features_filename'],
        train_smiles_column='smiles',
        train_label_column='activity',
        project_manifest_path=sample_train_data['manifest_path'],
        output_filename='trained_model',
        explanation='Test model',
        model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 10},
        random_state=42
    )
    
    # Check return structure
    assert 'output_filename' in result
    assert 'model_algorithm' in result
    assert 'n_train_samples' in result
    assert 'n_features' in result
    
    # Check values
    assert result['model_algorithm'] == 'random_forest_classifier'
    assert result['n_train_samples'] == 10
    assert result['n_features'] == 5
    assert 'trained_model_' in result['output_filename']


def test_train_single_ml_model_regression(sample_regression_data):
    """Test single model training with regression."""
    result = train_single_ml_model(
        train_input_filename=sample_regression_data['train_filename'],
        train_feature_vectors_filename=sample_regression_data['features_filename'],
        train_smiles_column='smiles',
        train_label_column='pIC50',
        project_manifest_path=sample_regression_data['manifest_path'],
        output_filename='regression_model',
        explanation='Regression test model',
        model_algorithm='ridge',
        hyperparameters={'alpha': 1.0},
        random_state=42
    )
    
    assert result['model_algorithm'] == 'ridge'
    assert result['n_train_samples'] == 5
    assert result['n_features'] == 3


def test_train_single_ml_model_error_handling(sample_train_data):
    """Test error handling for missing columns and invalid algorithm."""
    # Missing SMILES column
    with pytest.raises(ValueError, match="SMILES column .* not found"):
        train_single_ml_model(
            train_input_filename=sample_train_data['train_filename'],
            train_feature_vectors_filename=sample_train_data['features_filename'],
            train_smiles_column='wrong_column',
            train_label_column='activity',
            project_manifest_path=sample_train_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            random_state=42
        )
    
    # Missing label column
    with pytest.raises(ValueError, match="Label column .* not found"):
        train_single_ml_model(
            train_input_filename=sample_train_data['train_filename'],
            train_feature_vectors_filename=sample_train_data['features_filename'],
            train_smiles_column='smiles',
            train_label_column='wrong_column',
            project_manifest_path=sample_train_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            random_state=42
        )
    
    # Invalid algorithm
    with pytest.raises(ValueError, match="not supported"):
        train_single_ml_model(
            train_input_filename=sample_train_data['train_filename'],
            train_feature_vectors_filename=sample_train_data['features_filename'],
            train_smiles_column='smiles',
            train_label_column='activity',
            project_manifest_path=sample_train_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='invalid_model',
            random_state=42
        )


def test_train_single_ml_model_missing_features(sample_train_data):
    """Test error when some SMILES don't have feature vectors."""
    # Create incomplete feature dict
    incomplete_features = {k: v for k, v in list(sample_train_data['feature_dict'].items())[:5]}
    
    # Store incomplete features
    from chemlint.infrastructure.resources import _store_resource
    incomplete_filename = _store_resource(
        incomplete_features,
        sample_train_data['manifest_path'],
        'incomplete_features',
        'Incomplete features',
        'json'
    )
    
    with pytest.raises(ValueError, match="Missing feature vectors"):
        train_single_ml_model(
            train_input_filename=sample_train_data['train_filename'],
            train_feature_vectors_filename=incomplete_filename,
            train_smiles_column='smiles',
            train_label_column='activity',
            project_manifest_path=sample_train_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            random_state=42
        )


def test_train_single_ml_model_default_hyperparameters(sample_train_data):
    """Test training with default hyperparameters (None)."""
    result = train_single_ml_model(
        train_input_filename=sample_train_data['train_filename'],
        train_feature_vectors_filename=sample_train_data['features_filename'],
        train_smiles_column='smiles',
        train_label_column='activity',
        project_manifest_path=sample_train_data['manifest_path'],
        output_filename='model_defaults',
        explanation='Test with defaults',
        model_algorithm='decision_tree_classifier',
        hyperparameters=None,
        random_state=42
    )
    
    assert result['hyperparameters'] == {}


# ========== train_ml_models_cross_validation Tests ==========

def test_train_ml_models_cv_kfold(sample_cv_data):
    """Test CV training with kfold strategy."""
    result = train_ml_models_cross_validation(
        input_filename=sample_cv_data['train_filename'],
        feature_vectors_filename=sample_cv_data['features_filename'],
        smiles_column='smiles',
        label_column='activity',
        project_manifest_path=sample_cv_data['manifest_path'],
        output_filename='cv_kfold_model',
        explanation='KFold CV test',
        model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 5},
        cv_strategy='kfold',
        n_folds=3,
        random_state=42
    )
    
    # Check return structure
    assert 'output_filename' in result
    assert 'n_models' in result
    assert 'cv_strategy' in result
    assert 'model_algorithm' in result
    
    # Check values
    assert result['n_models'] == 3
    assert result['cv_strategy'] == 'kfold'
    assert result['n_folds'] == 3
    assert result['model_algorithm'] == 'random_forest_classifier'


def test_train_ml_models_cv_stratified(sample_cv_data):
    """Test CV training with stratified strategy."""
    result = train_ml_models_cross_validation(
        input_filename=sample_cv_data['train_filename'],
        feature_vectors_filename=sample_cv_data['features_filename'],
        smiles_column='smiles',
        label_column='activity',
        project_manifest_path=sample_cv_data['manifest_path'],
        output_filename='cv_stratified_model',
        explanation='Stratified CV test',
        model_algorithm='logistic_regression',
        cv_strategy='stratified',
        n_folds=2,
        random_state=42
    )
    
    assert result['n_models'] == 2
    assert result['cv_strategy'] == 'stratified'


def test_train_ml_models_cv_montecarlo(sample_cv_data):
    """Test CV training with montecarlo strategy."""
    result = train_ml_models_cross_validation(
        input_filename=sample_cv_data['train_filename'],
        feature_vectors_filename=sample_cv_data['features_filename'],
        smiles_column='smiles',
        label_column='activity',
        project_manifest_path=sample_cv_data['manifest_path'],
        output_filename='cv_montecarlo_model',
        explanation='Monte Carlo CV test',
        model_algorithm='random_forest_classifier',
        cv_strategy='montecarlo',
        n_folds=4,
        val_size=0.2,
        random_state=42
    )
    
    assert result['n_models'] == 4
    assert result['cv_strategy'] == 'montecarlo'


def test_train_ml_models_cv_cluster(sample_cv_data):
    """Test CV training with cluster strategy."""
    result = train_ml_models_cross_validation(
        input_filename=sample_cv_data['train_filename'],
        feature_vectors_filename=sample_cv_data['features_filename'],
        smiles_column='smiles',
        label_column='activity',
        project_manifest_path=sample_cv_data['manifest_path'],
        output_filename='cv_cluster_model',
        explanation='Cluster CV test',
        model_algorithm='random_forest_classifier',
        cv_strategy='cluster',
        n_folds=3,
        cluster_column='cluster',
        random_state=42
    )
    
    assert result['n_models'] == 3
    assert result['cv_strategy'] == 'cluster'


def test_train_ml_models_cv_scaffold(sample_cv_data):
    """Test CV training with scaffold strategy."""
    result = train_ml_models_cross_validation(
        input_filename=sample_cv_data['train_filename'],
        feature_vectors_filename=sample_cv_data['features_filename'],
        smiles_column='smiles',
        label_column='activity',
        project_manifest_path=sample_cv_data['manifest_path'],
        output_filename='cv_scaffold_model',
        explanation='Scaffold CV test',
        model_algorithm='random_forest_classifier',
        cv_strategy='scaffold',
        n_folds=2,
        scaffold_column='scaffold',
        random_state=42
    )
    
    assert result['n_models'] == 2
    assert result['cv_strategy'] == 'scaffold'


def test_train_ml_models_cv_error_handling(sample_cv_data):
    """Test error handling for missing columns in CV training."""
    # Missing SMILES column
    with pytest.raises(ValueError, match="SMILES column .* not found"):
        train_ml_models_cross_validation(
            input_filename=sample_cv_data['train_filename'],
            feature_vectors_filename=sample_cv_data['features_filename'],
            smiles_column='wrong_column',
            label_column='activity',
            project_manifest_path=sample_cv_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            cv_strategy='kfold',
            n_folds=3,
            random_state=42
        )
    
    # Missing label column
    with pytest.raises(ValueError, match="Label column .* not found"):
        train_ml_models_cross_validation(
            input_filename=sample_cv_data['train_filename'],
            feature_vectors_filename=sample_cv_data['features_filename'],
            smiles_column='smiles',
            label_column='wrong_column',
            project_manifest_path=sample_cv_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            cv_strategy='kfold',
            n_folds=3,
            random_state=42
        )


def test_train_ml_models_cv_cluster_without_column(sample_cv_data):
    """Test error when cluster strategy used without cluster_column."""
    with pytest.raises(ValueError, match="Cluster-based CV requires"):
        train_ml_models_cross_validation(
            input_filename=sample_cv_data['train_filename'],
            feature_vectors_filename=sample_cv_data['features_filename'],
            smiles_column='smiles',
            label_column='activity',
            project_manifest_path=sample_cv_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            cv_strategy='cluster',
            n_folds=3,
            cluster_column=None,
            random_state=42
        )


def test_train_ml_models_cv_scaffold_without_column(sample_cv_data):
    """Test error when scaffold strategy used without scaffold_column."""
    with pytest.raises(ValueError, match="Scaffold-based CV requires"):
        train_ml_models_cross_validation(
            input_filename=sample_cv_data['train_filename'],
            feature_vectors_filename=sample_cv_data['features_filename'],
            smiles_column='smiles',
            label_column='activity',
            project_manifest_path=sample_cv_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            cv_strategy='scaffold',
            n_folds=3,
            scaffold_column=None,
            random_state=42
        )


def test_train_ml_models_cv_invalid_algorithm(sample_cv_data):
    """Test error with invalid model algorithm in CV."""
    with pytest.raises(ValueError, match="not supported"):
        train_ml_models_cross_validation(
            input_filename=sample_cv_data['train_filename'],
            feature_vectors_filename=sample_cv_data['features_filename'],
            smiles_column='smiles',
            label_column='activity',
            project_manifest_path=sample_cv_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='invalid_model',
            cv_strategy='kfold',
            n_folds=3,
            random_state=42
        )


def test_train_ml_models_cv_cluster_column_not_found(sample_cv_data):
    """Test error when specified cluster column doesn't exist."""
    with pytest.raises(ValueError, match="Cluster column .* not found"):
        train_ml_models_cross_validation(
            input_filename=sample_cv_data['train_filename'],
            feature_vectors_filename=sample_cv_data['features_filename'],
            smiles_column='smiles',
            label_column='activity',
            project_manifest_path=sample_cv_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            cv_strategy='cluster',
            n_folds=3,
            cluster_column='nonexistent_column',
            random_state=42
        )


def test_train_ml_models_cv_scaffold_column_not_found(sample_cv_data):
    """Test error when specified scaffold column doesn't exist."""
    with pytest.raises(ValueError, match="Scaffold column .* not found"):
        train_ml_models_cross_validation(
            input_filename=sample_cv_data['train_filename'],
            feature_vectors_filename=sample_cv_data['features_filename'],
            smiles_column='smiles',
            label_column='activity',
            project_manifest_path=sample_cv_data['manifest_path'],
            output_filename='model',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            cv_strategy='scaffold',
            n_folds=3,
            scaffold_column='nonexistent_column',
            random_state=42
        )


def test_train_ml_models_cv_default_hyperparameters(sample_cv_data):
    """Test CV training with default hyperparameters."""
    result = train_ml_models_cross_validation(
        input_filename=sample_cv_data['train_filename'],
        feature_vectors_filename=sample_cv_data['features_filename'],
        smiles_column='smiles',
        label_column='activity',
        project_manifest_path=sample_cv_data['manifest_path'],
        output_filename='cv_defaults',
        explanation='CV with defaults',
        model_algorithm='decision_tree_classifier',
        hyperparameters=None,
        cv_strategy='kfold',
        n_folds=2,
        random_state=42
    )
    
    assert result['hyperparameters'] == {}
    assert result['n_models'] == 2


# ========== Integration Tests ==========

def test_single_and_cv_produce_valid_models(sample_train_data):
    """Test that both training functions produce valid, loadable models."""
    from chemlint.infrastructure.resources import _load_resource
    
    # Train single model
    result_single = train_single_ml_model(
        train_input_filename=sample_train_data['train_filename'],
        train_feature_vectors_filename=sample_train_data['features_filename'],
        train_smiles_column='smiles',
        train_label_column='activity',
        project_manifest_path=sample_train_data['manifest_path'],
        output_filename='single_model',
        explanation='Single model test',
        model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 5},
        random_state=42
    )
    
    # Load and verify single model
    single_model_data = _load_resource(
        sample_train_data['manifest_path'],
        result_single['output_filename']
    )
    
    assert 'models' in single_model_data
    assert len(single_model_data['models']) == 1
    assert hasattr(single_model_data['models'][0], 'predict')
    
    # Train CV models
    result_cv = train_ml_models_cross_validation(
        input_filename=sample_train_data['train_filename'],
        feature_vectors_filename=sample_train_data['features_filename'],
        smiles_column='smiles',
        label_column='activity',
        project_manifest_path=sample_train_data['manifest_path'],
        output_filename='cv_model',
        explanation='CV model test',
        model_algorithm='random_forest_classifier',
        hyperparameters={'n_estimators': 5},
        cv_strategy='kfold',
        n_folds=2,
        random_state=42
    )
    
    # Load and verify CV models
    cv_model_data = _load_resource(
        sample_train_data['manifest_path'],
        result_cv['output_filename']
    )
    
    assert 'models' in cv_model_data
    assert len(cv_model_data['models']) == 2
    assert all(hasattr(m, 'predict') for m in cv_model_data['models'])
