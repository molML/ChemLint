"""
Tests for hyperparam_tuning.py module.

Tests cover:
- _define_search_space: grid and random search space generation
- tune_hyperparameters: hyperparameter tuning with various strategies
- Error handling and validation
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from chemlint.tools.ml.hyperparam_tuning import (
    _define_search_space,
    tune_hyperparameters
)
from chemlint.infrastructure.resources import _store_resource


# Fixtures
@pytest.fixture
def sample_tuning_data(session_workdir):
    """Create sample dataset for hyperparameter tuning."""
    # Create training data with enough samples for meaningful CV
    np.random.seed(42)
    n_samples = 30
    
    train_df = pd.DataFrame({
        'smiles': [f'SMILES_{i}' for i in range(n_samples)],
        'activity': np.random.randint(0, 2, n_samples).tolist(),
        'cluster': [i % 5 for i in range(n_samples)],
        'scaffold': [f'scaffold_{i % 3}' for i in range(n_samples)]
    })
    
    # Create feature vectors
    feature_dict = {
        f'SMILES_{i}': np.random.randn(10).tolist() 
        for i in range(n_samples)
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
        "tuning_data",
        "Tuning dataset",
        "csv"
    )
    
    features_filename = _store_resource(
        feature_dict,
        str(manifest_path),
        "tuning_features",
        "Tuning feature vectors",
        "json"
    )
    
    return {
        'manifest_path': str(manifest_path),
        'train_filename': train_filename,
        'features_filename': features_filename,
        'train_df': train_df
    }


@pytest.fixture
def sample_regression_tuning_data(session_workdir):
    """Create sample regression dataset for hyperparameter tuning."""
    np.random.seed(42)
    n_samples = 25
    
    train_df = pd.DataFrame({
        'smiles': [f'MOL_{i}' for i in range(n_samples)],
        'pIC50': np.random.uniform(4.0, 8.0, n_samples).tolist()
    })
    
    feature_dict = {
        f'MOL_{i}': np.random.randn(8).tolist() 
        for i in range(n_samples)
    }
    
    manifest_path = session_workdir / "manifest.json"
    manifest_data = {"resources": []}
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)
    
    train_filename = _store_resource(
        train_df,
        str(manifest_path),
        "regression_tuning_data",
        "Regression tuning dataset",
        "csv"
    )
    
    features_filename = _store_resource(
        feature_dict,
        str(manifest_path),
        "regression_tuning_features",
        "Regression tuning features",
        "json"
    )
    
    return {
        'manifest_path': str(manifest_path),
        'train_filename': train_filename,
        'features_filename': features_filename
    }


# ========== _define_search_space Tests ==========

def test_define_search_space_grid():
    """Test grid search space generation."""
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [3, 5]
    }
    
    search_space = _define_search_space(param_grid, 'grid', n_searches=10, random_state=42)
    
    # Grid search should return all combinations
    assert len(search_space) == 4  # 2 * 2 = 4 combinations
    
    # Check all combinations are present
    expected = [
        {'n_estimators': 10, 'max_depth': 3},
        {'n_estimators': 10, 'max_depth': 5},
        {'n_estimators': 20, 'max_depth': 3},
        {'n_estimators': 20, 'max_depth': 5}
    ]
    assert all(combo in search_space for combo in expected)


def test_define_search_space_random():
    """Test random search space generation."""
    param_grid = {
        'n_estimators': [10, 20, 30, 40],
        'max_depth': [3, 5, 7, 9]
    }
    
    search_space = _define_search_space(param_grid, 'random', n_searches=5, random_state=42)
    
    # Random search should return exactly n_searches combinations
    assert len(search_space) == 5
    
    # All returned combos should be valid
    all_combos = list({'n_estimators': n, 'max_depth': d} 
                     for n in [10, 20, 30, 40] 
                     for d in [3, 5, 7, 9])
    assert all(combo in all_combos for combo in search_space)
    
    # No duplicates
    assert len(search_space) == len(set(tuple(sorted(d.items())) for d in search_space))


def test_define_search_space_random_exceeds_total():
    """Test random search when n_searches >= total combinations."""
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [3, 5]
    }
    
    # Request more searches than total combinations
    search_space = _define_search_space(param_grid, 'random', n_searches=10, random_state=42)
    
    # Should return all combinations (same as grid search)
    assert len(search_space) == 4


def test_define_search_space_deterministic():
    """Test that random search is deterministic with same random_state."""
    param_grid = {
        'n_estimators': [10, 20, 30, 40, 50],
        'max_depth': [3, 5, 7]
    }
    
    space1 = _define_search_space(param_grid, 'random', n_searches=5, random_state=42)
    space2 = _define_search_space(param_grid, 'random', n_searches=5, random_state=42)
    
    assert space1 == space2


def test_define_search_space_invalid_strategy():
    """Test that invalid search strategy raises ValueError."""
    param_grid = {'n_estimators': [10, 20]}
    
    with pytest.raises(ValueError, match="Unknown search strategy"):
        _define_search_space(param_grid, 'invalid', n_searches=5, random_state=42)


def test_define_search_space_single_param():
    """Test search space with single parameter."""
    param_grid = {'n_estimators': [10, 20, 30]}
    
    search_space = _define_search_space(param_grid, 'grid', n_searches=10, random_state=42)
    
    assert len(search_space) == 3
    assert all('n_estimators' in combo for combo in search_space)


# ========== tune_hyperparameters Tests ==========

def test_tune_hyperparameters_grid_search_basic(sample_tuning_data):
    """Test basic grid search hyperparameter tuning."""
    param_grid = {
        'n_estimators': [5, 10],
        'max_depth': [3, 5]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='best_params',
        explanation='Grid search test',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='kfold',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    # Check return structure
    assert 'output_filename' in result
    assert 'best_hyperparameters' in result
    assert 'best_score' in result
    assert 'n_successful' in result
    assert 'n_total' in result
    assert 'success_rate' in result
    
    # Check values
    assert result['n_total'] == 4  # 2 * 2 combinations
    assert result['n_successful'] > 0
    assert 0 <= result['success_rate'] <= 1.0
    assert isinstance(result['best_hyperparameters'], dict)
    assert 'n_estimators' in result['best_hyperparameters']
    assert 'max_depth' in result['best_hyperparameters']


def test_tune_hyperparameters_random_search(sample_tuning_data):
    """Test random search hyperparameter tuning."""
    param_grid = {
        'n_estimators': [5, 10, 15, 20],
        'max_depth': [3, 5, 7, None]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='random_search_params',
        explanation='Random search test',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='random',
        n_searches=5,
        cv_strategy='stratified',
        n_folds=3,
        metric='f1_score',
        higher_is_better=True,
        random_state=42
    )
    
    assert result['n_total'] == 5  # Requested 5 random searches
    assert result['n_successful'] > 0


def test_tune_hyperparameters_regression(sample_regression_tuning_data):
    """Test hyperparameter tuning with regression model."""
    param_grid = {
        'alpha': [0.1, 1.0, 10.0]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_regression_tuning_data['train_filename'],
        feature_vectors_filename=sample_regression_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='pIC50',
        project_manifest_path=sample_regression_tuning_data['manifest_path'],
        output_filename='ridge_params',
        explanation='Ridge regression tuning',
        model_algorithm='ridge',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='kfold',
        n_folds=3,
        metric='r2',
        higher_is_better=True,
        random_state=42
    )
    
    assert result['n_total'] == 3
    assert result['n_successful'] > 0
    assert 'alpha' in result['best_hyperparameters']


def test_tune_hyperparameters_minimize_metric(sample_regression_tuning_data):
    """Test hyperparameter tuning with metric to minimize."""
    param_grid = {
        'alpha': [0.1, 1.0]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_regression_tuning_data['train_filename'],
        feature_vectors_filename=sample_regression_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='pIC50',
        project_manifest_path=sample_regression_tuning_data['manifest_path'],
        output_filename='minimize_mse_params',
        explanation='Minimize MSE',
        model_algorithm='ridge',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='kfold',
        n_folds=3,
        metric='mse',
        higher_is_better=False,  # MSE should be minimized
        random_state=42
    )
    
    assert result['n_successful'] > 0
    assert isinstance(result['best_score'], (int, float))


def test_tune_hyperparameters_stratified_cv(sample_tuning_data):
    """Test hyperparameter tuning with stratified CV."""
    param_grid = {
        'max_depth': [3, 5]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='stratified_params',
        explanation='Stratified CV test',
        model_algorithm='decision_tree_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='stratified',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    assert result['n_successful'] > 0


def test_tune_hyperparameters_cluster_cv(sample_tuning_data):
    """Test hyperparameter tuning with cluster-based CV."""
    param_grid = {
        'n_estimators': [5, 10]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='cluster_cv_params',
        explanation='Cluster CV test',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='cluster',
        n_folds=3,
        cluster_column='cluster',
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    assert result['n_successful'] > 0


def test_tune_hyperparameters_scaffold_cv(sample_tuning_data):
    """Test hyperparameter tuning with scaffold-based CV."""
    param_grid = {
        'n_estimators': [5, 10]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='scaffold_cv_params',
        explanation='Scaffold CV test',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='scaffold',
        n_folds=2,
        scaffold_column='scaffold',
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    assert result['n_successful'] > 0


def test_tune_hyperparameters_montecarlo_cv(sample_tuning_data):
    """Test hyperparameter tuning with Monte Carlo CV."""
    param_grid = {
        'n_estimators': [5]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='montecarlo_params',
        explanation='Monte Carlo CV test',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='montecarlo',
        n_folds=3,
        val_size=0.3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    assert result['n_successful'] > 0


def test_tune_hyperparameters_logistic_regression(sample_tuning_data):
    """Test hyperparameter tuning with logistic regression."""
    param_grid = {
        'C': [0.1, 1.0],
        'max_iter': [100]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='logistic_params',
        explanation='Logistic regression tuning',
        model_algorithm='logistic_regression',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='kfold',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    assert result['n_successful'] > 0
    assert 'C' in result['best_hyperparameters']


def test_tune_hyperparameters_success_rate(sample_tuning_data):
    """Test that success_rate is calculated correctly."""
    param_grid = {
        'n_estimators': [5, 10, 15]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='success_rate_test',
        explanation='Test success rate',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='kfold',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    expected_success_rate = result['n_successful'] / result['n_total']
    assert pytest.approx(result['success_rate']) == expected_success_rate
    assert 0 < result['success_rate'] <= 1.0


def test_tune_hyperparameters_stored_resource(sample_tuning_data):
    """Test that best hyperparameters are stored as JSON resource."""
    from chemlint.infrastructure.resources import _load_resource
    
    param_grid = {
        'n_estimators': [5, 10]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='stored_params',
        explanation='Test storage',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='kfold',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    # Load the stored resource
    loaded_params = _load_resource(
        sample_tuning_data['manifest_path'],
        result['output_filename']
    )
    
    # Should match the best hyperparameters
    assert loaded_params == result['best_hyperparameters']


def test_tune_hyperparameters_different_metrics(sample_tuning_data):
    """Test tuning with different evaluation metrics."""
    param_grid = {'n_estimators': [5, 10]}
    
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score']
    
    for metric in metrics:
        result = tune_hyperparameters(
            input_filename=sample_tuning_data['train_filename'],
            feature_vectors_filename=sample_tuning_data['features_filename'],
            smiles_column='smiles',
            target_column='activity',
            project_manifest_path=sample_tuning_data['manifest_path'],
            output_filename=f'params_{metric}',
            explanation=f'Test {metric}',
            model_algorithm='random_forest_classifier',
            param_grid=param_grid,
            search_strategy='grid',
            cv_strategy='kfold',
            n_folds=3,
            metric=metric,
            higher_is_better=True,
            random_state=42
        )
        
        assert result['n_successful'] > 0
        assert isinstance(result['best_score'], (int, float))


def test_tune_hyperparameters_deterministic(sample_tuning_data):
    """Test that tuning is deterministic with same random_state."""
    param_grid = {
        'n_estimators': [5, 10, 15],
        'max_depth': [3, 5]
    }
    
    result1 = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='deterministic_test1',
        explanation='Deterministic test 1',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='random',
        n_searches=3,
        cv_strategy='kfold',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    result2 = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='deterministic_test2',
        explanation='Deterministic test 2',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='random',
        n_searches=3,
        cv_strategy='kfold',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    # Same parameters should be selected
    assert result1['best_hyperparameters'] == result2['best_hyperparameters']


def test_tune_hyperparameters_single_combination(sample_tuning_data):
    """Test tuning with single parameter combination."""
    param_grid = {
        'n_estimators': [10]
    }
    
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='single_combo',
        explanation='Single combination test',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='kfold',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    assert result['n_total'] == 1
    assert result['n_successful'] == 1
    assert result['success_rate'] == 1.0


# ========== Error Handling Tests ==========

def test_tune_hyperparameters_invalid_search_strategy(sample_tuning_data):
    """Test that invalid search strategy raises error."""
    param_grid = {'n_estimators': [5, 10]}
    
    with pytest.raises(ValueError, match="Unknown search strategy"):
        tune_hyperparameters(
            input_filename=sample_tuning_data['train_filename'],
            feature_vectors_filename=sample_tuning_data['features_filename'],
            smiles_column='smiles',
            target_column='activity',
            project_manifest_path=sample_tuning_data['manifest_path'],
            output_filename='invalid_strategy',
            explanation='Test',
            model_algorithm='random_forest_classifier',
            param_grid=param_grid,
            search_strategy='invalid_strategy',
            cv_strategy='kfold',
            n_folds=3,
            metric='accuracy',
            random_state=42
        )


# ========== Integration Tests ==========

def test_tune_and_use_best_params(sample_tuning_data):
    """Test that tuned hyperparameters can be loaded and used."""
    from chemlint.infrastructure.resources import _load_resource
    
    param_grid = {
        'n_estimators': [5, 10],
        'max_depth': [3, 5]
    }
    
    # Tune hyperparameters
    result = tune_hyperparameters(
        input_filename=sample_tuning_data['train_filename'],
        feature_vectors_filename=sample_tuning_data['features_filename'],
        smiles_column='smiles',
        target_column='activity',
        project_manifest_path=sample_tuning_data['manifest_path'],
        output_filename='best_params_integration',
        explanation='Integration test',
        model_algorithm='random_forest_classifier',
        param_grid=param_grid,
        search_strategy='grid',
        cv_strategy='kfold',
        n_folds=3,
        metric='accuracy',
        higher_is_better=True,
        random_state=42
    )
    
    # Load the best parameters
    best_params = _load_resource(
        sample_tuning_data['manifest_path'],
        result['output_filename']
    )
    
    # Verify they're valid hyperparameters
    assert isinstance(best_params, dict)
    assert 'n_estimators' in best_params
    assert 'max_depth' in best_params
    assert best_params['n_estimators'] in [5, 10]
    assert best_params['max_depth'] in [3, 5]
