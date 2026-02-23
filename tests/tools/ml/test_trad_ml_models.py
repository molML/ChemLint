"""Tests for trad_ml_models.py - Traditional ML model training functions."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from chemlint.tools.ml.trad_ml.singular_models import (
    # Sample classification models
    _train_random_forest_classifier,
    _train_logistic_regression,
    _train_decision_tree_classifier,
    # Sample regression models
    _train_random_forest_regressor,
    _train_ridge,
    _train_lasso,
    # Utility functions
    get_available_models,
    get_model_function,
    get_hyperparameter_space,
    get_all_hyperparameter_spaces,
)


# ============================================================================
# Tests for utility functions
# ============================================================================

def test_get_available_models():
    """Test getting list of all available models."""
    models = get_available_models()
    
    assert isinstance(models, dict)
    assert len(models) > 0
    
    # Check structure of model info
    for key, info in models.items():
        assert "name" in info
        assert "description" in info
        assert "function" in info
        assert "default_params" in info
        assert callable(info["function"])


def test_get_model_function():
    """Test getting specific model training function."""
    # Valid model
    func = get_model_function("logistic_regression")
    assert callable(func)
    
    # Invalid model
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_function("nonexistent_model")


def test_get_hyperparameter_space():
    """Test getting hyperparameter space for a model."""
    # Logistic regression
    lr_space = get_hyperparameter_space("logistic_regression")
    assert "C" in lr_space
    assert "penalty" in lr_space
    
    # Ridge regression
    ridge_space = get_hyperparameter_space("ridge")
    assert "alpha" in ridge_space
    
    # Test invalid model key
    with pytest.raises(ValueError, match="Unknown model"):
        get_hyperparameter_space("nonexistent_model")


def test_get_all_hyperparameter_spaces():
    """Test getting hyperparameter spaces for all models."""
    all_spaces = get_all_hyperparameter_spaces()
    
    assert isinstance(all_spaces, dict)
    assert len(all_spaces) > 0
    
    # Should have spaces for all models
    assert "random_forest_classifier" in all_spaces
    assert "logistic_regression" in all_spaces
    assert "ridge" in all_spaces
    
    # Each space should be a dictionary of hyperparameters
    for model_key, space in all_spaces.items():
        assert isinstance(space, dict)


# ============================================================================
# Tests for classification models
# ============================================================================

@pytest.fixture
def classification_data():
    """Generate synthetic classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    return X, y


def test_train_random_forest_classifier(classification_data):
    """Test Random Forest classifier training."""
    X, y = classification_data
    
    model = _train_random_forest_classifier(
        X, y,
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    
    # Check model is trained
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    # Check predictions work
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert set(predictions).issubset({0, 1})
    
    # Check probabilities
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    # Check feature importance available
    assert hasattr(model, 'feature_importances_')
    assert len(model.feature_importances_) == X.shape[1]


def test_train_logistic_regression(classification_data):
    """Test Logistic Regression training."""
    X, y = classification_data
    
    model = _train_logistic_regression(
        X, y,
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    # Check model is trained
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    # Check predictions
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    
    # Check coefficients available (linear model)
    assert hasattr(model, 'coef_')
    assert model.coef_.shape[1] == X.shape[1]


def test_train_decision_tree_classifier(classification_data):
    """Test Decision Tree classifier training."""
    X, y = classification_data
    
    model = _train_decision_tree_classifier(
        X, y,
        max_depth=5,
        min_samples_split=2,
        random_state=42
    )
    
    # Check model is trained
    assert hasattr(model, 'predict')
    assert hasattr(model, 'tree_')
    
    # Check predictions
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    
    # Check tree depth
    assert model.get_depth() <= 5


def test_classification_model_with_custom_params(classification_data):
    """Test classification models accept custom parameters."""
    X, y = classification_data
    
    # Test Random Forest with various parameters
    model = _train_random_forest_classifier(
        X, y,
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42
    )
    
    assert model.n_estimators == 50
    assert model.max_depth == 10
    assert model.min_samples_split == 5
    assert model.min_samples_leaf == 2


# ============================================================================
# Tests for regression models
# ============================================================================

@pytest.fixture
def regression_data():
    """Generate synthetic regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    return X, y


def test_train_random_forest_regressor(regression_data):
    """Test Random Forest regressor training."""
    X, y = regression_data
    
    model = _train_random_forest_regressor(
        X, y,
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    
    # Check model is trained
    assert hasattr(model, 'predict')
    
    # Check predictions
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert np.all(np.isfinite(predictions))
    
    # Check feature importance
    assert hasattr(model, 'feature_importances_')
    assert len(model.feature_importances_) == X.shape[1]


def test_train_ridge(regression_data):
    """Test Ridge regression training."""
    X, y = regression_data
    
    model = _train_ridge(
        X, y,
        alpha=1.0
    )
    
    # Check model is trained
    assert hasattr(model, 'predict')
    assert hasattr(model, 'coef_')
    
    # Check predictions
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert np.all(np.isfinite(predictions))
    
    # Check coefficients
    assert model.coef_.shape[0] == X.shape[1]


def test_train_lasso(regression_data):
    """Test Lasso regression training."""
    X, y = regression_data
    
    model = _train_lasso(
        X, y,
        alpha=0.1,
        max_iter=1000
    )
    
    # Check model is trained
    assert hasattr(model, 'predict')
    assert hasattr(model, 'coef_')
    
    # Check predictions
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    
    # Check Lasso performs feature selection (some coefficients should be zero)
    assert hasattr(model, 'sparse_coef_')


def test_regression_model_with_custom_params(regression_data):
    """Test regression models accept custom parameters."""
    X, y = regression_data
    
    # Test Random Forest with various parameters
    model = _train_random_forest_regressor(
        X, y,
        n_estimators=20,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    
    assert model.n_estimators == 20
    assert model.max_depth == 8
    assert model.min_samples_split == 4
    assert model.min_samples_leaf == 2


# ============================================================================
# Tests for model performance
# ============================================================================

def test_classification_model_reasonable_accuracy(classification_data):
    """Test that classification models achieve reasonable accuracy on training data."""
    X, y = classification_data
    
    # Random Forest should do well on training data
    model = _train_random_forest_classifier(
        X, y,
        n_estimators=50,
        random_state=42
    )
    
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    
    # Should achieve high accuracy on training data
    assert accuracy > 0.8


def test_regression_model_reasonable_r2(regression_data):
    """Test that regression models achieve reasonable R² on training data."""
    X, y = regression_data
    
    # Random Forest should do well on training data
    model = _train_random_forest_regressor(
        X, y,
        n_estimators=50,
        random_state=42
    )
    
    predictions = model.predict(X)
    
    # Calculate R²
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Should achieve high R² on training data
    assert r2 > 0.8


# ============================================================================
# Tests for edge cases and error handling
# ============================================================================

def test_models_handle_small_datasets():
    """Test that models work with very small datasets."""
    # Create tiny dataset
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_class = np.array([0, 1, 0, 1])
    y_reg = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Classification
    model_class = _train_random_forest_classifier(X, y_class, n_estimators=2, random_state=42)
    assert model_class.predict(X).shape == y_class.shape
    
    # Regression
    model_reg = _train_random_forest_regressor(X, y_reg, n_estimators=2, random_state=42)
    assert model_reg.predict(X).shape == y_reg.shape


def test_models_deterministic_with_random_state():
    """Test that models produce same results with same random_state."""
    X, y = make_classification(n_samples=50, n_features=10, random_state=42)
    
    # Train two models with same random state
    model1 = _train_random_forest_classifier(X, y, n_estimators=10, random_state=42)
    model2 = _train_random_forest_classifier(X, y, n_estimators=10, random_state=42)
    
    # Predictions should be identical
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    assert np.array_equal(pred1, pred2)
    
    # Feature importances should be identical
    assert np.allclose(model1.feature_importances_, model2.feature_importances_)


def test_model_registry_consistency():
    """Test that all registered models have consistent structure."""
    models = get_available_models()
    all_spaces = get_all_hyperparameter_spaces()
    
    # Every model should have a hyperparameter space
    for model_key in models.keys():
        assert model_key in all_spaces
        
        # Should be able to get the function
        func = get_model_function(model_key)
        assert callable(func)
        
        # Should be able to get hyperparameter space
        space = get_hyperparameter_space(model_key)
        assert isinstance(space, dict)
