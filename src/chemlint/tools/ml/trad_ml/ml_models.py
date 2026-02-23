
from typing import Dict, Any
from chemlint.tools.ml.trad_ml.singular_models import (
    _train_random_forest_classifier,
    _get_random_forest_classifier_hyperparams,
    _train_gradient_boosting_classifier,
    _get_gradient_boosting_classifier_hyperparams,
    _train_logistic_regression,
    _get_logistic_regression_hyperparams,
    _train_svc,
    _get_svc_hyperparams,
    _train_knn_classifier,
    _get_knn_classifier_hyperparams,
    _train_decision_tree_classifier,
    _get_decision_tree_classifier_hyperparams,
    _train_adaboost_classifier,
    _get_adaboost_classifier_hyperparams,
    _train_extra_trees_classifier,
    _get_extra_trees_classifier_hyperparams,
    _train_naive_bayes,
    _get_naive_bayes_hyperparams,
    _train_lda,
    _get_lda_hyperparams,
    _train_sgd_classifier,
    _get_sgd_classifier_hyperparams,
    _train_random_forest_regressor,
    _get_random_forest_regressor_hyperparams,
    _train_gradient_boosting_regressor,
    _get_gradient_boosting_regressor_hyperparams,
    _train_ridge,
    _get_ridge_hyperparams,
    _train_lasso,
    _get_lasso_hyperparams,
    _train_elastic_net,
    _get_elastic_net_hyperparams,
    _train_svr,
    _get_svr_hyperparams,
    _train_knn_regressor,
    _get_knn_regressor_hyperparams,
    _train_decision_tree_regressor,
    _get_decision_tree_regressor_hyperparams,
    _train_adaboost_regressor,
    _get_adaboost_regressor_hyperparams,
    _train_extra_trees_regressor,
    _get_extra_trees_regressor_hyperparams,
    _train_sgd_regressor,
    _get_sgd_regressor_hyperparams
)

from chemlint.tools.ml.trad_ml.ensembled_models import (
    _train_random_forest_classifier_w_uncertainty,
    _get_random_forest_classifier_w_uncertainty_hyperparams,
    _train_gradient_boosting_classifier_w_uncertainty,
    _get_gradient_boosting_classifier_w_uncertainty_hyperparams,
    _train_logistic_regression_w_uncertainty,
    _get_logistic_regression_w_uncertainty_hyperparams,
    _train_decision_tree_classifier_w_uncertainty,
    _get_decision_tree_classifier_w_uncertainty_hyperparams,
    _train_adaboost_classifier_w_uncertainty,
    _get_adaboost_classifier_w_uncertainty_hyperparams,
    _train_extra_trees_classifier_w_uncertainty,
    _get_extra_trees_classifier_w_uncertainty_hyperparams,
    _train_random_forest_regressor_w_uncertainty,
    _get_random_forest_regressor_w_uncertainty_hyperparams,
    _train_gradient_boosting_regressor_w_uncertainty,
    _get_gradient_boosting_regressor_w_uncertainty_hyperparams,
    _train_decision_tree_regressor_w_uncertainty,
    _get_decision_tree_regressor_w_uncertainty_hyperparams,
    _train_adaboost_regressor_w_uncertainty,
    _get_adaboost_regressor_w_uncertainty_hyperparams,
    _train_extra_trees_regressor_w_uncertainty,
    _get_extra_trees_regressor_w_uncertainty_hyperparams,
)   


# ============================================================================
# Model Registry
# ============================================================================

CLASSIFICATION_MODELS = {
    "random_forest_classifier": {
        "name": "Random Forest Classifier",
        "function": _train_random_forest_classifier,
        "hyperparams_function": _get_random_forest_classifier_hyperparams,
        "description": "Ensemble of decision trees using bagging. Good for most tasks, handles non-linear relationships.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt"
        }
    },
    "gradient_boosting_classifier": {
        "name": "Gradient Boosting Classifier",
        "function": _train_gradient_boosting_classifier,
        "hyperparams_function": _get_gradient_boosting_classifier_hyperparams,
        "description": "Sequential ensemble that corrects previous errors. Often highest accuracy but slower.",
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0
        }
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "function": _train_logistic_regression,
        "hyperparams_function": _get_logistic_regression_hyperparams,
        "description": "Linear classifier. Fast, interpretable, works well with high-dimensional sparse data.",
        "default_params": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000
        }
    },
    "svc": {
        "name": "Support Vector Classifier",
        "function": _train_svc,
        "hyperparams_function": _get_svc_hyperparams,
        "description": "Finds optimal separating hyperplane. Works well for high-dimensional data.",
        "default_params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True
        }
    },
    "knn_classifier": {
        "name": "K-Nearest Neighbors Classifier",
        "function": _train_knn_classifier,
        "hyperparams_function": _get_knn_classifier_hyperparams,
        "description": "Instance-based learning. Simple, no training phase, but slower prediction.",
        "default_params": {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski"
        }
    },
    "decision_tree_classifier": {
        "name": "Decision Tree Classifier",
        "function": _train_decision_tree_classifier,
        "hyperparams_function": _get_decision_tree_classifier_hyperparams,
        "description": "Simple tree-based model. Interpretable but prone to overfitting.",
        "default_params": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        }
    },
    "adaboost_classifier": {
        "name": "AdaBoost Classifier",
        "function": _train_adaboost_classifier,
        "hyperparams_function": _get_adaboost_classifier_hyperparams,
        "description": "Boosting ensemble focusing on misclassified samples. Good for binary classification.",
        "default_params": {
            "n_estimators": 50,
            "learning_rate": 1.0
        }
    },
    "extra_trees_classifier": {
        "name": "Extra Trees Classifier",
        "function": _train_extra_trees_classifier,
        "hyperparams_function": _get_extra_trees_classifier_hyperparams,
        "description": "Similar to Random Forest with random split thresholds. Faster, reduces overfitting.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": None,
            "max_features": "sqrt"
        }
    },
    "naive_bayes": {
        "name": "Gaussian Naive Bayes",
        "function": _train_naive_bayes,
        "hyperparams_function": _get_naive_bayes_hyperparams,
        "description": "Probabilistic classifier with independence assumption. Fast, works with small datasets.",
        "default_params": {}
    },
    "lda_classifier": {
        "name": "Linear Discriminant Analysis",
        "function": _train_lda,
        "hyperparams_function": _get_lda_hyperparams,
        "description": "Linear classifier with dimensionality reduction. Assumes Gaussian distributions.",
        "default_params": {
            "solver": "svd"
        }
    },
    "sgd_classifier": {
        "name": "SGD Classifier",
        "function": _train_sgd_classifier,
        "hyperparams_function": _get_sgd_classifier_hyperparams,
        "description": "Linear classifier with SGD training. Very efficient for large sparse datasets.",
        "default_params": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 1000
        }
    },
    # Ensemble models with uncertainty estimation
    "random_forest_classifier_w_uncertainty": {
        "name": "Random Forest Classifier (w/ Uncertainty)",
        "function": _train_random_forest_classifier_w_uncertainty,
        "hyperparams_function": _get_random_forest_classifier_w_uncertainty_hyperparams,
        "description": "Ensemble of Random Forest classifiers providing uncertainty estimates via prediction variance.",
        "default_params": {
            "ensemble_size": 10,
            "n_estimators": 100,
            "max_depth": None,
            "max_features": "sqrt"
        }
    },
    "gradient_boosting_classifier_w_uncertainty": {
        "name": "Gradient Boosting Classifier (w/ Uncertainty)",
        "function": _train_gradient_boosting_classifier_w_uncertainty,
        "hyperparams_function": _get_gradient_boosting_classifier_w_uncertainty_hyperparams,
        "description": "Ensemble of Gradient Boosting classifiers with uncertainty quantification.",
        "default_params": {
            "ensemble_size": 10,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        }
    },
    "logistic_regression_w_uncertainty": {
        "name": "Logistic Regression (w/ Uncertainty)",
        "function": _train_logistic_regression_w_uncertainty,
        "hyperparams_function": _get_logistic_regression_w_uncertainty_hyperparams,
        "description": "Ensemble of Logistic Regression models providing uncertainty estimates.",
        "default_params": {
            "ensemble_size": 10,
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 1000
        }
    },
    "decision_tree_classifier_w_uncertainty": {
        "name": "Decision Tree Classifier (w/ Uncertainty)",
        "function": _train_decision_tree_classifier_w_uncertainty,
        "hyperparams_function": _get_decision_tree_classifier_w_uncertainty_hyperparams,
        "description": "Ensemble of Decision Trees with uncertainty quantification.",
        "default_params": {
            "ensemble_size": 10,
            "max_depth": None,
            "criterion": "gini"
        }
    },
    "adaboost_classifier_w_uncertainty": {
        "name": "AdaBoost Classifier (w/ Uncertainty)",
        "function": _train_adaboost_classifier_w_uncertainty,
        "hyperparams_function": _get_adaboost_classifier_w_uncertainty_hyperparams,
        "description": "Ensemble of AdaBoost classifiers providing uncertainty estimates.",
        "default_params": {
            "ensemble_size": 10,
            "n_estimators": 50,
            "learning_rate": 1.0
        }
    },
    "extra_trees_classifier_w_uncertainty": {
        "name": "Extra Trees Classifier (w/ Uncertainty)",
        "function": _train_extra_trees_classifier_w_uncertainty,
        "hyperparams_function": _get_extra_trees_classifier_w_uncertainty_hyperparams,
        "description": "Ensemble of Extra Trees classifiers with uncertainty quantification.",
        "default_params": {
            "ensemble_size": 10,
            "n_estimators": 100,
            "max_depth": None,
            "max_features": "sqrt"
        }
    }
}

REGRESSION_MODELS = {
    "random_forest_regressor": {
        "name": "Random Forest Regressor",
        "function": _train_random_forest_regressor,
        "hyperparams_function": _get_random_forest_regressor_hyperparams,
        "description": "Ensemble of decision trees. Handles non-linear relationships, robust to outliers.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt"
        }
    },
    "gradient_boosting_regressor": {
        "name": "Gradient Boosting Regressor",
        "function": _train_gradient_boosting_regressor,
        "hyperparams_function": _get_gradient_boosting_regressor_hyperparams,
        "description": "Sequential tree ensemble. Often best accuracy for tabular data but slower.",
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0
        }
    },
    "ridge": {
        "name": "Ridge Regression",
        "function": _train_ridge,
        "hyperparams_function": _get_ridge_hyperparams,
        "description": "Linear regression with L2 regularization. Fast, stable, handles multicollinearity.",
        "default_params": {
            "alpha": 1.0,
            "solver": "auto"
        }
    },
    "lasso": {
        "name": "Lasso Regression",
        "function": _train_lasso,
        "hyperparams_function": _get_lasso_hyperparams,
        "description": "Linear regression with L1 regularization. Performs automatic feature selection.",
        "default_params": {
            "alpha": 1.0,
            "max_iter": 1000
        }
    },
    "elastic_net": {
        "name": "Elastic Net",
        "function": _train_elastic_net,
        "hyperparams_function": _get_elastic_net_hyperparams,
        "description": "Linear regression with L1 and L2 regularization. Combines Ridge and Lasso benefits.",
        "default_params": {
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 1000
        }
    },
    "svr": {
        "name": "Support Vector Regressor",
        "function": _train_svr,
        "hyperparams_function": _get_svr_hyperparams,
        "description": "Non-linear regression with kernel trick. Good for high-dimensional complex patterns.",
        "default_params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "epsilon": 0.1
        }
    },
    "knn_regressor": {
        "name": "K-Nearest Neighbors Regressor",
        "function": _train_knn_regressor,
        "hyperparams_function": _get_knn_regressor_hyperparams,
        "description": "Predicts based on k nearest neighbors. Non-parametric, simple, slower prediction.",
        "default_params": {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski"
        }
    },
    "decision_tree_regressor": {
        "name": "Decision Tree Regressor",
        "function": _train_decision_tree_regressor,
        "hyperparams_function": _get_decision_tree_regressor_hyperparams,
        "description": "Simple tree-based model. Interpretable but prone to overfitting.",
        "default_params": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "squared_error"
        }
    },
    "adaboost_regressor": {
        "name": "AdaBoost Regressor",
        "function": _train_adaboost_regressor,
        "hyperparams_function": _get_adaboost_regressor_hyperparams,
        "description": "Boosting ensemble focusing on difficult samples. Improves weak learners.",
        "default_params": {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "loss": "linear"
        }
    },
    "extra_trees_regressor": {
        "name": "Extra Trees Regressor",
        "function": _train_extra_trees_regressor,
        "hyperparams_function": _get_extra_trees_regressor_hyperparams,
        "description": "Similar to Random Forest with random splits. Faster, reduces overfitting.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": None,
            "max_features": "sqrt"
        }
    },
    "sgd_regressor": {
        "name": "SGD Regressor",
        "function": _train_sgd_regressor,
        "hyperparams_function": _get_sgd_regressor_hyperparams,
        "description": "Linear regression with SGD training. Very efficient for large sparse datasets.",
        "default_params": {
            "loss": "squared_error",
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 1000
        }
    },
    # Ensemble models with uncertainty estimation
    "random_forest_regressor_w_uncertainty": {
        "name": "Random Forest Regressor (w/ Uncertainty)",
        "function": _train_random_forest_regressor_w_uncertainty,
        "hyperparams_function": _get_random_forest_regressor_w_uncertainty_hyperparams,
        "description": "Ensemble of Random Forest regressors providing uncertainty estimates via prediction variance.",
        "default_params": {
            "ensemble_size": 10,
            "n_estimators": 100,
            "max_depth": None,
            "max_features": "sqrt"
        }
    },
    "gradient_boosting_regressor_w_uncertainty": {
        "name": "Gradient Boosting Regressor (w/ Uncertainty)",
        "function": _train_gradient_boosting_regressor_w_uncertainty,
        "hyperparams_function": _get_gradient_boosting_regressor_w_uncertainty_hyperparams,
        "description": "Ensemble of Gradient Boosting regressors with uncertainty quantification.",
        "default_params": {
            "ensemble_size": 10,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        }
    },
    "decision_tree_regressor_w_uncertainty": {
        "name": "Decision Tree Regressor (w/ Uncertainty)",
        "function": _train_decision_tree_regressor_w_uncertainty,
        "hyperparams_function": _get_decision_tree_regressor_w_uncertainty_hyperparams,
        "description": "Ensemble of Decision Trees with uncertainty quantification.",
        "default_params": {
            "ensemble_size": 10,
            "max_depth": None,
            "criterion": "squared_error"
        }
    },
    "adaboost_regressor_w_uncertainty": {
        "name": "AdaBoost Regressor (w/ Uncertainty)",
        "function": _train_adaboost_regressor_w_uncertainty,
        "hyperparams_function": _get_adaboost_regressor_w_uncertainty_hyperparams,
        "description": "Ensemble of AdaBoost regressors providing uncertainty estimates.",
        "default_params": {
            "ensemble_size": 10,
            "n_estimators": 50,
            "learning_rate": 1.0,
            "loss": "linear"
        }
    },
    "extra_trees_regressor_w_uncertainty": {
        "name": "Extra Trees Regressor (w/ Uncertainty)",
        "function": _train_extra_trees_regressor_w_uncertainty,
        "hyperparams_function": _get_extra_trees_regressor_w_uncertainty_hyperparams,
        "description": "Ensemble of Extra Trees regressors with uncertainty quantification.",
        "default_params": {
            "ensemble_size": 10,
            "n_estimators": 100,
            "max_depth": None,
            "max_features": "sqrt"
        }
    }
}


# Consolidated registry of all models
ALL_MODELS = {**CLASSIFICATION_MODELS, **REGRESSION_MODELS}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get registry of all available models.
    
    Returns:
        Dictionary mapping model keys to their metadata (name, description, default params)
    """
    return {k: {
        "name": v["name"],
        "description": v["description"],
        "default_params": v["default_params"]
    } for k, v in ALL_MODELS.items()}


def get_model_function(model_key: str):
    """
    Get the training function for a specific model.
    
    Args:
        model_key: Key identifying the model (e.g., "random_forest_classifier", "ridge")
    
    Returns:
        Training function for the specified model
    """
    if model_key not in ALL_MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(ALL_MODELS.keys())}")
    return ALL_MODELS[model_key]["function"]


def get_hyperparameter_space(model_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Get the hyperparameter search space for a specific model.
    
    Returns a dictionary describing each hyperparameter's type, range, and other properties
    that can be used to define a hyperparameter search space.
    
    Args:
        model_key: Key identifying the model (e.g., "random_forest_classifier", "ridge")
    
    Returns:
        Dictionary mapping hyperparameter names to their space definitions.
        Each space definition contains:
        - type: "int", "float", "categorical", "int_or_none", "float_or_none", etc.
        - range: [min, max] for numeric types (when applicable)
        - choices: List of valid options for categorical types
        - log_scale: Boolean indicating if log scale should be used for sampling
        - description: Human-readable description of the hyperparameter
    
    Example:
        >>> space = get_hyperparameter_space("random_forest_classifier")
        >>> space["n_estimators"]
        {
            "type": "int",
            "range": [10, 500],
            "log_scale": False,
            "description": "Number of trees in the forest"
        }
    """
    if model_key not in ALL_MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(ALL_MODELS.keys())}")
    
    # Call the hyperparameter function
    hyperparams_func = ALL_MODELS[model_key]["hyperparams_function"]
    return hyperparams_func()


def get_all_hyperparameter_spaces() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Get hyperparameter search spaces for all available models.
    
    Returns:
        Dictionary mapping model keys to their hyperparameter spaces
    
    Example:
        >>> spaces = get_all_hyperparameter_spaces()
        >>> list(spaces.keys())
        ['random_forest_classifier', 'gradient_boosting_classifier', 'logistic_regression', ...]
    """
    # Call each model's hyperparameter function
    spaces = {}
    for model_key, model_info in ALL_MODELS.items():
        hyperparams_func = model_info["hyperparams_function"]
        spaces[model_key] = hyperparams_func()
    
    return spaces
