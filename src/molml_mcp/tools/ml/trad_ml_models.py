"""
Traditional machine learning models from scikit-learn.

This module provides internal training functions for various ML models,
organized by task type (classification/regression).
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDClassifier,
    SGDRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ============================================================================
# Classification Models
# ============================================================================

def _train_random_forest_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Ensemble of decision trees using bagging. Good for most tasks,
    handles non-linear relationships, provides feature importance.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_gradient_boosting_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: int = 42,
    **kwargs
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier.
    
    Builds trees sequentially, each correcting errors of previous.
    Often achieves better accuracy than Random Forest but slower to train.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.
    
    Linear model for binary/multiclass classification. Fast, interpretable,
    works well with high-dimensional sparse data.
    """
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    probability: bool = True,
    random_state: int = 42,
    **kwargs
) -> SVC:
    """
    Train a Support Vector Classifier.
    
    Finds optimal hyperplane to separate classes. Works well for
    high-dimensional data, effective with clear margin of separation.
    """
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_knn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    n_jobs: int = -1,
    **kwargs
) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors classifier.
    
    Instance-based learning, classifies based on majority vote of k nearest neighbors.
    Simple, no training phase, but slower prediction and sensitive to feature scaling.
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_decision_tree_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = "gini",
    random_state: int = 42,
    **kwargs
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.
    
    Simple tree-based model. Interpretable, handles non-linear relationships,
    but prone to overfitting without pruning.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_adaboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    random_state: int = 42,
    **kwargs
) -> AdaBoostClassifier:
    """
    Train an AdaBoost classifier.
    
    Boosting ensemble that focuses on misclassified samples.
    Good for binary classification, less prone to overfitting than single tree.
    """
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_extra_trees_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> ExtraTreesClassifier:
    """
    Train an Extra Trees (Extremely Randomized Trees) classifier.
    
    Similar to Random Forest but uses random thresholds for splits.
    Faster to train, often reduces overfitting further.
    """
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs
) -> GaussianNB:
    """
    Train a Gaussian Naive Bayes classifier.
    
    Probabilistic classifier based on Bayes' theorem with independence assumption.
    Fast, works well with small datasets and high-dimensional data.
    """
    model = GaussianNB(**kwargs)
    model.fit(X_train, y_train)
    return model


def _train_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    solver: str = "svd",
    **kwargs
) -> LinearDiscriminantAnalysis:
    """
    Train a Linear Discriminant Analysis classifier.
    
    Linear classifier with dimensionality reduction. Assumes Gaussian distributions
    with same covariance. Works well when classes are well-separated.
    """
    model = LinearDiscriminantAnalysis(solver=solver, **kwargs)
    model.fit(X_train, y_train)
    return model


def _train_sgd_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    loss: str = "hinge",
    penalty: str = "l2",
    alpha: float = 0.0001,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> SGDClassifier:
    """
    Train a Stochastic Gradient Descent classifier.
    
    Linear classifier with SGD training. Very efficient for large datasets,
    supports online learning. Good for sparse data.
    """
    model = SGDClassifier(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


# ============================================================================
# Regression Models
# ============================================================================

def _train_random_forest_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> RandomForestRegressor:
    """
    Train a Random Forest regressor.
    
    Ensemble of decision trees using bagging. Handles non-linear relationships,
    robust to outliers, provides feature importance.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_gradient_boosting_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: int = 42,
    **kwargs
) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting regressor.
    
    Builds trees sequentially to minimize loss. Often best accuracy
    for tabular data but slower to train than Random Forest.
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    solver: str = "auto",
    **kwargs
) -> Ridge:
    """
    Train a Ridge regression model.
    
    Linear regression with L2 regularization. Prevents overfitting,
    works well with multicollinearity. Fast and stable.
    """
    model = Ridge(alpha=alpha, solver=solver, **kwargs)
    model.fit(X_train, y_train)
    return model


def _train_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    max_iter: int = 1000,
    **kwargs
) -> Lasso:
    """
    Train a Lasso regression model.
    
    Linear regression with L1 regularization. Performs feature selection
    by driving some coefficients to zero. Good for sparse models.
    """
    model = Lasso(alpha=alpha, max_iter=max_iter, **kwargs)
    model.fit(X_train, y_train)
    return model


def _train_elastic_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    **kwargs
) -> ElasticNet:
    """
    Train an Elastic Net regression model.
    
    Linear regression with L1 and L2 regularization. Combines benefits
    of Ridge and Lasso. Good when many correlated features.
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, **kwargs)
    model.fit(X_train, y_train)
    return model


def _train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    epsilon: float = 0.1,
    **kwargs
) -> SVR:
    """
    Train a Support Vector Regressor.
    
    Non-linear regression using kernel trick. Works well for
    high-dimensional data with complex non-linear patterns.
    """
    model = SVR(
        C=C,
        kernel=kernel,
        gamma=gamma,
        epsilon=epsilon,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_knn_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    n_jobs: int = -1,
    **kwargs
) -> KNeighborsRegressor:
    """
    Train a K-Nearest Neighbors regressor.
    
    Predicts based on average of k nearest neighbors. Non-parametric,
    simple, but sensitive to feature scaling and slower prediction.
    """
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_decision_tree_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = "squared_error",
    random_state: int = 42,
    **kwargs
) -> DecisionTreeRegressor:
    """
    Train a Decision Tree regressor.
    
    Simple tree-based model. Interpretable, handles non-linear relationships,
    but prone to overfitting without pruning.
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_adaboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    loss: str = "linear",
    random_state: int = 42,
    **kwargs
) -> AdaBoostRegressor:
    """
    Train an AdaBoost regressor.
    
    Boosting ensemble that focuses on difficult samples.
    Can improve weak learners, less prone to overfitting.
    """
    model = AdaBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss=loss,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_extra_trees_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> ExtraTreesRegressor:
    """
    Train an Extra Trees regressor.
    
    Similar to Random Forest but with random split thresholds.
    Faster to train, often reduces overfitting further.
    """
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def _train_sgd_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    loss: str = "squared_error",
    penalty: str = "l2",
    alpha: float = 0.0001,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> SGDRegressor:
    """
    Train a Stochastic Gradient Descent regressor.
    
    Linear regression with SGD training. Very efficient for large datasets,
    supports online learning. Good for sparse data.
    """
    model = SGDRegressor(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


# ============================================================================
# Model Registry
# ============================================================================

CLASSIFICATION_MODELS = {
    "random_forest": {
        "name": "Random Forest Classifier",
        "function": _train_random_forest_classifier,
        "description": "Ensemble of decision trees using bagging. Good for most tasks, handles non-linear relationships.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt"
        }
    },
    "gradient_boosting": {
        "name": "Gradient Boosting Classifier",
        "function": _train_gradient_boosting_classifier,
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
        "description": "Finds optimal separating hyperplane. Works well for high-dimensional data.",
        "default_params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True
        }
    },
    "knn": {
        "name": "K-Nearest Neighbors Classifier",
        "function": _train_knn_classifier,
        "description": "Instance-based learning. Simple, no training phase, but slower prediction.",
        "default_params": {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski"
        }
    },
    "decision_tree": {
        "name": "Decision Tree Classifier",
        "function": _train_decision_tree_classifier,
        "description": "Simple tree-based model. Interpretable but prone to overfitting.",
        "default_params": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        }
    },
    "adaboost": {
        "name": "AdaBoost Classifier",
        "function": _train_adaboost_classifier,
        "description": "Boosting ensemble focusing on misclassified samples. Good for binary classification.",
        "default_params": {
            "n_estimators": 50,
            "learning_rate": 1.0
        }
    },
    "extra_trees": {
        "name": "Extra Trees Classifier",
        "function": _train_extra_trees_classifier,
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
        "description": "Probabilistic classifier with independence assumption. Fast, works with small datasets.",
        "default_params": {}
    },
    "lda": {
        "name": "Linear Discriminant Analysis",
        "function": _train_lda,
        "description": "Linear classifier with dimensionality reduction. Assumes Gaussian distributions.",
        "default_params": {
            "solver": "svd"
        }
    },
    "sgd": {
        "name": "SGD Classifier",
        "function": _train_sgd_classifier,
        "description": "Linear classifier with SGD training. Very efficient for large sparse datasets.",
        "default_params": {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 1000
        }
    }
}

REGRESSION_MODELS = {
    "random_forest": {
        "name": "Random Forest Regressor",
        "function": _train_random_forest_regressor,
        "description": "Ensemble of decision trees. Handles non-linear relationships, robust to outliers.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt"
        }
    },
    "gradient_boosting": {
        "name": "Gradient Boosting Regressor",
        "function": _train_gradient_boosting_regressor,
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
        "description": "Linear regression with L2 regularization. Fast, stable, handles multicollinearity.",
        "default_params": {
            "alpha": 1.0,
            "solver": "auto"
        }
    },
    "lasso": {
        "name": "Lasso Regression",
        "function": _train_lasso,
        "description": "Linear regression with L1 regularization. Performs automatic feature selection.",
        "default_params": {
            "alpha": 1.0,
            "max_iter": 1000
        }
    },
    "elastic_net": {
        "name": "Elastic Net",
        "function": _train_elastic_net,
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
        "description": "Non-linear regression with kernel trick. Good for high-dimensional complex patterns.",
        "default_params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "epsilon": 0.1
        }
    },
    "knn": {
        "name": "K-Nearest Neighbors Regressor",
        "function": _train_knn_regressor,
        "description": "Predicts based on k nearest neighbors. Non-parametric, simple, slower prediction.",
        "default_params": {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski"
        }
    },
    "decision_tree": {
        "name": "Decision Tree Regressor",
        "function": _train_decision_tree_regressor,
        "description": "Simple tree-based model. Interpretable but prone to overfitting.",
        "default_params": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "squared_error"
        }
    },
    "adaboost": {
        "name": "AdaBoost Regressor",
        "function": _train_adaboost_regressor,
        "description": "Boosting ensemble focusing on difficult samples. Improves weak learners.",
        "default_params": {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "loss": "linear"
        }
    },
    "extra_trees": {
        "name": "Extra Trees Regressor",
        "function": _train_extra_trees_regressor,
        "description": "Similar to Random Forest with random splits. Faster, reduces overfitting.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": None,
            "max_features": "sqrt"
        }
    },
    "sgd": {
        "name": "SGD Regressor",
        "function": _train_sgd_regressor,
        "description": "Linear regression with SGD training. Very efficient for large sparse datasets.",
        "default_params": {
            "loss": "squared_error",
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 1000
        }
    }
}


def get_available_models(task: str = "classification") -> Dict[str, Dict[str, Any]]:
    """
    Get registry of available models for a given task.
    
    Args:
        task: Either "classification" or "regression"
    
    Returns:
        Dictionary mapping model keys to their metadata (name, description, default params)
    """
    if task == "classification":
        return {k: {
            "name": v["name"],
            "description": v["description"],
            "default_params": v["default_params"]
        } for k, v in CLASSIFICATION_MODELS.items()}
    elif task == "regression":
        return {k: {
            "name": v["name"],
            "description": v["description"],
            "default_params": v["default_params"]
        } for k, v in REGRESSION_MODELS.items()}
    else:
        raise ValueError(f"Unknown task '{task}'. Must be 'classification' or 'regression'")


def get_model_function(model_key: str, task: str = "classification"):
    """
    Get the training function for a specific model.
    
    Args:
        model_key: Key identifying the model (e.g., "random_forest", "ridge")
        task: Either "classification" or "regression"
    
    Returns:
        Training function for the specified model
    """
    if task == "classification":
        if model_key not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model '{model_key}'. Available: {list(CLASSIFICATION_MODELS.keys())}")
        return CLASSIFICATION_MODELS[model_key]["function"]
    elif task == "regression":
        if model_key not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model '{model_key}'. Available: {list(REGRESSION_MODELS.keys())}")
        return REGRESSION_MODELS[model_key]["function"]
    else:
        raise ValueError(f"Unknown task '{task}'. Must be 'classification' or 'regression'")
