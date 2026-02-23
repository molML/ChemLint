"""
Traditional machine learning models from scikit-learn.

This module provides internal training functions for various ML models,
organized by task type (classification/regression).
"""

import numpy as np
from typing import Dict, Any, Optional, List
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

def _get_random_forest_classifier_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Random Forest Classifier."""
    return {
        "n_estimators": {"type": "int", "range": [10, 500], "log_scale": False, "description": "Number of trees in the forest"},
        "max_depth": {"type": "int_or_none", "range": [3, 50], "log_scale": False, "description": "Maximum depth of trees (None for unlimited)"},
        "min_samples_split": {"type": "int", "range": [2, 20], "log_scale": False, "description": "Minimum samples required to split a node"},
        "min_samples_leaf": {"type": "int", "range": [1, 20], "log_scale": False, "description": "Minimum samples required at a leaf node"},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "description": "Number of features to consider for splits"}
    }


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


def _get_gradient_boosting_classifier_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Gradient Boosting Classifier."""
    return {
        "n_estimators": {"type": "int", "range": [50, 500], "log_scale": False, "description": "Number of boosting stages"},
        "learning_rate": {"type": "float", "range": [0.001, 0.3], "log_scale": True, "description": "Shrinks the contribution of each tree"},
        "max_depth": {"type": "int", "range": [3, 10], "log_scale": False, "description": "Maximum depth of trees"},
        "min_samples_split": {"type": "int", "range": [2, 20], "log_scale": False, "description": "Minimum samples required to split a node"},
        "min_samples_leaf": {"type": "int", "range": [1, 20], "log_scale": False, "description": "Minimum samples required at a leaf node"},
        "subsample": {"type": "float", "range": [0.5, 1.0], "log_scale": False, "description": "Fraction of samples for fitting trees"}
    }


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


def _get_logistic_regression_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Logistic Regression.
    
    Note: Penalty and solver combinations have constraints:
    - lbfgs: supports only l2, none
    - liblinear: supports l1, l2
    - saga: supports l1, l2, elasticnet, none
    """
    return {
        "penalty": {"type": "categorical", "choices": ["l2", "none"], "description": "Regularization type (use saga solver for l1/elasticnet)"},
        "C": {"type": "float", "range": [0.001, 100], "log_scale": True, "description": "Inverse of regularization strength"},
        "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear", "saga"], "description": "Optimization algorithm"},
        "max_iter": {"type": "int", "range": [100, 5000], "log_scale": False, "description": "Maximum iterations for convergence"}
    }


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


def _get_svc_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Support Vector Classifier."""
    return {
        "C": {"type": "float", "range": [0.001, 100], "log_scale": True, "description": "Regularization parameter"},
        "kernel": {"type": "categorical", "choices": ["linear", "poly", "rbf", "sigmoid"], "description": "Kernel type"},
        "gamma": {"type": "categorical_or_float", "choices": ["scale", "auto"], "range": [0.0001, 1.0], "log_scale": True, "description": "Kernel coefficient"},
        "degree": {"type": "int", "range": [2, 5], "log_scale": False, "description": "Degree for polynomial kernel"}
    }


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


def _get_knn_classifier_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for K-Nearest Neighbors Classifier."""
    return {
        "n_neighbors": {"type": "int", "range": [1, 50], "log_scale": False, "description": "Number of neighbors"},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"], "description": "Weight function for prediction"},
        "metric": {"type": "categorical", "choices": ["minkowski", "euclidean", "manhattan", "chebyshev"], "description": "Distance metric"},
        "p": {"type": "int", "range": [1, 3], "log_scale": False, "description": "Power parameter for Minkowski metric"}
    }


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


def _get_decision_tree_classifier_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Decision Tree Classifier."""
    return {
        "max_depth": {"type": "int_or_none", "range": [3, 30], "log_scale": False, "description": "Maximum depth of tree (None for unlimited)"},
        "min_samples_split": {"type": "int", "range": [2, 20], "log_scale": False, "description": "Minimum samples required to split a node"},
        "min_samples_leaf": {"type": "int", "range": [1, 20], "log_scale": False, "description": "Minimum samples required at a leaf node"},
        "criterion": {"type": "categorical", "choices": ["gini", "entropy"], "description": "Split quality measure"}
    }


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


def _get_adaboost_classifier_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for AdaBoost Classifier."""
    return {
        "n_estimators": {"type": "int", "range": [10, 200], "log_scale": False, "description": "Maximum number of estimators"},
        "learning_rate": {"type": "float", "range": [0.01, 2.0], "log_scale": True, "description": "Weight applied to each classifier"}
    }


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


def _get_extra_trees_classifier_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Extra Trees Classifier."""
    return {
        "n_estimators": {"type": "int", "range": [10, 500], "log_scale": False, "description": "Number of trees in the forest"},
        "max_depth": {"type": "int_or_none", "range": [3, 50], "log_scale": False, "description": "Maximum depth of trees (None for unlimited)"},
        "min_samples_split": {"type": "int", "range": [2, 20], "log_scale": False, "description": "Minimum samples required to split a node"},
        "min_samples_leaf": {"type": "int", "range": [1, 20], "log_scale": False, "description": "Minimum samples required at a leaf node"},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "description": "Number of features to consider for splits"}
    }


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


def _get_naive_bayes_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Gaussian Naive Bayes."""
    return {
        "var_smoothing": {"type": "float", "range": [1e-10, 1e-5], "log_scale": True, "description": "Portion of largest variance added to variances for stability"}
    }


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


def _get_lda_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Linear Discriminant Analysis.
    
    Note: Shrinkage only supported with 'lsqr' and 'eigen' solvers, not 'svd'.
    """
    return {
        "solver": {"type": "categorical", "choices": ["svd", "lsqr", "eigen"], "description": "Solver to use"},
        "shrinkage": {"type": "float_or_none", "range": [0.0, 1.0], "log_scale": False, "description": "Shrinkage parameter (only for lsqr/eigen solver, None for svd)"}
    }


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


def _get_sgd_classifier_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for SGD Classifier."""
    return {
        "loss": {"type": "categorical", "choices": ["hinge", "log_loss", "modified_huber", "perceptron"], "description": "Loss function"},
        "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"], "description": "Regularization type"},
        "alpha": {"type": "float", "range": [1e-6, 0.01], "log_scale": True, "description": "Regularization term"},
        "max_iter": {"type": "int", "range": [100, 5000], "log_scale": False, "description": "Maximum iterations"}
    }


# ============================================================================
# Regression Models
# ============================================================================

def _get_random_forest_regressor_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Random Forest Regressor."""
    return {
        "n_estimators": {"type": "int", "range": [10, 500], "log_scale": False, "description": "Number of trees in the forest"},
        "max_depth": {"type": "int_or_none", "range": [3, 50], "log_scale": False, "description": "Maximum depth of trees (None for unlimited)"},
        "min_samples_split": {"type": "int", "range": [2, 20], "log_scale": False, "description": "Minimum samples required to split a node"},
        "min_samples_leaf": {"type": "int", "range": [1, 20], "log_scale": False, "description": "Minimum samples required at a leaf node"},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "description": "Number of features to consider for splits"}
    }


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


def _get_gradient_boosting_regressor_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Gradient Boosting Regressor."""
    return {
        "n_estimators": {"type": "int", "range": [50, 500], "log_scale": False, "description": "Number of boosting stages"},
        "learning_rate": {"type": "float", "range": [0.001, 0.3], "log_scale": True, "description": "Shrinks the contribution of each tree"},
        "max_depth": {"type": "int", "range": [3, 10], "log_scale": False, "description": "Maximum depth of trees"},
        "min_samples_split": {"type": "int", "range": [2, 20], "log_scale": False, "description": "Minimum samples required to split a node"},
        "min_samples_leaf": {"type": "int", "range": [1, 20], "log_scale": False, "description": "Minimum samples required at a leaf node"},
        "subsample": {"type": "float", "range": [0.5, 1.0], "log_scale": False, "description": "Fraction of samples for fitting trees"}
    }


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


def _get_ridge_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Ridge Regression."""
    return {
        "alpha": {"type": "float", "range": [0.001, 100], "log_scale": True, "description": "Regularization strength"},
        "solver": {"type": "categorical", "choices": ["auto", "svd", "cholesky", "lsqr", "saga"], "description": "Solver to use"}
    }


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


def _get_lasso_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Lasso Regression."""
    return {
        "alpha": {"type": "float", "range": [0.001, 10], "log_scale": True, "description": "Regularization strength"},
        "max_iter": {"type": "int", "range": [100, 5000], "log_scale": False, "description": "Maximum iterations for convergence"}
    }


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


def _get_elastic_net_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Elastic Net."""
    return {
        "alpha": {"type": "float", "range": [0.001, 10], "log_scale": True, "description": "Regularization strength"},
        "l1_ratio": {"type": "float", "range": [0.0, 1.0], "log_scale": False, "description": "L1 penalty mixing parameter"},
        "max_iter": {"type": "int", "range": [100, 5000], "log_scale": False, "description": "Maximum iterations for convergence"}
    }


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


def _get_svr_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Support Vector Regressor."""
    return {
        "C": {"type": "float", "range": [0.001, 100], "log_scale": True, "description": "Regularization parameter"},
        "kernel": {"type": "categorical", "choices": ["linear", "poly", "rbf", "sigmoid"], "description": "Kernel type"},
        "gamma": {"type": "categorical_or_float", "choices": ["scale", "auto"], "range": [0.0001, 1.0], "log_scale": True, "description": "Kernel coefficient"},
        "epsilon": {"type": "float", "range": [0.01, 1.0], "log_scale": True, "description": "Epsilon in epsilon-SVR model"}
    }


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


def _get_knn_regressor_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for K-Nearest Neighbors Regressor."""
    return {
        "n_neighbors": {"type": "int", "range": [1, 50], "log_scale": False, "description": "Number of neighbors"},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"], "description": "Weight function for prediction"},
        "metric": {"type": "categorical", "choices": ["minkowski", "euclidean", "manhattan", "chebyshev"], "description": "Distance metric"},
        "p": {"type": "int", "range": [1, 3], "log_scale": False, "description": "Power parameter for Minkowski metric"}
    }


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


def _get_decision_tree_regressor_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Decision Tree Regressor."""
    return {
        "max_depth": {"type": "int_or_none", "range": [3, 30], "log_scale": False, "description": "Maximum depth of tree (None for unlimited)"},
        "min_samples_split": {"type": "int", "range": [2, 20], "log_scale": False, "description": "Minimum samples required to split a node"},
        "min_samples_leaf": {"type": "int", "range": [1, 20], "log_scale": False, "description": "Minimum samples required at a leaf node"},
        "criterion": {"type": "categorical", "choices": ["squared_error", "friedman_mse", "absolute_error"], "description": "Split quality measure"}
    }


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


def _get_adaboost_regressor_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for AdaBoost Regressor."""
    return {
        "n_estimators": {"type": "int", "range": [10, 200], "log_scale": False, "description": "Maximum number of estimators"},
        "learning_rate": {"type": "float", "range": [0.01, 2.0], "log_scale": True, "description": "Weight applied to each estimator"},
        "loss": {"type": "categorical", "choices": ["linear", "square", "exponential"], "description": "Loss function"}
    }


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


def _get_extra_trees_regressor_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Extra Trees Regressor."""
    return {
        "n_estimators": {"type": "int", "range": [10, 500], "log_scale": False, "description": "Number of trees in the forest"},
        "max_depth": {"type": "int_or_none", "range": [3, 50], "log_scale": False, "description": "Maximum depth of trees (None for unlimited)"},
        "min_samples_split": {"type": "int", "range": [2, 20], "log_scale": False, "description": "Minimum samples required to split a node"},
        "min_samples_leaf": {"type": "int", "range": [1, 20], "log_scale": False, "description": "Minimum samples required at a leaf node"},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "description": "Number of features to consider for splits"}
    }


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


def _get_sgd_regressor_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for SGD Regressor."""
    return {
        "loss": {"type": "categorical", "choices": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"], "description": "Loss function"},
        "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"], "description": "Regularization type"},
        "alpha": {"type": "float", "range": [1e-6, 0.01], "log_scale": True, "description": "Regularization term"},
        "max_iter": {"type": "int", "range": [100, 5000], "log_scale": False, "description": "Maximum iterations"}
    }


# ============================================================================
# Public API Functions
# ============================================================================

def get_available_models() -> Dict[str, Any]:
    """
    Get dictionary of all available ML models.
    
    Returns mapping of model_key -> model_info dict with:
    - name: Display name
    - description: Brief description
    - function: Training function
    - default_params: Default hyperparameters
    """
    return {
        # Classification models
        "random_forest_classifier": {
            "name": "Random Forest Classifier",
            "description": "Ensemble of decision trees for classification",
            "function": _train_random_forest_classifier,
            "default_params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
        },
        "decision_tree_classifier": {
            "name": "Decision Tree Classifier",
            "description": "Single decision tree for classification",
            "function": _train_decision_tree_classifier,
            "default_params": {"max_depth": None, "min_samples_split": 2, "criterion": "gini"}
        },
        "logistic_regression": {
            "name": "Logistic Regression",
            "description": "Linear model for binary/multiclass classification",
            "function": _train_logistic_regression,
            "default_params": {"C": 1.0, "penalty": "l2", "max_iter": 1000}
        },
        "gradient_boosting_classifier": {
            "name": "Gradient Boosting Classifier",
            "description": "Boosted ensemble of weak learners",
            "function": _train_gradient_boosting_classifier,
            "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
        },
        "svc": {
            "name": "Support Vector Classifier",
            "description": "SVM for classification with kernel methods",
            "function": _train_svc,
            "default_params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
        },
        "knn_classifier": {
            "name": "K-Nearest Neighbors Classifier",
            "description": "Instance-based classification",
            "function": _train_knn_classifier,
            "default_params": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"}
        },
        # Regression models
        "random_forest_regressor": {
            "name": "Random Forest Regressor",
            "description": "Ensemble of decision trees for regression",
            "function": _train_random_forest_regressor,
            "default_params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
        },
        "decision_tree_regressor": {
            "name": "Decision Tree Regressor",
            "description": "Single decision tree for regression",
            "function": _train_decision_tree_regressor,
            "default_params": {"max_depth": None, "min_samples_split": 2, "criterion": "squared_error"}
        },
        "ridge": {
            "name": "Ridge Regression",
            "description": "Linear regression with L2 regularization",
            "function": _train_ridge,
            "default_params": {"alpha": 1.0}
        },
        "lasso": {
            "name": "Lasso Regression",
            "description": "Linear regression with L1 regularization",
            "function": _train_lasso,
            "default_params": {"alpha": 1.0, "max_iter": 1000}
        },
        "svr": {
            "name": "Support Vector Regressor",
            "description": "SVM for regression with kernel methods",
            "function": _train_svr,
            "default_params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
        },
    }


def get_model_function(model_key: str):
    """Get training function for a specific model."""
    models = get_available_models()
    if model_key not in models:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(models.keys())}")
    return models[model_key]["function"]


def get_hyperparameter_space(model_key: str) -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space definition for a model."""
    hyperparam_getters = {
        "random_forest_classifier": _get_random_forest_classifier_hyperparams,
        "decision_tree_classifier": _get_decision_tree_classifier_hyperparams,
        "logistic_regression": _get_logistic_regression_hyperparams,
        "gradient_boosting_classifier": _get_gradient_boosting_classifier_hyperparams,
        "svc": _get_svc_hyperparams,
        "knn_classifier": _get_knn_classifier_hyperparams,
        "random_forest_regressor": _get_random_forest_regressor_hyperparams,
        "decision_tree_regressor": _get_decision_tree_regressor_hyperparams,
        "ridge": _get_ridge_hyperparams,
        "lasso": _get_lasso_hyperparams,
        "svr": _get_svr_hyperparams,
    }
    
    if model_key not in hyperparam_getters:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(hyperparam_getters.keys())}")
    
    return hyperparam_getters[model_key]()


def get_all_hyperparameter_spaces() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Get all hyperparameter spaces for all models."""
    models = get_available_models()
    return {key: get_hyperparameter_space(key) for key in models.keys()}
