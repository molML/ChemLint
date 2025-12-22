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
