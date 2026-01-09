
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator, clone
import numpy as np

class BayesianEnsemble(BaseEstimator):
    """
    Ensemble wrapper for uncertainty estimation.
    
    Wraps any sklearn estimator and trains n independent models
    with bootstrap sampling. Provides uncertainty estimates while
    maintaining sklearn API compatibility.
    """

    _ensemble_params = {'ensemble_size'}

    
    def __init__(self, base_estimator, ensemble_size=10, **kwargs):
        self.base_estimator = base_estimator
        self.ensemble_size = ensemble_size
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_scikit_base_params(self):
        """Extract only base estimator params"""
        all_params = self.get_params(deep=False)
        
        # Remove ensemble-specific and base_estimator itself
        base_params = {
            k: v for k, v in all_params.items() 
            if k not in self._ensemble_params and k != 'base_estimator'
        }
        
        return base_params
        
    def fit(self, X, y):
        base_params = self._get_scikit_base_params()
        self.estimators_ = []
        
        # Handle random_state: generate unique seeds for each model if provided
        random_states = None
        if 'random_state' in base_params and base_params['random_state'] is not None:
            rng = np.random.RandomState(base_params['random_state'])
            random_states = rng.randint(0, 100000, size=self.ensemble_size)
        
        for i in range(self.ensemble_size):

            if random_states is not None:
                hypers = base_params | {'random_state': random_states[i]}
            else:
                hypers = base_params.copy()

            estimator = self.base_estimator(**hypers)
            X_sample, y_sample = X, y
                
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)
            
        return self
    
    def predict(self, X):
        """Standard sklearn predict - returns mean"""
        predictions = self._get_predictions(X)
        return predictions.mean(axis=0)
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty estimates.
        
        Returns:
            tuple: (mean, std, predictions)
                - mean: Average predictions across all models
                - std: Standard deviation (uncertainty)
                - predictions: Full array of shape (n_models, n_samples)
        """
        predictions = self._get_predictions(X)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std, predictions
    
    def _get_predictions(self, X):
        """Get predictions from all estimators"""
        return np.array([est.predict(X) for est in self.estimators_])
    
    # Optional: Add predict_proba support for classifiers
    def predict_proba(self, X):
        """Average probability predictions (classifiers only)"""
        if not hasattr(self.estimators_[0], 'predict_proba'):
            raise AttributeError("Base estimator doesn't support predict_proba")
        
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return probas.mean(axis=0)
    
    def predict_proba_with_uncertainty(self, X):
        """
        Probability predictions with uncertainty.
        
        Returns:
            tuple: (mean, std, probas)
                - mean: Average probabilities across all models
                - std: Standard deviation (uncertainty) of probabilities
                - probas: Full array of shape (n_models, n_samples, n_classes)
        """
        if not hasattr(self.estimators_[0], 'predict_proba'):
            raise AttributeError("Base estimator doesn't support predict_proba")
            
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        mean = probas.mean(axis=0)
        std = probas.std(axis=0)
        
        return mean, std, probas

    def __len__(self):
        return len(self.estimators_)

    def __repr__(self):
        random_state = getattr(self, 'random_state', None)
        return f"BayesianEnsemble(base_estimator={self.base_estimator}, n_estimators={self.n_estimators}, random_state={random_state})"



###### Models that support ensembles ######

def _train_random_forest_classifier_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1
) -> BayesianEnsemble:
    """
    Train a Random Forest classifier.
    
    Ensemble of decision trees using bagging. Good for most tasks,
    handles non-linear relationships, provides feature importance.
    """
    from sklearn.ensemble import RandomForestClassifier
    model = BayesianEnsemble(
        base_estimator=RandomForestClassifier,
        ensemble_size=ensemble_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(X_train, y_train)
    return model

def _get_random_forest_classifier_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Random Forest Classifier w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_random_forest_classifier_hyperparams
    base_hypers = _get_random_forest_classifier_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}


def _train_gradient_boosting_classifier_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: int = 42
) -> BayesianEnsemble:
    """Train a Gradient Boosting classifier with uncertainty estimation."""
    from sklearn.ensemble import GradientBoostingClassifier
    model = BayesianEnsemble(
        base_estimator=GradientBoostingClassifier,
        ensemble_size=ensemble_size,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def _get_gradient_boosting_classifier_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Gradient Boosting Classifier w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_gradient_boosting_classifier_hyperparams
    base_hypers = _get_gradient_boosting_classifier_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_logistic_regression_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42
) -> BayesianEnsemble:
    """Train a Logistic Regression classifier with uncertainty estimation."""
    from sklearn.linear_model import LogisticRegression
    model = BayesianEnsemble(
        base_estimator=LogisticRegression,
        ensemble_size=ensemble_size,
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def _get_logistic_regression_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Logistic Regression w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_logistic_regression_hyperparams
    base_hypers = _get_logistic_regression_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_decision_tree_classifier_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = "gini",
    random_state: int = 42
) -> BayesianEnsemble:
    """Train a Decision Tree classifier with uncertainty estimation."""
    from sklearn.tree import DecisionTreeClassifier
    model = BayesianEnsemble(
        base_estimator=DecisionTreeClassifier,
        ensemble_size=ensemble_size,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def _get_decision_tree_classifier_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Decision Tree Classifier w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_decision_tree_classifier_hyperparams
    base_hypers = _get_decision_tree_classifier_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_adaboost_classifier_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    random_state: int = 42
) -> BayesianEnsemble:
    """Train an AdaBoost classifier with uncertainty estimation."""
    from sklearn.ensemble import AdaBoostClassifier
    model = BayesianEnsemble(
        base_estimator=AdaBoostClassifier,
        ensemble_size=ensemble_size,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def _get_adaboost_classifier_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for AdaBoost Classifier w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_adaboost_classifier_hyperparams
    base_hypers = _get_adaboost_classifier_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_extra_trees_classifier_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1
) -> BayesianEnsemble:
    """Train an Extra Trees classifier with uncertainty estimation."""
    from sklearn.ensemble import ExtraTreesClassifier
    model = BayesianEnsemble(
        base_estimator=ExtraTreesClassifier,
        ensemble_size=ensemble_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model

def _get_extra_trees_classifier_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Extra Trees Classifier w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_extra_trees_classifier_hyperparams
    base_hypers = _get_extra_trees_classifier_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_random_forest_regressor_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1
) -> BayesianEnsemble:
    """Train a Random Forest regressor with uncertainty estimation."""
    from sklearn.ensemble import RandomForestRegressor
    model = BayesianEnsemble(
        base_estimator=RandomForestRegressor,
        ensemble_size=ensemble_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model

def _get_random_forest_regressor_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Random Forest Regressor w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_random_forest_regressor_hyperparams
    base_hypers = _get_random_forest_regressor_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_gradient_boosting_regressor_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: int = 42
) -> BayesianEnsemble:
    """Train a Gradient Boosting regressor with uncertainty estimation."""
    from sklearn.ensemble import GradientBoostingRegressor
    model = BayesianEnsemble(
        base_estimator=GradientBoostingRegressor,
        ensemble_size=ensemble_size,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def _get_gradient_boosting_regressor_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Gradient Boosting Regressor w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_gradient_boosting_regressor_hyperparams
    base_hypers = _get_gradient_boosting_regressor_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_decision_tree_regressor_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = "squared_error",
    random_state: int = 42
) -> BayesianEnsemble:
    """Train a Decision Tree regressor with uncertainty estimation."""
    from sklearn.tree import DecisionTreeRegressor
    model = BayesianEnsemble(
        base_estimator=DecisionTreeRegressor,
        ensemble_size=ensemble_size,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def _get_decision_tree_regressor_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Decision Tree Regressor w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_decision_tree_regressor_hyperparams
    base_hypers = _get_decision_tree_regressor_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_adaboost_regressor_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    loss: str = "linear",
    random_state: int = 42
) -> BayesianEnsemble:
    """Train an AdaBoost regressor with uncertainty estimation."""
    from sklearn.ensemble import AdaBoostRegressor
    model = BayesianEnsemble(
        base_estimator=AdaBoostRegressor,
        ensemble_size=ensemble_size,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss=loss,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def _get_adaboost_regressor_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for AdaBoost Regressor w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_adaboost_regressor_hyperparams
    base_hypers = _get_adaboost_regressor_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

def _train_extra_trees_regressor_w_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble_size: int = 10,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1
) -> BayesianEnsemble:
    """Train an Extra Trees regressor with uncertainty estimation."""
    from sklearn.ensemble import ExtraTreesRegressor
    model = BayesianEnsemble(
        base_estimator=ExtraTreesRegressor,
        ensemble_size=ensemble_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model

def _get_extra_trees_regressor_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Extra Trees Regressor w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_extra_trees_regressor_hyperparams
    base_hypers = _get_extra_trees_regressor_hyperparams()
    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}
