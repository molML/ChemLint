
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

# random_forest_classifier_w_uncertainty

def _get_random_forest_classifier_w_uncertainty_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter space for Random Forest Classifier w Uncertainty."""
    from molml_mcp.tools.ml.trad_ml.singular_models import _get_random_forest_classifier_hyperparams
    base_hypers = _get_random_forest_classifier_hyperparams()

    return base_hypers | {'ensemble_size': {"type": "int", "range": [1, 1000], "log_scale": False, "description": "Number of models in the ensemble"}}

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



# Random_Forest
# Extra Trees
# Gradient Boosting
# AdaBoost
# Logistic Regression
# Ridge/Lasso/ElasticNet