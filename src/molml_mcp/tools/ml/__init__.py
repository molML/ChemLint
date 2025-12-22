"""Machine learning tools for model training and evaluation."""

from molml_mcp.tools.ml.metrics import calculate_metrics, list_all_supported_metrics
from molml_mcp.tools.ml.evaluation import predict_ml_model
from molml_mcp.tools.ml.training import train_ml_model
from molml_mcp.tools.ml.hyperparam_tuning import tune_hyperparameters
from molml_mcp.tools.ml.trad_ml_models import get_hyperparameter_space, get_all_hyperparameter_spaces


def get_all_ml_tools():
    """Get all ML tools for registration."""
    return [
        calculate_metrics,
        list_all_supported_metrics,
        predict_ml_model,
        train_ml_model,
        tune_hyperparameters,
        get_hyperparameter_space,
        get_all_hyperparameter_spaces,
    ]
