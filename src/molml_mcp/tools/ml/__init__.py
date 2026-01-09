"""Machine learning tools for model training and evaluation."""

from molml_mcp.tools.ml.metrics import calculate_metrics, list_all_supported_metrics
from molml_mcp.tools.ml.evaluation import predict_ml_model, evaluate_models
from molml_mcp.tools.ml.training import train_single_ml_model, train_ml_models_cross_validation
from molml_mcp.tools.ml.hyperparam_tuning import tune_hyperparameters


def get_all_ml_tools():
    """Get all ML tools for registration."""
    return [
        calculate_metrics,
        list_all_supported_metrics,
        predict_ml_model,
        evaluate_models,
        train_single_ml_model,
        train_ml_models_cross_validation,
        tune_hyperparameters,
    ]
