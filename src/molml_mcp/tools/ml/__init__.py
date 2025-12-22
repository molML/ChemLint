"""Machine learning tools for model training and evaluation."""

from molml_mcp.tools.ml.metrics import calculate_metrics, list_all_supported_metrics
from molml_mcp.tools.ml.evaluation import predict_ml_model
from molml_mcp.tools.ml.training import train_ml_model


def get_all_ml_tools():
    """Get all ML tools for registration."""
    return [
        calculate_metrics,
        list_all_supported_metrics,
        predict_ml_model,
        train_ml_model,
    ]
