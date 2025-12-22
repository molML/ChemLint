import numpy as np
from typing import Dict, Any


def _eval_single_ml_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metric: str
) -> float:
    """
    Evaluate a single ML model on a dataset using a specific metric.
    
    Args:
        model: Trained scikit-learn model
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,)
        metric: Metric name (e.g., "accuracy", "f1_score", "mse", "r2")
        task_type: Either "classification" or "regression"
    
    Returns:
        Metric value as float (or None if metric cannot be computed)
    """
    from molml_mcp.tools.ml.metrics import _get_metric_function
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Special handling for ROC AUC (requires probabilities)
    if metric == "roc_auc":
        if not hasattr(model, "predict_proba"):
            return None
        try:
            y_pred = model.predict_proba(X)[:, 1]
        except:
            return None
    
    # Get metric function and compute
    metric_func = _get_metric_function(metric)
    return metric_func(y, y_pred)


def predict_ml_model(
    ml_model_filename: str,
    test_input_filename: str,
    test_feature_vectors_filename: str,
    test_smiles_column: str,
    predict_column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
) -> dict:
    """
    Predict using a trained ML model on a test dataset.
    
    This function loads a trained model and applies it to a test dataset to generate
    predictions. The predictions are added as a new column to the test dataset and
    stored as a new resource.
    
    After making predictions, you can evaluate model performance by using the
    calculate_metrics() function from metrics.py to compare predictions against
    true labels (if available in the test dataset).
    
    Args:
        ml_model_filename: Filename of the trained model to use for predictions
        test_input_filename: Filename of the test dataset (CSV with SMILES)
        test_feature_vectors_filename: Filename of the feature vectors (JSON dict {smiles: [features]})
        test_smiles_column: Name of the SMILES column in the test dataset
        predict_column_name: Name for the new prediction column (e.g., "predicted_label")
        project_manifest_path: Path to manifest.json
        output_filename: Name for the output dataset with predictions
        explanation: Description of this prediction operation
    
    Returns:
        dict with output_filename, n_predictions, columns, and preview
        
    Example:
        >>> # Make predictions
        >>> result = predict_ml_model(
        ...     ml_model_filename="random_forest_A1B2C3D4.pkl",
        ...     test_input_filename="test_data_E5F6G7H8.csv",
        ...     test_feature_vectors_filename="test_features_I9J0K1L2.json",
        ...     test_smiles_column="smiles",
        ...     predict_column_name="predicted_activity",
        ...     project_manifest_path="/path/to/manifest.json",
        ...     output_filename="predictions",
        ...     explanation="Predictions on test set using Random Forest model"
        ... )
        >>> 
        >>> # Evaluate predictions (if test data has true labels)
        >>> from molml_mcp.tools.ml.metrics import calculate_metrics
        >>> metrics = calculate_metrics(
        ...     input_filename=result['output_filename'],
        ...     project_manifest_path="/path/to/manifest.json",
        ...     true_label_column="true_activity",
        ...     predicted_column="predicted_activity",
        ...     metrics=["accuracy", "precision", "recall", "f1_score"]
        ... )
    """
    from molml_mcp.infrastructure.resources import _load_resource, _store_resource
    import pandas as pd
    import numpy as np
    
    # Load model data (could be a dict structure or raw model for backwards compatibility)
    model_data = _load_resource(project_manifest_path, ml_model_filename)
    
    # Extract the actual model from the structure
    if isinstance(model_data, dict) and "models" in model_data:
        # New format from train_ml_model: {"models": [model], "data_splits": [...], ...}
        model = model_data["models"][0]
    else:
        # Backwards compatibility: assume it's the model directly
        model = model_data
    
    # Load test dataset
    test_df = _load_resource(project_manifest_path, test_input_filename)
    
    # Load feature vectors
    feature_vectors = _load_resource(project_manifest_path, test_feature_vectors_filename)
    
    # Validate SMILES column exists
    if test_smiles_column not in test_df.columns:
        raise ValueError(f"SMILES column '{test_smiles_column}' not found in test dataset")
    
    # Extract SMILES from test dataset
    test_smiles = test_df[test_smiles_column].tolist()
    
    # Build feature matrix (ensure order matches test_smiles)
    X_test = np.array([feature_vectors[smi] for smi in test_smiles])
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Add predictions to dataset
    output_df = test_df.copy()
    output_df[predict_column_name] = predictions
    
    # Store output dataset
    output_id = _store_resource(
        output_df,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_predictions": len(predictions),
        "columns": output_df.columns.tolist(),
        "preview": output_df.head(5).to_dict('records')
    }
