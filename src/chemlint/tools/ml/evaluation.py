import numpy as np
from typing import Dict, Any, List, Optional


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
    
    Returns:
        Metric value as float (or None if metric cannot be computed)
    """
    from chemlint.tools.ml.metrics import _get_metric_function
    
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
    Generate predictions from single model or CV ensemble. For CV models, outputs per-fold predictions plus aggregates (mean, std, entropy).
    For uncertainty-aware models (BayesianEnsemble), includes uncertainty estimates.
    
    Args:
        ml_model_filename: Model(s) from train_single_ml_model() or train_ml_models_cross_validation()
        test_input_filename: Test CSV with SMILES
        test_feature_vectors_filename: Feature vectors JSON
        test_smiles_column: SMILES column name
        predict_column_name: Base name for prediction columns
        project_manifest_path: Path to manifest.json
        output_filename: Output CSV name with predictions
        explanation: Description
    
    Returns:
        Dict with output_filename, n_models (1 or n_folds), and prediction column names
    """
    from chemlint.infrastructure.resources import _load_resource, _store_resource
    import pandas as pd
    import numpy as np
    from scipy.stats import entropy
    
    # Load model data
    model_data = _load_resource(project_manifest_path, ml_model_filename)
    
    # Extract models (support both single and multiple)
    if isinstance(model_data, dict) and "models" in model_data:
        models = model_data["models"]
        is_cv = len(models) > 1
    else:
        models = [model_data]
        is_cv = False
    
    # Load test dataset and features
    test_df = _load_resource(project_manifest_path, test_input_filename)
    feature_vectors = _load_resource(project_manifest_path, test_feature_vectors_filename)
    
    if test_smiles_column not in test_df.columns:
        raise ValueError(f"SMILES column '{test_smiles_column}' not found in test dataset")
    
    test_smiles = test_df[test_smiles_column].tolist()
    X_test = np.array([feature_vectors[smi] for smi in test_smiles])
    
    output_df = test_df.copy()
    
    # Check if models support uncertainty prediction
    has_uncertainty = hasattr(models[0], "predict_with_uncertainty")
    
    # For probability uncertainty, we need to check if the trained estimators support it
    # BayesianEnsemble wraps models, so check the underlying estimators
    has_proba_uncertainty = False
    if hasattr(models[0], "predict_proba_with_uncertainty"):
        # Check if the underlying estimators support predict_proba
        if hasattr(models[0], 'estimators_') and len(models[0].estimators_) > 0:
            has_proba_uncertainty = hasattr(models[0].estimators_[0], 'predict_proba')
        else:
            # Fallback: check if the model itself has predict_proba
            has_proba_uncertainty = hasattr(models[0], "predict_proba")
    
    if not is_cv:
        # Single model: simple prediction
        model = models[0]
        
        if has_uncertainty:
            # Uncertainty-aware model (e.g., BayesianEnsemble)
            mean, std, _ = model.predict_with_uncertainty(X_test)
            output_df[predict_column_name] = mean
            output_df[f"{predict_column_name}_uncertainty"] = std
        else:
            # Standard model
            predictions = model.predict(X_test)
            output_df[predict_column_name] = predictions
        
        # Add probability predictions with uncertainty if available
        if has_proba_uncertainty:
            proba_mean, proba_std, _ = model.predict_proba_with_uncertainty(X_test)
            # For binary classification, store probability of positive class
            if proba_mean.shape[1] == 2:
                output_df[f"{predict_column_name}_proba"] = proba_mean[:, 1]
                output_df[f"{predict_column_name}_proba_uncertainty"] = proba_std[:, 1]
            else:
                # Multi-class: store all class probabilities and uncertainties
                for class_idx in range(proba_mean.shape[1]):
                    output_df[f"{predict_column_name}_proba_class{class_idx}"] = proba_mean[:, class_idx]
                    output_df[f"{predict_column_name}_proba_class{class_idx}_uncertainty"] = proba_std[:, class_idx]
        elif hasattr(model, "predict_proba"):
            # Standard probability predictions without uncertainty
            # But skip if this is a BayesianEnsemble with regressors
            try:
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    output_df[f"{predict_column_name}_proba"] = proba[:, 1]
                else:
                    for class_idx in range(proba.shape[1]):
                        output_df[f"{predict_column_name}_proba_class{class_idx}"] = proba[:, class_idx]
            except AttributeError:
                # Model has predict_proba method but doesn't support it (e.g., BayesianEnsemble with regressor)
                pass
    else:
        # Multiple models: per-fold predictions + aggregates
        all_predictions = []
        all_uncertainties = [] if has_uncertainty else None
        
        for fold_idx, model in enumerate(models, 1):
            if has_uncertainty:
                mean, std, _ = model.predict_with_uncertainty(X_test)
                col_name = f"{predict_column_name}_{fold_idx}"
                output_df[col_name] = mean
                output_df[f"{col_name}_uncertainty"] = std
                all_predictions.append(mean)
                all_uncertainties.append(std)
            else:
                preds = model.predict(X_test)
                col_name = f"{predict_column_name}_{fold_idx}"
                output_df[col_name] = preds
                all_predictions.append(preds)
        
        # Convert to array for aggregation
        pred_array = np.array(all_predictions)  # Shape: (n_folds, n_samples)
        
        # Mean and std across folds
        output_df[f"{predict_column_name}_mean"] = np.mean(pred_array, axis=0)
        output_df[f"{predict_column_name}_std"] = np.std(pred_array, axis=0)
        
        # If uncertainty estimates are available, also aggregate those
        if all_uncertainties is not None:
            uncertainty_array = np.array(all_uncertainties)
            output_df[f"{predict_column_name}_uncertainty_mean"] = np.mean(uncertainty_array, axis=0)
        
        # Entropy for classification (check if predictions are discrete)
        if hasattr(models[0], "predict_proba") or has_proba_uncertainty:
            # Classification: compute entropy from prediction distribution
            entropies = []
            for sample_idx in range(len(test_smiles)):
                sample_preds = pred_array[:, sample_idx]
                # Count occurrences of each class
                unique, counts = np.unique(sample_preds, return_counts=True)
                probs = counts / len(sample_preds)
                entropies.append(entropy(probs, base=2))
            output_df[f"{predict_column_name}_entropy"] = entropies
    
    # Store output
    output_id = _store_resource(
        output_df,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_models": len(models),
        "n_predictions": len(test_smiles),
        "has_uncertainty": has_uncertainty,
        "columns": output_df.columns.tolist()
    }


def evaluate_models(
    model_filename: str,
    feature_vectors_filename: str,
    project_manifest_path: str,
    metrics: List[str],
    output_filename: str,
    test_input_filename: Optional[str] = None,
    test_smiles_column: Optional[str] = None,
    test_label_column: Optional[str] = None,
    explanation: str = "Model evaluation results",
    use_cv_validation_sets: bool = False,
    evaluate_training_sets: bool = False,
    test_feature_vectors_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate trained model(s) on test data. Handles both single models and CV ensembles automatically.
    
    For single models: computes metrics directly on test set (test params required).
    For CV models: can evaluate on CV validation sets, training sets, or external test set. Aggregates per-fold statistics.
    
    Args:
        model_filename: Model(s) from train_single_ml_model() or train_ml_models_cross_validation()
        feature_vectors_filename: Feature vectors JSON
        project_manifest_path: Path to manifest.json
        metrics: Metrics to compute (e.g., ["accuracy", "f1_score", "r2"])
        output_filename: Output JSON report name
        test_input_filename: Test CSV (required for single models and CV test mode)
        test_smiles_column: SMILES column (required for single models and CV test mode)
        test_label_column: Label column (required for single models and CV test mode)
        explanation: Description
        use_cv_validation_sets: For CV only - use CV validation splits instead of test set
        evaluate_training_sets: For CV only - also compute training metrics
        test_feature_vectors_filename: Optional different features for test set
        
    Returns:
        Dict with output_filename, metrics (single) or metrics_summary (CV), n_models, etc.
    """
    from chemlint.infrastructure.resources import _load_resource, _store_resource
    import pandas as pd
    
    # Load model data
    model_data = _load_resource(project_manifest_path, model_filename)
    
    # Detect single vs CV models
    if isinstance(model_data, dict) and "models" in model_data:
        models = model_data["models"]
        is_cv = len(models) > 1
        model_algorithm = model_data.get("model_algorithm", "unknown")
    else:
        models = [model_data]
        is_cv = False
        model_algorithm = "unknown"
    
    # Single model case or CV test mode requires test data
    if not is_cv or not use_cv_validation_sets:
        if not all([test_input_filename, test_smiles_column, test_label_column]):
            raise ValueError("test_input_filename, test_smiles_column, and test_label_column are required")
    
    # Load test data if needed
    if test_input_filename:
        test_df = _load_resource(project_manifest_path, test_input_filename)
        if test_smiles_column not in test_df.columns or test_label_column not in test_df.columns:
            raise ValueError(f"Required columns not found in test data")
        
        test_smiles = test_df[test_smiles_column].tolist()
        test_labels = test_df[test_label_column].values
    
    # Single model case
    if not is_cv:
        feature_vectors = _load_resource(project_manifest_path, feature_vectors_filename)
        missing = [s for s in test_smiles if s not in feature_vectors]
        if missing:
            raise ValueError(f"Missing {len(missing)} feature vectors")
        
        X_test = np.array([feature_vectors[s] for s in test_smiles])
        
        metrics_computed = {}
        for metric_name in metrics:
            try:
                metrics_computed[metric_name] = float(_eval_single_ml_model(models[0], X_test, test_labels, metric_name))
            except Exception:
                metrics_computed[metric_name] = None
        
        report = {
            "evaluation_type": "single_model_evaluation",
            "model_filename": model_filename,
            "model_algorithm": model_algorithm,
            "test_dataset": {
                "filename": test_input_filename,
                "n_samples": len(test_smiles),
                "smiles_column": test_smiles_column,
                "label_column": test_label_column
            },
            "metrics_requested": metrics,
            "metrics_computed": metrics_computed
        }
        
        output_id = _store_resource(report, project_manifest_path, output_filename, explanation, "json")
        
        return {
            "output_filename": output_id,
            "n_models": 1,
            "metrics_computed": metrics_computed,
            "n_samples": len(test_smiles)
        }
    
    # CV models case
    data_splits = model_data.get("data_splits", [])
    per_fold_metrics = []
    per_fold_training_metrics = [] if evaluate_training_sets else None
    
    if use_cv_validation_sets:
        # Use CV validation sets
        if len(data_splits) != len(models):
            raise ValueError(f"Model count ({len(models)}) doesn't match data splits ({len(data_splits)})")
        
        feature_vectors = _load_resource(project_manifest_path, feature_vectors_filename)
        
        for fold_idx, (model, split) in enumerate(zip(models, data_splits)):
            fold_result = _evaluate_fold_metrics(
                model, split.get("validation", {}), feature_vectors, metrics, fold_idx
            )
            if fold_result:
                per_fold_metrics.append(fold_result)
            
            if evaluate_training_sets:
                train_result = _evaluate_fold_metrics(
                    model, split.get("training", {}), feature_vectors, metrics, fold_idx
                )
                if train_result:
                    per_fold_training_metrics.append(train_result)
        
        evaluation_mode = "cv_validation"
    else:
        # Use external test set
        test_features = (_load_resource(project_manifest_path, test_feature_vectors_filename) 
                        if test_feature_vectors_filename 
                        else _load_resource(project_manifest_path, feature_vectors_filename))
        
        missing = [s for s in test_smiles if s not in test_features]
        if missing:
            raise ValueError(f"Missing {len(missing)} test feature vectors")
        
        X_test = np.array([test_features[s] for s in test_smiles])
        
        for fold_idx, model in enumerate(models):
            fold_metrics = {"fold": fold_idx + 1, "n_samples": len(test_smiles), "metrics": {}}
            for metric_name in metrics:
                try:
                    fold_metrics["metrics"][metric_name] = _eval_single_ml_model(
                        model, X_test, test_labels, metric_name
                    )
                except Exception:
                    fold_metrics["metrics"][metric_name] = None
            per_fold_metrics.append(fold_metrics)
        
        evaluation_mode = "test"
    
    # Aggregate metrics
    metrics_summary = _aggregate_metrics(per_fold_metrics, metrics)
    training_metrics_summary = (_aggregate_metrics(per_fold_training_metrics, metrics) 
                                if evaluate_training_sets and per_fold_training_metrics else None)
    
    # Build report
    report = {
        "evaluation_type": "cv_ensemble_evaluation",
        "model_filename": model_filename,
        "evaluation_mode": evaluation_mode,
        "n_models": len(models),
        "model_algorithm": model_algorithm,
        "cv_strategy": model_data.get("cv_strategy", "unknown"),
        "metrics_requested": metrics,
        "metrics_summary": metrics_summary,
        "per_fold_metrics": per_fold_metrics
    }
    
    # Add test dataset info if available
    if test_input_filename:
        report["test_dataset"] = {
            "filename": test_input_filename,
            "n_samples": len(test_smiles),
            "smiles_column": test_smiles_column,
            "label_column": test_label_column
        }
    
    if training_metrics_summary:
        report["training_metrics_summary"] = training_metrics_summary
        report["per_fold_training_metrics"] = per_fold_training_metrics
    
    output_id = _store_resource(report, project_manifest_path, output_filename, explanation, "json")
    
    return {
        "output_filename": output_id,
        "n_models": len(models),
        "cv_strategy": model_data.get("cv_strategy", "unknown"),
        "evaluation_mode": evaluation_mode,
        "metrics_summary": metrics_summary,
        "training_metrics_summary": training_metrics_summary,
        "n_folds_evaluated": len(per_fold_metrics)
    }


def _evaluate_fold_metrics(model, data_dict, feature_vectors, metrics, fold_idx):
    """Helper to evaluate metrics for a single fold."""
    if not data_dict:
        return None
    
    smiles_list = list(data_dict.keys())
    labels = np.array(list(data_dict.values()))
    
    missing = [s for s in smiles_list if s not in feature_vectors]
    if missing:
        raise ValueError(f"Fold {fold_idx + 1}: Missing {len(missing)} feature vectors")
    
    X = np.array([feature_vectors[s] for s in smiles_list])
    fold_metrics = {"fold": fold_idx + 1, "n_samples": len(smiles_list), "metrics": {}}
    
    for metric_name in metrics:
        try:
            fold_metrics["metrics"][metric_name] = _eval_single_ml_model(model, X, labels, metric_name)
        except Exception:
            fold_metrics["metrics"][metric_name] = None
    
    return fold_metrics


def _aggregate_metrics(per_fold_metrics, metrics):
    """Helper to aggregate metrics across folds."""
    summary = {}
    for metric_name in metrics:
        values = [f["metrics"][metric_name] for f in per_fold_metrics 
                  if f["metrics"].get(metric_name) is not None]
        
        if values:
            summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_folds": len(values),
                "values": [float(v) for v in values]
            }
        else:
            summary[metric_name] = {
                "mean": None, "std": None, "min": None, "max": None,
                "n_folds": 0, "values": []
            }
    return summary



