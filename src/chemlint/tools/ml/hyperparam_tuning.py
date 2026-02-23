from chemlint.tools.ml.cross_validation import _cross_validate_and_eval
from chemlint.infrastructure.resources import _load_resource, _store_resource


def _define_search_space(param_grid, search_strategy, n_searches, random_state=42):
    from sklearn.model_selection import ParameterGrid
    import random

    if search_strategy == "grid":
        # Full grid search
        return list(ParameterGrid(param_grid))
    elif search_strategy == "random":
        # Random search - sample without replacement to ensure each parameter set is tried only once
        all_params = list(ParameterGrid(param_grid))
        if n_searches >= len(all_params):
            # If n_searches >= total combinations, just return all (equivalent to grid search)
            return all_params
        # Set seed for reproducibility
        random.seed(random_state)
        return random.sample(all_params, n_searches)
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")


def tune_hyperparameters(
    input_filename: str,
    feature_vectors_filename: str,
    smiles_column: str,     
    target_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    model_algorithm: str = "random_forest_classifier",
    param_grid: dict = None,
    search_strategy: str = "grid",
    n_searches: int = 50,
    cv_strategy: str = "stratified",
    n_folds: int = 5,
    val_size: float = 0.2,
    scaffold_column: str = None,
    shuffle: bool = True,
    p: int = 1,
    max_splits: int = 100,
    cluster_column: str = None,
    higher_is_better: bool = True,
    metric: str = "auto",
    random_state: int = 42
) -> dict:
    """
    Perform hyperparameter tuning using cross-validation.
    
    Searches for best hyperparameters by evaluating parameter combinations with CV.
    Supports grid search (exhaustive) and random search (sampling).
    
    Parameters
    ----------
    input_filename : str
        Dataset filename (CSV).
    feature_vectors_filename : str
        Feature vectors filename (JSON dict {smiles: [features]}).
    smiles_column : str
        SMILES column name.
    target_column : str
        Label/target column name.
    project_manifest_path : str
        Path to manifest.json.
    output_filename : str
        Output filename for best hyperparameters (JSON).
    explanation : str
        Description for manifest.
    model_algorithm : str, default="random_forest_classifier"
        ML algorithm (e.g., "random_forest_classifier", "gradient_boosting_regressor", "svr").
    param_grid : dict, optional
        Hyperparameters to search {param_name: [values]}. If None, uses defaults.
    search_strategy : str, default="grid"
        "grid" (exhaustive) or "random" (sampling).
    n_searches : int, default=50
        Number of combinations for random search (ignored for grid).
    cv_strategy : str, default="stratified"
        CV strategy: "kfold", "stratified", "scaffold", "cluster", "montecarlo", "leavepout".
    n_folds : int, default=5
        Number of CV folds.
    val_size : float, default=0.2
        Validation size for montecarlo CV.
    scaffold_column : str, optional
        Scaffold column (required if cv_strategy="scaffold").
    shuffle : bool, default=True
        Whether to shuffle data.
    p : int, default=1
        Samples to leave out (leavepout strategy).
    max_splits : int, default=100
        Max splits for leavepout.
    cluster_column : str, optional
        Cluster column (required if cv_strategy="cluster").
    higher_is_better : bool, default=True
        True to maximize metric, False to minimize.
    metric : str, default="auto"
        Metric to optimize (classification: "accuracy", "f1_score", "roc_auc"; regression: "r2", "mse", "rmse").
    random_state : int, default=42
        Random seed.
    
    Returns
    -------
    dict
        Contains output_filename, best_hyperparameters, best_score, n_successful, n_total, success_rate.
    
    Notes
    -----
    - Grid search: evaluates all combinations
    - Random search: samples n_searches combinations without replacement
    - Use random search for large parameter spaces
    """
    
    # load training data
    train_df = _load_resource(project_manifest_path, input_filename)
    
    # load feature vectors dict
    feature_vectors_dict = _load_resource(project_manifest_path, feature_vectors_filename)

    # get list of dicts of hyperparams to explore from the param_grid (full grid for grid search, random with size n for random search)
    hyper_params = _define_search_space(param_grid, search_strategy, n_searches, random_state)  # list of dicts

    # perform cross-validation for each set of hyperparams and collect results
    cv_results = []
    error_log = []  # Track all errors for debugging
    
    for idx, params in enumerate(hyper_params):
        try:
            score = _cross_validate_and_eval(model_algorithm=model_algorithm,
                                             dataset=train_df,
                                             smiles_column=smiles_column,
                                             label_column=target_column,
                                             feature_vector_dict=feature_vectors_dict,
                                             cv_strategy=cv_strategy,
                                             n_folds=n_folds,
                                             random_state=random_state,
                                             metric=metric,
                                             hyperparameters=params,
                                             cluster_column=cluster_column,
                                             val_size=val_size,
                                             scaffold_column=scaffold_column,
                                             shuffle=shuffle,
                                             p=p,
                                             max_splits=max_splits)
            mean_score = sum(score) / len(score)
            cv_results.append(mean_score)

        except Exception as e:
            # If this parameter combination fails, record None and continue
            import traceback
            error_msg = f"Params {idx+1}/{len(hyper_params)}: {params}\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
            error_log.append(error_msg)
            cv_results.append(None)
        
    # Filter out failed runs (None values)
    valid_results = [(i, score) for i, score in enumerate(cv_results) if score is not None]
    
    if not valid_results:
        # Provide detailed error information
        error_summary = "\n\n".join(error_log[:3])  # Show first 3 errors
        raise ValueError(
            f"All hyperparameter combinations failed during cross-validation.\n\n"
            f"Total attempts: {len(hyper_params)}\n"
            f"All failed: {len(error_log)}\n\n"
            f"First errors:\n{error_summary}\n\n"
            f"Check your parameter grid, data, cv_strategy settings, and required columns."
        )
    
    # get the best hyperparams based on the best score among valid results
    if higher_is_better:
        best_index, best_score = max(valid_results, key=lambda x: x[1])
    else:
        best_index, best_score = min(valid_results, key=lambda x: x[1])

    best_hyperparams = hyper_params[best_index]

    # store the best hyperparams as a json resource
    output_filename = _store_resource(best_hyperparams, project_manifest_path, output_filename, explanation, 'json')
    
    # Calculate success rate
    n_successful = len(valid_results)
    n_total = len(hyper_params)
        
    return {
        "output_filename": output_filename,
        "best_hyperparameters": best_hyperparams,
        "best_score": best_score,
        "n_successful": n_successful,
        "n_total": n_total,
        "success_rate": n_successful / n_total
    }   


