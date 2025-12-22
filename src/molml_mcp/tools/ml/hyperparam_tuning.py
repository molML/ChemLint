from molml_mcp.tools.ml.cross_validation import _cross_validate_and_eval
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


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
    scaffold_type: str = "bemis_murcko",
    shuffle: bool = True,
    p: int = 1,
    max_splits: int = 100,
    cluster_column: str = None,
    higher_is_better: bool = True,
    metric: str = "auto",
    random_state: int = 42
) -> dict:
    """
    Perform hyperparameter tuning for a machine learning model using cross-validation.
    
    This function searches for the best hyperparameters by evaluating different parameter
    combinations using cross-validation. Supports both grid search (exhaustive) and random
    search (sampling) strategies. The best hyperparameters are stored as a JSON resource.
    
    Args:
        input_filename: Dataset filename with SMILES and labels (CSV)
        feature_vectors_filename: Feature vectors filename (JSON dict {smiles: [features]})
        smiles_column: Name of SMILES column in dataset
        target_column: Name of label/target column in dataset
        project_manifest_path: Path to manifest.json
        output_filename: Name for output file with best hyperparameters (JSON)
        explanation: Description of tuning operation
        model_algorithm: ML algorithm to use (e.g., "random_forest_classifier", "gradient_boosting_regressor", "svr")
        param_grid: Dictionary of hyperparameters to search {param_name: [values]}
                   If None, uses default parameter grid from get_hyperparameter_space()
        search_strategy: Search strategy - "grid" for exhaustive grid search, 
                        "random" for random sampling
        n_searches: Number of parameter combinations to try for random search
                   (ignored for grid search which tries all combinations)
        cv_strategy: Cross-validation strategy ("kfold", "stratified", "scaffold", 
                    "cluster", "montecarlo", "leavepout")
        n_folds: Number of CV folds
        val_size: Validation size fraction (used for montecarlo CV)
        scaffold_type: Type of scaffold for scaffold-based CV ("bemis_murcko", "generic", "cyclic_skeleton")
        shuffle: Whether to shuffle data before splitting
        p: Number of samples to leave out for leavepout strategy
        max_splits: Maximum number of splits for leavepout strategy
        cluster_column: Column name for cluster-based CV (required if cv_strategy="cluster")
        higher_is_better: If True, maximize the metric; if False, minimize it
                         (True for accuracy/f1/r2, False for mse/mae/rmse)
        metric: Metric to optimize ("auto" auto-detects, or specify like "accuracy", "f1_score", "r2", "mse")
        random_state: Random seed for reproducibility
    
    Returns:
        dict with:
            - output_filename: Filename of stored best hyperparameters (JSON)
            - best_hyperparameters: Dictionary of best hyperparameter values
            - best_score: Best cross-validation score achieved
            - n_successful: Number of parameter combinations that completed successfully
            - n_total: Total number of parameter combinations attempted
            - success_rate: Fraction of successful runs (n_successful / n_total)
    
    Example:
        >>> # Grid search for Random Forest Classifier
        >>> result = tune_hyperparameters(
        ...     input_filename="train_data_A1B2C3D4.csv",
        ...     feature_vectors_filename="features_E5F6G7H8.json",
        ...     smiles_column="smiles",
        ...     target_column="activity",
        ...     project_manifest_path="/path/to/manifest.json",
        ...     output_filename="rf_best_params",
        ...     explanation="Hyperparameter tuning for Random Forest classifier",
        ...     model_algorithm="random_forest_classifier",
        ...     param_grid={
        ...         "n_estimators": [50, 100, 200],
        ...         "max_depth": [3, 5, 10, None],
        ...         "min_samples_split": [2, 5, 10]
        ...     },
        ...     search_strategy="grid",
        ...     cv_strategy="stratified",
        ...     n_folds=5,
        ...     metric="accuracy",
        ...     higher_is_better=True
        ... )
        >>> print(f"Best accuracy: {result['best_score']:.4f}")
        >>> print(f"Best params: {result['best_hyperparameters']}")
        
        >>> # Random search with scaffold CV
        >>> result = tune_hyperparameters(
        ...     input_filename="train_data_A1B2C3D4.csv",
        ...     feature_vectors_filename="features_E5F6G7H8.json",
        ...     smiles_column="smiles",
        ...     target_column="ic50",
        ...     project_manifest_path="/path/to/manifest.json",
        ...     output_filename="svr_best_params",
        ...     explanation="Random search for SVR with scaffold CV",
        ...     model_algorithm="svr",
        ...     param_grid={
        ...         "C": [0.1, 1, 10, 100],
        ...         "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        ...         "kernel": ["rbf", "linear"]
        ...     },
        ...     search_strategy="random",
        ...     n_searches=20,
        ...     cv_strategy="scaffold",
        ...     n_folds=5,
        ...     metric="r2",
        ...     higher_is_better=True
        ... )
    
    Note:
        - Grid search evaluates all possible combinations of hyperparameters
        - Random search samples n_searches combinations without replacement
        - Use random search when the parameter space is very large
        - The best hyperparameters can be loaded and used with train_ml_model()
    """
    
    # load training data
    train_df = _load_resource(project_manifest_path, input_filename)
    
    # load feature vectors dict
    feature_vectors_dict = _load_resource(project_manifest_path, feature_vectors_filename)

    # get list of dicts of hyperparams to explore from the param_grid (full grid for grid search, random with size n for random search)
    hyper_params = _define_search_space(param_grid, search_strategy, n_searches, random_state)  # list of dicts

    # perform cross-validation for each set of hyperparams and collect results
    cv_results = []
    for params in hyper_params:
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
                                             scaffold_type=scaffold_type,
                                             shuffle=shuffle,
                                             p=p,
                                             max_splits=max_splits)
            cv_results.append(score)
        except Exception as e:
            # If this parameter combination fails, record None and continue
            print(f"Warning: Parameter combination {params} failed with error: {str(e)[:100]}")
            cv_results.append(None)
        
    # Filter out failed runs (None values)
    valid_results = [(i, score) for i, score in enumerate(cv_results) if score is not None]
    
    if not valid_results:
        raise ValueError("All hyperparameter combinations failed during cross-validation. Check your parameter grid and data.")
    
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


