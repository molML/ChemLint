from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def train_single_ml_model(
    train_input_filename: str,
    train_feature_vectors_filename: str,
    train_smiles_column: str,
    train_label_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    model_algorithm: str = "random_forest_classifier",
    hyperparameters: dict = None,
    random_state: int = 42
) -> dict:
    """
    Train single ML model on molecular data.
    
    Args:
        train_input_filename: Training CSV with SMILES and labels
        train_feature_vectors_filename: Feature vectors JSON
        train_smiles_column: SMILES column name
        train_label_column: Label column name
        project_manifest_path: Path to manifest.json
        output_filename: Output model name
        explanation: Description
        model_algorithm: Algorithm (e.g., "random_forest_classifier", "ridge", "svr")
        hyperparameters: Optional hyperparameter dict
        random_state: Random seed
    
    Returns:
        Dict with output_filename, model_algorithm, n_features
    """
    from molml_mcp.tools.ml.trad_ml.singular_models import get_available_models
    
    # Load training data
    train_df = _load_resource(project_manifest_path, train_input_filename)
    train_features_dict = _load_resource(project_manifest_path, train_feature_vectors_filename)
    
    # Validate required columns
    if train_smiles_column not in train_df.columns:
        raise ValueError(f"SMILES column '{train_smiles_column}' not found in {train_input_filename}")
    if train_label_column not in train_df.columns:
        raise ValueError(f"Label column '{train_label_column}' not found in {train_input_filename}")
    
    # Get SMILES and labels
    train_smiles = train_df[train_smiles_column].tolist()
    train_labels = train_df[train_label_column].values
    
    # Check that all SMILES have feature vectors
    missing_train = [smi for smi in train_smiles if smi not in train_features_dict]
    if missing_train:
        raise ValueError(f"Missing feature vectors for {len(missing_train)} training SMILES. First 5: {missing_train[:5]}")
    
    # Build feature matrix
    X_train = np.array([train_features_dict[smi] for smi in train_smiles])
    y_train = train_labels
    
    # Check if model algorithm is supported
    available_models = get_available_models()
    if model_algorithm not in available_models:
        raise ValueError(f"Model '{model_algorithm}' not supported. Available: {list(available_models.keys())}")
    
    # Train the model
    model = _train_ml_model(
        X=X_train,
        y=y_train,
        model_algorithm=model_algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state
    )
    
    # Prepare data structure matching train_ml_models_cv output format
    # Format: list of models and list of data splits (one split in this case)
    train_data_dict = {smi: label for smi, label in zip(train_smiles, y_train.tolist())}
    
    model_data = {
        "models": [model],
        "data_splits": [
            {
                "training": train_data_dict,
                "validation": {}
            }
        ],
        "model_algorithm": model_algorithm,
        "hyperparameters": hyperparameters or {},
        "random_state": random_state,
        "n_features": X_train.shape[1]
    }
    
    # Store the model
    output_id = _store_resource(
        model_data,
        project_manifest_path,
        output_filename,
        explanation,
        "model"
    )
    
    return {
        "output_filename": output_id,
        "model_algorithm": model_algorithm,
        "n_train_samples": len(train_smiles),
        "n_features": X_train.shape[1],
        "hyperparameters": hyperparameters or {}
    }


def train_ml_models_cross_validation(
    input_filename: str,
    feature_vectors_filename: str,
    smiles_column: str,
    label_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    model_algorithm: str = "random_forest_classifier",
    hyperparameters: dict = None,
    cv_strategy: str = "montecarlo",
    n_folds: int = 5,
    val_size: Optional[float] = None,
    cluster_column: Optional[str] = None,
    scaffold_column: Optional[str] = None,
    shuffle: bool = True,
    p: int = 1,
    max_splits: Optional[int] = None,
    random_state: int = 42
) -> dict:
    """
    Train multiple models using cross-validation. Creates train/val splits, trains one model per fold, stores all for ensemble use.
    
    CV strategies: 'kfold', 'stratified', 'montecarlo', 'scaffold' (needs scaffold_column), 'cluster' (needs cluster_column), 'leavepout' (needs p)
    
    Args:
        input_filename: Training CSV with SMILES and labels
        feature_vectors_filename: Feature vectors JSON
        smiles_column: SMILES column name
        label_column: Label column name
        project_manifest_path: Path to manifest.json
        output_filename: Output model name
        explanation: Description
        model_algorithm: Algorithm (e.g., "random_forest_classifier", "ridge", "svr")
        hyperparameters: Optional hyperparameter dict
        cv_strategy: CV strategy name
        n_folds: Number of folds/splits
        val_size: Validation fraction (for montecarlo)
        cluster_column: Cluster IDs column (for cluster strategy)
        scaffold_column: Scaffold IDs column (for scaffold strategy)
        shuffle: Shuffle data before splitting
        p: Samples to leave out (for leavepout)
        max_splits: Max splits for leavepout
        random_state: Random seed
    
    Returns:
        Dict with output_filename, n_models, cv_strategy, model_algorithm
    """
    from molml_mcp.tools.ml.cross_validation import get_cv_splits
    from molml_mcp.tools.ml.trad_ml.singular_models import get_available_models
    
    # Load data
    df = _load_resource(project_manifest_path, input_filename)
    feature_vectors_dict = _load_resource(project_manifest_path, feature_vectors_filename)
    
    # Validate required columns
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in {input_filename}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {input_filename}")
    
    # Validate strategy-specific columns
    if cv_strategy == 'cluster':
        if cluster_column is None:
            raise ValueError("Cluster-based CV requires 'cluster_column' parameter")
        if cluster_column not in df.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found in {input_filename}")
    
    if cv_strategy == 'scaffold':
        if scaffold_column is None:
            raise ValueError("Scaffold-based CV requires 'scaffold_column' parameter with pre-computed scaffolds")
        if scaffold_column not in df.columns:
            raise ValueError(f"Scaffold column '{scaffold_column}' not found in {input_filename}")
    
    # Check if model algorithm is supported
    available_models = get_available_models()
    if model_algorithm not in available_models:
        raise ValueError(f"Model '{model_algorithm}' not supported. Available: {list(available_models.keys())}")
    
    # Extract SMILES and labels
    smiles_list = df[smiles_column].tolist()
    labels = df[label_column].tolist()
    
    # Check that all SMILES have feature vectors
    missing_smiles = [smi for smi in smiles_list if smi not in feature_vectors_dict]
    if missing_smiles:
        raise ValueError(f"Missing feature vectors for {len(missing_smiles)} SMILES. First 5: {missing_smiles[:5]}")
    
    # Get clusters if needed
    clusters = None
    if cv_strategy == 'cluster':
        clusters = df[cluster_column].tolist()
    
    # Get scaffolds if needed
    scaffolds = None
    if cv_strategy == 'scaffold':
        scaffolds = df[scaffold_column].tolist()
    
    # Generate CV splits
    print(f"Generating {n_folds} CV splits using '{cv_strategy}' strategy...")
    splits = get_cv_splits(
        strategy=cv_strategy,
        smiles=smiles_list,
        n_folds=n_folds,
        random_state=random_state,
        labels=labels,
        clusters=clusters,
        val_size=val_size,
        scaffolds=scaffolds,
        shuffle=shuffle,
        p=p,
        max_splits=max_splits
    )
    
    print(f"Training {len(splits)} models...")
    
    # Create SMILES -> label mapping for quick lookup
    smiles_to_label = dict(zip(df[smiles_column], df[label_column]))
    
    # Train models on each split
    models = []
    data_splits = []
    
    for fold_idx, split in enumerate(splits):
        print(f"  Training model {fold_idx + 1}/{len(splits)}...")
        
        # Get train and val SMILES
        train_smiles = split['train_smiles']
        val_smiles = split['val_smiles']
        
        # Build feature matrices
        X_train = np.array([feature_vectors_dict[smi] for smi in train_smiles])
        X_val = np.array([feature_vectors_dict[smi] for smi in val_smiles])
        
        # Get labels
        y_train = np.array([smiles_to_label[smi] for smi in train_smiles])
        y_val = np.array([smiles_to_label[smi] for smi in val_smiles])
        
        # Train model
        model = _train_ml_model(
            X=X_train,
            y=y_train,
            model_algorithm=model_algorithm,
            hyperparameters=hyperparameters,
            random_state=random_state
        )
        
        models.append(model)
        
        # Store data split info (SMILES -> label mapping for each split)
        train_data_dict = {smi: label for smi, label in zip(train_smiles, y_train.tolist())}
        val_data_dict = {smi: label for smi, label in zip(val_smiles, y_val.tolist())}
        
        data_splits.append({
            'training': train_data_dict,
            'validation': val_data_dict
        })
    
    # Get feature dimensionality (use first model's training data)
    n_features = len(feature_vectors_dict[smiles_list[0]])
    
    # Prepare model data structure
    model_data = {
        "models": models,
        "data_splits": data_splits,
        "model_algorithm": model_algorithm,
        "hyperparameters": hyperparameters or {},
        "random_state": random_state,
        "n_features": n_features,
        "cv_strategy": cv_strategy,
        "cv_parameters": {
            "n_folds": n_folds,
            "val_size": val_size,
            "cluster_column": cluster_column,
            "scaffold_column": scaffold_column,
            "shuffle": shuffle,
            "p": p,
            "max_splits": max_splits
        }
    }
    
    # Store the models
    print(f"Saving {len(models)} trained models...")
    output_id = _store_resource(
        model_data,
        project_manifest_path,
        output_filename,
        explanation,
        "model"
    )
    
    print(f"✓ Training complete: {output_id}")
    
    return {
        "output_filename": output_id,
        "model_algorithm": model_algorithm,
        "n_models": len(models),
        "n_features": n_features,
        "cv_strategy": cv_strategy,
        "n_folds": n_folds,
        "hyperparameters": hyperparameters or {}
    }


def _train_ml_model(
    X: np.ndarray,
    y: np.ndarray,
    model_algorithm: str,
    hyperparameters: dict,
    random_state: int
):
    """
    Internal function to train a machine learning model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        model_algorithm: Algorithm name (e.g., "random_forest_classifier", "ridge", "svr")
        hyperparameters: Dictionary of hyperparameters to pass to the model.
                        If None or empty, the training function will use its defaults.
        random_state: Random seed for reproducibility
    
    Returns:
        Trained scikit-learn model object
    
    Raises:
        ValueError: If model_algorithm is not found
    """
    from molml_mcp.tools.ml.trad_ml.singular_models import get_model_function
    import inspect
    
    # Get the training function for the specified model
    try:
        train_func = get_model_function(model_algorithm)
    except ValueError as e:
        raise ValueError(f"Invalid model_algorithm '{model_algorithm}': {e}")
    
    # Prepare parameters
    params = hyperparameters.copy() if hyperparameters else {}
    
    # Check if the training function accepts random_state parameter
    sig = inspect.signature(train_func)
    if "random_state" in sig.parameters:
        params["random_state"] = random_state
    
    # Train the model
    model = train_func(X, y, **params)
    
    return model

