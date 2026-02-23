"""
Dimensionality reduction tools for molecular data visualization.
"""

import pandas as pd
import numpy as np
from chemlint.infrastructure.resources import _load_resource, _store_resource


def reduce_dimensions_pca(
    input_filename: str,
    feature_vectors_filename: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "PCA dimensionality reduction to 2D",
    smiles_column: str = "smiles",
    n_components: int = 2,
    use_svd_for_binary: bool = True,
    pc1_column_name: str = "PC1",
    pc2_column_name: str = "PC2"
) -> dict:
    """
    Perform dimensionality reduction using PCA and add 2D coordinates to dataset.
    
    Reduces high-dimensional feature vectors (e.g., Morgan fingerprints, descriptors)
    to 2D coordinates using PCA. Automatically detects binary data and uses
    TruncatedSVD when appropriate for better performance with sparse binary vectors.
    
    The resulting principal components are added as new columns to the dataset,
    enabling easy visualization with scatter plots.
    
    Args:
        input_filename: Input dataset filename
        feature_vectors_filename: Feature vectors to reduce (fingerprints/descriptors)
        project_manifest_path: Path to manifest.json
        output_filename: Output dataset name (with PC columns added)
        explanation: Description of reduction operation
        smiles_column: Column name containing SMILES that match feature vector keys
                      (default: \"smiles\"). Must be the SAME column used to compute
                      the feature vectors. Used to align dictionary-format features.
        n_components: Number of dimensions to reduce to (default: 2)
                     Only 2 is supported currently for visualization
        use_svd_for_binary: Use TruncatedSVD for binary data instead of PCA
                           (default: True, better for sparse fingerprints)
        pc1_column_name: Name for first principal component column (default: "PC1")
        pc2_column_name: Name for second principal component column (default: "PC2")
    
    Returns:
        dict with:
            - output_filename: Output dataset with PC columns
            - n_rows: Number of molecules
            - n_features: Original feature dimensionality
            - n_components: Number of components computed
            - method_used: "PCA" or "TruncatedSVD"
            - is_binary: Whether data was detected as binary
            - explained_variance: Variance explained by each component
            - total_variance_explained: Total variance explained by components
            - pc1_column: Name of first PC column
            - pc2_column: Name of second PC column
    
    """
    from sklearn.decomposition import PCA, TruncatedSVD
    
    # Validate n_components
    if n_components != 2:
        raise ValueError("Only n_components=2 is currently supported for visualization")
    
    # Load dataset and feature vectors
    df = _load_resource(project_manifest_path, input_filename)
    features = _load_resource(project_manifest_path, feature_vectors_filename)
    
    # Convert to numpy array if needed
    if hasattr(features, 'toarray'):
        # Sparse matrix
        features_array = features.toarray()
    elif isinstance(features, dict):
        # Dictionary mapping SMILES to arrays - need to align with dataset order
        if smiles_column not in df.columns:
            raise ValueError(
                f"Column '{smiles_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Extract features in the same order as the dataframe
        features_list = []
        for smiles in df[smiles_column]:
            if smiles not in features:
                raise ValueError(f"SMILES '{smiles}' not found in feature vectors")
            features_list.append(features[smiles])
        features_array = np.vstack(features_list)
    elif isinstance(features, list):
        # List of arrays - stack them
        features_array = np.vstack(features)
    elif isinstance(features, np.ndarray):
        # Already numpy array
        features_array = features
    else:
        # Try converting to numpy array
        features_array = np.array(features)
    
    # Ensure 2D array
    if features_array.ndim == 1:
        features_array = features_array.reshape(-1, 1)
    
    # Validate shapes match
    if len(features_array) != len(df):
        raise ValueError(
            f"Feature vectors length ({len(features_array)}) doesn't match "
            f"dataset length ({len(df)})"
        )
    
    n_samples, n_features = features_array.shape
    
    # Check if data is binary
    is_binary = np.all((features_array == 0) | (features_array == 1))
    
    # Choose method
    method_used = None
    if is_binary and use_svd_for_binary:
        # Use TruncatedSVD for binary/sparse data
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        method_used = "TruncatedSVD"
    else:
        # Use standard PCA
        reducer = PCA(n_components=n_components, random_state=42)
        method_used = "PCA"
    
    # Fit and transform
    components = reducer.fit_transform(features_array)
    
    # Get explained variance
    explained_variance = reducer.explained_variance_ratio_.tolist()
    total_variance = sum(explained_variance)
    
    # Add components to dataframe
    df_output = df.copy()
    df_output[pc1_column_name] = components[:, 0]
    df_output[pc2_column_name] = components[:, 1]
    
    # Store output
    output_id = _store_resource(
        df_output,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_rows": len(df_output),
        "n_features": n_features,
        "n_components": n_components,
        "method_used": method_used,
        "is_binary": is_binary,
        "explained_variance": [float(v) for v in explained_variance],
        "total_variance_explained": float(total_variance),
        "pc1_column": pc1_column_name,
        "pc2_column": pc2_column_name,
        "columns": df_output.columns.tolist(),
        "note": (
            f"{method_used} reduction: {n_samples} molecules, {n_features} features → "
            f"{n_components}D. Variance explained: {explained_variance[0]:.1%} (PC1), "
            f"{explained_variance[1]:.1%} (PC2), total: {total_variance:.1%}. "
            f"Added columns: '{pc1_column_name}', '{pc2_column_name}'."
        )
    }


def reduce_dimensions_tsne(
    input_filename: str,
    feature_vectors_filename: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "t-SNE dimensionality reduction to 2D",
    smiles_column: str = "smiles",
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
    tsne1_column_name: str = "tSNE1",
    tsne2_column_name: str = "tSNE2"
) -> dict:
    """
    Perform dimensionality reduction using t-SNE and add 2D coordinates to dataset.
    
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality
    reduction technique particularly good at preserving local structure and revealing
    clusters. Excellent for visualizing high-dimensional molecular data.
    
    Note: t-SNE is stochastic and computationally intensive. For large datasets (>10k),
    consider using PCA first to reduce to ~50 dimensions before t-SNE.
    
    Args:
        input_filename: Input dataset filename
        feature_vectors_filename: Feature vectors to reduce (fingerprints/descriptors)
        project_manifest_path: Path to manifest.json
        output_filename: Output dataset name (with t-SNE columns added)
        explanation: Description of reduction operation
        smiles_column: Column name containing SMILES that match feature vector keys
                      (default: \"smiles\"). Must be the SAME column used to compute
                      the feature vectors. Used to align dictionary-format features.
        n_components: Number of dimensions to reduce to (default: 2)
                     Only 2 is supported currently for visualization
        perplexity: Perplexity parameter (default: 30.0)
                   - Balance between local and global structure
                   - Should be between 5 and 50 (typically 5-50)
                   - Larger datasets can use larger perplexity
                   - Must be less than n_samples
        learning_rate: Learning rate for optimization (default: "auto")
                      - "auto": automatically determined (recommended)
                      - float: Typical range: 10 to 1000
                      - Higher = faster convergence but may miss structure
        max_iter: Maximum number of optimization iterations (default: 1000)
                 - Minimum 250 recommended
                 - More iterations = better results but slower
        metric: Distance metric (default: "euclidean")
               - "euclidean": Euclidean distance
               - "manhattan": Manhattan distance
               - "cosine": Cosine distance
               - "jaccard": Jaccard distance (good for binary data)
        tsne1_column_name: Name for first t-SNE component column (default: "tSNE1")
        tsne2_column_name: Name for second t-SNE component column (default: "tSNE2")
    
    Returns:
        dict with:
            - output_filename: Output dataset with t-SNE columns
            - n_rows: Number of molecules
            - n_features: Original feature dimensionality
            - n_components: Number of components computed
            - method_used: "t-SNE"
            - perplexity: Perplexity used
            - max_iter: Maximum number of iterations run
            - kl_divergence: Final KL divergence (lower is better)
            - tsne1_column: Name of first t-SNE column
            - tsne2_column: Name of second t-SNE column
    """
    from sklearn.manifold import TSNE
    
    # Validate n_components
    if n_components != 2:
        raise ValueError("Only n_components=2 is currently supported for visualization")
    
    # Load dataset and feature vectors
    df = _load_resource(project_manifest_path, input_filename)
    features = _load_resource(project_manifest_path, feature_vectors_filename)
    
    # Convert to numpy array if needed
    if hasattr(features, 'toarray'):
        # Sparse matrix
        features_array = features.toarray()
    elif isinstance(features, dict):
        # Dictionary mapping SMILES to arrays - need to align with dataset order
        if smiles_column not in df.columns:
            raise ValueError(
                f"Column '{smiles_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Extract features in the same order as the dataframe
        features_list = []
        for smiles in df[smiles_column]:
            if smiles not in features:
                raise ValueError(f"SMILES '{smiles}' not found in feature vectors")
            features_list.append(features[smiles])
        features_array = np.vstack(features_list)
    elif isinstance(features, list):
        # List of arrays - stack them
        features_array = np.vstack(features)
    elif isinstance(features, np.ndarray):
        # Already numpy array
        features_array = features
    else:
        # Try converting to numpy array
        features_array = np.array(features)
    
    # Ensure 2D array
    if features_array.ndim == 1:
        features_array = features_array.reshape(-1, 1)
    
    # Validate shapes match
    if len(features_array) != len(df):
        raise ValueError(
            f"Feature vectors length ({len(features_array)}) doesn't match "
            f"dataset length ({len(df)})"
        )
    
    n_samples, n_features = features_array.shape
    
    # Validate perplexity
    if perplexity >= n_samples:
        raise ValueError(
            f"Perplexity ({perplexity}) must be less than n_samples ({n_samples}). "
            f"Try a smaller perplexity (e.g., {max(5, n_samples // 3)})."
        )
    
    # Create t-SNE instance
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        random_state=42,
        verbose=0
    )
    
    # Fit and transform
    components = tsne.fit_transform(features_array)
    
    # Get KL divergence (measure of fit quality)
    kl_divergence = float(tsne.kl_divergence_)
    
    # Add components to dataframe
    df_output = df.copy()
    df_output[tsne1_column_name] = components[:, 0]
    df_output[tsne2_column_name] = components[:, 1]
    
    # Store output
    output_id = _store_resource(
        df_output,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_rows": len(df_output),
        "n_features": n_features,
        "n_components": n_components,
        "method_used": "t-SNE",
        "perplexity": perplexity,
        "max_iter": max_iter,
        "kl_divergence": kl_divergence,
        "metric": metric,
        "tsne1_column": tsne1_column_name,
        "tsne2_column": tsne2_column_name,
        "columns": df_output.columns.tolist(),
        "note": (
            f"t-SNE reduction: {n_samples} molecules, {n_features} features → "
            f"{n_components}D. Perplexity: {perplexity}, max_iter: {max_iter}, "
            f"metric: {metric}, KL divergence: {kl_divergence:.2f}. "
            f"Added columns: '{tsne1_column_name}', '{tsne2_column_name}'."
        )
    }