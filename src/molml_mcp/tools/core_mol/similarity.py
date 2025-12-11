import numpy as np
from scipy.spatial.distance import cdist
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def compute_similarity_matrix(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    feature_vectors_filename: str,
    output_filename: str,
    explanation: str,
    similarity_metric: str = 'tanimoto'
) -> dict:
    """
    Compute pairwise similarity matrix for molecules in a dataset based on precomputed feature vectors.
    
    This function efficiently computes an NxN similarity matrix for all molecules in the dataset
    using their precomputed feature vectors (e.g., ECFP, MACCS, RDKit fingerprints). The computation
    uses vectorized NumPy operations for maximum efficiency.
    
    Supported similarity metrics:
    - 'tanimoto' (default): Tanimoto coefficient, also known as Jaccard index for binary vectors
    - 'cosine': Cosine similarity
    - 'euclidean': Euclidean similarity (1 / (1 + distance))
    - 'manhattan': Manhattan similarity (1 / (1 + distance))
    - 'dice': Dice coefficient (similar to Tanimoto but with different normalization)
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource containing SMILES.
    project_manifest_path : str
        Path to the project manifest file for tracking resources.
    smiles_column : str
        Name of the column containing SMILES strings.
    feature_vectors_filename : str
        Filename of the precomputed feature vectors resource (joblib format).
        Should be a dict mapping SMILES to numpy arrays.
    output_filename : str
        Base filename for the stored similarity matrix resource (without extension).
    explanation : str
        Brief description of the similarity computation performed.
    similarity_metric : str, default='tanimoto'
        Similarity metric to use. Options: 'tanimoto', 'cosine', 'euclidean', 'manhattan', 'dice'.
    
    Returns
    -------
    dict
        {
            "output_filename": str,           # Filename for the similarity matrix resource
            "n_molecules": int,               # Number of molecules in the matrix
            "matrix_shape": tuple,            # Shape of the similarity matrix (N, N)
            "similarity_metric": str,         # Metric used
            "mean_similarity": float,         # Mean pairwise similarity (excluding diagonal)
            "median_similarity": float,       # Median pairwise similarity (excluding diagonal)
            "min_similarity": float,          # Minimum similarity value
            "max_similarity": float,          # Maximum similarity value (excluding diagonal)
            "note": str                       # Additional information
        }
    
    Raises
    ------
    ValueError
        If the SMILES column is not found, if feature vectors are missing for some molecules,
        or if an unsupported similarity metric is specified.
    
    Notes
    -----
    - The similarity matrix is symmetric with 1.0 on the diagonal
    - Computation is vectorized using NumPy for efficiency (O(N²) complexity unavoidable)
    - Memory requirement: O(N²) where N is the number of molecules
    - For large datasets (>10,000 molecules), consider computing similarities on-demand or
      using approximate nearest neighbor methods instead
    - The matrix is stored as a NumPy array in joblib format for efficient loading
    
    Examples
    --------
    # Compute Tanimoto similarity matrix for ECFP fingerprints
    result = compute_similarity_matrix(
        input_filename='molecules_cleaned_A3F2B1D4.csv',
        project_manifest_path='/path/to/manifest.json',
        smiles_column='smiles',
        feature_vectors_filename='ecfp_vectors_B2C3D4E5.joblib',
        output_filename='tanimoto_similarity',
        explanation='Tanimoto similarity matrix from ECFP4 fingerprints'
    )
    
    # Compute cosine similarity matrix
    result = compute_similarity_matrix(
        input_filename='molecules_cleaned_A3F2B1D4.csv',
        project_manifest_path='/path/to/manifest.json',
        smiles_column='smiles',
        feature_vectors_filename='rdkit_fp_C4D5E6F7.joblib',
        output_filename='cosine_similarity',
        explanation='Cosine similarity from RDKit fingerprints',
        similarity_metric='cosine'
    )
    
    See Also
    --------
    smiles_to_ecfp_dataset : Generate ECFP fingerprints
    smiles_to_maccs_dataset : Generate MACCS keys fingerprints
    smiles_to_rdkit_fp_dataset : Generate RDKit topological fingerprints
    """
    # Load dataset and feature vectors
    df = _load_resource(project_manifest_path, input_filename)
    feature_vectors = _load_resource(project_manifest_path, feature_vectors_filename)
    
    # Validate inputs
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    # Get SMILES list
    smiles_list = df[smiles_column].tolist()
    n_molecules = len(smiles_list)
    
    # Validate that all SMILES have feature vectors
    missing_smiles = [smi for smi in smiles_list if smi not in feature_vectors]
    if missing_smiles:
        raise ValueError(
            f"Feature vectors missing for {len(missing_smiles)} molecules. "
            f"Examples: {missing_smiles[:5]}. "
            f"Ensure all SMILES in the dataset have corresponding feature vectors."
        )
    
    # Build feature matrix (N × D) where N = number of molecules, D = feature dimension
    # Maintain order from SMILES list
    feature_matrix = np.array([feature_vectors[smi] for smi in smiles_list])
    
    # Compute similarity matrix based on metric
    if similarity_metric == 'tanimoto':
        # Tanimoto = dot(A,B) / (||A||² + ||B||² - dot(A,B))
        # For binary vectors: intersection / union
        dot_product = feature_matrix @ feature_matrix.T
        norms_squared = np.sum(feature_matrix ** 2, axis=1)
        # Broadcasting: norms_squared[:, None] makes it (N, 1), norms_squared makes it (N,)
        denominator = norms_squared[:, None] + norms_squared - dot_product
        # Avoid division by zero (occurs when both vectors are all zeros)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        similarity_matrix = dot_product / denominator
        
    elif similarity_metric == 'cosine':
        # Cosine similarity = dot(A,B) / (||A|| * ||B||)
        # Use cdist with 'cosine' distance, then convert to similarity
        distance_matrix = cdist(feature_matrix, feature_matrix, metric='cosine')
        similarity_matrix = 1 - distance_matrix
        
    elif similarity_metric == 'euclidean':
        # Euclidean similarity = 1 / (1 + distance)
        distance_matrix = cdist(feature_matrix, feature_matrix, metric='euclidean')
        similarity_matrix = 1 / (1 + distance_matrix)
        
    elif similarity_metric == 'manhattan':
        # Manhattan similarity = 1 / (1 + distance)
        distance_matrix = cdist(feature_matrix, feature_matrix, metric='cityblock')
        similarity_matrix = 1 / (1 + distance_matrix)
        
    elif similarity_metric == 'dice':
        # Dice coefficient = 2 * dot(A,B) / (||A||² + ||B||²)
        dot_product = feature_matrix @ feature_matrix.T
        norms_squared = np.sum(feature_matrix ** 2, axis=1)
        denominator = norms_squared[:, None] + norms_squared
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-10, denominator)
        similarity_matrix = 2 * dot_product / denominator
        
    else:
        supported_metrics = ['tanimoto', 'cosine', 'euclidean', 'manhattan', 'dice']
        raise ValueError(
            f"Unsupported similarity metric: '{similarity_metric}'. "
            f"Supported metrics: {supported_metrics}"
        )
    
    # Compute statistics (exclude diagonal for mean/median)
    # Use np.triu_indices to get upper triangle indices (excluding diagonal)
    triu_indices = np.triu_indices(n_molecules, k=1)
    off_diagonal_values = similarity_matrix[triu_indices]
    
    mean_similarity = float(np.mean(off_diagonal_values))
    median_similarity = float(np.median(off_diagonal_values))
    min_similarity = float(np.min(off_diagonal_values))
    max_similarity = float(np.max(off_diagonal_values))
    
    # Store similarity matrix as joblib (efficient for large matrices)
    output_filename = _store_resource(
        similarity_matrix, 
        project_manifest_path, 
        output_filename, 
        explanation, 
        'joblib'
    )
    
    return {
        "output_filename": output_filename,
        "n_molecules": n_molecules,
        "matrix_shape": similarity_matrix.shape,
        "similarity_metric": similarity_metric,
        "mean_similarity": mean_similarity,
        "median_similarity": median_similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "note": (
            f"Computed {n_molecules}×{n_molecules} similarity matrix using {similarity_metric} metric. "
            f"Matrix is symmetric with diagonal values = 1.0. "
            f"Average pairwise similarity: {mean_similarity:.3f}. "
            f"Use this matrix for clustering, nearest neighbor searches, or diversity analysis."
        )
    }


# TODO: Implement find_k_nearest_neighbors function
# TODO: Implement add_similarity_statistics_dataset function
