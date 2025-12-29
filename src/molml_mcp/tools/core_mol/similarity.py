import numpy as np
from scipy.spatial.distance import cdist
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def _tanimoto_similarity(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Tanimoto similarity matrix using RDKit's optimized implementation.
    
    Uses RDKit's C++ BulkTanimotoSimilarity for ~5x speedup over pure NumPy.
    Best for binary molecular fingerprints (ECFP, MACCS, Morgan, RDKit FP).
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Binary feature matrix of shape (n_molecules, n_features).
    
    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (n_molecules, n_molecules).
    """
    from rdkit import DataStructs
    
    n_molecules = len(feature_matrix)
    
    # Convert numpy arrays to RDKit ExplicitBitVect objects
    # Using CreateFromBitString is ~1.75x faster than SetBitsFromList
    rdkit_fps = []
    for fp_array in feature_matrix:
        bit_string = ''.join(map(str, fp_array.astype(int)))
        ebv = DataStructs.CreateFromBitString(bit_string)
        rdkit_fps.append(ebv)
    
    # Compute similarity matrix using RDKit's C++ implementation
    similarity_matrix = np.zeros((n_molecules, n_molecules))
    for i in range(n_molecules):
        sims = DataStructs.BulkTanimotoSimilarity(rdkit_fps[i], rdkit_fps)
        similarity_matrix[i, :] = sims
    
    return similarity_matrix


def _dice_similarity(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Dice coefficient similarity matrix.
    
    Dice coefficient = 2 * dot(A,B) / (||A||² + ||B||²)
    Similar to Tanimoto but with different normalization.
    Best for binary fingerprints.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Binary feature matrix of shape (n_molecules, n_features).
    
    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (n_molecules, n_molecules).
    """
    dot_product = feature_matrix @ feature_matrix.T
    norms_squared = np.sum(feature_matrix ** 2, axis=1)
    denominator = norms_squared[:, None] + norms_squared
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    similarity_matrix = 2 * dot_product / denominator
    
    return similarity_matrix


def _cosine_similarity(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix.
    
    Cosine similarity = dot(A,B) / (||A|| * ||B||)
    Measures angle between vectors, insensitive to magnitude.
    Best for continuous-valued descriptors or count fingerprints.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_molecules, n_features).
    
    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (n_molecules, n_molecules).
    """
    distance_matrix = cdist(feature_matrix, feature_matrix, metric='cosine')
    similarity_matrix = 1 - distance_matrix
    
    return similarity_matrix


def _euclidean_similarity(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance-based similarity matrix.
    
    Euclidean similarity = 1 / (1 + distance)
    Sensitive to feature magnitude and scale.
    Best for continuous-valued descriptors.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_molecules, n_features).
    
    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (n_molecules, n_molecules).
    """
    distance_matrix = cdist(feature_matrix, feature_matrix, metric='euclidean')
    similarity_matrix = 1 / (1 + distance_matrix)
    
    return similarity_matrix


def _manhattan_similarity(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Manhattan distance-based similarity matrix.
    
    Manhattan similarity = 1 / (1 + distance)
    Less sensitive to outliers than Euclidean.
    Best for continuous-valued descriptors.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_molecules, n_features).
    
    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (n_molecules, n_molecules).
    """
    distance_matrix = cdist(feature_matrix, feature_matrix, metric='cityblock')
    similarity_matrix = 1 / (1 + distance_matrix)
    
    return similarity_matrix


# Metric dispatch dictionary for easy extension
_SIMILARITY_METRICS = {
    'tanimoto': _tanimoto_similarity,
    'dice': _dice_similarity,
    'cosine': _cosine_similarity,
    'euclidean': _euclidean_similarity,
    'manhattan': _manhattan_similarity,
}


def _compute_pairwise_similarity(feature_matrix: np.ndarray, similarity_metric: str) -> np.ndarray:
    """
    Compute pairwise similarity matrix from feature matrix using specified metric.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_molecules, n_features).
    similarity_metric : str
        Similarity metric: 'tanimoto', 'dice', 'cosine', 'euclidean', or 'manhattan'.
    
    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (n_molecules, n_molecules).
    
    Raises
    ------
    ValueError
        If an unsupported similarity metric is specified.
    """
    if similarity_metric not in _SIMILARITY_METRICS:
        supported_metrics = list(_SIMILARITY_METRICS.keys())
        raise ValueError(
            f"Unsupported similarity metric: '{similarity_metric}'. "
            f"Supported metrics: {supported_metrics}"
        )
    
    # Dispatch to appropriate metric function
    similarity_func = _SIMILARITY_METRICS[similarity_metric]
    return similarity_func(feature_matrix)


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
    Compute pairwise similarity matrix for molecules using precomputed molecular fingerprints.
    
    Calculates an NxN similarity matrix for all molecule pairs in the dataset. The matrix is symmetric
    with diagonal values of 1.0 (each molecule has perfect similarity to itself).
    
    **Available Similarity Metrics:**
    
    - **tanimoto** (default, RECOMMENDED): Tanimoto coefficient (Jaccard index)
      - USE WITH: Binary fingerprints (ECFP, MACCS, RDKit FP, Morgan)
      - PERFORMANCE: Optimized with RDKit C++ implementation (~5x faster)
      - BEST FOR: Molecular similarity - the de facto standard in cheminformatics
    
    - **dice**: Dice coefficient
      - USE WITH: Binary fingerprints
      - Similar to Tanimoto but with different normalization (2ab/(a+b) vs ab/(a+b-ab))
    
    - **cosine**: Cosine similarity
      - USE WITH: Any continuous-valued vectors (count fingerprints, descriptors)
      - Measures angle between vectors, insensitive to magnitude
    
    - **euclidean**: Euclidean distance-based similarity (1/(1+distance))
      - USE WITH: Continuous-valued descriptors or count fingerprints
      - Sensitive to feature magnitude and scale
    
    - **manhattan**: Manhattan distance-based similarity (1/(1+distance))
      - USE WITH: Continuous-valued descriptors or count fingerprints
      - Less sensitive to outliers than Euclidean
    
    Parameters
    ----------
    input_filename : str
        Filename of the input dataset containing SMILES strings.
    project_manifest_path : str
        Path to the project's manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    feature_vectors_filename : str
        Filename of the precomputed feature vectors (joblib format).
        Must be a dictionary mapping SMILES to numpy arrays (fingerprints/descriptors).
    output_filename : str
        Base name for the output similarity matrix (extension added automatically).
    explanation : str
        Description of what this similarity computation represents.
    similarity_metric : str, default='tanimoto'
        Similarity metric to use:
        - 'tanimoto': For binary fingerprints (RECOMMENDED for molecules)
        - 'dice': For binary fingerprints
        - 'cosine': For count fingerprints or continuous descriptors
        - 'euclidean': For continuous descriptors
        - 'manhattan': For continuous descriptors
    
    Returns
    -------
    dict
        Dictionary containing:
        - output_filename: Stored similarity matrix filename
        - n_molecules: Number of molecules
        - matrix_shape: Dimensions (N, N)
        - similarity_metric: Metric used
        - mean_similarity: Average pairwise similarity (excluding self-similarity)
        - median_similarity: Median pairwise similarity
        - min_similarity: Minimum similarity value
        - max_similarity: Maximum similarity value (excluding diagonal)
        - note: Summary with usage suggestions
    
    Raises
    ------
    ValueError
        - If smiles_column doesn't exist in the dataset
        - If feature vectors are missing for any molecules
        - If an invalid similarity_metric is specified
    
    Examples
    --------
    Compute Tanimoto similarity for binary ECFP fingerprints:
    
        result = compute_similarity_matrix(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            feature_vectors_filename='ecfp_fingerprints_XY56ZW78.joblib',
            output_filename='tanimoto_similarity',
            explanation='Tanimoto similarity for drug-like molecules',
            similarity_metric='tanimoto'
        )
    
    Compute cosine similarity for continuous descriptors:
    
        result = compute_similarity_matrix(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            feature_vectors_filename='rdkit_descriptors_XY56ZW78.joblib',
            output_filename='cosine_similarity',
            explanation='Cosine similarity based on physicochemical descriptors',
            similarity_metric='cosine'
        )
    
    Notes
    -----
    - **Binary fingerprints**: Use tanimoto (recommended) or dice
    - **Count/continuous features**: Use cosine, euclidean, or manhattan
    - **Tanimoto**: Optimized with RDKit's C++ implementation for ~5x speedup
    - **Matrix storage**: Saved as joblib (efficient for large matrices)
    - **Symmetry**: Matrix is symmetric with diagonal = 1.0
       
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
    
    # Compute similarity matrix using helper function
    similarity_matrix = _compute_pairwise_similarity(feature_matrix, similarity_metric)
    
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


def find_k_nearest_neighbors(
    query_smiles: list[str],
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    feature_vectors_filename: str,
    k: int = 10,
    similarity_metric: str = 'tanimoto',
    exclude_self: bool = True
) -> dict:
    """
    Find k most similar molecules for one or more query molecules.
    
    Computes similarities between query molecule(s) and all molecules in the dataset on-the-fly,
    without precomputing a full similarity matrix. Returns the k nearest neighbors for each query.
    
    **Performance**: 
    - Single query on 10,000 molecules: ~1-2 seconds
    - Multiple queries: Linear scaling (Q queries = Qx1-2 seconds)
    - For <100 queries: This function is optimal
    - For >100 queries (>1% of dataset): Consider precomputing full similarity matrix instead
    
    **Note**: Not recommended for very large numbers of queries (Q > 100). For batch processing
    many molecules, consider computing a full similarity matrix once and doing lookups instead.
    
    Parameters
    ----------
    query_smiles : list of str
        List of SMILES strings for query molecule(s). Use a single-item list for one query.
    input_filename : str
        Filename of the dataset containing molecules to search.
    project_manifest_path : str
        Path to the project's manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    feature_vectors_filename : str
        Filename of the precomputed feature vectors (joblib format).
    k : int, default=10
        Number of nearest neighbors to return per query.
    similarity_metric : str, default='tanimoto'
        Similarity metric: 'tanimoto', 'dice', 'cosine', 'euclidean', or 'manhattan'.
    exclude_self : bool, default=True
        If True and query_smiles exists in dataset, exclude it from results.
    
    Returns
    -------
    dict
        Dictionary containing:
        - n_queries: Number of query molecules
        - k: Number of neighbors per query
        - n_candidates: Total molecules searched per query
        - similarity_metric: Metric used
        - results: List of dicts (one per query) with keys:
            - query_smiles: The query SMILES
            - neighbors: List of neighbor dicts with:
                - smiles: SMILES of neighbor
                - similarity: Similarity score
                - rank: Rank (1 = most similar)
            - mean_similarity: Average similarity to k neighbors
        - note: Summary with usage suggestions
    
    Raises
    ------
    ValueError
        - If any query_smiles not found in feature_vectors
        - If k is larger than number of available molecules
        - If an invalid similarity_metric is specified
    
    Examples
    --------
    Find neighbors for a single molecule:
    
        result = find_k_nearest_neighbors(
            query_smiles=['CCO'],
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            feature_vectors_filename='ecfp_fingerprints_XY56ZW78.joblib',
            k=10,
            similarity_metric='tanimoto'
        )
        
        # Access neighbors for the query
        for neighbor in result['results'][0]['neighbors']:
            print(f"Rank {neighbor['rank']}: {neighbor['smiles']} (similarity: {neighbor['similarity']:.3f})")
    
    Find neighbors for multiple molecules:
    
        result = find_k_nearest_neighbors(
            query_smiles=['CCO', 'CCC', 'c1ccccc1'],
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            feature_vectors_filename='ecfp_fingerprints_XY56ZW78.joblib',
            k=5
        )
        
        # Iterate through each query's results
        for query_result in result['results']:
            print(f"Query: {query_result['query_smiles']}")
            for neighbor in query_result['neighbors']:
                print(f"  {neighbor['rank']}. {neighbor['smiles']} ({neighbor['similarity']:.3f})")
    
    Notes
    -----
    - **Small queries (<100)**: Use this function (optimal)
    - **Large queries (>100)**: Consider precomputing full similarity matrix instead
    - For binary fingerprints, use 'tanimoto' (default, recommended)
    - All query molecules must exist in feature_vectors
    - Computes QxN similarities where Q = number of queries, N = dataset size
    
    See Also
    --------
    compute_similarity_matrix : Precompute full NxN similarity matrix for batch queries
    """
    query_smiles_list = list(query_smiles)
    n_queries = len(query_smiles_list)
    
    # Load dataset and feature vectors
    df = _load_resource(project_manifest_path, input_filename)
    feature_vectors = _load_resource(project_manifest_path, feature_vectors_filename)
    
    # Validate inputs
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    # Check all queries exist in feature_vectors
    missing_queries = [smi for smi in query_smiles_list if smi not in feature_vectors]
    if missing_queries:
        raise ValueError(
            f"Query SMILES not found in feature_vectors: {missing_queries[:5]}. "
            f"Generate fingerprints for all query + dataset molecules first."
        )
    
    # Get SMILES list from dataset
    smiles_list = df[smiles_column].tolist()
    n_molecules = len(smiles_list)
    
    # Build query feature matrix (Q × D) and dataset feature matrix (N × D)
    query_fp_matrix = np.array([feature_vectors[smi] for smi in query_smiles_list])
    dataset_fp_matrix = np.array([feature_vectors[smi] for smi in smiles_list])
    
    # Compute Q × N similarity matrix (queries vs dataset)
    if similarity_metric == 'tanimoto':
        # Use optimized RDKit implementation for Tanimoto
        from rdkit import DataStructs
        
        # Convert query fingerprints to RDKit
        query_rdkit_fps = []
        for fp_array in query_fp_matrix:
            bit_string = ''.join(map(str, fp_array.astype(int)))
            query_rdkit_fps.append(DataStructs.CreateFromBitString(bit_string))
        
        # Convert dataset fingerprints to RDKit
        dataset_rdkit_fps = []
        for fp_array in dataset_fp_matrix:
            bit_string = ''.join(map(str, fp_array.astype(int)))
            dataset_rdkit_fps.append(DataStructs.CreateFromBitString(bit_string))
        
        # Compute similarities: Q rows × N columns
        similarity_matrix = np.zeros((n_queries, n_molecules))
        for i in range(n_queries):
            sims = DataStructs.BulkTanimotoSimilarity(query_rdkit_fps[i], dataset_rdkit_fps)
            similarity_matrix[i, :] = sims
    
    elif similarity_metric == 'dice':
        # Dice coefficient: 2 * dot(A,B) / (||A||² + ||B||²)
        dot_product = query_fp_matrix @ dataset_fp_matrix.T  # Q × N
        query_norms_sq = np.sum(query_fp_matrix ** 2, axis=1)  # Q
        dataset_norms_sq = np.sum(dataset_fp_matrix ** 2, axis=1)  # N
        denominator = query_norms_sq[:, None] + dataset_norms_sq  # Q × N
        denominator = np.where(denominator == 0, 1e-10, denominator)
        similarity_matrix = 2 * dot_product / denominator
    
    elif similarity_metric == 'cosine':
        # Cosine similarity
        distance_matrix = cdist(query_fp_matrix, dataset_fp_matrix, metric='cosine')
        similarity_matrix = 1 - distance_matrix
    
    elif similarity_metric == 'euclidean':
        # Euclidean similarity
        distance_matrix = cdist(query_fp_matrix, dataset_fp_matrix, metric='euclidean')
        similarity_matrix = 1 / (1 + distance_matrix)
    
    elif similarity_metric == 'manhattan':
        # Manhattan similarity
        distance_matrix = cdist(query_fp_matrix, dataset_fp_matrix, metric='cityblock')
        similarity_matrix = 1 / (1 + distance_matrix)
    
    else:
        supported_metrics = list(_SIMILARITY_METRICS.keys())
        raise ValueError(
            f"Unsupported similarity metric: '{similarity_metric}'. "
            f"Supported metrics: {supported_metrics}"
        )
    
    # Process results for each query
    results = []
    for query_idx, query_smi in enumerate(query_smiles_list):
        # Get similarities for this query (one row of the Q × N matrix)
        query_similarities = similarity_matrix[query_idx, :]
        
        # Create list of (smiles, similarity) tuples
        candidates = list(zip(smiles_list, query_similarities))
        
        # Optionally exclude query itself
        if exclude_self:
            candidates = [(smi, sim) for smi, sim in candidates if smi != query_smi]
        
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Validate k
        n_available = len(candidates)
        if k > n_available:
            raise ValueError(
                f"Requested k={k} neighbors but only {n_available} molecules available "
                f"for query '{query_smi}' (after excluding query: exclude_self={exclude_self})."
            )
        
        # Get top k neighbors
        top_k = candidates[:k]
        
        # Format neighbors
        neighbors = [
            {
                'smiles': smi,
                'similarity': float(sim),
                'rank': i + 1
            }
            for i, (smi, sim) in enumerate(top_k)
        ]
        
        mean_similarity = float(np.mean([n['similarity'] for n in neighbors]))
        
        results.append({
            'query_smiles': query_smi,
            'neighbors': neighbors,
            'mean_similarity': mean_similarity
        })
    
    return {
        'n_queries': n_queries,
        'k': k,
        'n_candidates': n_molecules - (1 if exclude_self else 0),
        'similarity_metric': similarity_metric,
        'results': results,
        'note': (
            f"Found {k} nearest neighbors for {n_queries} query molecule(s) using {similarity_metric} metric. "
            f"Searched {n_molecules} candidate molecules. "
            f"For large numbers of queries (>100), consider precomputing a full similarity matrix instead."
        )
    }


def add_similarity_statistics_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    feature_vectors_filename: str,
    output_filename: str,
    explanation: str,
    similarity_metric: str = 'tanimoto',
    similarity_threshold: float = 0.8
) -> dict:
    """
    Add similarity statistics columns to a dataset based on pairwise molecular similarities.
    
    Computes the full similarity matrix (like compute_similarity_matrix) but instead of storing
    the matrix, calculates per-molecule statistics and adds them as new columns to the dataset.
    Useful for analyzing molecular diversity, finding outliers, or filtering by similarity criteria.
    
    **Added Columns**:
    - `mean_similarity`: Average similarity to all other molecules (excluding self)
    - `median_similarity`: Median similarity to all other molecules
    - `max_similarity`: Maximum similarity to any other molecule
    - `min_similarity`: Minimum similarity to any other molecule
    - `n_similar_above_threshold`: Count of molecules with similarity > similarity_threshold
    
    Parameters
    ----------
    input_filename : str
        Filename of the input dataset containing SMILES strings.
    project_manifest_path : str
        Path to the project's manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    feature_vectors_filename : str
        Filename of the precomputed feature vectors (joblib format).
        Must be a dictionary mapping SMILES to numpy arrays (fingerprints/descriptors).
    output_filename : str
        Base name for the output dataset (extension added automatically).
    explanation : str
        Description of what this operation represents.
    similarity_metric : str, default='tanimoto'
        Similarity metric: 'tanimoto', 'dice', 'cosine', 'euclidean', or 'manhattan'.
    similarity_threshold : float, default=0.8
        Threshold for counting similar molecules. A new column `n_similar_above_threshold`
        will count how many molecules have similarity > this value.
    
    Returns
    -------
    dict
        Dictionary containing:
        - output_filename: Stored dataset filename
        - n_molecules: Number of molecules
        - columns: List of column names in output
        - new_columns: List of newly added similarity statistic columns
        - similarity_metric: Metric used
        - similarity_threshold: Threshold used for counting similar molecules
        - overall_mean_similarity: Dataset-wide average pairwise similarity
        - overall_median_similarity: Dataset-wide median pairwise similarity
        - preview: First 5 rows as list of dicts
        - note: Summary with usage suggestions
    
    Raises
    ------
    ValueError
        - If smiles_column doesn't exist in the dataset
        - If feature vectors are missing for any molecules
        - If an invalid similarity_metric is specified
    
    Examples
    --------
    Add similarity statistics to a dataset:
    
        result = add_similarity_statistics_dataset(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            feature_vectors_filename='ecfp_fingerprints_XY56ZW78.joblib',
            output_filename='molecules_with_sim_stats',
            explanation='Added Tanimoto similarity statistics',
            similarity_metric='tanimoto',
            similarity_threshold=0.8
        )
        
        # New columns added: mean_similarity, median_similarity, max_similarity, etc.
        print(f"New columns: {result['new_columns']}")
    
    Filter for diverse molecules (low average similarity):
    
        # After adding statistics, load dataset and filter
        df = pd.read_csv(result['output_filename'])
        diverse = df[df['mean_similarity'] < 0.5]
        print(f"Found {len(diverse)} diverse molecules")
    
    Use custom similarity threshold:
    
        # Count molecules with similarity > 0.9 (very high similarity)
        result = add_similarity_statistics_dataset(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            feature_vectors_filename='ecfp_fingerprints_XY56ZW78.joblib',
            output_filename='molecules_with_sim_stats',
            explanation='High similarity threshold for near-duplicates',
            similarity_threshold=0.9
        )
    
    Notes
    -----
    - Computes full N×N similarity matrix internally (memory: O(N²))
    - For large datasets (>10,000 molecules), this may require significant memory
    - Statistics exclude self-similarity (diagonal values of 1.0)
    - Use for diversity analysis, outlier detection, or dataset characterization
    - Adjust similarity_threshold based on your similarity metric and use case
    - Common thresholds: 0.7-0.9 for finding similar molecules, 0.5-0.7 for moderate similarity
    
    See Also
    --------
    compute_similarity_matrix : Store full similarity matrix for advanced analysis
    find_k_nearest_neighbors : Find specific nearest neighbors for query molecules
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
    
    # Build feature matrix (N × D)
    feature_matrix = np.array([feature_vectors[smi] for smi in smiles_list])
    
    # Compute similarity matrix using helper function
    similarity_matrix = _compute_pairwise_similarity(feature_matrix, similarity_metric)
    
    # Create a copy of the dataframe for modifications
    df_out = df.copy()
    
    # Compute per-molecule statistics
    # For each molecule, get similarities to all other molecules (excluding self)
    mean_similarities = []
    median_similarities = []
    max_similarities = []
    min_similarities = []
    n_similar_above_threshold = []
    
    for i in range(n_molecules):
        # Get all similarities for molecule i
        row_similarities = similarity_matrix[i, :]
        
        # Exclude self-similarity (diagonal element)
        other_similarities = np.concatenate([row_similarities[:i], row_similarities[i+1:]])
        
        # Compute statistics
        mean_similarities.append(float(np.mean(other_similarities)))
        median_similarities.append(float(np.median(other_similarities)))
        max_similarities.append(float(np.max(other_similarities)))
        min_similarities.append(float(np.min(other_similarities)))
        
        # Count molecules above similarity threshold
        n_similar_above_threshold.append(int(np.sum(other_similarities > similarity_threshold)))
    
    # Add new columns to dataframe
    df_out['mean_similarity'] = mean_similarities
    df_out['median_similarity'] = median_similarities
    df_out['max_similarity'] = max_similarities
    df_out['min_similarity'] = min_similarities
    df_out['n_similar_above_threshold'] = n_similar_above_threshold
    
    # Define new columns list
    new_columns = [
        'mean_similarity', 
        'median_similarity', 
        'max_similarity', 
        'min_similarity',
        'n_similar_above_threshold'
    ]
    
    # Compute overall dataset statistics
    triu_indices = np.triu_indices(n_molecules, k=1)
    off_diagonal_values = similarity_matrix[triu_indices]
    overall_mean_similarity = float(np.mean(off_diagonal_values))
    overall_median_similarity = float(np.median(off_diagonal_values))
    
    # Store output dataset (always create new resource for traceability)
    output_filename = _store_resource(
        df_out,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_filename,
        "n_molecules": n_molecules,
        "columns": df_out.columns.tolist(),
        "new_columns": new_columns,
        "similarity_metric": similarity_metric,
        "similarity_threshold": similarity_threshold,
        "overall_mean_similarity": overall_mean_similarity,
        "overall_median_similarity": overall_median_similarity,
        "preview": df_out.head(5).to_dict('records'),
        "note": (
            f"Added {len(new_columns)} similarity statistic columns to dataset using {similarity_metric} metric. "
            f"Overall dataset mean similarity: {overall_mean_similarity:.3f}. "
            f"Threshold for n_similar_above_threshold: {similarity_threshold}. "
            f"Use these statistics for diversity analysis, outlier detection, or filtering similar molecules."
        )
    }