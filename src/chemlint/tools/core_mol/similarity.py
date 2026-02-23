import numpy as np
from scipy.spatial.distance import cdist
from typing import List
from chemlint.infrastructure.resources import _load_resource, _store_resource


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


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    
    The minimum number of single-character edits (insertions, deletions, substitutions)
    required to transform s1 into s2.
    
    Uses dynamic programming with O(min(m,n)) space complexity.
    
    Parameters
    ----------
    s1, s2 : str
        Input strings to compare.
        
    Returns
    -------
    int
        Edit distance between the strings.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    # Use rolling array to save space (only need previous row)
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _edit_distance_similarity(smiles_list: List[str]) -> np.ndarray:
    """
    Compute normalized edit distance similarity matrix for SMILES strings.
    
    Similarity = 1 - (edit_distance / max_length)
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings.
    
    Returns
    -------
    np.ndarray
        Symmetric similarity matrix (n_molecules × n_molecules), values 0-1, diagonal = 1.0.
    """
    n_molecules = len(smiles_list)
    similarity_matrix = np.zeros((n_molecules, n_molecules))
    
    # Compute pairwise edit distances
    for i in range(n_molecules):
        similarity_matrix[i, i] = 1.0  # Identical to self
        
        for j in range(i + 1, n_molecules):
            s1 = smiles_list[i]
            s2 = smiles_list[j]
            
            # Compute edit distance
            edit_dist = _levenshtein_distance(s1, s2)
            
            # Normalize by maximum possible distance (length of longer string)
            max_len = max(len(s1), len(s2))
            
            if max_len == 0:
                # Both empty strings
                normalized_sim = 1.0
            else:
                # Similarity = 1 - (normalized distance)
                normalized_sim = 1.0 - (edit_dist / max_len)
            
            # Matrix is symmetric
            similarity_matrix[i, j] = normalized_sim
            similarity_matrix[j, i] = normalized_sim
    
    return similarity_matrix


# Metric dispatch dictionary for easy extension
_SIMILARITY_METRICS = {
    'tanimoto': _tanimoto_similarity,
    'dice': _dice_similarity,
    'cosine': _cosine_similarity,
    'euclidean': _euclidean_similarity,
    'manhattan': _manhattan_similarity,
}

# String-based metrics that work directly on SMILES (no feature vectors needed)
_STRING_METRICS = {
    'edit_distance': _edit_distance_similarity,
}


def _compute_pairwise_similarity(feature_matrix: np.ndarray, similarity_metric: str) -> np.ndarray:
    """
    Compute pairwise similarity matrix from feature matrix using specified metric.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_molecules, n_features).
    similarity_metric : str
        Similarity metric: 'tanimoto', 'dice', 'cosine', 'euclidean', 'manhattan', or 'edit_distance'.
    
    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (n_molecules, n_molecules).
    
    Raises
    ------
    ValueError
        If an unsupported similarity metric is specified.
    """
    all_metrics = {**_SIMILARITY_METRICS, **_STRING_METRICS}
    
    if similarity_metric not in all_metrics:
        supported_metrics = list(all_metrics.keys())
        raise ValueError(
            f"Unsupported similarity metric: '{similarity_metric}'. "
            f"Supported metrics: {supported_metrics}"
        )
    
    # Dispatch to appropriate metric function
    similarity_func = all_metrics[similarity_metric]
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
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    project_manifest_path : str
        Path to manifest.json.
    smiles_column : str
        Column containing SMILES strings.
    feature_vectors_filename : str
        Precomputed feature vectors (joblib format, SMILES→arrays). Not required for 'edit_distance'.
    output_filename : str
        Base name for output matrix.
    explanation : str
        Description for manifest.
    similarity_metric : str, default='tanimoto'
        Metric: 'tanimoto' (binary FPs), 'dice' (binary FPs), 'cosine' (continuous),
        'euclidean' (continuous), 'manhattan' (continuous), 'edit_distance' (SMILES strings).
    
    Returns
    -------
    dict
        Contains output_filename, n_molecules, matrix_shape, similarity_metric, mean/median/min/max_similarity, note.
    
    Raises
    ------
    ValueError
        If smiles_column missing, feature vectors missing, or invalid similarity_metric.
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate inputs
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    # Get SMILES list
    smiles_list = df[smiles_column].tolist()
    n_molecules = len(smiles_list)
    
    # Check if this is a string-based metric (works directly on SMILES)
    if similarity_metric in _STRING_METRICS:
        # Edit distance works directly on SMILES strings
        similarity_matrix = _edit_distance_similarity(smiles_list)
    else:
        # Feature-based metrics need precomputed feature vectors
        feature_vectors = _load_resource(project_manifest_path, feature_vectors_filename)
        
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
    
    Parameters
    ----------
    query_smiles : list of str
        Query SMILES strings.
    input_filename : str
        Dataset filename to search.
    project_manifest_path : str
        Path to manifest.json.
    smiles_column : str
        Column containing SMILES strings.
    feature_vectors_filename : str
        Precomputed feature vectors (joblib).
    k : int, default=10
        Number of nearest neighbors per query.
    similarity_metric : str, default='tanimoto'
        Metric: 'tanimoto', 'dice', 'cosine', 'euclidean', 'manhattan'.
    exclude_self : bool, default=True
        Exclude query from results if in dataset.
    
    Returns
    -------
    dict
        Contains n_queries, k, n_candidates, similarity_metric, results (list of dicts with query_smiles,
        neighbors list, mean_similarity), note.
    
    Raises
    ------
    ValueError
        If query_smiles missing from feature_vectors, k too large, or invalid similarity_metric.
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
    
    Adds columns: mean_similarity, median_similarity, max_similarity, min_similarity, n_similar_above_threshold.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    project_manifest_path : str
        Path to manifest.json.
    smiles_column : str
        Column containing SMILES strings.
    feature_vectors_filename : str
        Precomputed feature vectors (joblib).
    output_filename : str
        Base name for output dataset.
    explanation : str
        Description for manifest.
    similarity_metric : str, default='tanimoto'
        Metric: 'tanimoto', 'dice', 'cosine', 'euclidean', 'manhattan'.
    similarity_threshold : float, default=0.8
        Threshold for counting similar molecules.
    
    Returns
    -------
    dict
        Contains output_filename, n_molecules, columns, new_columns, similarity_metric, similarity_threshold,
        overall_mean/median_similarity, preview, note.
    
    Raises
    ------
    ValueError
        If smiles_column missing, feature vectors missing, or invalid similarity_metric.
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


def add_training_set_similarity_statistics(
    test_input_filename: str,
    train_input_filename: str,
    project_manifest_path: str,
    test_smiles_column: str,
    train_smiles_column: str,
    test_feature_vectors_filename: str,
    train_feature_vectors_filename: str,
    output_filename: str,
    explanation: str,
    similarity_metric: str = 'tanimoto',
    k_nearest: int = 5
) -> dict:
    """
    Add similarity statistics to test molecules based on comparison to a training set.
    
    Useful for applicability domain analysis and novelty detection.
    Adds columns: mean/median/max/min_similarity_to_train, mean_top_k_similarity_to_train,
    nearest_train_smiles, nearest_train_similarity.
    
    Parameters
    ----------
    test_input_filename : str
        Test dataset filename.
    train_input_filename : str
        Training dataset filename.
    project_manifest_path : str
        Path to manifest.json.
    test_smiles_column : str
        SMILES column in test dataset.
    train_smiles_column : str
        SMILES column in training dataset.
    test_feature_vectors_filename : str
        Test feature vectors (joblib).
    train_feature_vectors_filename : str
        Training feature vectors (joblib).
    output_filename : str
        Base name for output test dataset.
    explanation : str
        Description for manifest.
    similarity_metric : str, default='tanimoto'
        Metric: 'tanimoto', 'dice', 'cosine', 'euclidean', 'manhattan'.
    k_nearest : int, default=5
        Number of nearest training neighbors for mean_top_k_similarity.
    
    Returns
    -------
    dict
        Contains output_filename, n_test/train_molecules, columns, new_columns, similarity_metric, k_nearest,
        test_mean/median_similarity_to_train, min/max_test_similarity_to_train, n_test_below_threshold, preview, note.
    
    Raises
    ------
    ValueError
        If SMILES columns missing, feature vectors missing, k_nearest too large, or invalid similarity_metric.
    """
    # Load datasets and feature vectors
    test_df = _load_resource(project_manifest_path, test_input_filename)
    train_df = _load_resource(project_manifest_path, train_input_filename)
    test_feature_vectors = _load_resource(project_manifest_path, test_feature_vectors_filename)
    train_feature_vectors = _load_resource(project_manifest_path, train_feature_vectors_filename)
    
    # Validate inputs
    if test_smiles_column not in test_df.columns:
        raise ValueError(f"Column '{test_smiles_column}' not found in test dataset.")
    if train_smiles_column not in train_df.columns:
        raise ValueError(f"Column '{train_smiles_column}' not found in training dataset.")
    
    # Get SMILES lists
    test_smiles_list = test_df[test_smiles_column].tolist()
    train_smiles_list = train_df[train_smiles_column].tolist()
    n_test = len(test_smiles_list)
    n_train = len(train_smiles_list)
    
    # Validate k_nearest
    if k_nearest > n_train:
        raise ValueError(
            f"k_nearest={k_nearest} is larger than number of training molecules ({n_train}). "
            f"Set k_nearest <= {n_train}."
        )
    
    # Validate that all SMILES have feature vectors
    missing_test_smiles = [smi for smi in test_smiles_list if smi not in test_feature_vectors]
    if missing_test_smiles:
        raise ValueError(
            f"Test feature vectors missing for {len(missing_test_smiles)} molecules. "
            f"Examples: {missing_test_smiles[:5]}. "
            f"Ensure all test SMILES have corresponding feature vectors."
        )
    
    missing_train_smiles = [smi for smi in train_smiles_list if smi not in train_feature_vectors]
    if missing_train_smiles:
        raise ValueError(
            f"Training feature vectors missing for {len(missing_train_smiles)} molecules. "
            f"Examples: {missing_train_smiles[:5]}. "
            f"Ensure all training SMILES have corresponding feature vectors."
        )
    
    # Build feature matrices
    test_feature_matrix = np.array([test_feature_vectors[smi] for smi in test_smiles_list])
    train_feature_matrix = np.array([train_feature_vectors[smi] for smi in train_smiles_list])
    
    # Compute M × N similarity matrix (test vs train)
    if similarity_metric == 'tanimoto':
        # Use optimized RDKit implementation for Tanimoto
        from rdkit import DataStructs
        
        # Convert test fingerprints to RDKit
        test_rdkit_fps = []
        for fp_array in test_feature_matrix:
            bit_string = ''.join(map(str, fp_array.astype(int)))
            test_rdkit_fps.append(DataStructs.CreateFromBitString(bit_string))
        
        # Convert training fingerprints to RDKit
        train_rdkit_fps = []
        for fp_array in train_feature_matrix:
            bit_string = ''.join(map(str, fp_array.astype(int)))
            train_rdkit_fps.append(DataStructs.CreateFromBitString(bit_string))
        
        # Compute similarities: M rows (test) × N columns (train)
        similarity_matrix = np.zeros((n_test, n_train))
        for i in range(n_test):
            sims = DataStructs.BulkTanimotoSimilarity(test_rdkit_fps[i], train_rdkit_fps)
            similarity_matrix[i, :] = sims
    
    elif similarity_metric == 'dice':
        # Dice coefficient: 2 * dot(A,B) / (||A||² + ||B||²)
        dot_product = test_feature_matrix @ train_feature_matrix.T  # M × N
        test_norms_sq = np.sum(test_feature_matrix ** 2, axis=1)  # M
        train_norms_sq = np.sum(train_feature_matrix ** 2, axis=1)  # N
        denominator = test_norms_sq[:, None] + train_norms_sq  # M × N
        denominator = np.where(denominator == 0, 1e-10, denominator)
        similarity_matrix = 2 * dot_product / denominator
    
    elif similarity_metric == 'cosine':
        # Cosine similarity
        distance_matrix = cdist(test_feature_matrix, train_feature_matrix, metric='cosine')
        similarity_matrix = 1 - distance_matrix
    
    elif similarity_metric == 'euclidean':
        # Euclidean similarity
        distance_matrix = cdist(test_feature_matrix, train_feature_matrix, metric='euclidean')
        similarity_matrix = 1 / (1 + distance_matrix)
    
    elif similarity_metric == 'manhattan':
        # Manhattan similarity
        distance_matrix = cdist(test_feature_matrix, train_feature_matrix, metric='cityblock')
        similarity_matrix = 1 / (1 + distance_matrix)
    
    else:
        supported_metrics = list(_SIMILARITY_METRICS.keys())
        raise ValueError(
            f"Unsupported similarity metric: '{similarity_metric}'. "
            f"Supported metrics: {supported_metrics}"
        )
    
    # Create a copy of the test dataframe for modifications
    test_df_out = test_df.copy()
    
    # Compute per-test-molecule statistics
    mean_similarities = []
    median_similarities = []
    max_similarities = []
    min_similarities = []
    mean_top_k_similarities = []
    nearest_train_smiles_list = []
    nearest_train_similarities = []
    
    for i in range(n_test):
        # Get all similarities from test molecule i to all training molecules
        test_to_train_sims = similarity_matrix[i, :]
        
        # Compute statistics
        mean_similarities.append(float(np.mean(test_to_train_sims)))
        median_similarities.append(float(np.median(test_to_train_sims)))
        max_similarities.append(float(np.max(test_to_train_sims)))
        min_similarities.append(float(np.min(test_to_train_sims)))
        
        # Find k nearest training neighbors
        top_k_indices = np.argsort(test_to_train_sims)[-k_nearest:][::-1]
        top_k_sims = test_to_train_sims[top_k_indices]
        mean_top_k_similarities.append(float(np.mean(top_k_sims)))
        
        # Store nearest neighbor info
        nearest_idx = np.argmax(test_to_train_sims)
        nearest_train_smiles_list.append(train_smiles_list[nearest_idx])
        nearest_train_similarities.append(float(test_to_train_sims[nearest_idx]))
    
    # Add new columns to test dataframe
    test_df_out['mean_similarity_to_train'] = mean_similarities
    test_df_out['median_similarity_to_train'] = median_similarities
    test_df_out['max_similarity_to_train'] = max_similarities
    test_df_out['min_similarity_to_train'] = min_similarities
    test_df_out[f'mean_top_{k_nearest}_similarity_to_train'] = mean_top_k_similarities
    test_df_out['nearest_train_smiles'] = nearest_train_smiles_list
    test_df_out['nearest_train_similarity'] = nearest_train_similarities
    
    # Define new columns list
    new_columns = [
        'mean_similarity_to_train',
        'median_similarity_to_train',
        'max_similarity_to_train',
        'min_similarity_to_train',
        f'mean_top_{k_nearest}_similarity_to_train',
        'nearest_train_smiles',
        'nearest_train_similarity'
    ]
    
    # Compute overall statistics
    test_mean_similarity = float(np.mean(mean_similarities))
    test_median_similarity = float(np.median(mean_similarities))
    min_test_similarity = float(np.min(max_similarities))
    max_test_similarity = float(np.max(max_similarities))
    n_test_below_threshold = int(np.sum(np.array(max_similarities) < 0.5))
    
    # Store output dataset
    output_filename = _store_resource(
        test_df_out,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_filename,
        "n_test_molecules": n_test,
        "n_train_molecules": n_train,
        "columns": test_df_out.columns.tolist(),
        "new_columns": new_columns,
        "similarity_metric": similarity_metric,
        "k_nearest": k_nearest,
        "test_mean_similarity_to_train": test_mean_similarity,
        "test_median_similarity_to_train": test_median_similarity,
        "min_test_similarity_to_train": min_test_similarity,
        "max_test_similarity_to_train": max_test_similarity,
        "n_test_below_threshold": n_test_below_threshold,
        "note": (
            f"Added {len(new_columns)} training set similarity statistics to {n_test} test molecules. "
            f"Compared against {n_train} training molecules using {similarity_metric} metric. "
            f"Average test-to-train similarity: {test_mean_similarity:.3f}. "
            f"{n_test_below_threshold} test molecules have max_similarity_to_train < 0.5 (novelty candidates). "
            f"Use max_similarity_to_train to assess applicability domain: "
            f">0.7=well-represented, 0.5-0.7=moderate, <0.5=extrapolation risk."
        )
    }
