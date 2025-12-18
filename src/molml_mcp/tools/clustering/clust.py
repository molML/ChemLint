# Clustering tools for molecular datasets

from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.tools.core_mol.similarity import compute_similarity_matrix
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import json
from kneed import KneeLocator
from scipy.sparse import csgraph
from scipy.linalg import eigh


def eigenvalue_cluster_approx(x: np.ndarray) -> int:
    """ We estimate the number of clusters we need for spectral clustering by using the Eigenvalues of the
    Laplacian. The Eigenvalues give you a nice curve from which we determine the elbow with the kneedle algorithm [1].

    [1] Satopaa, V. et al. (2011). Finding a" kneedle" in a haystack: Detecting knee points in system behavior. In
        2011 31st international conference on distributed computing systems workshops (pp. 166-171). IEEE.

    :param x: Similarity/affinity matrix (Tanimoto similarity matrix). Sqaure matrix
    :return: number of clusters
    """

    # Compute the Laplacian matrix. Make sure to compute the symmetrically normalized Laplacian to get a nice L-shape
    laplacian = csgraph.laplacian(x, normed=True)

    # Perform Eigen-decomposition to get the Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = eigh(laplacian)

    # Estimate the 'elbow'/'knee' of the curve using the kneedle algorithm.
    kn = KneeLocator(range(len(eigenvalues)), eigenvalues,
                     curve='concave', direction='increasing',
                     interp_method='interp1d', )

    n_clusters = kn.knee

    return n_clusters


def cluster_dbscan_on_similarity(
    input_filename: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "DBSCAN clustering on similarity matrix",
    similarity_matrix_filename: str | None = None,
    feature_vectors_filename: str | None = None,
    smiles_column: str = "smiles",
    eps: float = 0.3,
    min_samples: int = 5,
    similarity_metric: str = "tanimoto",
    cluster_column_name: str = "cluster"
) -> dict:
    """
    Perform DBSCAN clustering on molecules using similarity matrix.
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups
    molecules based on similarity, identifying dense regions as clusters and 
    marking outliers as noise (cluster -1).
    
    Automatically uses precomputed similarity matrix if available, otherwise
    computes it on-the-fly from feature vectors.
    
    Args:
        input_filename: Input dataset filename
        project_manifest_path: Path to manifest.json
        output_filename: Output dataset name (with cluster assignments)
        explanation: Description of clustering operation
        similarity_matrix_filename: Optional precomputed similarity matrix
                                   (if None, will be computed from feature_vectors)
        feature_vectors_filename: Required if similarity_matrix_filename is None
                                 (fingerprints/descriptors for computing similarity)
        smiles_column: Column name containing SMILES strings (default: "smiles")
        eps: Maximum distance (1-similarity) for neighborhood
             - Lower values = smaller, tighter clusters
             - For Tanimoto: 0.3 means 0.7 similarity threshold
             - Range: 0.0 to 1.0 (default: 0.3)
        min_samples: Minimum molecules to form a dense region (cluster)
                    - Higher values = fewer, larger clusters
                    - Lower values = more, smaller clusters
                    - Default: 5
        similarity_metric: Metric for computing similarity if needed
                          ('tanimoto', 'dice', 'cosine', etc.)
        cluster_column_name: Name for cluster assignment column (default: "cluster")
    
    Returns:
        dict with:
            - output_filename: Output dataset with clusters
            - n_rows: Number of molecules
            - n_clusters: Number of clusters found (excluding noise)
            - n_noise: Number of noise points (cluster -1)
            - cluster_sizes: Dictionary of cluster sizes
            - largest_cluster: Size of largest cluster
            - silhouette_score: Clustering quality metric (-1 to 1, higher is better)
            - used_precomputed_matrix: Whether precomputed matrix was used
            - eps: Distance threshold used
            - min_samples: Minimum samples parameter used
    
    Examples:
        # With precomputed similarity matrix
        result = cluster_dbscan_on_similarity(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            similarity_matrix_filename="similarity_E5F6G7H8.joblib",
            eps=0.3,
            min_samples=5
        )
        
        # Compute similarity on-the-fly
        result = cluster_dbscan_on_similarity(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            feature_vectors_filename="morgan_fps_E5F6G7H8.joblib",
            eps=0.25,
            min_samples=3
        )
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found. Available: {list(df.columns)}")
    
    # Get or compute similarity matrix
    used_precomputed = False
    if similarity_matrix_filename is not None:
        # Use precomputed matrix
        similarity_matrix = _load_resource(project_manifest_path, similarity_matrix_filename)
        used_precomputed = True
    else:
        # Compute on-the-fly
        if feature_vectors_filename is None:
            raise ValueError(
                "Either similarity_matrix_filename or feature_vectors_filename must be provided"
            )
        
        # Compute similarity matrix
        sim_result = compute_similarity_matrix(
            input_filename=input_filename,
            project_manifest_path=project_manifest_path,
            smiles_column=smiles_column,
            feature_vectors_filename=feature_vectors_filename,
            output_filename=f"temp_similarity_for_{output_filename}",
            explanation=f"Temporary similarity matrix for DBSCAN clustering",
            similarity_metric=similarity_metric
        )
        
        # Load the computed matrix
        similarity_matrix = _load_resource(project_manifest_path, sim_result['output_filename'])
    
    # Convert similarity to distance matrix (DBSCAN uses distances)
    distance_matrix = 1.0 - similarity_matrix
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed'
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Add cluster assignments to dataframe
    df[cluster_column_name] = cluster_labels
    
    # Calculate statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    # Cluster sizes (excluding noise)
    cluster_sizes = {}
    for label in set(cluster_labels):
        if label != -1:
            cluster_sizes[f"cluster_{label}"] = int(np.sum(cluster_labels == label))
    
    largest_cluster = max(cluster_sizes.values()) if cluster_sizes else 0
    
    # Calculate silhouette score (only if we have at least 2 clusters)
    silhouette = None
    if n_clusters >= 2 and n_noise < n_total:
        from sklearn.metrics import silhouette_score
        # Silhouette score only on non-noise points
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 0:
            try:
                silhouette = float(silhouette_score(
                    distance_matrix[non_noise_mask][:, non_noise_mask],
                    cluster_labels[non_noise_mask],
                    metric='precomputed'
                ))
            except:
                silhouette = None
    
    # Store output
    output_file = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_file,
        "n_rows": n_total,
        "columns": list(df.columns),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_percentage": f"{n_noise/n_total*100:.1f}%",
        "cluster_sizes": cluster_sizes,
        "largest_cluster": largest_cluster,
        "silhouette_score": silhouette,
        "used_precomputed_matrix": used_precomputed,
        "eps": eps,
        "min_samples": min_samples,
        "similarity_metric": similarity_metric if not used_precomputed else "precomputed",
        "note": (
            f"DBSCAN clustering: {n_total} molecules → {n_clusters} clusters, {n_noise} noise points. "
            f"Largest cluster: {largest_cluster} molecules. "
            f"Eps={eps} (similarity threshold={1-eps:.2f}), min_samples={min_samples}. "
            + (f"Silhouette score: {silhouette:.3f}." if silhouette else "")
        )
    }


def cluster_hierarchical_on_similarity(
    input_filename: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Hierarchical clustering on similarity matrix",
    similarity_matrix_filename: str | None = None,
    feature_vectors_filename: str | None = None,
    smiles_column: str = "smiles",
    n_clusters: int = 5,
    linkage_method: str = "average",
    similarity_metric: str = "tanimoto",
    cluster_column_name: str = "cluster"
) -> dict:
    """
    Perform hierarchical clustering on molecules using similarity matrix.
    
    Hierarchical clustering builds a tree (dendrogram) of nested clusters by
    iteratively merging or splitting groups based on similarity. Results in
    a fixed number of clusters without noise points (all molecules assigned).
    
    Automatically uses precomputed similarity matrix if available, otherwise
    computes it on-the-fly from feature vectors.
    
    Args:
        input_filename: Input dataset filename
        project_manifest_path: Path to manifest.json
        output_filename: Output dataset name (with cluster assignments)
        explanation: Description of clustering operation
        similarity_matrix_filename: Optional precomputed similarity matrix
                                   (if None, will be computed from feature_vectors)
        feature_vectors_filename: Required if similarity_matrix_filename is None
                                 (fingerprints/descriptors for computing similarity)
        smiles_column: Column name containing SMILES strings (default: "smiles")
        n_clusters: Number of clusters to create (default: 5)
                   - Must be between 1 and n_molecules
                   - All molecules will be assigned to a cluster
        linkage_method: How to measure distance between clusters
                       - 'average': Average distance between all pairs (RECOMMENDED)
                       - 'complete': Maximum distance between any pair (tight clusters)
                       - 'single': Minimum distance between any pair (loose clusters)
                       - 'ward': Minimize within-cluster variance (requires Euclidean)
                       - Default: 'average'
        similarity_metric: Metric for computing similarity if needed
                          ('tanimoto', 'dice', 'cosine', etc.)
        cluster_column_name: Name for cluster assignment column (default: "cluster")
    
    Returns:
        dict with:
            - output_filename: Output dataset with clusters
            - n_rows: Number of molecules
            - n_clusters: Number of clusters created
            - cluster_sizes: Dictionary of cluster sizes
            - largest_cluster: Size of largest cluster
            - smallest_cluster: Size of smallest cluster
            - silhouette_score: Clustering quality metric (-1 to 1, higher is better)
            - used_precomputed_matrix: Whether precomputed matrix was used
            - linkage_method: Linkage method used
    
    Examples:
        # With precomputed similarity matrix
        result = cluster_hierarchical_on_similarity(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            similarity_matrix_filename="similarity_E5F6G7H8.joblib",
            n_clusters=10,
            linkage_method="average"
        )
        
        # Compute similarity on-the-fly with tight clusters
        result = cluster_hierarchical_on_similarity(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            feature_vectors_filename="morgan_fps_E5F6G7H8.joblib",
            n_clusters=8,
            linkage_method="complete"
        )
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found. Available: {list(df.columns)}")
    
    if n_clusters < 1 or n_clusters > n_total:
        raise ValueError(f"n_clusters must be between 1 and {n_total}, got {n_clusters}")
    
    # Get or compute similarity matrix
    used_precomputed = False
    if similarity_matrix_filename is not None:
        # Use precomputed matrix
        similarity_matrix = _load_resource(project_manifest_path, similarity_matrix_filename)
        used_precomputed = True
    else:
        # Compute on-the-fly
        if feature_vectors_filename is None:
            raise ValueError(
                "Either similarity_matrix_filename or feature_vectors_filename must be provided"
            )
        
        # Compute similarity matrix
        sim_result = compute_similarity_matrix(
            input_filename=input_filename,
            project_manifest_path=project_manifest_path,
            smiles_column=smiles_column,
            feature_vectors_filename=feature_vectors_filename,
            output_filename=f"temp_similarity_for_{output_filename}",
            explanation=f"Temporary similarity matrix for hierarchical clustering",
            similarity_metric=similarity_metric
        )
        
        # Load the computed matrix
        similarity_matrix = _load_resource(project_manifest_path, sim_result['output_filename'])
    
    # Convert similarity to distance matrix
    distance_matrix = 1.0 - similarity_matrix
    
    # Convert distance matrix to condensed form (upper triangle)
    from scipy.spatial.distance import squareform
    condensed_distances = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method=linkage_method)
    
    # Cut dendrogram to get n_clusters
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Convert to 0-indexed (scipy returns 1-indexed)
    cluster_labels = cluster_labels - 1
    
    # Add cluster assignments to dataframe
    df[cluster_column_name] = cluster_labels
    
    # Calculate statistics
    cluster_sizes = {}
    for label in set(cluster_labels):
        cluster_sizes[f"cluster_{label}"] = int(np.sum(cluster_labels == label))
    
    largest_cluster = max(cluster_sizes.values())
    smallest_cluster = min(cluster_sizes.values())
    
    # Calculate silhouette score (only if we have at least 2 clusters)
    silhouette = None
    if n_clusters >= 2:
        try:
            silhouette = float(silhouette_score(
                distance_matrix,
                cluster_labels,
                metric='precomputed'
            ))
        except:
            silhouette = None
    
    # Store output
    output_file = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_file,
        "n_rows": n_total,
        "columns": list(df.columns),
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "largest_cluster": largest_cluster,
        "smallest_cluster": smallest_cluster,
        "balance_ratio": f"{smallest_cluster/largest_cluster:.2f}",
        "silhouette_score": silhouette,
        "used_precomputed_matrix": used_precomputed,
        "linkage_method": linkage_method,
        "similarity_metric": similarity_metric if not used_precomputed else "precomputed",
        "note": (
            f"Hierarchical clustering ({linkage_method} linkage): {n_total} molecules → {n_clusters} clusters. "
            f"Largest: {largest_cluster}, smallest: {smallest_cluster} molecules. "
            + (f"Silhouette score: {silhouette:.3f}." if silhouette else "")
        )
    }


def cluster_spectral_on_similarity(
    input_filename: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Spectral clustering on similarity matrix",
    similarity_matrix_filename: str | None = None,
    feature_vectors_filename: str | None = None,
    smiles_column: str = "smiles",
    n_clusters: int | None = None,
    auto_estimate_clusters: bool = True,
    assign_labels: str = "kmeans",
    similarity_metric: str = "tanimoto",
    cluster_column_name: str = "cluster"
) -> dict:
    """
    Perform spectral clustering on molecules using similarity matrix.
    
    Spectral clustering uses the eigenvalues of the graph Laplacian to identify
    clusters in the similarity network. Works well for non-convex cluster shapes
    and when clusters have different densities.
    
    Can automatically estimate optimal number of clusters using the eigenvalue
    elbow method (Kneedle algorithm), or use a specified number.
    
    Automatically uses precomputed similarity matrix if available, otherwise
    computes it on-the-fly from feature vectors.
    
    Args:
        input_filename: Input dataset filename
        project_manifest_path: Path to manifest.json
        output_filename: Output dataset name (with cluster assignments)
        explanation: Description of clustering operation
        similarity_matrix_filename: Optional precomputed similarity matrix
                                   (if None, will be computed from feature_vectors)
        feature_vectors_filename: Required if similarity_matrix_filename is None
                                 (fingerprints/descriptors for computing similarity)
        smiles_column: Column name containing SMILES strings (default: "smiles")
        n_clusters: Number of clusters to create (optional)
                   - If None and auto_estimate_clusters=True, will be estimated
                   - If provided, overrides auto-estimation
        auto_estimate_clusters: Automatically estimate n_clusters from eigenvalues
                               using Kneedle algorithm (default: True)
                               - Ignored if n_clusters is explicitly provided
        assign_labels: Strategy for assigning labels from eigenvectors
                      - 'kmeans': k-means on eigenvectors (default, more stable)
                      - 'discretize': discretization (faster but less stable)
        similarity_metric: Metric for computing similarity if needed
                          ('tanimoto', 'dice', 'cosine', etc.)
        cluster_column_name: Name for cluster assignment column (default: "cluster")
    
    Returns:
        dict with:
            - output_filename: Output dataset with clusters
            - n_rows: Number of molecules
            - n_clusters: Number of clusters created
            - n_clusters_estimated: Whether n_clusters was auto-estimated
            - cluster_sizes: Dictionary of cluster sizes
            - largest_cluster: Size of largest cluster
            - smallest_cluster: Size of smallest cluster
            - silhouette_score: Clustering quality metric (-1 to 1, higher is better)
            - used_precomputed_matrix: Whether precomputed matrix was used
            - assign_labels: Label assignment method used
    
    Examples:
        # Auto-estimate clusters from eigenvalues
        result = cluster_spectral_on_similarity(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            similarity_matrix_filename="similarity_E5F6G7H8.joblib",
            auto_estimate_clusters=True
        )
        
        # Specify exact number of clusters
        result = cluster_spectral_on_similarity(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            feature_vectors_filename="morgan_fps_E5F6G7H8.joblib",
            n_clusters=8,
            auto_estimate_clusters=False
        )
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found. Available: {list(df.columns)}")
    
    # Get or compute similarity matrix
    used_precomputed = False
    if similarity_matrix_filename is not None:
        # Use precomputed matrix
        similarity_matrix = _load_resource(project_manifest_path, similarity_matrix_filename)
        used_precomputed = True
    else:
        # Compute on-the-fly
        if feature_vectors_filename is None:
            raise ValueError(
                "Either similarity_matrix_filename or feature_vectors_filename must be provided"
            )
        
        # Compute similarity matrix
        sim_result = compute_similarity_matrix(
            input_filename=input_filename,
            project_manifest_path=project_manifest_path,
            smiles_column=smiles_column,
            feature_vectors_filename=feature_vectors_filename,
            output_filename=f"temp_similarity_for_{output_filename}",
            explanation=f"Temporary similarity matrix for spectral clustering",
            similarity_metric=similarity_metric
        )
        
        # Load the computed matrix
        similarity_matrix = _load_resource(project_manifest_path, sim_result['output_filename'])
    
    # Estimate or validate n_clusters
    n_clusters_estimated = False
    if n_clusters is None:
        if auto_estimate_clusters:
            # Estimate number of clusters from eigenvalues
            n_clusters = eigenvalue_cluster_approx(similarity_matrix)
            n_clusters_estimated = True
            if n_clusters is None or n_clusters < 2:
                # Fallback if estimation fails
                n_clusters = min(10, max(2, n_total // 20))
        else:
            raise ValueError("n_clusters must be provided if auto_estimate_clusters=False")
    
    if n_clusters < 2 or n_clusters > n_total:
        raise ValueError(f"n_clusters must be between 2 and {n_total}, got {n_clusters}")
    
    # Perform spectral clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels=assign_labels,
        random_state=42
    )
    
    cluster_labels = spectral.fit_predict(similarity_matrix)
    
    # Add cluster assignments to dataframe
    df[cluster_column_name] = cluster_labels
    
    # Calculate statistics
    cluster_sizes = {}
    for label in set(cluster_labels):
        cluster_sizes[f"cluster_{label}"] = int(np.sum(cluster_labels == label))
    
    largest_cluster = max(cluster_sizes.values())
    smallest_cluster = min(cluster_sizes.values())
    
    # Calculate silhouette score (only if we have at least 2 clusters)
    silhouette = None
    if n_clusters >= 2:
        try:
            # Use similarity matrix directly (not distance)
            distance_matrix = 1.0 - similarity_matrix
            silhouette = float(silhouette_score(
                distance_matrix,
                cluster_labels,
                metric='precomputed'
            ))
        except:
            silhouette = None
    
    # Store output
    output_file = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_file,
        "n_rows": n_total,
        "columns": list(df.columns),
        "n_clusters": n_clusters,
        "n_clusters_estimated": n_clusters_estimated,
        "cluster_sizes": cluster_sizes,
        "largest_cluster": largest_cluster,
        "smallest_cluster": smallest_cluster,
        "balance_ratio": f"{smallest_cluster/largest_cluster:.2f}",
        "silhouette_score": silhouette,
        "used_precomputed_matrix": used_precomputed,
        "assign_labels": assign_labels,
        "similarity_metric": similarity_metric if not used_precomputed else "precomputed",
        "note": (
            f"Spectral clustering ({assign_labels} assignment): {n_total} molecules → {n_clusters} clusters"
            + (" (auto-estimated)" if n_clusters_estimated else "")
            + f". Largest: {largest_cluster}, smallest: {smallest_cluster} molecules. "
            + (f"Silhouette score: {silhouette:.3f}." if silhouette else "")
        )
    }


def cluster_kmeans_on_features(
    input_filename: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "K-means clustering on feature vectors",
    feature_vectors_filename: str | None = None,
    smiles_column: str = "smiles",
    n_clusters: int = 5,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    cluster_column_name: str = "cluster"
) -> dict:
    """
    Perform k-means clustering on molecules using feature vectors.
    
    K-means clustering partitions molecules into k clusters by minimizing
    within-cluster variance. Works directly on feature vectors (fingerprints
    or descriptors) rather than similarity matrices.
    
    Fast and scalable for large datasets, but assumes spherical clusters
    and requires specifying the number of clusters in advance.
    
    Uses precomputed feature vectors if available, otherwise computes them
    on-the-fly using Morgan fingerprints.
    
    Args:
        input_filename: Input dataset filename
        project_manifest_path: Path to manifest.json
        output_filename: Output dataset name (with cluster assignments)
        explanation: Description of clustering operation
        feature_vectors_filename: Optional precomputed feature vectors
                                 (fingerprints/descriptors as numpy array)
                                 - If None, will compute Morgan fingerprints
        smiles_column: Column name containing SMILES strings (default: "smiles")
        n_clusters: Number of clusters to create (default: 5)
                   - Must be between 2 and n_molecules
                   - All molecules will be assigned to a cluster
        n_init: Number of times k-means runs with different centroid seeds
               - Higher values = more stable results but slower
               - Default: 10
        max_iter: Maximum iterations for convergence (default: 300)
        random_state: Random seed for reproducibility (default: 42)
        cluster_column_name: Name for cluster assignment column (default: "cluster")
    
    Returns:
        dict with:
            - output_filename: Output dataset with clusters
            - n_rows: Number of molecules
            - n_clusters: Number of clusters created
            - cluster_sizes: Dictionary of cluster sizes
            - largest_cluster: Size of largest cluster
            - smallest_cluster: Size of smallest cluster
            - silhouette_score: Clustering quality metric (-1 to 1, higher is better)
            - inertia: Sum of squared distances to nearest cluster center (lower is better)
            - n_iterations: Number of iterations until convergence
            - used_precomputed_features: Whether precomputed features were used
    
    Examples:
        # With precomputed feature vectors
        result = cluster_kmeans_on_features(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            feature_vectors_filename="morgan_fps_E5F6G7H8.joblib",
            n_clusters=10
        )
        
        # Compute Morgan fingerprints on-the-fly
        result = cluster_kmeans_on_features(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            n_clusters=8,
            n_init=20  # More runs for stability
        )
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found. Available: {list(df.columns)}")
    
    if n_clusters < 2 or n_clusters > n_total:
        raise ValueError(f"n_clusters must be between 2 and {n_total}, got {n_clusters}")
    
    # Get or compute feature vectors
    used_precomputed = False
    if feature_vectors_filename is not None:
        # Use precomputed feature vectors
        feature_vectors = _load_resource(project_manifest_path, feature_vectors_filename)
        used_precomputed = True
    else:
        # Compute Morgan fingerprints on-the-fly
        from molml_mcp.tools.featurization.supported.ecfps import smiles_to_ecfp_dataset
        
        fp_result = smiles_to_ecfp_dataset(
            input_filename=input_filename,
            project_manifest_path=project_manifest_path,
            smiles_column=smiles_column,
            output_filename=f"temp_fps_for_{output_filename}",
            explanation=f"Temporary Morgan fingerprints for k-means clustering",
            radius=2,
            nbits=2048
        )
        
        # Load the computed fingerprints
        feature_vectors = _load_resource(project_manifest_path, fp_result['output_filename'])
    
    # Convert dictionary to matrix (if needed)
    if isinstance(feature_vectors, dict):
        # Feature vectors are stored as {smiles: array}
        # Convert to matrix in same order as dataframe
        smiles_list = df[smiles_column].tolist()
        feature_vectors = np.array([feature_vectors[smi] for smi in smiles_list])
    
    # Ensure feature vectors are 2D
    if len(feature_vectors.shape) == 1:
        feature_vectors = feature_vectors.reshape(-1, 1)
    
    if feature_vectors.shape[0] != n_total:
        raise ValueError(
            f"Feature vector count ({feature_vectors.shape[0]}) doesn't match "
            f"dataset size ({n_total})"
        )
    
    # Perform k-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    
    cluster_labels = kmeans.fit_predict(feature_vectors)
    
    # Add cluster assignments to dataframe
    df[cluster_column_name] = cluster_labels
    
    # Calculate statistics
    cluster_sizes = {}
    for label in set(cluster_labels):
        cluster_sizes[f"cluster_{label}"] = int(np.sum(cluster_labels == label))
    
    largest_cluster = max(cluster_sizes.values())
    smallest_cluster = min(cluster_sizes.values())
    
    # Calculate silhouette score
    silhouette = None
    if n_clusters >= 2:
        try:
            silhouette = float(silhouette_score(
                feature_vectors,
                cluster_labels,
                metric='euclidean'
            ))
        except:
            silhouette = None
    
    # Get inertia (sum of squared distances to centers)
    inertia = float(kmeans.inertia_)
    
    # Get number of iterations
    n_iterations = int(kmeans.n_iter_)
    
    # Store output
    output_file = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_file,
        "n_rows": n_total,
        "columns": list(df.columns),
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "largest_cluster": largest_cluster,
        "smallest_cluster": smallest_cluster,
        "balance_ratio": f"{smallest_cluster/largest_cluster:.2f}",
        "silhouette_score": silhouette,
        "inertia": inertia,
        "n_iterations": n_iterations,
        "converged": n_iterations < max_iter,
        "used_precomputed_features": used_precomputed,
        "feature_dim": feature_vectors.shape[1],
        "note": (
            f"K-means clustering: {n_total} molecules → {n_clusters} clusters. "
            f"Largest: {largest_cluster}, smallest: {smallest_cluster} molecules. "
            f"Converged in {n_iterations} iterations. "
            + (f"Silhouette score: {silhouette:.3f}. " if silhouette else "")
            + f"Inertia: {inertia:.2f}."
        )
    }


def cluster_butina_on_similarity(
    input_filename: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Butina clustering on similarity matrix",
    similarity_matrix_filename: str | None = None,
    feature_vectors_filename: str | None = None,
    smiles_column: str = "smiles",
    distance_threshold: float = 0.35,
    similarity_metric: str = "tanimoto",
    cluster_column_name: str = "cluster"
) -> dict:
    """
    Perform Butina clustering on molecules using similarity matrix.
    
    Butina clustering is a deterministic, greedy algorithm that creates 
    non-overlapping spherical clusters. It iteratively:
    1. Finds the molecule with the most neighbors within distance_threshold
    2. Forms a cluster with that molecule and all its neighbors
    3. Removes those molecules from consideration
    4. Repeats until all molecules are assigned
    
    Results are deterministic (same input always produces same output) and
    all molecules are assigned to exactly one cluster. Particularly useful
    for diverse compound selection in drug discovery.
    
    Automatically uses precomputed similarity matrix if available, otherwise
    computes it on-the-fly from feature vectors.
    
    Args:
        input_filename: Input dataset filename
        project_manifest_path: Path to manifest.json
        output_filename: Output dataset name (with cluster assignments)
        explanation: Description of clustering operation
        similarity_matrix_filename: Optional precomputed similarity matrix
                                   (if None, will be computed from feature_vectors)
        feature_vectors_filename: Required if similarity_matrix_filename is None
                                 (fingerprints/descriptors for computing similarity)
        smiles_column: Column name containing SMILES strings (default: "smiles")
        distance_threshold: Maximum distance (1-similarity) for neighbors
                           - Lower values = smaller, more clusters
                           - For Tanimoto: 0.35 means 0.65 similarity threshold
                           - Range: 0.0 to 1.0 (default: 0.35)
        similarity_metric: Metric for computing similarity if needed
                          ('tanimoto', 'dice', 'cosine', etc.)
        cluster_column_name: Name for cluster assignment column (default: "cluster")
    
    Returns:
        dict with:
            - output_filename: Output dataset with clusters
            - n_rows: Number of molecules
            - n_clusters: Number of clusters found
            - cluster_sizes: Dictionary of cluster sizes
            - largest_cluster: Size of largest cluster
            - smallest_cluster: Size of smallest cluster
            - singleton_clusters: Number of single-molecule clusters
            - silhouette_score: Clustering quality metric (-1 to 1, higher is better)
            - used_precomputed_matrix: Whether precomputed matrix was used
            - distance_threshold: Distance threshold used
    
    Examples:
        # With precomputed similarity matrix
        result = cluster_butina_on_similarity(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            similarity_matrix_filename="similarity_E5F6G7H8.joblib",
            distance_threshold=0.3
        )
        
        # Compute similarity on-the-fly with tighter threshold
        result = cluster_butina_on_similarity(
            input_filename="molecules_A1B2C3D4.csv",
            project_manifest_path="/path/to/manifest.json",
            output_filename="molecules_clustered",
            feature_vectors_filename="morgan_fps_E5F6G7H8.joblib",
            distance_threshold=0.25
        )
    """
    from sklearn.metrics import silhouette_score
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found. Available: {list(df.columns)}")
    
    # Get or compute similarity matrix
    used_precomputed = False
    if similarity_matrix_filename is not None:
        # Use precomputed matrix
        similarity_matrix = _load_resource(project_manifest_path, similarity_matrix_filename)
        used_precomputed = True
    else:
        # Compute on-the-fly
        if feature_vectors_filename is None:
            raise ValueError(
                "Either similarity_matrix_filename or feature_vectors_filename must be provided"
            )
        
        # Compute similarity matrix
        sim_result = compute_similarity_matrix(
            input_filename=input_filename,
            project_manifest_path=project_manifest_path,
            smiles_column=smiles_column,
            feature_vectors_filename=feature_vectors_filename,
            output_filename=f"temp_similarity_for_{output_filename}",
            explanation=f"Temporary similarity matrix for Butina clustering",
            similarity_metric=similarity_metric
        )
        
        # Load the computed matrix
        similarity_matrix = _load_resource(project_manifest_path, sim_result['output_filename'])
    
    # Convert similarity to distance matrix
    distance_matrix = 1.0 - similarity_matrix
    
    # Butina clustering algorithm
    n_mols = distance_matrix.shape[0]
    
    # Track which molecules are still available for clustering
    available = set(range(n_mols))
    
    # Store cluster assignments (-1 means unassigned)
    cluster_labels = np.full(n_mols, -1, dtype=int)
    
    cluster_id = 0
    
    while available:
        # For each available molecule, count neighbors within threshold
        neighbor_counts = []
        for mol_idx in available:
            # Count how many available molecules are within distance threshold
            neighbors = [
                j for j in available 
                if distance_matrix[mol_idx, j] <= distance_threshold
            ]
            neighbor_counts.append((len(neighbors), mol_idx, neighbors))
        
        # Sort by neighbor count (descending) to get molecule with most neighbors
        neighbor_counts.sort(reverse=True, key=lambda x: x[0])
        
        # Create cluster with the molecule that has most neighbors
        _, center_mol, neighbors = neighbor_counts[0]
        
        # Assign cluster ID to center and all its neighbors
        for mol_idx in neighbors:
            cluster_labels[mol_idx] = cluster_id
            available.remove(mol_idx)
        
        cluster_id += 1
    
    # Add cluster assignments to dataframe
    df[cluster_column_name] = cluster_labels
    
    # Calculate statistics
    n_clusters = len(set(cluster_labels))
    
    cluster_sizes = {}
    for label in set(cluster_labels):
        cluster_sizes[f"cluster_{label}"] = int(np.sum(cluster_labels == label))
    
    largest_cluster = max(cluster_sizes.values())
    smallest_cluster = min(cluster_sizes.values())
    singleton_clusters = sum(1 for size in cluster_sizes.values() if size == 1)
    
    # Calculate silhouette score (only if we have at least 2 clusters)
    silhouette = None
    if n_clusters >= 2:
        try:
            silhouette = float(silhouette_score(
                distance_matrix,
                cluster_labels,
                metric='precomputed'
            ))
        except:
            silhouette = None
    
    # Store output
    output_file = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_file,
        "n_rows": n_total,
        "columns": list(df.columns),
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "largest_cluster": largest_cluster,
        "smallest_cluster": smallest_cluster,
        "singleton_clusters": singleton_clusters,
        "balance_ratio": f"{smallest_cluster/largest_cluster:.2f}",
        "silhouette_score": silhouette,
        "used_precomputed_matrix": used_precomputed,
        "distance_threshold": distance_threshold,
        "similarity_threshold": f"{1.0 - distance_threshold:.2f}",
        "similarity_metric": similarity_metric if not used_precomputed else "precomputed",
        "note": (
            f"Butina clustering: {n_total} molecules → {n_clusters} clusters. "
            f"Largest: {largest_cluster}, smallest: {smallest_cluster} molecules. "
            f"Singletons: {singleton_clusters}. "
            f"Distance threshold={distance_threshold} (similarity ≥ {1.0-distance_threshold:.2f}). "
            + (f"Silhouette score: {silhouette:.3f}." if silhouette else "")
        )
    }


def get_all_clustering_tools():
    """Return all clustering tools."""
    return [
        cluster_dbscan_on_similarity,
        cluster_hierarchical_on_similarity,
        cluster_spectral_on_similarity,
        cluster_kmeans_on_features,
        cluster_butina_on_similarity
    ]
