"""Tests for clustering functions."""

import pytest
import numpy as np
import pandas as pd
from chemlint.tools.clustering.clust import (
    eigenvalue_cluster_approx,
    cluster_dbscan_on_similarity,
    cluster_hierarchical_on_similarity,
    cluster_spectral_on_similarity,
    cluster_kmeans_on_features,
    cluster_butina_on_similarity
)
from chemlint.infrastructure.resources import _store_resource, read_project_manifest

# Cache for sample data files within a test session
_sample_data_cache = {}


def create_sample_dataset_and_features(session_workdir, test_name):
    """Helper to create sample dataset with SMILES and feature vectors.
    
    Checks if files are already created to avoid redundant creation.
    """
    # Create test-specific subdirectory
    test_dir = session_workdir / test_name
    test_dir.mkdir(exist_ok=True)
    
    manifest_path = test_dir / "test_manifest.json"
    cache_key = str(test_dir)
    
    # Check if we already created these files
    if cache_key in _sample_data_cache:
        return _sample_data_cache[cache_key]
    
    # Check if files exist in manifest
    try:
        manifest = read_project_manifest(str(manifest_path))
        existing_dataset = None
        existing_features = None
        
        for resource in manifest.get('resources', []):
            if resource['filename'].startswith('test_dataset_'):
                existing_dataset = resource['filename']
            elif resource['filename'].startswith('test_features_'):
                existing_features = resource['filename']
        
        # If both exist, return them
        if existing_dataset and existing_features:
            _sample_data_cache[cache_key] = (existing_dataset, existing_features, manifest_path)
            return existing_dataset, existing_features, manifest_path
    except FileNotFoundError:
        # Manifest doesn't exist yet, create it
        from chemlint.infrastructure.resources import create_project_manifest
        create_project_manifest(str(test_dir), "test")
    
    # Create sample dataset with SMILES
    df = pd.DataFrame({
        'smiles': [
            'CCO',           # ethanol
            'CC(C)O',        # isopropanol
            'CCCO',          # propanol
            'c1ccccc1',      # benzene
            'c1ccccc1C',     # toluene
            'c1ccccc1CC',    # ethylbenzene
        ],
        'id': [1, 2, 3, 4, 5, 6]
    })
    
    # Store dataset
    dataset_file = _store_resource(
        df, 
        str(manifest_path), 
        "test_dataset", 
        "Test dataset for clustering",
        'csv'
    )
    
    # Create simple feature vectors (mock fingerprints)
    # Make each one slightly different so k-means can form distinct clusters
    feature_vectors = np.array([
        [1, 0, 1, 0, 0, 0],  # ethanol
        [1, 0, 1, 0.1, 0, 0],  # isopropanol (similar to ethanol)
        [1, 0, 1, 0, 0.1, 0],  # propanol (similar to ethanol)
        [0, 1, 0, 1, 1, 0],  # benzene
        [0, 1, 0, 1, 1, 0.1],  # toluene (similar to benzene)
        [0, 1, 0.1, 1, 1, 0],  # ethylbenzene (similar to benzene)
    ], dtype=np.float32)
    
    # Store feature vectors
    features_file = _store_resource(
        feature_vectors,
        str(manifest_path),
        "test_features",
        "Test feature vectors for clustering",
        'model'  # Use joblib format
    )
    
    # Cache the result
    _sample_data_cache[cache_key] = (dataset_file, features_file, manifest_path)
    
    return dataset_file, features_file, manifest_path


def test_eigenvalue_cluster_approx():
    """Test eigenvalue-based cluster number estimation."""
    # Create a simple similarity matrix with clear structure
    # 2 groups of similar items
    sim_matrix = np.array([
        [1.0, 0.9, 0.8, 0.2, 0.1],
        [0.9, 1.0, 0.85, 0.15, 0.1],
        [0.8, 0.85, 1.0, 0.2, 0.15],
        [0.2, 0.15, 0.2, 1.0, 0.9],
        [0.1, 0.1, 0.15, 0.9, 1.0]
    ])
    
    n_clusters = eigenvalue_cluster_approx(sim_matrix)
    
    # Should return a reasonable number (int or None)
    assert n_clusters is None or isinstance(n_clusters, (int, np.integer))
    if n_clusters is not None:
        assert n_clusters >= 1
        assert n_clusters <= len(sim_matrix)


def test_cluster_dbscan_on_similarity(session_workdir, request):
    """Test DBSCAN clustering runs and produces reasonable output."""
    dataset_file, features_file, manifest_path = create_sample_dataset_and_features(session_workdir, request.node.name)
    
    result = cluster_dbscan_on_similarity(
        input_filename=dataset_file,
        project_manifest_path=str(manifest_path),
        output_filename="dbscan_test",
        feature_vectors_filename=features_file,
        similarity_metric="edit_distance",  # Use edit_distance (doesn't require feature dict)
        eps=0.4,
        min_samples=2
    )
    
    # Check result structure
    assert 'output_filename' in result
    assert 'n_clusters' in result
    assert 'n_noise' in result
    assert isinstance(result['n_clusters'], (int, np.integer))
    assert isinstance(result['n_noise'], (int, np.integer))
    assert result['n_clusters'] >= 0
    assert result['n_noise'] >= 0
    assert result['n_rows'] > 0


def test_cluster_hierarchical_on_similarity(session_workdir, request):
    """Test hierarchical clustering runs and produces reasonable output."""
    dataset_file, features_file, manifest_path = create_sample_dataset_and_features(session_workdir, request.node.name)
    
    result = cluster_hierarchical_on_similarity(
        input_filename=dataset_file,
        project_manifest_path=str(manifest_path),
        output_filename="hierarchical_test",
        feature_vectors_filename=features_file,
        similarity_metric="edit_distance",  # Use edit_distance (doesn't require feature dict)
        n_clusters=3
    )
    
    # Check result structure
    assert 'output_filename' in result
    assert 'n_clusters' in result
    assert result['n_clusters'] == 3
    assert 'cluster_sizes' in result
    assert len(result['cluster_sizes']) == 3
    assert 'largest_cluster' in result
    assert 'smallest_cluster' in result


def test_cluster_spectral_on_similarity(session_workdir, request):
    """Test spectral clustering runs and produces reasonable output."""
    dataset_file, features_file, manifest_path = create_sample_dataset_and_features(session_workdir, request.node.name)
    
    result = cluster_spectral_on_similarity(
        input_filename=dataset_file,
        project_manifest_path=str(manifest_path),
        output_filename="spectral_test",
        feature_vectors_filename=features_file,
        similarity_metric="edit_distance",  # Use edit_distance (doesn't require feature dict)
        n_clusters=3
    )
    
    # Check result structure
    assert 'output_filename' in result
    assert 'n_clusters' in result
    assert result['n_clusters'] == 3
    assert 'cluster_sizes' in result
    assert 'n_clusters_estimated' in result
    assert result['n_clusters_estimated'] is False  # We specified n_clusters


def test_cluster_kmeans_on_features(session_workdir, request):
    """Test k-means clustering runs and produces reasonable output."""
    dataset_file, features_file, manifest_path = create_sample_dataset_and_features(session_workdir, request.node.name)
    
    result = cluster_kmeans_on_features(
        input_filename=dataset_file,
        project_manifest_path=str(manifest_path),
        output_filename="kmeans_test",
        feature_vectors_filename=features_file,
        n_clusters=3
    )
    
    # Check result structure
    assert 'output_filename' in result
    assert 'n_clusters' in result
    assert result['n_clusters'] == 3
    assert 'inertia' in result
    assert 'n_iterations' in result
    assert isinstance(result['inertia'], float)
    assert isinstance(result['n_iterations'], int)


def test_cluster_butina_on_similarity(session_workdir, request):
    """Test Butina clustering runs and produces reasonable output."""
    dataset_file, features_file, manifest_path = create_sample_dataset_and_features(session_workdir, request.node.name)
    
    result = cluster_butina_on_similarity(
        input_filename=dataset_file,
        project_manifest_path=str(manifest_path),
        output_filename="butina_test",
        feature_vectors_filename=features_file,
        similarity_metric="edit_distance",  # Use edit_distance (doesn't require feature dict)
        distance_threshold=0.4
    )
    
    # Check result structure
    assert 'output_filename' in result
    assert 'n_clusters' in result
    assert isinstance(result['n_clusters'], int)
    assert result['n_clusters'] > 0
    assert 'singleton_clusters' in result
    assert 'distance_threshold' in result
    assert result['distance_threshold'] == 0.4
