import pytest
import numpy as np
import pandas as pd
from molml_mcp.tools.core_mol.similarity import (
    _levenshtein_distance,
    _edit_distance_similarity,
    _tanimoto_similarity,
    _dice_similarity,
    _cosine_similarity,
    _euclidean_similarity,
    _manhattan_similarity,
    compute_similarity_matrix,
    find_k_nearest_neighbors,
    add_similarity_statistics_dataset,
)
from molml_mcp.infrastructure.resources import _store_resource, _load_resource


def test_levenshtein_distance():
    """Test Levenshtein distance calculations."""
    # Identical strings
    assert _levenshtein_distance("CCO", "CCO") == 0
    
    # Single edits
    assert _levenshtein_distance("CCO", "CCC") == 1  # Substitution
    assert _levenshtein_distance("CCO", "CCCO") == 1  # Insertion
    assert _levenshtein_distance("CCCO", "CCO") == 1  # Deletion
    assert _levenshtein_distance("c1ccccc1", "c1cccnc1") == 1
    
    # Complex differences
    assert _levenshtein_distance("CCO", "c1ccccc1") == 8
    
    # Empty strings
    assert _levenshtein_distance("", "") == 0
    assert _levenshtein_distance("", "CCO") == 3
    assert _levenshtein_distance("CCO", "") == 3


def test_edit_distance_similarity():
    """Test edit distance similarity matrix calculations."""
    # Identical molecules
    result = _edit_distance_similarity(["CCO", "CCO"])
    assert result.shape == (2, 2)
    assert result[0, 1] == 1.0
    np.testing.assert_allclose(np.diag(result), 1.0)
    
    # Different molecules
    result = _edit_distance_similarity(["CCO", "CCCO", "c1ccccc1"])
    assert result.shape == (3, 3)
    assert abs(result[0, 1] - 0.75) < 0.01  # CCO vs CCCO
    assert result[0, 2] < 0.5  # CCO vs benzene
    np.testing.assert_allclose(result, result.T)  # Symmetry
    
    # Edge case: single molecule
    result = _edit_distance_similarity(["CCO"])
    assert result.shape == (1, 1)
    assert result[0, 0] == 1.0


def test_binary_fingerprint_similarities():
    """Test Tanimoto and Dice similarity for binary fingerprints."""
    # Identical fingerprints
    fp1 = np.array([1, 0, 1, 1, 0, 1])
    fp2 = np.array([1, 0, 1, 1, 0, 1])
    feature_matrix = np.array([fp1, fp2])
    
    result = _tanimoto_similarity(feature_matrix)
    assert result.shape == (2, 2)
    assert result[0, 1] == 1.0
    np.testing.assert_allclose(np.diag(result), 1.0)
    
    # Different fingerprints with known overlap
    fp1 = np.array([1, 1, 1, 0, 0])  # 3 bits set
    fp2 = np.array([1, 1, 0, 1, 0])  # 3 bits set, 2 in common
    feature_matrix = np.array([fp1, fp2])
    
    # Tanimoto = 2 / (3 + 3 - 2) = 0.5
    result = _tanimoto_similarity(feature_matrix)
    assert abs(result[0, 1] - 0.5) < 0.01
    np.testing.assert_allclose(result, result.T)
    
    # Dice = 2*2 / (3 + 3) = 0.667
    result = _dice_similarity(feature_matrix)
    assert abs(result[0, 1] - 0.667) < 0.01
    np.testing.assert_allclose(np.diag(result), 1.0)
    np.testing.assert_allclose(result, result.T)


def test_continuous_feature_similarities():
    """Test cosine, Euclidean, and Manhattan similarity for continuous features."""
    # Cosine similarity - parallel vectors
    fp1 = np.array([1.0, 2.0, 3.0])
    fp2 = np.array([2.0, 4.0, 6.0])  # Scaled version
    fp3 = np.array([1.0, 0.0, 0.0])
    feature_matrix = np.array([fp1, fp2, fp3])
    
    result = _cosine_similarity(feature_matrix)
    assert result.shape == (3, 3)
    assert abs(result[0, 1] - 1.0) < 0.01  # Parallel vectors
    np.testing.assert_allclose(np.diag(result), 1.0)
    np.testing.assert_allclose(result, result.T)
    
    # Euclidean similarity
    fp1 = np.array([1.0, 1.0])
    fp2 = np.array([1.0, 1.0])  # Identical
    fp3 = np.array([5.0, 5.0])  # Far away
    feature_matrix = np.array([fp1, fp2, fp3])
    
    result = _euclidean_similarity(feature_matrix)
    assert result[0, 1] == 1.0  # Identical
    assert result[0, 2] < 0.2  # Distant
    np.testing.assert_allclose(result, result.T)
    
    # Manhattan similarity
    fp1 = np.array([1.0, 1.0])
    fp2 = np.array([1.0, 1.0])  # Identical
    fp3 = np.array([4.0, 4.0])  # Distance = 6
    feature_matrix = np.array([fp1, fp2, fp3])
    
    result = _manhattan_similarity(feature_matrix)
    assert result[0, 1] == 1.0  # Identical
    assert abs(result[0, 2] - 0.143) < 0.01  # 1/(1+6)
    np.testing.assert_allclose(result, result.T)


# ============================================================================
# Main Function Tests
# ============================================================================


def test_compute_similarity_matrix(session_workdir):
    """Test compute_similarity_matrix with different metrics and error cases."""
    manifest_path = session_workdir / "test_manifest.json"
    
    # Test Tanimoto metric
    df = pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC', 'c1ccccc1'],
        'label': [1, 0, 1, 0]
    })
    input_filename = _store_resource(df, str(manifest_path), 'test_smiles', 'Test SMILES', 'csv')
    
    fingerprints = {
        'CCO': np.array([1, 0, 1, 1, 0]),
        'CCC': np.array([1, 1, 0, 1, 0]),
        'CCCC': np.array([1, 1, 1, 0, 0]),
        'c1ccccc1': np.array([0, 0, 1, 0, 1])
    }
    fp_filename = _store_resource(fingerprints, str(manifest_path), 'test_fps', 'Test fingerprints', 'joblib')
    
    result = compute_similarity_matrix(
        input_filename=input_filename,
        project_manifest_path=str(manifest_path),
        smiles_column='smiles',
        feature_vectors_filename=fp_filename,
        output_filename='similarity_matrix',
        explanation='Test Tanimoto similarity',
        similarity_metric='tanimoto'
    )
    
    assert result['n_molecules'] == 4
    assert result['matrix_shape'] == (4, 4)
    assert result['similarity_metric'] == 'tanimoto'
    assert 0.0 <= result['mean_similarity'] <= 1.0
    
    sim_matrix = _load_resource(str(manifest_path), result['output_filename'])
    np.testing.assert_allclose(np.diag(sim_matrix), 1.0)
    np.testing.assert_allclose(sim_matrix, sim_matrix.T)
    
    # Test edit distance (no feature vectors needed)
    df2 = pd.DataFrame({'smiles': ['CCO', 'CCCO', 'CCCCO']})
    input_filename2 = _store_resource(df2, str(manifest_path), 'test_smiles2', 'Test SMILES 2', 'csv')
    
    result = compute_similarity_matrix(
        input_filename=input_filename2,
        project_manifest_path=str(manifest_path),
        smiles_column='smiles',
        feature_vectors_filename='',
        output_filename='edit_distance_matrix',
        explanation='Test edit distance',
        similarity_metric='edit_distance'
    )
    
    sim_matrix = _load_resource(str(manifest_path), result['output_filename'])
    assert abs(sim_matrix[0, 1] - 0.75) < 0.01  # CCO vs CCCO
    assert abs(sim_matrix[0, 2] - 0.6) < 0.01  # CCO vs CCCCO
    
    # Test error cases
    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        compute_similarity_matrix(
            input_filename=input_filename,
            project_manifest_path=str(manifest_path),
            smiles_column='smiles',
            feature_vectors_filename=fp_filename,
            output_filename='invalid',
            explanation='Invalid metric',
            similarity_metric='invalid_metric'
        )
    
    # Missing feature vectors
    df3 = pd.DataFrame({'smiles': ['CCO', 'CCC', 'CCCC']})
    input_filename3 = _store_resource(df3, str(manifest_path), 'test_smiles3', 'Test SMILES 3', 'csv')
    
    partial_fps = {'CCO': np.array([1, 0, 1]), 'CCC': np.array([1, 1, 0])}
    partial_fp_filename = _store_resource(partial_fps, str(manifest_path), 'partial_fps', 'Partial FPs', 'joblib')
    
    with pytest.raises(ValueError, match="Feature vectors missing"):
        compute_similarity_matrix(
            input_filename=input_filename3,
            project_manifest_path=str(manifest_path),
            smiles_column='smiles',
            feature_vectors_filename=partial_fp_filename,
            output_filename='missing',
            explanation='Missing vectors',
            similarity_metric='tanimoto'
        )


def test_find_k_nearest_neighbors(session_workdir):
    """Test find_k_nearest_neighbors with various scenarios."""
    manifest_path = session_workdir / "test_manifest.json"
    
    # Single query test
    df = pd.DataFrame({'smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'c1ccccc1']})
    input_filename = _store_resource(df, str(manifest_path), 'test_smiles', 'Test SMILES', 'csv')
    
    fingerprints = {
        'CCO': np.array([1, 0, 1, 0, 0]),
        'CCC': np.array([1, 1, 0, 0, 0]),
        'CCCC': np.array([1, 1, 1, 0, 0]),
        'CCCCC': np.array([1, 1, 1, 1, 0]),
        'c1ccccc1': np.array([0, 0, 0, 0, 1])
    }
    fp_filename = _store_resource(fingerprints, str(manifest_path), 'test_fps', 'Test FPs', 'joblib')
    
    result = find_k_nearest_neighbors(
        query_smiles=['CCO'],
        input_filename=input_filename,
        project_manifest_path=str(manifest_path),
        smiles_column='smiles',
        feature_vectors_filename=fp_filename,
        k=3,
        similarity_metric='tanimoto',
        exclude_self=True
    )
    
    assert result['n_queries'] == 1
    assert result['k'] == 3
    assert result['n_candidates'] == 4
    assert len(result['results'][0]['neighbors']) == 3
    
    # Check sorting by similarity
    similarities = [n['similarity'] for n in result['results'][0]['neighbors']]
    assert similarities == sorted(similarities, reverse=True)
    
    # Check ranks
    ranks = [n['rank'] for n in result['results'][0]['neighbors']]
    assert ranks == [1, 2, 3]
    
    # Multiple queries test
    df2 = pd.DataFrame({'smiles': ['CCO', 'CCC', 'CCCC', 'c1ccccc1', 'c1cccnc1']})
    input_filename2 = _store_resource(df2, str(manifest_path), 'test_smiles2', 'Test SMILES 2', 'csv')
    
    fingerprints2 = {
        'CCO': np.array([1, 0, 0, 0]),
        'CCC': np.array([1, 1, 0, 0]),
        'CCCC': np.array([1, 1, 1, 0]),
        'c1ccccc1': np.array([0, 0, 0, 1]),
        'c1cccnc1': np.array([0, 0, 1, 1])
    }
    fp_filename2 = _store_resource(fingerprints2, str(manifest_path), 'test_fps2', 'Test FPs 2', 'joblib')
    
    result = find_k_nearest_neighbors(
        query_smiles=['CCO', 'c1ccccc1'],
        input_filename=input_filename2,
        project_manifest_path=str(manifest_path),
        smiles_column='smiles',
        feature_vectors_filename=fp_filename2,
        k=2,
        similarity_metric='dice',
        exclude_self=True
    )
    
    assert result['n_queries'] == 2
    assert len(result['results']) == 2
    assert len(result['results'][0]['neighbors']) == 2
    assert len(result['results'][1]['neighbors']) == 2
    
    # Test exclude_self=False
    df3 = pd.DataFrame({'smiles': ['CCO', 'CCC', 'CCCC']})
    input_filename3 = _store_resource(df3, str(manifest_path), 'test_smiles3', 'Test SMILES 3', 'csv')
    
    fingerprints3 = {'CCO': np.array([1, 0, 0]), 'CCC': np.array([0, 1, 0]), 'CCCC': np.array([0, 0, 1])}
    fp_filename3 = _store_resource(fingerprints3, str(manifest_path), 'test_fps3', 'Test FPs 3', 'joblib')
    
    result = find_k_nearest_neighbors(
        query_smiles=['CCO'],
        input_filename=input_filename3,
        project_manifest_path=str(manifest_path),
        smiles_column='smiles',
        feature_vectors_filename=fp_filename3,
        k=2,
        similarity_metric='tanimoto',
        exclude_self=False
    )
    
    neighbors = result['results'][0]['neighbors']
    assert neighbors[0]['smiles'] == 'CCO'
    assert neighbors[0]['similarity'] == 1.0
    
    # Test error: k too large
    with pytest.raises(ValueError, match="Requested k=5 neighbors but only 2 molecules available"):
        find_k_nearest_neighbors(
            query_smiles=['CCO'],
            input_filename=input_filename3,
            project_manifest_path=str(manifest_path),
            smiles_column='smiles',
            feature_vectors_filename=fp_filename3,
            k=5,
            similarity_metric='tanimoto',
            exclude_self=True
        )


def test_add_similarity_statistics_dataset(session_workdir):
    """Test add_similarity_statistics_dataset with various thresholds and scenarios."""
    manifest_path = session_workdir / "test_manifest.json"
    
    # Basic functionality test
    df = pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC', 'c1ccccc1'],
        'label': [1, 0, 1, 0]
    })
    input_filename = _store_resource(df, str(manifest_path), 'test_smiles', 'Test SMILES', 'csv')
    
    fingerprints = {
        'CCO': np.array([1, 0, 1, 0]),
        'CCC': np.array([1, 1, 0, 0]),
        'CCCC': np.array([1, 1, 1, 0]),
        'c1ccccc1': np.array([0, 0, 0, 1])
    }
    fp_filename = _store_resource(fingerprints, str(manifest_path), 'test_fps', 'Test FPs', 'joblib')
    
    result = add_similarity_statistics_dataset(
        input_filename=input_filename,
        project_manifest_path=str(manifest_path),
        smiles_column='smiles',
        feature_vectors_filename=fp_filename,
        output_filename='with_sim_stats',
        explanation='Added similarity statistics',
        similarity_metric='tanimoto',
        similarity_threshold=0.5
    )
    
    assert result['n_molecules'] == 4
    assert result['similarity_metric'] == 'tanimoto'
    
    expected_new_columns = [
        'mean_similarity', 'median_similarity', 'max_similarity',
        'min_similarity', 'n_similar_above_threshold'
    ]
    assert result['new_columns'] == expected_new_columns
    
    df_out = _load_resource(str(manifest_path), result['output_filename'])
    assert df_out.shape == (4, 7)  # 2 original + 5 new
    
    for col in expected_new_columns:
        assert col in df_out.columns
    
    # Verify statistics in valid ranges
    assert all(0.0 <= v <= 1.0 for v in df_out['mean_similarity'])
    assert all(0.0 <= v <= 1.0 for v in df_out['max_similarity'])
    assert all(v >= 0 for v in df_out['n_similar_above_threshold'])
    assert 0.0 <= result['overall_mean_similarity'] <= 1.0
    
    # Test with high threshold
    df2 = pd.DataFrame({'smiles': ['CCO', 'CCCO', 'CCCCO']})
    input_filename2 = _store_resource(df2, str(manifest_path), 'test_smiles2', 'Test SMILES 2', 'csv')
    
    fingerprints2 = {
        'CCO': np.array([1, 1, 1, 0, 0]),
        'CCCO': np.array([1, 1, 1, 1, 0]),
        'CCCCO': np.array([1, 1, 1, 1, 1])
    }
    fp_filename2 = _store_resource(fingerprints2, str(manifest_path), 'test_fps2', 'Test FPs 2', 'joblib')
    
    result = add_similarity_statistics_dataset(
        input_filename=input_filename2,
        project_manifest_path=str(manifest_path),
        smiles_column='smiles',
        feature_vectors_filename=fp_filename2,
        output_filename='with_sim_stats_high',
        explanation='High threshold test',
        similarity_metric='tanimoto',
        similarity_threshold=0.9
    )
    
    df_out = _load_resource(str(manifest_path), result['output_filename'])
    assert all(v <= 2 for v in df_out['n_similar_above_threshold'])
    
    # Test preservation of original columns
    df3 = pd.DataFrame({
        'smiles': ['CCO', 'CCC'],
        'label': [1, 0],
        'property': [10.5, 20.3]
    })
    input_filename3 = _store_resource(df3, str(manifest_path), 'test_smiles3', 'Test SMILES 3', 'csv')
    
    fingerprints3 = {'CCO': np.array([1, 0, 1]), 'CCC': np.array([0, 1, 1])}
    fp_filename3 = _store_resource(fingerprints3, str(manifest_path), 'test_fps3', 'Test FPs 3', 'joblib')
    
    result = add_similarity_statistics_dataset(
        input_filename=input_filename3,
        project_manifest_path=str(manifest_path),
        smiles_column='smiles',
        feature_vectors_filename=fp_filename3,
        output_filename='with_sim_stats_preserve',
        explanation='Preserve columns test',
        similarity_metric='cosine',
        similarity_threshold=0.7
    )
    
    df_out = _load_resource(str(manifest_path), result['output_filename'])
    assert 'smiles' in df_out.columns
    assert 'label' in df_out.columns
    assert 'property' in df_out.columns
    assert list(df_out['smiles']) == ['CCO', 'CCC']
    assert list(df_out['label']) == [1, 0]
    assert list(df_out['property']) == [10.5, 20.3]
