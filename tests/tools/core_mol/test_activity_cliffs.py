"""Simple tests for activity_cliffs.py functions."""
import pandas as pd
import numpy as np
import pytest


def test_compute_fold_difference_matrix_basic():
    """Test basic fold-difference calculation using ratio method.
    
    NOTE: Current implementation uses simple ratios (max/min), not log10.
    - 10 vs 100: ratio = 10.0 (this would be 1-fold in orders of magnitude)
    - 10 vs 1000: ratio = 100.0 (this would be 2-fold in orders of magnitude)
    """
    from chemlint.tools.core_mol.activity_cliffs import _compute_fold_difference_matrix
    
    # Simple activity values: 100, 10, 50 (e.g., IC50 in nM)
    activities = np.array([100.0, 10.0, 50.0])
    
    fold_matrix = _compute_fold_difference_matrix(activities)
    
    # Check shape
    assert fold_matrix.shape == (3, 3)
    
    # Check diagonal (molecule compared to itself) is 1.0
    assert fold_matrix[0, 0] == 1.0
    assert fold_matrix[1, 1] == 1.0
    assert fold_matrix[2, 2] == 1.0
    
    # Check specific fold differences (RATIO-based, not log10)
    # 100 vs 10: ratio = 100/10 = 10.0 (1 order of magnitude)
    assert fold_matrix[0, 1] == 10.0
    assert fold_matrix[1, 0] == 10.0  # Symmetric
    
    # 100 vs 50: ratio = 100/50 = 2.0 (0.3 orders of magnitude)
    assert fold_matrix[0, 2] == 2.0
    assert fold_matrix[2, 0] == 2.0
    
    # 50 vs 10: ratio = 50/10 = 5.0 (0.7 orders of magnitude)
    assert fold_matrix[1, 2] == 5.0
    assert fold_matrix[2, 1] == 5.0


def test_compute_fold_difference_matrix_always_gte_one():
    """Test that fold-differences (ratios) are always >= 1.0."""
    from chemlint.tools.core_mol.activity_cliffs import _compute_fold_difference_matrix
    
    activities = np.array([0.5, 5.0, 50.0, 500.0])
    
    fold_matrix = _compute_fold_difference_matrix(activities)
    
    # All fold differences should be >= 1.0
    assert np.all(fold_matrix >= 1.0)


def test_compute_fold_difference_matrix_zero_handling():
    """Test handling of zero activity values."""
    from chemlint.tools.core_mol.activity_cliffs import _compute_fold_difference_matrix
    
    # Include a zero value (should result in inf)
    activities = np.array([100.0, 0.0, 10.0])
    
    fold_matrix = _compute_fold_difference_matrix(activities)
    
    # Comparisons with zero should be inf
    assert fold_matrix[0, 1] == np.inf
    assert fold_matrix[1, 0] == np.inf
    assert fold_matrix[1, 2] == np.inf
    assert fold_matrix[2, 1] == np.inf
    
    # Normal comparison should still work
    assert fold_matrix[0, 2] == 10.0


def test_annotate_activity_cliff_molecules_basic(session_workdir):
    """Test basic activity cliff annotation."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.activity_cliffs import annotate_activity_cliff_molecules
    import joblib
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create a simple dataset with activity cliffs
    # Molecules 0 and 1 are similar (high similarity) but different activities (10x fold)
    # Molecules 2 and 3 are similar but similar activities (no cliff)
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CC(O)C', 'c1ccccc1', 'c1cccc1C'],
        'IC50_nM': [100.0, 10.0, 50.0, 45.0]
    })
    
    # Create similarity matrix
    # High similarity between 0-1 (cliff) and 2-3 (no cliff)
    similarity_matrix = np.array([
        [1.0, 0.85, 0.2, 0.1],
        [0.85, 1.0, 0.1, 0.15],
        [0.2, 0.1, 1.0, 0.9],
        [0.1, 0.15, 0.9, 1.0]
    ])
    
    dataset_filename = _store_resource(df, manifest_path, "test_dataset", "Test data", "csv")
    
    # Save similarity matrix
    sim_matrix_path = session_workdir / "similarity_matrix.pkl"
    joblib.dump(similarity_matrix, sim_matrix_path)
    sim_filename = _store_resource(similarity_matrix, manifest_path, "sim_matrix", "Similarity", "model")
    
    result = annotate_activity_cliff_molecules(
        dataset_filename=dataset_filename,
        project_manifest_path=manifest_path,
        smiles_column='SMILES',
        activity_column='IC50_nM',
        similarity_matrix_filename=sim_filename,
        output_filename='annotated',
        explanation='Test annotation',
        similarity_threshold=0.8,
        fold_difference_threshold=5.0
    )
    
    # Verify result structure
    assert "output_filename" in result
    assert "n_molecules" in result
    assert "n_cliff_molecules" in result
    assert "columns" in result
    assert "summary" in result
    
    # Check counts
    assert result["n_molecules"] == 4
    
    # Verify new columns added
    assert "is_activity_cliff_molecule" in result["columns"]
    assert "n_activity_cliff_partners" in result["columns"]


def test_annotate_activity_cliff_molecules_no_cliffs(session_workdir):
    """Test when no activity cliffs exist."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.activity_cliffs import annotate_activity_cliff_molecules
    import joblib
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # All molecules have similar activities (no cliffs)
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CC(O)C', 'c1ccccc1'],
        'IC50_nM': [50.0, 55.0, 48.0]
    })
    
    # High similarity matrix
    similarity_matrix = np.array([
        [1.0, 0.9, 0.85],
        [0.9, 1.0, 0.88],
        [0.85, 0.88, 1.0]
    ])
    
    dataset_filename = _store_resource(df, manifest_path, "no_cliff_data", "No cliffs", "csv")
    sim_filename = _store_resource(similarity_matrix, manifest_path, "sim_matrix", "Similarity", "model")
    
    result = annotate_activity_cliff_molecules(
        dataset_filename=dataset_filename,
        project_manifest_path=manifest_path,
        smiles_column='SMILES',
        activity_column='IC50_nM',
        similarity_matrix_filename=sim_filename,
        output_filename='no_cliffs',
        explanation='No cliffs test',
        similarity_threshold=0.8,
        fold_difference_threshold=10.0  # 10-fold difference required
    )
    
    # Should have 0 cliff molecules (activities 48-55nM are < 10-fold different)
    assert result["n_cliff_molecules"] == 0
    assert result["n_molecules"] == 3


def test_annotate_activity_cliff_molecules_all_cliffs(session_workdir):
    """Test when all similar molecules form activity cliffs."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.activity_cliffs import annotate_activity_cliff_molecules
    import joblib
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # All molecules similar but very different activities
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CC(O)C', 'CC(C)O'],
        'IC50_nM': [1000.0, 10.0, 100.0]  # Large differences
    })
    
    # High similarity matrix (all similar)
    similarity_matrix = np.array([
        [1.0, 0.9, 0.85],
        [0.9, 1.0, 0.88],
        [0.85, 0.88, 1.0]
    ])
    
    dataset_filename = _store_resource(df, manifest_path, "all_cliff_data", "All cliffs", "csv")
    sim_filename = _store_resource(similarity_matrix, manifest_path, "sim_matrix", "Similarity", "model")
    
    result = annotate_activity_cliff_molecules(
        dataset_filename=dataset_filename,
        project_manifest_path=manifest_path,
        smiles_column='SMILES',
        activity_column='IC50_nM',
        similarity_matrix_filename=sim_filename,
        output_filename='all_cliffs',
        explanation='All cliffs test',
        similarity_threshold=0.8,
        fold_difference_threshold=5.0
    )
    
    # All molecules should participate in cliffs
    assert result["n_cliff_molecules"] == 3
    assert result["n_molecules"] == 3


def test_annotate_activity_cliff_molecules_missing_columns(session_workdir):
    """Test error handling for missing columns."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.activity_cliffs import annotate_activity_cliff_molecules
    import joblib
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    df = pd.DataFrame({
        'SMILES': ['CCO', 'c1ccccc1'],
        'IC50_nM': [100.0, 10.0]
    })
    
    similarity_matrix = np.eye(2)
    
    dataset_filename = _store_resource(df, manifest_path, "test_data", "Test", "csv")
    sim_filename = _store_resource(similarity_matrix, manifest_path, "sim_matrix", "Similarity", "model")
    
    # Test with wrong SMILES column name
    with pytest.raises(ValueError, match="SMILES column .* not found"):
        annotate_activity_cliff_molecules(
            dataset_filename=dataset_filename,
            project_manifest_path=manifest_path,
            smiles_column='WRONG_COLUMN',
            activity_column='IC50_nM',
            similarity_matrix_filename=sim_filename,
            output_filename='output',
            explanation='Test'
        )
    
    # Test with wrong activity column name
    with pytest.raises(ValueError, match="Activity column .* not found"):
        annotate_activity_cliff_molecules(
            dataset_filename=dataset_filename,
            project_manifest_path=manifest_path,
            smiles_column='SMILES',
            activity_column='WRONG_COLUMN',
            similarity_matrix_filename=sim_filename,
            output_filename='output',
            explanation='Test'
        )


def test_annotate_activity_cliff_molecules_threshold_variations(session_workdir):
    """Test different threshold combinations.
    
    NOTE: Thresholds are ratio-based, not log10-based.
    - 100 vs 20 = 5.0 ratio (0.7 orders of magnitude)
    - threshold of 3.0 = ratio of 3 (0.48 orders of magnitude)
    - threshold of 10.0 = ratio of 10 (1.0 order of magnitude)
    """
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.activity_cliffs import annotate_activity_cliff_molecules
    import joblib
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CC(O)C'],
        'IC50_nM': [100.0, 20.0]  # 5.0 ratio difference (0.7 orders of magnitude)
    })
    
    similarity_matrix = np.array([
        [1.0, 0.85],
        [0.85, 1.0]
    ])
    
    dataset_filename = _store_resource(df, manifest_path, "threshold_data", "Threshold test", "csv")
    sim_filename = _store_resource(similarity_matrix, manifest_path, "sim_matrix", "Similarity", "model")
    
    # With fold threshold = 3.0, should detect cliff (5 > 3)
    result_low = annotate_activity_cliff_molecules(
        dataset_filename=dataset_filename,
        project_manifest_path=manifest_path,
        smiles_column='SMILES',
        activity_column='IC50_nM',
        similarity_matrix_filename=sim_filename,
        output_filename='low_threshold',
        explanation='Low threshold',
        similarity_threshold=0.8,
        fold_difference_threshold=3.0
    )
    
    assert result_low["n_cliff_molecules"] == 2  # Both molecules in cliff
    
    # With fold threshold = 10.0, should NOT detect cliff (5 < 10)
    result_high = annotate_activity_cliff_molecules(
        dataset_filename=dataset_filename,
        project_manifest_path=manifest_path,
        smiles_column='SMILES',
        activity_column='IC50_nM',
        similarity_matrix_filename=sim_filename,
        output_filename='high_threshold',
        explanation='High threshold',
        similarity_threshold=0.8,
        fold_difference_threshold=10.0
    )
    
    assert result_high["n_cliff_molecules"] == 0  # No cliffs detected
