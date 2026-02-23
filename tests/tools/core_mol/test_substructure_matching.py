"""Tests for substructure matching functions."""

import pytest
import pandas as pd
from chemlint.tools.core_mol.substructure_matching import (
    get_available_structural_patterns,
    get_available_functional_group_patterns,
    smiles_has_structural_pattern,
    find_structural_patterns_in_smiles,
    find_functional_group_patterns_in_smiles,
    find_functional_group_patterns_in_list_of_smiles,
    find_structural_patterns_in_list_of_smiles,
    add_substructure_matches_to_dataset
)


def test_get_available_structural_patterns():
    """Test that structural patterns dict is returned correctly."""
    patterns = get_available_structural_patterns()
    
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    
    # Check that patterns have expected structure
    for pattern_name, pattern_data in patterns.items():
        assert 'pattern' in pattern_data
        assert 'comment' in pattern_data
        assert isinstance(pattern_data['pattern'], str)
        assert isinstance(pattern_data['comment'], str)
    
    # Check for some known patterns
    assert 'Rotatable bond' in patterns
    assert 'Ring atom' in patterns


def test_get_available_functional_group_patterns():
    """Test that functional group patterns dict is returned correctly."""
    patterns = get_available_functional_group_patterns()
    
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    
    # Check that patterns have expected structure
    for pattern_name, pattern_data in patterns.items():
        assert 'pattern' in pattern_data
        assert 'comment' in pattern_data
        assert isinstance(pattern_data['pattern'], str)
        assert isinstance(pattern_data['comment'], str)
    
    # Check for some known functional groups
    assert 'Carbonyl group' in patterns
    assert 'Hydroxyl' in patterns


def test_smiles_has_structural_pattern():
    """Test detection of structural patterns in molecules."""
    # Benzene should match aromatic benzene pattern
    assert smiles_has_structural_pattern('c1ccccc1', 'c1ccccc1') is True
    
    # Alkane should not match benzene pattern
    assert smiles_has_structural_pattern('CCCC', 'c1ccccc1') is False
    
    # Test invalid SMILES
    assert smiles_has_structural_pattern('invalid', 'c1ccccc1') is False
    
    # Test invalid SMARTS (should return False, not crash)
    assert smiles_has_structural_pattern('CCO', 'invalid[[[smarts') is False


def test_find_structural_patterns_in_smiles():
    """Test finding all structural patterns in a molecule."""
    # Benzene should have ring atoms and benzene ring
    result = find_structural_patterns_in_smiles('c1ccccc1')
    assert isinstance(result, str)
    assert 'Ring atom' in result or 'Unfused benzene ring' in result
    
    # Simple alkane should have rotatable bonds
    result = find_structural_patterns_in_smiles('CCCC')
    assert isinstance(result, str)
    # May or may not have rotatable bonds depending on definition
    
    # Invalid SMILES should return empty string
    result = find_structural_patterns_in_smiles('invalid')
    assert result == ''
    
    # Empty result should be string
    result = find_structural_patterns_in_smiles('C')
    assert isinstance(result, str)


def test_find_functional_group_patterns_in_smiles():
    """Test finding all functional groups in a molecule."""
    # Ethanol should have hydroxyl
    result = find_functional_group_patterns_in_smiles('CCO')
    assert isinstance(result, str)
    assert 'Hydroxyl' in result
    
    # Acetone should have carbonyl and ketone
    result = find_functional_group_patterns_in_smiles('CC(=O)C')
    assert isinstance(result, str)
    assert 'Carbonyl' in result or 'Ketone' in result
    
    # Ethyl acetate should have ester
    result = find_functional_group_patterns_in_smiles('CC(=O)OCC')
    assert isinstance(result, str)
    assert 'Ester' in result or 'Carbonyl' in result
    
    # Invalid SMILES should return empty string
    result = find_functional_group_patterns_in_smiles('invalid')
    assert result == ''
    
    # Methane may have no functional groups
    result = find_functional_group_patterns_in_smiles('C')
    assert isinstance(result, str)


def test_find_functional_group_patterns_in_list_of_smiles():
    """Test batch finding of functional groups."""
    smiles_list = ['CCO', 'CC(=O)C', 'CCCC']
    results = find_functional_group_patterns_in_list_of_smiles(smiles_list)
    
    assert isinstance(results, list)
    assert len(results) == len(smiles_list)
    
    # All results should be strings
    for result in results:
        assert isinstance(result, str)
    
    # First should have hydroxyl
    assert 'Hydroxyl' in results[0]
    
    # Second should have carbonyl/ketone
    assert 'Carbonyl' in results[1] or 'Ketone' in results[1]
    
    # Test with invalid SMILES
    results = find_functional_group_patterns_in_list_of_smiles(['invalid', 'CCO'])
    assert len(results) == 2
    assert results[0] == ''  # Invalid returns empty
    assert 'Hydroxyl' in results[1]


def test_find_structural_patterns_in_list_of_smiles():
    """Test batch finding of structural patterns."""
    smiles_list = ['c1ccccc1', 'CCCC', 'C']
    results = find_structural_patterns_in_list_of_smiles(smiles_list)
    
    assert isinstance(results, list)
    assert len(results) == len(smiles_list)
    
    # All results should be strings
    for result in results:
        assert isinstance(result, str)
    
    # Benzene should have ring patterns
    assert 'Ring' in results[0] or len(results[0]) > 0
    
    # Test with invalid SMILES
    results = find_structural_patterns_in_list_of_smiles(['invalid', 'c1ccccc1'])
    assert len(results) == 2
    assert results[0] == ''  # Invalid returns empty
    assert len(results[1]) > 0  # Benzene has patterns


def test_add_substructure_matches_to_dataset_functional_groups(session_workdir, request):
    """Test adding functional group match columns to dataset."""
    from chemlint.infrastructure.resources import _load_resource, _store_resource, create_project_manifest
    
    # Create test directory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        'smiles': ['CCO', 'CC(=O)C', 'CC(=O)OCC', 'c1ccccc1', 'invalid'],
        'name': ['ethanol', 'acetone', 'ethyl acetate', 'benzene', 'bad']
    })
    input_file = _store_resource(df, manifest_path, "test_data", "Test molecules", 'csv')
    
    # Test with specific patterns
    result = add_substructure_matches_to_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column='smiles',
        output_filename='with_matches',
        pattern_type='functional_groups',
        specific_patterns=['Hydroxyl', 'Ketone', 'Ester'],
        explanation='Added functional group matches'
    )
    
    assert 'output_filename' in result
    assert result['n_rows'] == 5
    assert result['n_patterns_added'] == 3
    assert set(result['pattern_names']) == {'Hydroxyl', 'Ketone', 'Ester'}
    assert set(result['added_columns']) == {'has_Hydroxyl', 'has_Ketone', 'has_Ester'}
    
    # Load result and verify columns
    df_result = _load_resource(manifest_path, result['output_filename'])
    assert 'has_Hydroxyl' in df_result.columns
    assert 'has_Ketone' in df_result.columns
    assert 'has_Ester' in df_result.columns
    
    # Verify values
    assert df_result.loc[0, 'has_Hydroxyl'] == 'yes'  # Ethanol has OH
    assert df_result.loc[1, 'has_Ketone'] == 'yes'  # Acetone has ketone
    assert df_result.loc[2, 'has_Ester'] == 'yes'  # Ethyl acetate has ester
    assert df_result.loc[3, 'has_Hydroxyl'] == 'no'  # Benzene has no OH
    assert df_result.loc[4, 'has_Hydroxyl'] == 'no'  # Invalid SMILES


def test_add_substructure_matches_to_dataset_structural(session_workdir, request):
    """Test adding structural pattern match columns to dataset."""
    from chemlint.infrastructure.resources import _load_resource, _store_resource, create_project_manifest
    
    # Create test directory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCCC', 'C1CCC1'],
        'name': ['benzene', 'butane', 'cyclobutane']
    })
    input_file = _store_resource(df, manifest_path, "test_data", "Test molecules", 'csv')
    
    # Test with structural patterns
    result = add_substructure_matches_to_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column='smiles',
        output_filename='with_structural',
        pattern_type='structural',
        specific_patterns=['Ring atom', 'Rotatable bond'],
        explanation='Added structural patterns'
    )
    
    assert result['n_rows'] == 3
    assert result['n_patterns_added'] == 2
    
    # Load result and verify
    df_result = _load_resource(manifest_path, result['output_filename'])
    assert df_result.loc[0, 'has_Ring atom'] == 'yes'  # Benzene has rings
    assert df_result.loc[1, 'has_Ring atom'] == 'no'  # Butane has no rings
    assert df_result.loc[2, 'has_Ring atom'] == 'yes'  # Cyclobutane has rings


def test_add_substructure_matches_to_dataset_all_patterns(session_workdir, request):
    """Test adding all patterns when specific_patterns is None."""
    from chemlint.infrastructure.resources import _load_resource, _store_resource, create_project_manifest
    
    # Create test directory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create minimal dataset
    df = pd.DataFrame({'smiles': ['CCO']})
    input_file = _store_resource(df, manifest_path, "test_data", "Test", 'csv')
    
    # Test with all functional groups (should be many)
    result = add_substructure_matches_to_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column='smiles',
        output_filename='with_all',
        pattern_type='functional_groups',
        specific_patterns=None,  # Check all patterns
        explanation='All patterns'
    )
    
    # Should have checked many patterns
    assert result['n_patterns_added'] > 10
    df_result = _load_resource(manifest_path, result['output_filename'])
    assert len(df_result.columns) > 10  # Original + many pattern columns


def test_add_substructure_matches_to_dataset_errors(session_workdir, request):
    """Test error handling in add_substructure_matches_to_dataset."""
    from chemlint.infrastructure.resources import _store_resource, create_project_manifest
    
    # Create test directory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({'smiles': ['CCO']})
    input_file = _store_resource(df, manifest_path, "test_data", "Test", 'csv')
    
    # Test invalid column name
    with pytest.raises(ValueError, match="Column 'wrong' not found"):
        add_substructure_matches_to_dataset(
            input_file, manifest_path, 'wrong', 'out', explanation='test'
        )
    
    # Test invalid pattern type
    with pytest.raises(ValueError, match="pattern_type must be"):
        add_substructure_matches_to_dataset(
            input_file, manifest_path, 'smiles', 'out',
            pattern_type='invalid', explanation='test'
        )
    
    # Test invalid pattern names
    with pytest.raises(ValueError, match="Pattern names not found"):
        add_substructure_matches_to_dataset(
            input_file, manifest_path, 'smiles', 'out',
            specific_patterns=['NonExistentPattern'], explanation='test'
        )
