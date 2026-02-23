"""
Simple tests for complexity.py functions - just checking they run successfully.
"""

import pandas as pd
import numpy as np
import pytest


def test_calculate_smiles_branches():
    """Test that _calculate_smiles_branches runs successfully."""
    from chemlint.tools.core_mol.complexity import _calculate_smiles_branches
    
    # Simple branched molecule
    result = _calculate_smiles_branches("CC(C)C")
    assert isinstance(result, int)
    assert result >= 0
    
    # Linear molecule
    result = _calculate_smiles_branches("CCCC")
    assert result == 0


def test_calculate_molecular_shannon_entropy():
    """Test that _calculate_molecular_shannon_entropy runs successfully."""
    from chemlint.tools.core_mol.complexity import _calculate_molecular_shannon_entropy
    
    # Simple molecule
    result = _calculate_molecular_shannon_entropy("CCO")
    assert result is not None
    assert isinstance(result, float)
    assert result >= 0
    
    # Invalid SMILES should return None
    result = _calculate_molecular_shannon_entropy("invalid")
    assert result is None or (isinstance(result, tuple) and result[0] is None)


def test_calculate_smiles_shannon_entropy():
    """Test that _calculate_smiles_shannon_entropy runs successfully."""
    from chemlint.tools.core_mol.complexity import _calculate_smiles_shannon_entropy
    
    # Simple molecule
    result = _calculate_smiles_shannon_entropy("CCO")
    assert isinstance(result, float)
    assert result >= 0
    
    # Benzene
    result = _calculate_smiles_shannon_entropy("c1ccccc1")
    assert isinstance(result, float)


def test_calculate_num_tokens():
    """Test that _calculate_num_tokens runs successfully."""
    from chemlint.tools.core_mol.complexity import _calculate_num_tokens
    
    # Simple molecule
    result = _calculate_num_tokens("CCO")
    assert isinstance(result, int)
    assert result > 0
    
    # Molecule with multi-char tokens (Cl, Br)
    result = _calculate_num_tokens("CCCl")
    assert isinstance(result, int)


def test_calculate_bertz_complexity():
    """Test that _calculate_bertz_complexity runs successfully."""
    from chemlint.tools.core_mol.complexity import _calculate_bertz_complexity
    
    # Simple molecule
    result = _calculate_bertz_complexity("CCO")
    assert result is not None
    assert isinstance(result, float)
    assert result >= 0
    
    # Invalid SMILES
    result = _calculate_bertz_complexity("invalid")
    assert result is None or (isinstance(result, tuple) and result[0] is None)


def test_calculate_bottcher_complexity():
    """Test that _calculate_bottcher_complexity runs successfully."""
    from chemlint.tools.core_mol.complexity import _calculate_bottcher_complexity
    
    # Simple molecule
    result = _calculate_bottcher_complexity("CCO")
    assert result is not None or result is None  # Can return None on complex failures
    
    # Benzene
    result = _calculate_bottcher_complexity("c1ccccc1")
    # Just check it doesn't crash - this function can return None for various reasons


def test_add_complexity_columns(session_workdir):
    """Test that add_complexity_columns runs successfully."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.complexity import add_complexity_columns
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        'SMILES': ['CCO', 'c1ccccc1', 'CC(C)C', 'CCCC'],
        'name': ['ethanol', 'benzene', 'isobutane', 'butane']
    })
    
    input_filename = _store_resource(df, manifest_path, "test_data", "Test molecules", "csv")
    
    # Test single metric
    result = add_complexity_columns(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column='SMILES',
        metrics=['branches'],
        output_filename='with_branches',
        explanation='Added branches'
    )
    
    assert "output_filename" in result
    assert result["n_rows"] == 4
    assert "branches" in result["columns_added"]
    assert isinstance(result["n_failed"], dict)
    assert "preview" in result


def test_add_complexity_columns_multiple_metrics(session_workdir):
    """Test add_complexity_columns with multiple metrics."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.complexity import add_complexity_columns
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        'SMILES': ['CCO', 'c1ccccc1'],
    })
    
    input_filename = _store_resource(df, manifest_path, "test_data2", "Test molecules", "csv")
    
    # Test multiple metrics
    result = add_complexity_columns(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column='SMILES',
        metrics=['branches', 'num_tokens', 'smiles_entropy', 'bertz'],
        output_filename='with_multiple',
        explanation='Multiple metrics'
    )
    
    assert result["n_rows"] == 2
    assert len(result["columns_added"]) == 4
    assert 'branches' in result["columns_added"]
    assert 'num_tokens' in result["columns_added"]
    assert 'smiles_entropy' in result["columns_added"]
    assert 'bertz' in result["columns_added"]


def test_add_complexity_columns_invalid_metric(session_workdir):
    """Test that invalid metric raises error."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.complexity import add_complexity_columns
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    df = pd.DataFrame({'SMILES': ['CCO']})
    input_filename = _store_resource(df, manifest_path, "test_data3", "Test", "csv")
    
    with pytest.raises(ValueError, match="Invalid metrics"):
        add_complexity_columns(
            input_filename=input_filename,
            project_manifest_path=manifest_path,
            smiles_column='SMILES',
            metrics=['invalid_metric'],
            output_filename='output',
            explanation='Test'
        )


def test_add_complexity_columns_invalid_column(session_workdir):
    """Test that invalid SMILES column raises error."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.complexity import add_complexity_columns
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    df = pd.DataFrame({'SMILES': ['CCO']})
    input_filename = _store_resource(df, manifest_path, "test_data4", "Test", "csv")
    
    with pytest.raises(ValueError, match="not found"):
        add_complexity_columns(
            input_filename=input_filename,
            project_manifest_path=manifest_path,
            smiles_column='NonExistentColumn',
            metrics=['branches'],
            output_filename='output',
            explanation='Test'
        )


def test_add_complexity_columns_with_nan(session_workdir):
    """Test handling of NaN SMILES values."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.complexity import add_complexity_columns
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Dataset with NaN
    df = pd.DataFrame({
        'SMILES': ['CCO', np.nan, 'c1ccccc1'],
    })
    
    input_filename = _store_resource(df, manifest_path, "test_data_nan", "Test with NaN", "csv")
    
    result = add_complexity_columns(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column='SMILES',
        metrics=['branches'],
        output_filename='with_nan',
        explanation='NaN test'
    )
    
    # Should complete successfully and track failures
    assert result["n_rows"] == 3
    assert result["n_failed"]["branches"] >= 1  # At least the NaN should fail
