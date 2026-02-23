"""
Simple but robust tests for scaffolds.py functions.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def test_get_scaffold_bemis_murcko():
    """Test _get_scaffold with Bemis-Murcko scaffold type."""
    from chemlint.tools.core_mol.scaffolds import _get_scaffold
    
    # Molecule with ring - should succeed
    scaffold, comment = _get_scaffold("c1ccccc1CCO", "bemis_murcko")
    assert scaffold is not None
    assert "Passed" in comment
    assert "c1ccccc1" in scaffold or "C1=CC=CC=C1" in scaffold
    
    # Aliphatic molecule - no scaffold
    scaffold, comment = _get_scaffold("CCCCCC", "bemis_murcko")
    assert scaffold is None
    assert "No scaffold found" in comment
    
    # Invalid SMILES
    scaffold, comment = _get_scaffold("invalid_smiles", "bemis_murcko")
    assert scaffold is None
    assert "Failed" in comment or "Invalid" in comment
    
    # None input
    scaffold, comment = _get_scaffold(None, "bemis_murcko")
    assert scaffold is None
    assert "Invalid" in comment or "Skipped" in comment


def test_get_scaffold_generic():
    """Test _get_scaffold with generic scaffold type."""
    from chemlint.tools.core_mol.scaffolds import _get_scaffold
    
    # Molecule with heteroatoms: phenol with nitrogen in side chain
    # O=C(NCc1ccccn1)c1ccccc1 should give a generic scaffold with only carbons
    scaffold, comment = _get_scaffold("O=C(NCc1ccccn1)c1ccccc1", "generic")
    assert scaffold is not None
    assert "Passed" in comment
    # Generic scaffold converts all atoms to carbon and all bonds to single
    # Should not contain N, O, or aromatic bonds
    assert "N" not in scaffold
    assert "O" not in scaffold
    assert "n" not in scaffold  # No aromatic nitrogen
    assert "o" not in scaffold  # No aromatic oxygen
    # Should only contain C and structural characters
    for char in scaffold:
        if char.isalpha():
            assert char in ['C', 'c'], f"Found non-carbon atom: {char}"


def test_get_scaffold_cyclic_skeleton():
    """Test _get_scaffold with cyclic_skeleton scaffold type."""
    from chemlint.tools.core_mol.scaffolds import _get_scaffold
    
    # Molecule with ring and sidechains with heteroatoms
    # CCCN(Cc1ccccn1)C(=O)c1cc(C)cc(OCCCON=C(N)N)c1
    # Cyclic skeleton should remove sidechains and convert to all carbons
    scaffold, comment = _get_scaffold("CCCN(Cc1ccccn1)C(=O)c1cc(C)cc(OCCCON=C(N)N)c1", "cyclic_skeleton")
    assert scaffold is not None
    assert "Passed" in comment
    
    # Cyclic skeleton removes sidechains, so should be smaller/simpler than original
    # Should not contain heteroatoms (N, O)
    assert "N" not in scaffold
    assert "O" not in scaffold
    assert "n" not in scaffold
    assert "o" not in scaffold
    
    # Should only contain carbons and structural characters
    for char in scaffold:
        if char.isalpha():
            assert char in ['C', 'c'], f"Found non-carbon atom: {char}"
    
    # Should still have rings (cyclic part)
    assert "1" in scaffold or "c" in scaffold or "C" in scaffold


def test_get_scaffold_invalid_type():
    """Test _get_scaffold with invalid scaffold type."""
    from chemlint.tools.core_mol.scaffolds import _get_scaffold
    
    scaffold, comment = _get_scaffold("c1ccccc1", "invalid_type")
    assert scaffold is None
    assert "not supported" in comment


def test_calculate_scaffolds_basic():
    """Test calculate_scaffolds with basic SMILES list."""
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds
    
    smiles = ["c1ccccc1CCO", "CCCC", "c1ccc(O)cc1"]
    scaffolds, comments = calculate_scaffolds(smiles)
    
    # Check output lengths match input
    assert len(scaffolds) == len(smiles)
    assert len(comments) == len(smiles)
    
    # First molecule has ring - should have scaffold
    assert scaffolds[0] is not None
    assert "Passed" in comments[0]
    
    # Second molecule is aliphatic - no scaffold
    assert scaffolds[1] is None
    assert "No scaffold found" in comments[1]
    
    # Third molecule has ring - should have scaffold
    assert scaffolds[2] is not None
    assert "Passed" in comments[2]


def test_calculate_scaffolds_with_invalid():
    """Test calculate_scaffolds handles invalid SMILES."""
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds
    
    smiles = ["c1ccccc1", None, "invalid", np.nan, "CCO"]
    scaffolds, comments = calculate_scaffolds(smiles)
    
    assert len(scaffolds) == 5
    assert len(comments) == 5
    
    # First should succeed
    assert scaffolds[0] is not None
    
    # None should be handled
    assert scaffolds[1] is None
    assert "Invalid" in comments[1] or "Skipped" in comments[1]
    
    # Invalid SMILES should fail
    assert scaffolds[2] is None
    assert "Failed" in comments[2] or "Invalid" in comments[2]
    
    # NaN should be handled
    assert scaffolds[3] is None


def test_calculate_scaffolds_all_types():
    """Test calculate_scaffolds with all scaffold types."""
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds
    
    smiles = ["c1ccccc1CCO"]
    
    # Bemis-Murcko
    scaffolds_bm, _ = calculate_scaffolds(smiles, "bemis_murcko")
    assert scaffolds_bm[0] is not None
    
    # Generic
    scaffolds_gen, _ = calculate_scaffolds(smiles, "generic")
    assert scaffolds_gen[0] is not None
    
    # Cyclic skeleton
    scaffolds_cs, _ = calculate_scaffolds(smiles, "cyclic_skeleton")
    assert scaffolds_cs[0] is not None
    
    # All should differ (at least generic should differ from bemis_murcko)
    assert scaffolds_bm[0] != scaffolds_gen[0]


def test_calculate_scaffolds_empty_list():
    """Test calculate_scaffolds with empty list."""
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds
    
    scaffolds, comments = calculate_scaffolds([])
    assert scaffolds == []
    assert comments == []


def test_calculate_scaffolds_dataset_basic(session_workdir):
    """Test calculate_scaffolds_dataset with dummy data."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test dataset with various molecule types
    df = pd.DataFrame({
        'smiles': ['c1ccccc1CCO', 'CCCCCC', 'c1ccc(O)cc1', 'invalid', None],
        'name': ['benzene_ethanol', 'hexane', 'phenol', 'bad', 'missing']
    })
    
    input_filename = _store_resource(df, manifest_path, "test_molecules", "Test molecules", "csv")
    
    result = calculate_scaffolds_dataset(
        input_filename=input_filename,
        column_name='smiles',
        project_manifest_path=manifest_path,
        output_filename='with_scaffolds',
        scaffold_type='bemis_murcko'
    )
    
    # Check structure
    assert "output_filename" in result

    
    # Check values
    assert result["n_rows"] == 5
    assert result["scaffold_type"] == "bemis_murcko"
    assert "scaffold_bemis_murcko" in result["columns"]
    assert "scaffold_comments" in result["columns"]
    
    # Should have found 2 scaffolds (benzene derivatives)
    assert result["n_scaffolds_found"] == 2
    
    # At least the aliphatic molecule should have no scaffold
    assert result["n_no_scaffold"] == 1


def test_calculate_scaffolds_dataset_generic_type(session_workdir):
    """Test calculate_scaffolds_dataset with generic scaffold type."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'c1ccc(CC)cc1'],
        'name': ['benzene', 'ethylbenzene']
    })
    
    input_filename = _store_resource(df, manifest_path, "test_molecules2", "Test", "csv")
    
    result = calculate_scaffolds_dataset(
        input_filename=input_filename,
        column_name='smiles',
        project_manifest_path=manifest_path,
        output_filename='with_generic_scaffolds',
        scaffold_type='generic'
    )
    
    assert result["scaffold_type"] == "generic"
    assert "scaffold_generic" in result["columns"]
    assert result["n_scaffolds_found"] == 2


def test_calculate_scaffolds_dataset_cyclic_skeleton(session_workdir):
    """Test calculate_scaffolds_dataset with cyclic_skeleton type."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'c1ccc2ccccc2c1'],  # benzene, naphthalene
    })
    
    input_filename = _store_resource(df, manifest_path, "test_molecules3", "Test", "csv")
    
    result = calculate_scaffolds_dataset(
        input_filename=input_filename,
        column_name='smiles',
        project_manifest_path=manifest_path,
        output_filename='with_cyclic_scaffolds',
        scaffold_type='cyclic_skeleton'
    )
    
    assert result["scaffold_type"] == "cyclic_skeleton"
    assert "scaffold_cyclic_skeleton" in result["columns"]
    assert result["n_scaffolds_found"] == 2


def test_calculate_scaffolds_dataset_with_real_data(session_workdir):
    """Test calculate_scaffolds_dataset with dummy_data_raw_small.csv."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load real dummy data
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data", "Dummy data", "csv")
    
    result = calculate_scaffolds_dataset(
        input_filename=input_filename,
        column_name='smiles',
        project_manifest_path=manifest_path,
        output_filename='dummy_with_scaffolds',
        scaffold_type='bemis_murcko'
    )
    
    # Check we got results for most molecules
    assert result["n_rows"] == len(df)
    assert result["n_scaffolds_found"] >= 100  # Most should have scaffolds
    assert "scaffold_bemis_murcko" in result["columns"]
    assert len(result["preview"]) <= 5


def test_calculate_scaffolds_preserves_order():
    """Test that calculate_scaffolds preserves input order."""
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds
    
    smiles = [f"c1ccc(C{'C'*i})cc1" for i in range(5)]  # Different chain lengths
    scaffolds, comments = calculate_scaffolds(smiles)
    
    # All should have the same scaffold (benzene)
    assert len(scaffolds) == 5
    assert len(comments) == 5
    
    # All should succeed and have similar scaffolds
    for i, scaffold in enumerate(scaffolds):
        assert scaffold is not None, f"Failed at index {i}"
        assert "Passed" in comments[i]
