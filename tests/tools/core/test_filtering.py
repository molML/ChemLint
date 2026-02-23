"""Tests for filtering.py functions."""
import pandas as pd
import pytest
from pathlib import Path


def test_filter_by_property_range(session_workdir):
    """Test filtering by property ranges."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_property_range
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test data with numeric properties
    df = pd.DataFrame({
        'smiles': ['CCO', 'CCCO', 'CCCCO', 'CCCCCO'],
        'MolWt': [46.07, 60.10, 74.12, 88.15],
        'LogP': [-0.3, 0.3, 0.9, 1.5]
    })
    
    input_filename = _store_resource(
        df, manifest_path, "molecules_with_props", "Test molecules with properties", "csv"
    )
    
    # Filter by property range
    result = filter_by_property_range(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        property_ranges={'MolWt': (60, 80), 'LogP': (0, 1)},
        output_filename="filtered_molecules",
        explanation="Filter by MW and LogP"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert len(df_result) == 2  # Should keep CCCO and CCCCO
    assert set(df_result['smiles'].tolist()) == {'CCCO', 'CCCCO'}


def test_filter_by_lipinski_ro5(session_workdir):
    """Test Lipinski Rule of Five filtering."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_lipinski_ro5
    from pathlib import Path
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load real data
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(data_path).head(20)
    
    input_filename = _store_resource(
        df, manifest_path, "molecules", "Test molecules", "csv"
    )
    
    # Apply Lipinski filtering
    result = filter_by_lipinski_ro5(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="lipinski_filtered",
        explanation="Lipinski filtering"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "MolWt" in df_result.columns
    assert "MolLogP" in df_result.columns
    # Verify all molecules meet Lipinski criteria
    assert (df_result["MolWt"] <= 500).all()
    assert (df_result["MolLogP"] <= 5).all()
    assert (df_result["NumHDonors"] <= 5).all()
    assert (df_result["NumHAcceptors"] <= 10).all()
    # Verify all molecules meet Lipinski criteria
    assert (df_result["MolWt"] <= 500).all()
    assert (df_result["MolLogP"] <= 5).all()
    assert (df_result["NumHDonors"] <= 5).all()
    assert (df_result["NumHAcceptors"] <= 10).all()


def test_filter_by_veber_rules(session_workdir):
    """Test Veber rules filtering."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_veber_rules
    from pathlib import Path
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load real data
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(data_path).head(20)
    
    input_filename = _store_resource(
        df, manifest_path, "molecules", "Test molecules", "csv"
    )
    
    # Apply Veber filtering
    result = filter_by_veber_rules(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="veber_filtered",
        explanation="Veber filtering"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "TPSA" in df_result.columns
    assert "NumRotatableBonds" in df_result.columns
    # Verify all molecules meet Veber criteria
    assert (df_result["TPSA"] <= 140).all()
    assert (df_result["NumRotatableBonds"] <= 10).all()
    # Verify all molecules meet Veber criteria
    assert (df_result["TPSA"] <= 140).all()
    assert (df_result["NumRotatableBonds"] <= 10).all()


def test_filter_by_pains(session_workdir):
    """Test PAINS filtering."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_pains
    from pathlib import Path
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load dataset with problematic molecules
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_cleaning.csv"
    df = pd.read_csv(data_path)
    df.rename(columns={'SMILES': 'smiles'}, inplace=True)  # Standardize column name
    n_input = len(df)
    
    input_filename = _store_resource(
        df, manifest_path, "problematic_molecules", "Molecules with various issues", "csv"
    )
    
    # Apply PAINS filtering
    result = filter_by_pains(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="pains_filtered",
        explanation="PAINS filtering"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    # PAINS filtering should remove problematic molecules (invalid SMILES, etc)
    # The output should be less than input due to invalid SMILES and potential PAINS hits
    assert len(df_result) < n_input
    assert result["n_output"] == len(df_result)
    assert result["n_input"] == n_input


def test_filter_by_lead_likeness(session_workdir):
    """Test lead-likeness filtering."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_lead_likeness
    from pathlib import Path
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load real data
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(data_path).head(20)
    
    input_filename = _store_resource(
        df, manifest_path, "molecules", "Test molecules", "csv"
    )
    
    # Apply lead-likeness filtering
    result = filter_by_lead_likeness(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="lead_like_filtered",
        explanation="Lead-likeness filtering"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "MolWt" in df_result.columns
    # Verify all molecules meet lead-likeness criteria
    assert (df_result["MolWt"] >= 250).all() and (df_result["MolWt"] <= 350).all()
    assert (df_result["MolLogP"] <= 3.5).all()
    assert (df_result["NumRotatableBonds"] <= 7).all()


def test_filter_by_rule_of_three(session_workdir):
    """Test Rule of Three filtering."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_rule_of_three
    from pathlib import Path
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load real data
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(data_path).head(20)
    
    input_filename = _store_resource(
        df, manifest_path, "molecules", "Test molecules", "csv"
    )
    
    # Apply Rule of Three filtering
    result = filter_by_rule_of_three(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="ro3_filtered",
        explanation="Rule of Three filtering"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "MolWt" in df_result.columns
    # Verify all molecules meet Rule of Three criteria
    assert (df_result["MolWt"] <= 300).all()
    assert (df_result["MolLogP"] <= 3).all()
    assert (df_result["NumHDonors"] <= 3).all()
    assert (df_result["NumHAcceptors"] <= 3).all()
    assert (df_result["NumRotatableBonds"] <= 3).all()


def test_filter_by_qed(session_workdir):
    """Test QED filtering."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_qed
    from pathlib import Path
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load real data
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(data_path).head(20)
    
    input_filename = _store_resource(
        df, manifest_path, "molecules", "Test molecules", "csv"
    )
    
    # Apply QED filtering
    result = filter_by_qed(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        min_qed=0.5,
        output_filename="qed_filtered",
        explanation="QED filtering"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "QED" in df_result.columns
    # Verify all molecules meet QED threshold
    assert (df_result["QED"] >= 0.5).all()
    # Verify all molecules meet QED threshold
    assert (df_result["QED"] >= 0.5).all()


def test_filter_by_scaffold(session_workdir):
    """Test scaffold-based filtering."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_scaffold
    from pathlib import Path
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load real data
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(data_path).head(20)
    
    input_filename = _store_resource(
        df, manifest_path, "molecules", "Test molecules", "csv"
    )
    
    # Apply scaffold filtering (keep molecules with benzene scaffold)
    result = filter_by_scaffold(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        scaffold_smiles_list=["c1ccccc1"],
        action="keep",
        smiles_column="smiles",
        output_filename="scaffold_filtered",
        explanation="Keep benzene scaffolds"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert len(df_result) <= 20


def test_filter_by_functional_groups(session_workdir):
    """Test functional group filtering."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.filtering import filter_by_functional_groups
    from pathlib import Path
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load real data
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(data_path).head(20)
    
    input_filename = _store_resource(
        df, manifest_path, "molecules", "Test molecules", "csv"
    )
    
    # Apply functional group filtering
    result = filter_by_functional_groups(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        required=["Primary amine"],
        output_filename="fgroup_filtered",
        explanation="Keep molecules with primary amines"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert len(df_result) <= 20
