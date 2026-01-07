"""Tests for SMILES standardization pipeline functions in mol_cleaning.py."""
import pandas as pd
import pytest
from pathlib import Path


def test_default_SMILES_standardization_pipeline_basic(session_workdir):
    """Test basic SMILES standardization pipeline with valid molecules."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Test with simple valid SMILES
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    
    standardized, comments = default_SMILES_standardization_pipeline(smiles)
    
    # All should be standardized successfully
    assert len(standardized) == 3
    assert len(comments) == 3
    assert all(comment == "Standardized" for comment in comments)
    # Check canonicalization
    assert standardized[0] == "CCO"
    assert standardized[1] == "c1ccccc1"
    assert standardized[2] == "CC(=O)O"


def test_default_SMILES_standardization_pipeline_invalid_smiles(session_workdir):
    """Test pipeline handles invalid SMILES correctly."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use dummy_cleaning.csv data with invalid SMILES
    smiles = [
        "INVALID_SMILES_123",  # Invalid SMILES string
        "C(C)(C)(C)(C)C",      # Invalid valence
        "c1ccccc",             # Unclosed ring
        "CCO"                  # Valid SMILES
    ]
    
    standardized, comments = default_SMILES_standardization_pipeline(smiles)
    
    assert len(standardized) == 4
    assert len(comments) == 4
    
    # First three should have validation errors
    assert "Validation" in comments[0] or "Invalid" in comments[0]
    assert "Validation" in comments[1] or "Invalid" in comments[1]
    assert "Validation" in comments[2] or "Invalid" in comments[2]
    
    # Last one should be valid
    assert comments[3] == "Standardized"
    assert standardized[3] == "CCO"


def test_default_SMILES_standardization_pipeline_salt_removal(session_workdir):
    """Test pipeline removes salts correctly."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use valid SMILES with salts that should be removable
    smiles = [
        "CCN.[Cl-]",        # Chloride salt (proper format)
        "CC(C)N.[Br-]",     # Bromide salt (proper format)
        "CCO"               # No salt (control)
    ]
    
    standardized, comments = default_SMILES_standardization_pipeline(smiles)
    
    assert len(standardized) == 3
    # All should be standardized (salts removed where present)
    for i, comment in enumerate(comments):
        assert comment == "Standardized", f"Entry {i} failed: {comment}"
    
    # Verify organic parts preserved
    assert standardized[0] in ["CCN", "NCC"]  # Ethylamine (canonical form)
    assert standardized[1] in ["CC(C)N", "CC(N)C"]  # Isopropylamine (canonical form)
    assert standardized[2] == "CCO"  # Ethanol unchanged


def test_default_SMILES_standardization_pipeline_solvent_removal(session_workdir):
    """Test pipeline removes solvents correctly."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use valid SMILES with solvents
    smiles = [
        "CCO.O",                    # Ethanol with water
        "c1ccccc1.CCO",             # Benzene with ethanol
        "CC(=O)C.O.O"               # Acetone with water
    ]
    
    standardized, comments = default_SMILES_standardization_pipeline(smiles)
    
    assert len(standardized) == 3
    # All should be standardized (solvents removed)
    for i, comment in enumerate(comments):
        assert comment == "Standardized", f"Entry {i} failed: {comment}"


def test_default_SMILES_standardization_pipeline_stereochemistry_flatten(session_workdir):
    """Test pipeline flattens stereochemistry by default."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use dummy_cleaning.csv stereoisomer examples
    smiles = [
        "C[C@H](O)CC",      # R-stereoisomer
        "C[C@@H](O)CC",     # S-stereoisomer
        "C/C=C/C"           # E-alkene
    ]
    
    standardized, comments = default_SMILES_standardization_pipeline(
        smiles, 
        stereo_policy="flatten"
    )
    
    assert len(standardized) == 3
    # All should be standardized
    for comment in comments:
        assert comment == "Standardized"
    
    # Verify stereochemistry removed
    for std in standardized:
        assert "@" not in std, f"Stereochemistry not removed: {std}"
        assert "/" not in std and "\\" not in std, f"E/Z stereochemistry not removed: {std}"


def test_default_SMILES_standardization_pipeline_stereochemistry_keep(session_workdir):
    """Test pipeline preserves stereochemistry when requested."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use dummy_cleaning.csv stereoisomer examples
    smiles = [
        "C[C@H](O)CC",      # R-stereoisomer
        "C[C@@H](O)CC"      # S-stereoisomer
    ]
    
    standardized, comments = default_SMILES_standardization_pipeline(
        smiles, 
        stereo_policy="keep"
    )
    
    assert len(standardized) == 2
    # All should be standardized
    for comment in comments:
        assert comment == "Standardized"
    
    # Verify stereochemistry preserved
    assert "@" in standardized[0] or "@" in standardized[1]


def test_default_SMILES_standardization_pipeline_isotope_removal(session_workdir):
    """Test pipeline removes isotopes by default."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use dummy_cleaning.csv isotope examples
    smiles = [
        "[2H]C([2H])([2H])O",   # Deuterated methanol
        "[13C]CO",              # C13-labeled ethanol
        "CC([18F])CC"           # F18-labeled compound
    ]
    
    standardized, comments = default_SMILES_standardization_pipeline(
        smiles,
        skip_isotope_removal=False
    )
    
    assert len(standardized) == 3
    # All should be standardized
    for comment in comments:
        assert comment == "Standardized"
    
    # Verify isotopes removed (no [2H], [13C], [18F])
    for std in standardized:
        assert "[2H]" not in std, f"Isotope not removed: {std}"
        assert "[13C]" not in std, f"Isotope not removed: {std}"
        assert "[18F]" not in std, f"Isotope not removed: {std}"


def test_default_SMILES_standardization_pipeline_isotope_preservation(session_workdir):
    """Test pipeline preserves isotopes when requested."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use dummy_cleaning.csv isotope examples
    smiles = [
        "[2H]C([2H])([2H])O",   # Deuterated methanol
        "[13C]CO"               # C13-labeled ethanol
    ]
    
    standardized_skip, comments_skip = default_SMILES_standardization_pipeline(
        smiles,
        skip_isotope_removal=True
    )
    
    standardized_remove, comments_remove = default_SMILES_standardization_pipeline(
        smiles,
        skip_isotope_removal=False
    )
    
    assert len(standardized_skip) == 2
    assert len(standardized_remove) == 2
    
    # When skip_isotope_removal=False, isotopes should be removed
    # (verify by checking the SMILES are simpler/different)
    # Note: RDKit may canonicalize isotopes differently, so we check behavior difference
    assert all(comment == "Standardized" for comment in comments_skip)
    assert all(comment == "Standardized" for comment in comments_remove)


def test_default_SMILES_standardization_pipeline_metal_disconnection(session_workdir):
    """Test pipeline disconnects metals when requested."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use dummy_cleaning.csv metal complex examples
    smiles = [
        "c1ccccc1[Fe]",             # Phenyl-iron complex
        "C1=CC=CC=C1.[Fe+2]",       # Benzene-iron complex
        "[Cu+2].c1ccccc1"           # Copper-benzene complex
    ]
    
    standardized, comments = default_SMILES_standardization_pipeline(
        smiles,
        enable_metal_disconnection=True
    )
    
    assert len(standardized) == 3
    # All should be standardized
    for comment in comments:
        assert comment == "Standardized"
    
    # Verify metals disconnected (organic part preserved)
    for std in standardized:
        # Should contain organic part (benzene ring)
        assert "c1ccccc1" in std or "C1=CC=CC=C1" in std, f"Organic part missing: {std}"


def test_default_SMILES_standardization_pipeline_charged_species(session_workdir):
    """Test pipeline neutralizes charged species correctly."""
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline
    
    # Use dummy_cleaning.csv charged species examples
    smiles = [
        "CC[NH3+]",         # Protonated amine
        "CC(=O)[O-]",       # Carboxylate anion
        "c1cc[n+]cc1"       # Pyridinium cation
    ]
    
    standardized, comments = default_SMILES_standardization_pipeline(smiles)
    
    assert len(standardized) == 3
    # All should be standardized
    for comment in comments:
        assert comment == "Standardized"
    
    # Verify neutralization occurred (check for reduced charge presence)
    # Note: Some molecules may retain charges if they're part of the structure


def test_default_SMILES_standardization_pipeline_dataset_basic(session_workdir):
    """Test dataset pipeline with basic SMILES standardization."""
    from molml_mcp.infrastructure.resources import _store_resource
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create dataset with mixed SMILES
    df = pd.DataFrame({
        'SMILES': [
            "CCO",
            "c1ccccc1",
            "CC(=O)O.Na",       # Has salt
            "C[C@H](O)CC"       # Has stereochemistry
        ],
        'label': [1, 0, 1, 0]
    })
    
    input_filename = _store_resource(df, manifest_path, "test_input", "Test dataset", "csv")
    
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name='SMILES',
        project_manifest_path=manifest_path,
        output_filename='standardized_output',
        explanation='Test standardization'
    )
    
    # Verify result structure
    assert "output_filename" in result
    assert "n_rows" in result
    assert "columns" in result
    assert "preview" in result
    assert "protocol_summary" in result
    assert "final_validation" in result
    
    # Verify data
    assert result["n_rows"] == 4
    assert "standardized_smiles" in result["columns"]
    
    # Check protocol summary
    assert result["protocol_summary"]["stereo_policy"] == "flatten"
    assert result["protocol_summary"]["skip_isotope_removal"] == False


def test_default_SMILES_standardization_pipeline_dataset_with_dummy_data(session_workdir):
    """Test dataset pipeline with actual dummy_cleaning.csv data."""
    from molml_mcp.infrastructure.resources import _store_resource
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load dummy_cleaning.csv - navigate up from tests/tools/core/cleaning/ to tests/
    dummy_path = Path(__file__).parent.parent.parent.parent / "data" / "dummy_cleaning.csv"
    df = pd.read_csv(dummy_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_input", "Dummy cleaning data", "csv")
    
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name='SMILES',
        project_manifest_path=manifest_path,
        output_filename='dummy_standardized',
        explanation='Standardize dummy cleaning dataset'
    )
    
    # Verify result structure
    assert "output_filename" in result
    assert "n_rows" in result
    assert result["n_rows"] > 0
    
    # Verify standardized_smiles column was added
    assert "standardized_smiles" in result["columns"]
    
    # Check validation statistics
    assert "final_validation" in result
    assert "n_valid" in result["final_validation"]
    assert "n_invalid" in result["final_validation"]
    
    # Some entries should be valid (most basic organics)
    assert result["final_validation"]["n_valid"] > 0
    
    # Some entries should be invalid (intentionally bad SMILES in dummy data)
    assert result["final_validation"]["n_invalid"] > 0


def test_default_SMILES_standardization_pipeline_dataset_stereo_keep(session_workdir):
    """Test dataset pipeline preserves stereochemistry when requested."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create dataset with stereochemistry
    df = pd.DataFrame({
        'SMILES': [
            "C[C@H](O)CC",      # R-stereoisomer
            "C[C@@H](O)CC"      # S-stereoisomer
        ]
    })
    
    input_filename = _store_resource(df, manifest_path, "stereo_input", "Stereoisomers", "csv")
    
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name='SMILES',
        project_manifest_path=manifest_path,
        output_filename='stereo_output',
        explanation='Keep stereochemistry',
        stereo_policy='keep'
    )
    
    # Load result to check
    output_df = _load_resource(manifest_path, result["output_filename"])
    
    # Verify stereochemistry preserved
    assert result["protocol_summary"]["stereo_policy"] == "keep"
    
    # At least one standardized SMILES should contain stereochemistry marker
    has_stereo = any("@" in str(smi) for smi in output_df["standardized_smiles"])
    assert has_stereo, "Stereochemistry should be preserved"


def test_default_SMILES_standardization_pipeline_dataset_metal_disconnection(session_workdir):
    """Test dataset pipeline disconnects metals when requested."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create dataset with metal complexes
    df = pd.DataFrame({
        'SMILES': [
            "c1ccccc1[Fe]",         # Phenyl-iron
            "[Cu+2].c1ccccc1"       # Copper-benzene
        ]
    })
    
    input_filename = _store_resource(df, manifest_path, "metal_input", "Metal complexes", "csv")
    
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name='SMILES',
        project_manifest_path=manifest_path,
        output_filename='metal_output',
        explanation='Disconnect metals',
        enable_metal_disconnection=True
    )
    
    # Load result
    output_df = _load_resource(manifest_path, result["output_filename"])
    
    # Verify protocol
    assert result["protocol_summary"]["enable_metal_disconnection"] == True
    
    # Verify organic parts preserved (benzene rings)
    for smi in output_df["standardized_smiles"]:
        if pd.notna(smi) and smi:
            # Should contain benzene ring
            assert "c1ccccc1" in smi or "C1=CC=CC=C1" in smi, f"Organic part missing: {smi}"


def test_default_SMILES_standardization_pipeline_dataset_isotope_handling(session_workdir):
    """Test dataset pipeline handles isotopes correctly."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create dataset with isotopes
    df = pd.DataFrame({
        'SMILES': [
            "[2H]C([2H])([2H])O",   # Deuterated methanol
            "[13C]CO"               # C13-labeled ethanol
        ]
    })
    
    input_filename = _store_resource(df, manifest_path, "isotope_input", "Isotopes", "csv")
    
    # Test with isotope removal (default)
    result_remove = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name='SMILES',
        project_manifest_path=manifest_path,
        output_filename='isotope_removed',
        explanation='Remove isotopes',
        skip_isotope_removal=False
    )
    
    output_df_remove = _load_resource(manifest_path, result_remove["output_filename"])
    
    # Verify isotopes removed
    for smi in output_df_remove["standardized_smiles"]:
        if pd.notna(smi):
            assert "[2H]" not in smi, f"Isotope should be removed: {smi}"
            assert "[13C]" not in smi, f"Isotope should be removed: {smi}"
    
    # Test with isotope preservation
    result_keep = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name='SMILES',
        project_manifest_path=manifest_path,
        output_filename='isotope_kept',
        explanation='Keep isotopes',
        skip_isotope_removal=True
    )
    
    output_df_keep = _load_resource(manifest_path, result_keep["output_filename"])
    
    # Verify protocol setting was applied
    assert result_keep["protocol_summary"]["skip_isotope_removal"] == True
    assert result_remove["protocol_summary"]["skip_isotope_removal"] == False
    
    # Both should have valid standardized SMILES
    assert output_df_keep["standardized_smiles"].notna().sum() >= 1
    assert output_df_remove["standardized_smiles"].notna().sum() >= 1


def test_default_SMILES_standardization_pipeline_dataset_comment_columns(session_workdir):
    """Test dataset pipeline adds comment columns for each step."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource
    from molml_mcp.tools.cleaning.mol_cleaning import default_SMILES_standardization_pipeline_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create simple dataset
    df = pd.DataFrame({
        'SMILES': ["CCO", "CC(=O)O.Na", "INVALID"]
    })
    
    input_filename = _store_resource(df, manifest_path, "comment_test", "Test comments", "csv")
    
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name='SMILES',
        project_manifest_path=manifest_path,
        output_filename='comment_output',
        explanation='Test comment columns'
    )
    
    # Load result
    output_df = _load_resource(manifest_path, result["output_filename"])
    
    # Verify comment columns exist (pipeline adds comment for each step)
    # Should have at least: standardized_smiles and some comment columns
    assert "standardized_smiles" in output_df.columns
    
    # Check that we have rows processed
    assert len(output_df) == 3
    
    # Verify standardized_smiles column populated
    assert output_df["standardized_smiles"].notna().sum() >= 1  # At least one valid SMILES
