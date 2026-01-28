"""Tests for simple_descriptors.py - RDKit descriptor calculation."""

import pandas as pd
import numpy as np
import pytest
from molml_mcp.tools.featurization.simple_descriptors import (
    list_rdkit_descriptors,
    calculate_simple_descriptors,
    calculate_descriptor_vectors,
    normalize_feature_vectors,
)
from molml_mcp.infrastructure.resources import create_project_manifest, _store_resource, _load_resource


def test_list_rdkit_descriptors():
    """Test listing available RDKit descriptors."""
    descriptors = list_rdkit_descriptors()
    
    # Should return a non-empty list
    assert isinstance(descriptors, list)
    assert len(descriptors) > 100  # RDKit has 200+ descriptors
    
    # Each item should have required keys
    for desc in descriptors:
        assert "descriptor name" in desc
        assert "explanation" in desc
        assert isinstance(desc["descriptor name"], str)
        assert isinstance(desc["explanation"], str)
    
    # Check for some common descriptors
    descriptor_names = [d["descriptor name"] for d in descriptors]
    assert "MolWt" in descriptor_names
    assert "TPSA" in descriptor_names
    assert "MolLogP" in descriptor_names


def test_calculate_simple_descriptors_basic(session_workdir, request):
    """Test basic descriptor calculation."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(C)O"],
        "id": [1, 2, 3]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate some common descriptors
    result = calculate_simple_descriptors(
        input_filename=input_file,
        smiles_column="smiles",
        descriptor_names=["MolWt", "TPSA", "NumHDonors"],
        project_manifest_path=manifest_path,
        output_filename="with_descriptors",
        explanation="calculated descriptors"
    )
    
    # Check return structure
    assert "output_filename" in result
    assert "n_rows" in result
    assert "columns" in result
    assert "descriptors_added" in result
    assert "n_failed" in result
    
    # Check values
    assert result["n_rows"] == 3
    assert "MolWt" in result["columns"]
    assert "TPSA" in result["columns"]
    assert "NumHDonors" in result["columns"]
    assert result["descriptors_added"] == ["MolWt", "TPSA", "NumHDonors"]
    
    # Check preview has reasonable values
    preview = result["preview"]
    assert len(preview) == 3
    # Ethanol (CCO) should have MolWt around 46, TPSA around 20, 1 H-donor
    assert 45 < preview[0]["MolWt"] < 47
    assert preview[0]["NumHDonors"] == 1


def test_calculate_simple_descriptors_invalid_smiles(session_workdir, request):
    """Test descriptor calculation handles invalid SMILES."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create dataset with invalid SMILES
    df = pd.DataFrame({
        "smiles": ["CCO", "INVALID", None, "c1ccccc1"],
        "id": [1, 2, 3, 4]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate descriptors
    result = calculate_simple_descriptors(
        input_filename=input_file,
        smiles_column="smiles",
        descriptor_names=["MolWt"],
        project_manifest_path=manifest_path,
        output_filename="with_descriptors",
        explanation="calculated descriptors"
    )
    
    # Should complete but report failures
    assert result["n_rows"] == 4
    assert result["n_failed"]["MolWt"] >= 2  # At least 2 failed (INVALID and None)
    
    # Preview should have None or NaN for invalid entries
    import math
    preview = result["preview"]
    assert preview[0]["MolWt"] is not None and not (isinstance(preview[0]["MolWt"], float) and math.isnan(preview[0]["MolWt"]))  # Valid SMILES
    # Invalid SMILES should have None or NaN
    assert preview[1]["MolWt"] is None or (isinstance(preview[1]["MolWt"], float) and math.isnan(preview[1]["MolWt"]))
    assert preview[2]["MolWt"] is None or (isinstance(preview[2]["MolWt"], float) and math.isnan(preview[2]["MolWt"]))


def test_calculate_simple_descriptors_invalid_descriptor(session_workdir, request):
    """Test error handling for invalid descriptor names."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({"smiles": ["CCO"], "id": [1]})
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Try to use invalid descriptor name
    with pytest.raises(ValueError, match="Invalid descriptor names"):
        calculate_simple_descriptors(
            input_filename=input_file,
            smiles_column="smiles",
            descriptor_names=["FakeDescriptor"],
            project_manifest_path=manifest_path,
            output_filename="output",
            explanation="test"
        )


def test_calculate_simple_descriptors_multiple_descriptors(session_workdir, request):
    """Test calculating multiple descriptors at once."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset with drug-like molecules
    df = pd.DataFrame({
        "smiles": ["CCO", "CC(=O)O", "c1ccccc1"],
        "name": ["ethanol", "acetic acid", "benzene"]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate multiple descriptors
    descriptors = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]
    result = calculate_simple_descriptors(
        input_filename=input_file,
        smiles_column="smiles",
        descriptor_names=descriptors,
        project_manifest_path=manifest_path,
        output_filename="with_many_descriptors",
        explanation="multiple descriptors"
    )
    
    # All descriptors should be added
    assert result["n_rows"] == 3
    assert len(result["descriptors_added"]) == 5
    for desc in descriptors:
        assert desc in result["columns"]
        assert desc in result["descriptors_added"]
    


def test_calculate_descriptor_vectors_basic(session_workdir, request):
    """Test basic descriptor vector calculation."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(C)O"],
        "id": [1, 2, 3]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate descriptor vectors
    result = calculate_descriptor_vectors(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        descriptor_names=["MolWt", "TPSA", "NumHDonors"],
        output_filename="descriptor_vectors",
        explanation="test descriptor vectors"
    )
    
    # Check return structure
    assert "output_filename" in result
    assert "n_molecules" in result
    assert "descriptor_names" in result
    assert "n_descriptors" in result
    
    # Check values
    assert result["n_molecules"] == 3
    assert result["n_descriptors"] == 3
    assert result["descriptor_names"] == ["MolWt", "TPSA", "NumHDonors"]
    
    # Load and verify feature vectors
    vectors = _load_resource(manifest_path, result["output_filename"])
    assert isinstance(vectors, dict)
    assert len(vectors) == 3
    
    # Check vector structure
    for smiles, vec in vectors.items():
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (3,)  # 3 descriptors
        assert vec.dtype == np.float32
    
    # Check specific values for ethanol (CCO)
    ethanol_vec = vectors["CCO"]
    assert 45 < ethanol_vec[0] < 47  # MolWt around 46
    assert ethanol_vec[2] == 1  # 1 H-donor


def test_calculate_descriptor_vectors_invalid_smiles(session_workdir, request):
    """Test descriptor vectors with invalid SMILES."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create dataset with invalid SMILES
    df = pd.DataFrame({
        "smiles": ["CCO", "INVALID_SMILES", "CCC"]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Should raise error for invalid SMILES
    with pytest.raises(ValueError, match="Failed to parse SMILES string"):
        calculate_descriptor_vectors(
            input_filename=input_file,
            project_manifest_path=manifest_path,
            smiles_column="smiles",
            descriptor_names=["MolWt"],
            output_filename="descriptor_vectors",
            explanation="test"
        )


def test_calculate_descriptor_vectors_invalid_descriptor(session_workdir, request):
    """Test error handling for invalid descriptor names."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({"smiles": ["CCO"]})
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Try to use invalid descriptor name
    with pytest.raises(ValueError, match="Invalid descriptor names"):
        calculate_descriptor_vectors(
            input_filename=input_file,
            project_manifest_path=manifest_path,
            smiles_column="smiles",
            descriptor_names=["FakeDescriptor", "MolWt"],
            output_filename="output",
            explanation="test"
        )


def test_normalize_feature_vectors_train_only(session_workdir, request):
    """Test normalizing training set only."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(C)O", "CCCC"],
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate descriptor vectors
    vectors_result = calculate_descriptor_vectors(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        descriptor_names=["MolWt", "TPSA", "NumHDonors"],
        output_filename="train_vectors",
        explanation="test vectors"
    )
    
    # Normalize
    result = normalize_feature_vectors(
        train_filename=vectors_result["output_filename"],
        project_manifest_path=manifest_path,
        train_output_filename="normalized_train",
        explanation="normalize test"
    )
    
    # Check return structure
    assert "train_output_filename" in result
    assert "test_output_filenames" in result
    assert "n_features" in result
    assert "scaler_mean" in result
    assert "scaler_std" in result
    
    # Check values
    assert result["n_features"] == 3
    assert len(result["test_output_filenames"]) == 0
    assert len(result["scaler_mean"]) == 3
    assert len(result["scaler_std"]) == 3
    
    # Load normalized vectors
    normalized = _load_resource(manifest_path, result["train_output_filename"])
    assert len(normalized) == 4
    
    # Check that values are normalized (mean≈0, std≈1)
    all_vectors = np.vstack([normalized[smi] for smi in normalized.keys()])
    feature_means = np.mean(all_vectors, axis=0)
    feature_stds = np.std(all_vectors, axis=0)
    
    # Should be close to 0 mean and 1 std (with small sample size, not exact)
    assert np.allclose(feature_means, 0, atol=1e-6)
    assert np.allclose(feature_stds, 1, atol=1e-6)


def test_normalize_feature_vectors_with_test_set(session_workdir, request):
    """Test normalizing train and test sets together."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create train and test datasets
    train_df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(C)O", "CCCC"],
    })
    test_df = pd.DataFrame({
        "smiles": ["CCC", "CC(C)C"],
    })
    
    train_file = _store_resource(train_df, manifest_path, "train_molecules", "train data", "csv")
    test_file = _store_resource(test_df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate descriptor vectors for both
    train_vectors = calculate_descriptor_vectors(
        input_filename=train_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        descriptor_names=["MolWt", "TPSA"],
        output_filename="train_vectors",
        explanation="train vectors"
    )
    
    test_vectors = calculate_descriptor_vectors(
        input_filename=test_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        descriptor_names=["MolWt", "TPSA"],
        output_filename="test_vectors",
        explanation="test vectors"
    )
    
    # Normalize both sets
    result = normalize_feature_vectors(
        train_filename=train_vectors["output_filename"],
        project_manifest_path=manifest_path,
        train_output_filename="normalized_train",
        test_filenames=[test_vectors["output_filename"]],
        test_output_filenames=["normalized_test"],
        explanation="normalize both"
    )
    
    # Check return structure
    assert result["n_features"] == 2
    assert len(result["test_output_filenames"]) == 1
    
    # Load both normalized sets
    train_normalized = _load_resource(manifest_path, result["train_output_filename"])
    test_normalized = _load_resource(manifest_path, result["test_output_filenames"][0])
    
    assert len(train_normalized) == 4
    assert len(test_normalized) == 2
    
    # Check that test set uses same normalization as train set
    # (test set should NOT have mean=0, std=1, but should use train's scaler)
    train_matrix = np.vstack([train_normalized[smi] for smi in train_normalized.keys()])
    test_matrix = np.vstack([test_normalized[smi] for smi in test_normalized.keys()])
    
    # Train set should be normalized
    assert np.allclose(np.mean(train_matrix, axis=0), 0, atol=1e-6)
    
    # Test set might not have mean=0 (that's expected - it uses train's scaler)
    # But values should be reasonable (not wildly different)
    assert np.all(np.abs(test_matrix) < 10)  # Sanity check


def test_normalize_feature_vectors_multiple_test_sets(session_workdir, request):
    """Test normalizing with multiple test sets."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create datasets
    train_df = pd.DataFrame({"smiles": ["CCO", "CCC", "CCCC"]})
    test1_df = pd.DataFrame({"smiles": ["CC"]})
    test2_df = pd.DataFrame({"smiles": ["C"]})
    
    train_file = _store_resource(train_df, manifest_path, "train", "train", "csv")
    test1_file = _store_resource(test1_df, manifest_path, "test1", "test1", "csv")
    test2_file = _store_resource(test2_df, manifest_path, "test2", "test2", "csv")
    
    # Calculate vectors
    train_vec = calculate_descriptor_vectors(
        input_filename=train_file, project_manifest_path=manifest_path,
        smiles_column="smiles", descriptor_names=["MolWt"],
        output_filename="train_vec", explanation="train"
    )
    test1_vec = calculate_descriptor_vectors(
        input_filename=test1_file, project_manifest_path=manifest_path,
        smiles_column="smiles", descriptor_names=["MolWt"],
        output_filename="test1_vec", explanation="test1"
    )
    test2_vec = calculate_descriptor_vectors(
        input_filename=test2_file, project_manifest_path=manifest_path,
        smiles_column="smiles", descriptor_names=["MolWt"],
        output_filename="test2_vec", explanation="test2"
    )
    
    # Normalize all
    result = normalize_feature_vectors(
        train_filename=train_vec["output_filename"],
        project_manifest_path=manifest_path,
        train_output_filename="norm_train",
        test_filenames=[test1_vec["output_filename"], test2_vec["output_filename"]],
        test_output_filenames=["norm_test1", "norm_test2"],
        explanation="normalize all"
    )
    
    # Check all sets were created
    assert len(result["test_output_filenames"]) == 2
    
    # Load all sets
    train_norm = _load_resource(manifest_path, result["train_output_filename"])
    test1_norm = _load_resource(manifest_path, result["test_output_filenames"][0])
    test2_norm = _load_resource(manifest_path, result["test_output_filenames"][1])
    
    assert len(train_norm) == 3
    assert len(test1_norm) == 1
    assert len(test2_norm) == 1


def test_normalize_feature_vectors_mismatched_lengths(session_workdir, request):
    """Test error when test_filenames and test_output_filenames don't match."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({"smiles": ["CCO"]})
    input_file = _store_resource(df, manifest_path, "molecules", "data", "csv")
    
    vec = calculate_descriptor_vectors(
        input_filename=input_file, project_manifest_path=manifest_path,
        smiles_column="smiles", descriptor_names=["MolWt"],
        output_filename="vectors", explanation="test"
    )
    
    # Mismatched lengths should raise error
    with pytest.raises(ValueError, match="must have the same length"):
        normalize_feature_vectors(
            train_filename=vec["output_filename"],
            project_manifest_path=manifest_path,
            train_output_filename="norm_train",
            test_filenames=["file1", "file2"],
            test_output_filenames=["output1"],  # Only 1 output for 2 inputs
            explanation="test"
        )

