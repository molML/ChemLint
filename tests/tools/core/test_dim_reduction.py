"""Tests for dim_reduction.py functions."""
import pandas as pd
import pytest
import numpy as np
from pathlib import Path


def _get_or_create_test_data_with_ecfp(session_workdir, manifest_path):
    """Helper to create or reuse test molecular data with ECFP fingerprints.
    
    This function creates the dataset and fingerprints once and reuses them
    across multiple tests within the same session to speed up testing.
    """
    from chemlint.infrastructure.resources import _store_resource
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Load real data from tests/data directory
    data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df_real = pd.read_csv(data_path)
    
    # Take subset for testing (25 molecules - enough for both PCA and t-SNE)
    df_subset = df_real.head(25).copy()
    
    # Store dataset
    dataset_filename = _store_resource(
        df_subset, 
        manifest_path, 
        "shared_test_molecules", 
        "Shared molecular data for dim reduction tests", 
        "csv"
    )
    
    # Compute ECFP4 fingerprints (1024 bits - good balance for both PCA and t-SNE)
    features_dict = {}
    for smiles in df_subset["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            features_dict[smiles] = np.array(fp)
    
    features_filename = _store_resource(
        features_dict, 
        manifest_path, 
        "shared_ecfp4_fingerprints", 
        "Shared ECFP4 fingerprints", 
        "feature_vectors"
    )
    
    return dataset_filename, features_filename


def test_reduce_dimensions_pca_with_real_data_and_ecfp(session_workdir):
    """Test PCA with real molecular data and computed ECFP fingerprints."""
    from chemlint.infrastructure.resources import _load_resource
    from chemlint.tools.core.dim_reduction import reduce_dimensions_pca
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Get or create shared test data (reused across tests)
    dataset_filename, features_filename = _get_or_create_test_data_with_ecfp(session_workdir, manifest_path)
    
    # Run PCA
    result = reduce_dimensions_pca(
        input_filename=dataset_filename,
        feature_vectors_filename=features_filename,
        project_manifest_path=manifest_path,
        output_filename="molecules_with_pca",
        explanation="PCA dimensionality reduction on ECFP4 fingerprints",
        smiles_column="smiles"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "PC1" in df_result.columns
    assert "PC2" in df_result.columns
    
def test_reduce_dimensions_tsne_with_real_data_and_ecfp(session_workdir):
    """Test t-SNE with real molecular data and computed ECFP fingerprints."""
    from chemlint.infrastructure.resources import _load_resource
    from chemlint.tools.core.dim_reduction import reduce_dimensions_tsne
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Get or create shared test data (reused from PCA test if already created)
    dataset_filename, features_filename = _get_or_create_test_data_with_ecfp(session_workdir, manifest_path)
    
    # Run t-SNE
    result = reduce_dimensions_tsne(
        input_filename=dataset_filename,
        feature_vectors_filename=features_filename,
        project_manifest_path=manifest_path,
        output_filename="molecules_with_tsne",
        explanation="t-SNE dimensionality reduction on ECFP4 fingerprints",
        smiles_column="smiles",
        perplexity=15.0,
        max_iter=500
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "tSNE1" in df_result.columns
    assert "tSNE2" in df_result.columns
