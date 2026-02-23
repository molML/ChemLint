"""
Simple tests for data_splitting.py functions using dummy data.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def test_random_split_dataset_basic(session_workdir):
    """Test basic random split (80/20 train/test)."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import random_split_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load dummy data
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data", "Dummy molecules", "csv")
    
    result = random_split_dataset(
        project_manifest_path=manifest_path,
        input_filename=input_filename,
        train_df_output_filename="train",
        test_df_output_filename="test",
        test_size=0.2,
        random_state=42
    )
    
    # Check structure
    assert "train_df_output_filename" in result
    assert "test_df_output_filename" in result
    assert "n_train_rows" in result
    assert "n_test_rows" in result
    
    # Check splits are reasonable
    total_rows = len(df)
    assert result["n_train_rows"] + result["n_test_rows"] == total_rows
    # 80/20 split of 144 rows should be approximately 115/29 (±2)
    assert 113 <= result["n_train_rows"] <= 117
    assert 27 <= result["n_test_rows"] <= 31


def test_random_split_dataset_with_validation(session_workdir):
    """Test random split with validation set (70/20/10)."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import random_split_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data2", "Dummy molecules", "csv")
    
    result = random_split_dataset(
        project_manifest_path=manifest_path,
        input_filename=input_filename,
        train_df_output_filename="train",
        test_df_output_filename="test",
        val_df_output_filename="val",
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Check all three splits exist
    assert result["val_df_output_filename"] is not None
    
    # Check total
    total_rows = len(df)
    assert result["n_train_rows"] + result["n_test_rows"] + result["n_val_rows"] == total_rows
    # 70/20/10 split: first 20% test (~29), then 10% of remaining for val (~11-15), rest train (~99-104)
    assert 27 <= result["n_test_rows"] <= 31
    assert 11 <= result["n_val_rows"] <= 15
    assert 98 <= result["n_train_rows"] <= 106


def test_random_split_dataset_invalid_sizes(session_workdir):
    """Test that invalid split sizes raise errors."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import random_split_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data3", "Dummy molecules", "csv")
    
    # Test size >= 1.0
    with pytest.raises(ValueError):
        random_split_dataset(
            project_manifest_path=manifest_path,
            input_filename=input_filename,
            train_df_output_filename="train",
            test_df_output_filename="test",
            test_size=1.5
        )


def test_stratified_split_dataset_classification(session_workdir):
    """Test stratified split with classification labels."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import stratified_split_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data_strat", "Dummy molecules", "csv")
    
    result = stratified_split_dataset(
        input_filename=input_filename,
        label_column="class",  # Binary classification column
        project_manifest_path=manifest_path,
        train_output_filename="train_strat",
        test_output_filename="test_strat",
        train_ratio=0.8,
        test_ratio=0.2
    )
    
    # Check structure
    assert "train_output_filename" in result
    assert "test_output_filename" in result
    assert "train_label_distribution" in result
    assert "test_label_distribution" in result
    assert result["is_regression"] == False
    
    # Check splits
    assert result["n_train_rows"] + result["n_test_rows"] == len(df)
    # 80/20 split should be approximately 115/29
    assert result["n_train_rows"] >= 110
    assert result["n_test_rows"] >= 25


def test_stratified_split_dataset_regression(session_workdir):
    """Test stratified split with regression (continuous) labels."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import stratified_split_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data_strat_reg", "Dummy molecules", "csv")
    
    result = stratified_split_dataset(
        input_filename=input_filename,
        label_column="exp_mean [nM]",  # Continuous regression column
        project_manifest_path=manifest_path,
        train_output_filename="train_strat_reg",
        test_output_filename="test_strat_reg",
        train_ratio=0.8,
        test_ratio=0.2,
        n_bins=5
    )
    
    # Check it detected regression
    assert result["is_regression"] == True
    assert result["bin_edges"] is not None
    assert len(result["bin_edges"]) > 0
    
    # Check splits
    assert result["n_train_rows"] + result["n_test_rows"] == len(df)


def test_stratified_split_dataset_with_validation(session_workdir):
    """Test stratified split with validation set."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import stratified_split_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data_strat_val", "Dummy molecules", "csv")
    
    result = stratified_split_dataset(
        input_filename=input_filename,
        label_column="class",
        project_manifest_path=manifest_path,
        train_output_filename="train",
        test_output_filename="test",
        val_output_filename="val",
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1
    )
    
    # Check all three splits
    assert result["val_output_filename"] is not None
    assert result["n_train_rows"] + result["n_test_rows"] + result["n_val_rows"] == len(df)
    # 70/20/10 split: approximately 101/29/14
    assert result["n_train_rows"] >= 95
    assert result["n_test_rows"] >= 25
    assert result["n_val_rows"] >= 10


def test_scaffold_split_dataset_basic(session_workdir):
    """Test basic scaffold split."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import scaffold_split_dataset
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data_scaffold", "Dummy molecules", "csv")
    
    # First add scaffolds
    scaffold_result = calculate_scaffolds_dataset(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        column_name="smiles",
        scaffold_type="bemis_murcko",
        output_filename="with_scaffolds",
        explanation="Added scaffolds"
    )
    
    # Now do scaffold split
    result = scaffold_split_dataset(
        input_filename=scaffold_result["output_filename"],
        scaffold_column="scaffold_bemis_murcko",
        project_manifest_path=manifest_path,
        train_output_filename="train_scaffold",
        test_output_filename="test_scaffold",
        train_ratio=0.8,
        test_ratio=0.2
    )
    
    # Check structure
    assert "train_output_filename" in result
    assert "test_output_filename" in result
    assert "n_train_rows" in result
    assert "n_test_rows" in result
    assert "n_train_scaffolds" in result
    assert "n_test_scaffolds" in result
    
    # Check splits
    assert result["n_train_rows"] + result["n_test_rows"] == len(df)
    # Scaffold splits may vary but should have reasonable distribution
    assert result["n_train_rows"] >= 100
    assert result["n_test_rows"] >= 20


def test_scaffold_split_dataset_with_validation(session_workdir):
    """Test scaffold split with validation set."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import scaffold_split_dataset
    from chemlint.tools.core_mol.scaffolds import calculate_scaffolds_dataset
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data_scaffold2", "Dummy molecules", "csv")
    
    # Add scaffolds
    scaffold_result = calculate_scaffolds_dataset(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        column_name="smiles",
        scaffold_type="bemis_murcko",
        output_filename="with_scaffolds2",
        explanation="Added scaffolds"
    )
    
    # Scaffold split with validation
    result = scaffold_split_dataset(
        input_filename=scaffold_result["output_filename"],
        scaffold_column="scaffold_bemis_murcko",
        project_manifest_path=manifest_path,
        train_output_filename="train",
        test_output_filename="test",
        val_output_filename="val",
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1
    )
    
    # Check all three splits
    assert result["val_output_filename"] is not None
    assert result["n_train_rows"] + result["n_test_rows"] + result["n_val_rows"] == len(df)
    # Scaffold-based splits vary but should have reasonable distribution
    assert result["n_train_rows"] >= 85
    assert result["n_test_rows"] >= 20
    assert result["n_val_rows"] >= 5


def test_cluster_based_split_dataset_basic(session_workdir):
    """Test basic cluster-based split."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core_mol.data_splitting import cluster_based_split_dataset
    from sklearn.cluster import KMeans
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path)
    
    # Add cluster assignments
    fps = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fps.append(list(fp))
        else:
            fps.append([0] * 1024)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)  # Fewer clusters for smaller dataset
    df["cluster"] = kmeans.fit_predict(fps)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data_cluster", "Dummy molecules", "csv")
    
    result = cluster_based_split_dataset(
        input_filename=input_filename,
        cluster_column="cluster",
        project_manifest_path=manifest_path,
        train_output_filename="train_cluster",
        test_output_filename="test_cluster",
        train_ratio=0.8,
        test_ratio=0.2,
        random_state=42
    )
    
    # Check structure
    assert "train_output_filename" in result
    assert "test_output_filename" in result
    assert "n_train_rows" in result
    assert "n_test_rows" in result
    assert "n_train_clusters" in result
    assert "n_test_clusters" in result
    
    # Check splits
    assert result["n_train_rows"] + result["n_test_rows"] == len(df)
    # Cluster-based splits vary but should have reasonable distribution
    assert result["n_train_rows"] >= 30  # Adjusted for smaller dataset
    assert result["n_test_rows"] >= 10  # Adjusted for smaller dataset


def test_cluster_based_split_dataset_with_validation(session_workdir):
    """Test cluster-based split with validation set."""
    from chemlint.infrastructure.resources import _store_resource
    from chemlint.tools.core_mol.data_splitting import cluster_based_split_dataset
    from sklearn.cluster import KMeans
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    dummy_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_data_raw_small.csv"
    df = pd.read_csv(dummy_data_path).head(50)  # Use smaller dataset for speed
    
    # Add cluster assignments
    fps = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fps.append(list(fp))
        else:
            fps.append([0] * 1024)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)  # Fewer clusters for smaller dataset
    df["cluster"] = kmeans.fit_predict(fps)
    
    input_filename = _store_resource(df, manifest_path, "dummy_data_cluster2", "Dummy molecules", "csv")
    
    result = cluster_based_split_dataset(
        input_filename=input_filename,
        cluster_column="cluster",
        project_manifest_path=manifest_path,
        train_output_filename="train",
        test_output_filename="test",
        val_output_filename="val",
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1,
        random_state=42
    )
    
    # Check all three splits
    assert result["val_output_filename"] is not None
    assert result["n_train_rows"] + result["n_test_rows"] + result["n_val_rows"] == len(df)
    # Cluster-based splits vary but should have reasonable distribution
    assert result["n_train_rows"] >= 25  # Adjusted for smaller dataset
    assert result["n_test_rows"] >= 8  # Adjusted for smaller dataset
    assert result["n_val_rows"] >= 3  # Adjusted for smaller dataset
    assert result["n_val_clusters"] >= 1
