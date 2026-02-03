"""Tests for dataset_ops.py functions."""
import pandas as pd
import pytest
from pathlib import Path


def test_import_csv_from_path(session_workdir, request):
    """Test importing CSV file as dataset."""
    from molml_mcp.tools.core.dataset_ops import import_csv_from_path
    from molml_mcp.infrastructure.resources import _load_resource, create_project_manifest
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test CSV
    csv_path = test_dir / "test.csv"
    df_original = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df_original.to_csv(csv_path, index=False)
    
    # Store as dataset
    result = import_csv_from_path(
        file_path=str(csv_path),
        project_manifest_path=manifest_path,
        filename="test_data",
        explanation="Test dataset"
    )
    
    assert "output_filename" in result
    assert "n_rows" in result
    assert result["n_rows"] == 3
    assert "columns" in result
    assert set(result["columns"]) == {"A", "B"}
    assert "preview" in result
    assert len(result["preview"]) == 3
    
    # Verify data can be loaded back
    df_loaded = _load_resource(manifest_path, result["output_filename"])
    assert df_loaded.equals(df_original)


def test_import_csv_from_text(session_workdir, request):
    """Test importing CSV text as dataset."""
    from molml_mcp.tools.core.dataset_ops import import_csv_from_text
    from molml_mcp.infrastructure.resources import _load_resource, create_project_manifest
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create CSV text
    csv_text = "A,B\n1,4\n2,5\n3,6"
    
    # Store as dataset
    result = import_csv_from_text(
        csv_content=csv_text,
        project_manifest_path=manifest_path,
        filename="test_data_text",
        explanation="Test dataset from text"
    )
    
    assert "output_filename" in result
    assert "n_rows" in result
    assert result["n_rows"] == 3
    assert "columns" in result
    assert set(result["columns"]) == {"A", "B"}
    assert "preview" in result
    assert len(result["preview"]) == 3
    
    # Verify data can be loaded back and matches expected values
    df_loaded = _load_resource(manifest_path, result["output_filename"])
    assert len(df_loaded) == 3
    assert list(df_loaded["A"]) == [1, 2, 3]
    assert list(df_loaded["B"]) == [4, 5, 6]


def test_get_dataset_head(session_workdir, request):
    """Test getting dataset head."""
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import get_dataset_head
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store test dataset
    df = pd.DataFrame({"A": range(10), "B": range(10, 20)})
    filename = _store_resource(df, manifest_path, "test_data_head", "Test", "csv")
    
    # Get head with n_rows=5
    result = get_dataset_head(
        project_manifest_path=manifest_path,
        input_filename=filename,
        n_rows=5
    )
    
    assert "n_rows_returned" in result
    assert result["n_rows_returned"] == 5
    assert "n_rows_total" in result
    assert result["n_rows_total"] == 10
    assert "rows" in result
    assert len(result["rows"]) == 5
    # Verify we got the first 5 rows
    assert result["rows"][0]["A"] == 0
    assert result["rows"][4]["A"] == 4


def test_get_dataset_full(session_workdir, request):
    """Test getting full dataset."""
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import get_dataset_full
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store test dataset
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    filename = _store_resource(df, manifest_path, "test_data_full", "Test", "csv")
    
    # Get full dataset
    result = get_dataset_full(
        project_manifest_path=manifest_path,
        input_filename=filename
    )
    
    assert "n_rows_returned" in result
    assert result["n_rows_returned"] == 3
    assert "n_rows_total" in result
    assert result["n_rows_total"] == 3
    assert "rows" in result
    assert len(result["rows"]) == 3
    assert "truncated" in result
    assert result["truncated"] == False


def test_get_dataset_summary(session_workdir, request):
    """Test getting dataset summary."""
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import get_dataset_summary
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store test dataset
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    filename = _store_resource(df, manifest_path, "test_data_summary", "Test", "csv")
    
    # Get summary
    result = get_dataset_summary(
        project_manifest_path=manifest_path,
        input_filename=filename
    )
    
    assert "n_rows" in result
    assert result["n_rows"] == 3
    assert "n_columns" in result
    assert result["n_columns"] == 2
    assert "column_summaries" in result
    assert "A" in result["column_summaries"]
    assert "B" in result["column_summaries"]
    # Verify numeric summaries have mean
    assert "mean" in result["column_summaries"]["A"]
    assert result["column_summaries"]["A"]["mean"] == 2.0


def test_inspect_dataset_rows(session_workdir, request):
    """Test inspecting specific dataset rows with index and filter conditions."""
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import inspect_dataset_rows
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store test dataset
    df = pd.DataFrame({"A": range(10), "B": range(10, 20), "C": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]})
    filename = _store_resource(df, manifest_path, "test_data_inspect", "Test", "csv")
    
    # Test 1: Inspect specific rows by index
    result1 = inspect_dataset_rows(
        project_manifest_path=manifest_path,
        input_filename=filename,
        row_indices=[0, 5, 9]
    )
    
    assert "n_rows_returned" in result1
    assert result1["n_rows_returned"] == 3
    assert "rows" in result1
    assert len(result1["rows"]) == 3
    # Verify we got the correct rows
    assert result1["rows"][0]["A"] == 0
    assert result1["rows"][1]["A"] == 5
    assert result1["rows"][2]["A"] == 9
    
    # Test 2: Filter with > condition (numeric comparison)
    result2 = inspect_dataset_rows(
        project_manifest_path=manifest_path,
        input_filename=filename,
        filter_condition="A > 5"
    )
    
    assert result2["n_rows_returned"] == 4  # Rows with A = 6, 7, 8, 9
    assert "rows" in result2
    assert all(row["A"] > 5 for row in result2["rows"])
    assert result2["rows"][0]["A"] == 6
    
    # Test 3: Filter with < condition (numeric comparison)
    result3 = inspect_dataset_rows(
        project_manifest_path=manifest_path,
        input_filename=filename,
        filter_condition="B < 15"
    )
    
    assert result3["n_rows_returned"] == 5  # Rows with B = 10, 11, 12, 13, 14
    assert all(row["B"] < 15 for row in result3["rows"])
    
    # Test 4: Filter with >= condition
    result4 = inspect_dataset_rows(
        project_manifest_path=manifest_path,
        input_filename=filename,
        filter_condition="C >= 7.5"
    )
    
    assert result4["n_rows_returned"] == 4  # Rows with C = 7.5, 8.5, 9.5, 10.5
    assert all(row["C"] >= 7.5 for row in result4["rows"])
    
    # Test 5: Filter with range condition
    result5 = inspect_dataset_rows(
        project_manifest_path=manifest_path,
        input_filename=filename,
        filter_condition="3 <= A <= 7"
    )
    
    assert result5["n_rows_returned"] == 5  # Rows with A = 3, 4, 5, 6, 7
    assert all(3 <= row["A"] <= 7 for row in result5["rows"])
    
    # Test 6: Filter with multiple conditions using 'and'
    result6 = inspect_dataset_rows(
        project_manifest_path=manifest_path,
        input_filename=filename,
        filter_condition="A > 2 and B < 17"
    )
    
    assert result6["n_rows_returned"] == 4  # Rows with A > 2 AND B < 17 (A=3,4,5,6 with B=13,14,15,16)
    assert all(row["A"] > 2 and row["B"] < 17 for row in result6["rows"])


def test_drop_from_dataset(session_workdir, request):
    """Test dropping rows from dataset with various conditions."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import drop_from_dataset
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Test 1: Drop rows with exact numeric match
    df1 = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    filename1 = _store_resource(df1, manifest_path, "test_data_drop1", "Test", "csv")
    
    result1 = drop_from_dataset(
        input_filename=filename1,
        column_name="A",
        condition="5",
        project_manifest_path=manifest_path,
        output_filename="filtered_data1",
        explanation="Drop rows with A == 5"
    )
    
    assert "output_filename" in result1
    assert "n_rows" in result1
    assert result1["n_rows"] == 4  # Should have 4 rows left (A=1,2,3,4)
    df_result1 = _load_resource(manifest_path, result1["output_filename"])
    assert 5 not in df_result1["A"].values
    assert len(df_result1) == 4
    
    # Test 2: Drop rows with null values using "is None"
    df2 = pd.DataFrame({"A": [1, 2, None, 4, 5], "B": [10, 20, 30, 40, 50]})
    filename2 = _store_resource(df2, manifest_path, "test_data_drop2", "Test", "csv")
    
    result2 = drop_from_dataset(
        input_filename=filename2,
        column_name="A",
        condition="is None",
        project_manifest_path=manifest_path,
        output_filename="filtered_data2",
        explanation="Drop rows with null A"
    )
    
    assert result2["n_rows"] == 4  # Should have 4 rows (removed the null)
    df_result2 = _load_resource(manifest_path, result2["output_filename"])
    assert df_result2["A"].isnull().sum() == 0  # No nulls should remain
    assert len(df_result2) == 4
    
    # Test 3: Drop rows with exact string match
    df3 = pd.DataFrame({
        "status": ["Passed", "Failed", "Passed", "Failed", "Passed"],
        "value": [1, 2, 3, 4, 5]
    })
    filename3 = _store_resource(df3, manifest_path, "test_data_drop3", "Test", "csv")
    
    result3 = drop_from_dataset(
        input_filename=filename3,
        column_name="status",
        condition="Failed",
        project_manifest_path=manifest_path,
        output_filename="filtered_data3",
        explanation="Drop rows with status == Failed"
    )
    
    assert result3["n_rows"] == 3  # Should have 3 Passed rows
    df_result3 = _load_resource(manifest_path, result3["output_filename"])
    assert "Failed" not in df_result3["status"].values
    assert all(df_result3["status"] == "Passed")
    
    # Test 4: Drop rows with exact string match (complex string)
    df4 = pd.DataFrame({
        "comments": [
            "Valid",
            "Failed: Invalid SMILES string",
            "Valid",
            "Failed: Invalid SMILES string",
            "Valid"
        ],
        "value": [1, 2, 3, 4, 5]
    })
    filename4 = _store_resource(df4, manifest_path, "test_data_drop4", "Test", "csv")
    
    result4 = drop_from_dataset(
        input_filename=filename4,
        column_name="comments",
        condition="Failed: Invalid SMILES string",
        project_manifest_path=manifest_path,
        output_filename="filtered_data4",
        explanation="Drop rows with specific failure message"
    )
    
    assert result4["n_rows"] == 3  # Should have 3 Valid rows
    df_result4 = _load_resource(manifest_path, result4["output_filename"])
    assert "Failed: Invalid SMILES string" not in df_result4["comments"].values
    assert all(df_result4["comments"] == "Valid")
    
    # Test 5: Drop multiple matching numeric values in a single call
    df5 = pd.DataFrame({"A": [1, 1, 2, 3, 1], "B": [10, 20, 30, 40, 50]})
    filename5 = _store_resource(df5, manifest_path, "test_data_drop5", "Test", "csv")
    
    result5 = drop_from_dataset(
        input_filename=filename5,
        column_name="A",
        condition="1",
        project_manifest_path=manifest_path,
        output_filename="filtered_data5",
        explanation="Drop all rows with A == 1"
    )
    
    assert result5["n_rows"] == 2  # Should have 2 rows (A=2 and A=3)
    df_result5 = _load_resource(manifest_path, result5["output_filename"])
    assert 1 not in df_result5["A"].values
    assert list(df_result5["A"]) == [2, 3]


def test_subset_dataset(session_workdir, request):
    """Test subsetting rows with pandas query filters."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import subset_dataset
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Test 1: Exact match for numbers
    df1 = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    filename1 = _store_resource(df1, manifest_path, "test_data1", "Test", "csv")
    
    result1 = subset_dataset(
        input_filename=filename1,
        project_manifest_path=manifest_path,
        output_filename="result1",
        explanation="Keep A == 3",
        filter_condition="A == 3"
    )
    
    assert result1["n_rows"] == 1
    df_result1 = _load_resource(manifest_path, result1["output_filename"])
    assert df_result1["A"].iloc[0] == 3
    
    # Test 2: Exact match for strings
    df2 = pd.DataFrame({
        "status": ["Passed", "Failed", "Passed", "Failed", "Passed"],
        "value": [1, 2, 3, 4, 5]
    })
    filename2 = _store_resource(df2, manifest_path, "test_data2", "Test", "csv")
    
    result2 = subset_dataset(
        input_filename=filename2,
        project_manifest_path=manifest_path,
        output_filename="result2",
        explanation="Keep Passed",
        filter_condition='status == "Passed"'
    )
    
    assert result2["n_rows"] == 3
    df_result2 = _load_resource(manifest_path, result2["output_filename"])
    assert all(df_result2["status"] == "Passed")
    
    # Test 3: Comparison operators
    result3 = subset_dataset(
        input_filename=filename1,
        project_manifest_path=manifest_path,
        output_filename="result3",
        explanation="Keep A > 3",
        filter_condition="A > 3"
    )
    
    assert result3["n_rows"] == 2
    df_result3 = _load_resource(manifest_path, result3["output_filename"])
    assert all(df_result3["A"] > 3)
    
    # Test 4: Boolean columns
    df4 = pd.DataFrame({
        "is_active": [True, False, True, False, True],
        "value": [1, 2, 3, 4, 5]
    })
    filename4 = _store_resource(df4, manifest_path, "test_data4", "Test", "csv")
    
    result4 = subset_dataset(
        input_filename=filename4,
        project_manifest_path=manifest_path,
        output_filename="result4",
        explanation="Keep active",
        filter_condition="is_active == True"
    )
    
    assert result4["n_rows"] == 3
    df_result4 = _load_resource(manifest_path, result4["output_filename"])
    assert all(df_result4["is_active"] == True)
    
    # Test 5: Multiple conditions with AND
    df5 = pd.DataFrame({
        "TPSA": [10, 25, 30, 15, 40],
        "MolWt": [200, 400, 600, 300, 250]
    })
    filename5 = _store_resource(df5, manifest_path, "test_data5", "Test", "csv")
    
    result5 = subset_dataset(
        input_filename=filename5,
        project_manifest_path=manifest_path,
        output_filename="result5",
        explanation="TPSA > 20 and MolWt < 500",
        filter_condition="TPSA > 20 and MolWt < 500"
    )
    
    assert result5["n_rows"] == 2
    df_result5 = _load_resource(manifest_path, result5["output_filename"])
    assert all(df_result5["TPSA"] > 20)
    assert all(df_result5["MolWt"] < 500)
    
    # Test 6: Multiple conditions with OR
    result6 = subset_dataset(
        input_filename=filename1,
        project_manifest_path=manifest_path,
        output_filename="result6",
        explanation="A < 2 or A > 4",
        filter_condition="A < 2 or A > 4"
    )
    
    assert result6["n_rows"] == 2
    df_result6 = _load_resource(manifest_path, result6["output_filename"])
    assert set(df_result6["A"].tolist()) == {1, 5}
    
    # Test 7: Null checks
    df7 = pd.DataFrame({"A": [1, None, 3, None, 5], "B": [10, 20, 30, 40, 50]})
    filename7 = _store_resource(df7, manifest_path, "test_data7", "Test", "csv")
    
    result7 = subset_dataset(
        input_filename=filename7,
        project_manifest_path=manifest_path,
        output_filename="result7",
        explanation="Keep null A",
        filter_condition="A.isnull()"
    )
    
    assert result7["n_rows"] == 2
    df_result7 = _load_resource(manifest_path, result7["output_filename"])
    assert df_result7["A"].isnull().sum() == 2


def test_drop_duplicate_rows(session_workdir, request):
    """Test dropping duplicate rows."""
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import drop_duplicate_rows
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store test dataset with duplicates
    df = pd.DataFrame({"A": [1, 2, 2, 3], "B": [10, 20, 20, 30]})
    filename = _store_resource(df, manifest_path, "test_data_dedup", "Test", "csv")
    
    # Drop duplicates
    result = drop_duplicate_rows(
        input_filename=filename,
        subset_columns=None,
        project_manifest_path=manifest_path,
        output_filename="deduped_data",
        explanation="Remove duplicates"
    )
    
    assert "output_filename" in result
    assert "n_rows_after" in result
    assert result["n_rows_after"] == 3  # Should have 3 unique rows


def test_drop_empty_rows(session_workdir, request):
    """Test dropping rows where ALL values are null."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import drop_empty_rows
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store test dataset with one completely empty row
    df = pd.DataFrame({"A": [1, None, None, 4], "B": [10, None, None, 40]})
    filename = _store_resource(df, manifest_path, "test_data_empty", "Test", "csv")
    
    # Drop empty rows (should only drop row where BOTH A and B are None)
    result = drop_empty_rows(
        input_filename=filename,
        project_manifest_path=manifest_path,
        output_filename="clean_data",
        explanation="Remove completely empty rows"
    )
    
    assert "output_filename" in result
    assert "n_rows_after" in result
    assert "n_rows_before" in result
    assert result["n_rows_before"] == 4
    assert result["n_rows_after"] == 2  # Only rows 0 and 3 should remain
    
    # Verify the correct rows were kept
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert len(df_result) == 2
    assert list(df_result["A"].dropna()) == [1.0, 4.0]


def test_drop_columns(session_workdir, request):
    """Test dropping columns."""
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import drop_columns
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store test dataset
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    filename = _store_resource(df, manifest_path, "test_data_dropcol", "Test", "csv")
    
    # Drop column B
    result = drop_columns(
        input_filename=filename,
        columns_to_drop=["B"],
        project_manifest_path=manifest_path,
        output_filename="reduced_data",
        explanation="Remove column B"
    )
    
    assert "output_filename" in result
    assert "columns_remaining" in result
    assert "B" not in result["columns_remaining"]
    assert len(result["columns_remaining"]) == 2


def test_keep_columns(session_workdir, request):
    """Test keeping specific columns."""
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import keep_columns
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store test dataset
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    filename = _store_resource(df, manifest_path, "test_data_keepcol", "Test", "csv")
    
    # Keep only columns A and C
    result = keep_columns(
        input_filename=filename,
        columns_to_keep=["A", "C"],
        project_manifest_path=manifest_path,
        output_filename="reduced_data",
        explanation="Keep only A and C"
    )
    
    assert "output_filename" in result
    assert "columns_kept" in result
    assert set(result["columns_kept"]) == {"A", "C"}
    assert len(result["columns_kept"]) == 2


def test_transform_column(session_workdir, request):
    """Test transforming a column."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import transform_column
    import numpy as np
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Test 1: Simple multiplication transformation
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    filename1 = _store_resource(df1, manifest_path, "test_data_transform1", "Test", "csv")
    
    result1 = transform_column(
        input_filename=filename1,
        expression="A_doubled = A * 2",
        project_manifest_path=manifest_path,
        output_filename="transformed_data1",
        explanation="Multiply A by 2"
    )
    
    assert "output_filename" in result1
    assert "n_rows" in result1
    assert result1["n_rows"] == 3
    assert "preview" in result1
    assert "expression" in result1
    assert "columns" in result1
    assert "A_doubled" in result1["columns"]
    
    df_result1 = _load_resource(manifest_path, result1["output_filename"])
    assert "A_doubled" in df_result1.columns
    assert list(df_result1["A_doubled"]) == [2, 4, 6]
    
    # Test 2: Ki to pKi transformation (pKi = -log10(Ki * 1e-9))
    df2 = pd.DataFrame({"Ki": [1.0, 10.0, 100.0, 1000.0], "compound": ["A", "B", "C", "D"]})
    filename2 = _store_resource(df2, manifest_path, "test_data_transform2", "Test", "csv")
    
    result2 = transform_column(
        input_filename=filename2,
        expression="pKi = -log10(Ki * 1e-9)",
        project_manifest_path=manifest_path,
        output_filename="transformed_data2",
        explanation="Convert Ki (nM) to pKi"
    )
    
    assert "output_filename" in result2
    assert "pKi" in result2["columns"]
    
    df_result2 = _load_resource(manifest_path, result2["output_filename"])
    assert "pKi" in df_result2.columns
    # Verify pKi values: Ki=1nM -> pKi=9, Ki=10nM -> pKi=8, Ki=100nM -> pKi=7, Ki=1000nM -> pKi=6
    expected_pKi = [9.0, 8.0, 7.0, 6.0]
    assert np.allclose(df_result2["pKi"].values, expected_pKi, rtol=1e-9)
    
    # Test 3: String replacement
    df3 = pd.DataFrame({"text": ["Hello", "World", "Test"]})
    filename3 = _store_resource(df3, manifest_path, "test_data_transform3", "Test", "csv")
    
    result3 = transform_column(
        input_filename=filename3,
        expression="text = text.str.replace('o', 'X')",
        project_manifest_path=manifest_path,
        output_filename="transformed_data3",
        explanation="Replace 'o' with 'X'"
    )
    
    df_result3 = _load_resource(manifest_path, result3["output_filename"])
    assert list(df_result3["text"]) == ["HellX", "WXrld", "Test"]
    
    # Test 4: Type conversion
    df4 = pd.DataFrame({"value": ["1", "2", "3"]})
    filename4 = _store_resource(df4, manifest_path, "test_data_transform4", "Test", "csv")
    
    result4 = transform_column(
        input_filename=filename4,
        expression="value = value.astype(int)",
        project_manifest_path=manifest_path,
        output_filename="transformed_data4",
        explanation="Convert string to int"
    )
    
    df_result4 = _load_resource(manifest_path, result4["output_filename"])
    assert df_result4["value"].dtype == np.int64 or df_result4["value"].dtype == np.int32
    assert list(df_result4["value"]) == [1, 2, 3]
    
    # Test 5: Chained operations (string replacement + type conversion)
    df5 = pd.DataFrame({"Ki": ["<1.5", "10.0", "<2.5", "50.0"]})
    filename5 = _store_resource(df5, manifest_path, "test_data_transform5", "Test", "csv")
    
    result5 = transform_column(
        input_filename=filename5,
        expression="Ki = Ki.str.replace('<', '').astype(float)",
        project_manifest_path=manifest_path,
        output_filename="transformed_data5",
        explanation="Remove '<' and convert to float"
    )
    
    df_result5 = _load_resource(manifest_path, result5["output_filename"])
    assert df_result5["Ki"].dtype == np.float64
    assert list(df_result5["Ki"]) == [1.5, 10.0, 2.5, 50.0]


def test_scramble_column(session_workdir, request):
    """Test scrambling a column for permutation testing."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import scramble_column
    import numpy as np
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Test 1: Scramble column with fixed seed for reproducibility
    df1 = pd.DataFrame({
        "smiles": ["CCO", "CCC", "CCCO", "CCCC", "CCCCO"],
        "activity": [1, 2, 3, 4, 5]
    })
    filename1 = _store_resource(df1, manifest_path, "test_data_scramble", "Test", "csv")
    
    result1 = scramble_column(
        input_filename=filename1,
        column_name="activity",
        project_manifest_path=manifest_path,
        output_filename="scrambled_data",
        explanation="Scrambled activity for permutation test",
        random_seed=42
    )
    
    assert "output_filename" in result1
    assert result1["output_filename"] != filename1  # Should create new file
    assert "n_rows" in result1
    assert result1["n_rows"] == 5
    assert "scrambled_column" in result1
    assert result1["scrambled_column"] == "activity"
    assert "random_seed" in result1
    assert result1["random_seed"] == 42
    assert "columns" in result1
    assert set(result1["columns"]) == {"smiles", "activity"}
    assert "preview" in result1
    
    df_result1 = _load_resource(manifest_path, result1["output_filename"])
    
    # Verify that smiles column is unchanged
    assert list(df_result1["smiles"]) == list(df1["smiles"])
    
    # Verify that activity values are different order but same set
    assert set(df_result1["activity"]) == set(df1["activity"])
    assert list(df_result1["activity"]) != list(df1["activity"])  # Should be shuffled
    
    # Test 2: Same seed should give same result
    result2 = scramble_column(
        input_filename=filename1,
        column_name="activity",
        project_manifest_path=manifest_path,
        output_filename="scrambled_data2",
        explanation="Scrambled activity with same seed",
        random_seed=42
    )
    
    df_result2 = _load_resource(manifest_path, result2["output_filename"])
    assert list(df_result2["activity"]) == list(df_result1["activity"])
    
    # Test 3: Different seed should give different result
    result3 = scramble_column(
        input_filename=filename1,
        column_name="activity",
        project_manifest_path=manifest_path,
        output_filename="scrambled_data3",
        explanation="Scrambled activity with different seed",
        random_seed=123
    )
    
    df_result3 = _load_resource(manifest_path, result3["output_filename"])
    assert list(df_result3["activity"]) != list(df_result1["activity"])
    
    # Test 4: Error handling - invalid column name
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        scramble_column(
            input_filename=filename1,
            column_name="nonexistent",
            project_manifest_path=manifest_path,
            output_filename="should_fail",
            explanation="Should fail"
        )
    
    # Test 5: Scramble without seed (should work but not reproducible)
    result5 = scramble_column(
        input_filename=filename1,
        column_name="activity",
        project_manifest_path=manifest_path,
        output_filename="scrambled_no_seed",
        explanation="Scrambled without seed"
    )
    
    assert result5["random_seed"] is None
    df_result5 = _load_resource(manifest_path, result5["output_filename"])
    assert set(df_result5["activity"]) == set(df1["activity"])


def test_combine_datasets_vertical(session_workdir, request):
    """Test combining datasets vertically."""
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource, create_project_manifest
    from molml_mcp.tools.core.dataset_ops import combine_datasets_vertical
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store two test datasets
    df1 = pd.DataFrame({"smiles": ["CCO", "CCC"], "label": [0, 1]})
    df2 = pd.DataFrame({"smiles": ["CCCO", "CCCC"], "label": [1, 0]})
    
    filename1 = _store_resource(df1, manifest_path, "dataset1", "First dataset", "csv")
    filename2 = _store_resource(df2, manifest_path, "dataset2", "Second dataset", "csv")
    
    # Combine datasets (keep all)
    result = combine_datasets_vertical(
        input_filenames=[filename1, filename2],
        project_manifest_path=manifest_path,
        output_filename="combined",
        explanation="Combined datasets",
        handle_duplicates='keep_all',
        verify_columns=True
    )
    
    assert "output_filename" in result
    assert result["n_rows"] == 4
    assert result["n_rows_per_input"][filename1] == 2
    assert result["n_rows_per_input"][filename2] == 2
    assert result["n_duplicates_dropped"] == 0
    assert set(result["columns"]) == {"smiles", "label"}
    
    # Verify combined data
    df_combined = _load_resource(manifest_path, result["output_filename"])
    assert len(df_combined) == 4
    assert list(df_combined["smiles"]) == ["CCO", "CCC", "CCCO", "CCCC"]
    
    # Test drop_duplicates
    df3 = pd.DataFrame({"smiles": ["CCO", "CCCCO"], "label": [0, 1]})  # CCO is duplicate
    filename3 = _store_resource(df3, manifest_path, "dataset3", "Third dataset", "csv")
    
    result2 = combine_datasets_vertical(
        input_filenames=[filename1, filename3],
        project_manifest_path=manifest_path,
        output_filename="combined_dedup",
        explanation="Combined with deduplication",
        handle_duplicates='drop_duplicates'
    )
    
    assert result2["n_rows"] == 3  # 4 rows - 1 duplicate
    assert result2["n_duplicates_dropped"] == 1
    
    # Test column mismatch error
    df4 = pd.DataFrame({"smiles": ["CCCCCC"], "activity": [0.5]})  # Different columns
    filename4 = _store_resource(df4, manifest_path, "dataset4", "Fourth dataset", "csv")
    
    with pytest.raises(ValueError, match="Column mismatch"):
        combine_datasets_vertical(
            input_filenames=[filename1, filename4],
            project_manifest_path=manifest_path,
            output_filename="combined_fail",
            explanation="Should fail",
            verify_columns=True
        )
    
    # Test with verify_columns=False (should work with NaN fill)
    result3 = combine_datasets_vertical(
        input_filenames=[filename1, filename4],
        project_manifest_path=manifest_path,
        output_filename="combined_mixed",
        explanation="Combined with mismatched columns",
        verify_columns=False
    )
    
    assert result3["n_rows"] == 3
    df_mixed = _load_resource(manifest_path, result3["output_filename"])
    assert "activity" in df_mixed.columns
    assert "label" in df_mixed.columns


def test_combine_datasets_horizontal(session_workdir, request):
    """Test horizontal dataset combination (column-wise concatenation)."""
    from molml_mcp.tools.core.dataset_ops import combine_datasets_horizontal
    from molml_mcp.infrastructure.resources import create_project_manifest, _load_resource, _store_resource
    
    # Setup
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create left dataset (molecules with identifiers)
    left_data = pd.DataFrame({
        "smiles": ["CCO", "CC(C)O", "c1ccccc1"],
        "mol_id": ["mol1", "mol2", "mol3"]
    })
    left_file = _store_resource(left_data, manifest_path, "left_data", "Left dataset", 'csv')
    
    # Create right dataset (descriptors)
    right_data = pd.DataFrame({
        "mw": [46.07, 60.10, 78.11],
        "logp": [-0.3, 0.1, 1.6]
    })
    right_file = _store_resource(right_data, manifest_path, "right_data", "Right dataset", 'csv')
    
    # Test 1: Basic horizontal combination with verification
    result = combine_datasets_horizontal(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_file,
        output_filename="combined_horizontal",
        explanation="Combine molecules with descriptors",
        verify_alignment=True
    )
    
    assert "output_filename" in result
    assert result["n_rows"] == 3
    assert result["n_columns"] == 4  # 2 from left + 2 from right
    assert result["n_columns_left"] == 2
    assert result["n_columns_right"] == 2
    assert result["alignment_verified"] is True
    assert set(result["columns"]) == {"smiles", "mol_id", "mw", "logp"}
    
    # Verify actual data
    df_combined = _load_resource(manifest_path, result["output_filename"])
    assert len(df_combined) == 3
    assert list(df_combined["smiles"]) == ["CCO", "CC(C)O", "c1ccccc1"]
    assert list(df_combined["mw"]) == [46.07, 60.10, 78.11]
    
    # Test 2: Overlapping column names should raise error
    right_overlap = pd.DataFrame({
        "smiles": ["XXX", "YYY", "ZZZ"],  # Overlaps with left
        "mw": [46.07, 60.10, 78.11]
    })
    right_overlap_file = _store_resource(right_overlap, manifest_path, "right_overlap", "Overlap test", 'csv')
    
    with pytest.raises(ValueError, match="overlapping column names"):
        combine_datasets_horizontal(
            project_manifest_path=manifest_path,
            left_filename=left_file,
            right_filename=right_overlap_file,
            output_filename="should_fail",
            explanation="Should fail"
        )
    
    # Test 3: Mismatched row counts with verify_alignment=True should raise
    right_short = pd.DataFrame({
        "mw": [46.07, 60.10],  # Only 2 rows
        "logp": [-0.3, 0.1]
    })
    right_short_file = _store_resource(right_short, manifest_path, "right_short", "Short dataset", 'csv')
    
    with pytest.raises(ValueError, match="Row count mismatch"):
        combine_datasets_horizontal(
            project_manifest_path=manifest_path,
            left_filename=left_file,
            right_filename=right_short_file,
            output_filename="should_fail",
            explanation="Should fail",
            verify_alignment=True
        )
    
    # Test 4: Mismatched row counts with on_mismatch='warn' should work
    result_warn = combine_datasets_horizontal(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_short_file,
        output_filename="warn_test",
        explanation="Test warning mode",
        verify_alignment=True,
        on_mismatch='warn'
    )
    
    assert result_warn["alignment_verified"] is False
    # pandas concat with axis=1 will create NaN for missing values
    df_warn = _load_resource(manifest_path, result_warn["output_filename"])
    assert len(df_warn) == 3  # Takes max length
    
    # Test 5: No verification (verify_alignment=False)
    result_no_verify = combine_datasets_horizontal(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_short_file,
        output_filename="no_verify",
        explanation="No verification",
        verify_alignment=False
    )
    
    assert result_no_verify["alignment_verified"] is False
    assert result_no_verify["n_rows"] == 3
    
    # Test 6: Perfectly aligned datasets should pass verification
    result_aligned = combine_datasets_horizontal(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_file,
        output_filename="aligned_test",
        explanation="Test alignment verification passes",
        verify_alignment=True
    )
    
    assert result_aligned["alignment_verified"] is True
    assert result_aligned["n_rows"] == 3
    assert result_aligned["n_columns"] == 4


def test_merge_datasets_on_smiles(session_workdir, request):
    """Test merging datasets based on SMILES structures."""
    from molml_mcp.tools.core.dataset_ops import merge_datasets_on_smiles
    from molml_mcp.infrastructure.resources import create_project_manifest, _load_resource, _store_resource
    
    # Setup
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create left dataset (bioactivity data)
    left_data = pd.DataFrame({
        "smiles": ["CCO", "CC(C)O", "c1ccccc1", "CCCC"],
        "activity": [5.2, 6.1, 7.3, 4.8],
        "source": ["A", "A", "B", "B"]
    })
    left_file = _store_resource(left_data, manifest_path, "left_bio", "Bioactivity data", 'csv')
    
    # Create right dataset (descriptors) - with some overlap
    right_data = pd.DataFrame({
        "canonical_smiles": ["CCO", "c1ccccc1", "CC(C)C", "CCCCC"],  # Different notation for isopropanol
        "mw": [46.07, 78.11, 58.12, 72.15],
        "logp": [-0.3, 1.6, 0.8, 2.3]
    })
    right_file = _store_resource(right_data, manifest_path, "right_desc", "Descriptor data", 'csv')
    
    # Test 1: Inner join with canonicalization (default)
    result_inner = merge_datasets_on_smiles(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_file,
        output_filename="merged_inner",
        explanation="Inner join bioactivity with descriptors",
        left_smiles_col="smiles",
        right_smiles_col="canonical_smiles",
        how="inner"
    )
    
    assert "output_filename" in result_inner
    assert result_inner["n_rows_left"] == 4
    assert result_inner["n_rows_right"] == 4
    assert result_inner["n_matched"] == 2  # CCO and benzene match
    assert result_inner["n_rows"] == 2  # Only matched molecules
    assert result_inner["merge_type"] == "inner"
    assert result_inner["canonicalized"] is True
    assert result_inner["smiles_column"] == "smiles"
    
    # Verify data
    df_inner = _load_resource(manifest_path, result_inner["output_filename"])
    assert len(df_inner) == 2
    assert "smiles" in df_inner.columns
    assert "activity" in df_inner.columns
    assert "mw" in df_inner.columns
    assert set(df_inner["smiles"]) == {"CCO", "c1ccccc1"}
    
    # Test 2: Left join - keep all left molecules
    result_left = merge_datasets_on_smiles(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_file,
        output_filename="merged_left",
        explanation="Left join to keep all bioactivity data",
        left_smiles_col="smiles",
        right_smiles_col="canonical_smiles",
        how="left"
    )
    
    assert result_left["n_rows"] == 4  # All left molecules kept
    df_left = _load_resource(manifest_path, result_left["output_filename"])
    assert len(df_left) == 4
    # Check that unmatched molecules have NaN for right columns
    butane_row = df_left[df_left["smiles"] == "CCCC"].iloc[0]
    assert pd.isna(butane_row["mw"])
    
    # Test 3: Right join - keep all right molecules
    result_right = merge_datasets_on_smiles(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_file,
        output_filename="merged_right",
        explanation="Right join to keep all descriptor data",
        left_smiles_col="smiles",
        right_smiles_col="canonical_smiles",
        how="right"
    )
    
    assert result_right["n_rows"] == 4  # All right molecules kept
    df_right = _load_resource(manifest_path, result_right["output_filename"])
    assert len(df_right) == 4
    # Check that unmatched molecules have NaN for left columns
    isobutane_row = df_right[df_right["smiles"] == "CC(C)C"].iloc[0]
    assert pd.isna(isobutane_row["activity"])
    
    # Test 4: Outer join - keep all molecules
    result_outer = merge_datasets_on_smiles(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_file,
        output_filename="merged_outer",
        explanation="Outer join to keep everything",
        left_smiles_col="smiles",
        right_smiles_col="canonical_smiles",
        how="outer"
    )
    
    assert result_outer["n_rows"] == 6  # 4 unique from left + 2 unique from right
    df_outer = _load_resource(manifest_path, result_outer["output_filename"])
    assert len(df_outer) == 6
    
    # Test 5: Merge without canonicalization
    result_no_canon = merge_datasets_on_smiles(
        project_manifest_path=manifest_path,
        left_filename=left_file,
        right_filename=right_file,
        output_filename="merged_no_canon",
        explanation="Merge without canonicalization",
        left_smiles_col="smiles",
        right_smiles_col="canonical_smiles",
        how="inner",
        canonicalize=False
    )
    
    assert result_no_canon["canonicalized"] is False
    # Without canonicalization, fewer matches (exact string matching only)
    df_no_canon = _load_resource(manifest_path, result_no_canon["output_filename"])
    assert len(df_no_canon) <= 2
    
    # Test 6: Overlapping column names with suffixes
    left_overlap = pd.DataFrame({
        "smiles": ["CCO", "CC(C)O"],
        "value": [10, 20],
        "label": ["A", "B"]
    })
    left_overlap_file = _store_resource(left_overlap, manifest_path, "left_overlap", "Left overlap", 'csv')
    
    right_overlap = pd.DataFrame({
        "smiles": ["CCO", "CC(C)O"],
        "value": [100, 200],  # Same column name as left
        "type": ["X", "Y"]
    })
    right_overlap_file = _store_resource(right_overlap, manifest_path, "right_overlap", "Right overlap", 'csv')
    
    result_suffix = merge_datasets_on_smiles(
        project_manifest_path=manifest_path,
        left_filename=left_overlap_file,
        right_filename=right_overlap_file,
        output_filename="merged_suffix",
        explanation="Test suffix handling",
        left_smiles_col="smiles",
        right_smiles_col="smiles",
        how="inner",
        suffixes=("_left", "_right")
    )
    
    df_suffix = _load_resource(manifest_path, result_suffix["output_filename"])
    assert "value_left" in df_suffix.columns
    assert "value_right" in df_suffix.columns
    assert df_suffix.loc[0, "value_left"] == 10
    assert df_suffix.loc[0, "value_right"] == 100
    
    # Test 7: Invalid SMILES column name should raise error
    with pytest.raises(ValueError, match="not found"):
        merge_datasets_on_smiles(
            project_manifest_path=manifest_path,
            left_filename=left_file,
            right_filename=right_file,
            output_filename="should_fail",
            explanation="Should fail",
            left_smiles_col="nonexistent_col",
            right_smiles_col="canonical_smiles",
            how="inner"
        )
    
    # Test 8: Invalid merge type should raise error
    with pytest.raises(ValueError, match="Invalid how"):
        merge_datasets_on_smiles(
            project_manifest_path=manifest_path,
            left_filename=left_file,
            right_filename=right_file,
            output_filename="should_fail",
            explanation="Should fail",
            left_smiles_col="smiles",
            right_smiles_col="canonical_smiles",
            how="invalid_type"
        )
    
    # Test 9: Handle invalid SMILES gracefully
    left_invalid = pd.DataFrame({
        "smiles": ["CCO", "INVALID_SMILES", "c1ccccc1"],
        "activity": [5.2, 6.1, 7.3]
    })
    left_invalid_file = _store_resource(left_invalid, manifest_path, "left_invalid", "Invalid SMILES", 'csv')
    
    result_invalid = merge_datasets_on_smiles(
        project_manifest_path=manifest_path,
        left_filename=left_invalid_file,
        right_filename=right_file,
        output_filename="merged_invalid",
        explanation="Handle invalid SMILES",
        left_smiles_col="smiles",
        right_smiles_col="canonical_smiles",
        how="inner"
    )
    
    # Should succeed but drop invalid SMILES
    df_invalid = _load_resource(manifest_path, result_invalid["output_filename"])
    assert len(df_invalid) == 2  # Only valid matches


def test_get_all_dataset_tools():
    """Test getting all dataset tools."""
    from molml_mcp.tools.core.dataset_ops import get_all_dataset_tools
    
    tools = get_all_dataset_tools()
    
    assert isinstance(tools, list)
    assert len(tools) > 0


def test_read_txt(session_workdir, request):
    """Test reading text files from project resources."""
    from molml_mcp.tools.core.dataset_ops import read_txt
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store a text file
    test_content = "This is a test report.\nLine 2.\nLine 3."
    txt_filename = _store_resource(
        test_content,
        manifest_path,
        "test_report",
        "Test text report",
        'txt'
    )
    
    # Read the text file
    result = read_txt(txt_filename, manifest_path)
    
    assert "filename" in result
    assert result["filename"] == txt_filename
    assert "content" in result
    assert result["content"] == test_content
    assert "n_chars" in result
    assert result["n_chars"] == len(test_content)
    assert "n_lines" in result
    assert result["n_lines"] == 3


def test_read_json(session_workdir, request):
    """Test reading JSON files from project resources."""
    from molml_mcp.tools.core.dataset_ops import read_json
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    import json
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store a JSON file
    test_data = {
        "analysis": "scaffold_report",
        "n_molecules": 1000,
        "n_scaffolds": 150,
        "metrics": {
            "gini": 0.45,
            "shannon": 3.2
        }
    }
    json_filename = _store_resource(
        test_data,
        manifest_path,
        "test_analysis",
        "Test JSON data",
        'json'
    )
    
    # Read the JSON file
    result = read_json(json_filename, manifest_path)
    
    assert "filename" in result
    assert result["filename"] == json_filename
    assert "data" in result
    assert result["data"] == test_data
    assert "formatted_text" in result
    assert isinstance(result["formatted_text"], str)
    # Verify it's valid JSON
    parsed = json.loads(result["formatted_text"])
    assert parsed == test_data
    assert "type" in result
    assert result["type"] == "dict"


def test_read_json_list(session_workdir, request):
    """Test reading JSON files containing lists."""
    from molml_mcp.tools.core.dataset_ops import read_json
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store a list as JSON
    test_data = [
        {"scaffold": "c1ccccc1", "count": 50},
        {"scaffold": "CCO", "count": 30}
    ]
    json_filename = _store_resource(
        test_data,
        manifest_path,
        "test_list",
        "Test list data",
        'json'
    )
    
    # Read the JSON file
    result = read_json(json_filename, manifest_path)
    
    assert result["data"] == test_data
    assert result["type"] == "list"


def test_read_txt_invalid_type(session_workdir, request):
    """Test error handling when reading wrong resource type as text."""
    from molml_mcp.tools.core.dataset_ops import read_txt
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    import pandas as pd
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store a CSV file (DataFrame)
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    csv_filename = _store_resource(df, manifest_path, "test_csv", "Test CSV", 'csv')
    
    # Try to read as text - should fail
    with pytest.raises(ValueError, match="Expected text content"):
        read_txt(csv_filename, manifest_path)


def test_read_json_invalid_type(session_workdir, request):
    """Test error handling when reading wrong resource type as JSON."""
    from molml_mcp.tools.core.dataset_ops import read_json
    from molml_mcp.infrastructure.resources import _store_resource, create_project_manifest
    
    # Create test-specific subdirectory
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Store a text file
    txt_filename = _store_resource("Plain text", manifest_path, "test_txt", "Test text", 'txt')
    
    # Try to read as JSON - should fail
    with pytest.raises(ValueError, match="Expected JSON data"):
        read_json(txt_filename, manifest_path)

