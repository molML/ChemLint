import pytest
from pathlib import Path


def test_store_load_resource(session_workdir):
    from chemlint.infrastructure.resources import _store_resource, _load_resource

    data = {"a": 1, "b": 2}
    filename = _store_resource(data, session_workdir / "test_manifest.json", "store_resource_test_data", "Test data storage", "json")
    loaded_data = _load_resource(session_workdir / "test_manifest.json", filename)

    assert data == loaded_data


def test_create_project_manifest(session_workdir):
    from chemlint.infrastructure.resources import create_project_manifest
    
    manifest = create_project_manifest(str(session_workdir), "test_project")
    
    assert manifest["project_name"] == "test_project"
    assert "created_at" in manifest
    assert manifest["resources"] == []
    assert (session_workdir / "test_project_manifest.json").exists()


def test_read_project_manifest(session_workdir):
    from chemlint.infrastructure.resources import create_project_manifest, read_project_manifest
    
    create_project_manifest(str(session_workdir), "read_test")
    manifest = read_project_manifest(str(session_workdir / "read_test_manifest.json"))
    
    assert manifest["project_name"] == "read_test"
    assert "resources" in manifest


def test_add_to_project_manifest(session_workdir):
    from chemlint.infrastructure.resources import create_project_manifest, add_to_project_manifest, read_project_manifest
    
    manifest_path = str(session_workdir / "add_test_manifest.json")
    create_project_manifest(str(session_workdir), "add_test")
    
    add_to_project_manifest(
        manifest_path,
        "test_file_12345678.csv",
        "csv",
        "Test CSV file"
    )
    
    manifest = read_project_manifest(manifest_path)
    assert len(manifest["resources"]) == 1
    assert manifest["resources"][0]["filename"] == "test_file_12345678.csv"
    assert manifest["resources"][0]["type_tag"] == "csv"


def test_remove_from_project_manifest(session_workdir):
    from chemlint.infrastructure.resources import (
        create_project_manifest, 
        add_to_project_manifest, 
        remove_from_project_manifest,
        read_project_manifest
    )
    
    manifest_path = str(session_workdir / "remove_test_manifest.json")
    create_project_manifest(str(session_workdir), "remove_test")
    
    add_to_project_manifest(manifest_path, "file1.csv", "csv", "File 1")
    add_to_project_manifest(manifest_path, "file2.csv", "csv", "File 2")
    
    removed = remove_from_project_manifest(manifest_path, "file1.csv")
    
    assert removed["filename"] == "file1.csv"
    manifest = read_project_manifest(manifest_path)
    assert len(manifest["resources"]) == 1
    assert manifest["resources"][0]["filename"] == "file2.csv"


def test_list_untracked_resources(session_workdir):
    from chemlint.infrastructure.resources import (
        create_project_manifest,
        add_to_project_manifest,
        list_untracked_resources_in_project
    )
    
    manifest_path = str(session_workdir / "untracked_test_manifest.json")
    create_project_manifest(str(session_workdir), "untracked_test")
    
    # Create a file that's not in manifest
    (session_workdir / "orphan_file.txt").write_text("orphan")
    
    # Add a tracked file to manifest
    add_to_project_manifest(manifest_path, "tracked_file.csv", "csv", "Tracked")
    
    untracked = list_untracked_resources_in_project(manifest_path)
    
    assert "orphan_file.txt" in untracked
    assert "tracked_file.csv" not in untracked


def test_get_supported_resource_types():
    from chemlint.infrastructure.resources import get_supported_resource_types
    
    types = get_supported_resource_types()
    
    assert isinstance(types, list)
    assert "csv" in types
    assert "json" in types
    assert "model" in types


def test_store_and_load_csv(session_workdir):
    import pandas as pd
    from chemlint.infrastructure.resources import _store_resource, _load_resource, create_project_manifest
    
    manifest_path = str(session_workdir / "csv_test_manifest.json")
    create_project_manifest(str(session_workdir), "csv_test")
    
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    filename = _store_resource(df, manifest_path, "test_df", "Test DataFrame", "csv")
    loaded_df = _load_resource(manifest_path, filename)
    
    assert loaded_df.equals(df)


def test_store_and_load_model(session_workdir):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from chemlint.infrastructure.resources import _store_resource, _load_resource, create_project_manifest
    
    manifest_path = str(session_workdir / "model_test_manifest.json")
    create_project_manifest(str(session_workdir), "model_test")
    
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Use the standard model format (dict with models list)
    model_data = {
        "models": [model],
        "data_splits": [{"training": {}, "validation": {}}],
        "model_algorithm": "random_forest_classifier",
        "hyperparameters": {},
        "random_state": 42,
        "n_features": 2
    }
    
    filename = _store_resource(model_data, manifest_path, "test_model", "Test RF model", "model")
    loaded_model_data = _load_resource(manifest_path, filename)
    
    # Verify structure
    assert "models" in loaded_model_data
    assert isinstance(loaded_model_data["models"], list)
    assert len(loaded_model_data["models"]) == 1
    
    # Verify loaded model works
    loaded_model = loaded_model_data["models"][0]
    pred = loaded_model.predict([[2, 3]])
    assert pred.shape == (1,)


def test_load_nonexistent_resource(session_workdir):
    from chemlint.infrastructure.resources import _load_resource, create_project_manifest
    
    manifest_path = str(session_workdir / "error_test_manifest.json")
    create_project_manifest(str(session_workdir), "error_test")
    
    with pytest.raises(ValueError, match="not found in manifest"):
        _load_resource(manifest_path, "nonexistent_file.csv")


def test_create_manifest_twice_fails(session_workdir):
    from chemlint.infrastructure.resources import create_project_manifest
    
    create_project_manifest(str(session_workdir), "duplicate_test")
    
    with pytest.raises(FileExistsError):
        create_project_manifest(str(session_workdir), "duplicate_test")


def test_check_data_directory_content_with_projects(session_workdir):
    from chemlint.infrastructure.resources import (
        create_project_manifest, 
        _store_resource,
        check_data_directory_content
    )
    import pandas as pd
    
    # Create isolated test directory
    test_dir = session_workdir / "test_check_content"
    test_dir.mkdir()
    
    # Create two projects with resources
    manifest1 = str(test_dir / "project1_manifest.json")
    manifest2 = str(test_dir / "project2_manifest.json")
    
    create_project_manifest(str(test_dir), "project1")
    create_project_manifest(str(test_dir), "project2")
    
    # Add resources to project1
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _store_resource(df1, manifest1, "data1", "First dataset", "csv")
    _store_resource(df1, manifest1, "data2", "Second dataset", "csv")
    
    # Add resource to project2
    df2 = pd.DataFrame({"x": [5, 6]})
    _store_resource(df2, manifest2, "data3", "Third dataset", "csv")
    
    # Check directory content
    result = check_data_directory_content(str(test_dir))
    
    assert result["n_projects"] == 2
    assert len(result["projects"]) == 2
    assert "Found 2 projects" in result["message"]
    
    # Verify project details
    project_names = {p["project_name"] for p in result["projects"]}
    assert project_names == {"project1", "project2"}
    
    # Find project1 and verify its resources
    project1 = next(p for p in result["projects"] if p["project_name"] == "project1")
    assert project1["n_resources"] == 2
    assert len(project1["resources"]) == 2
    
    # Find project2 and verify its resources
    project2 = next(p for p in result["projects"] if p["project_name"] == "project2")
    assert project2["n_resources"] == 1
    assert len(project2["resources"]) == 1


def test_check_data_directory_content_no_manifests(session_workdir):
    from chemlint.infrastructure.resources import check_data_directory_content
    
    # Create empty directory
    empty_dir = session_workdir / "empty"
    empty_dir.mkdir()
    
    result = check_data_directory_content(str(empty_dir))
    
    assert result["n_projects"] == 0
    assert result["projects"] == []
    assert "No project manifests found" in result["message"]


def test_check_data_directory_content_nonexistent(session_workdir):
    from chemlint.infrastructure.resources import check_data_directory_content
    
    result = check_data_directory_content(str(session_workdir / "nonexistent"))
    
    assert result["n_projects"] == 0
    assert result["projects"] == []
    assert "does not exist" in result["message"]


def test_check_data_directory_content_single_project(session_workdir):
    from chemlint.infrastructure.resources import (
        create_project_manifest,
        _store_resource,
        check_data_directory_content
    )
    import pandas as pd
    
    # Create isolated test directory
    test_dir = session_workdir / "test_single_project"
    test_dir.mkdir()
    
    manifest = str(test_dir / "solo_manifest.json")
    create_project_manifest(str(test_dir), "solo")
    
    # Add some resources
    df = pd.DataFrame({"x": [1, 2, 3]})
    _store_resource(df, manifest, "data1", "Dataset 1", "csv")
    _store_resource(df, manifest, "data2", "Dataset 2", "csv")
    _store_resource(df, manifest, "data3", "Dataset 3", "csv")
    
    result = check_data_directory_content(str(test_dir))
    
    assert result["n_projects"] == 1
    assert "Found 1 project with 3 tracked resources" in result["message"]
