"""Tests for outliers.py functions."""
import pandas as pd
import numpy as np
from pathlib import Path


def test_detect_outliers_zscore(session_workdir):
    """Test Z-score outlier detection."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.outliers import detect_outliers_zscore
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test data with clear outliers - need more data points for Z-score
    df = pd.DataFrame({
        'values': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 100],  # 100 is a clear outlier
        'normal': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    })
    
    input_filename = _store_resource(
        df, manifest_path, "test_data", "Test data with outliers", "csv"
    )
    
    # Detect outliers
    result = detect_outliers_zscore(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        columns=['values'],
        output_filename="zscore_checked",
        explanation="Z-score outlier detection"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "values_zscore_pass" in df_result.columns
    # Verify 1000 is detected as outlier (False = outlier)
    assert df_result.loc[df_result['values'] == 100, 'values_zscore_pass'].iloc[0] == False
    # Verify normal values pass
    assert df_result.loc[df_result['values'] == 15, 'values_zscore_pass'].iloc[0] == True


def test_detect_outliers_modified_zscore(session_workdir):
    """Test Modified Z-score outlier detection."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.outliers import detect_outliers_modified_zscore
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test data
    df = pd.DataFrame({
        'values': [1, 2, 3, 4, 5, 100]
    })
    
    input_filename = _store_resource(
        df, manifest_path, "test_data", "Test data", "csv"
    )
    
    # Detect outliers
    result = detect_outliers_modified_zscore(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        columns=['values'],
        output_filename="modified_zscore_checked",
        explanation="Modified Z-score outlier detection"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "values_modified_zscore_pass" in df_result.columns
    # Verify 100 is detected as outlier
    assert df_result.loc[df_result['values'] == 100, 'values_modified_zscore_pass'].iloc[0] == False
    # Verify normal values pass
    assert df_result.loc[df_result['values'] == 3, 'values_modified_zscore_pass'].iloc[0] == True


def test_detect_outliers_iqr(session_workdir):
    """Test IQR outlier detection."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.outliers import detect_outliers_iqr
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test data
    df = pd.DataFrame({
        'values': [1, 2, 3, 4, 5, 100]
    })
    
    input_filename = _store_resource(
        df, manifest_path, "test_data", "Test data", "csv"
    )
    
    # Detect outliers
    result = detect_outliers_iqr(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        columns=['values'],
        output_filename="iqr_checked",
        explanation="IQR outlier detection"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "values_iqr_pass" in df_result.columns
    # Verify 100 is detected as outlier
    assert df_result.loc[df_result['values'] == 100, 'values_iqr_pass'].iloc[0] == False
    # Verify normal values pass
    assert df_result.loc[df_result['values'] == 3, 'values_iqr_pass'].iloc[0] == True


def test_detect_outliers_grubbs(session_workdir):
    """Test Grubbs' test outlier detection."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.outliers import detect_outliers_grubbs
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test data
    df = pd.DataFrame({
        'values': [1, 2, 3, 4, 5, 100]
    })
    
    input_filename = _store_resource(
        df, manifest_path, "test_data", "Test data", "csv"
    )
    
    # Detect outliers
    result = detect_outliers_grubbs(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        columns=['values'],
        output_filename="grubbs_checked",
        explanation="Grubbs test outlier detection"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "values_grubbs_pass" in df_result.columns
    # Verify 100 is detected as outlier
    assert df_result.loc[df_result['values'] == 100, 'values_grubbs_pass'].iloc[0] == False
    # Verify normal values pass
    assert df_result.loc[df_result['values'] == 3, 'values_grubbs_pass'].iloc[0] == True


def test_detect_outliers_gesd(session_workdir):
    """Test Generalized ESD outlier detection."""
    from chemlint.infrastructure.resources import _store_resource, _load_resource
    from chemlint.tools.core.outliers import detect_outliers_gesd
    
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Create test data
    df = pd.DataFrame({
        'values': [1, 2, 3, 4, 5, 100, 200]
    })
    
    input_filename = _store_resource(
        df, manifest_path, "test_data", "Test data", "csv"
    )
    
    # Detect outliers
    result = detect_outliers_gesd(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        columns=['values'],
        max_outliers=2,
        output_filename="gesd_checked",
        explanation="GESD outlier detection"
    )
    
    # Verify it ran successfully
    assert "output_filename" in result
    df_result = _load_resource(manifest_path, result["output_filename"])
    assert "values_gesd_pass" in df_result.columns
    # Verify both 100 and 200 are detected as outliers
    assert df_result.loc[df_result['values'] == 100, 'values_gesd_pass'].iloc[0] == False
    assert df_result.loc[df_result['values'] == 200, 'values_gesd_pass'].iloc[0] == False
    # Verify normal values pass
    assert df_result.loc[df_result['values'] == 3, 'values_gesd_pass'].iloc[0] == True
