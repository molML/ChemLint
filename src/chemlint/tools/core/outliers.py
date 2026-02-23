"""
Outlier detection methods for dataset analysis.

All methods operate on dataset columns and add boolean pass/fail columns:

Statistical Methods:
- Z-score method: Detects outliers beyond ±3 standard deviations
- Modified Z-score method: Uses median absolute deviation (MAD), more robust to outliers
- IQR method: Detects outliers beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR

Parametric Tests:
- Grubbs' test: Statistical test for single outlier (assumes normality)
- Generalized ESD test: Detects multiple outliers iteratively (assumes normality)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats
from chemlint.infrastructure.resources import _load_resource, _store_resource


def detect_outliers_zscore(
    input_filename: str,
    project_manifest_path: str,
    columns: List[str],
    output_filename: str,
    explanation: str,
    threshold: float = 3.0
) -> Dict:
    """
    Detect outliers using Z-score method. Formula: z = (x - μ) / σ. Outlier if |z| > threshold.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to manifest.json
        columns: Columns to check for outliers
        output_filename: Output dataset name
        explanation: Description of operation
        threshold: Number of standard deviations (default: 3.0)
        
    Returns:
        Dictionary with output_filename, n_rows, columns, columns_checked, outliers_per_column, 
        total_outliers, outlier_columns_added, threshold, preview, summary
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Always create copy for traceability
    df_result = df.copy()
    
    # Validate columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found: {missing_cols}. "
            f"Available: {list(df.columns)}"
        )
    
    outlier_columns_added = []
    outliers_per_column = {}
    
    for col in columns:
        # Calculate Z-scores
        data = df_result[col].dropna()
        
        if len(data) < 2:
            raise ValueError(f"Column '{col}' needs at least 2 non-NaN values")
        
        mean = data.mean()
        std = data.std(ddof=1)
        
        if std == 0:
            # All values are the same, no outliers
            z_scores = pd.Series(0, index=data.index)
        else:
            z_scores = (data - mean) / std
        
        # Create outlier mask (False = outlier, True = not outlier)
        is_not_outlier = np.abs(z_scores) <= threshold
        
        # For NaN values, mark as True (not outlier) to avoid dropping them unless intended
        outlier_col_name = f"{col}_zscore_pass"
        df_result[outlier_col_name] = True  # Default to pass
        df_result.loc[data.index, outlier_col_name] = is_not_outlier.values
        
        outlier_columns_added.append(outlier_col_name)
        n_outliers = (~is_not_outlier).sum()
        outliers_per_column[col] = int(n_outliers)
    
    # Count rows with at least one outlier
    outlier_mask = df_result[outlier_columns_added].all(axis=1)
    total_outliers = int((~outlier_mask).sum())
    
    # Store result as new dataset (traceability policy)
    output_id = _store_resource(
        df_result,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_rows": len(df_result),
        "columns": list(df_result.columns),
        "columns_checked": columns,
        "outliers_per_column": outliers_per_column,
        "total_outliers": total_outliers,
        "outlier_columns_added": outlier_columns_added,
        "threshold": threshold,
        "preview": df_result.head(5).to_dict('records'),
        "summary": f"Z-score outlier detection: {total_outliers} rows with outliers (threshold={threshold}σ). Columns: {outlier_columns_added}"
    }


def detect_outliers_modified_zscore(
    input_filename: str,
    project_manifest_path: str,
    columns: List[str],
    output_filename: str,
    explanation: str,
    threshold: float = 3.5
) -> Dict:
    """
    Detect outliers using Modified Z-score (robust to outliers). 
    Formula: M = 0.6745 × (x - median) / MAD, where MAD = median(|x - median|). Outlier if |M| > threshold.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to manifest.json
        columns: Columns to check for outliers
        output_filename: Output dataset name
        explanation: Description of operation
        threshold: Modified Z-score threshold (default: 3.5)
        
    Returns:
        Dictionary with output_filename, n_rows, columns, columns_checked, outliers_per_column,
        total_outliers, outlier_columns_added, threshold, preview, summary
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Always create copy for traceability
    df_result = df.copy()
    
    # Validate columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found: {missing_cols}. "
            f"Available: {list(df.columns)}"
        )
    
    outlier_columns_added = []
    outliers_per_column = {}
    
    for col in columns:
        data = df_result[col].dropna()
        
        if len(data) < 2:
            raise ValueError(f"Column '{col}' needs at least 2 non-NaN values")
        
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            # All values are the same or very concentrated, no outliers
            modified_z_scores = pd.Series(0, index=data.index)
        else:
            # 0.6745 is the constant to make MAD comparable to std for normal distribution
            modified_z_scores = 0.6745 * (data - median) / mad
        
        # Create outlier mask
        is_not_outlier = np.abs(modified_z_scores) <= threshold
        
        outlier_col_name = f"{col}_modified_zscore_pass"
        df_result[outlier_col_name] = True
        df_result.loc[data.index, outlier_col_name] = is_not_outlier.values
        
        outlier_columns_added.append(outlier_col_name)
        n_outliers = (~is_not_outlier).sum()
        outliers_per_column[col] = int(n_outliers)
    
    # Count rows with at least one outlier
    outlier_mask = df_result[outlier_columns_added].all(axis=1)
    total_outliers = int((~outlier_mask).sum())
    
    # Store result as new dataset (traceability policy)
    output_id = _store_resource(
        df_result,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_rows": len(df_result),
        "columns": list(df_result.columns),
        "columns_checked": columns,
        "outliers_per_column": outliers_per_column,
        "total_outliers": total_outliers,
        "outlier_columns_added": outlier_columns_added,
        "threshold": threshold,
        "preview": df_result.head(5).to_dict('records'),
        "summary": f"Modified Z-score outlier detection: {total_outliers} rows with outliers (threshold={threshold}). Columns: {outlier_columns_added}"
    }


def detect_outliers_iqr(
    input_filename: str,
    project_manifest_path: str,
    columns: List[str],
    output_filename: str,
    explanation: str,
    multiplier: float = 1.5
) -> Dict:
    """
    Detect outliers using IQR method. Outliers fall outside [Q1 - k×IQR, Q3 + k×IQR].
    Non-parametric, robust to extremes. Use k=1.5 for outliers, k=3.0 for far outliers.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to manifest.json
        columns: Columns to check for outliers
        output_filename: Output dataset name
        explanation: Description of operation
        multiplier: IQR multiplier k (default: 1.5)
        
    Returns:
        Dictionary with output_filename, n_rows, columns, columns_checked, outliers_per_column,
        total_outliers, outlier_columns_added, bounds_per_column, multiplier, preview, summary
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Always create copy for traceability
    df_result = df.copy()
    
    # Validate columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found: {missing_cols}. "
            f"Available: {list(df.columns)}"
        )
    
    outlier_columns_added = []
    outliers_per_column = {}
    bounds_per_column = {}
    
    for col in columns:
        data = df_result[col].dropna()
        
        if len(data) < 4:
            raise ValueError(f"Column '{col}' needs at least 4 non-NaN values for quartile calculation")
        
        # Calculate quartiles and IQR
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        # Calculate bounds
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # Create outlier mask
        is_not_outlier = (data >= lower_bound) & (data <= upper_bound)
        
        outlier_col_name = f"{col}_iqr_pass"
        df_result[outlier_col_name] = True
        df_result.loc[data.index, outlier_col_name] = is_not_outlier.values
        
        outlier_columns_added.append(outlier_col_name)
        n_outliers = (~is_not_outlier).sum()
        outliers_per_column[col] = int(n_outliers)
        bounds_per_column[col] = {
            "lower": float(lower_bound),
            "upper": float(upper_bound),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr)
        }
    
    # Count rows with at least one outlier
    outlier_mask = df_result[outlier_columns_added].all(axis=1)
    total_outliers = int((~outlier_mask).sum())
    
    # Store result as new dataset (traceability policy)
    output_id = _store_resource(
        df_result,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_rows": len(df_result),
        "columns": list(df_result.columns),
        "columns_checked": columns,
        "outliers_per_column": outliers_per_column,
        "total_outliers": total_outliers,
        "outlier_columns_added": outlier_columns_added,
        "bounds_per_column": bounds_per_column,
        "multiplier": multiplier,
        "preview": df_result.head(5).to_dict('records'),
        "summary": f"IQR outlier detection: {total_outliers} rows with outliers (multiplier={multiplier}). Columns: {outlier_columns_added}"
    }


def detect_outliers_grubbs(
    input_filename: str,
    project_manifest_path: str,
    columns: List[str],
    output_filename: str,
    explanation: str,
    alpha: float = 0.05
) -> Dict:
    """
    Detect outliers using Grubbs' test (single outlier, assumes normality).
    Test statistic: G = max|x_i - mean| / std. Detects ONE outlier per run.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to manifest.json
        columns: Columns to check for outliers
        output_filename: Output dataset name
        explanation: Description of operation
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with output_filename, n_rows, columns, columns_checked, outliers_per_column,
        total_outliers, outlier_columns_added, alpha, preview, summary
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Always create copy for traceability
    df_result = df.copy()
    
    # Validate columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found: {missing_cols}. "
            f"Available: {list(df.columns)}"
        )
    
    outlier_columns_added = []
    outliers_per_column = {}
    
    for col in columns:
        data = df_result[col].dropna()
        
        if len(data) < 3:
            raise ValueError(f"Column '{col}' needs at least 3 non-NaN values for Grubbs test")
        
        n = len(data)
        mean = data.mean()
        std = data.std(ddof=1)
        
        if std == 0:
            # All values are the same, no outliers
            outlier_col_name = f"{col}_grubbs_pass"
            df_result[outlier_col_name] = True
            outlier_columns_added.append(outlier_col_name)
            outliers_per_column[col] = {
                "outlier_detected": False,
                "n_outliers": 0
            }
            continue
        
        # Calculate G statistic for all points
        z_scores = np.abs(data - mean) / std
        max_idx = z_scores.idxmax()
        G = z_scores.max()
        
        # Calculate critical value
        # Critical value from t-distribution
        t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
        
        # Determine if outlier
        is_outlier = G > G_critical
        
        # Create pass/fail column
        outlier_col_name = f"{col}_grubbs_pass"
        df_result[outlier_col_name] = True
        if is_outlier:
            df_result.loc[max_idx, outlier_col_name] = False
        
        outlier_columns_added.append(outlier_col_name)
        
        if is_outlier:
            outliers_per_column[col] = {
                "outlier_detected": True,
                "n_outliers": 1,
                "outlier_index": int(max_idx),
                "outlier_value": float(data.loc[max_idx]),
                "G_statistic": float(G),
                "G_critical": float(G_critical),
                "is_significant": True
            }
        else:
            outliers_per_column[col] = {
                "outlier_detected": False,
                "n_outliers": 0,
                "max_G_statistic": float(G),
                "G_critical": float(G_critical),
                "is_significant": False
            }
    
    # Count rows with at least one outlier
    outlier_mask = df_result[outlier_columns_added].all(axis=1)
    total_outliers = int((~outlier_mask).sum())
    
    # Store result as new dataset (traceability policy)
    output_id = _store_resource(
        df_result,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_rows": len(df_result),
        "columns": list(df_result.columns),
        "columns_checked": columns,
        "outliers_per_column": outliers_per_column,
        "total_outliers": total_outliers,
        "outlier_columns_added": outlier_columns_added,
        "alpha": alpha,
        "preview": df_result.head(5).to_dict('records'),
        "summary": f"Grubbs test: {total_outliers} rows with outliers (α={alpha}). Columns: {outlier_columns_added}"
    }


def detect_outliers_gesd(
    input_filename: str,
    project_manifest_path: str,
    columns: List[str],
    output_filename: str,
    explanation: str,
    max_outliers: int = 10,
    alpha: float = 0.05
) -> Dict:
    """
    Detect outliers using Generalized ESD test (multiple outliers, assumes normality).
    Extension of Grubbs' test that iteratively removes extreme values up to max_outliers.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to manifest.json
        columns: Columns to check for outliers
        output_filename: Output dataset name
        explanation: Description of operation
        max_outliers: Maximum outliers to detect (default: 10)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with output_filename, n_rows, columns, columns_checked, outliers_per_column,
        total_outliers, outlier_columns_added, max_outliers, alpha, preview, summary
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Always create copy for traceability
    df_result = df.copy()
    
    # Validate columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found: {missing_cols}. "
            f"Available: {list(df.columns)}"
        )
    
    outlier_columns_added = []
    outliers_per_column = {}
    
    for col in columns:
        data = df_result[col].dropna()
        
        if len(data) < max_outliers + 2:
            raise ValueError(
                f"Column '{col}' needs at least {max_outliers + 2} non-NaN values "
                f"for GESD test with max_outliers={max_outliers}"
            )
        
        n = len(data)
        
        # Initialize
        outlier_indices = []
        test_statistics = []
        critical_values = []
        
        # Make a copy for iterative removal
        data_working = data.copy()
        
        # Perform GESD test iteratively
        for i in range(max_outliers):
            if len(data_working) < 3:
                break
            
            # Calculate test statistic
            mean = data_working.mean()
            std = data_working.std(ddof=1)
            
            if std == 0:
                break
            
            z_scores = np.abs(data_working - mean) / std
            max_idx = z_scores.idxmax()
            R_i = z_scores.max()
            
            # Calculate critical value for current iteration
            n_i = len(data_working)
            p = 1 - alpha / (2 * (n_i - i))
            t_dist = stats.t.ppf(p, n_i - i - 2)
            lambda_i = ((n_i - i - 1) * t_dist) / np.sqrt((n_i - i - 2 + t_dist**2) * (n_i - i))
            
            test_statistics.append(float(R_i))
            critical_values.append(float(lambda_i))
            
            # Store potential outlier
            outlier_indices.append(max_idx)
            
            # Remove the most extreme value for next iteration
            data_working = data_working.drop(max_idx)
        
        # Determine actual number of outliers
        # Work backwards to find where R_i > lambda_i stops being true
        num_outliers = 0
        for i in range(len(test_statistics)):
            if test_statistics[i] > critical_values[i]:
                num_outliers = i + 1
        
        # Create pass/fail column
        outlier_col_name = f"{col}_gesd_pass"
        df_result[outlier_col_name] = True
        
        if num_outliers > 0:
            actual_outliers = outlier_indices[:num_outliers]
            df_result.loc[actual_outliers, outlier_col_name] = False
            
            outliers_per_column[col] = {
                "n_outliers": num_outliers,
                "outlier_indices": [int(idx) for idx in actual_outliers],
                "outlier_values": [float(data.loc[idx]) for idx in actual_outliers],
                "test_statistics": test_statistics[:num_outliers],
                "critical_values": critical_values[:num_outliers]
            }
        else:
            outliers_per_column[col] = {
                "n_outliers": 0,
                "max_test_statistic": test_statistics[0] if test_statistics else 0,
                "critical_value": critical_values[0] if critical_values else 0
            }
        
        outlier_columns_added.append(outlier_col_name)
    
    # Count rows with at least one outlier
    outlier_mask = df_result[outlier_columns_added].all(axis=1)
    total_outliers = int((~outlier_mask).sum())
    
    # Store result as new dataset (traceability policy)
    output_id = _store_resource(
        df_result,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,
        "n_rows": len(df_result),
        "columns": list(df_result.columns),
        "columns_checked": columns,
        "outliers_per_column": outliers_per_column,
        "total_outliers": total_outliers,
        "outlier_columns_added": outlier_columns_added,
        "max_outliers": max_outliers,
        "alpha": alpha,
        "preview": df_result.head(5).to_dict('records'),
        "summary": f"Generalized ESD test: {total_outliers} rows with outliers (max={max_outliers}, α={alpha}). Columns: {outlier_columns_added}"
    }


def get_all_outlier_detection_tools():
    """
    Returns all MCP-exposed outlier detection functions.
    """
    return [
        detect_outliers_zscore,
        detect_outliers_modified_zscore,
        detect_outliers_iqr,
        detect_outliers_grubbs,
        detect_outliers_gesd,
    ]
