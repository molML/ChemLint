"""
Heatmap visualizations (correlation and grouped).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from chemlint.infrastructure.resources import _load_resource
from chemlint.tools.plotting.utils import (
    _active_plots, _server_lock, _PORT,
    _ensure_server_running, _update_layout
)


def add_correlation_heatmap(
    input_filename: str,
    project_manifest_path: str,
    columns: list[str],
    plot_name: str,
    explanation: str,
    method: str = "pearson"
) -> dict:
    """Add correlation matrix heatmap (Pearson or Spearman).
    
    Creates new dashboard tab showing pairwise correlations. Values range -1 to 1. Requires ≥2 numeric columns. Starts server automatically if needed.
    
    Parameters
    ----------
    input_filename : str - Dataset filename
    project_manifest_path : str - Path to manifest.json
    columns : list[str] - Columns to correlate (must be numeric)
    plot_name : str - Unique tab label
    explanation : str - Brief description
    method : str, default='pearson' - Correlation method: 'pearson' or 'spearman'
    
    Returns: dict with plot_name, plot_id, n_variables, method, url, message
    
    Example: add_correlation_heatmap("data_ABC123.csv", "/path/manifest.json", ["MW", "LogP", "TPSA"], "Property Correlations", "Descriptor analysis")
    """
    if method not in ["pearson", "spearman"]:
        raise ValueError(f"Invalid method '{method}'. Must be 'pearson' or 'spearman'")
    
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")
    
    # Select numeric columns only
    df_subset = df[columns].select_dtypes(include=[np.number])
    if df_subset.empty or len(df_subset.columns) < 2:
        raise ValueError(f"Need at least 2 numeric columns. Got: {list(df_subset.columns)}")
    
    # Calculate correlation matrix
    corr_matrix = df_subset.corr(method=method)
    
    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title=f"{method.capitalize()}<br>Correlation")
    ))
    
    fig.update_layout(
        title=f"{method.capitalize()} Correlation Matrix",
        xaxis_title="",
        yaxis_title="",
        width=max(600, len(corr_matrix.columns) * 80),
        height=max(600, len(corr_matrix.columns) * 80),
        template="plotly_white"
    )
    
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    plot_data = {
        'label': plot_name,
        'type': 'heatmap',
        'figure': fig,
        'explanation': explanation,
        'method': method,
        'n_variables': len(corr_matrix.columns),
        'show_structures': False
    }
    
    _ensure_server_running()
    
    with _server_lock:
        _active_plots[plot_id] = plot_data
        _update_layout()
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "plot_name": plot_name,
        "plot_id": plot_id,
        "n_variables": len(corr_matrix.columns),
        "method": method,
        "url": url,
        "message": f"Correlation heatmap '{plot_name}' added. Visit {url} to view."
    }


def add_grouped_heatmap(
    input_filename: str,
    project_manifest_path: str,
    row_column: str,
    col_column: str,
    value_column: str,
    plot_name: str,
    explanation: str,
    aggregation: str = "mean"
) -> dict:
    """Add pivot table heatmap with categorical grouping and aggregation.
    
    Creates new dashboard tab showing aggregated values in 2D grid. Useful for category-based analysis. Starts server automatically if needed.
    
    Parameters
    ----------
    input_filename : str - Dataset filename
    project_manifest_path : str - Path to manifest.json
    row_column : str - Column for rows (categorical)
    col_column : str - Column for columns (categorical)
    value_column : str - Column to aggregate (numeric)
    plot_name : str - Unique tab label
    explanation : str - Brief description
    aggregation : str, default='mean' - Aggregation: 'mean', 'median', 'count', 'std', 'min', 'max', 'sum'
    
    Returns: dict with plot_name, plot_id, n_rows, n_cols, aggregation, url, message
    
    Example: add_grouped_heatmap("data_ABC123.csv", "/path/manifest.json", "scaffold", "cluster", "pIC50", "Activity Map", "Scaffold-cluster analysis", "mean")
    """
    valid_agg = ["mean", "median", "count", "std", "min", "max", "sum"]
    if aggregation not in valid_agg:
        raise ValueError(f"Invalid aggregation '{aggregation}'. Must be one of {valid_agg}")
    
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    required = [row_column, col_column, value_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")
    
    # Check value column is numeric
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        raise ValueError(f"Value column '{value_column}' must be numeric. Got dtype: {df[value_column].dtype}")
    
    # Create pivot table with aggregation
    pivot = df.pivot_table(
        values=value_column,
        index=row_column,
        columns=col_column,
        aggfunc=aggregation,
        fill_value=np.nan
    )
    
    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        colorscale='Viridis',
        text=np.round(pivot.values, 2),
        texttemplate='%{text}',
        textfont={"size": 9},
        colorbar=dict(title=f"{aggregation.capitalize()}<br>{value_column}")
    ))
    
    fig.update_layout(
        title=f"{aggregation.capitalize()} of {value_column} by {row_column} and {col_column}",
        xaxis_title=col_column,
        yaxis_title=row_column,
        width=max(700, len(pivot.columns) * 60),
        height=max(500, len(pivot.index) * 30),
        template="plotly_white"
    )
    
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    plot_data = {
        'label': plot_name,
        'type': 'grouped_heatmap',
        'figure': fig,
        'explanation': explanation,
        'aggregation': aggregation,
        'n_rows': len(pivot.index),
        'n_cols': len(pivot.columns),
        'show_structures': False
    }
    
    _ensure_server_running()
    
    with _server_lock:
        _active_plots[plot_id] = plot_data
        _update_layout()
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "plot_name": plot_name,
        "plot_id": plot_id,
        "n_rows": len(pivot.index),
        "n_cols": len(pivot.columns),
        "aggregation": aggregation,
        "url": url,
        "message": f"Grouped heatmap '{plot_name}' added. Visit {url} to view."
    }
