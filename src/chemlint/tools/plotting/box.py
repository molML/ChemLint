"""
Box plot visualization.
"""

import plotly.graph_objects as go
from chemlint.infrastructure.resources import _load_resource
from chemlint.tools.plotting.utils import (
    _active_plots, _server_lock, _PORT,
    _ensure_server_running, _update_layout
)


def add_box_plot(
    input_filename: str,
    column: str,
    project_manifest_path: str,
    plot_name: str,
    explanation: str,
    group_column: str = None,
    color: str = "#577788",
    show_points: bool = False,
    notched: bool = False
) -> dict:
    """Add box-and-whisker plot with optional grouping and data points.
    
    Creates new dashboard tab showing distribution quartiles. Supports side-by-side grouped plots. Starts server automatically if needed.
    
    Parameters
    ----------
    input_filename : str - Dataset filename
    column : str - Column with values to plot
    project_manifest_path : str - Path to manifest.json
    plot_name : str - Unique tab label
    explanation : str - Brief description
    group_column : str, optional - Column for grouping (creates multiple boxes)
    color : str, default='#577788' - Hex color
    show_points : bool, default=False - Overlay individual points
    notched : bool, default=False - Show CI around median
    
    Returns: dict with plot_name, plot_id, url, n_values, statistics (mean/median/q1/q3/min/max/iqr), active_plots, message
    
    Example: add_box_plot("data_ABC123.csv", "Activity", "/path/manifest.json", "Activity Distribution", "pIC50 boxplot", group_column="Cluster")
    """
    global _active_plots, _PORT
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    
    if group_column and group_column not in df.columns:
        raise ValueError(
            f"Group column '{group_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Remove rows with NaN in required columns
    required_cols = [column]
    if group_column:
        required_cols.append(group_column)
    
    df_clean = df[required_cols].dropna()
    
    if len(df_clean) == 0:
        raise ValueError(f"No valid (non-NaN) data found in required columns")
    
    # Generate unique plot ID
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    # Check if plot already exists
    if plot_id in _active_plots:
        raise ValueError(
            f"Plot '{plot_name}' already exists. Use a different name or remove it first."
        )
    
    # Create box plot figure
    fig = go.Figure()
    
    if group_column:
        # Grouped box plots
        groups = df_clean[group_column].unique()
        
        for group in sorted(groups):
            group_data = df_clean[df_clean[group_column] == group][column]
            
            if show_points:
                fig.add_trace(go.Box(
                    y=group_data,
                    name=str(group),
                    marker_color=color,
                    boxmean=True,
                    notched=notched,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.5
                ))
            else:
                fig.add_trace(go.Box(
                    y=group_data,
                    name=str(group),
                    marker_color=color,
                    boxmean=True,
                    notched=notched
                ))
        
        # Calculate overall statistics
        all_data = df_clean[column]
        mean_val = float(all_data.mean())
        median_val = float(all_data.median())
        min_val = float(all_data.min())
        max_val = float(all_data.max())
        q1 = float(all_data.quantile(0.25))
        q3 = float(all_data.quantile(0.75))
        
        fig.update_layout(
            xaxis=dict(title=group_column),
            yaxis=dict(title=column),
            showlegend=True
        )
    else:
        # Single box plot
        data = df_clean[column]
        
        if show_points:
            fig.add_trace(go.Box(
                y=data,
                name=column,
                marker_color=color,
                boxmean=True,
                notched=notched,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.5
            ))
        else:
            fig.add_trace(go.Box(
                y=data,
                name=column,
                marker_color=color,
                boxmean=True,
                notched=notched
            ))
        
        mean_val = float(data.mean())
        median_val = float(data.median())
        min_val = float(data.min())
        max_val = float(data.max())
        q1 = float(data.quantile(0.25))
        q3 = float(data.quantile(0.75))
        
        fig.update_layout(
            yaxis=dict(title=column),
            showlegend=False
        )
    
    # Common layout updates
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.1)',
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='closest'
    )
    
    # Store plot data
    with _server_lock:
        _active_plots[plot_id] = {
            'label': plot_name,
            'figure': fig,
            'type': 'boxplot',
            'column': column,
            'group_column': group_column,
            'show_structures': False,
            'explanation': explanation,
            'statistics': {
                'n_values': len(df_clean),
                'mean': mean_val,
                'median': median_val,
                'q1': q1,
                'q3': q3,
                'min': min_val,
                'max': max_val,
                'iqr': q3 - q1
            }
        }
        
        # Ensure server is running
        _ensure_server_running()
        
        # Update layout
        _update_layout()
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "plot_name": plot_name,
        "plot_id": plot_id,
        "url": url,
        "n_values": len(df_clean),
        "statistics": {
            "mean": mean_val,
            "median": median_val,
            "q1": q1,
            "q3": q3,
            "min": min_val,
            "max": max_val,
            "iqr": q3 - q1
        },
        "active_plots": list(_active_plots.keys()),
        "message": f"Box plot '{plot_name}' added. Visit {url} to view all {len(_active_plots)} plot(s)."
    }
