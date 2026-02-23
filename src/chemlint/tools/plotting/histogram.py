"""
Histogram visualization.
"""

import plotly.graph_objects as go
from chemlint.infrastructure.resources import _load_resource
from chemlint.tools.plotting.utils import (
    _active_plots, _server_lock, _PORT,
    _ensure_server_running, _update_layout
)


def add_histogram(
    input_filename: str,
    column: str,
    project_manifest_path: str,
    plot_name: str,
    explanation: str,
    bins: int = 30,
    color: str = "#577788",
    show_mean_line: bool = True,
    show_median_line: bool = False
) -> dict:
    """Add histogram distribution plot with optional mean/median lines.
    
    Creates new dashboard tab showing value distribution. Starts server automatically if needed.
    
    Parameters
    ----------
    input_filename : str - Dataset filename
    column : str - Column to plot
    project_manifest_path : str - Path to manifest.json
    plot_name : str - Unique tab label
    explanation : str - Brief description
    bins : int, default=30 - Number of bins
    color : str, default='#577788' - Hex color
    show_mean_line : bool, default=True - Show mean line
    show_median_line : bool, default=False - Show median line
    
    Returns: dict with plot_name, plot_id, url, n_values, statistics (mean/median/min/max), active_plots, message
    
    Example: add_histogram("data_ABC123.csv", "MW", "/path/manifest.json", "MW Distribution", "Molecular weight histogram")
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
    
    # Extract data and remove NaN
    data = df[column].dropna()
    
    if len(data) == 0:
        raise ValueError(f"Column '{column}' contains no valid (non-NaN) values")
    
    # Calculate statistics
    mean_val = float(data.mean())
    median_val = float(data.median())
    min_val = float(data.min())
    max_val = float(data.max())
    
    # Generate unique plot ID
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    # Check if plot already exists
    if plot_id in _active_plots:
        raise ValueError(
            f"Plot '{plot_name}' already exists. Use a different name or remove it first."
        )
    
    # Create histogram figure
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        marker=dict(color=color, line=dict(color='white', width=1)),
        name='Distribution',
        opacity=0.85
    ))
    
    # Add mean line
    if show_mean_line:
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top"
        )
    
    # Add median line
    if show_median_line:
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="top"
        )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(title=column),
        yaxis=dict(title="Count"),
        plot_bgcolor='rgba(255,255,255,0.1)',
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x'
    )
    
    # Store plot data
    with _server_lock:
        _active_plots[plot_id] = {
            'label': plot_name,
            'figure': fig,
            'type': 'histogram',
            'column': column,
            'show_structures': False,  # Histograms don't have molecular tooltips
            'explanation': explanation,
            'statistics': {
                'n_values': len(data),
                'mean': mean_val,
                'median': median_val,
                'min': min_val,
                'max': max_val
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
        "n_values": len(data),
        "statistics": {
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val
        },
        "active_plots": list(_active_plots.keys()),
        "message": f"Histogram '{plot_name}' added. Visit {url} to view all {len(_active_plots)} plot(s)."
    }
