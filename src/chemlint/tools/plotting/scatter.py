"""
Scatter plot with molecular structure tooltips.
"""

from chemlint.infrastructure.resources import _load_resource
from chemlint.tools.plotting.utils import (
    _active_plots, _server_lock, _PORT,
    _create_scatter_figure, _ensure_server_running, _update_layout
)


def add_molecular_scatter_plot(
    input_filename: str,
    x_column: str,
    y_column: str,
    project_manifest_path: str,
    plot_name: str,
    explanation: str,
    smiles_column: str = 'smiles',
    color_column: str = None,
    size_column: str = None,
    show_structures_on_hover: bool = True,
    use_acs1996_style: bool = True
) -> dict:
    """Add interactive scatter plot with optional molecular structure tooltips.
    
    Creates new dashboard tab. Starts server automatically if needed. Hover over points to see molecular structures (if enabled).
    
    Parameters
    ----------
    input_filename : str - Dataset filename
    x_column : str - Column for x-axis
    y_column : str - Column for y-axis
    project_manifest_path : str - Path to manifest.json
    plot_name : str - Unique tab label
    explanation : str - Brief description
    smiles_column : str, default='smiles' - SMILES column
    color_column : str, optional - Column for point colors
    size_column : str, optional - Column for point sizes
    show_structures_on_hover : bool, default=True - Show structures on hover
    use_acs1996_style : bool, default=True - Use ACS1996 rendering style for molecules (matches smiles_to_acs1996_png)
    
    Returns: dict with plot_name, plot_id, url, n_molecules, x_column, y_column, show_structures, use_acs1996_style, active_plots, message
    
    Example: add_molecular_scatter_plot("data_ABC123.csv", "MW", "LogP", "/path/manifest.json", "MW vs LogP", "Lipophilicity analysis")
    """
    global _active_plots, _PORT
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    required_cols = [x_column, y_column]
    if show_structures_on_hover:
        required_cols.append(smiles_column)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Column(s) not found in dataset: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Generate unique plot ID
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    # Check if plot already exists
    if plot_id in _active_plots:
        raise ValueError(
            f"Plot '{plot_name}' already exists. Use a different name or remove it first."
        )
    
    # Create figure
    fig = _create_scatter_figure(df, x_column, y_column, color_column, size_column, plot_id)
    
    # Store plot data
    with _server_lock:
        _active_plots[plot_id] = {
            'label': plot_name,
            'dataframe': df,
            'figure': fig,
            'x_column': x_column,
            'y_column': y_column,
            'smiles_column': smiles_column,
            'color_column': color_column,
            'size_column': size_column,
            'show_structures': show_structures_on_hover,
            'use_acs1996': use_acs1996_style,
            'explanation': explanation
        }
        
        # Ensure server is running
        _ensure_server_running()
        
        # Update layout (no need to register callbacks - already done universally)
        _update_layout()
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "plot_name": plot_name,
        "plot_id": plot_id,
        "url": url,
        "n_molecules": len(df),
        "x_column": x_column,
        "y_column": y_column,
        "show_structures": show_structures_on_hover,
        "use_acs1996_style": use_acs1996_style,
        "active_plots": list(_active_plots.keys()),
        "message": f"Plot '{plot_name}' added. Visit {url} to view all {len(_active_plots)} plot(s)."
    }
