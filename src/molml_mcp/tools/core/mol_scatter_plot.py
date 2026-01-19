"""
Interactive molecular scatter plots with structure visualization on hover.

Uses a persistent Dash server with tabs - plots can be dynamically added/removed.
"""

from dash import Dash, dcc, html, Input, Output, no_update, callback, ctx
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO
import pandas as pd
import threading
import time
from molml_mcp.infrastructure.resources import _load_resource, _store_resource

# Global state for persistent Dash server
_dash_app = None
_dash_thread = None
_active_plots = {}  # plot_id -> plot_data dict
_server_lock = threading.RLock()  # Use RLock for reentrant locking
_PORT = 8050


def _mol_to_base64(smiles, size=(200, 200)):
    """
    Convert SMILES to base64 encoded PNG image.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    size : tuple
        Image size (width, height)
    
    Returns
    -------
    str or None
        Base64 encoded data URI or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def _create_scatter_figure(df, x_col, y_col, color_col=None, size_col=None, plot_id="scatter"):
    """
    Create a plotly scatter plot figure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    color_col : str, optional
        Column for point colors
    size_col : str, optional
        Column for point sizes
    plot_id : str
        Unique identifier for this plot
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    marker_kwargs = {"size": 10}
    
    if color_col and color_col in df.columns:
        marker_kwargs.update({
            "color": df[color_col],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": color_col}
        })
    
    if size_col and size_col in df.columns:
        marker_kwargs["size"] = df[size_col]
        marker_kwargs["sizemode"] = "diameter"
        marker_kwargs["sizeref"] = df[size_col].max() / 20
    
    fig = go.Figure(data=go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=marker_kwargs,
        customdata=list(range(len(df)))  # Store row indices
    ))
    
    # Turn off native plotly.js hover effects
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    
    fig.update_layout(
        xaxis=dict(title=x_col),
        yaxis=dict(title=y_col),
        plot_bgcolor='rgba(255,255,255,0.1)',
        hovermode='closest',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def _build_plot_tab(plot_id, plot_data):
    """Build a single plot tab with graph and tooltip."""
    fig = plot_data['figure']
    
    if plot_data['show_structures']:
        return dcc.Tab(
            label=plot_data['label'],
            value=plot_id,
            children=[
                html.Div([
                    dcc.Graph(
                        id={'type': 'graph', 'index': plot_id},
                        figure=fig,
                        clear_on_unhover=True,
                        style={'height': '80vh'}
                    ),
                    dcc.Tooltip(id={'type': 'tooltip', 'index': plot_id}),
                ])
            ]
        )
    else:
        return dcc.Tab(
            label=plot_data['label'],
            value=plot_id,
            children=[
                dcc.Graph(
                    id={'type': 'graph', 'index': plot_id},
                    figure=fig,
                    style={'height': '80vh'}
                )
            ]
        )


def _update_layout():
    """Update the Dash app layout with current plots."""
    global _dash_app, _active_plots
    
    if not _active_plots:
        _dash_app.layout = html.Div([
            html.H3("Molecular Visualization Dashboard", style={'textAlign': 'center', 'padding': '20px'}),
            html.P("No plots yet. Use add_molecular_scatter_plot to create visualizations.", 
                   style={'textAlign': 'center', 'color': 'gray'})
        ])
        return
    
    tabs = [_build_plot_tab(plot_id, plot_data) for plot_id, plot_data in _active_plots.items()]
    
    _dash_app.layout = html.Div([
        html.H3("Molecular Visualization Dashboard", style={'textAlign': 'center', 'padding': '10px'}),
        dcc.Tabs(id="plot-tabs", value=list(_active_plots.keys())[0], children=tabs)
    ])


def _setup_universal_callback():
    """Set up a single universal callback that handles all plot tooltips dynamically."""
    global _dash_app
    
    # Use pattern-matching callbacks to handle all tooltips with a single callback
    from dash import ALL
    
    @_dash_app.callback(
        Output({'type': 'tooltip', 'index': ALL}, 'show'),
        Output({'type': 'tooltip', 'index': ALL}, 'bbox'),
        Output({'type': 'tooltip', 'index': ALL}, 'children'),
        Input({'type': 'graph', 'index': ALL}, 'hoverData'),
        prevent_initial_call=True
    )
    def universal_hover(*hover_data_list):
        global _active_plots
        
        # Get the list of all registered plot IDs
        plot_ids = list(_active_plots.keys())
        n_plots = len(plot_ids)
        
        # Find which graph triggered the callback
        triggered = ctx.triggered_id
        if not triggered:
            return [no_update] * n_plots, [no_update] * n_plots, [no_update] * n_plots
        
        plot_id = triggered['index']
        
        # Find the index in our plot_ids list
        try:
            trigger_idx = plot_ids.index(plot_id)
        except ValueError:
            return [no_update] * n_plots, [no_update] * n_plots, [no_update] * n_plots
        
        # Get hover data
        if len(hover_data_list) == 0:
            return [no_update] * n_plots, [no_update] * n_plots, [no_update] * n_plots
            
        # hover_data_list[0] is a list with entries for all tabs
        # e.g., [{'points': [...]}, None] or [None, {'points': [...]}]
        all_hover_data = hover_data_list[0]
        
        if not isinstance(all_hover_data, list) or len(all_hover_data) == 0:
            return [no_update] * n_plots, [no_update] * n_plots, [no_update] * n_plots
        
        # Get the hover data for the triggered plot
        if trigger_idx >= len(all_hover_data):
            return [no_update] * n_plots, [no_update] * n_plots, [no_update] * n_plots
            
        hoverData = all_hover_data[trigger_idx]
        
        # Check if hoverData is valid
        if not hoverData or not isinstance(hoverData, dict) or "points" not in hoverData:
            return [no_update] * n_plots, [no_update] * n_plots, [no_update] * n_plots
        
        plot_data = _active_plots[plot_id]
        
        if not plot_data['show_structures']:
            return [no_update] * n_plots, [no_update] * n_plots, [no_update] * n_plots
        
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]

        df = plot_data['dataframe']
        df_row = df.iloc[num]
        smiles = df_row[plot_data['smiles_column']]
        
        img_src = _mol_to_base64(smiles)
        
        properties = []
        for col in df.columns:
            if col != plot_data['smiles_column']:
                value = df_row[col]
                properties.append(
                    html.P(
                        f"{col}: {value:.2f}" if isinstance(value, (int, float)) else f"{col}: {value}",
                        style={"margin": "2px"}
                    )
                )
        
        children = html.Div([
            html.Img(src=img_src, style={"width": "200px", "display": "block", "margin": "0 auto"}) if img_src else None,
            html.P(smiles, style={"font-family": "monospace", "text-align": "center", "margin": "5px"}),
            html.Hr(style={"margin": "5px 0"}),
            *properties
        ], style={"width": "220px", "padding": "10px"})
        
        # Prepare outputs for all tooltips - use n_plots consistently
        shows = [False] * n_plots
        bboxes = [no_update] * n_plots
        children_list = [no_update] * n_plots
        
        shows[trigger_idx] = True
        bboxes[trigger_idx] = bbox
        children_list[trigger_idx] = children
        
        return shows, bboxes, children_list


def _ensure_server_running():
    """Ensure the Dash server is running. Start it if not."""
    global _dash_app, _dash_thread, _PORT
    
    with _server_lock:
        if _dash_app is None:
            _dash_app = Dash(__name__, suppress_callback_exceptions=True)
            
            # Set initial empty layout
            _dash_app.layout = html.Div([
                html.H3("Molecular Visualization Dashboard", style={'textAlign': 'center', 'padding': '20px'}),
                html.P("Loading...", style={'textAlign': 'center', 'color': 'gray'})
            ])
            
            # Register the universal callback BEFORE adding any plots
            # This way pattern-matching will work for all future plot additions
            _setup_universal_callback()
            
            def run_server():
                # Suppress Flask/Werkzeug logging
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
                try:
                    _dash_app.run(debug=False, port=_PORT, use_reloader=False, host='127.0.0.1')
                except OSError as e:
                    # Port already in use - that's okay, server is already running
                    if "Address already in use" in str(e) or "address already in use" in str(e):
                        pass
                    else:
                        raise
            
            _dash_thread = threading.Thread(target=run_server, daemon=True)
            _dash_thread.start()
            
            # Give the server a moment to start
            time.sleep(0.5)


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
    show_structures_on_hover: bool = True
) -> dict:
    """
    Add an interactive scatter plot to the persistent visualization dashboard.
    
    Creates a new tab in the Dash visualization server. If the server isn't running,
    it will be started automatically. Multiple plots can coexist as tabs.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    x_column : str
        Column name for x-axis
    y_column : str
        Column name for y-axis
    project_manifest_path : str
        Path to manifest.json
    plot_name : str
        Unique name for this plot (used as tab label and identifier)
    explanation : str
        Brief description of the plot
    smiles_column : str, default='smiles'
        Column containing SMILES strings
    color_column : str, optional
        Column to use for point colors
    size_column : str, optional
        Column to use for point sizes
    show_structures_on_hover : bool, default=True
        If True, show molecular structures on hover
    
    Returns
    -------
    dict
        Contains plot_name, url, n_molecules, x_column, y_column, active_plots
    
    Examples
    --------
    >>> add_molecular_scatter_plot(
    ...     input_filename="dataset_A1B2C3D4.csv",
    ...     x_column="MW",
    ...     y_column="LogP",
    ...     project_manifest_path="/path/to/manifest.json",
    ...     plot_name="MW vs LogP",
    ...     explanation="Molecular weight vs lipophilicity"
    ... )
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
        "active_plots": list(_active_plots.keys()),
        "message": f"Plot '{plot_name}' added. Visit {url} to view all {len(_active_plots)} plot(s)."
    }


def remove_plot(plot_name: str) -> dict:
    """
    Remove a plot from the visualization dashboard.
    
    Parameters
    ----------
    plot_name : str
        Name of the plot to remove (case-insensitive)
    
    Returns
    -------
    dict
        Contains removed plot name, remaining plots, and url
    
    Examples
    --------
    >>> remove_plot("MW vs LogP")
    """
    global _active_plots, _PORT
    
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    with _server_lock:
        if plot_id not in _active_plots:
            available = [_active_plots[pid]['label'] for pid in _active_plots]
            raise ValueError(
                f"Plot '{plot_name}' not found. Available plots: {available if available else 'none'}"
            )
        
        removed_label = _active_plots[plot_id]['label']
        del _active_plots[plot_id]
        
        # Update layout (no need to re-register callbacks - universal callback handles all)
        _update_layout()
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "removed_plot": removed_label,
        "remaining_plots": list(_active_plots.keys()),
        "n_remaining": len(_active_plots),
        "url": url if _active_plots else None,
        "message": f"Plot '{removed_label}' removed. {len(_active_plots)} plot(s) remaining."
    }


def list_active_plots() -> dict:
    """
    List all active plots in the visualization dashboard.
    
    Returns
    -------
    dict
        Contains plot details, url, and count
    
    Examples
    --------
    >>> list_active_plots()
    """
    global _active_plots, _PORT
    
    if not _active_plots:
        return {
            "active_plots": [],
            "n_plots": 0,
            "url": None,
            "message": "No active plots. Use add_molecular_scatter_plot to create visualizations."
        }
    
    plots_info = []
    for plot_id, plot_data in _active_plots.items():
        plots_info.append({
            "name": plot_data['label'],
            "plot_id": plot_id,
            "x_column": plot_data['x_column'],
            "y_column": plot_data['y_column'],
            "n_points": len(plot_data['dataframe']),
            "show_structures": plot_data['show_structures'],
            "explanation": plot_data['explanation']
        })
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "active_plots": plots_info,
        "n_plots": len(_active_plots),
        "url": url,
        "message": f"{len(_active_plots)} plot(s) active. Visit {url} to view the dashboard."
    }


# Keep old function for backward compatibility (deprecated)
def create_molecular_scatter_plot(
    input_filename: str,
    x_column: str,
    y_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    smiles_column: str = 'smiles',
    color_column: str = None,
    size_column: str = None,
    show_structures_on_hover: bool = True,
    port: int = 8050
) -> dict:
    """
    DEPRECATED: Use add_molecular_scatter_plot instead.
    
    This function creates a standalone Dash server per plot, which is inefficient.
    The new approach uses a single persistent server with tabs.
    """
    return add_molecular_scatter_plot(
        input_filename=input_filename,
        x_column=x_column,
        y_column=y_column,
        project_manifest_path=project_manifest_path,
        plot_name=output_filename,
        explanation=explanation,
        smiles_column=smiles_column,
        color_column=color_column,
        size_column=size_column,
        show_structures_on_hover=show_structures_on_hover
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
    """
    Add an interactive histogram to the persistent visualization dashboard.
    
    Creates a new tab in the Dash visualization server. If the server isn't running,
    it will be started automatically. Multiple plots can coexist as tabs.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    column : str
        Column name to plot as histogram
    project_manifest_path : str
        Path to manifest.json
    plot_name : str
        Unique name for this plot (used as tab label and identifier)
    explanation : str
        Brief description of the plot
    bins : int, default=30
        Number of histogram bins
    color : str, default="#577788"
        Color for histogram bars (hex color)
    show_mean_line : bool, default=True
        If True, show a vertical line at the mean
    show_median_line : bool, default=False
        If True, show a vertical line at the median
    
    Returns
    -------
    dict
        Contains plot_name, url, n_values, statistics, active_plots
    
    Examples
    --------
    >>> add_histogram(
    ...     input_filename="dataset_A1B2C3D4.csv",
    ...     column="MW",
    ...     project_manifest_path="/path/to/manifest.json",
    ...     plot_name="Molecular Weight Distribution",
    ...     explanation="Distribution of molecular weights"
    ... )
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


# Example usage for testing
if __name__ == "__main__":
    df = pd.DataFrame({
        'smiles': ['CCO', 'CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'c1ccccc1'],
        'MW': [46.07, 180.16, 194.19, 206.28, 78.11],
        'logP': [-0.31, 1.19, -0.07, 3.97, 1.88],
        'activity': [5.2, 7.8, 6.1, 4.3, 3.9]
    })
    
    fig = _create_scatter_figure(df, 'MW', 'logP', 'activity')
    
    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="mol-scatter", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="mol-tooltip"),
    ])
    
    @callback(
        Output("mol-tooltip", "show"),
        Output("mol-tooltip", "bbox"),
        Output("mol-tooltip", "children"),
        Input("mol-scatter", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        df_row = df.iloc[num]
        smiles = df_row['smiles']
        img_src = _mol_to_base64(smiles)
        
        children = [
            html.Div([
                html.Img(src=img_src, style={"width": "200px", "display": "block", "margin": "0 auto"}),
                html.P(smiles, style={"font-family": "monospace", "text-align": "center", "margin": "5px"}),
                html.Hr(style={"margin": "5px 0"}),
                html.P(f"MW: {df_row['MW']:.2f}", style={"margin": "2px"}),
                html.P(f"LogP: {df_row['logP']:.2f}", style={"margin": "2px"}),
                html.P(f"Activity: {df_row['activity']:.2f}", style={"margin": "2px", "font-weight": "bold"}),
            ], style={"width": "220px", "padding": "10px"})
        ]

        return True, bbox, children
    
    app.run(debug=True)


