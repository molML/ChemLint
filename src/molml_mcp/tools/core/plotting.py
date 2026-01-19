"""
Interactive plotting functions for molecular data visualization.

Uses a persistent Dash server with tabs - plots can be dynamically added/removed.
Supports both scatter plots (with molecular structure tooltips) and histograms.
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
import numpy as np
from scipy import stats
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
            html.P("No plots yet. Use add_molecular_scatter_plot, add_histogram, add_density_plot, or add_box_plot to create visualizations.", 
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


def add_density_plot(
    input_filename: str,
    column: str,
    project_manifest_path: str,
    plot_name: str,
    explanation: str,
    bandwidth: str = 'scott',
    fill: bool = True,
    color: str = "#577788",
    show_rug: bool = True,
    show_mean_line: bool = True,
    show_median_line: bool = False
) -> dict:
    """
    Add an interactive density plot (KDE) to the persistent visualization dashboard.
    
    Creates a new tab in the Dash visualization server showing a kernel density estimate.
    If the server isn't running, it will be started automatically.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    column : str
        Column name to plot as density
    project_manifest_path : str
        Path to manifest.json
    plot_name : str
        Unique name for this plot (used as tab label and identifier)
    explanation : str
        Brief description of the plot
    bandwidth : str, default='scott'
        Method to determine bandwidth: 'scott', 'silverman', or a numeric value
    fill : bool, default=True
        If True, fill the area under the density curve
    color : str, default="#577788"
        Color for density curve (hex color)
    show_rug : bool, default=True
        If True, show rug plot (individual data points) at bottom
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
    >>> add_density_plot(
    ...     input_filename="dataset_A1B2C3D4.csv",
    ...     column="MW",
    ...     project_manifest_path="/path/to/manifest.json",
    ...     plot_name="Molecular Weight Density",
    ...     explanation="Density distribution of molecular weights"
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
    
    if len(data) < 2:
        raise ValueError(f"Need at least 2 data points for density estimation, got {len(data)}")
    
    # Calculate statistics
    mean_val = float(data.mean())
    median_val = float(data.median())
    min_val = float(data.min())
    max_val = float(data.max())
    std_val = float(data.std())
    
    # Generate unique plot ID
    plot_id = plot_name.lower().replace(" ", "-").replace("/", "-")
    
    # Check if plot already exists
    if plot_id in _active_plots:
        raise ValueError(
            f"Plot '{plot_name}' already exists. Use a different name or remove it first."
        )
    
    # Create density estimate using KDE
    data_array = np.array(data)
    
    # Determine bandwidth
    if bandwidth == 'scott':
        bw_method = 'scott'
    elif bandwidth == 'silverman':
        bw_method = 'silverman'
    else:
        try:
            bw_method = float(bandwidth)
        except (ValueError, TypeError):
            bw_method = 'scott'
    
    kde = stats.gaussian_kde(data_array, bw_method=bw_method)
    
    # Create x values for smooth curve
    x_range = max_val - min_val
    x_min = min_val - 0.1 * x_range
    x_max = max_val + 0.1 * x_range
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = kde(x_vals)
    
    # Create density plot figure
    fig = go.Figure()
    
    # Add density curve
    if fill:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Density',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=color,
            opacity=0.6
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Density',
            line=dict(color=color, width=2)
        ))
    
    # Add rug plot (individual data points)
    if show_rug:
        # Sample if too many points
        rug_data = data_array if len(data_array) <= 1000 else np.random.choice(data_array, 1000, replace=False)
        rug_y = [-0.02 * max(y_vals)] * len(rug_data)  # Small negative offset
        fig.add_trace(go.Scatter(
            x=rug_data,
            y=rug_y,
            mode='markers',
            name='Data points',
            marker=dict(
                color=color,
                size=4,
                symbol='line-ns-open',
                line=dict(width=1)
            ),
            showlegend=False
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
        yaxis=dict(title="Density"),
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
            'type': 'density',
            'column': column,
            'show_structures': False,
            'explanation': explanation,
            'statistics': {
                'n_values': len(data),
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
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
            "std": std_val,
            "min": min_val,
            "max": max_val
        },
        "active_plots": list(_active_plots.keys()),
        "message": f"Density plot '{plot_name}' added. Visit {url} to view all {len(_active_plots)} plot(s)."
    }


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
    """
    Add an interactive box plot to the persistent visualization dashboard.
    
    Creates a new tab in the Dash visualization server showing box-and-whisker plot(s).
    Supports grouping to show multiple box plots side by side.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    column : str
        Column name containing values to plot
    project_manifest_path : str
        Path to manifest.json
    plot_name : str
        Unique name for this plot (used as tab label and identifier)
    explanation : str
        Brief description of the plot
    group_column : str, optional
        Column name for grouping. If provided, creates separate box plots for each group
    color : str, default="#577788"
        Color for box plots (hex color)
    show_points : bool, default=False
        If True, show individual data points overlaid on boxes
    notched : bool, default=False
        If True, show notched boxes (indicates confidence interval around median)
    
    Returns
    -------
    dict
        Contains plot_name, url, n_values, statistics, active_plots
    
    Examples
    --------
    Single box plot:
    
    >>> add_box_plot(
    ...     input_filename="dataset_A1B2C3D4.csv",
    ...     column="MW",
    ...     project_manifest_path="/path/to/manifest.json",
    ...     plot_name="Molecular Weight Distribution",
    ...     explanation="Box plot of molecular weights"
    ... )
    
    Grouped box plot:
    
    >>> add_box_plot(
    ...     input_filename="dataset_A1B2C3D4.csv",
    ...     column="Activity",
    ...     group_column="Cluster",
    ...     project_manifest_path="/path/to/manifest.json",
    ...     plot_name="Activity by Cluster",
    ...     explanation="Box plot of activity values grouped by cluster"
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
            "message": "No active plots. Use add_molecular_scatter_plot, add_histogram, add_density_plot, or add_box_plot to create visualizations."
        }
    
    plots_info = []
    for plot_id, plot_data in _active_plots.items():
        plots_info.append({
            "name": plot_data['label'],
            "plot_id": plot_id,
            "type": plot_data.get('type', 'scatter'),
            "explanation": plot_data['explanation']
        })
    
    url = f"http://127.0.0.1:{_PORT}/"
    
    return {
        "plots": plots_info,
        "n_plots": len(_active_plots),
        "url": url,
        "message": f"{len(_active_plots)} plot(s) active. Visit {url} to view the dashboard."
    }


