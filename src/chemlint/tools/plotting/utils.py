"""
Shared utilities for plotting functions.

Manages global Dash server state, callback registration, and helper functions.
"""

from dash import Dash, dcc, html, Input, Output, no_update, ctx
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from io import BytesIO
import pandas as pd
import threading
import time
import numpy as np

# Optional dependency for high-quality rendering
try:
    import cairosvg
    HAS_CAIROSVG = True
except Exception:
    cairosvg = None
    HAS_CAIROSVG = False

# Global state for persistent Dash server
_dash_app = None
_dash_thread = None
_active_plots = {}  # plot_id -> plot_data dict
_server_lock = threading.RLock()  # Use RLock for reentrant locking
_PORT = 8050


def _mol_to_base64(smiles, size=(200, 200), use_acs1996=True):
    """
    Convert SMILES to base64 encoded PNG image.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    size : tuple
        Image size (width, height). If using ACS1996 with CairoSVG, output will be 4× this size.
    use_acs1996 : bool
        If True (default), use ACS1996 rendering style (same as smiles_to_acs1996_png)
    
    Returns
    -------
    str or None
        Base64 encoded data URI or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if use_acs1996:
        # ACS1996 rendering style (matches smiles_to_acs1996_png)
        m = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(m)
        
        if HAS_CAIROSVG:
            # High-quality SVG → PNG route with 4× scaling
            drawer = rdMolDraw2D.MolDraw2DSVG(-1, -1)  # flexicanvas
            opts = drawer.drawOptions()
            mean_bond_len = Draw.MeanBondLength(m) or 1.5
            Draw.SetACS1996Mode(opts, mean_bond_len)
            
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, m, legend='')
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            
            png_bytes = cairosvg.svg2png(
                bytestring=svg.encode("utf-8"),
                scale=4.0,
            )
        else:
            # Fallback: pure RDKit PNG at base_size
            w, h = size
            drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
            opts = drawer.drawOptions()
            mean_bond_len = Draw.MeanBondLength(m) or 1.5
            Draw.SetACS1996Mode(opts, mean_bond_len)
            
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, m, legend='')
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()
        
        img_str = base64.b64encode(png_bytes).decode()
    else:
        # Original simple rendering
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
            html.P("No plots yet. Use plotting functions to create visualizations.", 
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
        
        # Use ACS1996 style if specified in plot data (defaults to False for backward compatibility)
        use_acs1996 = plot_data.get('use_acs1996', False)
        img_src = _mol_to_base64(smiles, use_acs1996=use_acs1996)
        
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
                import sys
                import os
                
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
                # Redirect stdout/stderr to devnull to prevent MCP JSON errors
                devnull = open(os.devnull, 'w')
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                
                try:
                    _dash_app.run(debug=False, port=_PORT, use_reloader=False, host='127.0.0.1')
                except OSError as e:
                    # Port already in use - that's okay, server is already running
                    if "Address already in use" in str(e) or "address already in use" in str(e):
                        pass
                    else:
                        raise
                finally:
                    # Restore stdout/stderr
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    devnull.close()
            
            _dash_thread = threading.Thread(target=run_server, daemon=True)
            _dash_thread.start()
            
            # Give the server a moment to start
            time.sleep(0.5)
