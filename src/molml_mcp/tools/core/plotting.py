"""
Plotting functions for molecular data visualization.
"""

import pandas as pd
import io
from plotnine import (
    ggplot, aes, geom_histogram, geom_point, theme_minimal, theme, element_text, 
    element_line, element_rect, labs, scale_fill_manual, scale_color_gradient,
    scale_color_manual, scale_color_cmap, scale_color_cmap_d, guide_colorbar, guides
)
from mcp.server.fastmcp import Image
from molml_mcp.infrastructure.resources import _load_resource


def plot_histogram(
    input_filename: str,
    column: str,
    project_manifest_path: str,
    bins: int = 30,
    fill_color: str = "#577788",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Count",
    width: float = 6.0,
    height: float = 4.0,
    dpi: int = 300
) -> list:
    """
    Create a publication-quality histogram using plotnine (Nature paper style).
    
    Generates a clean, professional histogram with minimalist styling suitable
    for scientific publications. Returns the image for inline display.
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    column : str
        Name of the column to plot.
    project_manifest_path : str
        Path to the project manifest JSON file.
    bins : int
        Number of histogram bins (default: 30).
    fill_color : str
        Hex color for histogram bars (default: "#577788" - blue-gray).
    title : str | None
        Plot title (default: None for no title).
    xlabel : str | None
        X-axis label (default: column name).
    ylabel : str
        Y-axis label (default: "Count").
    width : float
        Figure width in inches (default: 6.0).
    height : float
        Figure height in inches (default: 4.0).
    dpi : int
        Resolution in dots per inch (default: 300).
    
    Returns
    -------
    list
        [Image, str] - FastMCP Image object and summary statistics string
    
    Examples
    --------
    Basic histogram:
    
        img, stats = plot_histogram(
            input_filename='dataset_AB12CD34.csv',
            column='molecular_weight',
            project_manifest_path='/path/to/manifest.json'
        )
    
    Customized histogram:
    
        img, stats = plot_histogram(
            input_filename='dataset_AB12CD34.csv',
            column='logP',
            project_manifest_path='/path/to/manifest.json',
            bins=40,
            fill_color='#3498DB',
            title='LogP Distribution',
            xlabel='LogP',
            ylabel='Frequency'
        )
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column exists
    if column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"Column '{column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Extract data and remove NaN values
    data = df[column].dropna()
    
    if len(data) == 0:
        raise ValueError(f"Column '{column}' contains no valid (non-NaN) values")
    
    # Calculate statistics
    stats = {
        "n_values": len(data),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "median": float(data.median())
    }
    
    # Create DataFrame for plotnine
    plot_df = pd.DataFrame({column: data})
    
    # Set default labels
    if xlabel is None:
        xlabel = column
    
    # Create the plot with Nature paper styling
    p = (
        ggplot(plot_df, aes(x=column))
        + geom_histogram(bins=bins, fill=fill_color, color='white', size=0.3, alpha=0.9)
        + labs(
            title=title if title else '',
            x=xlabel,
            y=ylabel
        )
        + theme_minimal()
        + theme(
            # Text elements - clean and legible
            text=element_text(family='Arial', size=11, color='#2C3E50'),
            plot_title=element_text(size=13, face='bold', margin={'b': 15}) if title else element_text(size=0),
            axis_title_x=element_text(size=11, face='bold', margin={'t': 10}),
            axis_title_y=element_text(size=11, face='bold', margin={'r': 10}),
            axis_text=element_text(size=9, color='#34495E'),
            
            # Grid - subtle and minimal
            panel_grid_major=element_line(color='#ECF0F1', size=0.5),
            panel_grid_minor=element_line(color='#ECF0F1', size=0.25),
            
            # Background - clean white
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            
            # Axes - subtle lines
            axis_line=element_line(color='#95A5A6', size=0.5),
            
            # Remove top and right spines for cleaner look
            panel_border=element_rect(color='none'),
            
            # Adjust plot margins
            plot_margin=0.05
        )
    )
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    p.save(buf, format='png', width=width, height=height, dpi=dpi, verbose=False)
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()
    
    # Create FastMCP Image object
    img = Image(data=png_bytes, format="png")
    
    # Create summary statistics string
    summary = (
        f"Histogram of '{column}': {stats['n_values']} values, "
        f"range [{stats['min']:.2f}, {stats['max']:.2f}], "
        f"mean={stats['mean']:.2f}, median={stats['median']:.2f}"
    )
    
    return [img, summary]


def plot_scatter(
    input_filename: str,
    x_column: str,
    y_column: str,
    project_manifest_path: str,
    color_column: str | None = None,
    color_palette: str = "viridis",
    treat_color_as_categorical: bool = False,
    point_size: float = 3.0,
    point_alpha: float = 0.7,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: float = 6.0,
    height: float = 5.0,
    dpi: int = 300
) -> list:
    """
    Create a publication-quality scatter plot using plotnine (Nature paper style).
    
    Generates a clean, professional scatter plot with optional color-coding by
    a third variable (categorical or continuous). Uses minimalist styling suitable
    for scientific publications.
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    x_column : str
        Name of the column for x-axis.
    y_column : str
        Name of the column for y-axis.
    project_manifest_path : str
        Path to the project manifest JSON file.
    color_column : str | None
        Optional column name for color-coding points. Can be categorical or continuous.
        If None, all points will be the same color (default: None).
    color_palette : str
        Color palette for continuous data or categorical data (default: "viridis").
        Options: "viridis", "plasma", "inferno", "magma", "cividis" for continuous,
        or "Set1", "Set2", "Set3", "Paired" for categorical.
    treat_color_as_categorical : bool
        Force color column to be treated as categorical even if it's numerical.
        Useful for numerical cluster IDs (default: False).
    point_size : float
        Size of scatter points (default: 3.0).
    point_alpha : float
        Transparency of points, 0.0 (transparent) to 1.0 (opaque) (default: 0.7).
    title : str | None
        Plot title (default: None for no title).
    xlabel : str | None
        X-axis label (default: x_column name).
    ylabel : str | None
        Y-axis label (default: y_column name).
    width : float
        Figure width in inches (default: 6.0).
    height : float
        Figure height in inches (default: 5.0).
    dpi : int
        Resolution in dots per inch (default: 300).
    
    Returns
    -------
    list
        [Image, str] - FastMCP Image object and summary statistics string
    
    Examples
    --------
    Basic scatter plot (no color):
    
        img, stats = plot_scatter(
            input_filename='dataset_AB12CD34.csv',
            x_column='molecular_weight',
            y_column='logP',
            project_manifest_path='/path/to/manifest.json'
        )
    
    Scatter plot with continuous color variable:
    
        img, stats = plot_scatter(
            input_filename='dataset_AB12CD34.csv',
            x_column='molecular_weight',
            y_column='logP',
            color_column='pIC50',
            project_manifest_path='/path/to/manifest.json',
            color_palette='viridis',
            title='MW vs LogP colored by activity'
        )
    
    Scatter plot with categorical color variable:
    
        img, stats = plot_scatter(
            input_filename='dataset_AB12CD34.csv',
            x_column='PC1',
            y_column='PC2',
            color_column='cluster',
            project_manifest_path='/path/to/manifest.json',
            color_palette='Set2'
        )
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns exist
    required_columns = [x_column, y_column]
    if color_column:
        required_columns.append(color_column)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"Column(s) {missing_columns} not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Create plot dataframe, removing rows with NaN in required columns
    plot_df = df[required_columns].dropna()
    
    if len(plot_df) == 0:
        raise ValueError(f"No valid (non-NaN) data points found")
    
    # Calculate statistics
    stats = {
        "n_points": len(plot_df),
        "x_range": [float(plot_df[x_column].min()), float(plot_df[x_column].max())],
        "y_range": [float(plot_df[y_column].min()), float(plot_df[y_column].max())],
        "x_mean": float(plot_df[x_column].mean()),
        "y_mean": float(plot_df[y_column].mean())
    }
    
    # Set default labels
    if xlabel is None:
        xlabel = x_column
    if ylabel is None:
        ylabel = y_column
    
    # Determine if color column is categorical or continuous
    is_categorical = False
    if color_column:
        # User can force categorical treatment
        if treat_color_as_categorical:
            is_categorical = True
        # Check if column is categorical (object/string type or few unique values)
        elif plot_df[color_column].dtype == 'object' or plot_df[color_column].dtype.name == 'category':
            is_categorical = True
        elif plot_df[color_column].nunique() <= 10:  # Heuristic: <= 10 unique values = categorical
            is_categorical = True
        
        # Convert numerical columns to strings if treating as categorical
        if is_categorical and plot_df[color_column].dtype != 'object':
            plot_df[color_column] = plot_df[color_column].astype(str)
    
    # Create the plot
    if color_column:
        p = ggplot(plot_df, aes(x=x_column, y=y_column, color=color_column))
    else:
        p = ggplot(plot_df, aes(x=x_column, y=y_column))
    
    # Add scatter points
    if color_column:
        p = p + geom_point(size=point_size, alpha=point_alpha)
    else:
        p = p + geom_point(size=point_size, alpha=point_alpha, color='#577788')
    
    # Add color scale
    if color_column:
        if is_categorical:
            # Categorical color scale - use discrete cmap
            p = p + scale_color_cmap_d(cmap_name=color_palette)
        else:
            # Continuous color scale
            p = (p + scale_color_cmap(cmap_name=color_palette)
                 + guides(color=guide_colorbar(title=color_column)))
    
    # Add labels and theme
    p = (p
        + labs(
            title=title if title else '',
            x=xlabel,
            y=ylabel
        )
        + theme_minimal()
        + theme(
            # Text elements - clean and legible
            text=element_text(family='Arial', size=11, color='#2C3E50'),
            plot_title=element_text(size=13, face='bold', margin={'b': 15}) if title else element_text(size=0),
            axis_title_x=element_text(size=11, face='bold', margin={'t': 10}),
            axis_title_y=element_text(size=11, face='bold', margin={'r': 10}),
            axis_text=element_text(size=9, color='#34495E'),
            
            # Grid - subtle and minimal
            panel_grid_major=element_line(color='#ECF0F1', size=0.5),
            panel_grid_minor=element_line(color='#ECF0F1', size=0.25),
            
            # Background - clean white
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            
            # Axes - subtle lines
            axis_line=element_line(color='#95A5A6', size=0.5),
            
            # Remove top and right spines for cleaner look
            panel_border=element_rect(color='none'),
            
            # Legend styling
            legend_background=element_rect(fill='white', color='none'),
            legend_key=element_rect(fill='white', color='none'),
            legend_title=element_text(size=10, face='bold'),
            legend_text=element_text(size=9),
            
            # Adjust plot margins
            plot_margin=0.05
        )
    )
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    p.save(buf, format='png', width=width, height=height, dpi=dpi, verbose=False)
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()
    
    # Create FastMCP Image object
    img = Image(data=png_bytes, format="png")
    
    # Create summary statistics string
    color_info = f", colored by '{color_column}'" if color_column else ""
    summary = (
        f"Scatter plot of '{y_column}' vs '{x_column}'{color_info}: "
        f"{stats['n_points']} points, "
        f"x range [{stats['x_range'][0]:.2f}, {stats['x_range'][1]:.2f}], "
        f"y range [{stats['y_range'][0]:.2f}, {stats['y_range'][1]:.2f}]"
    )
    
    return [img, summary]
