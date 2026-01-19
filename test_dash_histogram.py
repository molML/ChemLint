#!/usr/bin/env python3
"""
Quick test for add_histogram function with Dash dashboard.
Creates scatter plots AND histograms to verify they work together.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from molml_mcp.tools.core.plotting import add_histogram, add_molecular_scatter_plot, list_active_plots
from molml_mcp.infrastructure.resources import _store_resource
import pandas as pd
import tempfile
import time

def test_histogram_and_scatter():
    """Test histogram and scatter plot together."""
    # Create temp directory for manifest
    temp_dir = tempfile.mkdtemp()
    manifest_path = os.path.join(temp_dir, "manifest.json")
    
    print(f"Test manifest at: {manifest_path}")
    print()
    
    # Create test datasets
    df_alcohols = pd.DataFrame({
        'SMILES': ['CCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO'],
        'MW': [46.07, 60.10, 74.12, 88.15, 102.17],
        'LogP': [-0.18, 0.28, 0.77, 1.24, 1.73],
        'Activity': [5.2, 6.1, 7.3, 6.8, 5.9]
    })
    
    df_aromatics = pd.DataFrame({
        'SMILES': ['c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1', 'c1ccc(C)cc1C', 'c1ccc(CC)cc1C'],
        'MW': [78.11, 92.14, 106.17, 106.17, 120.19],
        'LogP': [1.98, 2.69, 3.15, 3.18, 3.63],
        'Activity': [4.5, 5.8, 7.2, 8.1, 6.9]
    })
    
    # Store datasets
    print("Storing datasets...")
    alcohols_file = _store_resource(
        df_alcohols, manifest_path, "alcohols", 
        "Alcohol test dataset", 'csv'
    )
    aromatics_file = _store_resource(
        df_aromatics, manifest_path, "aromatics",
        "Aromatic test dataset", 'csv'
    )
    print(f"  - {alcohols_file}")
    print(f"  - {aromatics_file}")
    print()
    
    # Test 1: Add scatter plot for alcohols
    print("Test 1: Adding scatter plot for alcohols...")
    result1 = add_molecular_scatter_plot(
        input_filename=alcohols_file,
        smiles_column='SMILES',
        x_column='MW',
        y_column='Activity',
        project_manifest_path=manifest_path,
        plot_name='Alcohols Activity',
        explanation='Scatter plot of MW vs Activity for alcohols',
        color_column='LogP',
        size_column=None
    )
    print(f"  ✓ Created: {result1['plot_name']}")
    print(f"  URL: {result1['url']}")
    print()
    
    # Test 2: Add histogram for MW
    print("Test 2: Adding histogram for MW distribution...")
    result2 = add_histogram(
        input_filename=alcohols_file,
        column='MW',
        project_manifest_path=manifest_path,
        plot_name='MW Distribution',
        explanation='Distribution of molecular weights',
        bins=10,
        color='#3498DB',
        show_mean_line=True,
        show_median_line=True
    )
    print(f"  ✓ Created: {result2['plot_name']}")
    print(f"  Statistics: mean={result2['statistics']['mean']:.2f}, median={result2['statistics']['median']:.2f}")
    print()
    
    # Test 3: Add another scatter plot for aromatics
    print("Test 3: Adding scatter plot for aromatics...")
    result3 = add_molecular_scatter_plot(
        input_filename=aromatics_file,
        smiles_column='SMILES',
        x_column='LogP',
        y_column='Activity',
        project_manifest_path=manifest_path,
        plot_name='Aromatics Activity',
        explanation='Scatter plot of LogP vs Activity for aromatics',
        color_column=None,
        size_column='MW'
    )
    print(f"  ✓ Created: {result3['plot_name']}")
    print()
    
    # Test 4: Add histogram for Activity
    print("Test 4: Adding histogram for Activity distribution...")
    result4 = add_histogram(
        input_filename=aromatics_file,
        column='Activity',
        project_manifest_path=manifest_path,
        plot_name='Activity Distribution',
        explanation='Distribution of activity values',
        bins=8,
        color='#E74C3C',
        show_mean_line=True,
        show_median_line=False
    )
    print(f"  ✓ Created: {result4['plot_name']}")
    print()
    
    # List all active plots
    print("All active plots:")
    plots = list_active_plots()
    for i, plot in enumerate(plots['plots'], 1):
        print(f"  {i}. {plot['label']} ({plot['type']})")
    print()
    
    print("=" * 60)
    print(f"Dashboard running at: {result1['url']}")
    print("=" * 60)
    print("\nTest completed! Server will stay running.")
    print("Press Ctrl+C to exit.")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    test_histogram_and_scatter()
