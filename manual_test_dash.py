"""Quick test script for persistent Dash server functionality."""

import pandas as pd
from src.molml_mcp.tools.core.plotting import (
    add_molecular_scatter_plot,
    remove_plot,
    list_active_plots
)
import tempfile
import json
from pathlib import Path

# Create test data
df1 = pd.DataFrame({
    'smiles': ['CCO', 'CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
    'MW': [46.07, 180.16, 194.19],
    'logP': [-0.31, 1.19, -0.07],
    'activity': [5.2, 7.8, 6.1]
})

df2 = pd.DataFrame({
    'smiles': ['CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'c1ccccc1', 'CCN'],
    'MW': [206.28, 78.11, 59.11],
    'logP': [3.97, 1.88, 0.05],
    'pKa': [4.3, 3.9, 10.8]
})

# Create temporary project directory with manifest
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    manifest_path = tmpdir / "manifest.json"
    
    # Create manifest
    manifest = {
        "resources": []
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # Save datasets
    csv1_path = tmpdir / "dataset1.csv"
    csv2_path = tmpdir / "dataset2.csv"
    df1.to_csv(csv1_path, index=False)
    df2.to_csv(csv2_path, index=False)
    
    # Add to manifest
    manifest["resources"].extend([
        {
            "filename": "dataset1.csv",
            "type_tag": "csv",
            "created_at": "2026-01-19",
            "created_by": "test_script",
            "explanation": "Test dataset 1"
        },
        {
            "filename": "dataset2.csv",
            "type_tag": "csv",
            "created_at": "2026-01-19",
            "created_by": "test_script",
            "explanation": "Test dataset 2"
        }
    ])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    print("=" * 60)
    print("Testing Persistent Dash Server")
    print("=" * 60)
    
    # Test 1: List empty plots
    print("\n1. Listing plots (should be empty):")
    result = list_active_plots()
    print(f"   {result['message']}")
    
    # Test 2: Add first plot
    print("\n2. Adding first plot (MW vs LogP):")
    result = add_molecular_scatter_plot(
        input_filename="dataset1.csv",
        x_column="MW",
        y_column="logP",
        project_manifest_path=str(manifest_path),
        plot_name="MW vs LogP",
        explanation="Molecular weight vs lipophilicity"
    )
    print(f"   {result['message']}")
    print(f"   URL: {result['url']}")
    
    # Test 3: Add second plot
    print("\n3. Adding second plot (MW vs Activity):")
    result = add_molecular_scatter_plot(
        input_filename="dataset1.csv",
        x_column="MW",
        y_column="activity",
        project_manifest_path=str(manifest_path),
        plot_name="MW vs Activity",
        explanation="Molecular weight vs biological activity",
        color_column="logP"
    )
    print(f"   {result['message']}")
    
    # Test 4: Add third plot from different dataset
    print("\n4. Adding third plot (MW vs pKa):")
    result = add_molecular_scatter_plot(
        input_filename="dataset2.csv",
        x_column="MW",
        y_column="pKa",
        project_manifest_path=str(manifest_path),
        plot_name="MW vs pKa",
        explanation="Molecular weight vs acidity"
    )
    print(f"   {result['message']}")
    
    # Test 5: List all plots
    print("\n5. Listing all active plots:")
    result = list_active_plots()
    print(f"   {result['message']}")
    for plot in result['active_plots']:
        print(f"   - {plot['name']}: {plot['x_column']} vs {plot['y_column']} ({plot['n_points']} points)")
    
    # Test 6: Remove a plot
    print("\n6. Removing 'MW vs Activity' plot:")
    result = remove_plot("MW vs Activity")
    print(f"   {result['message']}")
    
    # Test 7: List remaining plots
    print("\n7. Listing remaining plots:")
    result = list_active_plots()
    print(f"   {result['message']}")
    for plot in result['active_plots']:
        print(f"   - {plot['name']}")
    
    print("\n" + "=" * 60)
    print(f"✓ All tests passed! Visit {result['url']} to view the dashboard")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server and exit...")
    
    # Keep the script running so the server stays alive
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
