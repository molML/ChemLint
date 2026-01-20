"""Simple test for Dash molecular scatter plots - verifies tooltips work on multiple tabs."""

import pandas as pd
import tempfile
import json
from pathlib import Path
import time

if __name__ == "__main__":
    from src.molml_mcp.tools.core.plotting import (
        add_molecular_scatter_plot,
        list_active_plots,
    )

    # Create temp directory
    tmpdir = Path(tempfile.mkdtemp())
    manifest_path = tmpdir / "manifest.json"
    
    print("=" * 70)
    print("Testing Molecular Scatter Plots - Hover Tooltips on Multiple Tabs")
    print("=" * 70)
    print(f"\n📁 Temp directory: {tmpdir}\n")
    
    # Initialize manifest
    manifest = {"resources": []}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # Dataset 1: Alcohols
    df1 = pd.DataFrame({
        'smiles': ['CCO', 'CC(=O)O', 'CC(C)O'],
        'MW': [46.07, 60.05, 60.10],
        'logP': [-0.31, -0.17, 0.05],
    })
    csv1 = tmpdir / "alcohols.csv"
    df1.to_csv(csv1, index=False)
    manifest["resources"].append({
        "filename": "alcohols.csv",
        "type_tag": "csv",
        "created_at": "2026-01-19",
        "created_by": "test",
        "explanation": "Alcohols"
    })
    
    # Dataset 2: Aromatics
    df2 = pd.DataFrame({
        'smiles': ['c1ccccc1', 'c1ccccc1O', 'c1ccccc1N'],
        'MW': [78.11, 94.11, 93.13],
        'activity': [3.5, 7.2, 5.8],
    })
    csv2 = tmpdir / "aromatics.csv"
    df2.to_csv(csv2, index=False)
    manifest["resources"].append({
        "filename": "aromatics.csv",
        "type_tag": "csv",
        "created_at": "2026-01-19",
        "created_by": "test",
        "explanation": "Aromatics"
    })
    
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    print("✓ Created 2 datasets (3 molecules each)")
    
    # Add plots
    print("\n📊 Adding Plot 1: MW vs LogP (Alcohols)")
    result1 = add_molecular_scatter_plot(
        input_filename="alcohols.csv",
        x_column="MW",
        y_column="logP",
        project_manifest_path=str(manifest_path),
        plot_name="Alcohols",
        explanation="MW vs LogP"
    )
    print(f"   ✓ {result1['message']}")
    
    print("\n📊 Adding Plot 2: MW vs Activity (Aromatics)")
    result2 = add_molecular_scatter_plot(
        input_filename="aromatics.csv",
        x_column="MW",
        y_column="activity",
        project_manifest_path=str(manifest_path),
        plot_name="Aromatics",
        explanation="MW vs Activity"
    )
    print(f"   ✓ {result2['message']}")
    
    # Show active plots
    print("\n" + "=" * 70)
    plots = list_active_plots()
    print(f"🎉 Dashboard ready with {plots['n_plots']} tabs!")
    print(f"\n🌐 URL: {plots['url']}")
    print("\n" + "=" * 70)
    print("Instructions:")
    print("  1. Open browser: http://127.0.0.1:8050/")
    print("  2. You should see 2 tabs: 'Alcohols' and 'Aromatics'")
    print("  3. Hover over points in BOTH tabs to see molecular structures")
    print("  4. Press Ctrl+C when done")
    print("=" * 70)
    
    try:
        print("\n⏳ Server running... (Press Ctrl+C to stop)\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n✓ Stopped. Cleaning up...")
        import shutil
        shutil.rmtree(tmpdir)
        print(f"✓ Cleaned up: {tmpdir}")
        print("👋 Done!")
