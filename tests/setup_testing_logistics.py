"""
Example demonstrating the MCP-level text report generation function.

This shows the complete end-to-end workflow that MCP users will experience:
- Store datasets
- Call generate_split_quality_text_report() (one function, does everything)
- Get comprehensive text report + JSON analysis
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.infrastructure.resources import _store_resource, _load_resource
from molml_mcp.tools.reports.data_splitting import generate_split_quality_text_report

# Test manifest path
TEST_MANIFEST = Path(__file__).parent / 'data' / 'test_manifest.json'


def example_mcp_workflow():
    """
    Complete MCP workflow: datasets → text report (one function call!)
    """
    print("\n" + "="*80)
    print("MCP TOOL DEMO: generate_split_quality_text_report()")
    print("="*80 + "\n")
    
    # 1. Prepare datasets
    print("Step 1: Preparing datasets...")
    train_df = pd.DataFrame({
        'smiles': [
            'c1ccccc1',      # benzene
            'CCO',           # ethanol
            'CC(=O)C',       # acetone
            'CCN',           # ethylamine
            'CCC',           # propane
            'CCCC',          # butane
            'c1ccc(C)cc1',   # toluene
            'CC(C)O',        # isopropanol
        ],
        'label': [0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    test_df = pd.DataFrame({
        'smiles': [
            'CCCCC',         # pentane
            'c1ccc(CC)cc1',  # ethylbenzene (similar to toluene)
            'CC(=O)CC',      # butanone (similar to acetone)
        ],
        'label': [1, 0, 1]
    })
    
    # Store datasets
    train_file = _store_resource(train_df, str(TEST_MANIFEST), "mcp_train", "Train", "csv")
    test_file = _store_resource(test_df, str(TEST_MANIFEST), "mcp_test", "Test", "csv")
    
    print(f"  ✓ Train: {train_file} ({len(train_df)} molecules)")
    print(f"  ✓ Test:  {test_file} ({len(test_df)} molecules)")
    
    # 2. ONE FUNCTION CALL - generates everything!
    print("\n" + "="*80)
    print("Step 2: Calling MCP tool (ONE function does it all!)...")
    print("="*80)
    
    result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=str(TEST_MANIFEST),
        smiles_col='smiles',
        label_col='label',
        output_filename='mcp_demo_report'
    )
    
    # 3. Review results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"✓ Text Report: {result['output_filename']}")
    print(f"✓ JSON Report: {result['json_report_filename']}")
    print(f"✓ Overall Severity: {result['overall_severity']}")
    print(f"✓ Report Lines: {result['n_lines']}")
    print(f"✓ Report Sections: {len(result['report_sections'])}")
    
    print("\nKey Issues Found:")
    for issue_type, count in result['issues_found'].items():
        if count > 0:
            print(f"  - {issue_type}: {count}")
    
    # 4. Display the text report
    print("\n" + "="*80)
    print("TEXT REPORT PREVIEW (first 60 lines)")
    print("="*80 + "\n")
    
    text_content = _load_resource(str(TEST_MANIFEST), result['output_filename'])
    lines = text_content.split('\n')[:60]
    print('\n'.join(lines))
    print("\n... [report continues] ...")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    print("\n✅ MCP tool generated:")
    print(f"   1. Comprehensive JSON analysis: {result['json_report_filename']}")
    print(f"   2. Human-readable text report: {result['output_filename']}")
    print("\n💡 All from ONE function call to generate_split_quality_text_report()")
    
    return result


def example_with_critical_issues():
    """
    Example showing CRITICAL severity detection.
    """
    print("\n\n" + "="*80)
    print("BONUS EXAMPLE: Detecting CRITICAL Issues")
    print("="*80 + "\n")
    
    # Datasets with exact duplicates
    train_df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO', 'CC(=O)C'],
        'label': [0, 1, 0]
    })
    test_df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCCC'],  # benzene is duplicate!
        'label': [1, 0]
    })
    
    train_file = _store_resource(train_df, str(TEST_MANIFEST), "critical_train", "Train", "csv")
    test_file = _store_resource(test_df, str(TEST_MANIFEST), "critical_test", "Test", "csv")
    
    print("⚠️  Datasets contain exact duplicate: c1ccccc1 (benzene)")
    print("\nRunning analysis...")
    
    result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=str(TEST_MANIFEST),
        smiles_col='smiles',
        label_col='label',
        output_filename='critical_demo_report'
    )
    
    print(f"\n🔴 Overall Severity: {result['overall_severity']}")
    print(f"   Exact duplicates: {result['issues_found']['exact_duplicates']}")
    print(f"\n📄 Full report: {result['output_filename']}")


if __name__ == '__main__':
    print("\n🚀 MCP Tool Demonstration: generate_split_quality_text_report()")
    print("   A complete end-to-end solution for data splitting quality analysis\n")
    
    # Main example
    example_mcp_workflow()
    
    # Bonus: Critical issues
    example_with_critical_issues()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")
