import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.reports.quality import generate_quality_report

print("=" * 80)
print("TESTING QUALITY REPORT")
print("=" * 80)

# Setup test data
test_manifest = Path("tests/data/test_manifest.json")

# Create comprehensive test dataset with various issues
test_smiles = [
    'CCO',                           # Valid - Ethanol
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',   # Valid - Ibuprofen
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Valid - Caffeine
    'c1ccccc1',                      # Valid - Benzene
    'CC(=O)Oc1ccccc1C(=O)O',        # Valid - Aspirin
    'INVALID_SMILES',                # Invalid SMILES
    'CCO',                           # Duplicate of row 0
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',   # Duplicate of row 1
    'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC', # Very high MW - outlier
    'C',                             # Very low MW - methane
    'CC(C)(C)c1ccc(O)cc1',          # Might trigger PAINS
    'Cc1ccccc1N',                    # o-Toluidine
    'CC(C)NCC(COc1ccccc1)O',        # Propranolol
    'c1ccc2c(c1)ccc3c2cccc3',       # Anthracene
    'CCCCCCCCCCCCCCC',               # Pentadecane
]

# Create test activities for classification
test_activities = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

df = pd.DataFrame({
    'smiles': test_smiles,
    'activity': test_activities,
    'compound_id': [f'COMP_{i:03d}' for i in range(len(test_smiles))]
})

print(f"\nCreated test dataset with {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head(3))

# Store dataset
input_filename = _store_resource(
    df,
    str(test_manifest),
    "quality_test_input",
    "Test dataset for quality report",
    'csv'
)
print(f"\nStored test dataset as: {input_filename}")

# Test 1: Basic quality report (with activity)
print("\n" + "=" * 80)
print("TEST 1: Quality Report with Classification Activity")
print("=" * 80)

try:
    result = generate_quality_report(
        input_filename=input_filename,
        project_manifest_path=str(test_manifest),
        smiles_col='smiles',
        output_name='quality_test_output',
        activity_col='activity',
        activity_type='classification'
    )
    
    print("\n✅ SUCCESS - Quality report generated")
    print(f"JSON output: {result['calculations_json']}")
    print(f"Text output: {result['report_txt']}")
    print(f"Number of molecules: {result['n_molecules']}")
    print(f"Critical issues found: {result['n_issues']}")
    
    # Display key metrics
    print("\nKey Metrics from Report:")
    for key, value in result.get('key_metrics', {}).items():
        if key != 'scaffold_diversity':  # Skip nested dict
            print(f"  {key}: {value}")
    
    # Display warnings
    if result.get('warnings'):
        print(f"\nWarnings ({len(result['warnings'])}):")
        for warning in result['warnings'][:5]:
            print(f"  - {warning}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Quality report without activity
print("\n" + "=" * 80)
print("TEST 2: Quality Report WITHOUT Activity Column")
print("=" * 80)

# Create dataset without activity
df_no_activity = df[['smiles', 'compound_id']].copy()

input_filename_2 = _store_resource(
    df_no_activity,
    str(test_manifest),
    "quality_test_no_activity",
    "Test dataset without activity",
    'csv'
)

try:
    result2 = generate_quality_report(
        input_filename=input_filename_2,
        project_manifest_path=str(test_manifest),
        smiles_col='smiles',
        output_name='quality_test_no_activity_output'
    )
    
    print("\n✅ SUCCESS - Quality report generated (no activity)")
    print(f"JSON output: {result2['calculations_json']}")
    print(f"Text output: {result2['report_txt']}")
    print(f"Number of molecules: {result2['n_molecules']}")
    print(f"Critical issues found: {result2['n_issues']}")
    
    # Display key metrics
    print("\nKey Metrics from Report:")
    for key, value in result2.get('key_metrics', {}).items():
        if key != 'scaffold_diversity':  # Skip nested dict
            print(f"  {key}: {value}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Quality report with regression activity
print("\n" + "=" * 80)
print("TEST 3: Quality Report with Regression Activity")
print("=" * 80)

# Create regression activity values
import numpy as np
df_regression = df.copy()
df_regression['ic50'] = np.random.uniform(0.1, 100.0, len(df))

input_filename_3 = _store_resource(
    df_regression,
    str(test_manifest),
    "quality_test_regression",
    "Test dataset with regression activity",
    'csv'
)

try:
    result3 = generate_quality_report(
        input_filename=input_filename_3,
        project_manifest_path=str(test_manifest),
        smiles_col='smiles',
        output_name='quality_test_regression_output',
        activity_col='ic50',
        activity_type='regression',
        activity_units='nM'
    )
    
    print("\n✅ SUCCESS - Quality report generated (regression)")
    print(f"JSON output: {result3['calculations_json']}")
    print(f"Text output: {result3['report_txt']}")
    print(f"Number of molecules: {result3['n_molecules']}")
    print(f"Critical issues found: {result3['n_issues']}")
    
    # Display key metrics
    print("\nKey Metrics from Report:")
    for key, value in result3.get('key_metrics', {}).items():
        if key != 'scaffold_diversity':  # Skip nested dict
            print(f"  {key}: {value}")
    for key, value in result3.get('key_metrics', {}).items():
        print(f"  {key}: {value}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)
