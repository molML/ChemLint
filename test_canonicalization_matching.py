"""
Test to verify that salt/solvent matching works with non-canonical SMILES.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.reports.quality import generate_quality_report

print("=" * 80)
print("TESTING CANONICALIZATION IN PATTERN MATCHING")
print("=" * 80)

# Setup test data
test_manifest = Path("tests/data/test_manifest.json")

# Test SMILES with salts/solvents in different representations
# These are intentionally non-canonical to test the canonicalization
test_smiles = [
    'CCO.Cl',                           # Ethanol + HCl (non-canonical salt)
    'c1ccccc1.[Na+].[Cl-]',            # Benzene + NaCl (non-canonical)
    'CC(=O)O.O',                        # Acetic acid + Water (non-canonical water)
    'c1ccc(cc1)O.[Na]Cl',              # Phenol + NaCl (alternative representation)
    'CCO',                              # Just ethanol (no salt)
    'c1ccccc1.Cl',                      # Benzene + Cl (canonical Cl is [Cl-])
    'CC(C)O.C1CCOC1',                   # Isopropanol + THF
    'Cc1ccccc1.ClCCl',                  # Toluene + DCM (canonical)
]

df = pd.DataFrame({
    'smiles': test_smiles,
    'compound_id': [f'TEST_{i:03d}' for i in range(len(test_smiles))]
})

print(f"\nCreated test dataset with {len(df)} rows")
print(f"Testing salt/solvent detection with non-canonical SMILES:\n")
for i, smi in enumerate(test_smiles):
    print(f"  {i+1}. {smi}")

# Store dataset
input_filename = _store_resource(
    df,
    str(test_manifest),
    'canonicalization_test',
    'Test dataset for canonicalization in pattern matching',
    'csv'
)

print(f"\nStored test dataset: {input_filename}")

# Generate quality report
print("\n" + "=" * 80)
print("GENERATING QUALITY REPORT")
print("=" * 80)

result = generate_quality_report(
    input_filename=input_filename,
    project_manifest_path=str(test_manifest),
    smiles_col='smiles',
    output_name='canonicalization_test_output'
)

print("\n✅ Quality report generated successfully")

# Read the text report and extract the salt/solvent section
import re
txt_filename = result['report_txt']
with open(f"tests/data/{txt_filename}", 'r') as f:
    report_text = f.read()

# Find the Salts/Fragments/Solvents section
pattern = r"17\. SALTS/FRAGMENTS/SOLVENTS.*?(?=\n-{80}\n\d+\.|$)"
match = re.search(pattern, report_text, re.DOTALL)

if match:
    print("\n" + "=" * 80)
    print("SECTION 17: SALTS/FRAGMENTS/SOLVENTS")
    print("=" * 80)
    print(match.group(0))
else:
    print("\n⚠️  Could not find Section 17 in report")
    print("\nSearching for any mention of salts or solvents...")
    if 'salt' in report_text.lower() or 'solvent' in report_text.lower():
        lines = report_text.split('\n')
        for i, line in enumerate(lines):
            if 'salt' in line.lower() or 'solvent' in line.lower():
                print(f"Line {i}: {line}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nExpected results:")
print("  - Should detect HCl in multiple representations (Cl, [Cl-], etc.)")
print("  - Should detect NaCl in different formats")
print("  - Should detect water (O) even when part of mixture")
print("  - Should detect common solvents regardless of canonicalization")
