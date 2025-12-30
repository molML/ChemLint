"""
Comprehensive test suite for all internal _analyze... functions in quality.py

This test suite rigorously validates:
1. Edge cases (empty data, all NaN, single row)
2. Invalid inputs (bad SMILES, wrong types)
3. Boundary conditions (extreme values)
4. Data type handling (int, float, string variations)
5. Return value structure and types
6. Robustness to malformed data
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from molml_mcp.tools.reports.quality import (
    _analyze_smiles_validity,
    _analyze_activity_distribution,
    _analyze_functional_groups,
    _analyze_stereochemistry,
    _analyze_charge_state,
    _analyze_salts_fragments_solvents,
    _analyze_special_features,
    _analyze_activity_correlations
)

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_test(self, condition, test_name, error_msg=""):
        if condition:
            self.passed += 1
            print(f"✅ {test_name}")
        else:
            self.failed += 1
            msg = f"❌ {test_name}"
            if error_msg:
                msg += f": {error_msg}"
            print(msg)
            self.errors.append(test_name)
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        if self.errors:
            print("\nFailed tests:")
            for error in self.errors:
                print(f"  - {error}")
        print("=" * 80)

results = TestResults()

print("=" * 80)
print("COMPREHENSIVE QUALITY REPORT FUNCTION TESTS")
print("=" * 80)
print()

# =============================================================================
# TEST 1: _analyze_smiles_validity
# =============================================================================
print("TEST 1: _analyze_smiles_validity")
print("-" * 80)

# Test 1.1: Valid SMILES
df1 = pd.DataFrame({'smiles': ['CCO', 'c1ccccc1', 'CC(=O)O']})
result = _analyze_smiles_validity(df1, 'smiles')
results.assert_test(
    result['n_valid'] == 3 and result['n_invalid'] == 0,
    "1.1: All valid SMILES"
)

# Test 1.2: Mixed valid/invalid
df2 = pd.DataFrame({'smiles': ['CCO', 'INVALID', 'c1ccccc1', 'BadSMILES!!!']})
result = _analyze_smiles_validity(df2, 'smiles')
results.assert_test(
    result['n_valid'] == 2 and result['n_invalid'] == 2,
    "1.2: Mixed valid/invalid SMILES"
)

# Test 1.3: All invalid
df3 = pd.DataFrame({'smiles': ['INVALID1', 'BAD', '!!!']})
result = _analyze_smiles_validity(df3, 'smiles')
results.assert_test(
    result['n_valid'] == 0 and result['n_invalid'] == 3,
    "1.3: All invalid SMILES"
)

# Test 1.4: Empty DataFrame
df4 = pd.DataFrame({'smiles': []})
result = _analyze_smiles_validity(df4, 'smiles')
results.assert_test(
    result['n_valid'] == 0 and result['n_total'] == 0,
    "1.4: Empty DataFrame"
)

# Test 1.5: NaN values
df5 = pd.DataFrame({'smiles': ['CCO', np.nan, None, 'c1ccccc1', '']})
result = _analyze_smiles_validity(df5, 'smiles')
results.assert_test(
    result['n_valid'] == 2 and len(result['valid_flags']) == 5,
    "1.5: NaN and None values handled"
)

# Test 1.6: Non-string types
df6 = pd.DataFrame({'smiles': ['CCO', 123, ['list'], {'dict': 'val'}]})
result = _analyze_smiles_validity(df6, 'smiles')
results.assert_test(
    'n_valid' in result and 'n_invalid' in result,
    "1.6: Non-string types handled"
)

# Test 1.7: Return structure
results.assert_test(
    all(key in result for key in ['n_valid', 'n_invalid', 'n_total', 'pct_valid', 'pct_invalid', 'valid_flags']),
    "1.7: Return structure complete"
)

print()

# =============================================================================
# TEST 2: _analyze_activity_distribution
# =============================================================================
print("TEST 2: _analyze_activity_distribution")
print("-" * 80)

# Test 2.1: Continuous activity (regression)
activity1 = np.array([1.5, 2.3, 3.7, 4.2, 5.8, 6.1, 7.9, 8.5, 9.2, 10.3])
result = _analyze_activity_distribution(activity1, activity_type='continuous', units='nM')
results.assert_test(
    result['type'] == 'continuous' and 'linear_stats' in result and 'mean' in result['linear_stats'],
    "2.1: Continuous activity statistics"
)

# Test 2.2: Binary classification
activity2 = np.array([0, 1, 0, 1, 1, 0, 0, 1])
result = _analyze_activity_distribution(activity2, activity_type='classification')
results.assert_test(
    'n_positive' in result and 'n_negative' in result and result['n_positive'] == 4,
    "2.2: Binary classification counts"
)

# Test 2.3: Continuous with units
activity3 = np.array([12.5, 25.3, 48.7, 102.3, 256.8])
result = _analyze_activity_distribution(activity3, activity_type='continuous', units='μM')
results.assert_test(
    result['type'] == 'continuous' and result['units'] == 'μM',
    "2.3: Continuous with custom units"
)

# Test 2.4: Insufficient data
activity4 = np.array([5.0])
result = _analyze_activity_distribution(activity4, activity_type='continuous')
results.assert_test(
    'error' in result,
    "2.4: Insufficient data handled"
)

# Test 2.5: All same values
activity5 = np.array([3.0, 3.0, 3.0, 3.0])
result = _analyze_activity_distribution(activity5, activity_type='continuous')
results.assert_test(
    result['linear_stats']['std'] == 0.0,
    "2.5: All same values"
)

# Test 2.6: With NaN values
activity6 = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
result = _analyze_activity_distribution(activity6, activity_type='continuous')
results.assert_test(
    'n_missing' in result and result['n_missing'] == 2,
    "2.6: NaN values filtered"
)

# Test 2.7: Extreme values
activity7 = np.array([1e-10, 1e10, 0.0001, 100000])
result = _analyze_activity_distribution(activity7, activity_type='continuous')
results.assert_test(
    'log_stats' in result and 'mean' in result['log_stats'],
    "2.7: Extreme values handled"
)

# Test 2.8: Negative values (should still work)
activity8 = np.array([-5, -3, -1, 0, 1, 3, 5])
result = _analyze_activity_distribution(activity8, activity_type='continuous')
results.assert_test(
    'linear_stats' in result and result['linear_stats']['min'] == -5,
    "2.8: Negative values handled"
)

print()

# =============================================================================
# TEST 3: _analyze_functional_groups
# =============================================================================
print("TEST 3: _analyze_functional_groups")
print("-" * 80)

# Test 3.1: Molecules with various functional groups
smiles1 = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'CC(=O)C']
result = _analyze_functional_groups(smiles1)
results.assert_test(
    isinstance(result, dict) and len(result) > 0,
    "3.1: Functional groups detected"
)

# Test 3.2: Empty list
result = _analyze_functional_groups([])
results.assert_test(
    isinstance(result, dict),
    "3.2: Empty list handled"
)

# Test 3.3: Invalid SMILES
smiles3 = ['INVALID', 'BAD_SMILES', '!!!']
result = _analyze_functional_groups(smiles3)
results.assert_test(
    isinstance(result, dict),
    "3.3: Invalid SMILES handled"
)

# Test 3.4: Mixed valid/invalid
smiles4 = ['CCO', 'INVALID', 'c1ccccc1', None, np.nan, '']
result = _analyze_functional_groups(smiles4)
results.assert_test(
    isinstance(result, dict),
    "3.4: Mixed valid/invalid handled"
)

# Test 3.5: Return structure
if result:
    sample_group = list(result.keys())[0]
    results.assert_test(
        all(key in result[sample_group] for key in ['count', 'pct_dataset', 'avg_per_molecule']),
        "3.5: Return structure correct"
    )

# Test 3.6: Single molecule
result = _analyze_functional_groups(['CC(=O)O'])
results.assert_test(
    isinstance(result, dict) and len(result) > 0,
    "3.6: Single molecule processed"
)

# Test 3.7: Non-string types
smiles7 = ['CCO', 123, ['list'], None]
result = _analyze_functional_groups(smiles7)
results.assert_test(
    isinstance(result, dict),
    "3.7: Non-string types handled"
)

print()

# =============================================================================
# TEST 4: _analyze_stereochemistry
# =============================================================================
print("TEST 4: _analyze_stereochemistry")
print("-" * 80)

# Test 4.1: Molecules with stereochemistry
smiles1 = ['C[C@H](O)CC', 'C[C@@H](N)C', 'C/C=C/C']
result = _analyze_stereochemistry(smiles1)
results.assert_test(
    'chiral_centers' in result and 'ez_bonds' in result,
    "4.1: Stereochemistry detected"
)

# Test 4.2: No stereochemistry
smiles2 = ['CCO', 'c1ccccc1', 'CC(C)C']
result = _analyze_stereochemistry(smiles2)
results.assert_test(
    'n_molecules_with_stereo' in result,
    "4.2: No stereochemistry handled"
)

# Test 4.3: Empty list
result = _analyze_stereochemistry([])
results.assert_test(
    isinstance(result, dict) and 'total_molecules' in result,
    "4.3: Empty list handled"
)

# Test 4.4: Invalid SMILES
smiles4 = ['INVALID', 'BAD']
result = _analyze_stereochemistry(smiles4)
results.assert_test(
    isinstance(result, dict),
    "4.4: Invalid SMILES handled"
)

# Test 4.5: Mixed valid/invalid
smiles5 = ['C[C@H](O)CC', 'INVALID', None, 'CCO']
result = _analyze_stereochemistry(smiles5)
results.assert_test(
    'total_molecules' in result,
    "4.5: Mixed valid/invalid handled"
)

# Test 4.6: Unspecified stereo
smiles6 = ['CC(O)CC']  # Chiral but unspecified
result = _analyze_stereochemistry(smiles6)
results.assert_test(
    isinstance(result, dict),
    "4.6: Unspecified stereochemistry"
)

print()

# =============================================================================
# TEST 5: _analyze_charge_state
# =============================================================================
print("TEST 5: _analyze_charge_state")
print("-" * 80)

# Test 5.1: Neutral molecules
smiles1 = ['CCO', 'c1ccccc1', 'CC(=O)O']
result = _analyze_charge_state(smiles1)
results.assert_test(
    'n_charged' in result and 'n_neutral' in result,
    "5.1: Neutral molecules analyzed"
)

# Test 5.2: Charged molecules
smiles2 = ['CC[NH3+]', 'CC(=O)[O-]', '[Na+]', '[Cl-]']
result = _analyze_charge_state(smiles2)
results.assert_test(
    result['n_charged'] > 0,
    "5.2: Charged molecules detected"
)

# Test 5.3: Zwitterions
smiles3 = ['C[NH3+]CC(=O)[O-]']
result = _analyze_charge_state(smiles3)
results.assert_test(
    'n_zwitterions' in result,
    "5.3: Zwitterions detected"
)

# Test 5.4: Empty list
result = _analyze_charge_state([])
results.assert_test(
    isinstance(result, dict) and 'total_molecules' in result,
    "5.4: Empty list handled"
)

# Test 5.5: Invalid SMILES
smiles5 = ['INVALID', 'BAD']
result = _analyze_charge_state(smiles5)
results.assert_test(
    isinstance(result, dict),
    "5.5: Invalid SMILES handled"
)

# Test 5.6: Mixed valid/invalid
smiles6 = ['CCO', 'INVALID', '[Na+]', None, np.nan]
result = _analyze_charge_state(smiles6)
results.assert_test(
    'total_molecules' in result,
    "5.6: Mixed valid/invalid handled"
)

# Test 5.7: Charge distribution
smiles7 = ['[NH4+]', 'CC[N+](C)(C)C', '[O-]C(=O)C']
result = _analyze_charge_state(smiles7)
results.assert_test(
    'charge_distribution' in result,
    "5.7: Charge distribution calculated"
)

print()

# =============================================================================
# TEST 6: _analyze_salts_fragments_solvents
# =============================================================================
print("TEST 6: _analyze_salts_fragments_solvents")
print("-" * 80)

# Test 6.1: Fragmented SMILES
smiles1 = ['CCO.Cl', 'c1ccccc1.[Na+].[Cl-]', 'CC(=O)O.O']
result = _analyze_salts_fragments_solvents(smiles1)
results.assert_test(
    result['fragmented_molecules'] > 0 and result['multi_component_molecules'] > 0,
    "6.1: Fragmented SMILES detected"
)

# Test 6.2: No fragments
smiles2 = ['CCO', 'c1ccccc1', 'CC(=O)O']
result = _analyze_salts_fragments_solvents(smiles2)
results.assert_test(
    result['fragmented_molecules'] == 0,
    "6.2: No fragments detected"
)

# Test 6.3: Common salts
smiles3 = ['CCO.[Na+]', 'c1ccccc1.[Cl-]']
result = _analyze_salts_fragments_solvents(smiles3)
results.assert_test(
    'salt_counts' in result and len(result['salt_counts']) > 0,
    "6.3: Common salts detected"
)

# Test 6.4: Common solvents
smiles4 = ['CCO.O', 'c1ccccc1.C1CCOC1']  # Water and THF
result = _analyze_salts_fragments_solvents(smiles4)
results.assert_test(
    'solvent_counts' in result and len(result['solvent_counts']) > 0,
    "6.4: Common solvents detected"
)

# Test 6.5: Empty list
result = _analyze_salts_fragments_solvents([])
results.assert_test(
    isinstance(result, dict) and 'total_molecules' in result,
    "6.5: Empty list handled"
)

# Test 6.6: Invalid SMILES
smiles6 = ['INVALID.BAD', 'ERROR!!!']
result = _analyze_salts_fragments_solvents(smiles6)
results.assert_test(
    isinstance(result, dict),
    "6.6: Invalid SMILES handled"
)

# Test 6.7: Non-canonical SMILES (canonicalization test)
smiles7 = ['CCO.Cl', 'CCO.[Cl-]']  # Same salt, different representation
result = _analyze_salts_fragments_solvents(smiles7)
results.assert_test(
    result['fragmented_molecules'] == 2,
    "6.7: Non-canonical SMILES handled via canonicalization"
)

# Test 6.8: Mixed valid/invalid
smiles8 = ['CCO.Cl', 'INVALID', None, 'c1ccccc1']
result = _analyze_salts_fragments_solvents(smiles8)
results.assert_test(
    'total_molecules' in result,
    "6.8: Mixed valid/invalid handled"
)

print()

# =============================================================================
# TEST 7: _analyze_special_features
# =============================================================================
print("TEST 7: _analyze_special_features")
print("-" * 80)

# Test 7.1: Organometallic compounds
smiles1 = ['c1ccccc1[Fe]', 'CC[Zn]CC']
result = _analyze_special_features(smiles1)
results.assert_test(
    'organometallic_count' in result and result['organometallic_count'] > 0,
    "7.1: Organometallic compounds detected"
)

# Test 7.2: No special features
smiles2 = ['CCO', 'c1ccccc1', 'CC(=O)O']
result = _analyze_special_features(smiles2)
results.assert_test(
    result['organometallic_count'] == 0 and result['isotope_count'] == 0,
    "7.2: No special features"
)

# Test 7.3: Isotopes
smiles3 = ['[2H]C([2H])([2H])C', '[13C]CO']
result = _analyze_special_features(smiles3)
results.assert_test(
    result['isotope_count'] > 0,
    "7.3: Isotopes detected"
)

# Test 7.4: Ring sizes
smiles4 = ['C1CC1', 'C1CCC1', 'C1CCCC1', 'c1ccccc1']  # 3, 4, 5, 6-membered
result = _analyze_special_features(smiles4)
results.assert_test(
    'ring_sizes' in result and len(result['ring_sizes']) > 0,
    "7.4: Ring sizes analyzed"
)

# Test 7.5: Empty list
result = _analyze_special_features([])
results.assert_test(
    isinstance(result, dict) and 'total_molecules' in result,
    "7.5: Empty list handled"
)

# Test 7.6: Invalid SMILES
smiles6 = ['INVALID', 'BAD']
result = _analyze_special_features(smiles6)
results.assert_test(
    isinstance(result, dict),
    "7.6: Invalid SMILES handled"
)

# Test 7.7: Mixed valid/invalid
smiles7 = ['c1ccccc1[Fe]', 'INVALID', None, 'CCO']
result = _analyze_special_features(smiles7)
results.assert_test(
    'total_molecules' in result,
    "7.7: Mixed valid/invalid handled"
)

# Test 7.8: Large rings
smiles8 = ['C1CCCCCCC1', 'C1CCCCCCCC1']  # 8, 9-membered
result = _analyze_special_features(smiles8)
results.assert_test(
    'molecules_with_rings' in result,
    "7.8: Large rings handled"
)

print()

# =============================================================================
# TEST 8: _analyze_activity_correlations
# =============================================================================
print("TEST 8: _analyze_activity_correlations")
print("-" * 80)

# Test 8.1: Regression correlation
df1 = pd.DataFrame({
    'smiles': ['CCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO'],
    'activity': [1.0, 2.0, 3.0, 4.0, 5.0]
})
result = _analyze_activity_correlations(df1, 'smiles', 'activity', 'regression')
results.assert_test(
    'analysis_type' in result and result['analysis_type'] == 'regression',
    "8.1: Regression correlation"
)

# Test 8.2: Classification enrichment
df2 = pd.DataFrame({
    'smiles': ['CCO', 'CCCO', 'c1ccccc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1'],
    'activity': [0, 0, 1, 1, 1]
})
result = _analyze_activity_correlations(df2, 'smiles', 'activity', 'classification')
results.assert_test(
    'task_type' in result and result['task_type'] == 'classification',
    "8.2: Classification enrichment"
)

# Test 8.3: Insufficient samples
df3 = pd.DataFrame({
    'smiles': ['CCO', 'CCCO'],
    'activity': [1.0, 2.0]
})
result = _analyze_activity_correlations(df3, 'smiles', 'activity', 'regression')
results.assert_test(
    isinstance(result, dict),
    "8.3: Insufficient samples handled"
)

# Test 8.4: All invalid SMILES
df4 = pd.DataFrame({
    'smiles': ['INVALID', 'BAD', 'ERROR'],
    'activity': [1, 2, 3]
})
result = _analyze_activity_correlations(df4, 'smiles', 'activity', 'regression')
results.assert_test(
    isinstance(result, dict),
    "8.4: All invalid SMILES handled"
)

# Test 8.5: Mixed valid/invalid
df5 = pd.DataFrame({
    'smiles': ['CCO', 'INVALID', 'c1ccccc1', None, 'CCCO'],
    'activity': [1.0, 2.0, 3.0, 4.0, 5.0]
})
result = _analyze_activity_correlations(df5, 'smiles', 'activity', 'regression')
results.assert_test(
    isinstance(result, dict) and 'n_samples' in result,
    "8.5: Mixed valid/invalid handled"
)

# Test 8.6: NaN activities
df6 = pd.DataFrame({
    'smiles': ['CCO', 'CCCO', 'c1ccccc1'],
    'activity': [1.0, np.nan, 3.0]
})
result = _analyze_activity_correlations(df6, 'smiles', 'activity', 'regression')
results.assert_test(
    isinstance(result, dict) and 'n_samples' in result,
    "8.6: NaN activities handled"
)

# Test 8.7: Single class (classification)
df7 = pd.DataFrame({
    'smiles': ['CCO', 'CCCO', 'c1ccccc1'],
    'activity': [1, 1, 1]
})
result = _analyze_activity_correlations(df7, 'smiles', 'activity', 'classification')
results.assert_test(
    isinstance(result, dict),
    "8.7: Single class handled"
)

# Test 8.8: No variance in activity (regression)
df8 = pd.DataFrame({
    'smiles': ['CCO', 'CCCO', 'c1ccccc1'],
    'activity': [5.0, 5.0, 5.0]
})
result = _analyze_activity_correlations(df8, 'smiles', 'activity', 'regression')
results.assert_test(
    isinstance(result, dict),
    "8.8: No variance in activity handled"
)

print()

# =============================================================================
# STRESS TESTS - Large datasets and extreme conditions
# =============================================================================
print("STRESS TESTS")
print("-" * 80)

# Stress 1: Large dataset
large_smiles = ['CCO'] * 1000 + ['c1ccccc1'] * 1000 + ['CC(=O)O'] * 1000
result = _analyze_functional_groups(large_smiles)
results.assert_test(
    isinstance(result, dict) and len(result) > 0,
    "S1: Large dataset (3000 molecules)"
)

# Stress 2: All NaN
df_nan = pd.DataFrame({'smiles': [np.nan] * 100})
result = _analyze_smiles_validity(df_nan, 'smiles')
results.assert_test(
    result['n_total'] == 100 and result['n_valid'] == 0,
    "S2: All NaN values"
)

# Stress 3: Very long SMILES
long_smiles = ['C' * 1000, 'c1ccccc1' * 100]
result = _analyze_functional_groups(long_smiles)
results.assert_test(
    isinstance(result, dict),
    "S3: Very long SMILES strings"
)

# Stress 4: Unicode and special characters
special_smiles = ['CCO', 'c1ccccc1', '🔬', '你好', '\n\t']
result = _analyze_smiles_validity(pd.DataFrame({'smiles': special_smiles}), 'smiles')
results.assert_test(
    isinstance(result, dict) and 'n_valid' in result,
    "S4: Unicode and special characters"
)

# Stress 5: Extreme activity values
extreme_activity = np.array([1e-20, 1e20, -1e20, np.inf, -np.inf, 0])
try:
    result = _analyze_activity_distribution(extreme_activity[:-2], activity_type='continuous')
    results.assert_test(
        isinstance(result, dict) and 'mean' in result,
        "S5: Extreme activity values"
    )
except:
    results.assert_test(False, "S5: Extreme activity values", "Exception raised")

# Stress 6: Complex fragmented SMILES
complex_fragments = [
    'CCO.Cl.[Na+].[OH-].O.C1CCOC1',
    'c1ccccc1' + '.O' * 50,
    'C.' * 100
]
result = _analyze_salts_fragments_solvents(complex_fragments)
results.assert_test(
    isinstance(result, dict) and 'fragmented_molecules' in result,
    "S6: Complex fragmented SMILES"
)

# Stress 7: Mixed data types in activity
mixed_activity = np.array([1, 2.5, '3', None, np.nan, True, False])
try:
    numeric_activity = pd.to_numeric(mixed_activity, errors='coerce')
    result = _analyze_activity_distribution(numeric_activity, activity_type='continuous')
    results.assert_test(
        isinstance(result, dict),
        "S7: Mixed data types in activity"
    )
except:
    results.assert_test(False, "S7: Mixed data types in activity", "Exception raised")

print()

# =============================================================================
# EDGE CASE TESTS
# =============================================================================
print("EDGE CASE TESTS")
print("-" * 80)

# Edge 1: Empty string SMILES
df_empty = pd.DataFrame({'smiles': ['', '  ', '\t', '\n']})
result = _analyze_smiles_validity(df_empty, 'smiles')
results.assert_test(
    result['n_valid'] == 0,
    "E1: Empty string SMILES"
)

# Edge 2: Whitespace in SMILES
df_whitespace = pd.DataFrame({'smiles': [' CCO ', '  c1ccccc1  ', '\tCC(=O)O\n']})
result = _analyze_smiles_validity(df_whitespace, 'smiles')
results.assert_test(
    isinstance(result, dict) and 'n_valid' in result,
    "E2: Whitespace in SMILES"
)

# Edge 3: Case sensitivity
df_case = pd.DataFrame({'smiles': ['cco', 'CCO', 'C1CCCCC1', 'c1ccccc1']})
result = _analyze_smiles_validity(df_case, 'smiles')
results.assert_test(
    isinstance(result, dict) and result['n_total'] == 4,
    "E3: Case sensitivity in SMILES"
)

# Edge 4: Activity with single outlier
activity_outlier = np.array([1.0, 1.1, 1.2, 1.1, 1.0, 1000.0])
result = _analyze_activity_distribution(activity_outlier, activity_type='continuous')
results.assert_test(
    result['std'] > 100 and result['cv'] > 1.0,
    "E4: Activity with single outlier"
)

# Edge 5: Identical SMILES (canonicalization test)
identical_smiles = ['c1ccccc1', 'C1=CC=CC=C1', 'c1ccc(cc1)']
result = _analyze_salts_fragments_solvents(identical_smiles)
results.assert_test(
    isinstance(result, dict),
    "E5: Identical structures, different SMILES"
)

# Edge 6: Classification with imbalanced classes
df_imbalanced = pd.DataFrame({
    'smiles': ['CCO'] * 95 + ['c1ccccc1'] * 5,
    'activity': [0] * 95 + [1] * 5
})
result = _analyze_activity_correlations(df_imbalanced, 'smiles', 'activity', 'classification')
results.assert_test(
    isinstance(result, dict) and result.get('n_active', 0) == 5,
    "E6: Highly imbalanced classification"
)

# Edge 7: Single unique SMILES repeated
single_repeated = ['CCO'] * 100
result = _analyze_functional_groups(single_repeated)
results.assert_test(
    isinstance(result, dict) and len(result) > 0,
    "E7: Single unique SMILES repeated"
)

# Edge 8: Borderline valid SMILES
borderline = ['[H]', '[He]', '[C]', 'C', '[]', '[1H]']
result = _analyze_smiles_validity(pd.DataFrame({'smiles': borderline}), 'smiles')
results.assert_test(
    isinstance(result, dict) and 'n_valid' in result,
    "E8: Borderline valid SMILES"
)

print()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
results.print_summary()

if results.failed == 0:
    print("\n🎉 ALL TESTS PASSED! Quality report functions are robust.")
    sys.exit(0)
else:
    print(f"\n⚠️  {results.failed} tests failed. Review failures above.")
    sys.exit(1)
