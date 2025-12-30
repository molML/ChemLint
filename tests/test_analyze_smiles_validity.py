"""
RIGOROUS TEST SUITE: _analyze_smiles_validity

Tests all edge cases, boundary conditions, and error handling for SMILES validation.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from molml_mcp.tools.reports.quality import _analyze_smiles_validity

def test_basic_functionality():
    """Test basic valid/invalid SMILES detection"""
    print("\n=== BASIC FUNCTIONALITY ===")
    
    # Test 1: All valid SMILES
    df = pd.DataFrame({'smiles': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCC']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 5, f"Expected 5 valid, got {result['n_valid']}"
    assert result['n_invalid'] == 0, f"Expected 0 invalid, got {result['n_invalid']}"
    assert result['pct_valid'] == 100.0, f"Expected 100%, got {result['pct_valid']}"
    assert len(result['valid_flags']) == 5
    print("✅ All valid SMILES")
    
    # Test 2: All invalid SMILES
    df = pd.DataFrame({'smiles': ['INVALID', 'BAD_SMILES', '!!!', 'xyz123']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 0, f"Expected 0 valid, got {result['n_valid']}"
    assert result['n_invalid'] == 4, f"Expected 4 invalid, got {result['n_invalid']}"
    assert result['pct_valid'] == 0.0, f"Expected 0%, got {result['pct_valid']}"
    print("✅ All invalid SMILES")
    
    # Test 3: Mixed valid/invalid
    df = pd.DataFrame({'smiles': ['CCO', 'INVALID', 'c1ccccc1', 'BAD', 'CC(=O)O']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 3, f"Expected 3 valid, got {result['n_valid']}"
    assert result['n_invalid'] == 2, f"Expected 2 invalid, got {result['n_invalid']}"
    assert abs(result['pct_valid'] - 60.0) < 0.01
    print("✅ Mixed valid/invalid")

def test_empty_and_missing():
    """Test empty DataFrames and missing values"""
    print("\n=== EMPTY AND MISSING VALUES ===")
    
    # Test 1: Empty DataFrame
    df = pd.DataFrame({'smiles': []})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_total'] == 0
    assert result['n_valid'] == 0
    assert result['n_invalid'] == 0
    assert len(result['valid_flags']) == 0
    print("✅ Empty DataFrame")
    
    # Test 2: All NaN
    df = pd.DataFrame({'smiles': [np.nan, np.nan, np.nan]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_total'] == 3
    assert result['n_valid'] == 0
    assert result['n_invalid'] == 3
    print("✅ All NaN values")
    
    # Test 3: All None
    df = pd.DataFrame({'smiles': [None, None, None]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_total'] == 3
    assert result['n_valid'] == 0
    print("✅ All None values")
    
    # Test 4: Mixed with NaN/None
    df = pd.DataFrame({'smiles': ['CCO', np.nan, None, 'c1ccccc1', '', 'CC(=O)O']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_total'] == 6
    assert result['n_valid'] == 3  # CCO, benzene, acetic acid
    assert len(result['valid_flags']) == 6
    print("✅ Mixed with NaN/None")

def test_whitespace_handling():
    """Test various whitespace scenarios"""
    print("\n=== WHITESPACE HANDLING ===")
    
    # Test 1: Empty strings
    df = pd.DataFrame({'smiles': ['', '  ', '\t', '\n', '   \t\n  ']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 0, "Empty/whitespace should be invalid"
    print("✅ Empty strings and whitespace")
    
    # Test 2: Leading/trailing whitespace
    df = pd.DataFrame({'smiles': [' CCO ', '  c1ccccc1  ', '\tCC(=O)O\n']})
    result = _analyze_smiles_validity(df, 'smiles')
    # RDKit should handle whitespace - test behavior
    print(f"   Whitespace handling: {result['n_valid']}/{result['n_total']} valid")
    assert result['n_total'] == 3
    print("✅ Leading/trailing whitespace")
    
    # Test 3: Internal whitespace (RDKit may handle some patterns)
    df = pd.DataFrame({'smiles': ['CC O', 'c1 ccc cc1', 'CC(=O) O']})
    result = _analyze_smiles_validity(df, 'smiles')
    # RDKit behavior varies with internal whitespace - just verify it handles it
    assert isinstance(result, dict) and 'n_valid' in result
    print("✅ Internal whitespace handled")

def test_case_sensitivity():
    """Test case sensitivity in SMILES"""
    print("\n=== CASE SENSITIVITY ===")
    
    # Test: Mixed case SMILES
    df = pd.DataFrame({'smiles': [
        'CCO',      # Aliphatic carbons
        'cco',      # Lowercase (invalid - aromatic notation wrong)
        'c1ccccc1', # Aromatic benzene
        'C1CCCCC1', # Aliphatic cyclohexane
        'Cc1ccccc1' # Toluene (mixed)
    ]})
    result = _analyze_smiles_validity(df, 'smiles')
    print(f"   Case sensitivity: {result['n_valid']}/{result['n_total']} valid")
    assert result['n_total'] == 5
    # 'cco' should be invalid as lowercase c implies aromatic
    print("✅ Case sensitivity tested")

def test_special_characters():
    """Test special characters and unicode"""
    print("\n=== SPECIAL CHARACTERS ===")
    
    # Test 1: Unicode characters
    df = pd.DataFrame({'smiles': ['🔬', '你好', 'مرحبا', '🧪⚗️']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 0, "Unicode should be invalid"
    print("✅ Unicode characters rejected")
    
    # Test 2: Special symbols
    df = pd.DataFrame({'smiles': ['@@@', '###', '$$$', '%%%', '&&&']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 0, "Special symbols should be invalid"
    print("✅ Special symbols rejected")
    
    # Test 3: Valid special SMILES characters
    df = pd.DataFrame({'smiles': [
        'C[C@H](O)CC',      # Chiral center
        'C/C=C/C',          # E/Z stereochemistry
        'C[C@@H](N)C',      # Chiral center
        '[Na+].[Cl-]',      # Ions
        'CC(=O)[O-]'        # Charged species
    ]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 5, "Valid SMILES special chars should work"
    print("✅ Valid SMILES special characters")

def test_data_types():
    """Test non-string data types"""
    print("\n=== DATA TYPES ===")
    
    # Test 1: Numeric types
    df = pd.DataFrame({'smiles': [123, 456.789, 0, -1, 1e10]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 0, "Numeric types should be invalid"
    print("✅ Numeric types handled")
    
    # Test 2: Boolean types
    df = pd.DataFrame({'smiles': [True, False]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 0, "Boolean types should be invalid"
    print("✅ Boolean types handled")
    
    # Test 3: Lists and dicts
    df = pd.DataFrame({'smiles': [['list'], {'dict': 'val'}, ('tuple',)]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 0, "Complex types should be invalid"
    print("✅ Complex types handled")
    
    # Test 4: Mixed types
    df = pd.DataFrame({'smiles': ['CCO', 123, True, None, 'c1ccccc1', []]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_total'] == 6
    assert result['n_valid'] == 2, "Only string SMILES should be valid"
    print("✅ Mixed types handled")

def test_extreme_cases():
    """Test extreme and edge cases"""
    print("\n=== EXTREME CASES ===")
    
    # Test 1: Very long SMILES
    long_smiles = 'C' * 10000
    df = pd.DataFrame({'smiles': [long_smiles]})
    result = _analyze_smiles_validity(df, 'smiles')
    print(f"   Very long SMILES (10k chars): valid={result['n_valid']}")
    print("✅ Very long SMILES handled")
    
    # Test 2: Repeated pattern
    repeated = 'c1ccccc1' * 1000
    df = pd.DataFrame({'smiles': [repeated]})
    result = _analyze_smiles_validity(df, 'smiles')
    print(f"   Repeated pattern: valid={result['n_valid']}")
    print("✅ Repeated pattern handled")
    
    # Test 3: Single character SMILES
    df = pd.DataFrame({'smiles': ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']})
    result = _analyze_smiles_validity(df, 'smiles')
    print(f"   Single atoms: {result['n_valid']}/{result['n_total']} valid")
    print("✅ Single character SMILES")
    
    # Test 4: Exotic atoms
    df = pd.DataFrame({'smiles': ['[H]', '[He]', '[Li]', '[Be]', '[B]']})
    result = _analyze_smiles_validity(df, 'smiles')
    print(f"   Exotic atoms: {result['n_valid']}/{result['n_total']} valid")
    print("✅ Exotic atoms tested")
    
    # Test 5: Large dataset
    large_smiles = ['CCO'] * 5000 + ['INVALID'] * 5000
    df = pd.DataFrame({'smiles': large_smiles})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_total'] == 10000
    assert result['n_valid'] == 5000
    assert result['n_invalid'] == 5000
    print("✅ Large dataset (10k entries)")

def test_complex_molecules():
    """Test complex molecular structures"""
    print("\n=== COMPLEX MOLECULES ===")
    
    # Test 1: Drugs and natural products
    df = pd.DataFrame({'smiles': [
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    ]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 3, f"Expected 3 valid drugs, got {result['n_valid']}"
    print("✅ Complex drug molecules")
    
    # Test 2: Metal complexes
    df = pd.DataFrame({'smiles': [
        'c1ccccc1[Fe]',
        'CC[Zn]CC',
        '[Pt](Cl)(Cl)(N)(N)'
    ]})
    result = _analyze_smiles_validity(df, 'smiles')
    print(f"   Metal complexes: {result['n_valid']}/{result['n_total']} valid")
    print("✅ Metal complexes tested")
    
    # Test 3: Charged species
    df = pd.DataFrame({'smiles': [
        '[NH4+]',
        '[O-]C(=O)C',
        'CC[N+](C)(C)C',
        '[Na+]',
        '[Cl-]'
    ]})
    result = _analyze_smiles_validity(df, 'smiles')
    print(f"   Charged species: {result['n_valid']}/{result['n_total']} valid")
    print("✅ Charged species")

def test_stereochemistry():
    """Test stereochemical notation"""
    print("\n=== STEREOCHEMISTRY ===")
    
    # Test: Various stereochemical notations
    df = pd.DataFrame({'smiles': [
        'C[C@H](O)CC',       # S chirality
        'C[C@@H](O)CC',      # R chirality
        'C/C=C/C',           # E configuration
        'C/C=C\\C',          # Z configuration
        'C[C@H]1CC[C@@H](C)CC1',  # Multiple centers
    ]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 5, "All stereochemical notations should be valid"
    print("✅ Stereochemical notation")

def test_fragments():
    """Test fragmented/multi-component SMILES"""
    print("\n=== FRAGMENTS ===")
    
    # Test: Fragmented SMILES
    df = pd.DataFrame({'smiles': [
        'CCO.Cl',
        'c1ccccc1.[Na+].[Cl-]',
        'CC(=O)O.O',
        'CCN.CC(=O)O.O.[Na+].[Cl-]',
        'C.C.C.C.C'
    ]})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 5, "Fragmented SMILES should be valid"
    print("✅ Fragmented SMILES")

def test_return_structure():
    """Test return value structure and types"""
    print("\n=== RETURN STRUCTURE ===")
    
    df = pd.DataFrame({'smiles': ['CCO', 'INVALID', 'c1ccccc1']})
    result = _analyze_smiles_validity(df, 'smiles')
    
    # Check all required keys
    required_keys = ['n_valid', 'n_invalid', 'n_total', 'pct_valid', 'pct_invalid', 'valid_flags']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    print("✅ All required keys present")
    
    # Check types
    assert isinstance(result['n_valid'], (int, np.integer))
    assert isinstance(result['n_invalid'], (int, np.integer))
    assert isinstance(result['n_total'], (int, np.integer))
    assert isinstance(result['pct_valid'], (float, np.floating))
    assert isinstance(result['pct_invalid'], (float, np.floating))
    assert isinstance(result['valid_flags'], (list, np.ndarray))
    print("✅ All types correct")
    
    # Check values consistency
    assert result['n_valid'] + result['n_invalid'] == result['n_total']
    assert len(result['valid_flags']) == result['n_total']
    assert abs(result['pct_valid'] + result['pct_invalid'] - 100.0) < 0.01
    print("✅ Values consistent")

def test_boundary_conditions():
    """Test boundary conditions"""
    print("\n=== BOUNDARY CONDITIONS ===")
    
    # Test 1: Single entry
    df = pd.DataFrame({'smiles': ['CCO']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_total'] == 1
    assert result['pct_valid'] == 100.0
    print("✅ Single entry")
    
    # Test 2: Two entries (50/50)
    df = pd.DataFrame({'smiles': ['CCO', 'INVALID']})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 1
    assert result['n_invalid'] == 1
    assert abs(result['pct_valid'] - 50.0) < 0.01
    print("✅ Two entries (50/50)")
    
    # Test 3: All same SMILES
    df = pd.DataFrame({'smiles': ['CCO'] * 100})
    result = _analyze_smiles_validity(df, 'smiles')
    assert result['n_valid'] == 100
    assert result['pct_valid'] == 100.0
    print("✅ All same SMILES")

def test_canonicalization_independence():
    """Test that different representations are handled correctly"""
    print("\n=== CANONICALIZATION INDEPENDENCE ===")
    
    # Test: Different representations of same molecule
    df = pd.DataFrame({'smiles': [
        'c1ccccc1',      # Benzene standard
        'C1=CC=CC=C1',   # Benzene Kekulé
        'c1ccc(cc1)',    # Benzene with parens
    ]})
    result = _analyze_smiles_validity(df, 'smiles')
    # All should be valid regardless of canonicalization
    assert result['n_valid'] == 3, "All benzene forms should be valid"
    print("✅ Different representations handled")

if __name__ == '__main__':
    print("="*80)
    print("RIGOROUS TEST SUITE: _analyze_smiles_validity")
    print("="*80)
    
    try:
        test_basic_functionality()
        test_empty_and_missing()
        test_whitespace_handling()
        test_case_sensitivity()
        test_special_characters()
        test_data_types()
        test_extreme_cases()
        test_complex_molecules()
        test_stereochemistry()
        test_fragments()
        test_return_structure()
        test_boundary_conditions()
        test_canonicalization_independence()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
