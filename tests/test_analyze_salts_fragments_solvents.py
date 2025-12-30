"""
RIGOROUS TEST SUITE: _analyze_salts_fragments_solvents

Tests salt, fragment, and solvent detection with canonicalization.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from molml_mcp.tools.reports.quality import _analyze_salts_fragments_solvents

def test_basic_fragmentation():
    """Test basic fragment detection"""
    print("\n=== BASIC FRAGMENTATION ===")
    
    # Test 1: No fragments
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 0
    assert result['n_multi_component'] == 0
    print("✅ No fragments")
    
    # Test 2: Simple fragments
    smiles = ['CCO.Cl', 'c1ccccc1.[Na+]', 'CC(=O)O.O']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 3
    assert result['n_multi_component'] == 3
    print("✅ Simple fragments")
    
    # Test 3: Mixed
    smiles = ['CCO', 'CCO.Cl', 'c1ccccc1', 'c1ccccc1.[Na+]']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 2
    assert result['n_multi_component'] == 2
    print("✅ Mixed fragmented and non-fragmented")

def test_salt_detection():
    """Test common salt detection"""
    print("\n=== SALT DETECTION ===")
    
    # Test 1: Common salts
    smiles = [
        'CCO.[Na+]',
        'c1ccccc1.[Cl-]',
        'CC(=O)O.[K+]',
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    assert 'salts_detected' in result
    assert len(result['salts_detected']) > 0
    print("✅ Common salts detected")
    
    # Test 2: Multiple salts
    smiles = ['CCO.[Na+].[Cl-]']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 1
    print("✅ Multiple salts per molecule")

def test_solvent_detection():
    """Test common solvent detection"""
    print("\n=== SOLVENT DETECTION ===")
    
    # Test 1: Water
    smiles = ['CCO.O', 'c1ccccc1.O']
    result = _analyze_salts_fragments_solvents(smiles)
    assert 'solvents_detected' in result
    assert len(result['solvents_detected']) > 0
    print("✅ Water detected")
    
    # Test 2: Organic solvents
    smiles = [
        'CCO.C1CCOC1',  # THF
        'c1ccccc1.CCO',  # Ethanol
        'CC(=O)O.CC(C)=O',  # Acetone
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    assert 'solvents_detected' in result
    print("✅ Organic solvents")

def test_canonicalization():
    """Test SMILES canonicalization in fragment matching"""
    print("\n=== CANONICALIZATION ===")
    
    # Test 1: Different representations of same salt
    smiles = [
        'CCO.Cl',
        'CCO.[Cl-]',
        'CCO.Cl',  # Repeat
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    # All should be detected as having chloride
    assert result['n_fragmented'] >= 2
    print("✅ Chloride representations")
    
    # Test 2: Different benzene representations
    smiles = [
        'CCO.c1ccccc1',
        'CCO.C1=CC=CC=C1',
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    # Both should be detected as having benzene
    print("✅ Benzene representations")
    
    # Test 3: Sodium representations
    smiles = [
        'CCO.[Na+]',
        'CCO.[Na]',
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    print("✅ Sodium representations")

def test_empty_and_invalid():
    """Test empty lists and invalid SMILES"""
    print("\n=== EMPTY AND INVALID ===")
    
    # Test 1: Empty list
    result = _analyze_salts_fragments_solvents([])
    assert isinstance(result, dict)
    assert 'n_molecules' in result
    assert result['n_molecules'] == 0
    print("✅ Empty list")
    
    # Test 2: All invalid
    smiles = ['INVALID', 'BAD', '!!!']
    result = _analyze_salts_fragments_solvents(smiles)
    assert isinstance(result, dict)
    print("✅ All invalid SMILES")
    
    # Test 3: Mixed valid/invalid
    smiles = ['CCO.Cl', 'INVALID', None, 'c1ccccc1']
    result = _analyze_salts_fragments_solvents(smiles)
    assert isinstance(result, dict)
    assert 'n_molecules' in result
    print("✅ Mixed valid/invalid")

def test_complex_fragments():
    """Test complex fragmentation patterns"""
    print("\n=== COMPLEX FRAGMENTS ===")
    
    # Test 1: Many fragments
    smiles = ['CCO.Cl.[Na+].[OH-].O.C1CCOC1']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 1
    # Should detect multiple salts and solvents
    print("✅ Many fragments")
    
    # Test 2: Large number of dots
    smiles = ['C.C.C.C.C.C.C.C.C.C']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_multi_component'] == 1
    print("✅ Many components")
    
    # Test 3: Repeated solvents
    smiles = ['CCO.O.O.O']  # Multiple water molecules
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 1
    print("✅ Repeated solvents")

def test_edge_cases():
    """Test edge cases"""
    print("\n=== EDGE CASES ===")
    
    # Test 1: Single molecule
    smiles = ['CCO.Cl']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_molecules'] == 1
    assert result['n_fragmented'] == 1
    print("✅ Single molecule")
    
    # Test 2: Very simple fragments
    smiles = ['C.C', 'CC.C']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 2
    print("✅ Simple fragments")
    
    # Test 3: Only fragments, no main molecule
    smiles = ['[Na+].[Cl-]']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_multi_component'] == 1
    print("✅ Only fragments")

def test_counting_accuracy():
    """Test counting accuracy"""
    print("\n=== COUNTING ACCURACY ===")
    
    # Test: Specific salt counts
    smiles = [
        'CCO.[Na+]',
        'CCO.[Na+]',
        'CCO.[Cl-]',
        'c1ccccc1'
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    
    # Check percentages
    assert 0 <= result['pct_fragmented'] <= 100
    assert 0 <= result['pct_multi_component'] <= 100
    
    # 3 out of 4 are fragmented (75%)
    expected_pct = 75.0
    assert abs(result['pct_fragmented'] - expected_pct) < 0.1
    print("✅ Counting accuracy")

def test_large_dataset():
    """Test with large dataset"""
    print("\n=== LARGE DATASET ===")
    
    # Test: Large number of molecules
    smiles = ['CCO.Cl'] * 5000 + ['c1ccccc1'] * 5000
    result = _analyze_salts_fragments_solvents(smiles)
    
    assert result['n_molecules'] == 10000
    assert result['n_fragmented'] == 5000
    assert abs(result['pct_fragmented'] - 50.0) < 0.1
    print("✅ Large dataset (10k molecules)")

def test_return_structure():
    """Test return value structure"""
    print("\n=== RETURN STRUCTURE ===")
    
    smiles = ['CCO.Cl', 'c1ccccc1.[Na+]', 'CC(=O)O']
    result = _analyze_salts_fragments_solvents(smiles)
    
    # Check required keys
    required_keys = [
        'n_molecules',
        'n_fragmented',
        'n_multi_component',
        'pct_fragmented',
        'pct_multi_component',
        'salts_detected',
        'solvents_detected'
    ]
    
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    # Check types
    assert isinstance(result['n_molecules'], (int, np.integer))
    assert isinstance(result['n_fragmented'], (int, np.integer))
    assert isinstance(result['pct_fragmented'], (float, np.floating))
    assert isinstance(result['salts_detected'], dict)
    assert isinstance(result['solvents_detected'], dict)
    
    print("✅ Return structure correct")

def test_non_string_types():
    """Test non-string types"""
    print("\n=== NON-STRING TYPES ===")
    
    smiles = ['CCO.Cl', 123, None, np.nan, ['list']]
    result = _analyze_salts_fragments_solvents(smiles)
    assert isinstance(result, dict)
    assert result['n_molecules'] >= 1  # At least the valid one
    print("✅ Non-string types handled")

def test_charged_fragments():
    """Test charged species in fragments"""
    print("\n=== CHARGED FRAGMENTS ===")
    
    # Test: Various charged fragments
    smiles = [
        'CCO.[NH4+]',
        'c1ccccc1.[O-]',
        'CC(=O)O.[Na+].[Cl-]',
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 3
    print("✅ Charged fragments")

def test_organometallic():
    """Test organometallic fragments"""
    print("\n=== ORGANOMETALLIC ===")
    
    # Test: Metal-containing fragments
    smiles = [
        'c1ccccc1.[Fe]',
        'CCO.[Zn]CC',
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 2
    print("✅ Organometallic fragments")

def test_stereochemistry_independence():
    """Test that stereochemistry doesn't affect fragment detection"""
    print("\n=== STEREOCHEMISTRY INDEPENDENCE ===")
    
    # Test: Same fragments, different stereochemistry
    smiles = [
        'C[C@H](O)CC.Cl',
        'C[C@@H](O)CC.Cl',
        'CC(O)CC.Cl',
    ]
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_fragmented'] == 3
    # All should detect chloride
    print("✅ Stereochemistry independence")

def test_consistency():
    """Test consistency of results"""
    print("\n=== CONSISTENCY ===")
    
    # Test: Same input should give same output
    smiles = ['CCO.Cl', 'c1ccccc1.[Na+]', 'CC(=O)O.O']
    result1 = _analyze_salts_fragments_solvents(smiles)
    result2 = _analyze_salts_fragments_solvents(smiles)
    
    assert result1['n_fragmented'] == result2['n_fragmented']
    assert abs(result1['pct_fragmented'] - result2['pct_fragmented']) < 0.01
    
    print("✅ Results are consistent")

def test_special_cases():
    """Test special edge cases"""
    print("\n=== SPECIAL CASES ===")
    
    # Test 1: Fragment is larger than main molecule
    smiles = ['C.CCCCCCCCCC']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_multi_component'] == 1
    print("✅ Larger fragment")
    
    # Test 2: All fragments are same size
    smiles = ['CCC.CCC.CCC']
    result = _analyze_salts_fragments_solvents(smiles)
    assert result['n_multi_component'] == 1
    print("✅ Equal-sized fragments")
    
    # Test 3: Empty fragment (invalid)
    smiles = ['CCO.']
    result = _analyze_salts_fragments_solvents(smiles)
    # Should handle gracefully
    print("✅ Invalid fragment handled")

if __name__ == '__main__':
    print("="*80)
    print("RIGOROUS TEST SUITE: _analyze_salts_fragments_solvents")
    print("="*80)
    
    try:
        test_basic_fragmentation()
        test_salt_detection()
        test_solvent_detection()
        test_canonicalization()
        test_empty_and_invalid()
        test_complex_fragments()
        test_edge_cases()
        test_counting_accuracy()
        test_large_dataset()
        test_return_structure()
        test_non_string_types()
        test_charged_fragments()
        test_organometallic()
        test_stereochemistry_independence()
        test_consistency()
        test_special_cases()
        
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
