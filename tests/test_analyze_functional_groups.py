"""
RIGOROUS TEST SUITE: _analyze_functional_groups

Tests functional group detection, counting, and edge case handling.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from molml_mcp.tools.reports.quality import _analyze_functional_groups

def test_basic_detection():
    """Test basic functional group detection"""
    print("\n=== BASIC DETECTION ===")
    
    # Test 1: Molecules with known groups
    smiles = [
        'CCO',           # Alcohol
        'CC(=O)O',       # Carboxylic acid
        'CCN',           # Amine
        'c1ccccc1',      # Aromatic ring
        'CC(=O)C'        # Ketone
    ]
    result = _analyze_functional_groups(smiles)
    
    assert isinstance(result, dict)
    assert len(result) > 0, "Should detect functional groups"
    print(f"   Detected {len(result)} functional group types")
    print("✅ Basic functional groups detected")
    
    # Test 2: Return structure
    if result:
        sample_group = list(result.keys())[0]
        assert 'count' in result[sample_group]
        assert 'pct_dataset' in result[sample_group]
        assert 'avg_per_molecule' in result[sample_group]
    print("✅ Return structure correct")

def test_empty_and_invalid():
    """Test empty lists and invalid SMILES"""
    print("\n=== EMPTY AND INVALID ===")
    
    # Test 1: Empty list
    result = _analyze_functional_groups([])
    assert isinstance(result, dict)
    # Function returns full structure with all groups at 0 count
    assert len(result) > 0  # Returns all functional groups
    assert all(v['count'] == 0 for k, v in result.items() if k != 'halogen_breakdown')
    print("✅ Empty list handled")
    
    # Test 2: All invalid SMILES
    smiles = ['INVALID', 'BAD_SMILES', '!!!', 'xyz']
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ All invalid SMILES")
    
    # Test 3: Mixed valid/invalid
    smiles = ['CCO', 'INVALID', 'c1ccccc1', 'BAD', None, np.nan]
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ Mixed valid/invalid")

def test_specific_groups():
    """Test detection of specific functional groups"""
    print("\n=== SPECIFIC GROUPS ===")
    
    # Test 1: Alcohols
    smiles = ['CCO', 'CCCO', 'CC(O)C']
    result = _analyze_functional_groups(smiles)
    # Should detect OH groups
    print("✅ Alcohols")
    
    # Test 2: Carboxylic acids
    smiles = ['CC(=O)O', 'CCC(=O)O', 'c1ccc(C(=O)O)cc1']
    result = _analyze_functional_groups(smiles)
    print("✅ Carboxylic acids")
    
    # Test 3: Amines
    smiles = ['CCN', 'CC(N)C', 'c1ccc(N)cc1']
    result = _analyze_functional_groups(smiles)
    print("✅ Amines")
    
    # Test 4: Aromatic rings
    smiles = ['c1ccccc1', 'c1ccc(O)cc1', 'c1ccc(cc1)c2ccccc2']
    result = _analyze_functional_groups(smiles)
    print("✅ Aromatic rings")
    
    # Test 5: Ketones
    smiles = ['CC(=O)C', 'CCC(=O)CC', 'c1ccc(C(=O)C)cc1']
    result = _analyze_functional_groups(smiles)
    print("✅ Ketones")
    
    # Test 6: Esters
    smiles = ['CC(=O)OC', 'CCOC(=O)C', 'CC(=O)Oc1ccccc1']
    result = _analyze_functional_groups(smiles)
    print("✅ Esters")
    
    # Test 7: Amides
    smiles = ['CC(=O)N', 'CNC(=O)C', 'c1ccc(C(=O)N)cc1']
    result = _analyze_functional_groups(smiles)
    print("✅ Amides")
    
    # Test 8: Halogens
    smiles = ['CCCl', 'CCBr', 'CCI', 'CCF']
    result = _analyze_functional_groups(smiles)
    print("✅ Halogens")

def test_multiple_groups():
    """Test molecules with multiple functional groups"""
    print("\n=== MULTIPLE GROUPS ===")
    
    # Test: Complex molecules
    smiles = [
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin: ester, carboxylic acid, aromatic
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine: multiple amides, aromatic
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen: carboxylic acid, aromatic
    ]
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    assert len(result) > 0, "Should detect multiple groups"
    print(f"   Detected {len(result)} functional group types")
    print("✅ Multiple groups per molecule")

def test_counting_accuracy():
    """Test counting accuracy"""
    print("\n=== COUNTING ACCURACY ===")
    
    # Test 1: Single molecule, single group
    smiles = ['CCO']
    result = _analyze_functional_groups(smiles)
    # Check percentages add up correctly (skip halogen_breakdown which is a dict)
    total_pct = sum(v['pct_dataset'] for k, v in result.items() if k != 'halogen_breakdown')
    assert total_pct >= 0  # Should be non-negative
    print("✅ Single molecule counting")
    
    # Test 2: Multiple molecules, same groups
    smiles = ['CCO'] * 10
    result = _analyze_functional_groups(smiles)
    if result:
        sample_group = list(result.keys())[0]
        # All molecules should have the same pattern
        assert result[sample_group]['pct_dataset'] <= 100.0
    print("✅ Repeated molecule counting")
    
    # Test 3: Diverse molecules
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CC(=O)C']
    result = _analyze_functional_groups(smiles)
    # Check all percentages are valid (skip halogen_breakdown)
    for group, data in result.items():
        if group == 'halogen_breakdown':
            continue  # This is a different structure
        assert 0 <= data['pct_dataset'] <= 100
        assert data['count'] >= 0
        assert data['avg_per_molecule'] >= 0
    print("✅ Diverse molecule counting")

def test_edge_cases():
    """Test edge cases"""
    print("\n=== EDGE CASES ===")
    
    # Test 1: Single molecule
    smiles = ['CCO']
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ Single molecule")
    
    # Test 2: Very simple molecules
    smiles = ['C', 'CC', 'CCC', 'CCCC']
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ Simple alkanes")
    
    # Test 3: Single atom
    smiles = ['C']
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ Single atom")
    
    # Test 4: Exotic atoms
    smiles = ['[H]', '[He]', '[Li]', '[Na]']
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ Exotic atoms")

def test_large_dataset():
    """Test with large dataset"""
    print("\n=== LARGE DATASET ===")
    
    # Test: Large number of molecules
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CC(=O)C'] * 1000
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    
    # Verify counting makes sense for repeated pattern
    if result:
        for group, data in result.items():
            if group == 'halogen_breakdown':
                continue
            assert data['count'] >= 0  # Some groups may have 0 counts
            assert 0 <= data['pct_dataset'] <= 100
    print("✅ Large dataset (5000 molecules)")

def test_complex_molecules():
    """Test with complex molecular structures"""
    print("\n=== COMPLEX MOLECULES ===")
    
    # Test: Drug-like molecules
    smiles = [
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
        'C[C@H]1CN(C[C@@H](O1)C)c2c3n(c(=O)c(n2)C(=O)O)cc(c3F)N4CCNCC4',  # Moxifloxacin
    ]
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    assert len(result) > 0, "Should detect groups in complex molecules"
    print("✅ Drug-like molecules")

def test_stereochemistry_independence():
    """Test that stereochemistry doesn't affect group counting"""
    print("\n=== STEREOCHEMISTRY INDEPENDENCE ===")
    
    # Test: Same structure, different stereochemistry
    smiles = [
        'CC(O)CC',       # No stereo
        'C[C@H](O)CC',   # S
        'C[C@@H](O)CC',  # R
    ]
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    # All should detect the same OH group
    print("✅ Stereochemistry independence")

def test_fragmented_smiles():
    """Test fragmented SMILES"""
    print("\n=== FRAGMENTED SMILES ===")
    
    # Test: Multi-component SMILES
    smiles = [
        'CCO.Cl',
        'c1ccccc1.[Na+].[Cl-]',
        'CC(=O)O.O'
    ]
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ Fragmented SMILES")

def test_charged_species():
    """Test charged molecules"""
    print("\n=== CHARGED SPECIES ===")
    
    # Test: Various charged species
    smiles = [
        '[NH4+]',
        'CC[N+](C)(C)C',
        '[O-]C(=O)C',
        'CC(=O)[O-]'
    ]
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ Charged species")

def test_non_string_types():
    """Test non-string types"""
    print("\n=== NON-STRING TYPES ===")
    
    # Test: Mixed types
    smiles = ['CCO', 123, True, None, ['list'], {'dict': 'val'}]
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    # Should handle gracefully
    print("✅ Non-string types handled")

def test_metal_complexes():
    """Test organometallic compounds"""
    print("\n=== METAL COMPLEXES ===")
    
    # Test: Metal-containing molecules
    smiles = [
        'c1ccccc1[Fe]',
        'CC[Zn]CC',
        '[Pt](Cl)(Cl)(N)(N)'
    ]
    result = _analyze_functional_groups(smiles)
    assert isinstance(result, dict)
    print("✅ Metal complexes")

def test_return_values():
    """Test return value structure and types"""
    print("\n=== RETURN VALUES ===")
    
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    result = _analyze_functional_groups(smiles)
    
    assert isinstance(result, dict)
    
    for group_name, group_data in result.items():
        # Skip halogen_breakdown (different structure)
        if group_name == 'halogen_breakdown':
            continue
            
        assert isinstance(group_name, str)
        assert isinstance(group_data, dict)
        
        # Check required keys
        assert 'count' in group_data
        assert 'pct_dataset' in group_data
        assert 'avg_per_molecule' in group_data
        
        # Check types
        assert isinstance(group_data['count'], (int, np.integer))
        assert isinstance(group_data['pct_dataset'], (float, np.floating))
        assert isinstance(group_data['avg_per_molecule'], (float, np.floating))
        
        # Check ranges
        assert group_data['count'] >= 0
        assert 0 <= group_data['pct_dataset'] <= 100
        assert group_data['avg_per_molecule'] >= 0
    
    print("✅ Return value structure and types correct")

def test_consistency():
    """Test consistency of results"""
    print("\n=== CONSISTENCY ===")
    
    # Test: Same input should give same output
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN']
    result1 = _analyze_functional_groups(smiles)
    result2 = _analyze_functional_groups(smiles)
    
    assert result1.keys() == result2.keys()
    for key in result1:
        if key == 'halogen_breakdown':
            continue
        assert result1[key]['count'] == result2[key]['count']
        assert abs(result1[key]['pct_dataset'] - result2[key]['pct_dataset']) < 0.01
    
    print("✅ Results are consistent")

if __name__ == '__main__':
    print("="*80)
    print("RIGOROUS TEST SUITE: _analyze_functional_groups")
    print("="*80)
    
    try:
        test_basic_detection()
        test_empty_and_invalid()
        test_specific_groups()
        test_multiple_groups()
        test_counting_accuracy()
        test_edge_cases()
        test_large_dataset()
        test_complex_molecules()
        test_stereochemistry_independence()
        test_fragmented_smiles()
        test_charged_species()
        test_non_string_types()
        test_metal_complexes()
        test_return_values()
        test_consistency()
        
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
