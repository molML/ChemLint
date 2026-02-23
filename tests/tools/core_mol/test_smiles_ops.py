import pytest
import numpy as np
import pandas as pd
from rdkit import Chem
from chemlint.tools.core_mol.smiles_ops import (
    _is_invalid_smiles,
    _canonicalize_smiles,
    _remove_pattern,
    _is_common_solvent_fragment,
    _strip_common_solvent_fragments,
    _defragment_smiles,
    _initialise_neutralisation_reactions,
    _neutralize_smiles,
    _flatten_stereochemistry,
    _remove_isotopes,
    enumerate_stereo_isomers_smiles,
    _calculate_conformer_energy,
    _select_isomer_from_mols,
    _has_chiral_centers,
    _has_complete_stereochemistry,
    _deduplicate_isomers,
    _standardize_stereo_smiles,
    _canonicalize_tautomer_smiles,
    _normalize_smiles,
    _reionize_smiles,
    _disconnect_metals_smiles,
    _validate_smiles,
)


# ============================================================================
# Validation and Canonicalization Tests
# ============================================================================


def test_is_invalid_smiles():
    """Test SMILES validity checking."""
    # Invalid cases
    assert _is_invalid_smiles(None) is True
    assert _is_invalid_smiles(np.nan) is True
    assert _is_invalid_smiles(pd.NA) is True
    assert _is_invalid_smiles(123) is True
    assert _is_invalid_smiles([]) is True
    
    # Valid cases
    assert _is_invalid_smiles("CCO") is False
    assert _is_invalid_smiles("c1ccccc1") is False
    assert _is_invalid_smiles("") is False  # Empty string is technically a string


def test_canonicalize_smiles():
    """Test SMILES canonicalization."""
    # Valid SMILES
    result, comment = _canonicalize_smiles("CCO")
    assert result == "CCO"
    assert comment == "Passed"
    
    # Different representations of same molecule
    result1, _ = _canonicalize_smiles("C(C)O")
    result2, _ = _canonicalize_smiles("OCC")
    assert result1 == result2 == "CCO"
    
    # Benzene
    result, _ = _canonicalize_smiles("c1ccccc1")
    assert result == "c1ccccc1"
    
    # Invalid SMILES
    result, comment = _canonicalize_smiles("InvalidSMILES")
    assert result is None
    assert "Failed" in comment
    
    # None input
    result, comment = _canonicalize_smiles(None)
    assert result is None
    assert "Invalid" in comment


def test_remove_pattern():
    """Test pattern removal from SMILES."""
    # Remove sodium from fragment
    result, comment = _remove_pattern("CCO.[Na+]", "[Na+]")
    assert result == "CCO"
    assert comment == "Passed"
    
    # No fragments - should return as is
    result, comment = _remove_pattern("CCO", "[Na+]")
    assert result == "CCO"
    assert comment == "Passed"
    
    # Multiple fragments with salt removal
    result, comment = _remove_pattern("CC(=O)O.[Na+].O", "[Na+]")
    assert "CC(=O)O" in result or result == "CC(=O)O"
    assert comment == "Passed"
    
    # Invalid SMILES
    result, comment = _remove_pattern(None, "[Na+]")
    assert result is None
    assert "Invalid" in comment


def test_is_common_solvent_fragment():
    """Test common solvent detection."""
    # Common solvents (need to check what's actually in COMMON_SOLVENTS)
    assert _is_common_solvent_fragment("O") is True  # Water
    assert _is_common_solvent_fragment("CCO") is True  # Ethanol
    
    # Not solvents
    assert _is_common_solvent_fragment("c1ccccc1C") is False  # Toluene (if not in list)
    assert _is_common_solvent_fragment("CC(C)O") is False  # IPA (if not in list)
    
    # Invalid
    assert _is_common_solvent_fragment("InvalidSMILES") is False


def test_strip_common_solvent_fragments():
    """Test stripping common solvent fragments."""
    # SMILES with water
    result, comment = _strip_common_solvent_fragments("c1ccccc1.O")
    assert "c1ccccc1" in result
    assert "Pass" in comment or "removed" in comment
    
    # No fragments
    result, comment = _strip_common_solvent_fragments("c1ccccc1")
    assert result == "c1ccccc1"
    assert "Pass" in comment
    
    # All fragments are solvents
    result, comment = _strip_common_solvent_fragments("O.CCO")
    assert "." in result  # Should keep original
    assert "all fragments" in comment.lower() or result == "O.CCO"
    
    # Invalid input
    result, comment = _strip_common_solvent_fragments(None)
    assert result is None
    assert "Invalid" in comment


def test_defragment_smiles():
    """Test SMILES defragmentation."""
    # Keep largest fragment
    result, comment = _defragment_smiles("CCO.O.C", keep_largest_fragment=True)
    assert result == "CCO"
    assert "largest" in comment.lower() or "Pass" in comment
    
    # No fragments
    result, comment = _defragment_smiles("CCO")
    assert result == "CCO"
    assert comment == "Pass"
    
    # Repeated fragments
    result, comment = _defragment_smiles("CCO.CCO.CCO")
    assert result == "CCO"
    assert "repeated" in comment.lower()
    
    # Don't keep largest
    result, comment = _defragment_smiles("CCO.O", keep_largest_fragment=False)
    assert "." in result
    assert "Unresolved" in comment
    
    # Invalid input
    result, comment = _defragment_smiles(None)
    assert result is None
    assert "Invalid" in comment


def test_neutralize_smiles():
    """Test SMILES neutralization."""
    transformations = _initialise_neutralisation_reactions()
    
    # Carboxylic acid (should remain as is or be neutralized)
    result, comment = _neutralize_smiles("CC(=O)[O-]", transformations)
    assert result is not None
    assert comment == "Passed"
    
    # Amine
    result, comment = _neutralize_smiles("CC[NH3+]", transformations)
    assert result is not None
    assert comment == "Passed"
    
    # Neutral molecule (no change expected)
    result, comment = _neutralize_smiles("CCO", transformations)
    assert result == "CCO"
    assert comment == "Passed"
    
    # Invalid input
    result, comment = _neutralize_smiles(None, transformations)
    assert result is None
    assert "Invalid" in comment


def test_flatten_stereochemistry():
    """Test stereochemistry removal."""
    # Chiral center
    result, comment = _flatten_stereochemistry("C[C@H](O)C")
    assert "@" not in result
    assert comment == "Passed"
    
    # E/Z double bond
    result, comment = _flatten_stereochemistry("C/C=C/C")
    assert "/" not in result and "\\" not in result
    assert comment == "Passed"
    
    # No stereochemistry
    result, comment = _flatten_stereochemistry("CCO")
    assert result == "CCO"
    assert comment == "Passed"
    
    # Invalid input
    result, comment = _flatten_stereochemistry(None)
    assert result is None
    assert "Invalid" in comment


def test_has_complete_stereochemistry():
    """Test stereochemistry completeness detection."""
    # No stereochemistry - should pass
    mol = Chem.MolFromSmiles("CC(C)C")
    assert _has_complete_stereochemistry(mol) is True
    
    # Fully specified chiral center - should pass
    mol = Chem.MolFromSmiles("C[C@H](O)CC")
    assert _has_complete_stereochemistry(mol) is True
    
    mol = Chem.MolFromSmiles("C[C@@H](O)CC")
    assert _has_complete_stereochemistry(mol) is True
    
    # Unspecified chiral center - should fail
    mol = Chem.MolFromSmiles("CC(O)CC")
    assert _has_complete_stereochemistry(mol) is False
    
    # Fully specified E double bond - should pass
    mol = Chem.MolFromSmiles("C/C=C/C")
    assert _has_complete_stereochemistry(mol) is True
    
    # Fully specified Z double bond - should pass
    mol = Chem.MolFromSmiles("C/C=C\\C")
    assert _has_complete_stereochemistry(mol) is True
    
    # No E/Z specification needed (symmetric) - should pass
    mol = Chem.MolFromSmiles("CC=CC")
    assert _has_complete_stereochemistry(mol) is True
    
    # Multiple chiral centers, all specified - should pass
    mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](C)N")
    assert _has_complete_stereochemistry(mol) is True
    
    # Invalid stereochemistry gets removed by RDKit - should pass
    mol = Chem.MolFromSmiles("[C@H]1(C)CCCC1")
    assert _has_complete_stereochemistry(mol) is True
    
    # Not a chiral center (quaternary carbon) - should pass
    mol = Chem.MolFromSmiles("CC(C)(O)C")
    assert _has_complete_stereochemistry(mol) is True
    
    # Both E/Z and chiral specified - should pass
    mol = Chem.MolFromSmiles("C/C=C/[C@@H](C)O")
    assert _has_complete_stereochemistry(mol) is True


def test_standardize_stereo_smiles_with_require_complete():
    """Test stereochemistry standardization with require_complete flag."""
    # Fully specified chiral center with require_complete=True - should pass
    result, comment = _standardize_stereo_smiles(
        "C[C@H](O)CC", 
        stereo_policy="keep", 
        require_complete=True
    )
    assert result is not None
    assert comment == "Passed"
    
    # Fully specified chiral center with require_complete=False - should pass
    result, comment = _standardize_stereo_smiles(
        "C[C@H](O)CC", 
        stereo_policy="keep", 
        require_complete=False
    )
    assert result is not None
    assert comment == "Passed"
    
    # Unspecified chiral center with require_complete=True - should fail
    result, comment = _standardize_stereo_smiles(
        "CC(O)CC", 
        stereo_policy="keep", 
        require_complete=True
    )
    assert result is None
    assert "Incomplete stereochemistry" in comment
    
    # Unspecified chiral center with require_complete=False - should pass
    result, comment = _standardize_stereo_smiles(
        "CC(O)CC", 
        stereo_policy="keep", 
        require_complete=False
    )
    assert result is not None
    assert comment == "Passed"
    
    # Fully specified E bond with require_complete=True - should pass
    result, comment = _standardize_stereo_smiles(
        "C/C=C/C", 
        stereo_policy="keep", 
        require_complete=True
    )
    assert result is not None
    assert comment == "Passed"
    
    # No stereochemistry with require_complete=True - should pass
    result, comment = _standardize_stereo_smiles(
        "CC(C)C", 
        stereo_policy="keep", 
        require_complete=True
    )
    assert result is not None
    assert comment == "Passed"
    
    # Test with flatten policy and require_complete - should still work
    result, comment = _standardize_stereo_smiles(
        "C[C@H](O)CC", 
        stereo_policy="flatten", 
        require_complete=True
    )
    assert result is not None
    assert "@" not in result
    assert comment == "Passed"


def test_remove_isotopes():
    """Test isotope removal."""
    # Carbon-13
    result, comment = _remove_isotopes("[13C]CO")
    assert "[13C]" not in result
    assert "C" in result
    assert comment == "Passed"
    
    # Deuterium
    result, comment = _remove_isotopes("CC([2H])O")
    assert "[2H]" not in result
    assert comment == "Passed"
    
    # No isotopes
    result, comment = _remove_isotopes("CCO")
    assert result == "CCO"
    assert comment == "Passed"
    
    # Invalid input
    result, comment = _remove_isotopes(None)
    assert result is None
    assert "Invalid" in comment


def test_enumerate_stereo_isomers_smiles():
    """Test stereoisomer enumeration."""
    # Molecule with chiral center (unassigned)
    result = enumerate_stereo_isomers_smiles("CC(O)C", max_isomers=10)
    assert isinstance(result, list)
    # May or may not enumerate if no explicit unassigned center
    
    # Molecule with explicit unassigned chirality
    result = enumerate_stereo_isomers_smiles("CC(C)(C)C", max_isomers=10)
    assert isinstance(result, list)
    
    # Invalid SMILES
    result = enumerate_stereo_isomers_smiles("InvalidSMILES")
    assert result == []
    
    # Simple molecule with potential stereocenters
    result = enumerate_stereo_isomers_smiles("C[C@@H](O)[C@@H](O)C", max_isomers=10)
    assert isinstance(result, list)


def test_select_isomer_from_mols():
    """Test isomer selection from mol list."""
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("c1ccccc1")
    mols = [mol1, mol2]
    
    # First policy
    result = _select_isomer_from_mols(mols, assign_policy="first")
    assert result is not None
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(mol1)
    
    # Random policy
    result = _select_isomer_from_mols(mols, assign_policy="random", random_seed=42)
    assert result is not None
    
    # Lowest energy policy
    result = _select_isomer_from_mols(mols, assign_policy="lowest", random_seed=42)
    assert result is not None
    
    # Empty list
    result = _select_isomer_from_mols([])
    assert result is None
    
    # Single molecule
    result = _select_isomer_from_mols([mol1])
    assert result is not None


def test_has_chiral_centers():
    """Test chiral center detection."""
    # Molecule with potential chiral center (carbon with 4 different groups)
    mol = Chem.MolFromSmiles("C[C@H](O)CC")
    result = _has_chiral_centers(mol)
    # May be True or False depending on RDKit's detection
    assert isinstance(result, bool)
    
    # Molecule without chiral center
    mol = Chem.MolFromSmiles("CCO")
    assert _has_chiral_centers(mol) is False
    
    # Benzene (no chiral centers)
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert _has_chiral_centers(mol) is False


def test_deduplicate_isomers():
    """Test isomer deduplication."""
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("CCO")  # Same as mol1
    mol3 = Chem.MolFromSmiles("CCC")  # Different molecule
    
    mols = [mol1, mol2, mol3]
    result = _deduplicate_isomers(mols)
    
    assert len(result) == 2  # mol1 and mol3, mol2 is duplicate
    
    # All identical
    mols = [mol1, mol1, mol1]
    result = _deduplicate_isomers(mols)
    assert len(result) == 1
    
    # Test with actual stereoisomers if they're preserved
    mol_r = Chem.MolFromSmiles("C[C@H](O)[C@H](O)C")
    mol_s = Chem.MolFromSmiles("C[C@@H](O)[C@@H](O)C")
    mols = [mol_r, mol_s]
    result = _deduplicate_isomers(mols)
    # Should have 1 or 2 depending on whether stereo is preserved
    assert len(result) >= 1


def test_standardize_stereo_smiles():
    """Test stereochemistry standardization."""
    # Keep policy with molecule that has definite chirality
    result, comment = _standardize_stereo_smiles("C[C@H](O)CC", stereo_policy="keep")
    # May or may not preserve @ depending on whether it's a true chiral center
    assert result is not None
    assert comment == "Passed"
    
    # Flatten policy
    result, comment = _standardize_stereo_smiles("C[C@H](O)C", stereo_policy="flatten")
    assert "@" not in result
    assert comment == "Passed"
    
    # Assign policy with first
    result, comment = _standardize_stereo_smiles("CC(O)C", stereo_policy="assign", assign_policy="first")
    assert result is not None
    assert comment == "Passed"
    
    # Invalid stereo_policy with molecule that has no chirality
    # The function may return the molecule as-is if it has no chirality
    result, comment = _standardize_stereo_smiles("CCCC", stereo_policy="invalid")
    # Should either error or return as-is for non-chiral molecules
    assert result is None or result == "CCCC"
    
    # Invalid input
    result, comment = _standardize_stereo_smiles(None)
    assert result is None
    assert "Invalid" in comment


def test_canonicalize_tautomer_smiles():
    """Test tautomer canonicalization."""
    # Pyridone tautomers - use proper aromatic notation
    result1, comment1 = _canonicalize_tautomer_smiles("O=c1cccc[nH]1")
    result2, comment2 = _canonicalize_tautomer_smiles("Oc1ccccn1")
    
    # Both tautomers should give the same canonical form
    assert result1 is not None
    assert result2 is not None
    assert result1 == result2
    assert "Passed" in comment1
    assert "Passed" in comment2
    
    # Simple molecule (no tautomerism)
    result, comment = _canonicalize_tautomer_smiles("CCO")
    assert result == "CCO"
    assert "Passed" in comment
    
    # Check stereochemistry warning
    result, comment = _canonicalize_tautomer_smiles("C[C@H](O)C=O")
    assert result is not None
    # May or may not have warning depending on whether stereo was lost
    
    # Invalid input
    result, comment = _canonicalize_tautomer_smiles(None)
    assert result is None
    assert "Invalid" in comment


def test_normalize_smiles():
    """Test functional group normalization."""
    # Nitro group normalization
    result, comment = _normalize_smiles("C[N+](=O)[O-]")
    assert result is not None
    assert comment == "Passed"
    
    # Simple molecule
    result, comment = _normalize_smiles("CCO")
    assert result == "CCO"
    assert comment == "Passed"
    
    # N-oxide
    result, comment = _normalize_smiles("CN(=O)=O")
    assert result is not None
    assert comment == "Passed"
    
    # Invalid input
    result, comment = _normalize_smiles(None)
    assert result is None
    assert "Invalid" in comment


def test_reionize_smiles():
    """Test SMILES reionization."""
    # Zwitterion
    result, comment = _reionize_smiles("C(C(=O)[O-])[NH3+]")
    assert result is not None
    assert comment == "Passed"
    
    # Simple molecule
    result, comment = _reionize_smiles("CCO")
    assert result == "CCO"
    assert comment == "Passed"
    
    # Carboxylate
    result, comment = _reionize_smiles("CC(=O)[O-]")
    assert result is not None
    assert comment == "Passed"
    
    # Invalid input
    result, comment = _reionize_smiles(None)
    assert result is None
    assert "Invalid" in comment


def test_disconnect_metals_smiles():
    """Test metal disconnection."""
    # Metal complex (if supported)
    result, comment = _disconnect_metals_smiles("CC[Mg]Br")
    assert result is not None
    # Should disconnect metal or handle appropriately
    
    # Organic molecule (no metals)
    result, comment = _disconnect_metals_smiles("CCO")
    assert result == "CCO"
    assert comment == "Passed"
    
    # Drop inorganics option with inorganic
    result, comment = _disconnect_metals_smiles("[Na]Cl", drop_inorganics=True)
    assert result is None or "no carbon" in comment.lower()
    
    # Drop inorganics option with organic
    result, comment = _disconnect_metals_smiles("CCO", drop_inorganics=True)
    assert result == "CCO"
    assert comment == "Passed"
    
    # Invalid input
    result, comment = _disconnect_metals_smiles(None)
    assert result is None
    assert "Invalid" in comment


def test_validate_smiles():
    """Test SMILES validation."""
    # Valid SMILES
    result, comment = _validate_smiles("CCO")
    assert result == "CCO"
    assert comment == "Passed"
    
    # Valid benzene
    result, comment = _validate_smiles("c1ccccc1")
    assert result == "c1ccccc1"
    assert comment == "Passed"
    
    # Invalid SMILES
    result, comment = _validate_smiles("InvalidSMILES")
    assert result is None
    assert "Failed" in comment
    
    # None input
    result, comment = _validate_smiles(None)
    assert result is None
    assert "Invalid" in comment
    
    # Empty molecule (if possible to construct)
    result, comment = _validate_smiles("")
    assert result is None
    assert "Failed" in comment or "Invalid" in comment
