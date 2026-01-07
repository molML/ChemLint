from __future__ import annotations
from rdkit.Chem import MolFromSmiles, MolToSmiles, MolFromSmarts, RemoveStereochemistry
from rdkit.Chem.AllChem import ReplaceSubstructs
from rdkit.Chem.rdchem import Mol
from molml_mcp.constants import COMMON_SOLVENTS, SMARTS_NEUTRALIZATION_PATTERNS

from typing import List, Optional
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem import FindMolChiralCenters
from rdkit.Chem.MolStandardize import rdMolStandardize

# create one global enumerator so I don't reallocate it on every call
_TAUT_ENUM = rdMolStandardize.TautomerEnumerator()
_NORMALIZER = rdMolStandardize.Normalizer()
_REIONIZER = rdMolStandardize.Reionizer()
_METAL_DISCONNECTOR = rdMolStandardize.MetalDisconnector()


def _is_invalid_smiles(smi) -> bool:
    """Check if SMILES is None, NaN, or otherwise invalid."""
    if smi is None:
        return True
    # Check if it's a string - the only valid type
    if isinstance(smi, str):
        return False
    # For non-strings, check specific types to avoid pd.isna() returning arrays
    import numpy as np
    # Arrays and lists are not valid SMILES
    if isinstance(smi, (np.ndarray, list, tuple)):
        return True
    # For scalars (float, int, pd.NA), check if it's NaN/NA, otherwise invalid
    try:
        if pd.isna(smi):
            return True
    except (TypeError, ValueError):
        pass
    # Any non-string type is invalid
    return True


def _canonicalize_smiles(smi: str) -> tuple[str, str]: 
    """ Convert a SMILES string to its canonical form. Failed conversions are treated as None. Return both canonical SMILES and comment."""

    # Handle None or NaN input (missing data)
    if _is_invalid_smiles(smi):
        return None, "Failed: Invalid SMILES string"

    mol = MolFromSmiles(smi)

    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        smi_canon = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi_canon, "Passed"
    
    except Exception as e:
        return None, f"Failed: {str(e)}"



def _remove_pattern(smi: str, smarts_pattern: str) -> tuple[str, str]: 
    """ Remove some pattern from a SMILES string using the specified SMARTS.

    :param smi: single SMILES string
    :param smarts_pattern: SMARTS pattern (e.g., "[Cl,Na,Mg]")
    :return: cleaned SMILES without pattern, comment
    """
    from rdkit.Chem.SaltRemover import SaltRemover

    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smi):
        return None, "Skipped: Invalid SMILES string"

    # Create SaltRemover with the provided SMARTS pattern
    # defnData format: "SMARTS<tab>name" per line
    remover = SaltRemover(defnData=f"{smarts_pattern}\tsalts")

    # If no fragments, no salts to remove
    if '.' not in smi:
        return smi, "Passed"

    # Try to parse the SMILES with fragments
    try:
        mol = MolFromSmiles(smi)
        
        if mol is None:
            return None, "Failed: Invalid SMILES string"

        # Remove salts
        cleaned_mol = remover.StripMol(mol, dontRemoveEverything=True)
        
        if cleaned_mol is None or cleaned_mol.GetNumAtoms() == 0:
            return None, "Failed: All fragments were salts"
        
        cleaned_smi = MolToSmiles(cleaned_mol)
        
        # If still has fragments after salt removal, keep the largest
        if '.' in cleaned_smi:
            frags = cleaned_smi.split('.')
            # Sort by length and take longest
            largest_frag = max(frags, key=len)
            return largest_frag, "Passed"
        else:
            return cleaned_smi, "Passed"
            
    except Exception as e:
        return None, f"Failed: {str(e)}"



def _is_common_solvent_fragment(smiles_frag: str) -> bool:
    """
    Return True if this standalone fragment is one of the known common solvents.
    Uses canonical SMILES matching; independent of any broader cleaning pipeline.
    """
    mol = MolFromSmiles(smiles_frag)
    if mol is None:
        return False
    can = MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return can in COMMON_SOLVENTS


def _strip_common_solvent_fragments(smi: str) -> tuple[str, str]: 
    """
    Remove known common solvent fragments from a fragmented SMILES string.

    Behavior:
    - If the SMILES has no '.' (single component), it is returned unchanged.
    - If some fragments match the solvent list and at least one fragment
      does NOT match, solvent fragments are dropped and the rest is joined.
    - If *all* fragments would be removed as solvents, the original SMILES
      is returned unchanged (assumed main molecule of interest).
    """

    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smi):
        return None, "Skipped: Invalid SMILES string"

    # Only act on fragmented SMILES
    if '.' not in smi:
        return smi, 'Pass'

    try:
        frags = [f.strip() for f in smi.split('.') if f.strip()]
        
        kept: list[str] = []
        any_removed = False

        for frag in frags:
            if _is_common_solvent_fragment(frag):
                any_removed = True
            else:
                kept.append(frag)

        # Case 1: nothing matched as solvent → return original
        if not any_removed:
            return smi, 'SMILES string is fragmented, but found no common solvents'

        # Case 2: everything would be removed → keep original (your preference)
        if not kept:
            return smi, 'SMILES string is fragmented, but all fragments are common solvents. Kept original SMILES'

        # Case 3: we removed some solvents but kept at least one fragment
        return '.'.join(kept), 'Pass, removed solvents'

    except Exception as e:
        return smi, f"Failed: {str(e)}"



def _defragment_smiles(smiles: str, keep_largest_fragment: bool = True) -> tuple[str, str]:
    """ Defragment a SMILES string by removing smaller fragments. """

    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smiles):
        return None, "Skipped: Invalid SMILES string"

    # If no fragments, nothing to do
    if '.' not in smiles:
        return smiles, "Pass"

    try:
        frags = [f.strip() for f in smiles.split('.') if f.strip()]
        
        # if the fragment is repeated, keep only one instance
        if len(frags) != len(set(frags)):
            return frags[0], "Pass, kept one instance of repeated fragments"
        
        if keep_largest_fragment:
            # Find the largest fragment by length
            largest_frag = max(frags, key=len)
            return largest_frag, "Pass, defragmented to largest component"
        else:
            return smiles, "Unresolved, contains fragments and keep_largest_fragment is False"

    except Exception as e:
        return None, f"Failed: {str(e)}"
    

def _initialise_neutralisation_reactions() -> list[(Mol, Mol)]:
    """ adapted from the rdkit contribution of Hans de Winter """
    return [(MolFromSmarts(x), MolFromSmiles(y, False)) for x, y in SMARTS_NEUTRALIZATION_PATTERNS]


def _neutralize_smiles(smiles: str, transformations: list[(Mol, Mol)]) -> tuple[str, str]:
    """ Use several neutralisation reactions based on patterns defined in SMARTS_NEUTRALIZATION_PATTERNS to neutralize charged
    molecules. Transformations should be pre-initialized via _initialise_neutralisation_reactions().

    :param smiles: Canonical SMILES string
    :return: SMILES of the neutralized molecule
    """
    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smiles):
        return None, "Skipped: Invalid SMILES string"
    
    mol = MolFromSmiles(smiles)

    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        # applies the transformations
        for i, (reactant, product) in enumerate(transformations):   
            while mol.HasSubstructMatch(reactant):
                rms = ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]

        # converts back the molecule to smiles
        smiles = MolToSmiles(mol, canonical=True, isomericSmiles=True)

        return smiles, "Passed"
    
    except Exception as e:
        return None, f"Failed: {str(e)}"


def _flatten_stereochemistry(smiles: str) -> tuple[str, str]:
    """
    Remove all stereochemistry (chiral centers + E/Z double bonds) 
    from a SMILES string using RDKit.
    """
    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smiles):
        return None, "Skipped: Invalid SMILES string"
    
    mol = MolFromSmiles(smiles)

    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:

        RemoveStereochemistry(mol)

        # isomericSmiles=False makes the intent explicit: no stereo in the output
        return MolToSmiles(mol, isomericSmiles=False, canonical=True), "Passed"
    except Exception as e:
        return None, f"Failed: {str(e)}"
    

def _remove_isotopes(smiles: str) -> tuple[str, str]:
    """
    Replace all isotopically-labeled atoms in a SMILES string
    by their default (non-isotopic) form.

    Examples:
        [13CH3][18F]   ->  CCF
        CC([2H])O      ->  CCO
    """
    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smiles):
        return None, "Skipped: Invalid SMILES string"
    
    mol = MolFromSmiles(smiles)

    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        # convert all isotope flags to 0 (default) i.e. convert C13 to C12
        for atom in mol.GetAtoms():
            if atom.GetIsotope() != 0:
                atom.SetIsotope(0)
        # Keep stereo etc.; isotopes are gone because they're all 0 now
        return MolToSmiles(mol, isomericSmiles=True, canonical=True), "Passed"
    except Exception as e:
        return None, f"Failed: {str(e)}"


def enumerate_stereo_isomers_smiles(smiles: str, max_isomers: int = 32, try_embedding: bool = False, only_unassigned: bool = True, 
                                    random_seed: int = 42) -> List[str]:
    """
    Enumerate stereoisomers for a SMILES string.

    Args:
        smiles: Input SMILES.
        max_isomers: Maximum number of stereoisomers to enumerate.
        try_embedding: Let RDKit try 3D embedding to prune degenerates.
        only_unassigned: Only enumerate unassigned stereocenters if True.
        random_seed: Random seed for RDKit's enumeration.

    Returns:
        List of isomeric SMILES strings (possibly empty if invalid or no isomers).
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return []

    np.random.seed(random_seed)

    opts = StereoEnumerationOptions(
        tryEmbedding=try_embedding,
        maxIsomers=max_isomers,
        onlyUnassigned=only_unassigned,
        rand=random_seed,
    )

    try:
        isomers = list(EnumerateStereoisomers(mol, options=opts))
    except Exception:
        return []

    if not isomers:
        return []

    # deduplicate
    isomers = _deduplicate_isomers(isomers)

    # Return canonical isomeric SMILES for each isomer
    return [MolToSmiles(iso, canonical=True, isomericSmiles=True) for iso in isomers]


def _calculate_conformer_energy(mol: Chem.Mol, random_seed: int = 42) -> Optional[float]:
    """
    Calculate MMFF94 (or UFF) energy for a single conformer.
    Returns None if embedding or energy evaluation fails.
    """
    try:
        mol_h = Chem.AddHs(mol)

        if AllChem.EmbedMolecule(mol_h, randomSeed=random_seed) != 0:
            return None

        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            props = AllChem.MMFFGetMoleculeProperties(mol_h)
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
            ff.Minimize()
            return ff.CalcEnergy()
        else:
            AllChem.UFFOptimizeMolecule(mol_h)
            ff = AllChem.UFFGetMoleculeForceField(mol_h)
            return ff.CalcEnergy()
    except Exception:
        return None


def _select_isomer_from_mols(
    isomer_mols: List[Chem.Mol],
    assign_policy: str = "first",  # "first" | "random" | "lowest"
    random_seed: int = 42,
) -> Optional[Chem.Mol]:
    """
    Select a single isomer from a list of RDKit mols.
    """
    if not isomer_mols:
        return None

    if len(isomer_mols) == 1 or assign_policy == "first":
        return isomer_mols[0]

    if assign_policy == "random":
        np.random.seed(random_seed)
        idx = np.random.randint(len(isomer_mols))
        return isomer_mols[idx]

    if assign_policy == "lowest":
        scored = []
        for iso in isomer_mols:
            e = _calculate_conformer_energy(iso, random_seed=random_seed)
            if e is not None:
                scored.append((e, iso))
        if scored:
            return min(scored, key=lambda x: x[0])[1]
        else:
            # fall back to first if all energies failed
            return isomer_mols[0]

    # Fallback
    return isomer_mols[0]


def _has_chiral_centers(mol: Chem.Mol) -> bool:
    """
    Check if molecule has any chiral centers (assigned or unassigned).
    Returns False if detection fails.
    """
    try:
        chiral_centers = FindMolChiralCenters(mol, includeUnassigned=True)
        return len(chiral_centers) > 0
    except Exception:
        return False


def _deduplicate_isomers(isomer_mols: List[Chem.Mol]) -> List[Chem.Mol]:
    """
    Remove duplicate stereoisomers based on canonical SMILES.
    """
    seen = set()
    unique = []
    for iso in isomer_mols:
        smi = MolToSmiles(iso, canonical=True, isomericSmiles=True)
        if smi not in seen:
            seen.add(smi)
            unique.append(iso)
    return unique


def _standardize_stereo_smiles(
    smiles: str,
    stereo_policy: str = "keep",          # "keep" | "assign" | "flatten"
    assign_policy: str = "first",         # "first" | "random" | "lowest"
    max_isomers: int = 32,
    try_embedding: bool = False,
    only_unassigned: bool = True,
    only_unique: bool = True,
    random_seed: int = 42,
) -> tuple[str, str]:
    """
    1→1 stereochemistry handling for use in a cleaning/standardization pipeline.

    Args:
        smiles: Input SMILES.
        stereo_policy:
            - "keep":    return canonical isomeric SMILES (no stereo changes)
            - "assign":  enumerate stereoisomers and pick one (see `assign_policy`)
            - "flatten": remove all stereochemistry (chiral centers + E/Z bonds)
        assign_policy: How to pick a single isomer when stereo_policy == "assign":
            - "first":   first enumerated
            - "random":  random choice
            - "lowest":  lowest MMFF94/UFF energy
        max_isomers, try_embedding, only_unassigned, only_unique, random_seed:
            Parameters forwarded to enumeration / energy evaluation.

    Returns:
        Tuple of (SMILES string, comment). Returns (None, error message) if the input is invalid.
    """
    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smiles):
        return None, "Skipped: Invalid SMILES string"
    
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        # Handle flatten policy
        if stereo_policy == "flatten":
            return _flatten_stereochemistry(smiles)
        
        # Detect chiral centers (including unassigned)
        has_chirality = _has_chiral_centers(mol)
        
        # Handle "keep" policy or molecules without chirality
        if stereo_policy == "keep" or not has_chirality:
            return MolToSmiles(mol, canonical=True, isomericSmiles=True), "Passed"

        if stereo_policy != "assign":
            return None, f"Failed: Unsupported stereo_policy '{stereo_policy}'. Use 'keep', 'assign', or 'flatten'."

        # --- Assign policy: enumerate isomers, then choose one ---
        isomer_smiles = enumerate_stereo_isomers_smiles(
            smiles=smiles,
            max_isomers=max_isomers,
            try_embedding=try_embedding,
            only_unassigned=only_unassigned,
            random_seed=random_seed,
        )

        if not isomer_smiles:
            return MolToSmiles(mol, canonical=True, isomericSmiles=True), "Passed"

        # Convert SMILES back to mol objects for selection
        isomer_mols = [MolFromSmiles(smi) for smi in isomer_smiles]
        isomer_mols = [m for m in isomer_mols if m is not None]

        if not isomer_mols:
            return None, "Failed: Could not parse enumerated stereoisomers"

        selected = _select_isomer_from_mols(
            isomer_mols,
            assign_policy=assign_policy,
            random_seed=random_seed,
        )

        if selected is None:
            return MolToSmiles(mol, canonical=True, isomericSmiles=True), "Passed"

        return MolToSmiles(selected, canonical=True, isomericSmiles=True), "Passed"
    
    except Exception as e:
        return None, f"Failed: {str(e)}"
    

def _canonicalize_tautomer_smiles(smiles: str) -> tuple[str, str]:
    """
    Standardize a SMILES string to RDKit's canonical tautomer.

    Returns a single canonical SMILES for all tautomers of the same scaffold.
    This ensures that different tautomeric forms of the same molecule are 
    represented by the same SMILES string.
    
    **WARNING**: This function can REMOVE or CHANGE stereochemistry. This is
    a known limitation of RDKit's tautomer enumerator.

    Args:
        smiles: Input SMILES string.

    Returns:
        Tuple of (SMILES string, comment). Returns (None, error message) if 
        the input is invalid. Comment includes warning if stereochemistry was lost.

    Examples:
        >>> canonicalize_tautomer_smiles("O=C1NC=CC=C1")
        ('O=C1NC=CC=C1', 'Passed')
        >>> canonicalize_tautomer_smiles("OC1=NC=CC=C1")
        ('O=C1NC=CC=C1', 'Passed')   # same output as above
    """
    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smiles):
        return None, "Skipped: Invalid SMILES string"
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        # Check if input has stereochemistry
        has_stereo_input = '@' in smiles or '/' in smiles or '\\' in smiles
        
        can_mol = _TAUT_ENUM.Canonicalize(mol)
        canonical_smiles = Chem.MolToSmiles(can_mol, canonical=True, isomericSmiles=True)
        
        # Check if stereochemistry was lost
        has_stereo_output = '@' in canonical_smiles or '/' in canonical_smiles or '\\' in canonical_smiles
        
        if has_stereo_input and not has_stereo_output:
            return canonical_smiles, "Passed (WARNING: Stereochemistry was removed by tautomer canonicalization)"
        
        return canonical_smiles, "Passed"
    except Exception as e:
        return None, f"Failed: {str(e)}"


def _normalize_smiles(smi: str) -> tuple[str | None, str]:
    """
    Normalize functional groups (e.g. nitro, N-oxide, azides) using
    RDKit's rdMolStandardize.Normalizer to fix "weird valence forms"

    Returns:
        (canonical isomeric SMILES or None, comment)
    """
    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smi):
        return None, "Skipped: Invalid SMILES string"
    
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        mol = _NORMALIZER.normalize(mol)
        smi_norm = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi_norm, "Passed"
    except Exception as e:
        return None, f"Failed: Normalization error: {e}"
    

def _reionize_smiles(smi: str) -> tuple[str | None, str]:
    """
    Reionize a SMILES string to a preferred charge distribution using
    RDKit's rdMolStandardize.Reionizer.

    Expects a "reasonable" structure (ideally already normalized). 
    Also helps with zwitterions and multi-site ionizable systems before 
    neutralization in case that is performed later on.

    Returns:
        (canonical isomeric SMILES or None, comment)
    """
    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smi):
        return None, "Skipped: Invalid SMILES string"
    
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        mol = _REIONIZER.reionize(mol)
        smi_reion = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi_reion, "Passed"
    except Exception as e:
        return None, f"Failed: Reionization error: {e}"
    

def _disconnect_metals_smiles(
    smi: str,
    drop_inorganics: bool = False,
) -> tuple[str | None, str]:
    """
    Disconnect coordinate bonds to metals and optionally drop purely inorganic
    molecules (no carbon atoms).

    Returns transformed SMILES (or None) and a comment.
    """
    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smi):
        return None, "Skipped: Invalid SMILES string"
    
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        mol = _METAL_DISCONNECTOR.Disconnect(mol)

        if drop_inorganics:
            has_carbon = any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
            if not has_carbon:
                return None, "Failed: Inorganic molecule (no carbon atoms)"

        smi_out = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi_out, "Passed"
    except Exception as e:
        return None, f"Failed: Metal disconnection error: {e}"


def _validate_smiles(smi: str) -> tuple[str | None, str]:
    """
    Lightweight validation that a SMILES string is still a sane RDKit molecule.

    - Parses with MolFromSmiles (includes sanitization).
    - Requires at least one atom.

    Returns:
        (original SMILES or None, comment).
    """
    if _is_invalid_smiles(smi):
        return None, "Skipped: Invalid SMILES string"
    
    try:
        mol = MolFromSmiles(smi)
    except Exception as e:
        return None, f"Failed: Exception during parsing: {e}"

    if mol is None:
        return None, "Failed: Invalid SMILES string"

    if mol.GetNumAtoms() == 0:
        return None, "Failed: Empty molecule (0 atoms)"

    return smi, "Passed"