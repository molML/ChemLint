from rdkit.Chem import MolFromSmiles, MolToSmiles, MolFromSmarts, RemoveStereochemistry
from rdkit.Chem.AllChem import ReplaceSubstructs
from rdkit.Chem.rdchem import Mol
from molml_mcp.constants import COMMON_SOLVENTS, SMARTS_NEUTRALIZATION_PATTERNS


def _canonicalize_smiles(smi: str) -> tuple[str, str]: 
    """ Convert a SMILES string to its canonical form. Failed conversions are treated as None. Return both canonical SMILES and comment."""

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
    mol = MolFromSmiles(smiles)
    
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:

        RemoveStereochemistry(mol)

        # isomericSmiles=False makes the intent explicit: no stereo in the output
        return MolToSmiles(mol, isomericSmiles=False, canonical=True), "Passed"
    except Exception as e:
        return None, f"Failed: {str(e)}"
    