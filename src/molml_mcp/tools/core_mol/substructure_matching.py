from __future__ import annotations
from typing import Dict, Mapping
from rdkit.Chem.rdchem import Mol
from molml_mcp.constants import STRUCTURAL_PATTERNS, FUNCTIONAL_GROUP_PATTERNS
from rdkit.Chem import MolFromSmiles, MolFromSmarts


def get_available_structural_patterns() -> Mapping[str, Dict[str, str]]:
    """
    Return a mapping of available structural SMARTS patterns for exploratory analysis.

    The returned dict has the form:
        {
            "<pattern name>": {
                "pattern": "<SMARTS string>",
                "comment": "<human-readable description>",
            },
            ...
        }

    Available structural pattern names and their meanings:

    - "Specified chiral carbon":
        Explicitly specified sp3 stereocentres on carbon (using @/@H), but not
        implicitly chiral centres or atoms with unspecified stereo.

    - "Quaternary Nitrogen":
        Tetravalent nitrogen atoms (quaternary or formally N(IV)), typically positively
        charged or in N=X arrangements (non-aromatic).

    - "S double-bonded to Carbon":
        Terminal sulfur (1-connected S) double-bonded to carbon, e.g. terminal C=S groups.

    - "Triply bonded N":
        Nitrogen atoms engaged in a triple bond (nitriles, isonitriles, etc.).

    - "Divalent Oxygen":
        Generic divalent oxygen centres (alcohols, ethers, carbonyls, etc.).

    - "Long_chain groups":
        Linear aliphatic chains with at least 8 consecutive non-aromatic atoms.

    - "Carbon_isolating":
        CLOGP-style isolating carbons: neutral carbons that are not CF3, not aromatic
        carbons between two aromatic heteroatoms, and not multiply bonded to heteroatoms.

    - "Rotatable bond":
        Non-terminal, non-ring, non-triple single bonds between two heavy atoms;
        standard medicinal chemistry rotatable-bond definition.

    - "Bicyclic":
        Molecules containing at least two bridgehead-like atoms (R2 with three ring
        neighbours); a coarse flag for bicyclic/polycyclic scaffolds.

    - "Ortho", "Meta", "Para":
        Ortho-, meta-, and para-substitution patterns on aromatic rings based on the
        distance in aromatic bonds between substituents.

    - "Acylic-bonds":
        Any non-ring bond (at least one endpoint is not in a ring).

    - "Single bond and not in a ring":
        Single, non-ring bonds between two atoms.

    - "Non-ring atom", "Ring atom":
        Atoms that are outside or inside rings, respectively.

    - "Macrocycle groups":
        Atoms belonging to rings larger than 7 members (macrocyclic atoms).

    - "S in aromatic 5-ring with lone pair":
        Divalent sulfur in a 5-membered aromatic ring (thiophene-like).

    - "Aromatic 5-Ring O with Lone Pair":
        Divalent oxygen in a 5-membered aromatic ring (furan-like).

    - "Spiro-ring center":
        Tetra-coordinated atom that is part of exactly two rings and connects
        multiple 4-6 membered rings (spiro junction).

    - "N in 5-ring arom":
        Anionic sp2 nitrogens in 5-membered aromatic rings (e.g. deprotonated azoles).

    - "CIS or TRANS double bond in a ring":
        Ring double bonds with explicit cis/trans stereochemistry.

    - "CIS or TRANS double or aromatic bond in a ring":
        Double or aromatic bonds in rings with explicit stereochemical annotation.

    - "Unfused benzene ring":
        Benzene rings where each aromatic carbon is only part of a single ring
        (non-fused benzene).

    - "Multiple non-fused benzene rings":
        Molecules containing at least two such non-fused benzene rings.

    - "Fused benzene rings":
        Fused benzene systems such as naphthalene-like scaffolds.

    - "Alkene (C=C)":
        Non-aromatic C=C double bonds between sp2 carbons.

    - "Alkyne (C#C)":
        C≡C triple bonds between sp carbons.

    - "Epoxide (3-membered cyclic ether)":
        Oxygen atoms in 3-membered rings (epoxides and related motifs).

    - "Aziridine (3-membered cyclic amine)":
        Nitrogen atoms in 3-membered rings (aziridines and related small azacycles).

    This is intended for exploratory data analysis and feature extraction in
    drug-discovery contexts; patterns are deliberately broad detector motifs
    rather than strict IUPAC definitions.
    """
    # Returning as a Mapping discourages callers from mutating the internal dict.
    return STRUCTURAL_PATTERNS


def get_available_functional_group_patterns() -> Mapping[str, Dict[str, str]]:
    """
    Return a mapping of available functional-group SMARTS patterns.

    The returned dict has the form:
        {
            "<functional group name>": {
                "pattern": "<SMARTS string>",
                "comment": "<human-readable description>",
            },
            ...
        }

    Available functional group names and their meanings (high-level):

    Carbonyl / carboxyl family:
    - "Carbonyl group":
        Generic C=O carbonyl (including zwitterionic C(+)-O(-) depiction).
    - "Aldehyde":
        Aldehyde groups (R-CHO).
    - "Amide":
        Amide carbonyl N-C(=O)-C (secondary/tertiary amides).
    - "Carbamate":
        Carbamate N-C(=O)-O motifs (carbamic esters/acids).
    - "Carboxylate Ion":
        Deprotonated carboxylic/carbonic/carbamic acids, C(=O)-O(-).
    - "Carbonic Acid or Carbonic Ester":
        Carbonic acid/ester-like C(=O)(O)O.
    - "Carboxylic acid":
        Carboxylic acids and their conjugate bases, C(=O)-OH / C(=O)-O(-).
    - "Ester Also hits anhydrides":
        Simple esters R-C(=O)-O-R'; comment is legacy, pattern is ester-centric.
    - "Ketone":
        Ketones R-C(=O)-R'.

    Urea / guanidine / amidine:
    - "Urea":
        Urea-like cores: C(=O) flanked by two nitrogens.
    - "Thiourea":
        Thiourea cores: C(=S) flanked by two nitrogens.
    - "Guanidine":
        Guanidine/guanidinium motifs N-C(=N)-N, strongly basic/cationic.
    - "Amidine":
        Neutral amidines C(=NR)-NH.
    - "Amidinium":
        Protonated amidines (amidinium cations).

    Ethers and simple O/N patterns:
    - "Ether":
        Dialkyl/aryl ethers R-O-R'.
    - "Mono-Hydrogenated Cation":
        Atoms with + charge and exactly one attached H.
    - "Not Mono-Hydrogenated":
        Atoms that do not have exactly one attached H (broad filter).
    - "Cyanamide":
        Cyanamide N-C≡N motifs.

    Amines and conjugated N:
    - "Primary or secondary amine, not amide":
        Sp3 amines (1-2 H) not directly bound to C=O (includes some cyanamides/thioamides).
    - "Enamine":
        Enamine N-C=C motifs.
    - "Enamine or Aniline Nitrogen":
        Sp3 N attached to vinyl or aromatic carbon (enamine/aniline-like).

    Aromatic heterocycles / azoles:
    - "Azole":
        5-membered aromatic heterocycles with N and another heteroatom (N/O/S).

    Hydrazine / hydrazone / imine:
    - "Hydrazine H2NNH2":
        Hydrazine-like N-N single bonds.
    - "Hydrazone C=NNH2":
        Hydrazone-like C=N-NH motifs.
    - "Substituted imine":
        Substituted imines (Schiff bases) with C=N-R and two carbon substituents on C.
    - "Substituted or un-substituted imine":
        Broader imine detector (C=N-R or C=NH with at least one carbon substituent).
    - "Iminium":
        Positively charged C=N+ species.

    Imides:
    - "Unsubstituted dicarboximide":
        Dicarboximides with N-H.
    - "Substituted dicarboximide":
        Dicarboximides with N-alkyl/aryl substitution.

    Nitrate / nitro / nitrile:
    - "Nitrate group":
        Nitrate-like N(=O)(=O)-O and related zwitterions.
    - "Nitrile":
        Nitrile groups -C≡N.
    - "Nitro group":
        Nitro groups -NO2 (neutral or zwitterionic), excluding nitrate.

    Hydroxyl / alcohol / phenol:
    - "Hydroxyl":
        Hydroxyl groups -OH attached to any atom.
    - "Hydroxyl in Alcohol":
        Alcoholic C-OH groups.
    - "Enol":
        Enolic OH attached to an sp2 carbon in C=C.
    - "Phenol":
        Phenolic OH attached to aromatic carbons.

    Thio / sulfur-containing:
    - "Carbo-Thioester":
        Thioesters R-S-C(=O)-R'.
    - "Thio analog of carbonyl":
        C=S analogues of C=O not classified as thioamides.
    - "Thiol, Sulfide or Disulfide Sulfur":
        Divalent sulfur atoms in thiols, sulfides, and disulfides.
    - "Thioamide":
        Thioamides N-C(=S)-.

    Sulfide / sulfone / sulfonamide / sulfoxide:
    - "Sulfide":
        Divalent sulfur (excludes thiols); sulfides/disulfides.
    - "Mono-sulfide":
        Mono-sulfide R-S-R' (S bound only to non-sulfur atoms).
    - "Two Sulfides":
        Molecules with at least two mono-sulfide motifs.
    - "Sulfone":
        Sulfones and sulfonyl-derived acids/esters (S(=O)2).
    - "Sulfonamide":
        Sulfonamides (sulfonyl-N motifs).
    - "Sulfoxide":
        Sulfoxides/sulfinyl species S(=O).

    Strong acids:
    - "Sulfonic acid or sulfonate":
        Sulfonic acids and sulfonates S(=O)2(OH/ O-).
    - "Phosphoric or phosphonic acid/ester":
        Phosphate/phosphonate-like P(=O)(O)(O/O-) groups.

    Aromatic nitrogens (6-ring):
    - "Ring sp2 N (pyridine-like)":
        Ring sp2 N in 6-membered aromatics (pyridine/diazine).
    - "Diazine-like ring (two ring Ns in 6-ring)":
        Approximate diazine motif with two ring nitrogens.

    Unsaturation / warheads:
    - "Alkene (non-aromatic)":
        Non-aromatic C=C double bonds.
    - "Alkyne (non-aromatic)":
        C≡C triple bonds.
    - "Michael acceptor (alpha,beta-unsat. carbonyl)":
        Approximate Michael acceptor: α,β-unsaturated carbonyls (enone-like).

    Small strained ring warheads:
    - "Epoxide":
        Explicit epoxide motifs (3-membered cyclic ethers).
    - "Aziridine":
        Explicit aziridine motifs (3-membered cyclic amines).

    Halogens:
    - "Any carbon attached to any halogen":
        Carbons directly bonded to F, Cl, Br, or I.
    - "Halogen":
        Halogen atoms themselves (F, Cl, Br, I).
    - "Three_halides groups":
        Molecules containing at least three halogen atoms (polyhalogenated flag).

    These patterns are intended as broad functional-group detectors for exploratory
    data analysis and medicinal-chemistry feature extraction, not as strict
    IUPAC-level classifications.
    """
    return FUNCTIONAL_GROUP_PATTERNS




# from rdkit.Chem import MolFromSmarts

def _mol_has_pattern(mol: Mol, smarts: str):
    """Check if a molecule has a substructure match for a given SMARTS pattern.
    
    Args:
        mol: RDKit molecule object to search
        smarts: SMARTS pattern string
        
    Returns:
        bool: True if pattern matches, False if no match or invalid pattern
    """
    pattern = MolFromSmarts(smarts)
    if pattern is None:
        # Invalid SMARTS pattern - return False instead of crashing
        return False
    return mol.HasSubstructMatch(pattern)


def _find_all_patterns_in_smiles(smi: str, smarts_dict: dict) -> list[str]:
    """Find all substructures that are present in a given SMILES string using a dict of SMARTS.

    Args:
        smi (str): The SMILES string to analyze.
        smarts_dict (dict): A dictionary where keys are pattern names and values are SMARTS strings.

    Returns:
        list[str]: List of pattern names that matched, or empty list if invalid/no matches.
    """
    # Handle invalid input types
    if not isinstance(smi, str):
        return []
    
    mol = MolFromSmiles(smi)
    if mol is None:
        return []

    matched_patterns = []
    for pattern_name, smarts in smarts_dict.items():
        if _mol_has_pattern(mol, smarts['pattern']):
            matched_patterns.append(pattern_name)

    return matched_patterns


def smiles_has_structural_pattern(smiles: str, smarts_pattern) -> bool:
    """Check if a SMILES string contains a specified structural pattern.

    Args:
        smi (str): The SMILES string to analyze.
        smarts_pattern (str): The SMARTS pattern string to search for.

    Returns:
        bool: True if the pattern is found, False otherwise.
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return False
    return _mol_has_pattern(mol, smarts_pattern)


def find_structural_patterns_in_smiles(smiles: str) -> str:
    """Find all structural patterns in a given SMILES string.
    
    Args:
        smiles: SMILES string to analyze
        
    Returns:
        str: Comma-separated pattern names, or empty string if no matches or invalid input
    """
    try:
        pattern_matches = _find_all_patterns_in_smiles(smiles, get_available_structural_patterns())
        return ', '.join(pattern_matches)
    except Exception:
        return ''


def find_functional_group_patterns_in_smiles(smiles: str) -> str:
    """Find all functional group patterns in a given SMILES string.
    
    Args:
        smiles: SMILES string to analyze
        
    Returns:
        str: Comma-separated pattern names, or empty string if no matches or invalid input
    """
    try:
        pattern_matches = _find_all_patterns_in_smiles(smiles, get_available_functional_group_patterns())
        return ', '.join(pattern_matches)
    except Exception:
        return ''   



# smi = 'CC(=O)Oc1ccccc1C(=O)O'

# find_structural_patterns_in_smiles('smi')
# find_functional_group_patterns_in_smiles(smi)