"""
Constants for molml_mcp package.

COMMON_SOLVENTS: List of common solvent SMILES strings.
SMARTS_NEUTRALIZATION_PATTERNS: Patterns for neutralizing common charged groups.
SMARTS_COMMON_SALTS: SMARTS pattern for common inorganic salts.
STRUCTURAL_PATTERNS: Dictionary of structural SMARTS patterns with comments.
FUNCTIONAL_GROUP_PATTERNS: Dictionary of functional group SMARTS patterns with comments.

"""

from __future__ import annotations
from typing import Dict, Mapping


COMMON_SOLVENTS = [
    'O',
    'O=[N+]([O-])O',
    'F[P-](F)(F)(F)(F)F',
    'O=C([O-])C(F)(F)F',
    'O=C(O)CC(O)(CC(=O)O)C(=O)O',
    'CCO',
    'CCN(CC)CC',
    '[O-][Cl+3]([O-])([O-])O',
    'O=P(O)(O)O',
    'O=C(O)/C=C/C(=O)O',
    'O=C(O)/C=C\\C(=O)O',
    '[O-][Cl+3]([O-])([O-])[O-]',
    'CS(=O)(=O)O',
    'O=C(O)C(=O)O',
    'F[B-](F)(F)F',
    'C',
    'Cc1ccc(S(=O)(=O)[O-])cc1',
    'C1CCC(NC2CCCCC2)CC1',
    'O=CO',
    'O=S(=O)([O-])O',
    'O=C(O)C(F)(F)F',
    'COS(=O)(=O)[O-]',
    'CN(C)C=O',
    'Cc1ccc(S(=O)(=O)O)cc1',
    'O=C(O)CCC(=O)O',
    'O=C(O)[C@H](O)[C@@H](O)C(=O)O',
    'CS(=O)(=O)[O-]',
    'c1ccncc1',
    'NCCO',
    'O=S(=O)([O-])C(F)(F)F',
    'CNC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO',
    'O=C(O)C(O)C(O)C(=O)O',
    'CC(=O)O',
    'NC(CO)(CO)CO',
    'O=S(=O)(O)O'
    ]

SMARTS_NEUTRALIZATION_PATTERNS = (
    # Imidazoles
    ('[n+;H]', 'n'),
    # Amines
    ('[N+;!H0]', 'N'),
    # Carboxylic acids and alcohols
    ('[$([O-]);!$([O-][#7])]', 'O'),
    # Thiols
    ('[S-;X1]', 'S'),
    # Sulfonamides
    ('[$([N-;X2]S(=O)=O)]', 'N'),
    # Enamines
    ('[$([N-;X2][C,N]=C)]', 'N'),
    # Tetrazoles
    ('[n-]', '[nH]'),
    # Sulfoxides
    ('[$([S-]=O)]', 'S'),
    # Amides
    ('[$([N-]C=O)]', 'N'),
)

SMARTS_COMMON_SALTS = "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]"


STRUCTURAL_PATTERNS: Dict[str, Dict[str, str]] = {
    # chirality
    "Specified chiral carbon": {
        "pattern": "[$([#6X4@](*)(*)(*)*),$([#6X4@H](*)(*)*)]",
        "comment": (
            "sp3 carbons whose chirality is explicitly specified (@/@H). "
            "Does not include potentially chiral centres with implicit H or unspecified stereo."
        ),
    },

    # connectivity
    "Quaternary Nitrogen": {
        "pattern": "[$([NX4+]),$([NX4]=*)]",
        "comment": "Formally tetravalent (quaternary) nitrogen, including cationic and N=X forms (non-aromatic).",
    },
    "S double-bonded to Carbon": {
        "pattern": "[$([SX1]=[#6])]",
        "comment": "Terminal sulfur (1-connected) double-bonded to carbon (e.g. C=S at the end of a chain).",
    },
    "Triply bonded N": {
        "pattern": "[$([NX1]#*)]",
        "comment": "Nitrogen in a triple bond (e.g. nitriles, isonitriles).",
    },
    "Divalent Oxygen": {
        "pattern": "[$([OX2])]",
        "comment": "Generic divalent oxygen (e.g. in alcohols, ethers, carbonyls, etc.).",
    },

    # chains and branching
    "Long_chain groups": {
        "pattern": "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]",
        "comment": "Linear aliphatic chain with at least 8 non-aromatic atoms in a row (no ring atoms).",
    },
    "Carbon_isolating": {
        "pattern": "[$([#6+0]);!$(C(F)(F)F);!$(c(:[!c]):[!c])!$([#6]=,#[!#6])]",
        "comment": (
            "CLOGP-style isolating carbon: neutral carbon that is not CF3, "
            "not an aromatic carbon flanked by two aromatic heteroatoms, and not "
            "multiply bonded to a heteroatom."
        ),
    },

    # rotation
    "Rotatable bond": {
        "pattern": "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]",
        "comment": (
            "Non-terminal, non-triple, non-ring single bond between two heavy atoms; "
            "standard rotatable bond definition excluding ring bonds and terminal atoms."
        ),
    },

    # cyclic features
    "Bicyclic": {
        "pattern": "[$([*R2]([*R])([*R])([*R]))].[$([*R2]([*R])([*R])([*R]))]",
        "comment": (
            "Molecules containing at least two bridgehead-like atoms: each atom is in two rings "
            "and connected to three other ring atoms. Acts as a coarse bicyclic/polycyclic flag."
        ),
    },
    "Ortho": {
        "pattern": "*-!:aa-!:*",
        "comment": "Ortho-substituted aromatic ring: two substituents separated by 1 aromatic bond.",
    },
    "Meta": {
        "pattern": "*-!:aaa-!:*",
        "comment": "Meta-substituted aromatic ring: two substituents separated by 2 aromatic bonds.",
    },
    "Para": {
        "pattern": "*-!:aaaa-!:*",
        "comment": "Para-substituted aromatic ring: two substituents separated by 3 aromatic bonds.",
    },
    "Acylic-bonds": {
        "pattern": "*!@*",
        "comment": "Any non-ring bond (at least one endpoint is not in a ring).",
    },
    "Single bond and not in a ring": {
        "pattern": "*-!@*",
        "comment": "Single, non-ring bond between two atoms.",
    },
    "Non-ring atom": {
        "pattern": "[!R]",
        "comment": "Atom that is not a member of any ring.",
    },
    "Ring atom": {
        "pattern": "[R]",
        "comment": "Atom that is a member of at least one ring.",
    },
    "Macrocycle groups": {
        "pattern": "[r;!r3;!r4;!r5;!r6;!r7]",
        "comment": "Atoms belonging to rings larger than 7 members (macrocyclic environment).",
    },
    "S in aromatic 5-ring with lone pair": {
        "pattern": "[sX2r5]",
        "comment": "Divalent sulfur in a 5-membered aromatic ring (thiophene-like S).",
    },
    "Aromatic 5-Ring O with Lone Pair": {
        "pattern": "[oX2r5]",
        "comment": "Divalent oxygen in a 5-membered aromatic ring (furan-like O).",
    },
    "Spiro-ring center": {
        "pattern": "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",
        "comment": "Spiro centre joining two 4-6 membered rings (tetra-coordinated spiro atom in two rings).",
    },
    "N in 5-ring arom": {
        "pattern": "[$([nX2r5]:[a-]),$([nX2r5]:[a]:[a-])]",
        "comment": "Anionic sp2 nitrogen in a 5-membered aromatic ring (e.g. deprotonated azoles).",
    },

    "CIS or TRANS double bond in a ring": {
        "pattern": "*/,\\[R]=;@[R]/,\\*",
        "comment": "Isomeric double bond inside a ring with explicit cis/trans stereochemistry.",
    },
    "CIS or TRANS double or aromatic bond in a ring": {
        "pattern": "*/,\\[R]=,:;@[R]/,\\*",
        "comment": "Isomeric double or aromatic bond in a ring with explicit stereochemical annotation.",
    },

    "Unfused benzene ring": {
        "pattern": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
        "comment": "A benzene ring where each aromatic carbon is only in one ring (non-fused benzene).",
    },
    "Multiple non-fused benzene rings": {
        "pattern": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
        "comment": "Molecules containing at least two non-fused benzene rings.",
    },
    "Fused benzene rings": {
        "pattern": "c12ccccc1cccc2",
        "comment": "Fused benzene ring system (e.g. naphthalene-like).",
    },

    # simple unsaturation
    "Alkene (C=C)": {
        "pattern": "[#6X3]=[#6X3]",
        "comment": "Non-aromatic C=C double bond between two sp2 carbons.",
    },
    "Alkyne (C#C)": {
        "pattern": "[#6X2]#[#6X2]",
        "comment": "C≡C triple bond between two sp carbons.",
    },

    # small strained rings / warheads (structural)
    "Epoxide (3-membered cyclic ether)": {
        "pattern": "[OX2r3]",
        "comment": "Oxygen atom in a 3-membered ring (typically epoxides after normalization).",
    },
    "Aziridine (3-membered cyclic amine)": {
        "pattern": "[NX3r3]",
        "comment": "Nitrogen atom in a 3-membered ring (aziridines / related small azacycles).",
    },
}


FUNCTIONAL_GROUP_PATTERNS: Dict[str, Dict[str, str]] = {
    # carbonyl core
    "Carbonyl group": {
        "pattern": "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "comment": "C=O carbonyl group, including its zwitterionic depiction (C(+)-O(-)).",
    },
    "Aldehyde": {
        "pattern": "[CX3H1](=O)[#6]",
        "comment": "Aldehyde: -CHO attached to a carbon (R-CHO).",
    },
    "Amide": {
        "pattern": "[NX3][CX3](=[OX1])[#6]",
        "comment": "Amide: N-C(=O)-C core (secondary/tertiary amides).",
    },
    "Carbamate": {
        "pattern": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "comment": "Carbamate: N-C(=O)-O motif (carbamic esters/acids/zwitterions).",
    },
    "Carboxylate Ion": {
        "pattern": "[CX3](=O)[O-]",
        "comment": "Carboxylate: conjugate base of carboxylic/carbonic/carbamic acids.",
    },
    "Carbonic Acid or Carbonic Ester": {
        "pattern": "[CX3](=[OX1])(O)O",
        "comment": "Carbonic acid / carbonic ester type: C(=O)(O)O.",
    },
    "Carboxylic acid": {
        "pattern": "[CX3](=O)[OX1H0-,OX2H1]",
        "comment": "Carboxylic acid or its conjugate base: C(=O)-OH / C(=O)-O(-).",
    },
    "Ester Also hits anhydrides": {
        "pattern": "[#6][CX3](=O)[OX2H0][#6]",
        "comment": "Simple ester R-C(=O)-O-R'; does not generally capture anhydrides.",
    },
    "Ketone": {
        "pattern": "[#6][CX3](=O)[#6]",
        "comment": "Ketone: C=O flanked by two carbons (R-C(=O)-R').",
    },

    # urea / guanidine / amidine families
    "Urea": {
        "pattern": "[NX3,NX4+][CX3](=[OX1])[NX3,NX4+]",
        "comment": "Urea-like: C(=O) flanked by two nitrogens (incl. substituted/charged).",
    },
    "Thiourea": {
        "pattern": "[NX3,NX4+][CX3](=[SX1])[NX3,NX4+]",
        "comment": "Thiourea: C(=S) flanked by two nitrogens.",
    },
    "Guanidine": {
        "pattern": "[NX3,NX4+][CX3](=[NX3,NX4+])[NX3,NX4+]",
        "comment": "Guanidine/guanidinium core: N-C(=N)-N, often highly basic/cationic.",
    },
    "Amidine": {
        "pattern": "[NX3][CX3]=[NX2]",
        "comment": "Amidine: C(=NR)-NH motif in neutral form.",
    },
    "Amidinium": {
        "pattern": "[NX3][CX3]=[NX3+]",
        "comment": "Protonated amidine (amidinium cation).",
    },

    # ether
    "Ether": {
        "pattern": "[OD2]([#6])[#6]",
        "comment": "Simple dialkyl/aryl ether: R-O-R'.",
    },

    # hydrogen/charge patterns
    "Mono-Hydrogenated Cation": {
        "pattern": "[+H]",
        "comment": "Any atom with a positive charge and exactly one attached hydrogen.",
    },
    "Not Mono-Hydrogenated": {
        "pattern": "[!H1]",
        "comment": "Atoms that do not have exactly one attached hydrogen (broad detector).",
    },

    # amide-related N
    "Cyanamide": {
        "pattern": "[NX3][CX2]#[NX1]",
        "comment": "Cyanamide: N-C≡N motif.",
    },

    # amine-like / conjugated N
    "Primary or secondary amine, not amide": {
        "pattern": "[NX3;H2,H1;!$(NC=O)]",
        "comment": "Sp3 N with 1-2 H, not directly bound to C=O (includes some cyanamides/thioamides).",
    },
    "Enamine": {
        "pattern": "[NX3][CX3]=[CX3]",
        "comment": "Enamine: N-C=C motif (N attached to an sp2 carbon).",
    },
    "Enamine or Aniline Nitrogen": {
        "pattern": "[NX3][$(C=C),$(cc)]",
        "comment": "Sp3 N attached to a vinyl or aromatic carbon (enamine/aniline-like).",
    },

    # aromatic heterocycles / azoles
    "Azole": {
        "pattern": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "comment": "5-membered aromatic heterocycles with N and another hetero (N/O/S): azole-type rings.",
    },

    # hydrazine / hydrazone
    "Hydrazine H2NNH2": {
        "pattern": "[NX3][NX3]",
        "comment": "Hydrazine-like N-N single bonds (H2NNH2 and substituted analogues).",
    },
    "Hydrazone C=NNH2": {
        "pattern": "[NX3][NX2]=[*]",
        "comment": "Hydrazone-like N-N=C motifs.",
    },

    # imine / iminium
    "Substituted imine": {
        "pattern": "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "comment": "Substituted imine (Schiff base): C=N-R with C bearing two carbon substituents.",
    },
    "Substituted or un-substituted imine": {
        "pattern": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "comment": "Broader imine detector: C=N-R or C=NH with at least one carbon substituent on C.",
    },
    "Iminium": {
        "pattern": "[NX3+]=[CX3]",
        "comment": "Iminium: positively charged C=N+ species.",
    },

    # imide
    "Unsubstituted dicarboximide": {
        "pattern": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "comment": "Dicarboximide with N-H (unsubstituted imide).",
    },
    "Substituted dicarboximide": {
        "pattern": "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "comment": "Dicarboximide with N-alkyl/aryl substitution.",
    },

    # nitrate / nitro / nitrile
    "Nitrate group": {
        "pattern": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "comment": "Nitrate and nitrate-like groups: N(=O)(=O)-O / N(+)(=O)(O(-))-O.",
    },
    "Nitrile": {
        "pattern": "[NX1]#[CX2]",
        "comment": "Nitrile: -C≡N group.",
    },
    "Nitro group": {
        "pattern": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "comment": "Nitro: -NO2 in neutral or zwitterionic depiction, excluding nitrate.",
    },

    # hydroxyl / alcohol / phenol
    "Hydroxyl": {
        "pattern": "[OX2H]",
        "comment": "Hydroxyl group (-OH) attached to any atom.",
    },
    "Hydroxyl in Alcohol": {
        "pattern": "[#6][OX2H]",
        "comment": "Alcohol: C-OH group.",
    },
    "Enol": {
        "pattern": "[OX2H][#6X3]=[#6]",
        "comment": "Enol: -OH attached to an sp2 carbon which is double-bonded to another carbon.",
    },
    "Phenol": {
        "pattern": "[OX2H][cX3]:[c]",
        "comment": "Phenol: -OH attached to an aromatic carbon (Ar-OH).",
    },

    # thio groups
    "Carbo-Thioester": {
        "pattern": "S([#6])[CX3](=O)[#6]",
        "comment": "Thioester: R-S-C(=O)-R'.",
    },
    "Thio analog of carbonyl": {
        "pattern": "[#6X3](=[SX1])([!N])[!N]",
        "comment": "Thio-carbonyl C=S analogues of C=O that are not thioamides (no N substituents).",
    },
    "Thiol, Sulfide or Disulfide Sulfur": {
        "pattern": "[SX2]",
        "comment": "Divalent sulfur atoms in thiols, sulfides, and disulfides.",
    },
    "Thioamide": {
        "pattern": "[NX3][CX3]=[SX1]",
        "comment": "Thioamide: N-C(=S)- pattern.",
    },

    # sulfide / sulfone / sulfonamide / sulfoxide
    "Sulfide": {
        "pattern": "[#16X2H0]",
        "comment": "Divalent sulfur (excludes thiols); matches sulfides and disulfides.",
    },
    "Mono-sulfide": {
        "pattern": "[#16X2H0][!#16]",
        "comment": "R-S-R' where S is bonded to non-sulfur atoms on both sides (mono-sulfide).",
    },
    "Two Sulfides": {
        "pattern": "[#16X2H0][!#16].[#16X2H0][!#16]",
        "comment": "Molecules containing at least two mono-sulfide motifs.",
    },
    "Sulfone": {
        "pattern": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "comment": "Sulfones and sulfonyl-derived acids/esters: S(=O)2 core.",
    },
    "Sulfonamide": {
        "pattern": "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        "comment": "Sulfonamides: sulfonyl group bound to nitrogen (sulfa-drug motif).",
    },
    "Sulfoxide": {
        "pattern": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "comment": "Sulfoxides / sulfinyl species: S(=O) with two substituents on S.",
    },

    # sulfonic / sulfonate
    "Sulfonic acid or sulfonate": {
        "pattern": "[$([SX4](=[OX1])(=[OX1])[OX2H,OX1-])]",
        "comment": "Strongly acidic sulfonic acids/sulfonates: S(=O)2(OH/ O-).",
    },

    # phosphorous-containing groups
    "Phosphoric or phosphonic acid/ester": {
        "pattern": "[$([PX4](=[OX1])(O)(O)),$([PX4](=[OX1])(O)([OX1-]))]",
        "comment": "Phosphate/phosphonate-like P(=O)(O)(O/O-) cores (acidic P(V) groups).",
    },

    # aromatic N / heterocycles (6-ring)
    "Ring sp2 N (pyridine-like)": {
        "pattern": "[nX2r6]",
        "comment": "Aromatic sp2 nitrogen in a 6-membered ring (pyridine/diazine-like).",
    },
    "Diazine-like ring (two ring Ns in 6-ring)": {
        "pattern": "[nX2r6]1cccc[nX2r6]1",
        "comment": "Approximate diazine motif: 6-membered aromatic ring with two ring nitrogens.",
    },

    # simple unsaturation / warheads
    "Alkene (non-aromatic)": {
        "pattern": "[#6X3]=[#6X3]",
        "comment": "Non-aromatic C=C double bond between sp2 carbons.",
    },
    "Alkyne (non-aromatic)": {
        "pattern": "[#6X2]#[#6X2]",
        "comment": "C≡C triple bond between sp carbons.",
    },
    "Michael acceptor (alpha,beta-unsat. carbonyl)": {
        "pattern": "[CX3]=[CX3]-[CX3](=O)[#6]",
        "comment": "Approximate Michael acceptor: α,β-unsaturated carbonyl (enone-type).",
    },

    # small strained rings / warheads (functional view)
    "Epoxide": {
        "pattern": "[OX2r3][CX4r3][CX4r3]",
        "comment": "Explicit epoxide motif: 3-membered cyclic ether.",
    },
    "Aziridine": {
        "pattern": "[NX3r3][CX4r3][CX4r3]",
        "comment": "Explicit aziridine motif: 3-membered cyclic amine.",
    },

    # halogens
    "Any carbon attached to any halogen": {
        "pattern": "[#6][F,Cl,Br,I]",
        "comment": "Carbons directly bonded to F, Cl, Br, or I.",
    },
    "Halogen": {
        "pattern": "[F,Cl,Br,I]",
        "comment": "Halogen atoms (F, Cl, Br, I).",
    },
    "Three_halides groups": {
        "pattern": "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
        "comment": "Molecules containing at least three halogen atoms (polyhalogenated flag).",
    },
}
