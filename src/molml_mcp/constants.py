"""
Constants for molml_mcp package.
"""

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

SMARTS_COMMON_ISOTOPES = [
    '[11c]',
    '[14C]',
    '[10B]',
    '[11C]',
    '[15n]',
    '[14c]',
    '[17F]',
    '[3H]',
    '[18F]',
    '[13C]',
    '[19F]',
    '[18O]',
    '[2H]',
]

SMARTS_COMMON_SALTS = "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]"

