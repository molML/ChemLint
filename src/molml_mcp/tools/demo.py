# server.py
from rdkit.Chem import MolFromSmiles, MolToSmiles


# Canonicalize a SMILES string
@mcp.tool() 
def canonicalize_smiles(smiles: str) -> str: 
    """ Convert a SMILES string to its canonical form. """
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    return MolToSmiles(mol, canonical=True)
