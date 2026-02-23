import numpy as np
from rdkit.Chem import MolFromSmiles, MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray

from chemlint.infrastructure.resources import _store_resource, _load_resource


def _smiles_to_maccs(smiles: list[str]) -> dict[str, np.ndarray]:
    """Generate MACCS keys fingerprints from SMILES strings.
    
    Args:
        smiles: List of SMILES strings
    
    Returns:
        Dict mapping SMILES to numpy arrays of shape (167,)
        Note: MACCS keys are 167 bits (indices 0-166, where 0 is not used)
    """

    fingerprints = {}
    for smi in smiles:
        mol = MolFromSmiles(smi)
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((fp.GetNumBits(),))
        ConvertToNumpyArray(fp, arr)
        fingerprints[smi] = arr

    return fingerprints


def smiles_to_maccs_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = 'Calculate MACCS keys fingerprints'
) -> dict:
    """Calculate MACCS keys fingerprints for molecules in a dataset.
    
    Args:
        input_filename: Input dataset resource filename
        project_manifest_path: Path to manifest.json
        smiles_column: Column name containing SMILES strings
        output_filename: Name for output resource (without extension)
        explanation: Description of this operation
    
    Returns:
        Dict with output_filename and fingerprint details
    """
    df = _load_resource(project_manifest_path, input_filename)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    fingerprints = _smiles_to_maccs(df[smiles_column].tolist())

    output_filename = _store_resource(fingerprints, project_manifest_path, output_filename, explanation, 'feature_vectors')

    return {
        "output_filename": output_filename,
        "n_molecules": len(fingerprints),
        "fingerprint_type": "MACCS",
        "nbits": 167
    }
