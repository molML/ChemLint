import numpy as np
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
from rdkit.DataStructs import ConvertToNumpyArray

from chemlint.infrastructure.resources import _store_resource, _load_resource


def _smiles_to_ecfp(smiles: list[str], radius: int = 2, nbits: int = 2048) -> dict[str, np.ndarray]:
    """Generate ECFP (Morgan) fingerprints from SMILES strings.
    
    Args:
        smiles: List of SMILES strings
        radius: ECFP radius (default: 2)
        nbits: Fingerprint size in bits (default: 2048)
    
    Returns:
        Dict mapping SMILES to numpy arrays of shape (nbits,)
    """

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    fingerprints = {}
    for smi in smiles:
        fp = mfpgen.GetFingerprint(MolFromSmiles(smi))
        arr = np.zeros((nbits,))
        ConvertToNumpyArray(fp, arr)
        fingerprints[smi] = arr

    return fingerprints


def smiles_to_ecfp_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = 'Calculate ECFP fingerprints',
    radius: int = 2,
    nbits: int = 2048
) -> dict:
    """Calculate ECFP (Morgan) fingerprints for molecules in a dataset.
    
    Args:
        input_filename: Input dataset resource filename
        project_manifest_path: Path to manifest.json
        smiles_column: Column name containing SMILES strings
        output_filename: Name for output resource (without extension)
        explanation: Description of this operation
        radius: ECFP radius (default: 2)
        nbits: Fingerprint size in bits (default: 2048)
    
    Returns:
        Dict with output_filename and fingerprint details
    """
    df = _load_resource(project_manifest_path, input_filename)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    fingerprints = _smiles_to_ecfp(df[smiles_column].tolist(), radius=radius, nbits=nbits)

    output_filename = _store_resource(fingerprints, project_manifest_path, output_filename, explanation, 'feature_vectors')

    return {
        "output_filename": output_filename,
        "n_molecules": len(fingerprints),
        "fingerprint_type": "ECFP",
        "radius": radius,
        "nbits": nbits
    }
