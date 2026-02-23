import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import rdMolDescriptors
from rdkit.Chem import Descriptors

from chemlint.infrastructure.resources import _store_resource, _load_resource


def _smiles_to_rdkit_fp(smiles: list[str], fp_size: int = 2048) -> dict[str, np.ndarray]:
    """Generate RDKit topological fingerprints from SMILES strings.
    
    Args:
        smiles: List of SMILES strings
        fp_size: Fingerprint size in bits (default: 2048)
    
    Returns:
        Dict mapping SMILES to numpy arrays of shape (fp_size,)
    """
    from rdkit.Chem import RDKFingerprint
    from rdkit.DataStructs import ConvertToNumpyArray

    fingerprints = {}
    for smi in smiles:
        mol = MolFromSmiles(smi)
        fp = RDKFingerprint(mol, fpSize=fp_size)
        arr = np.zeros((fp_size,))
        ConvertToNumpyArray(fp, arr)
        fingerprints[smi] = arr

    return fingerprints


def smiles_to_rdkit_fp_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = 'Calculate RDKit topological fingerprints',
    fp_size: int = 2048
) -> dict:
    """Calculate RDKit topological fingerprints for molecules in a dataset.
    
    Args:
        input_filename: Input dataset resource filename
        project_manifest_path: Path to manifest.json
        smiles_column: Column name containing SMILES strings
        output_filename: Name for output resource (without extension)
        explanation: Description of this operation
        fp_size: Fingerprint size in bits (default: 2048)
    
    Returns:
        Dict with output_filename and fingerprint details
    """
    df = _load_resource(project_manifest_path, input_filename)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    fingerprints = _smiles_to_rdkit_fp(df[smiles_column].tolist(), fp_size=fp_size)

    output_filename = _store_resource(fingerprints, project_manifest_path, output_filename, explanation, 'feature_vectors')

    return {
        "output_filename": output_filename,
        "n_molecules": len(fingerprints),
        "fingerprint_type": "RDKit",
        "fp_size": fp_size
    }
