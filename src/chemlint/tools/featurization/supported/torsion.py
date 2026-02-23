import numpy as np
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
from rdkit.DataStructs import ConvertToNumpyArray

from chemlint.infrastructure.resources import _store_resource, _load_resource


def _smiles_to_torsion(smiles: list[str], nbits: int = 2048) -> dict[str, np.ndarray]:
    """Generate Topological Torsion fingerprints from SMILES strings.
    
    Args:
        smiles: List of SMILES strings
        nbits: Fingerprint size in bits (default: 2048)
    
    Returns:
        Dict mapping SMILES to numpy arrays of shape (nbits,)
    """
    ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nbits)

    fingerprints = {}
    for smi in smiles:
        mol = MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES string: {smi}")
        fp = ttgen.GetFingerprint(mol)
        arr = np.zeros((nbits,))
        ConvertToNumpyArray(fp, arr)
        fingerprints[smi] = arr

    return fingerprints


def smiles_to_torsion_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = 'Calculate Topological Torsion fingerprints',
    nbits: int = 2048
) -> dict:
    """Calculate Topological Torsion fingerprints for molecules in a dataset.
    
    Topological Torsion fingerprints encode information about sequences of four connected atoms
    (torsion angles), capturing local chemical environment and conformational information.
    
    Args:
        input_filename: Input dataset resource filename
        project_manifest_path: Path to manifest.json
        smiles_column: Column name containing SMILES strings
        output_filename: Name for output resource (without extension)
        explanation: Description of this operation
        nbits: Fingerprint size in bits (default: 2048)
    
    Returns:
        Dict containing:
            - output_filename: Full filename with unique ID
            - n_molecules: Number of molecules processed
            - fingerprint_type: "TopologicalTorsion"
            - nbits: Fingerprint size
    """
    df = _load_resource(project_manifest_path, input_filename)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    fingerprints = _smiles_to_torsion(df[smiles_column].tolist(), nbits=nbits)

    output_filename = _store_resource(fingerprints, project_manifest_path, output_filename, explanation, 'feature_vectors')

    return {
        "output_filename": output_filename,
        "n_molecules": len(fingerprints),
        "fingerprint_type": "TopologicalTorsion",
        "nbits": nbits
    }
