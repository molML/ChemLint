from rdkit.Chem import Descriptors, MolFromSmiles
from chemlint.infrastructure.resources import _load_resource, _store_resource
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional



DESCRIPTOR_REGISTRY: dict[str, callable] = {
    name: fn for name, fn in Descriptors._descList
}

def list_rdkit_descriptors() -> list[dict]:
    """
    List all available RDKit 2D molecular descriptors.
    
    Returns
    -------
    list[dict]
        Dicts with 'name' (descriptor function name) and 'explanation' (brief description).
    """
    out = []
    for name, fn in Descriptors._descList:
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        out.append({
            "descriptor name": name,
            "explanation": doc,
        })
    return out


def _smiles_to_descriptors(smiles: list[str], descriptor_names: list[str]) -> dict[str, np.ndarray]:
    """Generate RDKit descriptor vectors from SMILES strings.
    
    Args:
        smiles: List of SMILES strings
        descriptor_names: List of RDKit descriptor names to calculate
    
    Returns:
        Dict mapping SMILES to numpy arrays of shape (n_descriptors,)
    
    Raises:
        ValueError: If invalid SMILES or descriptor calculation fails
    """
    feature_vectors = {}
    
    for smi in smiles:
        mol = MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES string: {smi}")
        
        # Calculate all descriptors for this molecule
        descriptor_values = []
        for desc_name in descriptor_names:
            try:
                value = DESCRIPTOR_REGISTRY[desc_name](mol)
                descriptor_values.append(value)
            except Exception as e:
                raise ValueError(f"Failed to calculate descriptor '{desc_name}' for SMILES '{smi}': {str(e)}")
        
        feature_vectors[smi] = np.array(descriptor_values, dtype=np.float32)
    
    return feature_vectors


def calculate_descriptor_vectors(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    descriptor_names: list[str],
    output_filename: str,
    explanation: str = 'Calculate RDKit descriptor vectors'
) -> dict:
    """Calculate RDKit molecular descriptors as feature vectors for machine learning.
    
    This function creates feature vectors (numpy arrays) suitable for ML model training,
    unlike calculate_simple_descriptors() which adds columns to a CSV for analysis.
    
    Common descriptors:
    - TPSA: Topological Polar Surface Area (Å²)
    - MolWt: Molecular weight (g/mol)
    - MolLogP: Octanol-water partition coefficient (log P)
    - NumHDonors: Number of hydrogen bond donors
    - NumHAcceptors: Number of hydrogen bond acceptors
    - HeavyAtomCount: Number of heavy (non-hydrogen) atoms
    - NumRotatableBonds: Number of rotatable bonds
    - NumAromaticRings: Number of aromatic rings
    - RingCount: Total number of rings
    - FractionCSP3: Fraction of sp³ hybridized carbons
    - qed: Quantitative Estimate of Drug-likeness
    
    Args:
        input_filename: Input dataset resource filename
        project_manifest_path: Path to manifest.json
        smiles_column: Column name containing SMILES strings
        descriptor_names: List of RDKit descriptor names. Use list_rdkit_descriptors() to see all 217 available names
        output_filename: Name for output resource (without extension)
        explanation: Description of this operation
    
    Returns:
        Dict containing:
            - output_filename: Full filename with unique ID
            - n_molecules: Number of molecules processed
            - descriptor_names: List of descriptors calculated
            - n_descriptors: Number of descriptors
    
    Raises:
        ValueError: If SMILES column not found or invalid descriptor names provided
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    # Validate descriptor names
    invalid_descriptors = [name for name in descriptor_names if name not in DESCRIPTOR_REGISTRY]
    if invalid_descriptors:
        available = list(DESCRIPTOR_REGISTRY.keys())[:10]
        raise ValueError(
            f"Invalid descriptor names: {invalid_descriptors}. "
            f"Use list_rdkit_descriptors() to see all available descriptors. "
            f"Examples: {available}"
        )
    
    # Calculate descriptor vectors
    feature_vectors = _smiles_to_descriptors(df[smiles_column].tolist(), descriptor_names)
    
    # Store feature vectors
    output_filename = _store_resource(feature_vectors, project_manifest_path, output_filename, explanation, 'feature_vectors')
    
    return {
        "output_filename": output_filename,
        "n_molecules": len(feature_vectors),
        "descriptor_names": descriptor_names,
        "n_descriptors": len(descriptor_names)
    }

def normalize_feature_vectors(
    train_filename: str,
    project_manifest_path: str,
    train_output_filename: str,
    test_filenames: Optional[list[str]] = None,
    test_output_filenames: Optional[list[str]] = None,
    explanation: str = 'Normalize feature vectors using StandardScaler'
) -> dict:
    """Normalize feature vectors using StandardScaler fitted on training data.
    
    Fits a StandardScaler on the training set and applies the same transformation
    to all provided datasets. This ensures consistent normalization across train/test/validation splits.
    
    Args:
        train_filename: Training feature vectors filename (used to fit the scaler)
        project_manifest_path: Path to manifest.json
        train_output_filename: Output name for normalized training vectors
        test_filenames: Optional list of test/validation feature vector filenames
        test_output_filenames: Optional list of output names for normalized test vectors (must match length of test_filenames)
        explanation: Description of this operation
    
    Returns:
        Dict containing:
            - train_output_filename: Normalized training vectors filename
            - test_output_filenames: List of normalized test vectors filenames (if provided)
            - n_features: Number of features
            - scaler_mean: Mean values used for normalization
            - scaler_std: Standard deviation values used for normalization
    
    Raises:
        ValueError: If test_filenames and test_output_filenames have different lengths
    """
    # Validate inputs
    if test_filenames is None:
        test_filenames = []
    if test_output_filenames is None:
        test_output_filenames = []
    
    if len(test_filenames) != len(test_output_filenames):
        raise ValueError(
            f"test_filenames ({len(test_filenames)}) and test_output_filenames ({len(test_output_filenames)}) "
            "must have the same length"
        )
    
    # Load training feature vectors
    train_vectors = _load_resource(project_manifest_path, train_filename)
    
    # Convert dict of vectors to matrix (preserve SMILES order)
    train_smiles = list(train_vectors.keys())
    train_matrix = np.vstack([train_vectors[smi] for smi in train_smiles])
    
    # Fit scaler on training data
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_matrix)
    
    # Create normalized training dict
    train_normalized_dict = {
        smi: train_normalized[i] for i, smi in enumerate(train_smiles)
    }
    
    # Store normalized training vectors
    train_output = _store_resource(
        train_normalized_dict,
        project_manifest_path,
        train_output_filename,
        f"{explanation} (training set)",
        'feature_vectors'
    )
    
    # Normalize test sets using the same scaler
    test_outputs = []
    for test_file, test_output_name in zip(test_filenames, test_output_filenames):
        test_vectors = _load_resource(project_manifest_path, test_file)
        test_smiles = list(test_vectors.keys())
        test_matrix = np.vstack([test_vectors[smi] for smi in test_smiles])
        
        # Transform using fitted scaler
        test_normalized = scaler.transform(test_matrix)
        
        # Create normalized test dict
        test_normalized_dict = {
            smi: test_normalized[i] for i, smi in enumerate(test_smiles)
        }
        
        # Store normalized test vectors
        test_output = _store_resource(
            test_normalized_dict,
            project_manifest_path,
            test_output_name,
            f"{explanation} (test set)",
            'feature_vectors'
        )
        test_outputs.append(test_output)
    
    return {
        "train_output_filename": train_output,
        "test_output_filenames": test_outputs,
        "n_features": train_matrix.shape[1],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist()
    }


def calculate_simple_descriptors(input_filename: str, smiles_column: str, descriptor_names: list[str], project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Calculate RDKit molecular descriptors for molecules in a dataset.
    
    Common descriptors:
    - TPSA: Topological Polar Surface Area (Å²)
    - MolWt: Molecular weight (g/mol)
    - MolLogP: Octanol-water partition coefficient (log P)
    - NumHDonors: Number of hydrogen bond donors
    - NumHAcceptors: Number of hydrogen bond acceptors
    - HeavyAtomCount: Number of heavy (non-hydrogen) atoms
    - NumRotatableBonds: Number of rotatable bonds
    - NumAromaticRings: Number of aromatic rings
    - RingCount: Total number of rings
    - FractionCSP3: Fraction of sp³ hybridized carbons
    - qed: Quantitative Estimate of Drug-likeness
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    smiles_column : str
        Column containing SMILES strings.
    descriptor_names : list[str]
        RDKit descriptor names to calculate. Use list_rdkit_descriptors() to see all 210+ available names.
    project_manifest_path : str
        Path to manifest.json.
    output_filename : str
        Base name for output file.
    explanation : str
        Description for manifest.
    
    Returns
    -------
    dict
        Contains output_filename, n_rows, columns, descriptors_added, n_failed, note, preview.
    
    Raises
    ------
    ValueError
        If SMILES column not found or invalid descriptor names provided.
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate inputs
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    # Validate descriptor names
    invalid_descriptors = [name for name in descriptor_names if name not in DESCRIPTOR_REGISTRY]
    if invalid_descriptors:
        available = list(DESCRIPTOR_REGISTRY.keys())
        raise ValueError(
            f"Invalid descriptor names: {invalid_descriptors}. "
            f"Use list_rdkit_descriptors() to see all available descriptors. "
            f"Examples: {available}"
        )
    
    # Calculate descriptors for each molecule
    n_failed = {name: 0 for name in descriptor_names}
    descriptor_data = {name: [] for name in descriptor_names}
    
    for smiles in df[smiles_column]:
        mol = MolFromSmiles(smiles) if pd.notna(smiles) else None
        
        for desc_name in descriptor_names:
            try:
                if mol is not None:
                    value = DESCRIPTOR_REGISTRY[desc_name](mol)
                else:
                    value = None
                    n_failed[desc_name] += 1
            except Exception:
                value = None
                n_failed[desc_name]  += 1
            
            descriptor_data[desc_name].append(value)
    
    # Add descriptor columns to dataframe
    for desc_name, values in descriptor_data.items():
        df[desc_name] = values
    
    # Store the updated dataset
    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "descriptors_added": descriptor_names,
        "n_failed": n_failed,
        "note": "If molecules fails a descriptor calculation, a None value is assigned for that descriptor.",
        "preview": df.head(5).to_dict(orient="records"),
    }

