from rdkit.Chem import Descriptors, MolFromSmiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable



DESCRIPTOR_REGISTRY: dict[str, callable] = {
    name: fn for name, fn in Descriptors._descList
}

def list_rdkit_descriptors() -> list[dict]:
    """
    List all available RDKit 2D molecular descriptors.
    
    Returns a list of descriptor metadata including names and descriptions.
    Use this to discover which descriptors are available before computing them
    with calculate_descriptors().
    
    Returns:
        List of dicts with keys:
            - name: Descriptor function name (e.g. "MolWt", "TPSA")
            - explanation: Brief description from the descriptor's docstring
    
    Example return:
        [
            {"name": "MolWt", "explanation": "Molecular weight"},
            {"name": "TPSA", "explanation": "Topological polar surface area"},
            ...
        ]
    """
    out = []
    for name, fn in Descriptors._descList:
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        out.append({
            "descriptor name": name,
            "explanation": doc,
        })
    return out


@loggable
def calculate_descriptors(input_filename: str, smiles_column: str, descriptor_names: list[str], project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Calculate RDKit molecular descriptors for molecules in a dataset.
    
    This function computes the specified RDKit 2D descriptors for all molecules 
    in the given SMILES column and adds each descriptor as a new column to the dataset.
    Invalid SMILES or calculation failures result in None values for that descriptor.
    
    Most commonly used descriptors:
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
        Base filename of the input dataset resource.
    smiles_column : str
        Name of the column containing SMILES strings.
    descriptor_names : list[str]
        List of RDKit descriptor names to calculate.
        Use list_rdkit_descriptors() to see all 210+ available descriptor names.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of what descriptors were calculated.
    
    Returns
    -------
    dict
        {
            "output_filename": str,          # filename for the new dataset
            "n_rows": int,                   # number of rows
            "columns": list[str],            # all column names including new descriptors
            "descriptors_added": list[str],  # names of descriptors successfully added
            "n_failed": int,                 # number of molecules that failed calculation
            "preview": list[dict],           # first 5 rows as records
        }
    
    Raises
    ------
    ValueError
        If the SMILES column is not found or if invalid descriptor names are provided.
    
    Examples
    --------
    # Calculate common drug-like properties
    calculate_descriptors(rid, "smiles", ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"])
    
    # Calculate additional structural properties
    calculate_descriptors(rid, "smiles", ["HeavyAtomCount", "NumRotatableBonds", "RingCount", "qed"])
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

