from rdkit.Chem import MolFromSmiles, MolToSmiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable


@loggable
def canonicalize_smiles(smiles: list[str]) -> list[str]: 
    """ Convert a SMILES string to its canonical form. Failed conversions are treated as None."""

    canonic = []
    for smi in smiles:
        mol = MolFromSmiles(smi)

        try:
            smi_canon = MolToSmiles(mol, canonical=True)
        except Exception as e:
            smi_canon = None
        
        canonic.append(smi_canon)
        
    return canonic


@loggable
def canonicalize_smiles_dataset(resource_id:str, column_name:str) -> dict:
    """
    Canonicalize all SMILES strings in a specified column of a tabular dataset. 
    A new column of canonicalized SMILES is added to the dataframe.

    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource.
    column_name : str
        Name of the column containing SMILES strings to be canonicalized.

    Returns
    -------
    dict
        Updated dataset information with canonicalized SMILES in the specified column.
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    canonical_smiles = canonicalize_smiles(smiles_list)

    df['canonical_smiles'] = canonical_smiles

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }
