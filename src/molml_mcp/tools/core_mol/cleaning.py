from collections import Counter
from rdkit.Chem import MolFromSmiles, MolToSmiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable
from molml_mcp.tools.core_mol.smiles_ops import _canonicalize_smiles


@loggable
def canonicalize_smiles_dataset(resource_id:str, column_name:str) -> dict:
    """
    Canonicalize all SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by canonicalizing SMILES strings in the 
    specified column. It adds two new columns to the dataframe: one containing the 
    canonicalized SMILES and another with comments logged during the canonicalization 
    process (e.g., invalid SMILES, conversion failures).
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be canonicalized.
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with canonicalized data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            canonicalization (e.g., number of failed conversions, invalid SMILES).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - suggestions : str
            Recommendations for additional cleaning steps that may be beneficial.
        - question_to_user : str
            Question directed at the user/client regarding next steps for handling 
            problematic entries.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_canonicalization': Contains the canonicalized SMILES strings.
    - 'comments_after_canonicalization': Contains any comments or warnings from the 
      canonicalization process.
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    canonical_smiles, comments = _canonicalize_smiles(smiles_list)

    df['smiles_after_canonicalization'] = canonical_smiles
    df['comments_after_canonicalization'] = comments



    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "suggestions": "Consider further cleaning steps such as salt removal, tautomer canonicalization, charge neutralization, and stereochemistry handling.",
        "question_to_user": "Would you like to review failed SMILES entries or drop them from the dataset?",
    }



# Salt removal

# def _desalt_molecule(self, mol: Chem.Mol) -> Tuple[Chem.Mol, bool]:
    # """Remove salts from molecule."""
    # if '.' not in Chem.MolToSmiles(mol):
    #     return mol, True
    
    # if self.desalt_policy == 'remove':
    #     return None, False
    
    # if self.desalt_policy == 'keep':
    #     mol = self.salt_remover.StripMol(mol, dontRemoveEverything=False)
    #     desalted_smiles = Chem.MolToSmiles(mol)
        
    #     if '.' in desalted_smiles:
    #         if self.brute_force_desalt:
    #             # Keep largest fragment
    #             mol = max(
    #                 Chem.GetMolFrags(mol, asMols=True), 
    #                 key=lambda x: x.GetNumAtoms()
    #             )
    #         else:
    #             return None, False
        
    #     if mol and mol.GetNumAtoms() > 0:
    #         return mol, True
    
    # return None, False

# Tautomer canonicalization
# Charge neutralization
# Stereochemistry handling
# remove duplicates

