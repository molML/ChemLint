from collections import Counter
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd


def _is_invalid_smiles(smi) -> bool:
    """Check if SMILES is None, NaN, or otherwise invalid."""
    if smi is None:
        return True
    # Check for pandas NA, numpy NaN, or float NaN
    if pd.isna(smi):
        return True
    # Check if it's not a string
    if not isinstance(smi, str):
        return True
    return False


def _get_scaffold(smiles: str, scaffold_type: str = 'bemis_murcko') -> tuple[str | None, str]:
    """ Get the molecular scaffold from a SMILES string.

    :param smiles: SMILES string
    :param scaffold_type: 'bemis_murcko' (rings + linkers), 'generic' (skeleton), 'cyclic_skeleton' (skeleton without sidechains)
    :return: Tuple of (scaffold SMILES or None, comment)
    """
    all_scaffs = ['bemis_murcko', 'generic', 'cyclic_skeleton']
    if scaffold_type not in all_scaffs:
        return None, f"Failed: scaffold_type='{scaffold_type}' is not supported. Pick from: {all_scaffs}"

    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smiles):
        return None, "Skipped: Invalid SMILES string"

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        # designed to match atoms that are doubly bonded to another atom.
        PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
        # replacement SMARTS (matches any atom)
        REPL = Chem.MolFromSmarts("[*]")

        Chem.RemoveStereochemistry(mol)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold_type == 'bemis_murcko':
            pass  # scaffold already set

        elif scaffold_type == 'generic':
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

        elif scaffold_type == 'cyclic_skeleton':
            scaffold = AllChem.ReplaceSubstructs(scaffold, PATT, REPL, replaceAll=True)[0]
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)

        # Convert to SMILES and validate
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None, "Failed: No scaffold found (molecule may lack ring systems)"
        
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        
        # Validate output SMILES
        if Chem.MolFromSmiles(scaffold_smiles) is None:
            return None, "Failed: Generated invalid scaffold SMILES"
        
        return scaffold_smiles, "Passed"
    
    except Exception as e:
        return None, f"Failed: {str(e)}"


def calculate_scaffolds(smiles: list[str], scaffold_type: str = 'bemis_murcko') -> tuple[list[str], list[str]]:
    """
    Calculate molecular scaffolds for a list of SMILES strings.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings.
    scaffold_type : str, default='bemis_murcko'
        Type: 'bemis_murcko' (rings + linkers), 'generic' (skeleton), 'cyclic_skeleton' (skeleton without sidechains).
    
    Returns
    -------
    tuple[list[str], list[str]]
        scaffolds: Scaffold SMILES (None if failed); comments: status messages ("Passed" or "Failed: <reason>").
    """
    scaffold_list, comment_list = [], []
    for smi in smiles:
        scaffold, comment = _get_scaffold(smi, scaffold_type)
        scaffold_list.append(scaffold)
        comment_list.append(comment)
    
    return scaffold_list, comment_list


def calculate_scaffolds_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    scaffold_type: str = 'bemis_murcko',
    explanation: str = "Calculate molecular scaffolds"
) -> dict:
    """
    Calculate molecular scaffolds for all SMILES strings in a dataset column.
    
    Parameters
    ----------
    input_filename : str
        Input CSV filename from manifest.
    column_name : str
        Column containing SMILES strings.
    project_manifest_path : str
        Path to manifest.json.
    output_filename : str
        Base name for output file.
    scaffold_type : str, default='bemis_murcko'
        Type: 'bemis_murcko' (rings + linkers), 'generic' (skeleton), 'cyclic_skeleton' (skeleton without sidechains).
    explanation : str
        Description for manifest logging.
    
    Returns
    -------
    dict
        Contains output_filename, n_rows, columns, comments (dict with counts), n_scaffolds_found,
        n_no_scaffold, scaffold_type, preview, and note.
    
    Raises
    ------
    ValueError
        If column_name not found in dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'scaffold_{scaffold_type}': Contains the scaffold SMILES strings or None.
    - 'scaffold_comments': Contains any comments or warnings from the extraction process.
    
    Molecules without ring systems (e.g., aliphatic chains) will have None in the 
    scaffold column with comment "Failed: No scaffold found (molecule may lack ring systems)".
    
    See Also
    --------
    calculate_scaffolds : For processing a list of SMILES strings
    _get_scaffold : Low-level helper function for single SMILES
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    scaffolds, comments = calculate_scaffolds(smiles_list, scaffold_type)

    df[f'scaffold_{scaffold_type}'] = scaffolds
    df['scaffold_comments'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    # Count successes and failures
    n_scaffolds_found = sum(1 for s in scaffolds if s is not None)
    n_no_scaffold = sum(1 for c in comments if 'No scaffold found' in c)

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "n_scaffolds_found": n_scaffolds_found,
        "n_no_scaffold": n_no_scaffold,
        "scaffold_type": scaffold_type,
        "preview": df.head(5).to_dict(orient="records"),
        "note": f"Scaffold column: 'scaffold_{scaffold_type}'. Successful extraction is marked by 'Passed' in scaffold_comments, failure is marked by 'Failed: <reason>'.",
    }


def get_all_scaffold_tools():
    """Return a list of all molecular scaffold tools."""
    return [
        calculate_scaffolds,
        calculate_scaffolds_dataset,
    ]
