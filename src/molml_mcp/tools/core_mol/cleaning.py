from collections import Counter
from rdkit.Chem import MolFromSmiles, MolToSmiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable
from molml_mcp.tools.core_mol.smiles_ops import _canonicalize_smiles, _remove_pattern, _strip_common_solvent_fragments, _defragment_smiles

from molml_mcp.constants import SMARTS_COMMON_SALTS, SMARTS_COMMON_ISOTOPES, SMARTS_NEUTRALIZATION_PATTERNS, COMMON_SOLVENTS



@loggable
def canonicalize_smiles(smiles: list[str]) -> tuple[list[str], list[str]]:
    """
    Convert SMILES strings to their canonical form.
    
    This function processes a list of SMILES strings and converts each to its 
    canonical representation using RDKit. Canonicalization ensures that equivalent 
    molecular structures have identical SMILES representations, which is essential 
    for deduplication, comparison, and downstream processing.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to canonicalize.
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - canonical_smiles : list[str]
            Canonicalized SMILES strings. Length matches input list.
            Failed conversions return the original SMILES or None.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Canonicalization successful
            - "Failed: <reason>": An error occurred (e.g., invalid SMILES)
    
    Examples
    --------
    # Canonicalize a list of SMILES
    smiles = ["CCO", "C(C)O", "c1ccccc1"]
    canonical, comments = canonicalize_smiles(smiles)
    # Returns: ["CCO", "CCO", "c1ccccc1"], ["Passed", "Passed", "Passed"]
    
    # Invalid SMILES handling
    smiles = ["CCO", "invalid", "c1ccccc1"]
    canonical, comments = canonicalize_smiles(smiles)
    # Returns with "Failed: <reason>" in comments for invalid entry
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Failed conversions are handled gracefully with error messages in comments
    - Canonicalization is idempotent: canonical SMILES remain unchanged
    - Output lists have the same length and order as input list
    
    See Also
    --------
    canonicalize_smiles_dataset : For dataset-level canonicalization
    """
    canonic, comment = [], []
    for smi in smiles:
        c, com = _canonicalize_smiles(smi)
        canonic.append(c)
        comment.append(com)

    return canonic, comment


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
    canonical_smiles, comments = canonicalize_smiles(smiles_list)

    df['smiles_after_canonicalization'] = canonical_smiles
    df['comments_after_canonicalization'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful canonicalization is marked by 'Passed' in comments, failure is marked by 'Failed: <reason>'.",
        "suggestions": "Consider further cleaning steps such as salt removal, tautomer canonicalization, charge neutralization, and stereochemistry handling.",
        "question_to_user": "Would you like to review failed SMILES entries or drop them from the dataset?",
    }


@loggable
def remove_salts(smiles: list[str], salt_smarts: str = SMARTS_COMMON_SALTS) -> tuple[list[str], list[str]]:
    """
    Remove common salt ions from a list of SMILES strings.
    
    This function strips common salt counterions (Cl, Na, Mg, Ca, K, Br, Zn, Ag, Al, 
    Li, I, O, N, H) from molecular structures. It processes a list of SMILES strings
    and returns cleaned versions without salt counterions.
    
    **IMPORTANT**: The default salt pattern works well for most use cases and should 
    typically NOT be changed. Only modify `salt_smarts` if you have specific requirements 
    for a specialized dataset (e.g., organometallic compounds where metals are part of 
    the active structure).
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to process. Each may contain salt counterions.
    salt_smarts : str, optional
        SMARTS pattern defining which atoms/ions to remove.
        **Default: "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]"**
        This default covers common pharmaceutical salts and should be used in most cases.
        Only change this if you have a specific reason (e.g., you want to keep certain 
        ions, or you're working with unusual salt forms).
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - new_smiles : list[str]
            SMILES strings with salts removed. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Salt removal successful
            - "Failed: <reason>": An error occurred (e.g., invalid SMILES)
    
    Examples
    --------
    # Remove common salts (typical usage - don't change salt_smarts)
    smiles = ["CC(=O)O.Na", "c1ccccc1.HCl", "CCO"]
    clean_smiles, comments = remove_salts(smiles)
    # Returns: ["CC(=O)O", "c1ccccc1", "CCO"], ["Passed", "Passed", "Passed"]
    
    # Only change salt_smarts if you have a specific reason:
    # (e.g., removing only chloride and bromide)
    smiles = ["CC(=O)O.Na", "c1ccccc1.Cl"]
    clean_smiles, comments = remove_salts(smiles, salt_smarts="[Cl,Br]")
    # Returns: ["CC(=O)O.Na", "c1ccccc1"], ["Passed", "Passed"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - The default pattern removes the most common pharmaceutical salt counterions
    - If a molecule consists ONLY of salt ions, the result may be empty or fail
    - For complex salt forms, multiple passes may be needed
    - The function preserves the largest fragment if multiple fragments remain after 
      salt removal
    - Output lists have the same length and order as input list
    
    See Also
    --------
    remove_salts_dataset: For dataset-level salt removal
    """
    new_smiles, comments = [], []
    for smi in smiles:
        cleaned_smi, comment = _remove_pattern(smi, salt_smarts)
        new_smiles.append(cleaned_smi)
        comments.append(comment)
    
    return new_smiles, comments


@loggable
def remove_salts_dataset(
    resource_id: str,
    column_name: str,
    salt_smarts: str = SMARTS_COMMON_SALTS
) -> dict:
    """
    Remove common salt ions from SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by removing salt counterions from SMILES 
    strings in the specified column. It adds two new columns to the dataframe: one 
    containing the desalted SMILES and another with comments logged during the salt 
    removal process (e.g., invalid SMILES, processing failures).
    
    **IMPORTANT**: The default salt pattern works well for most use cases and should 
    typically NOT be changed. Only modify `salt_smarts` if you have specific requirements 
    (e.g., organometallic compounds where metals are part of the active structure).
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be desalted.
    salt_smarts : str, optional
        SMARTS pattern defining which atoms/ions to remove.
        **Default: "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]"**
        This default covers common pharmaceutical salts and should be used in most cases.
        Only change this if you have a specific reason.
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with desalted data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            salt removal (e.g., number of successful removals, failures).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the comment system.
        - suggestions : str
            Recommendations for additional cleaning steps that may be beneficial.
        - question_to_user : str
            Question directed at the user/client regarding next steps.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_salt_removal': Contains the desalted SMILES strings.
    - 'comments_after_salt_removal': Contains any comments or warnings from the 
      salt removal process.
    
    Examples
    --------
    # Typical usage with default salt pattern
    result = remove_salts_dataset(resource_id="20251203T120000_csv_ABC123.csv", 
                                   column_name="smiles")
    
    See Also
    --------
    remove_salts : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For dataset-level canonicalization
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    desalted_smiles, comments = remove_salts(smiles_list, salt_smarts)

    df['smiles_after_salt_removal'] = desalted_smiles
    df['comments_after_salt_removal'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful salt removal is marked by 'Passed' in comments, failure is marked by 'Failed: <reason>'.",
        "suggestions": "Consider further cleaning steps such as canonicalization, charge neutralization, and stereochemistry handling.",
        "question_to_user": "Would you like to review failed SMILES entries or drop them from the dataset?",
    }


@loggable
def remove_common_solvents(smiles: list[str]) -> tuple[list[str], list[str]]:
    """
    Remove known common solvent fragments from a list of SMILES strings.
    
    This function processes fragmented SMILES strings and removes fragments that match
    a curated list of common laboratory solvents (e.g., water, ethanol, DMF, DMSO).
    It intelligently handles edge cases where all fragments are solvents or no solvents
    are present.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to process. May contain fragmented SMILES (with '.').
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - new_smiles : list[str]
            SMILES strings with common solvent fragments removed. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Pass": No fragments or no solvents found
            - "Removed solvents": Successfully removed solvent fragments
            - "All fragments are common solvents, kept original SMILES": All fragments 
              were solvents, so original kept (assumed molecule of interest)
            - "SMILES string is fragmented, but found no common solvents": Has fragments 
              but none match the solvent list
    
    Examples
    --------
    # Remove ethanol from benzene
    smiles = ["c1ccccc1.CCO", "CCO", "CC(=O)O.O"]
    clean, comments = remove_common_solvents(smiles)
    # Returns: ["c1ccccc1", "CCO", "CC(=O)O.O"], 
    #          ["Removed solvents", "Pass", "All fragments are common solvents..."]
    
    # Non-solvent fragments
    smiles = ["CC(=O)O.CC(=O)N"]
    clean, comments = remove_common_solvents(smiles)
    # Returns: ["CC(=O)N"], ["Removed solvents"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Only removes fragments that exactly match the common solvent list
    - If all fragments are solvents, keeps the original (assumes it's the target molecule)
    - Single-fragment SMILES are returned unchanged
    - Output lists have the same length and order as input list
    - The solvent list includes common lab solvents: water, alcohols, DMF, DMSO, 
      acetonitrile, pyridine, and many others
    
    See Also
    --------
    remove_common_solvents_dataset : For dataset-level solvent removal
    remove_salts : For removing salt counterions
    """
    new_smiles, comments = [], []
    for smi in smiles:
        cleaned_smi, comment = _strip_common_solvent_fragments(smi)
        new_smiles.append(cleaned_smi)
        comments.append(comment)
    
    return new_smiles, comments


@loggable
def remove_common_solvents_dataset(
    resource_id: str,
    column_name: str
) -> dict:
    """
    Remove known common solvent fragments from SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by removing common laboratory solvent
    fragments from SMILES strings in the specified column. It adds two new columns to 
    the dataframe: one containing the SMILES with solvents removed and another with 
    comments logged during the removal process.
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be processed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with solvent-free data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            solvent removal (e.g., number of successful removals, no changes).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the comment system.
        - suggestions : str
            Recommendations for additional cleaning steps that may be beneficial.
        - question_to_user : str
            Question directed at the user/client regarding next steps.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_solvent_removal': Contains the SMILES with solvent fragments removed.
    - 'comments_after_solvent_removal': Contains any comments or warnings from the 
      removal process.
    
    Examples
    --------
    # Typical usage
    result = remove_common_solvents_dataset(resource_id="20251203T120000_csv_ABC123.csv", 
                                            column_name="smiles")
    
    See Also
    --------
    remove_common_solvents : For processing a list of SMILES strings
    remove_salts_dataset : For dataset-level salt removal
    canonicalize_smiles_dataset : For dataset-level canonicalization
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    cleaned_smiles, comments = remove_common_solvents(smiles_list)

    df['smiles_after_solvent_removal'] = cleaned_smiles
    df['comments_after_solvent_removal'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successfully removed solvents marked by 'Pass, removed solvents' in comments or by 'Pass' when no fragments were found.",
        "suggestions": "Consider further cleaning steps such as salt removal, canonicalization, and charge neutralization.",
        "question_to_user": "Would you like to review SMILES where all fragments were solvents and/or SMILES where fragmented, but no solvents were found?",
    }


@loggable
def defragment_smiles(smiles: list[str], keep_largest_fragment: bool = True) -> tuple[list[str], list[str]]:
    """
    Remove smaller fragments from a list of SMILES strings by keeping only the largest component.
    
    This function processes fragmented SMILES strings (containing '.') and simplifies them 
    by retaining only the largest fragment. This is useful for removing counterions, salt 
    fragments, or unwanted co-crystallized molecules.
    
    **STRONGLY RECOMMENDED**: Remove common solvents BEFORE running this function using 
    `remove_common_solvents()` or `remove_common_solvents_dataset()`. This ensures that 
    you don't accidentally keep a solvent as the "largest fragment" when it happens to be 
    larger than your molecule of interest.
    
    **IMPORTANT LIMITATION**: By default, this function keeps the largest fragment based 
    on SMILES string length, which is NOT bulletproof. String length does not always 
    correspond to molecular weight or atom count. For example, a long SMILES string with 
    many branches might represent a smaller molecule than a compact aromatic system. Use 
    with caution and verify results, especially for complex molecules.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to defragment. May contain fragmented SMILES (with '.').
    keep_largest_fragment : bool, optional
        If True (default), keeps the largest fragment based on SMILES string length.
        If False, returns the original SMILES unchanged if fragments are detected.
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - new_smiles : list[str]
            Defragmented SMILES strings. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Pass": No fragments detected, returned as-is
            - "Pass, kept one instance of repeated fragments": Duplicate fragments found
            - "Pass, defragmented to largest component": Successfully kept largest fragment
            - "Unresolved, contains fragments and keep_largest_fragment is False": 
              Fragments present but keep_largest_fragment was False
            - "Failed: <reason>": An error occurred during processing
    
    Examples
    --------
    # Remove salt fragment (after removing common solvents)
    smiles = ["c1ccccc1.Cl", "CCO", "CC(=O)O.Na"]
    clean, comments = defragment_smiles(smiles)
    # Returns: ["c1ccccc1", "CCO", "CC(=O)O"], 
    #          ["Pass, defragmented to largest component", "Pass", "Pass, defragmented to largest component"]
    
    # Handle repeated fragments
    smiles = ["c1ccccc1.c1ccccc1"]
    clean, comments = defragment_smiles(smiles)
    # Returns: ["c1ccccc1"], ["Pass, kept one instance of repeated fragments"]
    
    # Disable defragmentation
    smiles = ["c1ccccc1.Cl"]
    clean, comments = defragment_smiles(smiles, keep_largest_fragment=False)
    # Returns: ["c1ccccc1.Cl"], ["Unresolved, contains fragments and keep_largest_fragment is False"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - **CRITICAL**: Always run `remove_common_solvents()` first to avoid keeping solvents
    - The "largest fragment" is determined by SMILES string length, which is imperfect:
      * Does not account for implicit hydrogens
      * Does not correlate directly with molecular weight
      * May fail for highly branched vs. compact structures
    - Single-fragment SMILES are returned unchanged with "Pass" status
    - Duplicate fragments are collapsed to a single instance
    - Output lists have the same length and order as input list
    
    Warnings
    --------
    The largest-fragment heuristic based on string length is NOT bulletproof. Always 
    verify that the correct fragment was kept, especially for:
    - Molecules with similar-sized fragments
    - Drug-protein or drug-nucleic acid complexes
    - Coordination complexes where ligands might be larger than metal centers
    
    See Also
    --------
    defragment_smiles_dataset : For dataset-level defragmentation
    remove_common_solvents : Should be run BEFORE this function
    remove_salts : For removing salt counterions using chemical knowledge
    """
    new_smiles, comments = [], []
    for smi in smiles:
        cleaned_smi, comment = _defragment_smiles(smi, keep_largest_fragment)
        new_smiles.append(cleaned_smi)
        comments.append(comment)
    
    return new_smiles, comments


@loggable
def defragment_smiles_dataset(
    resource_id: str,
    column_name: str,
    keep_largest_fragment: bool = True
) -> dict:
    """
    Remove smaller fragments from SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by defragmenting SMILES strings in the 
    specified column. It adds two new columns to the dataframe: one containing the 
    defragmented SMILES and another with comments logged during the defragmentation 
    process.
    
    **STRONGLY RECOMMENDED**: Remove common solvents BEFORE running this function using 
    `remove_common_solvents_dataset()`. This ensures that you don't accidentally keep a 
    solvent as the "largest fragment" when it happens to be larger than your molecule 
    of interest.
    
    **IMPORTANT LIMITATION**: By default, this function keeps the largest fragment based 
    on SMILES string length, which is NOT bulletproof. String length does not always 
    correspond to molecular weight or atom count. For example, a long SMILES string with 
    many branches might represent a smaller molecule than a compact aromatic system. Use 
    with caution and verify results, especially for complex molecules.
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be defragmented.
    keep_largest_fragment : bool, optional
        If True (default), keeps the largest fragment based on SMILES string length.
        If False, returns the original SMILES unchanged if fragments are detected.
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with defragmented data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            defragmentation (e.g., number of successful defragmentations, unresolved).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the comment system.
        - warning : str
            Important warning about the limitations of the largest-fragment heuristic.
        - suggestions : str
            Recommendations for additional cleaning steps that may be beneficial.
        - question_to_user : str
            Question directed at the user/client regarding next steps.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_defragmentation': Contains the defragmented SMILES strings.
    - 'comments_after_defragmentation': Contains any comments or warnings from the 
      defragmentation process.
    
    Warnings
    --------
    The largest-fragment heuristic based on string length is NOT bulletproof. Always 
    verify that the correct fragment was kept, especially for:
    - Molecules with similar-sized fragments
    - Drug-protein or drug-nucleic acid complexes
    - Coordination complexes where ligands might be larger than metal centers
    
    Examples
    --------
    # Typical usage (after removing common solvents)
    # Step 1: Remove solvents first
    result1 = remove_common_solvents_dataset(resource_id="20251203T120000_csv_ABC123.csv", 
                                             column_name="smiles_after_salt_removal")
    # Step 2: Then defragment
    result2 = defragment_smiles_dataset(resource_id=result1["resource_id"], 
                                        column_name="smiles_after_solvent_removal")
    
    See Also
    --------
    defragment_smiles : For processing a list of SMILES strings
    remove_common_solvents_dataset : Should be run BEFORE this function
    remove_salts_dataset : For dataset-level salt removal
    canonicalize_smiles_dataset : For dataset-level canonicalization
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    defragmented_smiles, comments = defragment_smiles(smiles_list, keep_largest_fragment)

    df['smiles_after_defragmentation'] = defragmented_smiles
    df['comments_after_defragmentation'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successfully defragmented SMILES marked by 'Pass' or 'Pass, defragmented to largest component' in comments.",
        "warning": "The largest-fragment heuristic is based on SMILES string length and is NOT bulletproof. Always verify results, especially for complex molecules.",
        "suggestions": "Review entries marked as 'Unresolved' if keep_largest_fragment was False. Consider canonicalization and charge neutralization as next steps.",
        "question_to_user": "Would you like to review defragmented SMILES to verify the correct fragments were kept?",
    }


def get_all_cleaning_tools():
    """Return a list of all molecular cleaning tools."""
    return [
        canonicalize_smiles,
        canonicalize_smiles_dataset,
        remove_salts,
        remove_salts_dataset,
        remove_common_solvents,
        remove_common_solvents_dataset,
        defragment_smiles,
        defragment_smiles_dataset,
    ]







# Tautomer canonicalization
# Charge neutralization
# Stereochemistry handling
# remove duplicates

