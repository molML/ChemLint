from collections import Counter
from rdkit.Chem import MolFromSmiles, MolToSmiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable
from molml_mcp.tools.core_mol.smiles_ops import _canonicalize_smiles, _remove_pattern, _strip_common_solvent_fragments, _defragment_smiles

from molml_mcp.constants import SMARTS_COMMON_SALTS, SMARTS_NEUTRALIZATION_PATTERNS, COMMON_SOLVENTS



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
    Remove smaller fragments from a list of SMILES strings by keeping only the largest component or unrepeating fragments.
    
    This function processes fragmented SMILES strings (containing '.') and simplifies them. 
    This is useful for removing counterions, salt fragments, or unwanted co-crystallized molecules.
    
    **STRONGLY RECOMMENDED**: Remove common salts and solvents BEFORE running this function using 
    `remove_common_solvents()` or `remove_common_solvents_dataset()` and `remove_common_salts()` 
    or `remove_common_salts_dataset()`. This ensures that you don't accidentally keep a solvent as 
    the "largest fragment" when it happens to be larger than your molecule of interest.
    
    **IMPORTANT LIMITATION**: By default, this function keeps the largest fragment based 
    on SMILES string length, which is NOT bulletproof. Use  with caution and verify results.
    
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
    
    **STRONGLY RECOMMENDED**: Remove common salts and solvents BEFORE running this function using 
    `remove_common_solvents_dataset()` and `remove_common_salts_dataset()`. This ensures that you 
    don't accidentally keep a solvent as the "largest fragment" when it happens to be larger than 
    your molecule of interest.
    
    **IMPORTANT LIMITATION**: By default, this function keeps the largest fragment based 
    on SMILES string length, which is NOT bulletproof. Use with caution and verify results.
    
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


@loggable
def neutralize_smiles(smiles: list[str]) -> tuple[list[str], list[str]]:
    """
    Neutralize charged molecules by converting them to their neutral forms.
    
    This function processes a list of SMILES strings and neutralizes charged functional 
    groups using a set of predefined chemical transformation patterns. Common neutralizations 
    include: protonated amines → neutral amines, carboxylate anions → carboxylic acids, 
    protonated imidazoles → neutral imidazoles, and thiolate anions → thiols.
    
    **IMPORTANT**: The output SMILES are both neutralized AND canonicalized. No additional 
    canonicalization step is needed after running this function, as RDKit's MolToSmiles 
    with canonical=True is automatically applied to the neutralized structures.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to neutralize. May contain charged species.
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - neutralized_smiles : list[str]
            Neutralized AND canonicalized SMILES strings. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Neutralization successful (or no charges found)
            - "Failed: Invalid SMILES string": Could not parse SMILES
            - "Failed: <reason>": An error occurred during neutralization
    
    Examples
    --------
    # Neutralize a protonated amine
    smiles = ["CC[NH3+]"]
    neutral, comments = neutralize_smiles(smiles)
    # Returns: ["CCN"], ["Passed"]
    
    # Neutralize a carboxylate
    smiles = ["CC(=O)[O-]"]
    neutral, comments = neutralize_smiles(smiles)
    # Returns: ["CC(=O)O"], ["Passed"]
    
    # Multiple charged groups
    smiles = ["CC[NH3+].[O-]C(=O)C"]
    neutral, comments = neutralize_smiles(smiles)
    # Returns: ["CCN.CC(=O)O"], ["Passed"]
    
    # Already neutral molecule (no change)
    smiles = ["c1ccccc1"]
    neutral, comments = neutralize_smiles(smiles)
    # Returns: ["c1ccccc1"], ["Passed"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Output is BOTH neutralized AND canonicalized - no additional canonicalization needed
    - Neutralization patterns include:
      * Protonated amines ([N+;!H0]) → neutral amines (N)
      * Protonated imidazoles ([n+;H]) → neutral imidazoles (n)
      * Carboxylates/alkoxides ([O-]) → alcohols/acids (O)
      * Thiolates ([S-]) → thiols (S)
    - The function applies transformations iteratively until no more matches are found
    - Molecules without charged groups are returned unchanged (but still canonicalized)
    - Output lists have the same length and order as input list
    - Based on neutralization patterns adapted from Hans de Winter's RDKit contributions
    
    Warnings
    --------
    - Some charged species are intentional (e.g., quaternary ammonium salts, zwitterions)
      and may lose chemical meaning when neutralized
    - For zwitterionic amino acids or betaines, neutralization may not preserve the 
      original chemical structure appropriately
    - Always verify that neutralization is appropriate for your specific use case
    
    See Also
    --------
    neutralize_smiles : For processing a list of SMILES strings
    canonicalize_smiles : For canonicalization without neutralization
    remove_salts : For removing salt counterions
    """
    from molml_mcp.tools.core_mol.smiles_ops import _initialise_neutralisation_reactions, _neutralize_smiles
    
    neutralization_transformations = _initialise_neutralisation_reactions()

    neutralized_smiles, comments = [], []   
    for smi in smiles:
        new_smi, comment = _neutralize_smiles(smi, neutralization_transformations)
        neutralized_smiles.append(new_smi)
        comments.append(comment)

    return neutralized_smiles, comments


@loggable
def neutralize_smiles_dataset(
    resource_id: str,
    column_name: str
) -> dict:
    """
    Neutralize charged molecules in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by neutralizing charged functional groups 
    in SMILES strings in the specified column. It adds two new columns to the dataframe: 
    one containing the neutralized SMILES and another with comments logged during the 
    neutralization process.
    
    **IMPORTANT**: The output SMILES are both neutralized AND canonicalized. No additional 
    canonicalization step is needed after running this function, as RDKit's MolToSmiles 
    with canonical=True is automatically applied to all neutralized structures.
    
    Common neutralizations include:
    - Protonated amines → neutral amines
    - Carboxylate anions → carboxylic acids
    - Protonated imidazoles → neutral imidazoles
    - Thiolate anions → thiols
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be neutralized.
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with neutralized data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            neutralization (e.g., number of successful neutralizations, failures).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the comment system and canonicalization behavior.
        - warning : str
            Important warnings about cases where neutralization may not be appropriate.
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
    - 'smiles_after_neutralization': Contains the neutralized AND canonicalized SMILES strings.
    - 'comments_after_neutralization': Contains any comments or warnings from the 
      neutralization process.
    
    Neutralization patterns (adapted from Hans de Winter's RDKit contributions):
    - Protonated amines ([N+;!H0]) → neutral amines (N)
    - Protonated imidazoles ([n+;H]) → neutral imidazoles (n)
    - Carboxylates/alkoxides ([O-]) → alcohols/acids (O)
    - Thiolates ([S-]) → thiols (S)
    
    Warnings
    --------
    Some charged species are intentional and may lose chemical meaning when neutralized:
    - Quaternary ammonium salts (e.g., choline, betaine)
    - Zwitterionic amino acids at physiological pH
    - Permanently charged drug molecules (e.g., some muscle relaxants)
    - Ionic liquids where charge is essential to structure
    
    Always verify that neutralization is appropriate for your specific use case.
    
    Examples
    --------
    # Typical usage after salt removal and defragmentation
    result = neutralize_smiles_dataset(resource_id="20251203T120000_csv_ABC123.csv", 
                                       column_name="smiles_after_defragmentation")
    
    # Or as part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(resource_id="initial.csv", column_name="smiles")
    # Step 2: Remove solvents
    result2 = remove_common_solvents_dataset(resource_id=result1["resource_id"], 
                                             column_name="smiles_after_salt_removal")
    # Step 3: Defragment
    result3 = defragment_smiles_dataset(resource_id=result2["resource_id"], 
                                        column_name="smiles_after_solvent_removal")
    # Step 4: Neutralize (already canonicalized, no additional step needed)
    result4 = neutralize_smiles_dataset(resource_id=result3["resource_id"], 
                                        column_name="smiles_after_defragmentation")
    
    See Also
    --------
    neutralize_smiles : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For canonicalization without neutralization
    remove_salts_dataset : For dataset-level salt removal
    defragment_smiles_dataset : For dataset-level defragmentation
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    neutralized_smiles, comments = neutralize_smiles(smiles_list)

    df['smiles_after_neutralization'] = neutralized_smiles
    df['comments_after_neutralization'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful neutralization is marked by 'Passed' in comments. Output SMILES are both neutralized AND canonicalized - no additional canonicalization step is needed.",
        "warning": "Neutralization may not be appropriate for quaternary ammonium salts, zwitterions, or permanently charged drug molecules. Review your dataset to ensure neutralization is chemically meaningful.",
        "suggestions": "Consider reviewing molecules that failed neutralization. You may also want to perform tautomer canonicalization or stereochemistry handling as next steps.",
        "question_to_user": "Would you like to review failed neutralizations or molecules with specific charge states before proceeding?",
    }


@loggable
def standardize_stereochemistry(
    smiles: list[str],
    stereo_policy: str = "keep",
    assign_policy: str = "first",
    max_isomers: int = 32,
    try_embedding: bool = False,
    only_unassigned: bool = True,
    random_seed: int = 42
) -> tuple[list[str], list[str]]:
    """
    Standardize stereochemistry in a list of SMILES strings with flexible handling options.
    
    This function provides comprehensive stereochemistry handling with three policies:
    keep (preserve existing), assign (enumerate and select), or flatten (remove all).
    This is useful for standardizing molecular representations based on your specific 
    analysis requirements.
    
    **IMPORTANT**: The output SMILES are always canonicalized. No additional 
    canonicalization step is needed after running this function.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to process. May contain stereochemical information.
    stereo_policy : str, optional
        Policy for handling stereochemistry. Options:
        - "keep" (default): Preserve existing stereochemistry, canonicalize
        - "assign": Enumerate stereoisomers and select one based on assign_policy
        - "flatten": Remove all stereochemistry (chiral centers + E/Z bonds)
    assign_policy : str, optional
        When stereo_policy="assign", how to select from enumerated isomers:
        - "first" (default): Select first enumerated stereoisomer
        - "random": Randomly select one stereoisomer
        - "lowest": Select stereoisomer with lowest MMFF94/UFF energy
    max_isomers : int, optional
        Maximum number of stereoisomers to enumerate (default: 32)
    try_embedding : bool, optional
        Try 3D embedding to prune degenerate stereoisomers (default: False)
    only_unassigned : bool, optional
        Only enumerate unassigned stereocenters if True (default: True)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - standardized_smiles : list[str]
            SMILES strings with standardized stereochemistry. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Standardization successful
            - "Failed: Invalid SMILES string": Could not parse SMILES
            - "Failed: <reason>": An error occurred during processing
    
    Examples
    --------
    # Keep existing stereochemistry
    smiles = ["C[C@H](O)CC", "C[C@@H](O)CC"]
    std, comments = standardize_stereochemistry(smiles, stereo_policy="keep")
    # Returns: ["C[C@H](O)CC", "C[C@@H](O)CC"], ["Passed", "Passed"]
    
    # Flatten all stereochemistry
    smiles = ["C[C@H](O)CC", "C[C@@H](O)CC"]
    std, comments = standardize_stereochemistry(smiles, stereo_policy="flatten")
    # Returns: ["CC(O)CC", "CC(O)CC"], ["Passed", "Passed"]
    # Note: Both enantiomers become identical after flattening
    
    # Assign stereochemistry to unspecified centers
    smiles = ["CC(O)C(=O)O"]
    std, comments = standardize_stereochemistry(smiles, stereo_policy="assign", assign_policy="first")
    # Returns: ["C[C@H](O)C(=O)O"], ["Passed"]
    
    # Assign with energy-based selection
    smiles = ["CC(O)C(N)C"]
    std, comments = standardize_stereochemistry(smiles, stereo_policy="assign", assign_policy="lowest")
    # Returns lowest-energy stereoisomer
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Output is ALWAYS canonicalized - no additional canonicalization needed
    - Three stereo_policy options:
      * "keep": Preserves existing stereochemistry (@ and / symbols)
      * "assign": Enumerates and selects stereoisomers for unspecified centers
      * "flatten": Removes all stereochemistry completely
    - When using "assign" with "lowest", conformer energy calculation may be slow
    - Output lists have the same length and order as input list
    
    Warnings
    --------
    - Flattening: Loss of stereochemical information may not be appropriate for:
      * Drug molecules where stereochemistry affects activity
      * Structure-activity relationship studies
    - Assigning: Automated stereoisomer selection may not reflect biological relevance
    - This operation modifies molecular representation - verify appropriateness for your use case
    
    See Also
    --------
    standardize_stereochemistry_dataset : For dataset-level stereochemistry standardization
    canonicalize_smiles : For canonicalization that preserves stereochemistry
    """
    from molml_mcp.tools.core_mol.smiles_ops import _standardize_stereo_smiles
    
    standardized_smiles, comments = [], []
    for smi in smiles:
        std_smi, comment = standardize_stereo_smiles(
            smi,
            stereo_policy=stereo_policy,
            assign_policy=assign_policy,
            max_isomers=max_isomers,
            try_embedding=try_embedding,
            only_unassigned=only_unassigned,
            random_seed=random_seed
        )
        standardized_smiles.append(std_smi)
        comments.append(comment)
    
    return standardized_smiles, comments


@loggable
def standardize_stereochemistry_dataset(
    resource_id: str,
    column_name: str,
    stereo_policy: str = "keep",
    assign_policy: str = "first",
    max_isomers: int = 32,
    try_embedding: bool = False,
    only_unassigned: bool = True,
    random_seed: int = 42
) -> dict:
    """
    Standardize stereochemistry in SMILES strings in a specified column of a tabular dataset.
    
    This function provides comprehensive stereochemistry handling with three policies:
    keep (preserve existing), assign (enumerate and select), or flatten (remove all).
    It adds two new columns to the dataframe: one containing the standardized SMILES 
    and another with comments logged during the standardization process.
    
    **IMPORTANT**: The output SMILES are always canonicalized. No additional 
    canonicalization step is needed after running this function.
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be standardized.
    stereo_policy : str, optional
        Policy for handling stereochemistry. Options:
        - "keep" (default): Preserve existing stereochemistry, canonicalize
        - "assign": Enumerate stereoisomers and select one based on assign_policy
        - "flatten": Remove all stereochemistry (chiral centers + E/Z bonds)
    assign_policy : str, optional
        When stereo_policy="assign", how to select from enumerated isomers:
        - "first" (default): Select first enumerated stereoisomer
        - "random": Randomly select one stereoisomer
        - "lowest": Select stereoisomer with lowest MMFF94/UFF energy
    max_isomers : int, optional
        Maximum number of stereoisomers to enumerate (default: 32)
    try_embedding : bool, optional
        Try 3D embedding to prune degenerate stereoisomers (default: False)
    only_unassigned : bool, optional
        Only enumerate unassigned stereocenters if True (default: True)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with standardized data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            stereochemistry standardization (e.g., number of successful operations, failures).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the operation and canonicalization behavior.
        - suggestions : str
            Recommendations for next steps.
        - question_to_user : str
            Question directed at the user/client regarding stereoisomer handling.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_stereo_standardization': Contains the standardized SMILES, canonicalized.
    - 'comments_after_stereo_standardization': Contains any comments or warnings from the 
      standardization process.
    
    Three stereo_policy options:
    - "keep": Preserves existing stereochemistry (@ and / symbols)
    - "assign": Enumerates and selects stereoisomers for unspecified centers
    - "flatten": Removes all stereochemistry (chiral centers + E/Z bonds)
    
    Warnings
    --------
    - Flattening: Loss of stereochemical information may impact drug activity analysis
    - Assigning: Automated selection may not reflect biological relevance
    - Energy-based selection (assign_policy="lowest") can be computationally expensive
    
    Examples
    --------
    # Keep existing stereochemistry (default)
    result = standardize_stereochemistry_dataset(
        resource_id="20251204T120000_csv_ABC123.csv",
        column_name="smiles_after_neutralization",
        stereo_policy="keep"
    )
    
    # Flatten all stereochemistry
    result = standardize_stereochemistry_dataset(
        resource_id="20251204T120000_csv_ABC123.csv",
        column_name="smiles_after_neutralization",
        stereo_policy="flatten"
    )
    
    # Assign stereochemistry with random selection
    result = standardize_stereochemistry_dataset(
        resource_id="20251204T120000_csv_ABC123.csv",
        column_name="smiles_after_neutralization",
        stereo_policy="assign",
        assign_policy="random",
        random_seed=42
    )
    
    # As part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(resource_id="initial.csv", column_name="smiles")
    # Step 2: Neutralize
    result2 = neutralize_smiles_dataset(resource_id=result1["resource_id"], 
                                        column_name="smiles_after_salt_removal")
    # Step 3: Standardize stereochemistry
    result3 = standardize_stereochemistry_dataset(
        resource_id=result2["resource_id"], 
        column_name="smiles_after_neutralization",
        stereo_policy="flatten"  # or "keep" or "assign"
    )
    
    See Also
    --------
    standardize_stereochemistry : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For canonicalization that preserves stereochemistry
    neutralize_smiles_dataset : For charge neutralization
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    standardized_smiles, comments = standardize_stereochemistry(
        smiles_list,
        stereo_policy=stereo_policy,
        assign_policy=assign_policy,
        max_isomers=max_isomers,
        try_embedding=try_embedding,
        only_unassigned=only_unassigned,
        random_seed=random_seed
    )

    df['smiles_after_stereo_standardization'] = standardized_smiles
    df['comments_after_stereo_standardization'] = comments

    new_resource_id = _store_resource(df, 'csv')

    policy_notes = {
        "keep": "Existing stereochemistry has been preserved and canonicalized.",
        "assign": f"Stereoisomers have been enumerated and selected using '{assign_policy}' policy.",
        "flatten": "All stereochemical information has been removed. Stereoisomers are now indistinguishable."
    }

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": f"Successful standardization is marked by 'Passed' in comments. Output SMILES are canonicalized. Policy used: '{stereo_policy}'. {policy_notes.get(stereo_policy, '')}",
        "suggestions": "Review molecules that failed standardization. If using 'flatten' policy, consider deduplicating the dataset as stereoisomers are now identical. If using 'assign', verify that the automated selection is appropriate for your use case.",
        "question_to_user": f"You used stereo_policy='{stereo_policy}'. Is this appropriate for your analysis? Would you like to review the results or try a different policy?",
    }


@loggable
def remove_isotopes(smiles: list[str]) -> tuple[list[str], list[str]]:
    """
    Remove isotopic labels from a list of SMILES strings.
    
    This function processes a list of SMILES strings and removes all isotopic labels,
    converting isotopically-labeled atoms to their default (most common) isotope forms.
    For example, deuterium ([2H]), carbon-13 ([13C]), and fluorine-18 ([18F]) are 
    converted to their standard forms.
    
    **IMPORTANT**: The output SMILES are de-isotoped AND canonicalized. No additional 
    canonicalization step is needed after running this function, as RDKit's MolToSmiles 
    with canonical=True is automatically applied.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to process. May contain isotopic labels.
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - clean_smiles : list[str]
            SMILES strings without isotopic labels, canonicalized. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Isotope removal successful (or no isotopes present)
            - "Failed: Invalid SMILES string": Could not parse SMILES
            - "Failed: <reason>": An error occurred during processing
    
    Examples
    --------
    # Remove carbon-13 and fluorine-18 labels
    smiles = ["[13CH3][18F]"]
    clean, comments = remove_isotopes(smiles)
    # Returns: ["CCF"], ["Passed"]
    
    # Remove deuterium label
    smiles = ["CC([2H])O", "CCO"]
    clean, comments = remove_isotopes(smiles)
    # Returns: ["CCO", "CCO"], ["Passed", "Passed"]
    
    # Mixed isotope labels
    smiles = ["[13C]C([2H])([2H])[18O]"]
    clean, comments = remove_isotopes(smiles)
    # Returns: ["CCO"], ["Passed"]
    
    # Already unlabeled molecule (no change)
    smiles = ["c1ccccc1", "CCO"]
    clean, comments = remove_isotopes(smiles)
    # Returns: ["c1ccccc1", "CCO"], ["Passed", "Passed"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Output is BOTH de-isotoped AND canonicalized - no additional canonicalization needed
    - Removes ALL isotopic labels:
      * Deuterium ([2H]) → hydrogen (H)
      * Tritium ([3H]) → hydrogen (H)
      * Carbon-13 ([13C]) → carbon-12 (C)
      * Carbon-14 ([14C]) → carbon-12 (C)
      * Nitrogen-15 ([15N]) → nitrogen-14 (N)
      * Oxygen-18 ([18O]) → oxygen-16 (O)
      * Fluorine-18 ([18F]) → fluorine-19 (F)
      * And all other isotopic variants
    - Molecules without isotopic labels are returned unchanged (but canonicalized)
    - Stereochemistry is preserved during isotope removal
    - Output lists have the same length and order as input list
    
    Warnings
    --------
    - Loss of isotopic information may not be appropriate for all applications:
      * Radiolabeled compounds used in pharmacokinetic studies
      * NMR spectroscopy experiments requiring specific isotope labels
      * Mass spectrometry studies tracking isotope incorporation
      * Metabolic flux analysis using isotope tracers
    - This operation is irreversible - isotope labels cannot be recovered
    
    See Also
    --------
    remove_isotopes_dataset : For dataset-level isotope removal
    canonicalize_smiles : For canonicalization without isotope removal
    """
    from molml_mcp.tools.core_mol.smiles_ops import _remove_isotopes
    
    clean_smiles, comments = [], []
    for smi in smiles:
        clean_smi, comment = _remove_isotopes(smi)
        clean_smiles.append(clean_smi)
        comments.append(comment)
    
    return clean_smiles, comments


@loggable
def remove_isotopes_dataset(
    resource_id: str,
    column_name: str
) -> dict:
    """
    Remove isotopic labels from SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by removing all isotopic labels from SMILES 
    strings in the specified column. It adds two new columns to the dataframe: one 
    containing the de-isotoped SMILES and another with comments logged during the removal 
    process.
    
    **IMPORTANT**: The output SMILES are de-isotoped AND canonicalized. No additional 
    canonicalization step is needed after running this function, as RDKit's MolToSmiles 
    with canonical=True is automatically applied.
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be de-isotoped.
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with de-isotoped data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            isotope removal (e.g., number of successful operations, failures).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the operation and canonicalization behavior.
        - warning : str
            Important warnings about loss of isotopic information.
        - suggestions : str
            Recommendations for next steps.
        - question_to_user : str
            Question directed at the user/client regarding isotope handling.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_isotope_removal': Contains the SMILES without isotopic labels, canonicalized.
    - 'comments_after_isotope_removal': Contains any comments or warnings from the 
      removal process.
    
    All isotopic labels are removed:
    - Deuterium ([2H]), tritium ([3H])
    - Carbon-13 ([13C]), carbon-14 ([14C])
    - Nitrogen-15 ([15N])
    - Oxygen-18 ([18O])
    - Fluorine-18 ([18F])
    - All other isotopic variants
    
    Stereochemistry is preserved during isotope removal.
    
    Warnings
    --------
    Loss of isotopic information may significantly impact certain analyses:
    - Radiolabeling studies (e.g., PET tracers with [18F])
    - NMR experiments requiring deuterium or carbon-13 labels
    - Mass spectrometry isotope tracing experiments
    - Metabolic flux analysis using stable isotopes
    - This operation is irreversible
    
    Examples
    --------
    # Typical usage when isotope labels are not relevant
    result = remove_isotopes_dataset(resource_id="20251204T120000_csv_ABC123.csv", 
                                     column_name="smiles_after_neutralization")
    
    # As part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(resource_id="initial.csv", column_name="smiles")
    # Step 2: Neutralize
    result2 = neutralize_smiles_dataset(resource_id=result1["resource_id"], 
                                        column_name="smiles_after_salt_removal")
    # Step 3: Remove isotopes if not needed
    result3 = remove_isotopes_dataset(resource_id=result2["resource_id"], 
                                      column_name="smiles_after_neutralization")
    
    See Also
    --------
    remove_isotopes : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For canonicalization without isotope removal
    neutralize_smiles_dataset : For charge neutralization
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    clean_smiles, comments = remove_isotopes(smiles_list)

    df['smiles_after_isotope_removal'] = clean_smiles
    df['comments_after_isotope_removal'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful isotope removal is marked by 'Passed' in comments. Output SMILES are both de-isotoped AND canonicalized - no additional canonicalization step is needed.",
        "warning": "All isotopic labels have been removed. This may affect studies involving radiolabeling, NMR spectroscopy, or isotope tracing experiments. This operation is irreversible.",
        "suggestions": "Review molecules that failed isotope removal. Consider whether this affects your analysis if working with radiolabeled compounds or isotope tracing studies.",
        "question_to_user": "Are you working with any isotope-labeled compounds that require special handling?",
    }


@loggable
def canonicalize_tautomers(smiles: list[str]) -> tuple[list[str], list[str]]:
    """
    Canonicalize tautomeric forms of molecules in a list of SMILES strings.
    
    This function processes a list of SMILES strings and standardizes each to RDKit's 
    canonical tautomer representation. This ensures that different tautomeric forms of 
    the same molecule are represented by the same SMILES string, which is essential for 
    deduplication, comparison, and ensuring that tautomers are treated as equivalent 
    structures in downstream analyses.
    
    **IMPORTANT**: The output SMILES are tautomer-canonicalized AND canonicalized. No 
    additional canonicalization step is needed after running this function, as RDKit's 
    MolToSmiles with canonical=True is automatically applied.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to process. May contain different tautomeric forms.
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - canonical_tautomers : list[str]
            Tautomer-canonicalized SMILES strings. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Tautomer canonicalization successful
            - "Failed: Invalid SMILES string": Could not parse SMILES
            - "Failed: <reason>": An error occurred during processing
    
    Examples
    --------
    # Canonicalize keto-enol tautomers
    smiles = ["O=C1NC=CC=C1", "OC1=NC=CC=C1"]
    canonical, comments = canonicalize_tautomers(smiles)
    # Returns: ["O=C1NC=CC=C1", "O=C1NC=CC=C1"], ["Passed", "Passed"]
    # Note: Both tautomers become identical
    
    # Canonicalize amide-imidic acid tautomers
    smiles = ["CC(=O)N", "CC(O)=N"]
    canonical, comments = canonicalize_tautomers(smiles)
    # Returns: ["CC(=O)N", "CC(=O)N"], ["Passed", "Passed"]
    
    # Already canonical tautomer (no change)
    smiles = ["c1ccccc1", "CCO"]
    canonical, comments = canonicalize_tautomers(smiles)
    # Returns: ["c1ccccc1", "CCO"], ["Passed", "Passed"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Output is BOTH tautomer-canonicalized AND canonicalized - no additional steps needed
    - Uses RDKit's TautomerEnumerator with canonical tautomer selection
    - Common tautomerizations handled:
      * Keto-enol tautomers
      * Imine-enamine tautomers
      * Amide-imidic acid tautomers
      * Lactam-lactim tautomers
      * Nitroso-oxime tautomers
    - Different tautomers of the same molecule will have identical output SMILES
    - Molecules without tautomeric forms are returned unchanged (but canonicalized)
    - Output lists have the same length and order as input list
    
    Warnings
    --------
    - Tautomer canonicalization may affect analysis where specific tautomeric forms 
      are biologically or chemically relevant
    - The canonical tautomer selected may not be the predominant form under specific 
      conditions (pH, solvent, etc.)
    - Some edge cases with complex tautomeric systems may not be handled perfectly
    
    See Also
    --------
    canonicalize_tautomers_dataset : For dataset-level tautomer canonicalization
    canonicalize_smiles : For standard canonicalization without tautomer handling
    """
    from molml_mcp.tools.core_mol.smiles_ops import _canonicalize_tautomer_smiles
    
    canonical_tautomers, comments = [], []
    for smi in smiles:
        canon_smi, comment = _canonicalize_tautomer_smiles(smi)
        canonical_tautomers.append(canon_smi)
        comments.append(comment)
    
    return canonical_tautomers, comments


@loggable
def canonicalize_tautomers_dataset(
    resource_id: str,
    column_name: str
) -> dict:
    """
    Canonicalize tautomeric forms of molecules in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by standardizing SMILES strings to their 
    canonical tautomer representations in the specified column. It adds two new columns 
    to the dataframe: one containing the tautomer-canonicalized SMILES and another with 
    comments logged during the canonicalization process.
    
    **IMPORTANT**: The output SMILES are tautomer-canonicalized AND canonicalized. No 
    additional canonicalization step is needed after running this function, as RDKit's 
    MolToSmiles with canonical=True is automatically applied.
    
    This ensures that different tautomeric forms of the same molecule are represented 
    by the same SMILES string, which is essential for:
    - Deduplication of datasets where tautomers should be treated as equivalent
    - Machine learning where tautomers should have the same representation
    - Consistent database searches and comparisons
    
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
            Identifier for the new resource with tautomer-canonicalized data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            tautomer canonicalization (e.g., number of successful operations, failures).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the operation and canonicalization behavior.
        - warning : str
            Important warnings about tautomer canonicalization.
        - suggestions : str
            Recommendations for next steps.
        - question_to_user : str
            Question directed at the user/client regarding tautomer handling.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_tautomer_canonicalization': Contains the tautomer-canonicalized SMILES.
    - 'comments_after_tautomer_canonicalization': Contains any comments or warnings from 
      the canonicalization process.
    
    Common tautomerizations handled:
    - Keto-enol tautomers
    - Imine-enamine tautomers
    - Amide-imidic acid tautomers
    - Lactam-lactim tautomers
    - Nitroso-oxime tautomers
    
    Warnings
    --------
    - The canonical tautomer selected may not be the predominant form under specific 
      experimental conditions (pH, solvent, temperature)
    - For some molecules, the biologically active form may be a non-canonical tautomer
    - Tautomer canonicalization may affect structure-activity relationships if specific 
      tautomeric forms have different activities
    
    Examples
    --------
    # Typical usage after basic cleaning steps
    result = canonicalize_tautomers_dataset(
        resource_id="20251204T120000_csv_ABC123.csv", 
        column_name="smiles_after_neutralization"
    )
    
    # As part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(resource_id="initial.csv", column_name="smiles")
    # Step 2: Neutralize
    result2 = neutralize_smiles_dataset(resource_id=result1["resource_id"], 
                                        column_name="smiles_after_salt_removal")
    # Step 3: Canonicalize tautomers
    result3 = canonicalize_tautomers_dataset(resource_id=result2["resource_id"], 
                                             column_name="smiles_after_neutralization")
    
    See Also
    --------
    canonicalize_tautomers : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For standard canonicalization without tautomer handling
    neutralize_smiles_dataset : For charge neutralization
    standardize_stereochemistry_dataset : For stereochemistry handling
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    canonical_tautomers, comments = canonicalize_tautomers(smiles_list)

    df['smiles_after_tautomer_canonicalization'] = canonical_tautomers
    df['comments_after_tautomer_canonicalization'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful tautomer canonicalization is marked by 'Passed' in comments. Output SMILES are both tautomer-canonicalized AND canonicalized - no additional canonicalization step is needed. Different tautomers of the same molecule now have identical SMILES representations.",
        "warning": "The canonical tautomer selected may not be the predominant form under your specific experimental conditions (pH, solvent, etc.). For some molecules, the biologically active form may be a non-canonical tautomer.",
        "suggestions": "Review molecules that failed tautomer canonicalization. Consider deduplicating the dataset as different tautomers are now identical. If specific tautomeric forms are important for your analysis, verify that canonicalization is appropriate.",
        "question_to_user": "Are there any molecules in your dataset where specific tautomeric forms are biologically or chemically important? Would you like to review the tautomer canonicalization results?",
    }





def get_molecule_standardization_recommendations():
    """Return recommendations for molecular standardization steps."""
    return (
        "Recommended molecular standardization steps include:\n"
        "1. Remove common salts using `remove_salts()` or `remove_salts_dataset()`.\n"
        "2. Remove common solvents using `remove_common_solvents()` or `remove_common_solvents_dataset()`.\n"
        "3. Defragment molecules to keep the largest fragment using `defragment_smiles()` or `defragment_smiles_dataset()`.\n"
        "4. Neutralize charged species using `neutralize_smiles()` or `neutralize_smiles_dataset()`.\n"
        "5. Canonicalize SMILES strings using `canonicalize_smiles()` or `canonicalize_smiles_dataset()`.\n"
        "6. Optionally, flatten stereochemistry if not relevant using `flatten_stereochemistry()` or `flatten_stereochemistry_dataset()`.\n"
        "These steps help ensure consistency and reliability in molecular representations for downstream analyses."
    )


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
        neutralize_smiles,
        neutralize_smiles_dataset,
        standardize_stereochemistry,
        standardize_stereochemistry_dataset,
        remove_isotopes,
        remove_isotopes_dataset,
        canonicalize_tautomers,
        canonicalize_tautomers_dataset,
    ]





