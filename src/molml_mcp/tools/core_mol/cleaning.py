from collections import Counter
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable
from molml_mcp.tools.core_mol.smiles_ops import _canonicalize_smiles, _remove_pattern, _strip_common_solvent_fragments, _defragment_smiles, _normalize_smiles, _reionize_smiles, _disconnect_metals_smiles, _validate_smiles

from molml_mcp.constants import SMARTS_COMMON_SALTS


def get_SMILES_standardization_guidelines() -> str:
    """
    Return comprehensive guidelines for the default SMILES standardization protocol.

    It is STRONGLY recommended to stick to these steps unless you have a specific reason to deviate.
    
    This function provides detailed documentation of the recommended 11-step molecular
    standardization pipeline designed for general-purpose molecular machine learning
    applications. The protocol prioritizes structural cleanup, functional group 
    standardization, chemical canonicalization, and final validation.
    
    Returns
    -------
    str
        Multi-line string containing:
        - Complete protocol description with all 11 steps
        - Rationale for each step and their ordering
        - Mandatory policy decisions (e.g., stereochemistry flattening)
        - Optional steps (isotope removal, metal disconnection) with guidance
        - Critical dependencies and warnings
        - Use case recommendations
    
    Examples
    --------
    # Get the standardization guidelines
    guidelines = get_SMILES_standardization_guidelines()
    print(guidelines)
    
    # Use for documentation or client suggestions
    protocol_info = get_SMILES_standardization_guidelines()
    
    Notes
    -----
    This function is intended to provide standardization guidance to MCP clients
    even if they don't use the automated pipeline functions. It serves as both
    documentation and a reference for manual implementation of the protocol.
    
    The protocol is designed for GENERAL-PURPOSE ML applications. Specific use cases
    (drug discovery, SAR studies, metallodrug analysis, isotope tracing) may require
    modifications to the default policy decisions.
    
    See Also
    --------
    default_SMILES_standardization_pipeline_dataset : Automated implementation of this protocol
    """
    
    guidelines = """
================================================================================
DEFAULT SMILES STANDARDIZATION PROTOCOL
================================================================================

A comprehensive, scientifically-grounded molecular standardization pipeline 
designed for general-purpose molecular machine learning applications.

This protocol consists of 11 sequential steps, prioritizing structural cleanup
first, then functional group standardization, followed by chemical 
canonicalization and final validation.

--------------------------------------------------------------------------------
PROTOCOL STEPS
--------------------------------------------------------------------------------

**PHASE 1: INITIAL CANONICALIZATION & STRUCTURAL CLEANUP**

Step 1: Initial Canonicalization
    Function: canonicalize_smiles_dataset(input_filename, column_name="smiles", project_manifest_path, output_filename)
    Purpose: Establish baseline consistency and enable early detection of 
             invalid structures
    Rationale: Start with canonical form to identify parsing issues immediately
    Output Column: smiles_after_canonicalization

Step 2: Salt Removal
    Function: remove_salts_dataset(input_filename, column_name, project_manifest_path, output_filename, 
                                    salt_smarts=SMARTS_COMMON_SALTS)
    Purpose: Remove common pharmaceutical salt counterions (Cl, Na, Mg, Ca, K, 
             Br, Zn, Ag, Al, Li, I, O, N, H)
    Rationale: Salts are typically not part of the active molecular structure
    WARNING: NEVER modify salt_smarts unless working with organometallics where
             metals are part of the active structure
    Output Column: smiles_after_salt_removal

Step 3: Solvent Removal
    Function: remove_common_solvents_dataset(input_filename, column_name, project_manifest_path, output_filename)
    Purpose: Strip 35 common laboratory solvents (water, ethanol, DMF, DMSO, etc.)
    Rationale: Solvents are co-crystallization artifacts, not target molecules
    CRITICAL: Must be done BEFORE defragmentation to prevent accidentally 
              keeping a solvent as the "largest fragment"
    Output Column: smiles_after_solvent_removal

Step 4: Defragmentation
    Function: defragment_smiles_dataset(input_filename, column_name, project_manifest_path, output_filename, 
                                         keep_largest_fragment=True)
    Purpose: Isolate largest molecular component
    Rationale: After salt/solvent removal, keep the main molecule
    LIMITATION: Uses SMILES string length heuristic (not bulletproof - doesn't
                account for implicit hydrogens or molecular weight)
    DEPENDENCY: MUST run after salt and solvent removal
    Output Column: smiles_after_defragmentation

**PHASE 2: FUNCTIONAL GROUP STANDARDIZATION**

Step 5: Functional Group Normalization
    Function: normalize_functional_groups_dataset(input_filename, column_name, project_manifest_path, output_filename)
    Purpose: Standardize nitro groups, N-oxides, azides, diazo compounds, 
             sulfoxides, and phosphates to preferred representations
    Rationale: RDKit's Normalizer ensures consistent functional group encoding
    Output: Normalized AND canonicalized (no additional canonicalization needed)
    Output Column: smiles_after_normalization

Step 6: Reionization
    Function: reionize_smiles_dataset(input_filename, column_name, project_manifest_path, output_filename)
    Purpose: Adjust charge distributions to chemically preferred forms for 
             zwitterions and multi-ionizable compounds
    Rationale: RDKit's Reionizer applies chemical knowledge to set reasonable
               charge states
    MANDATORY: Must be done BEFORE neutralization if you want controlled charges
    WARNING: May change charge states in ways not matching experimental conditions
    Output: Reionized AND canonicalized (no additional canonicalization needed)
    Output Column: smiles_after_reionization

Step 7: Charge Neutralization
    Function: neutralize_smiles_dataset(input_filename, column_name, project_manifest_path, output_filename)
    Purpose: Convert charged species to neutral forms (protonated amines→amines,
             carboxylates→acids)
    Rationale: Most ML models work better with neutral forms; reduces complexity
    NOTE: Automatically skips quaternary ammonium salts and preserves essential
          charges where neutralization would be chemically inappropriate
    WARNING: Not suitable for all molecules (zwitterions, ionic liquids, 
             permanently charged drugs)
    Output: Neutralized AND canonicalized (no additional canonicalization needed)
    Output Column: smiles_after_neutralization

**PHASE 3: OPTIONAL SPECIALIZED CLEANING**

Step 8 (OPTIONAL): Isotope Removal
    Function: remove_isotopes_dataset(input_filename, column_name, project_manifest_path, output_filename)
    Purpose: Remove isotopic labels ([2H], [13C], [18F], etc.) to standard isotopes
    Rationale: Isotope labels are typically irrelevant for structure-based ML
    DEFAULT: Included in standard protocol (most use cases don't need isotopes)
    WHEN TO SKIP: Radiolabeling studies, NMR experiments, mass spec isotope 
                  tracing, metabolic flux analysis
    ARGUMENT: remove_isotopes=True (default) or False
    WARNING: This operation is IRREVERSIBLE
    Output: De-isotoped AND canonicalized (no additional canonicalization needed)
    Output Column: smiles_after_isotope_removal

Step 9 (OPTIONAL): Metal Disconnection
    Function: disconnect_metals_smiles_dataset(input_filename, column_name, project_manifest_path, output_filename, 
                                                 drop_inorganics=False)
    Purpose: Break metal-ligand coordinate bonds; optionally filter inorganics
    Rationale: Separates organic ligands from metal centers for structure-based ML
    DEFAULT: NOT included in standard protocol (most datasets don't have metals)
    WHEN TO INCLUDE: Datasets with coordination complexes or organometallics
    WHEN TO SKIP: Metallodrugs where coordination is essential for activity
    ARGUMENT: disconnect_metals=False (default) or True
    NOTE: If enabled, typically followed by re-defragmentation to remove 
          disconnected metal fragments
    WARNING: Destroys coordination geometry information
    Output: Metal-disconnected AND canonicalized
    Output Column: smiles_after_metal_disconnection

Step 9b (CONDITIONAL): Re-defragmentation After Metal Disconnection
    Function: defragment_smiles_dataset(input_filename, column_name, project_manifest_path, output_filename, 
                                         keep_largest_fragment=True)
    Purpose: Remove disconnected metal fragments after coordinate bond breaking
    Rationale: Keep organic ligand, discard metal center
    ONLY IF: disconnect_metals=True in Step 9
    Output Column: smiles_after_re_defragmentation

**PHASE 4: CHEMICAL CANONICALIZATION**

Step 10: Tautomer Canonicalization
    Function: canonicalize_tautomers_dataset(input_filename, column_name, project_manifest_path, output_filename)
    Purpose: Standardize keto-enol, imine-enamine, and other tautomeric forms
             to RDKit's canonical tautomer
    Rationale: Tautomers are chemically equivalent and should have identical
               representations for deduplication and ML
    WARNING: Canonical tautomer may not be the predominant form under specific
             conditions (pH, solvent)
    Output: Tautomer-canonicalized AND canonicalized
    Output Column: smiles_after_tautomer_canonicalization

Step 11: Stereochemistry Standardization
    Function: standardize_stereochemistry_dataset(input_filename, column_name, project_manifest_path, output_filename, 
                                                    stereo_policy="flatten")
    Purpose: Remove all stereochemical information (chiral centers + E/Z bonds)
             to treat stereoisomers as identical
    Rationale: For general ML, stereochemistry often adds noise without improving
               predictions; reduces dataset size by deduplicating stereoisomers
    POLICY DECISION: stereo_policy="flatten" (default for general ML)
    ALTERNATIVE POLICIES:
        - stereo_policy="keep": Preserve existing stereochemistry for SAR studies
                                or drug discovery where enantiomers have different
                                activities
        - stereo_policy="assign": Enumerate and assign undefined stereocenters
    ARGUMENT: stereo_policy="flatten" (default), "keep", or "assign"
    WARNING: Flattening loses information that may be critical for drug activity
    Output: Standardized AND canonicalized
    Output Column: smiles_after_stereo_standardization

**PHASE 5: VALIDATION**

Step 12: Final Validation
    Function: validate_smiles_dataset(input_filename, column_name, project_manifest_path, output_filename)
    Purpose: Verify parseability, sanitization, and compute validation statistics
    Rationale: Final quality control check; flag problematic entries for review
    NOTE: Returns ORIGINAL SMILES unchanged (validation only, not modification)
    Output: Validation statistics (n_valid, n_invalid, validation_rate)
    Output Columns: validation_status, validation_comments

--------------------------------------------------------------------------------
CRITICAL DEPENDENCIES
--------------------------------------------------------------------------------

1. Solvents BEFORE Defragmentation (Steps 3→4)
   Failure to follow: May keep solvent as "largest fragment" instead of molecule
   
2. Normalization BEFORE Reionization (Steps 5→6)
   Failure to follow: Reionizer may not work correctly on non-normalized structures
   
3. Reionization BEFORE Neutralization (Steps 6→7)
   Failure to follow: Lose control over charge distribution
   
4. Metal Disconnection BEFORE Re-defragmentation (Steps 9→9b)
   Failure to follow: Disconnected metal fragments remain in SMILES

--------------------------------------------------------------------------------
POLICY DECISIONS & ARGUMENTS
--------------------------------------------------------------------------------

The standard protocol makes the following policy decisions for general-purpose ML:

1. **Stereochemistry**: FLATTEN (treat stereoisomers as identical)
   Argument: stereo_policy="flatten"
   Alternatives: "keep" (SAR studies), "assign" (fill undefined stereocenters)
   
2. **Isotope Removal**: ENABLED (remove isotope labels)
   Argument: remove_isotopes=True
   Alternative: False (keep isotopes for specialized studies)
   
3. **Metal Disconnection**: DISABLED (keep metal complexes intact)
   Argument: disconnect_metals=False
   Alternative: True (break metal-ligand bonds)

4. **Salt Pattern**: DEFAULT (pharmaceutical salts)
   Argument: salt_smarts=SMARTS_COMMON_SALTS
   Alternative: Custom SMARTS (only for specialized datasets)

5. **Defragmentation**: ENABLED (keep largest fragment)
   Argument: keep_largest_fragment=True
   Alternative: False (keep all fragments)

--------------------------------------------------------------------------------
USE CASE RECOMMENDATIONS
--------------------------------------------------------------------------------

**GENERAL ML / QSAR / ADMET Prediction:**
Use default protocol as-is:
- stereo_policy="flatten"
- remove_isotopes=True
- disconnect_metals=False

**Drug Discovery / SAR Studies:**
Modify stereochemistry policy:
- stereo_policy="keep"  (enantiomers may have different activities)
- remove_isotopes=True
- disconnect_metals=False

**Organometallic Chemistry:**
Disable metal disconnection and modify salt removal:
- stereo_policy="keep"
- remove_isotopes=True
- disconnect_metals=False
- salt_smarts=<custom pattern excluding essential metals>

**Coordination Chemistry:**
Enable metal disconnection:
- stereo_policy="flatten"
- remove_isotopes=True
- disconnect_metals=True  (separate ligands from metal centers)

**Radiolabeling / Isotope Tracing:**
Disable isotope removal:
- stereo_policy="flatten"
- remove_isotopes=False  (isotopes are essential data)
- disconnect_metals=False

**Metallodrug Analysis:**
Keep metal complexes intact:
- stereo_policy="keep"
- remove_isotopes=True
- disconnect_metals=False  (coordination is essential)

--------------------------------------------------------------------------------
LIMITATIONS & WARNINGS
--------------------------------------------------------------------------------

1. **Defragmentation Heuristic**: Uses SMILES string length (NOT bulletproof)
   - Doesn't account for implicit hydrogens
   - Doesn't correlate with molecular weight
   - May fail for similar-sized fragments
   - MITIGATION: Always remove salts/solvents first

2. **Neutralization Edge Cases**: May not be appropriate for:
   - Quaternary ammonium salts (e.g., choline)
   - Zwitterions (e.g., amino acids)
   - Permanently charged drugs (e.g., muscle relaxants)
   - Ionic liquids
   - MITIGATION: Function automatically preserves essential charges

3. **Stereochemistry Flattening**: Loses chirality information
   - Enantiomers become identical
   - May affect SAR predictions
   - MITIGATION: Use stereo_policy="keep" for drug discovery

4. **Tautomer Selection**: Canonical tautomer may not be predominant form
   - RDKit's choice may not match biological conditions
   - pH/solvent effects ignored
   - MITIGATION: Accept as standardization convention

5. **Isotope Removal**: Irreversible operation
   - Cannot recover isotope labels
   - May affect specialized studies
   - MITIGATION: Use remove_isotopes=False when needed

6. **Metal Disconnection**: Destroys coordination information
   - Loses geometry
   - Inappropriate for metallodrugs
   - MITIGATION: Use disconnect_metals=False (default)

--------------------------------------------------------------------------------
IMPLEMENTATION NOTES
--------------------------------------------------------------------------------

- Most functions automatically canonicalize output (no redundant canonicalization)
- Each step adds new columns with descriptive names
- Comments columns track processing status for each molecule
- Failed operations preserve original values with failure reasons
- Resource IDs are chained: output of step N becomes input of step N+1
- All steps use @loggable decorator for automatic operation history logging
- Preview shows first 5 rows after each step for visual inspection

--------------------------------------------------------------------------------
EXPECTED OUTCOMES
--------------------------------------------------------------------------------

After running the full protocol:
- Structurally equivalent molecules have identical SMILES
- Salts, solvents, and fragments removed
- Functional groups standardized
- Charges normalized (neutral or chemically reasonable)
- Isotopes removed (if enabled)
- Metals disconnected (if enabled)
- Tautomers deduplicated
- Stereoisomers deduplicated (if flattened)
- Invalid molecules flagged for review

For deduplication, use the final SMILES column to identify duplicates.

================================================================================
END OF PROTOCOL
================================================================================
"""
    return guidelines



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
def canonicalize_smiles_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Canonicalize all SMILES strings"
) -> dict:
    """
    Canonicalize all SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by canonicalizing SMILES strings in the 
    specified column. It adds two new columns to the dataframe: one containing the 
    canonicalized SMILES and another with comments logged during the canonicalization 
    process (e.g., invalid SMILES, conversion failures).
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset (e.g., 'dataset_raw_A3F2B1D4').
    column_name : str
        Name of the column containing SMILES strings to be canonicalized.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., 'dataset_cleaned').
    explanation : str
        Brief description of the canonicalization performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with canonicalized data.
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
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    canonical_smiles, comments = canonicalize_smiles(smiles_list)

    df['smiles_after_canonicalization'] = canonical_smiles
    df['comments_after_canonicalization'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
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
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Remove salt counterions from molecules",
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
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to be desalted.
    salt_smarts : str, optional
        SMARTS pattern defining which atoms/ions to remove.
        **Default: "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]"**
        This default covers common pharmaceutical salts and should be used in most cases.
        Only change this if you have a specific reason.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the salt removal performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with desalted data.
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
    result = remove_salts_dataset(input_filename="dataset_raw_A3F2B1D4", 
                                   column_name="smiles")
    
    See Also
    --------
    remove_salts : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For dataset-level canonicalization
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    desalted_smiles, comments = remove_salts(smiles_list, salt_smarts)

    df['smiles_after_salt_removal'] = desalted_smiles
    df['comments_after_salt_removal'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
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
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Remove common solvent molecules"
) -> dict:
    """
    Remove known common solvent fragments from SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by removing common laboratory solvent
    fragments from SMILES strings in the specified column. It adds two new columns to 
    the dataframe: one containing the SMILES with solvents removed and another with 
    comments logged during the removal process.
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to be processed.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the solvent removal performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with solvent-free data.
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
    result = remove_common_solvents_dataset(input_filename="dataset_raw_A3F2B1D4", 
                                            column_name="smiles")
    
    See Also
    --------
    remove_common_solvents : For processing a list of SMILES strings
    remove_salts_dataset : For dataset-level salt removal
    canonicalize_smiles_dataset : For dataset-level canonicalization
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    cleaned_smiles, comments = remove_common_solvents(smiles_list)

    df['smiles_after_solvent_removal'] = cleaned_smiles
    df['comments_after_solvent_removal'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
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
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Remove disconnected fragments from SMILES",
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
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to be defragmented.
    keep_largest_fragment : bool, optional
        If True (default), keeps the largest fragment based on SMILES string length.
        If False, returns the original SMILES unchanged if fragments are detected.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the defragmentation performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with defragmented data.
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
    result1 = remove_common_solvents_dataset(input_filename="dataset_raw_A3F2B1D4", 
                                             column_name="smiles_after_salt_removal")
    # Step 2: Then defragment
    result2 = defragment_smiles_dataset(input_filename=result1["output_filename_id"], 
                                        column_name="smiles_after_solvent_removal")
    
    See Also
    --------
    defragment_smiles : For processing a list of SMILES strings
    remove_common_solvents_dataset : Should be run BEFORE this function
    remove_salts_dataset : For dataset-level salt removal
    canonicalize_smiles_dataset : For dataset-level canonicalization
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    defragmented_smiles, comments = defragment_smiles(smiles_list, keep_largest_fragment)

    df['smiles_after_defragmentation'] = defragmented_smiles
    df['comments_after_defragmentation'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
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
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Neutralize charges in molecules"
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
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to be neutralized.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the neutralization performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with neutralized data.
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
    result = neutralize_smiles_dataset(input_filename="dataset_raw_A3F2B1D4", 
                                       column_name="smiles_after_defragmentation")
    
    # Or as part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(input_filename="dataset_raw", column_name="smiles")
    # Step 2: Remove solvents
    result2 = remove_common_solvents_dataset(input_filename=result1["output_filename_id"], 
                                             column_name="smiles_after_salt_removal")
    # Step 3: Defragment
    result3 = defragment_smiles_dataset(input_filename=result2["output_filename_id"], 
                                        column_name="smiles_after_solvent_removal")
    # Step 4: Neutralize (already canonicalized, no additional step needed)
    result4 = neutralize_smiles_dataset(input_filename=result3["output_filename_id"], 
                                        column_name="smiles_after_defragmentation")
    
    See Also
    --------
    neutralize_smiles : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For canonicalization without neutralization
    remove_salts_dataset : For dataset-level salt removal
    defragment_smiles_dataset : For dataset-level defragmentation
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    neutralized_smiles, comments = neutralize_smiles(smiles_list)

    df['smiles_after_neutralization'] = neutralized_smiles
    df['comments_after_neutralization'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
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
        std_smi, comment = _standardize_stereo_smiles(
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
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Standardize stereochemistry in SMILES strings",
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
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
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
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the stereochemistry standardization performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with standardized data.
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
        input_filename="dataset_raw_A3F2B1D4",
        column_name="smiles_after_neutralization",
        stereo_policy="keep"
    )
    
    # Flatten all stereochemistry
    result = standardize_stereochemistry_dataset(
        input_filename="dataset_raw_A3F2B1D4",
        column_name="smiles_after_neutralization",
        stereo_policy="flatten"
    )
    
    # Assign stereochemistry with random selection
    result = standardize_stereochemistry_dataset(
        input_filename="dataset_raw_A3F2B1D4",
        column_name="smiles_after_neutralization",
        stereo_policy="assign",
        assign_policy="random",
        random_seed=42
    )
    
    # As part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(input_filename="dataset_raw", column_name="smiles")
    # Step 2: Neutralize
    result2 = neutralize_smiles_dataset(input_filename=result1["output_filename_id"], 
                                        column_name="smiles_after_salt_removal")
    # Step 3: Standardize stereochemistry
    result3 = standardize_stereochemistry_dataset(
        input_filename=result2["output_filename_id"], 
        column_name="smiles_after_neutralization",
        stereo_policy="flatten"  # or "keep" or "assign"
    )
    
    See Also
    --------
    standardize_stereochemistry : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For canonicalization that preserves stereochemistry
    neutralize_smiles_dataset : For charge neutralization
    """
    df = _load_resource(project_manifest_path, input_filename)
    
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

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    policy_notes = {
        "keep": "Existing stereochemistry has been preserved and canonicalized.",
        "assign": f"Stereoisomers have been enumerated and selected using '{assign_policy}' policy.",
        "flatten": "All stereochemical information has been removed. Stereoisomers are now indistinguishable."
    }

    return {
        "output_filename_id": output_filename_id,
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
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Remove isotope labels from molecules"
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
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to be de-isotoped.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the isotope removal performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with de-isotoped data.
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
    result = remove_isotopes_dataset(input_filename="dataset_raw_A3F2B1D4", 
                                     column_name="smiles_after_neutralization")
    
    # As part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(input_filename="dataset_raw", column_name="smiles")
    # Step 2: Neutralize
    result2 = neutralize_smiles_dataset(input_filename=result1["output_filename_id"], 
                                        column_name="smiles_after_salt_removal")
    # Step 3: Remove isotopes if not needed
    result3 = remove_isotopes_dataset(input_filename=result2["output_filename_id"], 
                                      column_name="smiles_after_neutralization")
    
    See Also
    --------
    remove_isotopes : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For canonicalization without isotope removal
    neutralize_smiles_dataset : For charge neutralization
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    clean_smiles, comments = remove_isotopes(smiles_list)

    df['smiles_after_isotope_removal'] = clean_smiles
    df['comments_after_isotope_removal'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
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
    - **CRITICAL**: Tautomer canonicalization can REMOVE or CHANGE stereochemistry. 
      This is a known RDKit limitation. If stereochemistry is important for your 
      analysis, consider running this step BEFORE stereochemistry standardization, 
      or skip tautomer canonicalization entirely.
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
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Canonicalize tautomers to standard forms"
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
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to be canonicalized.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the tautomer canonicalization performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with tautomer-canonicalized data.
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
        input_filename="dataset_raw_A3F2B1D4", 
        column_name="smiles_after_neutralization"
    )
    
    # As part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(input_filename="dataset_raw", column_name="smiles")
    # Step 2: Neutralize
    result2 = neutralize_smiles_dataset(input_filename=result1["output_filename_id"], 
                                        column_name="smiles_after_salt_removal")
    # Step 3: Canonicalize tautomers
    result3 = canonicalize_tautomers_dataset(input_filename=result2["output_filename_id"], 
                                             column_name="smiles_after_neutralization")
    
    See Also
    --------
    canonicalize_tautomers : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For standard canonicalization without tautomer handling
    neutralize_smiles_dataset : For charge neutralization
    standardize_stereochemistry_dataset : For stereochemistry handling
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    canonical_tautomers, comments = canonicalize_tautomers(smiles_list)

    df['smiles_after_tautomer_canonicalization'] = canonical_tautomers
    df['comments_after_tautomer_canonicalization'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful tautomer canonicalization is marked by 'Passed' in comments. Output SMILES are both tautomer-canonicalized AND canonicalized - no additional canonicalization step is needed. Different tautomers of the same molecule now have identical SMILES representations.",
        "warning": "The canonical tautomer selected may not be the predominant form under your specific experimental conditions (pH, solvent, etc.). For some molecules, the biologically active form may be a non-canonical tautomer.",
        "suggestions": "Review molecules that failed tautomer canonicalization. Consider deduplicating the dataset as different tautomers are now identical. If specific tautomeric forms are important for your analysis, verify that canonicalization is appropriate.",
        "question_to_user": "Are there any molecules in your dataset where specific tautomeric forms are biologically or chemically important? Would you like to review the tautomer canonicalization results?",
    }


@loggable
def normalize_functional_groups(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Normalize functional groups (nitro, N-oxide, azide, diazo, sulfoxide, phosphate) to preferred representations.
    
    Args:
        smiles: List of SMILES strings
        
    Returns:
        (normalized_smiles, comments) where comments are "Passed" or "Failed: <reason>"
    """
    results = [_normalize_smiles(smi) for smi in smiles]
    normalized_smiles = [smi for smi, _ in results]
    comments = [cmt for _, cmt in results]
    return normalized_smiles, comments


@loggable
def normalize_functional_groups_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Normalize functional groups to preferred representations"
) -> dict:
    """Normalize functional groups (nitro, N-oxide, azide, etc.) to preferred representations in a dataset column.
    
    Args:
        input_filename: Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\')
        column_name: Column with SMILES to normalize
        project_manifest_path: Path to the project manifest file for tracking this resource
        output_filename: Base filename for the stored resource (without extension)
        explanation: Brief description of the normalization performed
        
    Returns:
        dict with output_filename_id, n_rows, columns, comments (counts), preview, note, suggestions
        
    Adds columns: smiles_after_functional_group_normalization, comments_after_functional_group_normalization
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    normalized_smiles, comments = normalize_functional_groups(smiles_list)

    df['smiles_after_functional_group_normalization'] = normalized_smiles
    df['comments_after_functional_group_normalization'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful normalization is marked by 'Passed' in comments. Functional groups have been standardized to their preferred representations. The output SMILES are canonical and isomeric.",
        "suggestions": "Review molecules that failed normalization. Consider proceeding with charge neutralization and tautomer canonicalization steps.",
    }


@loggable
def reionize_smiles(smiles: list[str]) -> tuple[list[str], list[str]]:
    """
    Reionize molecules to their preferred charge distribution using RDKit's Reionizer.
    
    This function processes a list of SMILES strings and adjusts their charge distribution 
    to a chemically preferred form. Reionization is particularly useful for handling 
    zwitterions, molecules with multiple ionizable sites, and ensuring consistent charge 
    states across a dataset. It should typically be applied AFTER functional group 
    normalization but BEFORE charge neutralization.
    
    **IMPORTANT**: The output SMILES are reionized AND canonicalized. No additional 
    canonicalization step is needed after running this function, as RDKit's MolToSmiles 
    with canonical=True is automatically applied.
    
    Common reionizations include:
    - Amino acids: Correcting zwitterionic forms to appropriate charge states
    - Multi-ionizable compounds: Setting charges to preferred positions
    - Protonation state correction: Ensuring chemically reasonable charge distribution
    - pH-dependent ionization: Standardizing to a consistent protonation state
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to reionize. Should ideally be already normalized.
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - reionized_smiles : list[str]
            Reionized canonical SMILES strings. Length matches input list.
            Failed reionizations return None.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Reionization successful
            - "Failed: Invalid SMILES string": Input could not be parsed
            - "Failed: Reionization error: <details>": An error occurred during reionization
    
    Examples
    --------
    # Reionize a zwitterionic amino acid
    smiles = ["C([C@@H](C(=O)[O-])[NH3+])O"]
    reionized, comments = reionize_smiles(smiles)
    # Returns with adjusted charge distribution
    
    # Handle molecules with multiple ionizable sites
    smiles = ["c1ccc(cc1)C(=O)[O-]", "c1ccc(cc1)[NH3+]"]
    reionized, comments = reionize_smiles(smiles)
    # Returns with preferred charge states
    
    # Already optimally ionized (may return unchanged)
    smiles = ["c1ccccc1", "CCO"]
    reionized, comments = reionize_smiles(smiles)
    # Returns: ["c1ccccc1", "CCO"], ["Passed", "Passed"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Output is BOTH reionized AND canonicalized - no additional canonicalization needed
    - Best applied after functional group normalization (normalize_functional_groups)
    - Should be applied BEFORE neutralization if you want controlled charge states
    - Expects "reasonable" molecular structures (not highly unusual valence states)
    - Particularly useful for:
      * Zwitterionic compounds (amino acids, betaines)
      * Molecules with multiple ionizable groups
      * Ensuring consistent protonation states
    - Output lists have the same length and order as input list
    
    Warnings
    --------
    - Reionization may change the charge state in ways that don't match specific 
      experimental conditions (pH, solvent, temperature)
    - For molecules with complex ionization patterns, the "preferred" state may not 
      match biological or experimental relevance
    - Zwitterions may be converted to neutral or differently charged forms
    
    See Also
    --------
    reionize_smiles_dataset : Dataset version of this function
    normalize_functional_groups : Should be run BEFORE reionization
    neutralize_smiles : For complete charge removal (applied AFTER reionization)
    """
    results = [_reionize_smiles(smi) for smi in smiles]
    reionized_smiles = [smi for smi, _ in results]
    comments = [cmt for _, cmt in results]
    return reionized_smiles, comments


@loggable
def reionize_smiles_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Reionize molecules to preferred charge distribution"
) -> dict:
    """Reionize molecules to preferred charge distribution (zwitterions, multi-ionizable compounds) in a dataset column.
    
    Args:
        input_filename: Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\')
        column_name: Column with SMILES to reionize
        project_manifest_path: Path to the project manifest file for tracking this resource
        output_filename: Base filename for the stored resource (without extension)
        explanation: Brief description of the reionization performed
        
    Returns:
        dict with output_filename_id, n_rows, columns, comments (counts), preview, note, warning, suggestions
        
    Adds columns: smiles_after_reionization, comments_after_reionization
    Output is reionized AND canonicalized.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    reionized_smiles, comments = reionize_smiles(smiles_list)

    df['smiles_after_reionization'] = reionized_smiles
    df['comments_after_reionization'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename_id": output_filename_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful reionization is marked by 'Passed' in comments. Charge distributions have been adjusted to chemically preferred forms. The output SMILES are canonical and isomeric.",
        "warning": "Reionization may change charge states in ways that don't match specific experimental conditions (pH, solvent, etc.). Zwitterions may be converted to different forms.",
        "suggestions": "Review molecules that failed reionization. Consider proceeding with charge neutralization (if complete charge removal desired) or tautomer canonicalization steps.",
    }


@loggable
def disconnect_metals_smiles(smiles: list[str], drop_inorganics: bool = False) -> tuple[list[str], list[str]]:
    """
    Disconnect metal-ligand coordinate bonds in a list of SMILES strings.
    
    This function processes a list of SMILES strings and disconnects coordinate bonds 
    between metal atoms and ligands using RDKit's MetalDisconnector. This is useful for 
    standardizing organometallic compounds, metal complexes, and coordination compounds 
    by breaking dative/coordinate bonds while preserving the organic ligand structures.
    
    **IMPORTANT**: The output SMILES have disconnected metals AND are canonicalized. 
    No additional canonicalization step is needed after running this function.
    
    Common applications include:
    - Metal complexes: Separating metal centers from organic ligands
    - Organometallic compounds: Isolating the organic portion
    - Coordination compounds: Breaking coordinate bonds to metals
    - Drug discovery: Focusing on organic ligands without metal coordination
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to process. May contain metal-ligand bonds.
    drop_inorganics : bool, optional
        If True, molecules without carbon atoms (purely inorganic) are filtered out 
        and returned as None with a failure comment. Default is False.
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - disconnected_smiles : list[str]
            SMILES strings with metal-ligand bonds disconnected, canonicalized. 
            Length matches input list. Dropped inorganics return None.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Metal disconnection successful (or no metals present)
            - "Failed: Invalid SMILES string": Input could not be parsed
            - "Failed: Inorganic molecule (no carbon atoms)": Removed due to drop_inorganics=True
            - "Failed: Metal disconnection error: <details>": An error occurred during processing
    
    Examples
    --------
    # Disconnect a simple metal complex
    smiles = ["[Fe](Cl)(Cl)(Cl)(Cl)(Cl)Cl"]
    disconnected, comments = disconnect_metals_smiles(smiles)
    # Returns with metal-ligand bonds broken
    
    # Organometallic with organic ligand
    smiles = ["c1ccccc1[Fe]"]
    disconnected, comments = disconnect_metals_smiles(smiles)
    # Returns benzene separated from iron
    
    # Drop purely inorganic molecules
    smiles = ["[Fe]Cl6", "c1ccccc1", "[Na]Cl"]
    disconnected, comments = disconnect_metals_smiles(smiles, drop_inorganics=True)
    # Returns: [None, "c1ccccc1", None] with appropriate failure comments for inorganics
    
    # Already metal-free molecule (no change)
    smiles = ["c1ccccc1", "CCO"]
    disconnected, comments = disconnect_metals_smiles(smiles)
    # Returns: ["c1ccccc1", "CCO"], ["Passed", "Passed"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Output is BOTH metal-disconnected AND canonicalized - no additional steps needed
    - Breaks dative/coordinate bonds between metals and ligands
    - Does NOT remove metal atoms themselves (unless drop_inorganics=True for inorganics)
    - Typical workflow position: After defragmentation, before or after functional group normalization
    - Particularly useful for:
      * Organometallic chemistry datasets
      * Drug discovery focusing on organic scaffolds
      * Removing metal artifacts from screening libraries
      * Standardizing metal-containing compounds
    - Output lists have the same length and order as input list
    
    Warnings
    --------
    - Metal coordination may be essential for biological activity in some cases:
      * Metallodrugs where metal is part of the active compound
      * Enzyme inhibitors that coordinate to metal centers
      * Porphyrins and other metal-cofactor systems
    - Disconnection removes structural information about metal coordination geometry
    - When drop_inorganics=True, purely inorganic salts and minerals are lost
    
    See Also
    --------
    disconnect_metals_smiles_dataset : Dataset version of this function
    defragment_smiles : For removing disconnected fragments after metal disconnection
    remove_salts : For removing salt counterions
    """
    results = [_disconnect_metals_smiles(smi, drop_inorganics=drop_inorganics) for smi in smiles]
    disconnected_smiles = [smi for smi, _ in results]
    comments = [cmt for _, cmt in results]
    return disconnected_smiles, comments


@loggable
def disconnect_metals_smiles_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Disconnect metal-ligand coordinate bonds",
    drop_inorganics: bool = False
) -> dict:
    """
    Disconnect metal-ligand coordinate bonds in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by disconnecting coordinate bonds between 
    metal atoms and ligands in SMILES strings. It adds two new columns to the dataframe: 
    one containing the metal-disconnected SMILES and another with comments logged during 
    the disconnection process.
    
    **IMPORTANT**: The output SMILES have disconnected metals AND are canonicalized. 
    No additional canonicalization step is needed after running this function.
    
    This is useful for:
    - Standardizing organometallic and coordination compound datasets
    - Isolating organic ligands from metal complexes
    - Removing metal coordination artifacts from screening libraries
    - Focusing analysis on organic scaffolds without metal centers
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to process.
    drop_inorganics : bool, optional
        If True, molecules without carbon atoms (purely inorganic) are filtered out 
        and returned as None. Default is False.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the metal disconnection performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with metal-disconnected data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            metal disconnection (e.g., number of successful operations, failures, dropped inorganics).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the operation.
        - warning : str
            Important warnings about metal disconnection.
        - suggestions : str
            Recommendations for next steps.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_metal_disconnection': Contains the SMILES with disconnected metal bonds.
    - 'comments_after_metal_disconnection': Contains any comments or warnings from the 
      disconnection process.
    
    Typical workflow position:
    1. Remove salts
    2. Remove solvents
    3. Defragment (or after metal disconnection)
    4. **Disconnect metals** ← This step
    5. Defragment again (to remove disconnected metal fragments if desired)
    6. Normalize functional groups
    7. Reionize/neutralize
    8. Canonicalize tautomers
    
    Warnings
    --------
    - Metal coordination may be essential for activity in metallodrugs
    - Disconnection loses information about coordination geometry
    - Purely inorganic molecules are dropped when drop_inorganics=True
    - Consider whether metal-free ligands are appropriate for your analysis
    
    Examples
    --------
    # Typical usage after defragmentation
    result = disconnect_metals_smiles_dataset(
        input_filename="dataset_raw_A3F2B1D4", 
        column_name="smiles_after_defragmentation"
    )
    
    # Drop purely inorganic molecules
    result = disconnect_metals_smiles_dataset(
        input_filename="dataset_raw_A3F2B1D4", 
        column_name="smiles_after_defragmentation",
        drop_inorganics=True
    )
    
    # As part of a cleaning pipeline
    # Step 1: Remove salts
    result1 = remove_salts_dataset(input_filename="dataset_raw", column_name="smiles")
    # Step 2: Defragment
    result2 = defragment_smiles_dataset(input_filename=result1["output_filename_id"], 
                                        column_name="smiles_after_salt_removal")
    # Step 3: Disconnect metals
    result3 = disconnect_metals_smiles_dataset(input_filename=result2["output_filename_id"], 
                                               column_name="smiles_after_defragmentation",
                                               drop_inorganics=False)
    # Step 4: Defragment again to remove disconnected metal fragments
    result4 = defragment_smiles_dataset(input_filename=result3["output_filename_id"], 
                                        column_name="smiles_after_metal_disconnection")
    # Step 5: Normalize functional groups
    result5 = normalize_functional_groups_dataset(input_filename=result4["output_filename_id"], 
                                                   column_name="smiles_after_defragmentation")
    
    See Also
    --------
    disconnect_metals_smiles : For processing a list of SMILES strings
    defragment_smiles_dataset : Should be run AFTER metal disconnection to remove fragments
    remove_salts_dataset : For removing salt counterions
    normalize_functional_groups_dataset : For functional group standardization
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    disconnected_smiles, comments = disconnect_metals_smiles(smiles_list, drop_inorganics=drop_inorganics)

    df['smiles_after_metal_disconnection'] = disconnected_smiles
    df['comments_after_metal_disconnection'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    inorganic_note = " Purely inorganic molecules (no carbon) have been dropped." if drop_inorganics else ""

    return {
        "output_filename_id": output_filename_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": f"Successful metal disconnection is marked by 'Passed' in comments. Coordinate bonds between metals and ligands have been broken. The output SMILES are canonical and isomeric.{inorganic_note}",
        "warning": "Metal coordination may be essential for biological activity in metallodrugs and metal-dependent enzyme inhibitors. Metal disconnection removes structural information about coordination geometry. Consider whether this is appropriate for your analysis.",
        "suggestions": "Review molecules that failed metal disconnection. Consider running defragment_smiles_dataset again to remove disconnected metal fragments. Proceed with functional group normalization and other cleaning steps.",
    }


@loggable
def validate_smiles(smiles: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate a list of SMILES strings for correctness and chemical sanity.
    
    This function performs lightweight validation to check whether SMILES strings can be 
    parsed by RDKit, pass sanitization checks, and represent valid molecular structures. 
    This is useful as a final quality control step after cleaning, or to filter out 
    problematic molecules before processing.
    
    Validation checks include:
    - Parseable by RDKit (MolFromSmiles succeeds)
    - Passes RDKit's sanitization (valence, aromaticity, etc.)
    - Contains at least one atom (not empty)
    - No parsing exceptions
    
    **Note**: This function returns the ORIGINAL SMILES strings unchanged (not canonicalized).
    Use this for validation purposes, not for standardization.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to validate.
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - validated_smiles : list[str]
            Original SMILES strings if valid, None if invalid. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating validation status. Length matches input list.
            - "Passed": SMILES is valid
            - "Failed: Invalid SMILES string": Could not parse SMILES
            - "Failed: Empty molecule (0 atoms)": Molecule has no atoms
            - "Failed: Exception during parsing: <details>": An error occurred during parsing
    
    Examples
    --------
    # Validate valid SMILES
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    validated, comments = validate_smiles(smiles)
    # Returns: ["CCO", "c1ccccc1", "CC(=O)O"], ["Passed", "Passed", "Passed"]
    
    # Detect invalid SMILES
    smiles = ["CCO", "invalid_smiles", "c1ccccc"]
    validated, comments = validate_smiles(smiles)
    # Returns: ["CCO", None, None] with failure comments for invalid entries
    
    # Detect empty molecules
    smiles = ["", "CCO"]
    validated, comments = validate_smiles(smiles)
    # Returns with appropriate failure for empty string
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Returns ORIGINAL SMILES unchanged - does NOT canonicalize
    - Useful for quality control and filtering
    - Best applied:
      * As a final validation step after cleaning pipeline
      * Before expensive computations or database insertions
      * To identify problematic molecules for manual review
    - Validation is lightweight (just parsing + sanitization)
    - Does NOT check for chemical reasonableness beyond RDKit's sanitization
    - Output lists have the same length and order as input list
    
    Warnings
    --------
    - This function only validates RDKit parseability, not chemical realism
    - Some chemically unreasonable structures may still pass validation
    - Does not check for specific structural features or drug-likeness
    - Passing validation does not guarantee the molecule is scientifically meaningful
    
    See Also
    --------
    validate_smiles_dataset : Dataset version of this function
    canonicalize_smiles : For standardization (which includes implicit validation)
    """
    results = [_validate_smiles(smi) for smi in smiles]
    validated_smiles = [smi for smi, _ in results]
    comments = [cmt for _, cmt in results]
    return validated_smiles, comments


@loggable
def validate_smiles_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Validate SMILES strings for correctness"
) -> dict:
    """
    Validate SMILES strings in a specified column of a tabular dataset.
    
    This function performs lightweight validation on a tabular dataset to check whether 
    SMILES strings are parseable, pass sanitization, and represent valid molecular 
    structures. It adds two new columns to the dataframe: one containing the validated 
    SMILES (original, not canonicalized) and another with validation status comments.
    
    **Note**: This function returns ORIGINAL SMILES unchanged, making it suitable for 
    final quality control without altering the data.
    
    Validation checks:
    - Parseable by RDKit
    - Passes sanitization (valence, aromaticity)
    - Contains at least one atom
    - No parsing exceptions
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to validate.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the validation performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with validation results.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different validation statuses 
            (e.g., number of passed, failed validations).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the validation results.
        - n_valid : int
            Number of molecules that passed validation.
        - n_invalid : int
            Number of molecules that failed validation.
        - validation_rate : float
            Percentage of molecules that passed validation.
        - suggestions : str
            Recommendations for handling invalid molecules.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'validated_smiles': Contains the original SMILES if valid, None if invalid.
    - 'validation_status': Contains validation status for each molecule.
    
    Typical workflow position:
    - **As a final step** after all cleaning operations
    - Before expensive downstream processing
    - To identify problematic molecules for removal or manual review
    
    Use cases:
    - Quality control after data cleaning
    - Filtering datasets before machine learning
    - Identifying molecules that need manual curation
    - Final sanity check before database insertion
    
    Examples
    --------
    # Final validation after cleaning pipeline
    result = validate_smiles_dataset(
        input_filename="dataset_raw_A3F2B1D4", 
        column_name="smiles_after_tautomer_canonicalization"
    )
    
    # Check validation statistics
    print(f"Valid: {result['n_valid']}/{result['n_rows']}")
    print(f"Validation rate: {result['validation_rate']:.2f}%")
    
    # As part of a complete cleaning and validation pipeline
    # Step 1-8: Various cleaning steps...
    result8 = canonicalize_tautomers_dataset(input_filename=result7["output_filename_id"], 
                                             column_name="smiles_after_reionization")
    # Step 9: Final validation
    result9 = validate_smiles_dataset(input_filename=result8["output_filename_id"], 
                                      column_name="smiles_after_tautomer_canonicalization")
    
    # Filter to only valid molecules
    df = _load_resource(result9["output_filename_id"])
    df_valid = df[df["validation_status"] == "Passed"]
    
    See Also
    --------
    validate_smiles : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For standardization with implicit validation
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    validated_smiles, comments = validate_smiles(smiles_list)

    df['validated_smiles'] = validated_smiles
    df['validation_status'] = comments

    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    # Calculate validation statistics
    n_valid = sum(1 for c in comments if c == "Passed")
    n_invalid = len(comments) - n_valid
    validation_rate = (n_valid / len(comments) * 100) if comments else 0.0

    return {
        "output_filename_id": output_filename_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": f"Validation complete. {n_valid} molecules passed, {n_invalid} failed. Validation rate: {validation_rate:.2f}%. Original SMILES strings are preserved unchanged. Valid molecules marked as 'Passed' in validation_status column.",
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "validation_rate": validation_rate,
        "suggestions": "Review molecules that failed validation. Consider removing invalid molecules or investigating the cause of failures. Invalid molecules have None in the validated_smiles column and can be filtered out for downstream processing.",
    }



@loggable
def default_SMILES_standardization_pipeline(
    smiles: list[str],
    stereo_policy: str = "flatten",
    skip_isotope_removal: bool = False,
    enable_metal_disconnection: bool = False,
    drop_inorganics: bool = False,
    salt_smarts: str = SMARTS_COMMON_SALTS
) -> tuple[list[str], list[str]]:
    """
    Apply the default SMILES standardization protocol to a list of SMILES strings.
    
    This function implements the complete 11-step (or 12-step with metal disconnection)
    standardization pipeline optimized for general-purpose molecular machine learning.
    It returns clean SMILES with concise, non-redundant comments.
    
    **RECOMMENDED**: Use this function for standardizing SMILES lists. For dataset-level
    operations with full audit trails, use `default_SMILES_standardization_pipeline_dataset()`.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to standardize.
    stereo_policy : str, optional
        Stereochemistry handling policy (default: "flatten"):
        - "flatten": Remove all stereochemistry (general ML use case)
        - "keep": Preserve existing stereochemistry (SAR/drug discovery)
        - "assign": Enumerate and assign undefined stereocenters
    remove_isotopes : bool, optional
        Whether to remove isotope labels (default: True).
        Set to False for radiolabeling/NMR/mass spec studies.
    disconnect_metals : bool, optional
        Whether to disconnect metal-ligand bonds (default: False).
        Set to True for coordination chemistry analysis.
    drop_inorganics : bool, optional
        When disconnect_metals=True, whether to drop purely inorganic fragments
        (default: False).
    salt_smarts : str, optional
        SMARTS pattern for salt removal (default: SMARTS_COMMON_SALTS).
        **WARNING**: Only change for specialized datasets (e.g., organometallics).
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - standardized_smiles : list[str]
            Fully standardized SMILES strings. Length matches input list.
        - comments : list[str]
            Concise, non-redundant comments for each SMILES. Length matches input list.
            Only reports issues, warnings, or notable transformations.
            Successfully standardized molecules: "Standardized"
            Molecules with issues: "<step>: <issue>" (e.g., "Validation: Invalid SMILES")
    
    Examples
    --------
    # General ML use case (default settings)
    smiles = ["CC(=O)O.Na", "C[C@H](O)CC", "c1ccccc1"]
    clean, comments = default_SMILES_standardization_pipeline(smiles)
    # Returns: ["CC(=O)O", "CC(O)CC", "c1ccccc1"], 
    #          ["Standardized", "Standardized", "Standardized"]
    
    # Drug discovery (keep stereochemistry)
    smiles = ["C[C@H](O)CC", "C[C@@H](O)CC"]
    clean, comments = default_SMILES_standardization_pipeline(
        smiles, stereo_policy="keep"
    )
    # Returns: ["C[C@H](O)CC", "C[C@@H](O)CC"], ["Standardized", "Standardized"]
    
    # Coordination chemistry (disconnect metals)
    smiles = ["c1ccccc1.[Cu+2]"]
    clean, comments = default_SMILES_standardization_pipeline(
        smiles, enable_metal_disconnection=True
    )
    # Returns: ["c1ccccc1"], ["Standardized"]
    
    # Handle invalid SMILES
    smiles = ["CCO", "invalid_smiles", "c1ccccc1"]
    clean, comments = default_SMILES_standardization_pipeline(smiles)
    # Returns with appropriate failure comment for invalid entry
    
    Notes
    -----
    - This function applies ALL 11 steps of the standardization protocol
    - Comments are concise and non-redundant (no repeated "Passed" messages)
    - The function chains: canonicalize → remove_salts → remove_solvents → 
      defragment → normalize → reionize → neutralize → [optional: remove_isotopes] → 
      [optional: disconnect_metals + re-defragment] → canonicalize_tautomers → 
      standardize_stereochemistry → validate
    - **IMPORTANT**: Step 10 (tautomer canonicalization) can REMOVE or CHANGE 
      stereochemistry. This is a known RDKit limitation. The pipeline runs stereo 
      standardization AFTER tautomer canonicalization, but stereochemistry may 
      already be lost by that point. If stereochemistry preservation is critical, 
      consider using a custom pipeline that skips tautomer canonicalization.
    - For detailed step-by-step logging, use the dataset version
    - Output SMILES are fully canonicalized and standardized
    
    See Also
    --------
    default_SMILES_standardization_pipeline_dataset : Dataset version with full audit trail
    get_SMILES_standardization_guidelines : Detailed protocol documentation
    """
    from molml_mcp.tools.core_mol.smiles_ops import _initialise_neutralisation_reactions
    
    # Track which molecules fail at which step
    n_smiles = len(smiles)
    failure_info = ["" for _ in range(n_smiles)]  # Track first failure per molecule
    
    # Step 1: Initial canonicalization
    current_smiles, step_comments = canonicalize_smiles(smiles)
    for i, comment in enumerate(step_comments):
        if "Failed" in comment and not failure_info[i]:
            failure_info[i] = f"Canonicalization: {comment}"
    
    # Step 2: Salt removal
    current_smiles, step_comments = remove_salts(current_smiles, salt_smarts=salt_smarts)
    for i, comment in enumerate(step_comments):
        if "Failed" in comment and not failure_info[i]:
            failure_info[i] = f"Salt removal: {comment}"
    
    # Step 3: Solvent removal
    current_smiles, step_comments = remove_common_solvents(current_smiles)
    # Solvent removal doesn't typically fail, skip tracking
    
    # Step 4: Defragmentation
    current_smiles, step_comments = defragment_smiles(current_smiles, keep_largest_fragment=True)
    for i, comment in enumerate(step_comments):
        if "Unresolved" in comment and not failure_info[i]:
            failure_info[i] = f"Defragmentation: {comment}"
    
    # Step 5: Functional group normalization
    current_smiles, step_comments = normalize_functional_groups(current_smiles)
    for i, comment in enumerate(step_comments):
        if "Failed" in comment and not failure_info[i]:
            failure_info[i] = f"Normalization: {comment}"
    
    # Step 6: Reionization
    current_smiles, step_comments = reionize_smiles(current_smiles)
    for i, comment in enumerate(step_comments):
        if "Failed" in comment and not failure_info[i]:
            failure_info[i] = f"Reionization: {comment}"
    
    # Step 7: Neutralization
    current_smiles, step_comments = neutralize_smiles(current_smiles)
    for i, comment in enumerate(step_comments):
        if "Failed" in comment and not failure_info[i]:
            failure_info[i] = f"Neutralization: {comment}"
    
    # Step 8 (OPTIONAL): Isotope removal
    if not skip_isotope_removal:
        from molml_mcp.tools.core_mol import cleaning
        current_smiles, step_comments = cleaning.remove_isotopes(current_smiles)
        for i, comment in enumerate(step_comments):
            if "Failed" in comment and not failure_info[i]:
                failure_info[i] = f"Isotope removal: {comment}"
    
    # Step 9 (OPTIONAL): Metal disconnection
    if enable_metal_disconnection:
        current_smiles, step_comments = disconnect_metals_smiles(
            current_smiles, drop_inorganics=drop_inorganics
        )
        for i, comment in enumerate(step_comments):
            if "Failed" in comment and not failure_info[i]:
                failure_info[i] = f"Metal disconnection: {comment}"
        
        # Step 9b: Re-defragmentation after metal disconnection
        current_smiles, step_comments = defragment_smiles(current_smiles, keep_largest_fragment=True)
        for i, comment in enumerate(step_comments):
            if "Unresolved" in comment and not failure_info[i]:
                failure_info[i] = f"Re-defragmentation: {comment}"
    
    # Step 10: Tautomer canonicalization
    current_smiles, step_comments = canonicalize_tautomers(current_smiles)
    for i, comment in enumerate(step_comments):
        if "Failed" in comment and not failure_info[i]:
            failure_info[i] = f"Tautomer canonicalization: {comment}"
        # Track stereochemistry warnings (even if processing succeeded)
        elif "WARNING" in comment and "Stereochemistry" in comment:
            if not failure_info[i]:  # Only if no prior failure
                failure_info[i] = f"Tautomer canonicalization: {comment}"
    
    # Step 11: Stereochemistry standardization
    current_smiles, step_comments = standardize_stereochemistry(
        current_smiles,
        stereo_policy=stereo_policy,
        assign_policy="first",
        max_isomers=32,
        try_embedding=False,
        only_unassigned=True,
        random_seed=42
    )
    for i, comment in enumerate(step_comments):
        if "Failed" in comment and not failure_info[i]:
            failure_info[i] = f"Stereochemistry: {comment}"
    
    # Step 12: Final validation
    validated_smiles, step_comments = validate_smiles(current_smiles)
    for i, comment in enumerate(step_comments):
        if "Failed" in comment and not failure_info[i]:
            failure_info[i] = f"Validation: {comment}"
    
    # Generate final concise comments
    final_comments = []
    for i in range(n_smiles):
        if failure_info[i]:
            final_comments.append(failure_info[i])
        else:
            final_comments.append("Standardized")
    
    return validated_smiles, final_comments


@loggable
def default_SMILES_standardization_pipeline_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Apply default SMILES standardization protocol",
    stereo_policy: str = "flatten",
    skip_isotope_removal: bool = False,
    enable_metal_disconnection: bool = False,
    drop_inorganics: bool = False,
    salt_smarts: str = SMARTS_COMMON_SALTS
) -> dict:
    """
    Apply the default SMILES standardization protocol to a dataset with full audit trail.
    
    This function implements the complete 11-step (or 12-step with metal disconnection)
    standardization pipeline optimized for general-purpose molecular machine learning.
    It processes a tabular dataset and adds comment columns for EVERY step, plus a final
    'standardized_smiles' column.
    
    **RECOMMENDED**: Use this function for dataset-level standardization with complete
    documentation of all transformations. For simple list operations, use
    `default_SMILES_standardization_pipeline()`.
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\').
    column_name : str
        Name of the column containing SMILES strings to standardize.
    stereo_policy : str, optional
        Stereochemistry handling policy (default: "flatten"):
        - "flatten": Remove all stereochemistry (general ML use case)
        - "keep": Preserve existing stereochemistry (SAR/drug discovery)
        - "assign": Enumerate and assign undefined stereocenters
    remove_isotopes : bool, optional
        Whether to remove isotope labels (default: True).
        Set to False for radiolabeling/NMR/mass spec studies.
    disconnect_metals : bool, optional
        Whether to disconnect metal-ligand bonds (default: False).
        Set to True for coordination chemistry analysis.
    drop_inorganics : bool, optional
        When disconnect_metals=True, whether to drop purely inorganic fragments
        (default: False).
    salt_smarts : str, optional
        SMARTS pattern for salt removal (default: SMARTS_COMMON_SALTS).
        **WARNING**: Only change for specialized datasets (e.g., organometallics).
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., \'dataset_cleaned\').
    explanation : str
        Brief description of the standardization pipeline performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename_id : str
            Full filename with unique ID for the new resource with standardized data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names (includes comments for each step).
        - preview : list of dict
            Preview of the first 5 rows of the final dataset.
        - protocol_summary : dict
            Summary of the protocol settings used:
            - stereo_policy, remove_isotopes, disconnect_metals, drop_inorganics
        - final_validation : dict
            Validation statistics (n_valid, n_invalid, validation_rate)
        - note : str
            Explanation of the standardization process.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Examples
    --------
    # General ML use case (default settings)
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename="dataset_raw_A3F2B1D4",
        column_name="smiles"
    )
    # Returns dataset with all intermediate columns plus 'standardized_smiles'
    
    # Drug discovery (keep stereochemistry)
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename="dataset_raw_A3F2B1D4",
        column_name="smiles",
        stereo_policy="keep"
    )
    
    # Coordination chemistry (disconnect metals)
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename="dataset_raw_A3F2B1D4",
        column_name="smiles",
        enable_metal_disconnection=True,
        drop_inorganics=True
    )
    
    # Radiolabeling study (keep isotopes)
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename="dataset_raw_A3F2B1D4",
        column_name="smiles",
        skip_isotope_removal=True
    )
    
    Notes
    -----
    This function creates many intermediate columns with detailed comments:
    - smiles_after_canonicalization + comments_after_canonicalization
    - smiles_after_salt_removal + comments_after_salt_removal
    - smiles_after_solvent_removal + comments_after_solvent_removal
    - smiles_after_defragmentation + comments_after_defragmentation
    - smiles_after_normalization + comments_after_normalization
    - smiles_after_reionization + comments_after_reionization
    - smiles_after_neutralization + comments_after_neutralization
    - [optional] smiles_after_isotope_removal + comments_after_isotope_removal
    - [optional] smiles_after_metal_disconnection + comments_after_metal_disconnection
    - [optional] smiles_after_re_defragmentation + comments_after_re_defragmentation
    - smiles_after_tautomer_canonicalization + comments_after_tautomer_canonicalization
    - smiles_after_stereo_standardization + comments_after_stereo_standardization
    - validation_status + validation_comments
    - **standardized_smiles** (final output, copy of last valid step)
    
    **IMPORTANT**: The tautomer_canonicalization step can REMOVE or CHANGE 
    stereochemistry. Check the comments_after_tautomer_canonicalization column 
    for warnings. This is a known RDKit limitation.
    
    The full audit trail allows you to:
    - Track exactly what happened at each step
    - Identify which steps caused the most issues
    - Debug problematic molecules
    - Understand the complete transformation history
    - **Detect when stereochemistry was lost** during tautomer canonicalization
    
    See Also
    --------
    default_SMILES_standardization_pipeline : Simpler list-based version
    get_SMILES_standardization_guidelines : Detailed protocol documentation
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}")
    
    current_filename = input_filename
    current_column = column_name
    
    # Step 1: Initial canonicalization
    result = canonicalize_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step1_canonicalized", "Step 1: Initial canonicalization"
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_canonicalization"
    
    # Step 2: Salt removal
    result = remove_salts_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step2_desalted", "Step 2: Salt removal",
        salt_smarts=salt_smarts
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_salt_removal"
    
    # Step 3: Solvent removal
    result = remove_common_solvents_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step3_desolvated", "Step 3: Solvent removal"
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_solvent_removal"
    
    # Step 4: Defragmentation
    result = defragment_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step4_defragmented", "Step 4: Defragmentation",
        keep_largest_fragment=True
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_defragmentation"
    
    # Step 5: Functional group normalization
    result = normalize_functional_groups_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step5_normalized", "Step 5: Functional group normalization"
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_functional_group_normalization"
    
    # Step 6: Reionization
    result = reionize_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step6_reionized", "Step 6: Reionization"
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_reionization"
    
    # Step 7: Neutralization
    result = neutralize_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step7_neutralized", "Step 7: Neutralization"
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_neutralization"
    
    # Step 8 (OPTIONAL): Isotope removal
    if not skip_isotope_removal:
        result = remove_isotopes_dataset(
            current_filename, current_column, project_manifest_path,
            f"{output_filename}_step8_deisotoped", "Step 8: Isotope removal"
        )
        current_filename = result["output_filename_id"]
        current_column = "smiles_after_isotope_removal"
    
    # Step 9 (OPTIONAL): Metal disconnection
    if enable_metal_disconnection:
        result = disconnect_metals_smiles_dataset(
            current_filename, current_column, project_manifest_path,
            f"{output_filename}_step9_metals_disconnected", "Step 9: Metal disconnection",
            drop_inorganics=drop_inorganics
        )
        current_filename = result["output_filename_id"]
        current_column = "smiles_after_metal_disconnection"
        
        # Step 9b: Re-defragmentation after metal disconnection
        result = defragment_smiles_dataset(
            current_filename, current_column, project_manifest_path,
            f"{output_filename}_step9b_redefragmented", "Step 9b: Re-defragmentation",
            keep_largest_fragment=True
        )
        current_filename = result["output_filename_id"]
        current_column = "smiles_after_re_defragmentation"
    
    # Step 10: Tautomer canonicalization
    result = canonicalize_tautomers_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step10_tautomers", "Step 10: Tautomer canonicalization"
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_tautomer_canonicalization"
    
    # Step 11: Stereochemistry standardization
    result = standardize_stereochemistry_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step11_stereo", "Step 11: Stereochemistry standardization",
        stereo_policy=stereo_policy,
        assign_policy="first",
        max_isomers=32,
        try_embedding=False,
        only_unassigned=True,
        random_seed=42
    )
    current_filename = result["output_filename_id"]
    current_column = "smiles_after_stereo_standardization"
    
    # Step 12: Final validation
    result = validate_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step12_validated", "Step 12: Final validation"
    )
    current_filename = result["output_filename_id"]
    
    # Add final 'standardized_smiles' column (copy of the last valid SMILES column)
    df_final = _load_resource(project_manifest_path, current_filename)
    df_final['standardized_smiles'] = df_final[current_column]
    
    output_filename_id = _store_resource(df_final, project_manifest_path, output_filename, explanation, 'csv')
    
    # Compute validation statistics from the validation result
    n_valid = result.get("n_valid", 0)
    n_invalid = result.get("n_invalid", 0)
    validation_rate = result.get("validation_rate", 0.0)
    
    return {
        "output_filename_id": output_filename_id,
        "n_rows": len(df_final),
        "columns": list(df_final.columns),
        "preview": df_final.head(5).to_dict(orient="records"),
        "protocol_summary": {
            "stereo_policy": stereo_policy,
            "skip_isotope_removal": skip_isotope_removal,
            "enable_metal_disconnection": enable_metal_disconnection,
            "drop_inorganics": drop_inorganics if enable_metal_disconnection else "N/A",
            "salt_smarts": "default (SMARTS_COMMON_SALTS)" if salt_smarts == SMARTS_COMMON_SALTS else "custom"
        },
        "final_validation": {
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "validation_rate": validation_rate
        },
        "note": (
            f"Applied default SMILES standardization protocol with {11 + (0 if skip_isotope_removal else 1) + (2 if enable_metal_disconnection else 0)} steps. "
            f"Settings: stereo_policy='{stereo_policy}', skip_isotope_removal={skip_isotope_removal}, enable_metal_disconnection={enable_metal_disconnection}. "
            f"Final standardized SMILES are in 'standardized_smiles' column. "
            f"All intermediate steps have dedicated comment columns for full audit trail. "
            f"Validation: {n_valid}/{len(df_final)} molecules passed ({validation_rate:.1f}%)."
        )
    }


from molml_mcp.tools.core_mol.pains import _check_smiles_for_pains

@loggable
def check_smiles_for_pains(smiles: list[str]) -> list[str]:
    """Screen multiple molecules for PAINS (Pan-Assay INterference compoundS) substructures.
    
    Screens a list of SMILES against 480 PAINS patterns from Baell & Holloway 2010.
    PAINS are substructures that cause false positives in biological assays through
    non-specific binding, aggregation, redox activity, or assay interference.
    
    Args:
        smiles: List of SMILES strings to screen (e.g., ['CCO', 'O=C1C=CC(=O)C=C1'])
        
    Returns:
        list[str]: Screening results for each molecule, in one of three formats:
                   - 'Passed' if no PAINS patterns detected (molecule is clean)
                   - 'PAINS: <reasons>' if PAINS detected, with comma-separated explanations
                   - 'Failed: <error>' if input is invalid or cannot be parsed
                   
    Example:
        check_smiles_for_pains(['CCO', 'O=C1C=CC(=O)C=C1', 'Oc1ccccc1O'])
        # Returns:
        # ['Passed',
        #  'PAINS: quinone; redox cycling electrophile',
        #  'PAINS: catechol; redox-active metal chelator']
        
    Note:
        PAINS patterns are highly specific to problematic scaffolds observed in
        high-throughput screening, not generic functional group filters.
        
    Reference:
        Baell JB, Holloway GA. J Med Chem 53 (2010) 2719-2740. doi:10.1021/jm901137j
    """

    pains_hits = [_check_smiles_for_pains(smi) for smi in smiles]

    return pains_hits


@loggable
def check_smiles_for_pains_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Screen molecules for PAINS patterns"
) -> dict:
    """Screen molecules for PAINS (Pan-Assay INterference compoundS) patterns in a dataset column.
    
    Args:
        input_filename: Base filename of the input dataset (e.g., \'dataset_raw_A3F2B1D4\')
        column_name: Column with SMILES to screen
        project_manifest_path: Path to the project manifest file for tracking this resource
        output_filename: Base filename for the stored resource (without extension)
        explanation: Brief description of the PAINS screening performed
        
    Returns:
        dict with output_filename_id, n_rows, columns, preview, pains_summary (counts), 
        n_clean, n_flagged, n_failed, flagged_rate, note, suggestions
        
    Adds column: pains_screening with results "Passed", "PAINS: <reasons>", or "Failed: <error>"
    PAINS are context-specific; not all flagged molecules are problematic in all assays.
    Reference: Baell & Holloway, J Med Chem 53 (2010) 2719-2740
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}")
    
    smiles_list = df[column_name].tolist()
    pains_results = check_smiles_for_pains(smiles_list)
    
    df['pains_screening'] = pains_results
    
    output_filename_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    # Calculate screening statistics
    n_clean = sum(1 for r in pains_results if r == "Passed")
    n_flagged = sum(1 for r in pains_results if r.startswith("PAINS:"))
    n_failed = sum(1 for r in pains_results if r.startswith("Failed:"))
    flagged_rate = (n_flagged / len(pains_results) * 100) if pains_results else 0.0
    
    # Count different result types
    from collections import Counter
    pains_summary = dict(Counter(pains_results))
    
    return {
        "output_filename_id": output_filename_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
        "pains_summary": pains_summary,
        "n_clean": n_clean,
        "n_flagged": n_flagged,
        "n_failed": n_failed,
        "flagged_rate": flagged_rate,
        "note": (
            f"PAINS screening complete. {n_clean} molecules passed (clean), "
            f"{n_flagged} flagged as PAINS ({flagged_rate:.1f}%), {n_failed} failed screening. "
            f"Results are in 'pains_screening' column. PAINS patterns are context-specific "
            f"and were identified from high-throughput screening data. Not all flagged molecules "
            f"are necessarily problematic in all assays."
        ),
        "suggestions": (
            "Review flagged molecules in the 'pains_screening' column. Consider filtering out "
            "PAINS-containing molecules before HTS campaigns, but evaluate on a case-by-case basis "
            "for hit validation. Some approved drugs contain PAINS-like substructures. "
            "Filter clean molecules with: df[df['pains_screening'] == 'Passed']"
        )
    }


def get_all_cleaning_tools():
    """Return a list of all molecular cleaning tools."""
    return [
        default_SMILES_standardization_pipeline,
        default_SMILES_standardization_pipeline_dataset,
        get_SMILES_standardization_guidelines,
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
        normalize_functional_groups,
        normalize_functional_groups_dataset,
        reionize_smiles,
        reionize_smiles_dataset,
        disconnect_metals_smiles,
        disconnect_metals_smiles_dataset,
        validate_smiles,
        validate_smiles_dataset,
        check_smiles_for_pains,
        check_smiles_for_pains_dataset,
    ]



