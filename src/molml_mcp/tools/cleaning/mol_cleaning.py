from collections import Counter
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.tools.core_mol.smiles_ops import _canonicalize_smiles, _remove_pattern, _strip_common_solvent_fragments, _defragment_smiles, _normalize_smiles, _reionize_smiles, _disconnect_metals_smiles, _validate_smiles

from molml_mcp.constants import SMARTS_COMMON_SALTS


def get_SMILES_standardization_guidelines() -> str:
    """Return comprehensive guidelines for the default SMILES standardization protocol.
    
    Returns
    -------
    str
        Multi-line string with 11-step protocol, rationale, policy decisions, and use case recommendations.
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

Step 10 (OPTIONAL): Tautomer Canonicalization
    Function: canonicalize_tautomers_dataset(input_filename, column_name, project_manifest_path, output_filename)
    Purpose: Standardize keto-enol, imine-enamine, and other tautomeric forms
             to RDKit's canonical tautomer
    Rationale: Tautomers are chemically equivalent and should have identical
               representations for deduplication and ML
    DEFAULT: Included in standard protocol (most use cases benefit from tautomer standardization)
    WHEN TO SKIP: When specific tautomeric forms are biologically relevant, or when
                  stereochemistry preservation is critical (tautomer canonicalization
                  can remove/change stereochemistry - known RDKit limitation)
    ARGUMENT: skip_tautomer_canonicalization=False (default) or True
    WARNING: Canonical tautomer may not be the predominant form under specific
             conditions (pH, solvent). Can REMOVE or CHANGE stereochemistry.
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
   Argument: skip_isotope_removal=False
   Alternative: True (keep isotopes for specialized studies)
   
3. **Tautomer Canonicalization**: ENABLED (standardize tautomers)
   Argument: skip_tautomer_canonicalization=False
   Alternative: True (preserve specific tautomeric forms or protect stereochemistry)

4. **Metal Disconnection**: DISABLED (keep metal complexes intact)
   Argument: enable_metal_disconnection=False
   Alternative: True (break metal-ligand bonds)

5. **Salt Pattern**: DEFAULT (pharmaceutical salts)
   Argument: salt_smarts=SMARTS_COMMON_SALTS
   Alternative: Custom SMARTS (only for specialized datasets)

6. **Defragmentation**: ENABLED (keep largest fragment)
   Argument: keep_largest_fragment=True
   Alternative: False (keep all fragments)

--------------------------------------------------------------------------------
USE CASE RECOMMENDATIONS
--------------------------------------------------------------------------------

**GENERAL ML / QSAR / ADMET Prediction:**
Use default protocol as-is:
- stereo_policy="flatten"
- skip_isotope_removal=False
- skip_tautomer_canonicalization=False
- enable_metal_disconnection=False

**Drug Discovery / SAR Studies:**
Modify stereochemistry policy:
- stereo_policy="keep"  (enantiomers may have different activities)
- skip_isotope_removal=False
- skip_tautomer_canonicalization=False (or True if tautomer forms are activity-relevant)
- enable_metal_disconnection=False

**Stereochemistry-Critical Analysis:**
Disable tautomer canonicalization to protect stereochemistry:
- stereo_policy="keep"
- skip_isotope_removal=False
- skip_tautomer_canonicalization=True  (prevents loss of stereochemistry)
- enable_metal_disconnection=False

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

For deduplication, use the final SMILES column to identify duplicates. It is 
recommended to find_duplicates_dataset and deduplicate_dataset after standardization 
to make informed decisions on handling duplicates before actually deduplicating.

================================================================================
END OF PROTOCOL
================================================================================
"""
    return guidelines



def canonicalize_smiles(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Convert SMILES strings to canonical form using RDKit.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to canonicalize.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (canonical_smiles, comments). Comments: "Passed" or "Failed: <reason>".
    """
    canonic, comment = [], []
    for smi in smiles:
        c, com = _canonicalize_smiles(smi)
        canonic.append(c)
        comment.append(com)

    return canonic, comment


def canonicalize_smiles_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Canonicalize all SMILES strings"
) -> dict:
    """Canonicalize SMILES strings in a dataset column.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to canonicalize.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments (counts), preview, note, suggestions, question_to_user.
        Adds columns: smiles_after_canonicalization, comments_after_canonicalization.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    canonical_smiles, comments = canonicalize_smiles(smiles_list)

    df['smiles_after_canonicalization'] = canonical_smiles
    df['comments_after_canonicalization'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful canonicalization is marked by 'Passed' in comments, failure is marked by 'Failed: <reason>'.",
        "suggestions": "Consider further cleaning steps such as salt removal, tautomer canonicalization, charge neutralization, and stereochemistry handling.",
        "question_to_user": "Would you like to review failed SMILES entries or drop them from the dataset?",
    }


def remove_salts(smiles: list[str], salt_smarts: str = SMARTS_COMMON_SALTS) -> tuple[list[str], list[str]]:
    """Remove common salt ions from SMILES strings.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to process.
    salt_smarts : str
        SMARTS pattern for salts. Default: "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]". Do not change unless specialized use case.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (desalted_smiles, comments). Comments: "Passed" or "Failed: <reason>".
    """
    new_smiles, comments = [], []
    for smi in smiles:
        cleaned_smi, comment = _remove_pattern(smi, salt_smarts)
        new_smiles.append(cleaned_smi)
        comments.append(comment)
    
    return new_smiles, comments


def remove_salts_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Remove salt counterions from molecules",
    salt_smarts: str = SMARTS_COMMON_SALTS
) -> dict:
    """Remove salt ions from SMILES in a dataset column.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to desalt.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    salt_smarts : str
        SMARTS pattern for salts. Default: common pharmaceutical salts. Do not change unless specialized use case.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments (counts), preview, note, suggestions, question_to_user.
        Adds columns: smiles_after_salt_removal, comments_after_salt_removal.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    desalted_smiles, comments = remove_salts(smiles_list, salt_smarts)

    df['smiles_after_salt_removal'] = desalted_smiles
    df['comments_after_salt_removal'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful salt removal is marked by 'Passed' in comments, failure is marked by 'Failed: <reason>'.",
        "suggestions": "Consider further cleaning steps such as canonicalization, charge neutralization, and stereochemistry handling.",
        "question_to_user": "Would you like to review failed SMILES entries or drop them from the dataset?",
    }


def remove_common_solvents(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Remove common solvent fragments from SMILES strings.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to process. May contain fragmented SMILES.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (cleaned_smiles, comments). Comments: "Pass", "Removed solvents", "All fragments are common solvents...", or "SMILES string is fragmented...".
    """
    new_smiles, comments = [], []
    for smi in smiles:
        cleaned_smi, comment = _strip_common_solvent_fragments(smi)
        new_smiles.append(cleaned_smi)
        comments.append(comment)
    
    return new_smiles, comments


def remove_common_solvents_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Remove common solvent molecules"
) -> dict:
    """Remove common solvent fragments from SMILES in a dataset column.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to process.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments (counts), preview, note, suggestions, question_to_user.
        Adds columns: smiles_after_solvent_removal, comments_after_solvent_removal.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    cleaned_smiles, comments = remove_common_solvents(smiles_list)

    df['smiles_after_solvent_removal'] = cleaned_smiles
    df['comments_after_solvent_removal'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successfully removed solvents marked by 'Pass, removed solvents' in comments or by 'Pass' when no fragments were found.",
        "suggestions": "Consider further cleaning steps such as salt removal, canonicalization, and charge neutralization.",
        "question_to_user": "Would you like to review SMILES where all fragments were solvents and/or SMILES where fragmented, but no solvents were found?",
    }


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


def defragment_smiles_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Remove disconnected fragments from SMILES",
    keep_largest_fragment: bool = True
) -> dict:
    """Remove smaller fragments from SMILES in a dataset column.
    
    WARNING: Remove salts/solvents BEFORE this. Uses SMILES string length (imperfect heuristic).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to defragment.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    keep_largest_fragment : bool
        If True, keeps largest fragment. If False, keeps original if fragmented.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments (counts), preview, note, warning, suggestions, question_to_user.
        Adds columns: smiles_after_defragmentation, comments_after_defragmentation.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    defragmented_smiles, comments = defragment_smiles(smiles_list, keep_largest_fragment)

    df['smiles_after_defragmentation'] = defragmented_smiles
    df['comments_after_defragmentation'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successfully defragmented SMILES marked by 'Pass' or 'Pass, defragmented to largest component' in comments.",
        "warning": "The largest-fragment heuristic is based on SMILES string length and is NOT bulletproof. Always verify results, especially for complex molecules.",
        "suggestions": "Review entries marked as 'Unresolved' if keep_largest_fragment was False. Consider canonicalization and charge neutralization as next steps.",
        "question_to_user": "Would you like to review defragmented SMILES to verify the correct fragments were kept?",
    }


def neutralize_smiles(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Neutralize charged molecules to neutral forms (amines, carboxylic acids, etc.).
    
    Output is neutralized AND canonicalized.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to neutralize.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (neutralized_smiles, comments). Comments: "Passed" or "Failed: <reason>".
    """
    from molml_mcp.tools.core_mol.smiles_ops import _initialise_neutralisation_reactions, _neutralize_smiles
    
    neutralization_transformations = _initialise_neutralisation_reactions()

    neutralized_smiles, comments = [], []   
    for smi in smiles:
        new_smi, comment = _neutralize_smiles(smi, neutralization_transformations)
        neutralized_smiles.append(new_smi)
        comments.append(comment)

    return neutralized_smiles, comments


def neutralize_smiles_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Neutralize charges in molecules"
) -> dict:
    """Neutralize charged molecules in a dataset column. Output is neutralized AND canonicalized.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to neutralize.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments (counts), preview, note, warning, suggestions, question_to_user.
        Adds columns: smiles_after_neutralization, comments_after_neutralization.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    neutralized_smiles, comments = neutralize_smiles(smiles_list)

    df['smiles_after_neutralization'] = neutralized_smiles
    df['comments_after_neutralization'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful neutralization is marked by 'Passed' in comments. Output SMILES are both neutralized AND canonicalized - no additional canonicalization step is needed.",
        "warning": "Neutralization may not be appropriate for quaternary ammonium salts, zwitterions, or permanently charged drug molecules. Review your dataset to ensure neutralization is chemically meaningful.",
        "suggestions": "Consider reviewing molecules that failed neutralization. You may also want to perform tautomer canonicalization or stereochemistry handling as next steps.",
        "question_to_user": "Would you like to review failed neutralizations or molecules with specific charge states before proceeding?",
    }


def standardize_stereochemistry(
    smiles: list[str],
    stereo_policy: str = "keep",
    assign_policy: str = "first",
    max_isomers: int = 32,
    try_embedding: bool = False,
    only_unassigned: bool = True,
    random_seed: int = 42
) -> tuple[list[str], list[str]]:
    """Standardize stereochemistry: keep, assign, or flatten. Output is always canonicalized.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to process.
    stereo_policy : str
        "keep" (preserve), "assign" (enumerate & select), or "flatten" (remove all).
    assign_policy : str
        For stereo_policy="assign": "first", "random", or "lowest" (energy-based).
    max_isomers : int
        Max stereoisomers to enumerate.
    try_embedding : bool
        Try 3D embedding to prune degenerate stereoisomers.
    only_unassigned : bool
        Only enumerate unassigned stereocenters.
    random_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (standardized_smiles, comments). Comments: "Passed" or "Failed: <reason>".
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
    """Standardize stereochemistry in dataset: keep, assign, or flatten. Output always canonicalized.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to process.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    stereo_policy : str
        "keep", "assign", or "flatten".
    assign_policy : str
        For assign: "first", "random", or "lowest".
    max_isomers : int
        Max stereoisomers to enumerate.
    try_embedding : bool
        Try 3D embedding.
    only_unassigned : bool
        Only enumerate unassigned stereocenters.
    random_seed : int
        Random seed.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments, preview, note, suggestions, question_to_user.
        Adds columns: smiles_after_stereo_standardization, comments_after_stereo_standardization.
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

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    policy_notes = {
        "keep": "Existing stereochemistry has been preserved and canonicalized.",
        "assign": f"Stereoisomers have been enumerated and selected using '{assign_policy}' policy.",
        "flatten": "All stereochemical information has been removed. Stereoisomers are now indistinguishable."
    }

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": f"Successful standardization is marked by 'Passed' in comments. Output SMILES are canonicalized. Policy used: '{stereo_policy}'. {policy_notes.get(stereo_policy, '')}",
        "suggestions": "Review molecules that failed standardization. If using 'flatten' policy, consider deduplicating the dataset as stereoisomers are now identical. If using 'assign', verify that the automated selection is appropriate for your use case.",
        "question_to_user": f"You used stereo_policy='{stereo_policy}'. Is this appropriate for your analysis? Would you like to review the results or try a different policy?",
    }


def remove_isotopes(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Remove isotopic labels (e.g., [2H], [13C], [18F]). Output is de-isotoped AND canonicalized.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to process.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (de_isotoped_smiles, comments). Comments: "Passed" or "Failed: <reason>".
    """
    from molml_mcp.tools.core_mol.smiles_ops import _remove_isotopes
    
    clean_smiles, comments = [], []
    for smi in smiles:
        clean_smi, comment = _remove_isotopes(smi)
        clean_smiles.append(clean_smi)
        comments.append(comment)
    
    return clean_smiles, comments


def remove_isotopes_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Remove isotope labels from molecules"
) -> dict:
    """Remove isotopic labels from SMILES in dataset. Output is de-isotoped AND canonicalized.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to process.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments, preview, note, warning, suggestions, question_to_user.
        Adds columns: smiles_after_isotope_removal, comments_after_isotope_removal.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    clean_smiles, comments = remove_isotopes(smiles_list)

    df['smiles_after_isotope_removal'] = clean_smiles
    df['comments_after_isotope_removal'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful isotope removal is marked by 'Passed' in comments. Output SMILES are both de-isotoped AND canonicalized - no additional canonicalization step is needed.",
        "warning": "All isotopic labels have been removed. This may affect studies involving radiolabeling, NMR spectroscopy, or isotope tracing experiments. This operation is irreversible.",
        "suggestions": "Review molecules that failed isotope removal. Consider whether this affects your analysis if working with radiolabeled compounds or isotope tracing studies.",
        "question_to_user": "Are you working with any isotope-labeled compounds that require special handling?",
    }


def canonicalize_tautomers(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Canonicalize tautomers to standard forms. Output is tautomer-canonicalized AND canonicalized.
    
    WARNING: Can REMOVE or CHANGE stereochemistry (RDKit limitation).
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to process.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (canonical_tautomers, comments). Comments: "Passed" or "Failed: <reason>".
    """
    from molml_mcp.tools.core_mol.smiles_ops import _canonicalize_tautomer_smiles
    
    canonical_tautomers, comments = [], []
    for smi in smiles:
        canon_smi, comment = _canonicalize_tautomer_smiles(smi)
        canonical_tautomers.append(canon_smi)
        comments.append(comment)
    
    return canonical_tautomers, comments


def canonicalize_tautomers_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Canonicalize tautomers to standard forms"
) -> dict:
    """Canonicalize tautomers in dataset. Output is tautomer-canonicalized AND canonicalized.
    
    WARNING: Can REMOVE or CHANGE stereochemistry (RDKit limitation).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to process.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments, preview, note, warning, suggestions, question_to_user.
        Adds columns: smiles_after_tautomer_canonicalization, comments_after_tautomer_canonicalization.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    canonical_tautomers, comments = canonicalize_tautomers(smiles_list)

    df['smiles_after_tautomer_canonicalization'] = canonical_tautomers
    df['comments_after_tautomer_canonicalization'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful tautomer canonicalization is marked by 'Passed' in comments. Output SMILES are both tautomer-canonicalized AND canonicalized - no additional canonicalization step is needed. Different tautomers of the same molecule now have identical SMILES representations.",
        "warning": "The canonical tautomer selected may not be the predominant form under your specific experimental conditions (pH, solvent, etc.). For some molecules, the biologically active form may be a non-canonical tautomer.",
        "suggestions": "Review molecules that failed tautomer canonicalization. Consider deduplicating the dataset as different tautomers are now identical. If specific tautomeric forms are important for your analysis, verify that canonicalization is appropriate.",
        "question_to_user": "Are there any molecules in your dataset where specific tautomeric forms are biologically or chemically important? Would you like to review the tautomer canonicalization results?",
    }


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
        dict with output_filename, n_rows, columns, comments (counts), preview, note, suggestions
        
    Adds columns: smiles_after_functional_group_normalization, comments_after_functional_group_normalization
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    normalized_smiles, comments = normalize_functional_groups(smiles_list)

    df['smiles_after_functional_group_normalization'] = normalized_smiles
    df['comments_after_functional_group_normalization'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful normalization is marked by 'Passed' in comments. Functional groups have been standardized to their preferred representations. The output SMILES are canonical and isomeric.",
        "suggestions": "Review molecules that failed normalization. Consider proceeding with charge neutralization and tautomer canonicalization steps.",
    }


def reionize_smiles(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Reionize molecules to preferred charge distribution. Output is reionized AND canonicalized.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to process.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (reionized_smiles, comments). Comments: "Passed" or "Failed: <reason>".
    """
    results = [_reionize_smiles(smi) for smi in smiles]
    reionized_smiles = [smi for smi, _ in results]
    comments = [cmt for _, cmt in results]
    return reionized_smiles, comments


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
        dict with output_filename, n_rows, columns, comments (counts), preview, note, warning, suggestions
        
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

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful reionization is marked by 'Passed' in comments. Charge distributions have been adjusted to chemically preferred forms. The output SMILES are canonical and isomeric.",
        "warning": "Reionization may change charge states in ways that don't match specific experimental conditions (pH, solvent, etc.). Zwitterions may be converted to different forms.",
        "suggestions": "Review molecules that failed reionization. Consider proceeding with charge neutralization (if complete charge removal desired) or tautomer canonicalization steps.",
    }


def disconnect_metals_smiles(smiles: list[str], drop_inorganics: bool = False) -> tuple[list[str], list[str]]:
    """Disconnect metal-ligand bonds. Output is disconnected AND canonicalized.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to process.
    drop_inorganics : bool
        If True, filter out molecules without carbon.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (disconnected_smiles, comments). Comments: "Passed", "Failed: <reason>", or "Failed: Inorganic...".
    """
    results = [_disconnect_metals_smiles(smi, drop_inorganics=drop_inorganics) for smi in smiles]
    disconnected_smiles = [smi for smi, _ in results]
    comments = [cmt for _, cmt in results]
    return disconnected_smiles, comments


def disconnect_metals_smiles_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Disconnect metal-ligand coordinate bonds",
    drop_inorganics: bool = False
) -> dict:
    """Disconnect metal-ligand bonds in dataset. Output is disconnected AND canonicalized.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to process.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    drop_inorganics : bool
        If True, filter out molecules without carbon.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, comments, preview, note, warning, suggestions.
        Adds columns: smiles_after_metal_disconnection, comments_after_metal_disconnection.
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    disconnected_smiles, comments = disconnect_metals_smiles(smiles_list, drop_inorganics=drop_inorganics)

    df['smiles_after_metal_disconnection'] = disconnected_smiles
    df['comments_after_metal_disconnection'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    inorganic_note = " Purely inorganic molecules (no carbon) have been dropped." if drop_inorganics else ""

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": f"Successful metal disconnection is marked by 'Passed' in comments. Coordinate bonds between metals and ligands have been broken. The output SMILES are canonical and isomeric.{inorganic_note}",
        "warning": "Metal coordination may be essential for biological activity in metallodrugs and metal-dependent enzyme inhibitors. Metal disconnection removes structural information about coordination geometry. Consider whether this is appropriate for your analysis.",
        "suggestions": "Review molecules that failed metal disconnection. Consider running defragment_smiles_dataset again to remove disconnected metal fragments. Proceed with functional group normalization and other cleaning steps.",
    }


def validate_smiles(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Validate SMILES for parseability and sanitization. Returns ORIGINAL SMILES unchanged.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to validate.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (validated_smiles, comments). Validated_smiles: original if valid, None if invalid. Comments: "Passed" or "Failed: <reason>".
    """
    results = [_validate_smiles(smi) for smi in smiles]
    validated_smiles = [smi for smi, _ in results]
    comments = [cmt for _, cmt in results]
    return validated_smiles, comments


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
        - output_filename : str
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
    result8 = canonicalize_tautomers_dataset(input_filename=result7["output_filename"], 
                                             column_name="smiles_after_reionization")
    # Step 9: Final validation
    result9 = validate_smiles_dataset(input_filename=result8["output_filename"], 
                                      column_name="smiles_after_tautomer_canonicalization")
    
    # Filter to only valid molecules
    df = _load_resource(project_manifest_path, result9["output_filename"])
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

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    # Calculate validation statistics
    n_valid = sum(1 for c in comments if c == "Passed")
    n_invalid = len(comments) - n_valid
    validation_rate = (n_valid / len(comments) * 100) if comments else 0.0

    return {
        "output_filename": output_filename,
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



def default_SMILES_standardization_pipeline(
    smiles: list[str],
    stereo_policy: str = "flatten",
    skip_isotope_removal: bool = False,
    skip_tautomer_canonicalization: bool = False,
    enable_metal_disconnection: bool = False,
    drop_inorganics: bool = False,
    salt_smarts: str = SMARTS_COMMON_SALTS
) -> tuple[list[str], list[str]]:
    """Apply 11-step SMILES standardization protocol. Returns concise comments.
    
    Parameters
    ----------
    smiles : list[str]
        SMILES strings to standardize.
    stereo_policy : str
        "flatten" (default), "keep", or "assign".
    skip_isotope_removal : bool
        Skip removing isotopes (default: False).
    skip_tautomer_canonicalization : bool
        Skip tautomer canonicalization (default: False). Set True to protect stereochemistry.
    enable_metal_disconnection : bool
        Disconnect metal-ligand bonds (default: False).
    drop_inorganics : bool
        Drop inorganics when disconnecting metals.
    salt_smarts : str
        SMARTS for salt removal. Do not change unless specialized use case.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (standardized_smiles, comments). Comments: "Standardized" or "<step>: <issue>".
    
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
    - This function applies 11 steps of the standardization protocol (some optional)
    - Comments are concise and non-redundant (no repeated "Passed" messages)
    - The function chains: canonicalize → remove_salts → remove_solvents → 
      defragment → normalize → reionize → neutralize → [optional: remove_isotopes] → 
      [optional: disconnect_metals + re-defragment] → [optional: canonicalize_tautomers] → 
      standardize_stereochemistry → validate
    - **IMPORTANT**: Tautomer canonicalization (Step 10) can REMOVE or CHANGE 
      stereochemistry. This is a known RDKit limitation. If stereochemistry 
      preservation is critical, set skip_tautomer_canonicalization=True.
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
        current_smiles, step_comments = remove_isotopes(current_smiles)
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
    
    # Step 10 (OPTIONAL): Tautomer canonicalization
    if not skip_tautomer_canonicalization:
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


def default_SMILES_standardization_pipeline_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Apply default SMILES standardization protocol",
    stereo_policy: str = "flatten",
    skip_isotope_removal: bool = False,
    skip_tautomer_canonicalization: bool = False,
    enable_metal_disconnection: bool = False,
    drop_inorganics: bool = False,
    salt_smarts: str = SMARTS_COMMON_SALTS
) -> dict:
    """Apply 11-step standardization protocol to dataset with full audit trail. Adds comment columns for every step.
    
    WARNING: Tautomer canonicalization can REMOVE/CHANGE stereochemistry. Check comments_after_tautomer_canonicalization.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename.
    column_name : str
        Column with SMILES to standardize.
    project_manifest_path : str
        Path to project manifest.
    output_filename : str
        Output filename (without extension).
    explanation : str
        Description of operation.
    stereo_policy : str
        "flatten" (default), "keep", or "assign".
    skip_isotope_removal : bool
        Skip removing isotopes.
    skip_tautomer_canonicalization : bool
        Skip tautomer canonicalization (protects stereochemistry).
    enable_metal_disconnection : bool
        Disconnect metal-ligand bonds.
    drop_inorganics : bool
        Drop inorganics when disconnecting metals.
    salt_smarts : str
        SMARTS for salt removal. Do not change unless specialized.
    
    Returns
    -------
    dict
        output_filename, n_rows, columns, preview, protocol_summary, final_validation, note, suggestions.
        Adds many intermediate columns with comments for each step, plus 'standardized_smiles'.
    
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
    
    **IMPORTANT**: The tautomer_canonicalization step (if enabled) can REMOVE 
    or CHANGE stereochemistry. Check the comments_after_tautomer_canonicalization 
    column for warnings. This is a known RDKit limitation. To preserve stereochemistry, 
    set skip_tautomer_canonicalization=True.
    
    The full audit trail allows you to:
    - Track exactly what happened at each step
    - Identify which steps caused the most issues
    - Debug problematic molecules
    - Understand the complete transformation history
    - **Detect when stereochemistry was lost** during tautomer canonicalization

    It is recommended to find_duplicates_dataset and deduplicate_dataset after standardization 
    
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
    current_filename = result["output_filename"]
    current_column = "smiles_after_canonicalization"
    
    # Step 2: Salt removal
    result = remove_salts_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step2_desalted", "Step 2: Salt removal",
        salt_smarts=salt_smarts
    )
    current_filename = result["output_filename"]
    current_column = "smiles_after_salt_removal"
    
    # Step 3: Solvent removal
    result = remove_common_solvents_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step3_desolvated", "Step 3: Solvent removal"
    )
    current_filename = result["output_filename"]
    current_column = "smiles_after_solvent_removal"
    
    # Step 4: Defragmentation
    result = defragment_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step4_defragmented", "Step 4: Defragmentation",
        keep_largest_fragment=True
    )
    current_filename = result["output_filename"]
    current_column = "smiles_after_defragmentation"
    
    # Step 5: Functional group normalization
    result = normalize_functional_groups_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step5_normalized", "Step 5: Functional group normalization"
    )
    current_filename = result["output_filename"]
    current_column = "smiles_after_functional_group_normalization"
    
    # Step 6: Reionization
    result = reionize_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step6_reionized", "Step 6: Reionization"
    )
    current_filename = result["output_filename"]
    current_column = "smiles_after_reionization"
    
    # Step 7: Neutralization
    result = neutralize_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step7_neutralized", "Step 7: Neutralization"
    )
    current_filename = result["output_filename"]
    current_column = "smiles_after_neutralization"
    
    # Step 8 (OPTIONAL): Isotope removal
    if not skip_isotope_removal:
        result = remove_isotopes_dataset(
            current_filename, current_column, project_manifest_path,
            f"{output_filename}_step8_deisotoped", "Step 8: Isotope removal"
        )
        current_filename = result["output_filename"]
        current_column = "smiles_after_isotope_removal"
    
    # Step 9 (OPTIONAL): Metal disconnection
    if enable_metal_disconnection:
        result = disconnect_metals_smiles_dataset(
            current_filename, current_column, project_manifest_path,
            f"{output_filename}_step9_metals_disconnected", "Step 9: Metal disconnection",
            drop_inorganics=drop_inorganics
        )
        current_filename = result["output_filename"]
        current_column = "smiles_after_metal_disconnection"
        
        # Step 9b: Re-defragmentation after metal disconnection
        # Note: defragment_smiles_dataset always outputs 'smiles_after_defragmentation'
        result = defragment_smiles_dataset(
            current_filename, current_column, project_manifest_path,
            f"{output_filename}_step9b_redefragmented", "Step 9b: Re-defragmentation",
            keep_largest_fragment=True
        )
        current_filename = result["output_filename"]
        current_column = "smiles_after_defragmentation"  # Fixed: use actual output column name
    
    # Step 10 (OPTIONAL): Tautomer canonicalization
    if not skip_tautomer_canonicalization:
        result = canonicalize_tautomers_dataset(
            current_filename, current_column, project_manifest_path,
            f"{output_filename}_step10_tautomers", "Step 10: Tautomer canonicalization"
        )
        current_filename = result["output_filename"]
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
    current_filename = result["output_filename"]
    current_column = "smiles_after_stereo_standardization"
    
    # Step 12: Final validation
    result = validate_smiles_dataset(
        current_filename, current_column, project_manifest_path,
        f"{output_filename}_step12_validated", "Step 12: Final validation"
    )
    current_filename = result["output_filename"]
    
    # Add final 'standardized_smiles' column (copy of the last valid SMILES column)
    df_final = _load_resource(project_manifest_path, current_filename)
    df_final['standardized_smiles'] = df_final[current_column]
    
    output_filename = _store_resource(df_final, project_manifest_path, output_filename, explanation, 'csv')
    
    # Compute validation statistics from the validation result
    n_valid = result.get("n_valid", 0)
    n_invalid = result.get("n_invalid", 0)
    validation_rate = result.get("validation_rate", 0.0)
    
    return {
        "output_filename": output_filename,
        "n_rows": len(df_final),
        "columns": list(df_final.columns),
        "preview": df_final.head(5).to_dict(orient="records"),
        "suggestions": "Check the comments_after_tautomer_canonicalization column for warnings of changes in stereochemistry. After standardization, drop invalid SMILES. In next steps, consider label curation and de-duplication using find_duplicates_dataset and deduplicate_dataset. When working with experimental data, you might want to filter out PAINS molecules too.",
        "protocol_summary": {
            "stereo_policy": stereo_policy,
            "skip_isotope_removal": skip_isotope_removal,
            "skip_tautomer_canonicalization": skip_tautomer_canonicalization,
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
            f"Applied default SMILES standardization protocol with {11 + (0 if skip_isotope_removal else 1) + (0 if skip_tautomer_canonicalization else 1) + (2 if enable_metal_disconnection else 0)} steps. "
            f"Settings: stereo_policy='{stereo_policy}', skip_isotope_removal={skip_isotope_removal}, skip_tautomer_canonicalization={skip_tautomer_canonicalization}, enable_metal_disconnection={enable_metal_disconnection}. "
            f"Final standardized SMILES are in 'standardized_smiles' column. "
            f"It is recommended to find_duplicates_dataset and deduplicate_dataset after standardization."
            f"All intermediate steps have dedicated comment columns for full audit trail. "
            f"Validation: {n_valid}/{len(df_final)} molecules passed ({validation_rate:.1f}%)."
        )
    }


from molml_mcp.tools.core_mol.pains import _check_smiles_for_pains

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
        dict with output_filename, n_rows, columns, preview, pains_summary (counts), 
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
    
    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    # Calculate screening statistics
    n_clean = sum(1 for r in pains_results if r == "Passed")
    n_flagged = sum(1 for r in pains_results if r.startswith("PAINS:"))
    n_failed = sum(1 for r in pains_results if r.startswith("Failed:"))
    flagged_rate = (n_flagged / len(pains_results) * 100) if pains_results else 0.0
    
    # Count different result types
    from collections import Counter
    pains_summary = dict(Counter(pains_results))
    
    return {
        "output_filename": output_filename,
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
    """Return a list of all molecular cleaning tools exposed to MCP server.
    
    Only exposes dataset-level functions and selected SMILES-level functions:
    - canonicalize_smiles, validate_smiles, default_SMILES_standardization_pipeline
    """
    return [
        # Guidelines
        get_SMILES_standardization_guidelines,
        
        # SMILES-level functions (selected)
        canonicalize_smiles,
        validate_smiles,
        default_SMILES_standardization_pipeline,
        
        # Dataset-level functions
        default_SMILES_standardization_pipeline_dataset,
        canonicalize_smiles_dataset,
        remove_salts_dataset,
        remove_common_solvents_dataset,
        defragment_smiles_dataset,
        neutralize_smiles_dataset,
        standardize_stereochemistry_dataset,
        remove_isotopes_dataset,
        canonicalize_tautomers_dataset,
        normalize_functional_groups_dataset,
        reionize_smiles_dataset,
        disconnect_metals_smiles_dataset,
        validate_smiles_dataset,
        check_smiles_for_pains_dataset,
    ]



