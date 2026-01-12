"""
Dataset filtering tools for property-based selection.
"""
import pandas as pd
from rdkit.Chem import Descriptors, Lipinski
from rdkit import Chem
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.tools.core_mol.pains import _check_smiles_for_pains
from molml_mcp.tools.core_mol.scaffolds import _get_scaffold
from molml_mcp.tools.core_mol.substructure_matching import find_functional_group_patterns_in_smiles


def filter_by_property_range(
    input_filename: str,
    project_manifest_path: str,
    property_ranges: dict[str, tuple[float, float]],
    output_filename: str,
    explanation: str
) -> dict:
    """
    Filter dataset by property ranges (AND logic, inclusive min/max).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    property_ranges : dict[str, tuple[float, float]]
        Dict mapping columns to (min, max) tuples
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, percent_retained, filters_applied, columns, warning, note
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate inputs
    if not property_ranges:
        raise ValueError("property_ranges cannot be empty. Provide at least one property range filter.")
    
    # Check all property columns exist
    missing_columns = [col for col in property_ranges.keys() if col not in df.columns]
    if missing_columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"Property columns not found in dataset: {missing_columns}. "
            f"Available columns: {available_columns}"
        )
    
    # Validate ranges
    for prop, (min_val, max_val) in property_ranges.items():
        if min_val > max_val:
            raise ValueError(
                f"Invalid range for property '{prop}': min ({min_val}) > max ({max_val}). "
                f"Range must have min ≤ max."
            )
    
    # Apply filters
    df_filtered = df.copy()
    filters_applied = []
    
    for prop, (min_val, max_val) in property_ranges.items():
        # Create filter mask
        mask = (df_filtered[prop] >= min_val) & (df_filtered[prop] <= max_val)
        
        # Count before filtering
        n_before = len(df_filtered)
        
        # Apply filter
        df_filtered = df_filtered[mask]
        
        # Track what was applied
        n_after = len(df_filtered)
        n_removed_this_filter = n_before - n_after
        filters_applied.append({
            'property': prop,
            'min': float(min_val),
            'max': float(max_val),
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_filename,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "percent_retained": percent_retained,
        "filters_applied": filters_applied,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ This is a crude filtering tool. Results should be manually validated and used with caution.",
        "note": (
            f"Filtered dataset from {n_input} to {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Applied {len(property_ranges)} property range filters. "
            f"Removed {n_removed} molecules not passing all criteria."
        )
    }


def filter_by_lipinski_ro5(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str
) -> dict:
    """
    Filter by Lipinski's Rule of Five (MW≤500, LogP≤5, HBD≤5, HBA≤10).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    smiles_column : str
        SMILES column name
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles, percent_retained, filters_applied, lipinski_properties_added, columns, warning, note
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Calculate Lipinski properties on-the-fly
    df_with_props = df.copy()
    
    mw_list = []
    logp_list = []
    hbd_list = []
    hba_list = []
    valid_mask = []
    
    for smiles in df_with_props[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw_list.append(Descriptors.MolWt(mol))
            logp_list.append(Descriptors.MolLogP(mol))
            hbd_list.append(Lipinski.NumHDonors(mol))
            hba_list.append(Lipinski.NumHAcceptors(mol))
            valid_mask.append(True)
        else:
            # Invalid SMILES - will be filtered out
            mw_list.append(None)
            logp_list.append(None)
            hbd_list.append(None)
            hba_list.append(None)
            valid_mask.append(False)
    
    # Add properties as columns
    df_with_props['MolWt'] = mw_list
    df_with_props['MolLogP'] = logp_list
    df_with_props['NumHDonors'] = hbd_list
    df_with_props['NumHAcceptors'] = hba_list
    
    # Remove invalid SMILES first
    df_with_props = df_with_props[valid_mask].copy()
    n_invalid = n_input - len(df_with_props)
    
    # Apply Lipinski Rule of Five filters
    property_ranges = {
        'MolWt': (0, 500),
        'MolLogP': (-float('inf'), 5),
        'NumHDonors': (0, 5),
        'NumHAcceptors': (0, 10)
    }
    
    df_filtered = df_with_props.copy()
    filters_applied = []
    
    for prop, (min_val, max_val) in property_ranges.items():
        mask = (df_filtered[prop] >= min_val) & (df_filtered[prop] <= max_val)
        n_before = len(df_filtered)
        df_filtered = df_filtered[mask]
        n_after = len(df_filtered)
        n_removed_this_filter = n_before - n_after
        filters_applied.append({
            'property': prop,
            'min': float(min_val) if min_val != -float('inf') else 'no_limit',
            'max': float(max_val),
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    lipinski_properties = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors']
    
    return {
        "output_filename": output_filename,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "filters_applied": filters_applied,
        "lipinski_properties_added": lipinski_properties,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ Lipinski's Rule of Five is a crude guideline, not a strict rule. Many successful drugs violate these criteria. Use with caution and validate results.",
        "note": (
            f"Lipinski Rule of Five filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Added properties: {', '.join(lipinski_properties)}. "
            f"Criteria: MW≤500, LogP≤5, HBD≤5, HBA≤10."
        )
    }


def filter_by_veber_rules(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str
) -> dict:
    """
    Filter by Veber's rules for oral bioavailability (TPSA≤140, RotBonds≤10).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    smiles_column : str
        SMILES column name
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles, percent_retained, filters_applied, veber_properties_added, columns, warning, note
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Calculate Veber properties on-the-fly
    df_with_props = df.copy()
    
    tpsa_list = []
    rotatable_bonds_list = []
    valid_mask = []
    
    for smiles in df_with_props[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            tpsa_list.append(Descriptors.TPSA(mol))
            rotatable_bonds_list.append(Descriptors.NumRotatableBonds(mol))
            valid_mask.append(True)
        else:
            # Invalid SMILES - will be filtered out
            tpsa_list.append(None)
            rotatable_bonds_list.append(None)
            valid_mask.append(False)
    
    # Add properties as columns
    df_with_props['TPSA'] = tpsa_list
    df_with_props['NumRotatableBonds'] = rotatable_bonds_list
    
    # Remove invalid SMILES first
    df_with_props = df_with_props[valid_mask].copy()
    n_invalid = n_input - len(df_with_props)
    
    # Apply Veber rules filters
    property_ranges = {
        'TPSA': (0, 140),
        'NumRotatableBonds': (0, 10)
    }
    
    df_filtered = df_with_props.copy()
    filters_applied = []
    
    for prop, (min_val, max_val) in property_ranges.items():
        mask = (df_filtered[prop] >= min_val) & (df_filtered[prop] <= max_val)
        n_before = len(df_filtered)
        df_filtered = df_filtered[mask]
        n_after = len(df_filtered)
        n_removed_this_filter = n_before - n_after
        filters_applied.append({
            'property': prop,
            'min': float(min_val),
            'max': float(max_val),
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    veber_properties = ['TPSA', 'NumRotatableBonds']
    
    return {
        "output_filename": output_filename,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "filters_applied": filters_applied,
        "veber_properties_added": veber_properties,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ Veber's rules are crude predictors of oral bioavailability. Many factors beyond TPSA and rotatable bonds affect absorption. Use with caution.",
        "note": (
            f"Veber rules filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Added properties: {', '.join(veber_properties)}. "
            f"Criteria: TPSA≤140, RotatableBonds≤10."
        )
    }


def filter_by_pains(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = "Remove PAINS flagged molecules",
    action: str = 'drop'
) -> dict:
    """
    Filter by PAINS (Pan-Assay INterference compoundS) patterns.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    smiles_column : str
        SMILES column name
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    action : str, default='drop'
        'drop' removes PAINS hits, 'keep' retains only PAINS hits
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_pains_flagged, n_invalid_smiles, percent_retained, action, columns, warning, note
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Validate action
    if action not in ['drop', 'keep']:
        raise ValueError(
            f"Invalid action '{action}'. Must be 'drop' (remove PAINS) or 'keep' (retain only PAINS)."
        )
    
    # Check all molecules for PAINS
    df_with_pains = df.copy()
    pains_results = []
    
    for smiles in df_with_pains[smiles_column]:
        pains_result = _check_smiles_for_pains(smiles)
        pains_results.append(pains_result)
    
    # Add PAINS check results as column
    df_with_pains['pains_check'] = pains_results
    
    # Count different categories
    n_passed = sum(1 for r in pains_results if r == 'Passed')
    n_pains = sum(1 for r in pains_results if r.startswith('PAINS:'))
    n_failed = sum(1 for r in pains_results if r.startswith('Failed:'))
    
    # Apply filter based on action
    if action == 'drop':
        # Keep only molecules that passed PAINS check
        df_filtered = df_with_pains[df_with_pains['pains_check'] == 'Passed'].copy()
    else:  # action == 'keep'
        # Keep only PAINS-flagged molecules
        df_filtered = df_with_pains[df_with_pains['pains_check'].str.startswith('PAINS:')].copy()
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    if action == 'drop':
        note_text = (
            f"PAINS filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_pains} PAINS-flagged and {n_failed} invalid molecules. "
            f"Kept {n_passed} clean molecules."
        )
    else:
        note_text = (
            f"PAINS extraction: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Kept {n_pains} PAINS-flagged molecules for analysis. "
            f"Removed {n_passed} clean and {n_failed} invalid molecules."
        )
    
    return {
        "output_filename": output_filename,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_pains_flagged": n_pains,
        "n_invalid_smiles": n_failed,
        "percent_retained": percent_retained,
        "action": action,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ PAINS filters are controversial and may remove valid compounds. Context-dependent - a PAINS hit in one assay may be fine in another. Use with caution.",
        "note": note_text
    }


def filter_by_lead_likeness(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = "Filter by lead-likeness criteria",
    strict: bool = True
) -> dict:
    """
    Filter by lead-likeness rules (strict: MW:200-350, LogP≤3.5, RotBonds≤7, Rings≥1).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    smiles_column : str
        SMILES column name
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    strict : bool, default=True
        Use strict (True) or lenient (False) criteria
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles, percent_retained, criteria_mode, filters_applied, lead_properties_added, columns, warning, note
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Calculate lead-likeness properties on-the-fly
    df_with_props = df.copy()
    
    mw_list = []
    logp_list = []
    rot_bonds_list = []
    ring_count_list = []
    aromatic_rings_list = []
    total_rings_list = []
    valid_mask = []
    
    for smiles in df_with_props[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            ring_count = Lipinski.RingCount(mol)
            aromatic_rings = Lipinski.NumAromaticRings(mol)
            total_rings = ring_count  # Total ring count
            
            mw_list.append(mw)
            logp_list.append(logp)
            rot_bonds_list.append(rot_bonds)
            ring_count_list.append(ring_count)
            aromatic_rings_list.append(aromatic_rings)
            total_rings_list.append(total_rings)
            valid_mask.append(True)
        else:
            # Invalid SMILES - will be filtered out
            mw_list.append(None)
            logp_list.append(None)
            rot_bonds_list.append(None)
            ring_count_list.append(None)
            aromatic_rings_list.append(None)
            total_rings_list.append(None)
            valid_mask.append(False)
    
    # Add properties as columns
    df_with_props['MolWt'] = mw_list
    df_with_props['MolLogP'] = logp_list
    df_with_props['NumRotatableBonds'] = rot_bonds_list
    df_with_props['RingCount'] = ring_count_list
    df_with_props['NumAromaticRings'] = aromatic_rings_list
    df_with_props['TotalRings'] = total_rings_list
    
    # Remove invalid SMILES first
    df_with_props = df_with_props[valid_mask].copy()
    n_invalid = n_input - len(df_with_props)
    
    # Define lead-likeness criteria based on strict/lenient mode
    if strict:
        property_ranges = {
            'MolWt': (200, 350),
            'MolLogP': (-float('inf'), 3.5),
            'NumRotatableBonds': (0, 7),
            'TotalRings': (1, float('inf'))  # At least 1 ring
        }
        criteria_mode = "strict"
    else:
        property_ranges = {
            'MolWt': (150, 400),
            'MolLogP': (-float('inf'), 4.0),
            'NumRotatableBonds': (0, 10),
            'TotalRings': (1, float('inf'))  # At least 1 ring
        }
        criteria_mode = "lenient"
    
    # Apply filters
    df_filtered = df_with_props.copy()
    filters_applied = []
    
    for prop, (min_val, max_val) in property_ranges.items():
        mask = (df_filtered[prop] >= min_val) & (df_filtered[prop] <= max_val)
        n_before = len(df_filtered)
        df_filtered = df_filtered[mask]
        n_after = len(df_filtered)
        n_removed_this_filter = n_before - n_after
        filters_applied.append({
            'property': prop,
            'min': float(min_val) if min_val != -float('inf') else 'no_limit',
            'max': float(max_val) if max_val != float('inf') else 'no_limit',
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename_stored = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    lead_properties = ['MolWt', 'MolLogP', 'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'TotalRings']
    
    if strict:
        criteria_text = "MW:200-350, LogP≤3.5, RotBonds≤7, Rings≥1"
    else:
        criteria_text = "MW:150-400, LogP≤4.0, RotBonds≤10, Rings≥1"
    
    return {
        "output_filename": output_filename_stored,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "criteria_mode": criteria_mode,
        "filters_applied": filters_applied,
        "lead_properties_added": lead_properties,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ Lead-likeness rules are crude guidelines for hit-to-lead optimization. Optimal ranges vary by target class and project goals. Use with caution.",
        "note": (
            f"Lead-likeness filtering ({criteria_mode}): {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Added properties: {', '.join(lead_properties)}. "
            f"Criteria: {criteria_text}."
        )
    }


def filter_by_rule_of_three(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = "Filter by Rule of Three",
    strict: bool = True
) -> dict:
    """
    Filter by Rule of Three for fragments (strict: MW≤300, LogP≤3, HBD≤3, HBA≤3, RotBonds≤3, TPSA≤60).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    smiles_column : str
        SMILES column name
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    strict : bool, default=True
        Use strict (True) or lenient (False) criteria
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles, percent_retained, criteria_mode, filters_applied, ro3_properties_added, columns, warning, note
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Calculate Rule of Three properties on-the-fly
    df_with_props = df.copy()
    
    mw_list = []
    logp_list = []
    hbd_list = []
    hba_list = []
    rot_bonds_list = []
    tpsa_list = []
    valid_mask = []
    
    for smiles in df_with_props[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw_list.append(Descriptors.MolWt(mol))
            logp_list.append(Descriptors.MolLogP(mol))
            hbd_list.append(Lipinski.NumHDonors(mol))
            hba_list.append(Lipinski.NumHAcceptors(mol))
            rot_bonds_list.append(Descriptors.NumRotatableBonds(mol))
            tpsa_list.append(Descriptors.TPSA(mol))
            valid_mask.append(True)
        else:
            # Invalid SMILES - will be filtered out
            mw_list.append(None)
            logp_list.append(None)
            hbd_list.append(None)
            hba_list.append(None)
            rot_bonds_list.append(None)
            tpsa_list.append(None)
            valid_mask.append(False)
    
    # Add properties as columns
    df_with_props['MolWt'] = mw_list
    df_with_props['MolLogP'] = logp_list
    df_with_props['NumHDonors'] = hbd_list
    df_with_props['NumHAcceptors'] = hba_list
    df_with_props['NumRotatableBonds'] = rot_bonds_list
    df_with_props['TPSA'] = tpsa_list
    
    # Remove invalid SMILES first
    df_with_props = df_with_props[valid_mask].copy()
    n_invalid = n_input - len(df_with_props)
    
    # Define Rule of Three criteria based on strict/lenient mode
    if strict:
        property_ranges = {
            'MolWt': (0, 300),
            'MolLogP': (-float('inf'), 3),
            'NumHDonors': (0, 3),
            'NumHAcceptors': (0, 3),
            'NumRotatableBonds': (0, 3),
            'TPSA': (0, 60)
        }
        criteria_mode = "strict"
    else:
        property_ranges = {
            'MolWt': (0, 350),
            'MolLogP': (-float('inf'), 3.5),
            'NumHDonors': (0, 4),
            'NumHAcceptors': (0, 6),
            'NumRotatableBonds': (0, 5),
            'TPSA': (0, 90)
        }
        criteria_mode = "lenient"
    
    # Apply filters
    df_filtered = df_with_props.copy()
    filters_applied = []
    
    for prop, (min_val, max_val) in property_ranges.items():
        mask = (df_filtered[prop] >= min_val) & (df_filtered[prop] <= max_val)
        n_before = len(df_filtered)
        df_filtered = df_filtered[mask]
        n_after = len(df_filtered)
        n_removed_this_filter = n_before - n_after
        filters_applied.append({
            'property': prop,
            'min': float(min_val) if min_val != -float('inf') else 'no_limit',
            'max': float(max_val),
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename_stored = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    ro3_properties = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'TPSA']
    
    if strict:
        criteria_text = "MW≤300, LogP≤3, HBD≤3, HBA≤3, RotBonds≤3, TPSA≤60"
    else:
        criteria_text = "MW≤350, LogP≤3.5, HBD≤4, HBA≤6, RotBonds≤5, TPSA≤90"
    
    return {
        "output_filename": output_filename_stored,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "criteria_mode": criteria_mode,
        "filters_applied": filters_applied,
        "ro3_properties_added": ro3_properties,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ Rule of Three is a crude guideline for fragment libraries. Fragment quality depends heavily on binding mode and target. Use with caution.",
        "note": (
            f"Rule of Three filtering ({criteria_mode}): {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Added properties: {', '.join(ro3_properties)}. "
            f"Criteria: {criteria_text}."
        )
    }


def filter_by_qed(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = "Filter by QED score",
    min_qed: float = 0.5
) -> dict:
    """
    Filter by QED (Quantitative Estimate of Drug-likeness) score (0-1 scale).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    smiles_column : str
        SMILES column name
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    min_qed : float, default=0.5
        Minimum QED threshold (0-1)
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles, percent_retained, min_qed_threshold, mean_qed, median_qed, columns, warning, note
    """
    from rdkit.Chem import QED
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Validate min_qed threshold
    if not 0 <= min_qed <= 1:
        raise ValueError(
            f"min_qed must be between 0 and 1, got {min_qed}."
        )
    
    # Calculate QED scores on-the-fly
    df_with_qed = df.copy()
    
    qed_scores = []
    valid_mask = []
    
    for smiles in df_with_qed[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                qed_score = QED.qed(mol)
                qed_scores.append(qed_score)
                valid_mask.append(True)
            except:
                # QED calculation failed
                qed_scores.append(None)
                valid_mask.append(False)
        else:
            # Invalid SMILES
            qed_scores.append(None)
            valid_mask.append(False)
    
    # Add QED as column
    df_with_qed['QED'] = qed_scores
    
    # Remove invalid SMILES first
    df_with_qed = df_with_qed[valid_mask].copy()
    n_invalid = n_input - len(df_with_qed)
    
    # Apply QED filter
    df_filtered = df_with_qed[df_with_qed['QED'] >= min_qed].copy()
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Calculate QED statistics for filtered set
    if n_output > 0:
        mean_qed = float(df_filtered['QED'].mean())
        median_qed = float(df_filtered['QED'].median())
    else:
        mean_qed = 0.0
        median_qed = 0.0
    
    # Store output
    output_filename_stored = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_filename_stored,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "min_qed_threshold": min_qed,
        "mean_qed": mean_qed,
        "median_qed": median_qed,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ QED is a crude composite drug-likeness score. High QED doesn't guarantee success, and many approved drugs have low QED. Use with caution.",
        "note": (
            f"QED filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Threshold: QED ≥ {min_qed}. "
            f"Filtered set: mean QED = {mean_qed:.3f}, median QED = {median_qed:.3f}."
        )
    }


def filter_by_scaffold(
    input_filename: str,
    project_manifest_path: str,
    output_filename: str,
    scaffold_smiles_list: list[str],
    explanation: str = "Filter by scaffold",
    action: str = 'keep',
    smiles_column: str = 'smiles'
) -> dict:
    """
    Filter by Bemis-Murcko scaffold membership.
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    scaffold_smiles_list : list[str]
        List of scaffold SMILES to filter by
    explanation : str
        Brief description
    action : str, default='keep'
        'keep' retains matching scaffolds, 'drop' removes them
    smiles_column : str, default='smiles'
        SMILES column name (used if scaffold column missing)
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_matching_scaffold, n_invalid_smiles, n_no_scaffold, percent_retained, action, scaffolds_used, scaffold_column_existed, columns, warning, note
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate action
    if action not in ['keep', 'drop']:
        raise ValueError(
            f"Invalid action '{action}'. Must be 'keep' (retain matching) or 'drop' (remove matching)."
        )
    
    # Validate scaffold list
    if not scaffold_smiles_list:
        raise ValueError("scaffold_smiles_list cannot be empty. Provide at least one scaffold SMILES.")
    
    # Canonicalize scaffold SMILES for comparison
    canonical_scaffolds = set()
    invalid_scaffolds = []
    
    for scaffold_smi in scaffold_smiles_list:
        mol = Chem.MolFromSmiles(scaffold_smi)
        if mol is not None:
            canonical_smi = Chem.MolToSmiles(mol)
            canonical_scaffolds.add(canonical_smi)
        else:
            invalid_scaffolds.append(scaffold_smi)
    
    if invalid_scaffolds:
        raise ValueError(
            f"Invalid scaffold SMILES provided: {invalid_scaffolds}. "
            f"All scaffolds must be valid SMILES strings."
        )
    
    if not canonical_scaffolds:
        raise ValueError("No valid scaffolds provided after canonicalization.")
    
    # Check if scaffold column already exists
    scaffold_column_existed = 'scaffold_bemis_murcko' in df.columns
    
    df_with_scaffold = df.copy()
    
    if not scaffold_column_existed:
        # Need to calculate scaffolds from SMILES
        if smiles_column not in df.columns:
            available_columns = df.columns.tolist()
            raise ValueError(
                f"SMILES column '{smiles_column}' not found and no 'scaffold_bemis_murcko' column exists. "
                f"Available columns: {available_columns}"
            )
        
        # Calculate Bemis-Murcko scaffolds on-the-fly
        scaffold_list = []
        valid_mask = []
        
        for smiles in df_with_scaffold[smiles_column]:
            scaffold_smi, comment = _get_scaffold(smiles, scaffold_type='bemis_murcko')
            if scaffold_smi is None:
                scaffold_list.append(None)
                # Check if it's a valid SMILES but no scaffold vs invalid SMILES
                if comment.startswith('Failed: Invalid'):
                    valid_mask.append(False)
                else:
                    valid_mask.append(True)  # Valid SMILES but no scaffold
            else:
                scaffold_list.append(scaffold_smi)
                valid_mask.append(True)
        
        df_with_scaffold['scaffold_bemis_murcko'] = scaffold_list
        n_invalid = sum(1 for v in valid_mask if not v)
        n_no_scaffold = sum(1 for s in scaffold_list if s is None and valid_mask[scaffold_list.index(s)])
    else:
        # Use existing scaffold column
        n_invalid = 0
        n_no_scaffold = sum(1 for s in df_with_scaffold['scaffold_bemis_murcko'] if pd.isna(s) or s == 'No scaffold' or s == '')
    
    # Filter by scaffold membership
    def matches_scaffold(scaffold_smi):
        if pd.isna(scaffold_smi) or scaffold_smi == 'No scaffold' or scaffold_smi == '' or scaffold_smi is None:
            return False
        return scaffold_smi in canonical_scaffolds
    
    df_with_scaffold['_matches_scaffold'] = df_with_scaffold['scaffold_bemis_murcko'].apply(matches_scaffold)
    
    n_matching = df_with_scaffold['_matches_scaffold'].sum()
    
    # Apply filter based on action
    if action == 'keep':
        df_filtered = df_with_scaffold[df_with_scaffold['_matches_scaffold']].copy()
    else:  # action == 'drop'
        df_filtered = df_with_scaffold[~df_with_scaffold['_matches_scaffold']].copy()
    
    # Drop temporary column
    df_filtered = df_filtered.drop(columns=['_matches_scaffold'])
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename_stored = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    if action == 'keep':
        note_text = (
            f"Scaffold filtering (keep): {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Kept {n_matching} molecules matching {len(canonical_scaffolds)} scaffold(s). "
        )
    else:
        note_text = (
            f"Scaffold filtering (drop): {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_matching} molecules matching {len(canonical_scaffolds)} scaffold(s). "
        )
    
    if not scaffold_column_existed:
        note_text += f"Calculated scaffolds on-the-fly. {n_invalid} invalid SMILES, {n_no_scaffold} molecules with no scaffold."
    else:
        note_text += f"Used existing scaffold column. {n_no_scaffold} molecules with no scaffold."
    
    return {
        "output_filename": output_filename_stored,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_matching_scaffold": int(n_matching),
        "n_invalid_smiles": n_invalid,
        "n_no_scaffold": n_no_scaffold,
        "percent_retained": percent_retained,
        "action": action,
        "scaffolds_used": list(canonical_scaffolds),
        "scaffold_column_existed": scaffold_column_existed,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ Scaffold filtering is a crude structural filter. Molecules with the same scaffold can have vastly different properties. Use with caution.",
        "note": note_text
    }


def filter_by_functional_groups(
    input_filename: str,
    smiles_column: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Filter by functional groups",
    required: list[str] | None = None,
    forbidden: list[str] | None = None
) -> dict:
    """
    Filter by functional group presence/absence (58 patterns detected).
    
    Parameters
    ----------
    input_filename : str
        Input dataset filename
    smiles_column : str
        SMILES column name
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    required : list[str] | None
        Functional groups that MUST be present (AND logic)
    forbidden : list[str] | None
        Functional groups that MUST NOT be present (AND logic)
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles, percent_retained, required_groups, forbidden_groups, filter_summary, columns, warning, note
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Validate inputs
    if required is None:
        required = []
    if forbidden is None:
        forbidden = []
    
    if not required and not forbidden:
        raise ValueError(
            "At least one of 'required' or 'forbidden' must be specified. "
            "Both cannot be None/empty."
        )
    
    # Detect functional groups for each molecule
    df_with_groups = df.copy()
    functional_groups_list = []
    valid_mask = []
    
    for smiles in df_with_groups[smiles_column]:
        if pd.isna(smiles) or smiles == '' or smiles is None:
            functional_groups_list.append(set())
            valid_mask.append(False)
        else:
            groups_str = find_functional_group_patterns_in_smiles(smiles)
            if groups_str:
                # Parse comma-separated groups into set
                groups_set = set(g.strip() for g in groups_str.split(','))
                functional_groups_list.append(groups_set)
                valid_mask.append(True)
            else:
                # Valid SMILES but no functional groups detected
                functional_groups_list.append(set())
                valid_mask.append(True)
    
    df_with_groups['_functional_groups'] = functional_groups_list
    df_with_groups['_valid'] = valid_mask
    
    # Count invalid SMILES
    n_invalid = sum(1 for v in valid_mask if not v)
    
    # Apply filtering logic
    def passes_filter(groups_set):
        # Check required groups (must have ALL)
        if required:
            has_all_required = all(req in groups_set for req in required)
            if not has_all_required:
                return False
        
        # Check forbidden groups (must have NONE)
        if forbidden:
            has_any_forbidden = any(forb in groups_set for forb in forbidden)
            if has_any_forbidden:
                return False
        
        return True
    
    # Filter only valid molecules, then apply functional group filter
    df_valid = df_with_groups[df_with_groups['_valid']].copy()
    df_valid['_passes'] = df_valid['_functional_groups'].apply(passes_filter)
    df_filtered = df_valid[df_valid['_passes']].copy()
    
    # Drop temporary columns
    df_filtered = df_filtered.drop(columns=['_functional_groups', '_valid', '_passes'])
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Calculate filter statistics
    n_failed_required = 0
    n_failed_forbidden = 0
    
    if required:
        for groups_set in df_valid['_functional_groups']:
            if not all(req in groups_set for req in required):
                n_failed_required += 1
    
    if forbidden:
        for groups_set in df_valid['_functional_groups']:
            if any(forb in groups_set for forb in forbidden):
                n_failed_forbidden += 1
    
    # Store output
    output_filename_stored = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    # Build filter summary
    filter_parts = []
    if required:
        filter_parts.append(f"Required: {', '.join(required)}")
    if forbidden:
        filter_parts.append(f"Forbidden: {', '.join(forbidden)}")
    filter_summary = " | ".join(filter_parts)
    
    note_text = (
        f"Functional group filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
    )
    
    if required:
        note_text += f"Required ALL of {len(required)} group(s). "
    if forbidden:
        note_text += f"Forbidden ANY of {len(forbidden)} group(s). "
    
    note_text += f"Removed {n_invalid} invalid SMILES. "
    
    if required and n_failed_required > 0:
        note_text += f"{n_failed_required} failed required groups. "
    if forbidden and n_failed_forbidden > 0:
        note_text += f"{n_failed_forbidden} failed forbidden groups."
    
    return {
        "output_filename": output_filename_stored,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "required_groups": required if required else [],
        "forbidden_groups": forbidden if forbidden else [],
        "filter_summary": filter_summary,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ Functional group detection uses SMARTS patterns which may have false positives/negatives. Manual validation recommended for critical applications.",
        "note": note_text
    }
