"""
Activity cliff detection for identifying structurally similar molecules with large activity differences.

Activity cliffs are pairs of molecules that are highly similar in structure but have significantly
different biological activities. These are important for SAR analysis and understanding the 
molecular basis of activity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from chemlint.infrastructure.resources import _load_resource, _store_resource


def _compute_fold_difference_matrix(activity_values: np.ndarray) -> np.ndarray:
    """
    Compute pairwise fold-difference matrix. Formula: max(val_i, val_j) / min(val_i, val_j).
    ⚠️ Requires LINEAR SCALE (IC50_nM, Ki_nM). DO NOT use log-scale (pIC50, pKi).
    
    Parameters
    ----------
    activity_values : np.ndarray
        1D array in LINEAR SCALE (IC50 nM, Ki nM, EC50 μM, etc.)
        
    Returns
    -------
    np.ndarray
        Symmetric matrix (n_molecules × n_molecules) with fold-differences, diagonal = 1.0
    """
    n_molecules = len(activity_values)
    fold_diff_matrix = np.ones((n_molecules, n_molecules))
    
    for i in range(n_molecules):
        for j in range(i + 1, n_molecules):
            # Simple ratio: bigger / smaller
            val_i = activity_values[i]
            val_j = activity_values[j]
            
            if val_i == 0 or val_j == 0:
                # Avoid division by zero - set to very large fold-difference
                fold_diff = np.inf
            else:
                fold_diff = max(val_i, val_j) / min(val_i, val_j)
            
            fold_diff_matrix[i, j] = fold_diff
            fold_diff_matrix[j, i] = fold_diff  # Symmetric
    
    return fold_diff_matrix


def annotate_activity_cliff_molecules(
    dataset_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    activity_column: str,
    similarity_matrix_filename: str,
    output_filename: str,
    explanation: str = "Annotated molecules with activity cliff information",
    similarity_threshold: float = 0.8,
    fold_difference_threshold: float = 10.0,
) -> Dict:
    """
    Annotate molecules with activity cliff partners (similar structure, large activity difference).
    ⚠️ CRITICAL: activity_column MUST be LINEAR SCALE (IC50_nM, Ki_nM). DO NOT use log-scale (pIC50, pKi).
    Conversion: IC50_nM = 10^(9 - pIC50)
    
    Parameters
    ----------
    dataset_filename : str
        Input dataset from manifest
    project_manifest_path : str
        Path to manifest.json
    smiles_column : str
        Column with SMILES
    activity_column : str
        Activity column in LINEAR SCALE (✅ IC50_nM, Ki_nM | ❌ pIC50, pKi)
    similarity_matrix_filename : str
        Precomputed similarity matrix (from compute_similarity_matrix)
    output_filename : str
        Output dataset name
    explanation : str
        Operation description
    similarity_threshold : float, default=0.8
        Minimum similarity for "structurally similar" (typical: 0.85-0.95 for Tanimoto)
    fold_difference_threshold : float, default=10.0
        Minimum fold-difference for activity cliff (typical: 10-100)
        
    Returns
    -------
    dict
        output_filename, n_molecules, n_cliff_molecules, n_non_cliff_molecules, n_total_cliff_pairs, summary
        
    New Columns Added
    -----------------
    - is_activity_cliff_molecule : 'yes' or 'no'
    - n_activity_cliff_partners : int (count of partners)
    - strongest_cliff_partner_idx : int or NaN (index of partner with largest fold-difference)
    - strongest_cliff_partner_smiles : str or NaN
    """
    # Print warning about linear scale requirement
    print(f"⚠️  IMPORTANT: Activity values must be in LINEAR SCALE (IC50_nM, Ki_nM, etc.)")
    print(f"             DO NOT use log-scale (pIC50, pKi, etc.)!")
    print(f"             Analyzing column '{activity_column}' - please verify this is LINEAR SCALE.\n")
    
    # Load dataset
    df = _load_resource(project_manifest_path, dataset_filename)
    n_molecules = len(df)
    
    # Validate columns
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found. Available: {df.columns.tolist()}")
    if activity_column not in df.columns:
        raise ValueError(f"Activity column '{activity_column}' not found. Available: {df.columns.tolist()}")
    
    # Load similarity matrix
    sim_matrix = _load_resource(project_manifest_path, similarity_matrix_filename)
    
    # Validate matrix dimensions
    if sim_matrix.shape[0] != n_molecules or sim_matrix.shape[1] != n_molecules:
        raise ValueError(
            f"Similarity matrix shape {sim_matrix.shape} does not match dataset size {n_molecules}. "
            f"Ensure the matrix was computed for this exact dataset."
        )
    
    # Get activity values and handle NaN
    activity_values = df[activity_column].values
    valid_mask = ~np.isnan(activity_values)
    n_valid = valid_mask.sum()
    n_nan = (~valid_mask).sum()
    
    if n_nan > 0:
        print(f"⚠️  Warning: {n_nan} molecules have NaN activity values and will be marked as non-cliff molecules.\n")
    
    # Compute fold-difference matrix
    fold_diff_matrix = _compute_fold_difference_matrix(activity_values)
    
    # Initialize annotation columns with proper dtypes
    df['is_activity_cliff_molecule'] = 'no'
    df['n_activity_cliff_partners'] = 0
    df['strongest_cliff_partner_idx'] = pd.Series([np.nan] * len(df), dtype='Int64')  # Nullable integer
    df['strongest_cliff_partner_smiles'] = pd.Series([None] * len(df), dtype='object')  # Object for strings
    
    # Find cliffs for each molecule
    total_cliff_pairs = 0
    
    for i in range(n_molecules):
        # Skip if this molecule has NaN activity
        if not valid_mask[i]:
            continue
            
        # Find all molecules that form cliffs with molecule i
        # Requirements: similarity > threshold, fold-diff > threshold, valid activity, not self
        cliff_mask = (
            (sim_matrix[i, :] > similarity_threshold) &
            (fold_diff_matrix[i, :] > fold_difference_threshold) &
            valid_mask &
            (np.arange(n_molecules) != i)
        )
        
        cliff_indices = np.where(cliff_mask)[0]
        n_cliffs = len(cliff_indices)
        
        if n_cliffs > 0:
            df.at[i, 'is_activity_cliff_molecule'] = 'yes'
            df.at[i, 'n_activity_cliff_partners'] = n_cliffs
            total_cliff_pairs += n_cliffs
            
            # Find the strongest cliff partner (largest fold-difference among cliffs)
            cliff_fold_diffs = fold_diff_matrix[i, cliff_indices]
            strongest_idx_in_cliffs = np.argmax(cliff_fold_diffs)
            strongest_partner_idx = cliff_indices[strongest_idx_in_cliffs]
            
            df.at[i, 'strongest_cliff_partner_idx'] = int(strongest_partner_idx)
            df.at[i, 'strongest_cliff_partner_smiles'] = df.iloc[strongest_partner_idx][smiles_column]
    
    # Count each pair once (currently counted twice)
    total_cliff_pairs = total_cliff_pairs // 2
    
    # Count cliff molecules
    n_cliff_molecules = (df['is_activity_cliff_molecule'] == 'yes').sum()
    n_non_cliff_molecules = n_molecules - n_cliff_molecules
    
    # Store annotated dataset
    output_id = _store_resource(
        df,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    # Create summary message
    if n_cliff_molecules == 0:
        summary_msg = (
            f"No activity cliff molecules found with similarity > {similarity_threshold} "
            f"and fold-difference > {fold_difference_threshold}. "
            f"Consider lowering thresholds or checking data quality."
        )
    else:
        pct_cliff = 100.0 * n_cliff_molecules / n_molecules
        summary_msg = (
            f"Found {n_cliff_molecules} activity cliff molecules out of {n_molecules} total ({pct_cliff:.1f}%). "
            f"These molecules participate in {total_cliff_pairs} activity cliff pairs. "
            f"Activity column: {activity_column}."
        )
    
    return {
        "output_filename": output_id,
        "n_molecules": n_molecules,
        "n_molecules_with_nan_activity": int(n_nan),
        "n_cliff_molecules": int(n_cliff_molecules),
        "n_non_cliff_molecules": int(n_non_cliff_molecules),
        "n_total_cliff_pairs": int(total_cliff_pairs),
        "similarity_threshold": similarity_threshold,
        "fold_difference_threshold": fold_difference_threshold,
        "activity_column": activity_column,
        "columns": df.columns.tolist(),
        "summary": summary_msg
    }


def get_all_activity_cliff_tools():
    """
    Returns all MCP-exposed activity cliff functions.
    """
    return [
        annotate_activity_cliff_molecules,
    ]
