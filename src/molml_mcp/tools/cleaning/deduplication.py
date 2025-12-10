"""
Entry deduplication tools for molecular datasets.

This module will contain functions for identifying and removing duplicate entries
in molecular datasets based on SMILES, InChI, or other identifiers.

Planned functionality:
- Detect exact duplicates (same SMILES string)
- Detect structural duplicates (same molecule, different representations)
- Handle duplicate aggregation strategies (keep first, average values, etc.)
- Inspect duplicate statistics before removal
"""

# Placeholder for future deduplication functions
# TODO: Implement deduplication functions

from typing import Optional, List, Dict, Any
from molml_mcp.infrastructure.resources import _load_resource


def inspect_duplicates_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_col: str,
    label_col: Optional[str] = None,
    group_by_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Inspect a dataset for duplicate molecules and label conflicts (read-only).
    
    Identifies duplicate molecules (same SMILES, optionally same group_by values) 
    and analyzes conflicts in a single label column if provided. Automatically 
    detects whether the label is numeric (regression) or categorical (classification):
    - For regression: conflicts when values vary by >1% relative to mean
    - For classification: conflicts when multiple distinct values exist
    
    Provides comprehensive information to help decide how to merge duplicates:
    - Conflict statistics (for regression: mean, median, std, range, CV)
    - Value distribution (for classification: most common conflicting values)
    - Per-duplicate statistics in preview
    - Recommended merge strategies based on data characteristics

    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    project_manifest_path : str
        Path to the project manifest JSON file.
    smiles_col : str
        Column containing molecule identifiers (use of standardized SMILES HIGHLY RECOMMENDED).
    label_col : str | None
        Single label column to check for conflicts among duplicates. Works with both
        numeric (regression) and categorical (classification) labels.
    group_by_cols : list[str] | None
        Additional columns to group by before checking duplicates (e.g., ["protein_target", "assay_id"]).
        Duplicates will only be detected when BOTH SMILES and group_by values match.

    Returns
    -------
    dict
        {
            "input_filename": str,
            "n_rows": int,
            "n_unique_molecules": int,
            "n_duplicate_groups": int,
            "n_duplicate_rows": int,
            "n_rows_after_dedup": int,
            "label_analysis": {  # Only present if label_col provided
                "label_col": str,
                "label_type": "regression" | "classification",
                "n_conflicts": int,
                "conflict_rate": float,
                "conflict_statistics": {...}  # regression only
                "conflicting_value_distribution": {...}  # classification only
            },
            "merge_strategies": [...]  # Only present if label_col provided
            "duplicate_preview": [
                {
                    "smiles": str,
                    "n_occurrences": int,
                    "row_indices": list[int],
                    "group": {...}  # if group_by_cols provided
                    "label_values": list,  # if label_col provided
                    "label_stats": {...}   # if label_col provided
                }
            ]
        }
    """
    df = _load_resource(project_manifest_path, input_filename)

    # Validate columns exist
    all_cols = [smiles_col] + ([label_col] if label_col else []) + (group_by_cols or [])
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}. Available: {list(df.columns)}")

    # Define duplicate key (smiles + optional grouping)
    key_cols = (group_by_cols or []) + [smiles_col]

    # Find duplicates
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_df = df[dup_mask].copy()
    n_dup_rows = len(dup_df)
    n_dup_groups = dup_df.groupby(key_cols, dropna=False).ngroups if n_dup_rows > 0 else 0

    result: Dict[str, Any] = {
        "input_filename": input_filename,
        "n_rows": len(df),
        "n_unique_molecules": df[smiles_col].nunique(dropna=False),
        "n_duplicate_groups": n_dup_groups,
        "n_duplicate_rows": n_dup_rows,
        "n_rows_after_dedup": len(df) - n_dup_rows + n_dup_groups,
    }

    # Analyze label conflicts if requested
    if label_col and n_dup_groups > 0:
        import pandas as pd
        import numpy as np
        
        # Determine if label is numeric (regression) or categorical (classification)
        values = df[label_col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(values)
        
        def has_conflict(g):
            vals = g[label_col].dropna()
            if len(vals) <= 1:
                return False
            
            if is_numeric:
                # For regression: check if std deviation > 0 (values vary)
                # Also check relative variation to handle different scales
                std = vals.std()
                mean_abs = vals.abs().mean()
                # Consider conflict if std > 0 and relative variation > 1%
                if std == 0:
                    return False
                if mean_abs == 0:
                    return std > 0  # If mean is 0, any variation is a conflict
                return (std / mean_abs) > 0.01
            else:
                # For classification: check if more than one unique value
                return len(vals.unique()) > 1

        grouped = dup_df.groupby(key_cols, dropna=False)
        conflict_flags = grouped.apply(has_conflict)
        n_conflicts = int(conflict_flags.sum())

        # Build label analysis
        label_analysis = {
            "label_col": label_col,
            "label_type": "regression" if is_numeric else "classification",
            "n_conflicts": n_conflicts,
            "conflict_rate": round(n_conflicts / n_dup_groups * 100, 1) if n_dup_groups > 0 else 0.0,
        }
        
        # Add statistics for conflicting groups if numeric
        if is_numeric and n_conflicts > 0:
            def get_stats(g):
                vals = g[label_col].dropna()
                if len(vals) <= 1:
                    return None
                return {
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "range": float(vals.max() - vals.min()),
                    "cv": float(vals.std() / vals.mean()) if vals.mean() != 0 else float('inf'),
                }
            
            stats_by_group = grouped.apply(get_stats)
            valid_stats = [s for s in stats_by_group if s is not None]
            
            if valid_stats:
                label_analysis["conflict_statistics"] = {
                    "avg_std": round(np.mean([s["std"] for s in valid_stats]), 3),
                    "avg_range": round(np.mean([s["range"] for s in valid_stats]), 3),
                    "max_range": round(max([s["range"] for s in valid_stats]), 3),
                    "avg_cv": round(np.mean([s["cv"] for s in valid_stats if np.isfinite(s["cv"])]), 3),
                }
        
        # Add value distribution info for classification
        elif not is_numeric and n_conflicts > 0:
            # Count most common values across all conflicting duplicates
            conflict_vals = []
            for _, group in grouped:
                vals = group[label_col].dropna()
                if len(vals.unique()) > 1:  # Only conflicting groups
                    conflict_vals.extend(vals.tolist())
            
            if conflict_vals:
                from collections import Counter
                value_counts = Counter(conflict_vals)
                label_analysis["conflicting_value_distribution"] = {
                    str(k): v for k, v in value_counts.most_common(10)
                }

        result["label_analysis"] = label_analysis
        
        # Suggest merge strategies based on the data characteristics
        strategies = []
        
        if is_numeric:
            conflict_stats = label_analysis.get("conflict_statistics", {})
            avg_cv = conflict_stats.get("avg_cv", 0)
            max_range = conflict_stats.get("max_range", 0)
            avg_range = conflict_stats.get("avg_range", 0)
            
            # Check if values are unreliable (high CV suggests high uncertainty)
            high_variability = bool(avg_cv > 0.5)  # CV > 50% is very high
            extreme_range = bool(max_range > avg_range * 3) if avg_range > 0 else False  # Outlier detection
            very_high_conflict_rate = bool(label_analysis["conflict_rate"] > 80)
            
            # Only recommend drop if variability is truly problematic
            should_drop = high_variability or extreme_range or very_high_conflict_rate
            
            strategies.append({
                "strategy": "mean",
                "description": "Average all values (good for measurements with random error)",
                "recommended": bool(avg_cv < 0.2 and not should_drop)
            })
            strategies.append({
                "strategy": "median", 
                "description": "Take median value (robust to outliers)",
                "recommended": bool(0.2 <= avg_cv < 0.5 and not should_drop)
            })
            strategies.append({
                "strategy": "first",
                "description": "Keep first occurrence (preserves original data order)",
                "recommended": False
            })
            strategies.append({
                "strategy": "max",
                "description": "Keep maximum value (useful for potency/activity)",
                "recommended": False
            })
            strategies.append({
                "strategy": "min",
                "description": "Keep minimum value (useful for IC50/EC50 where lower is better)",
                "recommended": False
            })
            strategies.append({
                "strategy": "drop",
                "description": "Remove all conflicting duplicates (values too unreliable to merge)",
                "recommended": should_drop
            })
        else:  # classification
            strategies.append({
                "strategy": "mode",
                "description": "Most common value (majority vote)",
                "recommended": True
            })
            strategies.append({
                "strategy": "first",
                "description": "Keep first occurrence",
                "recommended": False
            })
            if n_conflicts > 0:
                strategies.append({
                    "strategy": "drop",
                    "description": "Remove all conflicting duplicates (conservative approach)",
                    "recommended": bool(label_analysis["conflict_rate"] > 30)
                })
        
        result["merge_strategies"] = strategies

    # Preview of duplicate groups (first 20)
    if n_dup_groups > 0:
        preview = []
        grouped = dup_df.groupby(key_cols, dropna=False)

        for i, (key, group) in enumerate(grouped):
            if i >= 20:
                break

            entry = {
                "smiles": key[-1] if isinstance(key, tuple) else key,
                "n_occurrences": len(group),
                "row_indices": group.index.tolist(),
            }

            if group_by_cols:
                entry["group"] = dict(zip(group_by_cols, key[:-1] if isinstance(key, tuple) else []))

            if label_col:
                import pandas as pd
                vals = group[label_col].dropna()
                entry["label_values"] = group[label_col].tolist()
                
                # Add statistics/summary for the label in preview
                if len(vals) > 0:
                    is_col_numeric = pd.api.types.is_numeric_dtype(vals)
                    if is_col_numeric:
                        entry["label_stats"] = {
                            "mean": round(float(vals.mean()), 3),
                            "median": round(float(vals.median()), 3),
                            "std": round(float(vals.std()), 3),
                            "min": round(float(vals.min()), 3),
                            "max": round(float(vals.max()), 3),
                            "has_conflict": bool(len(vals.unique()) > 1 and (vals.std() / vals.abs().mean() > 0.01 if vals.abs().mean() != 0 else vals.std() > 0))
                        }
                    else:
                        unique_vals = vals.unique()
                        mode_val = vals.mode()[0] if len(vals.mode()) > 0 else None
                        entry["label_stats"] = {
                            "unique_values": unique_vals.tolist(),
                            "n_unique": int(len(unique_vals)),
                            "has_conflict": bool(len(unique_vals) > 1),
                            "most_common": str(mode_val) if mode_val is not None else None
                        }

            preview.append(entry)

        result["duplicate_preview"] = preview

    return result


def deduplicate_molecules_dataset(input_filename: str, molecule_id_column: str, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Remove duplicate entries from a dataset based on a specified molecule identifier column. This should be a unique identifier for each molecule, 
    ideally after SMILES standardization.

    **IT IS STRONGLY RECOMMENDED TO USE inspect_duplicates_dataset function FIRST TO REVIEW DUPLICATES BEFORE REMOVAL.**

    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    molecule_id_column : str
        Name of the column containing unique molecule identifiers.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of the deduplication performed.

    Returns
    -------
    dict
        Updated dataset information after removing duplicates.
    """
    import pandas as pd

    df = _load_resource(project_manifest_path, input_filename)
    n_rows_before = len(df)

    if molecule_id_column not in df.columns:
        raise ValueError(f"Column {molecule_id_column} not found in dataset.")

    df_deduplicated = df.drop_duplicates(subset=[molecule_id_column])

    output_filename = _store_resource(df_deduplicated, project_manifest_path, output_filename, explanation, 'csv')
    return {
        "output_filename": output_filename,
        "n_rows_before": n_rows_before,
        "n_rows_after": len(df_deduplicated),
        "columns": list(df_deduplicated.columns),
        "preview": df_deduplicated.head(5).to_dict(orient="records"),
    }

