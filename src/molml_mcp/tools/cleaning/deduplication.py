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
import pandas as pd
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def find_duplicates_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_col: str,
    output_filename: str,
    explanation: str,
    label_col: Optional[str] = None,
    group_by_cols: Optional[List[str]] = None,
    is_binary_label: bool = True,
    alpha: float = 0.05,
    cv_threshold: float = 0.30,
) -> Dict[str, Any]:
    """
    Inspect a dataset for duplicate entries and recommend merging strategies based on label conflicts.
    
    This function identifies duplicate entries based on SMILES (and optional grouping columns),
    analyzes label conflicts using statistical tests, and appends a 'merge_strategy' column
    with recommendations for each entry.
    
    Workflow:
    1. Group rows by SMILES (and optional group_by_cols like protein ID)
    2. If no labels: mark duplicates with 'drop' strategy (identical entries)
    3. If labels exist: use should_merge_binary/continuous to analyze conflicts
    4. Unique entries (no duplicates) are marked as 'unique'
    5. Store annotated dataset and return summary statistics
    
    Parameters
    ----------
    input_filename : str
        Name of the input dataset resource (e.g., "cleaned_data_A3F2B1D4.csv")
    project_manifest_path : str
        Path to the project manifest.json file
    smiles_col : str
        Column name containing SMILES strings for duplicate detection
    output_filename : str
        Base name for output resource (will get unique ID appended)
    explanation : str
        Description of this inspection operation for manifest
    label_col : Optional[str], default=None
        Column name containing labels to check for conflicts. If None, all duplicates
        will be marked as 'drop' (no label analysis performed)
    group_by_cols : Optional[List[str]], default=None
        Additional columns to group by (e.g., ["protein_id"] to only consider entries
        as duplicates if they have same SMILES AND same protein). If None, only SMILES
        is used for grouping
    is_binary_label : bool, default=True
        Whether labels are binary (0/1) or continuous. Determines which statistical
        test to use (binomial test vs CV bootstrap)
    alpha : float, default=0.05
        Significance level for statistical tests (binomial test for binary labels)
    cv_threshold : float, default=0.30
        Coefficient of variation threshold for continuous labels (30% = high variability)
    
    Returns
    -------
    Dict[str, Any]
        Summary containing:
        - output_filename: Full name of annotated dataset resource
        - n_rows: Total number of rows
        - n_unique: Number of unique entries (no duplicates)
        - n_duplicate_groups: Number of duplicate groups found
        - strategy_counts: Dict with counts for each merge strategy
        - reason_summary: Dict with counts for each reason for strategy recommendation
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    if group_by_cols is not None:
        missing_cols = [col for col in group_by_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Group-by columns {missing_cols} not found in dataset. Available columns: {list(df.columns)}")
    
    # Prepare grouping columns
    grouping_cols = [smiles_col]
    if group_by_cols is not None:
        grouping_cols.extend(group_by_cols)
    
    # Create a copy to add merge_strategy and reason columns
    df_annotated = df.copy()
    df_annotated['merge_strategy'] = 'unique'  # Default for unique entries
    df_annotated['strategy_reason'] = ''  # Default empty reason
    
    # Handle NaN SMILES first - mark as 'drop' before grouping
    nan_smiles_mask = df_annotated[smiles_col].isna()
    if nan_smiles_mask.any():
        df_annotated.loc[nan_smiles_mask, 'merge_strategy'] = 'drop'
        df_annotated.loc[nan_smiles_mask, 'strategy_reason'] = 'NaN SMILES, recommend dropping'
    
    # Group by SMILES (and optional additional columns)
    # Note: NaN values create their own group in pandas groupby
    grouped = df_annotated.groupby(grouping_cols, dropna=False)
    
    # Track statistics
    strategy_counts = {'unique': 0, 'drop': 0, 'keep_first': 0, 'majority_vote': 0, 'mean': 0, 'median': 0}
    n_duplicate_groups = 0
    duplicate_examples = []
    reason_counts = []  # Track all reasons for summary
    
    # Process each group
    for group_keys, group_df in grouped:
        group_size = len(group_df)
        indices = group_df.index
        
        # Check if this group has NaN SMILES (already handled, just count and skip)
        if group_df[smiles_col].isna().any():
            nan_count = group_df[smiles_col].isna().sum()
            strategy_counts['drop'] += nan_count
            reason_counts.append('NaN SMILES, recommend dropping')
            
            # If there are non-NaN entries in mixed group, process them separately
            non_nan_indices = group_df[~group_df[smiles_col].isna()].index
            if len(non_nan_indices) == 0:
                # All NaN in this group
                if len(duplicate_examples) < 10:
                    duplicate_examples.extend(df_annotated.loc[indices].head(3).to_dict('records'))
                continue
            # If mixed, the non-NaN entries would have been in a different group
            # So we can continue here
            if len(duplicate_examples) < 10:
                duplicate_examples.extend(df_annotated.loc[indices].head(3).to_dict('records'))
            continue
        
        # Single entry - mark as unique
        if group_size == 1:
            strategy_counts['unique'] += 1
            continue
        
        # Multiple entries - duplicates found
        n_duplicate_groups += 1
        
        # No labels provided - mark all as 'drop' (identical structures)
        if label_col is None:
            df_annotated.loc[indices, 'merge_strategy'] = 'drop'
            strategy_counts['drop'] += group_size
            
            # Store example (read updated data from df_annotated)
            if len(duplicate_examples) < 10:
                duplicate_examples.extend(df_annotated.loc[indices].head(3).to_dict('records'))
            continue
        
        # Labels provided - separate NaN and valid labels
        nan_mask = group_df[label_col].isna()
        nan_indices = group_df[nan_mask].index
        valid_indices = group_df[~nan_mask].index
        
        # Handle NaN labels first - always drop with specific reason
        if len(nan_indices) > 0:
            df_annotated.loc[nan_indices, 'merge_strategy'] = 'drop'
            df_annotated.loc[nan_indices, 'strategy_reason'] = 'NaN label, recommend dropping'
            strategy_counts['drop'] += len(nan_indices)
            reason_counts.append('NaN label, recommend dropping')
        
        # Get valid (non-NaN) labels for analysis
        labels = group_df[label_col].dropna().tolist()
        
        # No valid labels in this group - all were NaN, already handled above
        if len(labels) == 0:
            if len(duplicate_examples) < 10:
                duplicate_examples.extend(df_annotated.loc[indices].head(3).to_dict('records'))
            continue
        
        # Analyze label conflicts using appropriate statistical test (only on valid labels)
        if is_binary_label:
            # Convert to int if needed (handles both int and float representations)
            labels_int = [int(label) for label in labels]
            method, reason = should_merge_binary(labels_int, alpha=alpha)
        else:
            # Continuous labels
            labels_float = [float(label) for label in labels]
            method, reason = should_merge_continuous(labels_float, cv_threshold=cv_threshold, alpha=alpha)
        
        # Apply recommended strategy and reason to entries with valid labels
        df_annotated.loc[valid_indices, 'merge_strategy'] = method
        df_annotated.loc[valid_indices, 'strategy_reason'] = reason
        strategy_counts[method] += len(valid_indices)
        
        # Track reason for summary
        reason_counts.append(reason)
        
        # Store example with the reason (read updated data from df_annotated)
        if len(duplicate_examples) < 10:
            example_rows = df_annotated.loc[indices].head(3).to_dict('records')
            for row in example_rows:
                row['_strategy_reason'] = reason  # Add reason for context
            duplicate_examples.extend(example_rows)
    
    # Store annotated dataset
    output_id = _store_resource(
        df_annotated,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    # Create reason summary using Counter
    from collections import Counter
    reason_summary = dict(Counter(reason_counts))
    
    # Return summary
    return {
        "output_filename": output_id,
        "n_rows": len(df_annotated),
        "n_unique": strategy_counts['unique'],
        "n_duplicate_groups": n_duplicate_groups,
        "strategy_counts": strategy_counts,
        "reason_summary": reason_summary
    }


def _analyze_regression_conflicts(values: List[float]) -> str:
    # function that determined if entries with regression labels should be merged or dropped based on statistical analysis
    # - given a list of values (labels), determine if they can be merged (mean/median) or should be dropped due to high variability
    import numpy as np

    if len(values) <= 1:
        return "merge"  # No conflict

    vals = np.array(values)
    mean = np.mean(vals)
    std = np.std(vals)
    if mean == 0:
        rel_cv = float('inf') if std > 0 else 0.0
    else:
        rel_cv = std / abs(mean)

    # Determine if variability is too high
    if rel_cv > 0.5:  # CV > 50% is very high variability
        return "drop"
    
    # Check for extreme outliers
    data_range = np.max(vals) - np.min(vals)
    if data_range > abs(mean) * 3:  # Range more than 3x mean suggests outliers
        return "drop"

    return "merge"


from scipy.stats import binomtest

def should_merge_binary(labels, alpha=0.05):
    """
    Analyze binary label conflicts using binomial test to recommend merge strategy.
    
    Uses a one-sided binomial test to determine if agreement among duplicate labels
    is significantly better than random chance (50%). This helps decide whether
    duplicates represent genuine signal or noise.
    
    H0: Agreement = 0.5 (random/noise)
    HA: Agreement > 0.5 (signal/consensus)
    
    Parameters
    ----------
    labels : list[int]
        Binary labels (0 or 1) from duplicate entries
    alpha : float, optional
        Significance level for binomial test (default: 0.05)
    
    Returns
    -------
    tuple[str, str]
        (recommended_method, reason)
        - recommended_method: 'majority_vote' or 'drop'
        - reason: Detailed explanation including p-value and label distribution
    
    Examples
    --------
    >>> should_merge_binary([1, 1, 1, 1, 0, 0])
    ('majority_vote', 'Significant agreement (p=0.023): 4/6 labels are 1 (67%)')
    
    >>> should_merge_binary([1, 1, 0, 0])
    ('drop', 'No significant agreement (p=0.500): 2/4 labels are 1 (50%)')
    """
    n = len(labels)
    n_ones = sum(labels)
    n_zeros = n - n_ones

    if n == 1:
        reason = f"Single value, no action needed"
        return 'keep_first', reason
    
    
    # Check if all labels are identical
    if len(set(labels)) == 1:
        reason = f"All values identical, recommend keeping first"
        return 'keep_first', reason
    
    n_majority = max(n_ones, n_zeros)
    majority_label = 1 if n_ones > n_zeros else 0
    agreement_pct = (n_majority / n) * 100
    
    # One-sided binomial test
    result = binomtest(n_majority, n, 0.5, alternative='greater')
    p_value = result.pvalue
    
    if p_value < alpha:
        # Significant agreement - recommend majority vote
        reason = f"Significant agreement (p<{alpha}), recommend majority vote"
        return 'majority_vote', reason
    else:
        # No significant agreement - recommend dropping
        reason = f"No significant agreement (p>{alpha}), recommend dropping"
        return 'drop', reason
    

import numpy as np
from scipy import stats

def should_merge_continuous(values, cv_threshold=0.30, alpha=0.05):
    """
    Analyze continuous value conflicts using CV and outlier detection to recommend merge strategy.
    
    Uses bootstrap-estimated coefficient of variation (CV) confidence interval to
    assess measurement reliability. Also detects outliers using IQR method to
    recommend appropriate aggregation method (mean vs median).
    
    Parameters
    ----------
    values : list[float]
        Continuous values from duplicate entries
    cv_threshold : float, optional
        Maximum acceptable CV for merging (default: 0.30 = 30%)
    alpha : float, optional
        Confidence level for CV estimation (default: 0.05 for 95% CI)
    
    Returns
    -------
    tuple[str, str]
        (recommended_method, reason)
        - recommended_method: 'mean', 'median', or 'drop'
        - reason: Detailed explanation including CV, outliers, and statistics
    
    Examples
    --------
    >>> should_merge_continuous([10.0, 10.2, 9.8, 10.1])
    ('mean', 'Low variation (CV=0.017, CI upper=0.025): values consistent...')
    
    >>> should_merge_continuous([10.0, 10.1, 50.0])
    ('drop', 'High variation (CV=0.994, CI upper=1.250): excessive uncertainty...')
    
    >>> should_merge_continuous([10.0, 10.1, 10.2, 50.0])
    ('median', 'Moderate variation (CV=0.850) with outliers detected...')
    """
    values = np.array(values)
    n = len(values)
    
    if n == 1:
        reason = f"Single value, no action needed"
        return 'keep_first', reason
    
    # Compute CV and basic statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values, ddof=1)
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Check if all values are identical (or nearly identical)
    if std_val == 0 or (std_val / abs(mean_val) < 1e-12 if mean_val != 0 else std_val < 1e-12):
        reason = f"All values (nearly) identical, recommend keeping first"
        return 'keep_first', reason
    value_range = max_val - min_val
    cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
    
    # Bootstrap 95% CI for CV
    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    cv_bootstrap = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        if sample_mean != 0:
            cv_bootstrap.append(sample_std / abs(sample_mean))
    
    cv_lower = np.percentile(cv_bootstrap, 2.5)
    cv_upper = np.percentile(cv_bootstrap, 97.5)
    
    # Outlier detection (IQR method)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    outlier_mask = (values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)
    has_outliers = outlier_mask.any()
    n_outliers = outlier_mask.sum()
    
    # Decision logic
    if cv_upper <= cv_threshold:
        # Acceptable variation - can merge
        if has_outliers:
            reason = f"CV lower than threshold (<={cv_threshold:.2f}), found outlier(s), recommend median"
            return 'median', reason
        else:
            reason = f"CV lower than threshold (<={cv_threshold:.2f}), no outliers, recommend mean"
            return 'mean', reason
    else:
        # Excessive variation - recommend dropping
        outlier_note = f", {n_outliers} outlier(s)" if has_outliers else ""
        reason = f"CV exceeds threshold (>{cv_threshold:.2f}), excessive uncertainty, recommend dropping"
        return 'drop', reason


def _execute_keep_first(group_df: pd.DataFrame, label_col: Optional[str]) -> pd.Series:
    """Keep the first entry in a duplicate group."""
    return group_df.iloc[0]


def _execute_drop(group_df: pd.DataFrame, label_col: Optional[str]) -> Optional[pd.Series]:
    """Drop all entries in a duplicate group (return None to signal removal)."""
    return None


def _execute_majority_vote(group_df: pd.DataFrame, label_col: str) -> pd.Series:
    """Keep first entry but use majority vote for the label."""
    result = group_df.iloc[0].copy()
    # Get valid (non-NaN) labels
    valid_labels = group_df[label_col].dropna()
    if len(valid_labels) > 0:
        # Majority vote (mode)
        majority_label = valid_labels.mode().iloc[0]
        result[label_col] = majority_label
    return result


def _execute_mean(group_df: pd.DataFrame, label_col: str) -> pd.Series:
    """Keep first entry but use mean for the label."""
    result = group_df.iloc[0].copy()
    # Get valid (non-NaN) labels
    valid_labels = group_df[label_col].dropna()
    if len(valid_labels) > 0:
        mean_label = valid_labels.mean()
        result[label_col] = mean_label
    return result


def _execute_median(group_df: pd.DataFrame, label_col: str) -> pd.Series:
    """Keep first entry but use median for the label."""
    result = group_df.iloc[0].copy()
    # Get valid (non-NaN) labels
    valid_labels = group_df[label_col].dropna()
    if len(valid_labels) > 0:
        median_label = valid_labels.median()
        result[label_col] = median_label
    return result


def deduplicate_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_col: str,
    output_filename: str,
    explanation: str,
    label_col: Optional[str] = None,
    group_by_cols: Optional[List[str]] = None,
    drop_all_duplicates: bool = False,
) -> Dict[str, Any]:
    """
    Execute deduplication on a dataset with merge_strategy column from find_duplicates_dataset.
    
    This function reads a dataset annotated with merge strategies (typically from find_duplicates_dataset)
    and executes the recommended merging operations. Each duplicate group is processed according to
    its assigned strategy:
    
    - 'unique': Keep as-is (single entry, no duplicates)
    - 'keep_first': Keep the first entry in the group
    - 'drop': Remove all entries in the group
    - 'majority_vote': Keep first entry, use majority vote for binary label
    - 'mean': Keep first entry, use mean for continuous label
    - 'median': Keep first entry, use median for continuous label
    
    **IMPORTANT**: This function expects the input dataset to have a 'merge_strategy' column created
    by find_duplicates_dataset(). Run find_duplicates_dataset() first to inspect duplicates and get 
    recommendations before executing deduplication.
    
    Parameters
    ----------
    input_filename : str
        Name of the input dataset resource with 'merge_strategy' column
        (e.g., output from find_duplicates_dataset)
    project_manifest_path : str
        Path to the project manifest.json file
    smiles_col : str
        Column name containing SMILES strings for duplicate detection
    output_filename : str
        Base name for output resource (will get unique ID appended)
    explanation : str
        Description of this deduplication operation for manifest
    label_col : Optional[str], default=None
        Column name containing labels. Required for strategies that aggregate labels
        (majority_vote, mean, median). Can be None if no label aggregation needed
    group_by_cols : Optional[List[str]], default=None
        Additional columns to group by (must match what was used in find_duplicates_dataset).
        E.g., ["protein_id"] to deduplicate within protein groups
    drop_all_duplicates : bool, default=False
        If True, override all merge strategy recommendations and drop all entries that are
        not unique (i.e., drop all duplicate groups regardless of their merge_strategy).
        Only entries marked as 'unique' will be kept. This is useful for strict deduplication
        when you want to remove all ambiguous entries regardless of label agreement
    
    Returns
    -------
    Dict[str, Any]
        Summary containing:
        - output_filename: Full name of deduplicated dataset resource
        - n_rows_before: Number of rows before deduplication
        - n_rows_after: Number of rows after deduplication
        - n_rows_removed: Number of rows removed
        - n_nan_smiles: Number of entries with NaN SMILES detected
        - n_nan_labels: Number of entries with NaN labels detected (0 if label_col is None)
        - suggestions: List of recommendations for handling NaN values
        - operations_summary: Dict with counts of each merge operation executed
        - preview: First 5 rows of deduplicated dataset
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_rows_before = len(df)
    
    # Validate merge_strategy column exists
    if 'merge_strategy' not in df.columns:
        raise ValueError(
            f"Input dataset must have 'merge_strategy' column. "
            f"Run find_duplicates_dataset() first to generate merge strategies. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Validate required columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    if group_by_cols is not None:
        missing_cols = [col for col in group_by_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Group-by columns {missing_cols} not found in dataset. Available columns: {list(df.columns)}")
    
    # Count NaN values before deduplication
    n_nan_smiles = df[smiles_col].isna().sum()
    n_nan_labels = df[label_col].isna().sum() if label_col is not None else 0
    
    # Prepare grouping columns
    grouping_cols = [smiles_col]
    if group_by_cols is not None:
        grouping_cols.extend(group_by_cols)
    
    # Track operations
    operations_summary = {
        'unique': 0,
        'keep_first': 0,
        'drop': 0,
        'majority_vote': 0,
        'mean': 0,
        'median': 0
    }
    
    # Process groups and collect deduplicated rows
    deduplicated_rows = []
    
    # Group by SMILES (and optional additional columns)
    grouped = df.groupby(grouping_cols, dropna=False)
    
    for group_keys, group_df in grouped:
        # Get the strategy for this group (all rows in group should have same strategy)
        strategy = group_df['merge_strategy'].iloc[0]
        
        # Override: if drop_all_duplicates is True, drop everything except unique entries
        if drop_all_duplicates and strategy != 'unique':
            # Drop all duplicate entries
            operations_summary['drop'] += 1
            continue
        
        # Execute appropriate merge operation
        if strategy == 'unique':
            # Single entry, keep as-is
            deduplicated_rows.append(group_df.iloc[0])
            operations_summary['unique'] += 1
            
        elif strategy == 'keep_first':
            # Keep first entry
            result = _execute_keep_first(group_df, label_col)
            deduplicated_rows.append(result)
            operations_summary['keep_first'] += 1
            
        elif strategy == 'drop':
            # Drop all entries in this group
            result = _execute_drop(group_df, label_col)
            if result is not None:
                deduplicated_rows.append(result)
            operations_summary['drop'] += 1
            
        elif strategy == 'majority_vote':
            # Keep first entry, aggregate label with majority vote
            if label_col is None:
                raise ValueError(f"label_col required for majority_vote strategy in group {group_keys}")
            result = _execute_majority_vote(group_df, label_col)
            deduplicated_rows.append(result)
            operations_summary['majority_vote'] += 1
            
        elif strategy == 'mean':
            # Keep first entry, aggregate label with mean
            if label_col is None:
                raise ValueError(f"label_col required for mean strategy in group {group_keys}")
            result = _execute_mean(group_df, label_col)
            deduplicated_rows.append(result)
            operations_summary['mean'] += 1
            
        elif strategy == 'median':
            # Keep first entry, aggregate label with median
            if label_col is None:
                raise ValueError(f"label_col required for median strategy in group {group_keys}")
            result = _execute_median(group_df, label_col)
            deduplicated_rows.append(result)
            operations_summary['median'] += 1
            
        else:
            raise ValueError(f"Unknown merge strategy '{strategy}' in group {group_keys}")
    
    # Create deduplicated dataframe
    df_deduplicated = pd.DataFrame(deduplicated_rows)
    
    # Remove merge_strategy and strategy_reason columns (they were for inspection only)
    cols_to_drop = ['merge_strategy', 'strategy_reason']
    cols_to_drop = [col for col in cols_to_drop if col in df_deduplicated.columns]
    if cols_to_drop:
        df_deduplicated = df_deduplicated.drop(columns=cols_to_drop)
    
    # Reset index
    df_deduplicated = df_deduplicated.reset_index(drop=True)
    
    n_rows_after = len(df_deduplicated)
    n_rows_removed = n_rows_before - n_rows_after
    
    # Store deduplicated dataset
    output_id = _store_resource(
        df_deduplicated,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    # Create suggestions based on NaN counts
    suggestions = []
    if n_nan_smiles > 0:
        suggestions.append(f"Found {n_nan_smiles} entries with NaN SMILES - recommend removing these invalid entries")
    if n_nan_labels > 0:
        suggestions.append(f"Found {n_nan_labels} entries with NaN labels - recommend removing these missing values")
    
    # Return summary
    return {
        "output_filename": output_id,
        "n_rows_before": n_rows_before,
        "n_rows_after": n_rows_after,
        "n_rows_removed": n_rows_removed,
        "n_nan_smiles": n_nan_smiles,
        "n_nan_labels": n_nan_labels,
        "suggestions": suggestions,
        "operations_summary": operations_summary,
        "preview": df_deduplicated.head(5).to_dict('records'),
    }

