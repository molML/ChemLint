"""
Client-facing dataset manipulation tools.

These functions provide MCP-accessible operations for working with tabular datasets,
including loading, inspecting, filtering, and manipulating dataset rows and columns.
"""

from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from typing import Any, Dict, List, Optional



def import_csv_from_path(file_path: str, project_manifest_path: str, filename: str, explanation: str) -> dict:
    """
    🚀 ENTRY POINT: Import a CSV file from a filesystem path into the project.
    
    ⚠️ USE THIS FUNCTION when the user provides a file path like "/path/to/data.csv"
    
    This is the PRIMARY way to bring external CSV files into the MCP project system.
    After calling this function, use the returned output_filename for all subsequent
    operations (NOT the original file_path).

    When user says: "Use /Users/me/compounds.csv", call this function!!!
    
    Parameters
    ----------
    file_path : str
        Full filesystem path to the CSV file (e.g., "/Users/name/data.csv")
    project_manifest_path : str
        Path to the project's manifest.json file
    filename : str
        Descriptive name for the dataset (no .csv extension needed)
    explanation : str
        Brief description of what this dataset contains

    Returns
    -------
    dict
        Contains:
        - output_filename: Use this for all future operations on this dataset
        - n_rows: Number of rows loaded
        - columns: List of column names
        - preview: First 5 rows
    """
    import pandas as pd

    df = pd.read_csv(file_path)

    output_filename = _store_resource(df, project_manifest_path, filename, explanation, "csv")

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


def import_csv_from_text(csv_content: str, project_manifest_path: str, filename: str, explanation: str) -> dict:
    """
    Import CSV data from text string.
    
    Parameters
    ----------
    csv_content : str
        CSV content as string
    project_manifest_path : str
        Path to manifest.json
    filename : str
        Name for stored dataset (no extension)
    explanation : str
        Brief description
    
    Returns
    -------
    dict
        Contains output_filename, n_rows, columns, preview
    """
    import pandas as pd
    from io import StringIO
    
    # Read CSV from string content
    df = pd.read_csv(StringIO(csv_content))
    
    output_filename = _store_resource(df, project_manifest_path, filename, explanation, "csv")
    
    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


def get_dataset_head(project_manifest_path: str, input_filename: str, n_rows: int = 10) -> dict:
    """
    Get the first n rows of a dataset.

    Parameters
    ----------
    project_manifest_path : str
        Path to manifest.json
    input_filename : str
        Dataset filename
    n_rows : int, default=10
        Number of rows to return

    Returns
    -------
    dict
        Contains input_filename, n_rows_returned, n_rows_total, columns, rows
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    # Get the head
    head_df = df.head(n_rows)
    
    return {
        "input_filename": input_filename,
        "n_rows_returned": len(head_df),
        "n_rows_total": n_total,
        "columns": list(df.columns),
        "rows": head_df.to_dict(orient="records"),
    }


def get_dataset_full(project_manifest_path: str, input_filename: str, max_rows: int = 10000) -> dict:
    """
    Get entire dataset content (up to max_rows limit). WARNING: may be large!

    Parameters
    ----------
    project_manifest_path : str
        Path to manifest.json
    input_filename : str
        Dataset filename
    max_rows : int, default=10000
        Maximum rows to return

    Returns
    -------
    dict
        Contains input_filename, n_rows_returned, n_rows_total, columns, rows, truncated
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    # Check if we need to truncate
    truncated = n_total > max_rows
    result_df = df.head(max_rows) if truncated else df
    
    return {
        "input_filename": input_filename,
        "n_rows_returned": len(result_df),
        "n_rows_total": n_total,
        "columns": list(df.columns),
        "rows": result_df.to_dict(orient="records"),
        "truncated": truncated,
    }


def get_dataset_summary(project_manifest_path: str, input_filename: str, columns: list[str] | None = None) -> dict:
    """
    Get statistical summary of dataset columns.

    Parameters
    ----------
    project_manifest_path : str
        Path to manifest.json
    input_filename : str
        Dataset filename
    columns : list[str] | None
        Specific columns to summarize (None = all)

    Returns
    -------
    dict
        Contains input_filename, n_rows, n_columns, n_columns_summarized, column_summaries
        (numeric: min/max/mean/median/std; non-numeric: n_unique/top_value/top_freq)
    """
    import pandas as pd
    import numpy as np
    
    df = _load_resource(project_manifest_path, input_filename)
    n_rows = len(df)
    n_columns = len(df.columns)
    
    # Determine which columns to summarize
    if columns is None:
        cols_to_summarize = df.columns
    else:
        # Validate that all requested columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Column(s) not found in dataset: {missing_cols}")
        cols_to_summarize = columns
    
    column_summaries = {}
    
    for col in cols_to_summarize:
        col_data = df[col]
        n_missing = col_data.isna().sum()
        count = col_data.notna().sum()
        dtype_str = str(col_data.dtype)
        
        # Check if numeric (including int, float, but excluding bool)
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            # Numeric column statistics
            if count > 0:
                summary = {
                    "dtype": dtype_str,
                    "count": int(count),
                    "n_missing": int(n_missing),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                }
            else:
                # All values are missing
                summary = {
                    "dtype": dtype_str,
                    "count": 0,
                    "n_missing": int(n_missing),
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "std": None,
                }
        else:
            # Non-numeric column statistics
            if count > 0:
                value_counts = col_data.value_counts()
                top_value = value_counts.index[0]
                top_freq = int(value_counts.iloc[0])
                n_unique = col_data.nunique()
                
                # Convert top_value to a JSON-serializable type
                if pd.isna(top_value):
                    top_value = None
                elif isinstance(top_value, (np.integer, np.floating)):
                    top_value = float(top_value)
                elif isinstance(top_value, np.bool_):
                    top_value = bool(top_value)
                else:
                    top_value = str(top_value)
                
                summary = {
                    "dtype": dtype_str,
                    "count": int(count),
                    "n_missing": int(n_missing),
                    "n_unique": int(n_unique),
                    "top_value": top_value,
                    "top_freq": top_freq,
                }
            else:
                # All values are missing
                summary = {
                    "dtype": dtype_str,
                    "count": 0,
                    "n_missing": int(n_missing),
                    "n_unique": 0,
                    "top_value": None,
                    "top_freq": 0,
                }
        
        column_summaries[col] = summary
    
    return {
        "input_filename": input_filename,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_columns_summarized": len(cols_to_summarize),
        "column_summaries": column_summaries,
    }


def inspect_dataset_rows(project_manifest_path: str, input_filename: str, row_indices: list[int] | None = None, 
                         filter_condition: str | None = None, max_rows: int = 100) -> dict:
    """
    Inspect rows by index or pandas query filter (supports >, <, ==, and, or).

    Parameters
    ----------
    project_manifest_path : str
        Path to manifest.json
    input_filename : str
        Dataset filename
    row_indices : list[int] | None
        Row indices to retrieve (0-based)
    filter_condition : str | None
        Pandas query string (e.g., "TPSA > 20 and MolWt < 500", 'status == "active"')
        Use backticks for columns with spaces: `column name`
    max_rows : int, default=100
        Maximum rows to return

    Returns
    -------
    dict
        Contains input_filename, n_rows_returned, n_rows_total, columns, rows, selection_method
    
    Examples
    --------
    inspect_dataset_rows(path, fname, filter_condition="TPSA > 20")
    inspect_dataset_rows(path, fname, filter_condition='status == "active"')
    inspect_dataset_rows(path, fname, row_indices=[0, 5, 10])
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    # Select rows by index or filter
    if row_indices is not None:
        # Validate indices
        invalid_indices = [i for i in row_indices if i < 0 or i >= n_total]
        if invalid_indices:
            raise ValueError(f"Invalid row indices (out of range): {invalid_indices}. Dataset has {n_total} rows (0-{n_total-1}).")
        
        selected_df = df.iloc[row_indices[:max_rows]]
        selection_method = "indices"
        
    elif filter_condition is not None:
        try:
            selected_df = df.query(filter_condition)
        except Exception as e:
            raise ValueError(f"Invalid filter condition '{filter_condition}': {e}")
        
        # Limit results
        if len(selected_df) > max_rows:
            selected_df = selected_df.head(max_rows)
        
        selection_method = "filter"
    else:
        raise ValueError("Must provide either row_indices or filter_condition")
    
    return {
        "input_filename": input_filename,
        "n_rows_returned": len(selected_df),
        "n_rows_total": n_total,
        "columns": list(df.columns),
        "rows": selected_df.to_dict(orient="records"),
        "selection_method": selection_method,
    }


def drop_from_dataset(input_filename: str, column_name: str, condition: str, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Drop rows by exact match or null check. For complex filters, use inspect_dataset_rows().

    Parameters
    ----------
    input_filename : str
        Input dataset filename
    column_name : str
        Column to check
    condition : str
        Either "is None" (drops nulls) or exact value to match (e.g., "Failed")
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description

    Returns
    -------
    dict
        Contains output_filename, n_rows, columns, preview
    
    Examples
    --------
    drop_from_dataset(fname, "status", "Failed", path, "cleaned", "Dropped failed rows")
    drop_from_dataset(fname, "smiles", "is None", path, "no_nulls", "Dropped null SMILES")
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    if condition == 'is None':
        df_cleaned = df[df[column_name].notnull()]
    else:
        # Attempt to convert condition to match column dtype for proper comparison
        col_data = df[column_name]
        try:
            # If column is numeric, try to convert condition to numeric
            if pd.api.types.is_numeric_dtype(col_data):
                condition_value = pd.to_numeric(condition)
            else:
                condition_value = condition
        except (ValueError, TypeError):
            # If conversion fails, use as-is
            condition_value = condition
        
        df_cleaned = df[df[column_name] != condition_value]

    output_filename = _store_resource(df_cleaned, project_manifest_path, output_filename, explanation, 'csv')
    return {
        "output_filename": output_filename,
        "n_rows": len(df_cleaned),
        "columns": list(df_cleaned.columns),
        "preview": df_cleaned.head(5).to_dict(orient="records"),
    }


def subset_dataset(input_filename: str, project_manifest_path: str, output_filename: str, explanation: str,
                   filter_condition: str) -> dict:
    """
    Subset rows using pandas query filter expressions.
    
    Parameters
    ----------
    input_filename : str
    project_manifest_path : str
    output_filename : str
    explanation : str
    filter_condition : str
        Pandas query string. Use backticks for columns with spaces.
        Examples: 'status == "Passed"' | "value == 3" | "TPSA > 50" | "is_active == True" |
        "TPSA > 20 and MolWt < 500" | "`Molecular Weight` > 300" | "smiles.notnull()" | "label.isnull()"
    
    Returns
    -------
    dict
        Contains output_filename, n_rows, columns, preview
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    
    try:
        df_filtered = df.query(filter_condition)
    except Exception as e:
        raise ValueError(f"Invalid filter condition '{filter_condition}': {e}")

    output_filename = _store_resource(df_filtered, project_manifest_path, output_filename, explanation, 'csv')
    return {
        "output_filename": output_filename,
        "n_rows": len(df_filtered),
        "columns": list(df_filtered.columns),
        "preview": df_filtered.head(5).to_dict(orient="records"),
    }


def drop_duplicate_rows(input_filename: str, subset_columns: list[str] | None, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Remove duplicate rows based on column subset (None = all columns).

    Parameters
    ----------
    input_filename : str
        Input dataset filename
    subset_columns : list[str] | None
        Columns to check for duplicates (None = all)
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description

    Returns
    -------
    dict
        Contains output_filename, n_rows_before, n_rows_after, columns, preview
    """
    import pandas as pd

    df = _load_resource(project_manifest_path, input_filename)
    n_rows_before = len(df)

    if subset_columns is not None:
        for col in subset_columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in dataset.")

    df_deduplicated = df.drop_duplicates(subset=subset_columns)

    output_filename = _store_resource(df_deduplicated, project_manifest_path, output_filename, explanation, 'csv')
    return {
        "output_filename": output_filename,
        "n_rows_before": n_rows_before,
        "n_rows_after": len(df_deduplicated),
        "columns": list(df_deduplicated.columns),
        "preview": df_deduplicated.head(5).to_dict(orient="records"),
    }


def drop_empty_rows(input_filename: str, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Remove rows where all columns are null.

    Parameters
    ----------
    input_filename : str
        Input dataset filename
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description

    Returns
    -------
    dict
        Contains output_filename, n_rows_before, n_rows_after, columns, preview
    """
    import pandas as pd

    df = _load_resource(project_manifest_path, input_filename)
    n_rows_before = len(df)

    df_non_empty = df.dropna(how='all')

    output_filename = _store_resource(df_non_empty, project_manifest_path, output_filename, explanation, 'csv')
    return {
        "output_filename": output_filename,
        "n_rows_before": n_rows_before,
        "n_rows_after": len(df_non_empty),
        "columns": list(df_non_empty.columns),
        "preview": df_non_empty.head(5).to_dict(orient="records"),
    }


def drop_columns(
    input_filename: str,
    columns_to_drop: list[str],
    project_manifest_path: str,
    output_filename: str,
    explanation: str = 'Dropped specified columns from dataset'
) -> dict:
    """
    Drop specified columns from dataset.

    Parameters
    ----------
    input_filename : str
        Input dataset filename
    columns_to_drop : list[str]
        Columns to remove
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description

    Returns
    -------
    dict
        Contains output_filename, n_rows, n_columns_before/after, columns_dropped/remaining, preview
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    n_columns_before = len(df.columns)
    
    # Validate that all columns to drop exist
    missing_columns = [col for col in columns_to_drop if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Column(s) not found in dataset: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Validate that we're not dropping all columns
    if len(columns_to_drop) >= len(df.columns):
        raise ValueError(
            f"Cannot drop all columns. Dataset has {len(df.columns)} columns, "
            f"attempting to drop {len(columns_to_drop)}."
        )
    
    # Drop the columns
    df_reduced = df.drop(columns=columns_to_drop)
    
    output_filename = _store_resource(df_reduced, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_filename,
        "n_rows": len(df_reduced),
        "n_columns_before": n_columns_before,
        "n_columns_after": len(df_reduced.columns),
        "columns_dropped": columns_to_drop,
        "columns_remaining": list(df_reduced.columns),
        "preview": df_reduced.head(5).to_dict(orient="records"),
    }


def keep_columns(
    input_filename: str,
    columns_to_keep: list[str],
    project_manifest_path: str,
    output_filename: str,
    explanation: str = 'Kept specified columns from dataset'
) -> dict:
    """
    Keep only specified columns, drop all others.

    Parameters
    ----------
    input_filename : str
        Input dataset filename
    columns_to_keep : list[str]
        Columns to retain
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description

    Returns
    -------
    dict
        Contains output_filename, n_rows, n_columns_before/after, columns_kept/dropped, preview
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    n_columns_before = len(df.columns)
    
    # Validate that we have columns to keep
    if not columns_to_keep:
        raise ValueError("Must specify at least one column to keep.")
    
    # Validate that all columns to keep exist
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Column(s) not found in dataset: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Keep only the specified columns
    df_reduced = df[columns_to_keep]
    
    # Determine which columns were dropped
    columns_dropped = [col for col in df.columns if col not in columns_to_keep]
    
    output_filename = _store_resource(df_reduced, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_filename,
        "n_rows": len(df_reduced),
        "n_columns_before": n_columns_before,
        "n_columns_after": len(df_reduced.columns),
        "columns_kept": columns_to_keep,
        "columns_dropped": columns_dropped,
        "preview": df_reduced.head(5).to_dict(orient="records"),
    }


def transform_column(
    input_filename: str,
    expression: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str
) -> dict:
    """
    Create new column using pandas.eval() expression.

    Parameters
    ----------
    input_filename : str
        Input dataset filename
    expression : str
        Pandas eval expression with assignment (e.g., "pKi = -log10(Ki_nM / 1e9)")
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description

    Returns
    -------
    dict
        Contains output_filename, n_rows, columns, expression, preview
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    # Safe evaluation with numpy functions available
    import numpy as np
    df.eval(expression, inplace=True)
    
    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "expression": expression,
        "preview": df.head().to_dict('records')
    }


def scramble_column(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    explanation: str = 'Scrambled column for permutation testing',
    random_seed: Optional[int] = None
) -> dict:
    """
    Randomly shuffle values in a column (e.g., for y-scrambling in permutation tests).
    
    This is commonly used in machine learning to establish a baseline by randomizing
    the target variable while keeping the features intact. If the model performs
    significantly better on real data than scrambled data, it validates the model.

    Parameters
    ----------
    input_filename : str
        Input dataset filename
    column_name : str
        Name of the column to scramble
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    random_seed : int, optional
        Random seed for reproducibility. If None, results will vary each time.

    Returns
    -------
    dict
        Contains output_filename, n_rows, columns, scrambled_column, random_seed, preview
    """
    import pandas as pd
    import numpy as np
    
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column exists
    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Scramble the column by shuffling its values
    df[column_name] = np.random.permutation(df[column_name].values)
    
    # Store as new resource for full traceability
    output_id = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "scrambled_column": column_name,
        "random_seed": random_seed,
        "preview": df.head(5).to_dict(orient="records"),
    }


def combine_datasets_vertical(
    input_filenames: list[str],
    project_manifest_path: str,
    output_filename: str,
    explanation: str,
    handle_duplicates: str = 'keep_all',
    verify_columns: bool = True
) -> dict:
    """
    Stack datasets vertically (concatenate rows).

    Parameters
    ----------
    input_filenames : list[str]
        List of dataset filenames to combine
    project_manifest_path : str
        Path to manifest.json
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    handle_duplicates : str, default='keep_all'
        'keep_all', 'drop_duplicates', or 'raise_error'
    verify_columns : bool, default=True
        If True, verify all datasets have same columns

    Returns
    -------
    dict
        Contains output_filename, n_rows, n_rows_per_input, n_duplicates_dropped, columns, preview
    """
    import pandas as pd
    
    if len(input_filenames) < 2:
        raise ValueError("Must provide at least 2 datasets to combine.")
    
    # Load all datasets
    dfs = []
    row_counts = {}
    for fname in input_filenames:
        df = _load_resource(project_manifest_path, fname)
        dfs.append(df)
        row_counts[fname] = len(df)
    
    # Verify columns if requested
    if verify_columns:
        first_cols = set(dfs[0].columns)
        for i, df in enumerate(dfs[1:], start=1):
            if set(df.columns) != first_cols:
                raise ValueError(
                    f"Column mismatch: {input_filenames[0]} has {list(dfs[0].columns)}, "
                    f"but {input_filenames[i]} has {list(df.columns)}"
                )
    
    # Combine datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    n_total_before = len(combined_df)
    
    # Handle duplicates
    n_duplicates_dropped = 0
    if handle_duplicates == 'drop_duplicates':
        combined_df = combined_df.drop_duplicates()
        n_duplicates_dropped = n_total_before - len(combined_df)
    elif handle_duplicates == 'raise_error':
        n_duplicates = combined_df.duplicated().sum()
        if n_duplicates > 0:
            raise ValueError(f"Found {n_duplicates} duplicate rows. Set handle_duplicates='keep_all' or 'drop_duplicates'")
    elif handle_duplicates != 'keep_all':
        raise ValueError(f"Invalid handle_duplicates='{handle_duplicates}'. Must be 'keep_all', 'drop_duplicates', or 'raise_error'")
    
    # Store result
    output_filename_stored = _store_resource(combined_df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_filename_stored,
        "n_rows": len(combined_df),
        "n_rows_per_input": row_counts,
        "n_duplicates_dropped": n_duplicates_dropped,
        "columns": list(combined_df.columns),
        "preview": combined_df.head(5).to_dict(orient="records"),
    }


def combine_datasets_horizontal(
    project_manifest_path: str,
    left_filename: str,
    right_filename: str,
    output_filename: str,
    explanation: str,
    verify_alignment: bool = True,
    on_mismatch: str = 'raise'
) -> dict:
    """
    Combine datasets horizontally (side-by-side, adding columns).

    Parameters
    ----------
    project_manifest_path : str
        Path to manifest.json
    left_filename : str
        Left dataset filename
    right_filename : str
        Right dataset filename
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    verify_alignment : bool, default=True
        Verify datasets have same row count and index order
    on_mismatch : str, default='raise'
        'raise' or 'warn' when alignment fails

    Returns
    -------
    dict
        Contains output_filename, n_rows, n_columns, n_columns_left/right, alignment_verified, columns, preview
    """
    import pandas as pd
    
    # Load datasets
    df_left = _load_resource(project_manifest_path, left_filename)
    df_right = _load_resource(project_manifest_path, right_filename)
    
    # Check for column name overlap
    left_cols = set(df_left.columns)
    right_cols = set(df_right.columns)
    overlap = left_cols & right_cols
    if overlap:
        raise ValueError(
            f"Datasets have overlapping column names: {sorted(overlap)}. "
            "Column names must be unique across both datasets."
        )
    
    # Verify alignment if requested
    alignment_verified = False
    if verify_alignment:
        # Check row counts
        if len(df_left) != len(df_right):
            msg = (
                f"Row count mismatch: left has {len(df_left)} rows, "
                f"right has {len(df_right)} rows"
            )
            if on_mismatch == 'raise':
                raise ValueError(msg)
            elif on_mismatch == 'warn':
                print(f"WARNING: {msg}")
            else:
                raise ValueError(f"Invalid on_mismatch='{on_mismatch}'. Must be 'raise' or 'warn'")
        
        # Check index alignment
        elif not df_left.index.equals(df_right.index):
            msg = (
                f"Index mismatch: datasets have same row count ({len(df_left)}) "
                "but indices do not match. Rows may not be aligned."
            )
            if on_mismatch == 'raise':
                raise ValueError(msg)
            elif on_mismatch == 'warn':
                print(f"WARNING: {msg}")
        else:
            alignment_verified = True
    
    # Combine horizontally
    combined_df = pd.concat([df_left, df_right], axis=1)
    
    # Store result
    output_filename_stored = _store_resource(
        combined_df, 
        project_manifest_path, 
        output_filename, 
        explanation, 
        'csv'
    )
    
    return {
        "output_filename": output_filename_stored,
        "n_rows": len(combined_df),
        "n_columns": len(combined_df.columns),
        "n_columns_left": len(df_left.columns),
        "n_columns_right": len(df_right.columns),
        "alignment_verified": alignment_verified,
        "columns": list(combined_df.columns),
        "preview": combined_df.head(5).to_dict(orient="records"),
    }


def merge_datasets_on_smiles(
    project_manifest_path: str,
    left_filename: str,
    right_filename: str,
    output_filename: str,
    explanation: str,
    left_smiles_col: str,
    right_smiles_col: str,
    how: str = 'inner',
    canonicalize: bool = True,
    suffixes: tuple = ('_x', '_y')
) -> dict:
    """
    Merge datasets using SMILES as join key (with optional canonicalization).

    Parameters
    ----------
    project_manifest_path : str
        Path to manifest.json
    left_filename : str
        Left dataset filename
    right_filename : str
        Right dataset filename
    output_filename : str
        Output dataset name (no extension)
    explanation : str
        Brief description
    left_smiles_col : str
        SMILES column name in left dataset
    right_smiles_col : str
        SMILES column name in right dataset
    how : str, default='inner'
        'inner', 'left', 'right', or 'outer'
    canonicalize : bool, default=True
        Canonicalize SMILES before merging
    suffixes : tuple, default=('_x', '_y')
        Suffixes for overlapping columns

    Returns
    -------
    dict
        Contains output_filename, n_rows, n_rows_left/right, n_matched, merge_type, canonicalized, smiles_column, columns, preview
    """
    import pandas as pd
    from rdkit import Chem
    
    # Validate merge type
    valid_how = ['inner', 'left', 'right', 'outer']
    if how not in valid_how:
        raise ValueError(f"Invalid how='{how}'. Must be one of {valid_how}")
    
    # Load datasets
    df_left = _load_resource(project_manifest_path, left_filename)
    df_right = _load_resource(project_manifest_path, right_filename)
    
    # Store original row counts
    n_rows_left = len(df_left)
    n_rows_right = len(df_right)
    
    # Validate SMILES columns exist
    if left_smiles_col not in df_left.columns:
        raise ValueError(
            f"Left SMILES column '{left_smiles_col}' not found. "
            f"Available: {list(df_left.columns)}"
        )
    if right_smiles_col not in df_right.columns:
        raise ValueError(
            f"Right SMILES column '{right_smiles_col}' not found. "
            f"Available: {list(df_right.columns)}"
        )
    
    # Prepare left dataset
    df_left_merge = df_left.copy()
    if canonicalize:
        # Canonicalize left SMILES
        def canon_smiles(smi):
            if pd.isna(smi):
                return None
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    return None
                return Chem.MolToSmiles(mol)
            except:
                return None
        
        df_left_merge['_smiles_canon'] = df_left_merge[left_smiles_col].apply(canon_smiles)
        # Drop rows with invalid SMILES
        n_invalid_left = df_left_merge['_smiles_canon'].isna().sum()
        if n_invalid_left > 0:
            print(f"Warning: Dropped {n_invalid_left} rows from left dataset with invalid SMILES")
        df_left_merge = df_left_merge.dropna(subset=['_smiles_canon'])
        merge_key_left = '_smiles_canon'
    else:
        merge_key_left = left_smiles_col
    
    # Prepare right dataset
    df_right_merge = df_right.copy()
    if canonicalize:
        # Canonicalize right SMILES
        df_right_merge['_smiles_canon'] = df_right_merge[right_smiles_col].apply(canon_smiles)
        # Drop rows with invalid SMILES
        n_invalid_right = df_right_merge['_smiles_canon'].isna().sum()
        if n_invalid_right > 0:
            print(f"Warning: Dropped {n_invalid_right} rows from right dataset with invalid SMILES")
        df_right_merge = df_right_merge.dropna(subset=['_smiles_canon'])
        merge_key_right = '_smiles_canon'
    else:
        merge_key_right = right_smiles_col
    
    # Calculate number of matched molecules (before merge)
    left_smiles_set = set(df_left_merge[merge_key_left].dropna())
    right_smiles_set = set(df_right_merge[merge_key_right].dropna())
    n_matched = len(left_smiles_set & right_smiles_set)
    
    # Perform merge
    merged_df = pd.merge(
        df_left_merge,
        df_right_merge,
        left_on=merge_key_left,
        right_on=merge_key_right,
        how=how,
        suffixes=suffixes
    )
    
    # Clean up merge keys and create final SMILES column
    if canonicalize:
        # Use canonical SMILES as the output
        merged_df['smiles'] = merged_df['_smiles_canon']
        
        # Drop original SMILES columns if they still exist
        cols_to_drop = ['_smiles_canon']
        if left_smiles_col in merged_df.columns and left_smiles_col != 'smiles':
            cols_to_drop.append(left_smiles_col)
        if right_smiles_col in merged_df.columns and right_smiles_col != 'smiles':
            cols_to_drop.append(right_smiles_col)
        # Handle suffixed versions
        if f"{left_smiles_col}{suffixes[0]}" in merged_df.columns:
            cols_to_drop.append(f"{left_smiles_col}{suffixes[0]}")
        if f"{right_smiles_col}{suffixes[1]}" in merged_df.columns:
            cols_to_drop.append(f"{right_smiles_col}{suffixes[1]}")
        
        merged_df = merged_df.drop(columns=cols_to_drop)
    else:
        # Use left SMILES as primary, fill with right where needed
        if merge_key_left != merge_key_right:
            # Different column names - both will exist in merged result
            # Use left as primary, fill with right for right-only rows
            if merge_key_left in merged_df.columns and merge_key_right in merged_df.columns:
                merged_df['smiles'] = merged_df[merge_key_left].fillna(merged_df[merge_key_right])
                cols_to_drop = [c for c in [merge_key_left, merge_key_right] if c in merged_df.columns and c != 'smiles']
            elif merge_key_left in merged_df.columns:
                merged_df['smiles'] = merged_df[merge_key_left]
                cols_to_drop = [merge_key_left] if merge_key_left != 'smiles' else []
            elif merge_key_right in merged_df.columns:
                merged_df['smiles'] = merged_df[merge_key_right]
                cols_to_drop = [merge_key_right] if merge_key_right != 'smiles' else []
            else:
                # Should not happen, but just in case
                merged_df['smiles'] = None
                cols_to_drop = []
            
            merged_df = merged_df.drop(columns=cols_to_drop, errors='ignore')
        else:
            # Same column name - just rename it if needed
            if merge_key_left in merged_df.columns:
                if merge_key_left != 'smiles':
                    merged_df = merged_df.rename(columns={merge_key_left: 'smiles'})
            else:
                # Column might have gotten a suffix if there were overlaps
                if f"{merge_key_left}{suffixes[0]}" in merged_df.columns:
                    merged_df = merged_df.rename(columns={f"{merge_key_left}{suffixes[0]}": 'smiles'})
                    # Drop the right version if it exists
                    if f"{merge_key_right}{suffixes[1]}" in merged_df.columns:
                        merged_df = merged_df.drop(columns=[f"{merge_key_right}{suffixes[1]}"])
                elif f"{merge_key_right}{suffixes[1]}" in merged_df.columns:
                    # Use right if left doesn't exist
                    merged_df = merged_df.rename(columns={f"{merge_key_right}{suffixes[1]}": 'smiles'})


    
    # Reorder columns to put SMILES first
    cols = ['smiles'] + [col for col in merged_df.columns if col != 'smiles']
    merged_df = merged_df[cols]
    
    # Store result
    output_filename_stored = _store_resource(
        merged_df,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_filename_stored,
        "n_rows": len(merged_df),
        "n_rows_left": n_rows_left,
        "n_rows_right": n_rows_right,
        "n_matched": n_matched,
        "merge_type": how,
        "canonicalized": canonicalize,
        "smiles_column": "smiles",
        "columns": list(merged_df.columns),
        "preview": merged_df.head(5).to_dict(orient="records"),
    }


def read_txt(input_filename: str, project_manifest_path: str) -> Dict[str, Any]:
    """
    Read a text file from project resources and return its contents.
    
    Parameters
    ----------
    input_filename : str
        Text file name from manifest.
    project_manifest_path : str
        Path to project manifest.json.
        
    Returns
    -------
    dict
        Contains 'content' (text content) and 'filename'.
    """
    content = _load_resource(project_manifest_path, input_filename)
    
    if not isinstance(content, str):
        raise ValueError(f"Expected text content, got {type(content).__name__}")
    
    return {
        "filename": input_filename,
        "content": content,
        "n_chars": len(content),
        "n_lines": content.count('\n') + 1 if content else 0
    }


def read_json(input_filename: str, project_manifest_path: str) -> Dict[str, Any]:
    """
    Read a JSON file from project resources and return its contents as formatted text.
    
    Parameters
    ----------
    input_filename : str
        JSON file name from manifest.
    project_manifest_path : str
        Path to project manifest.json.
        
    Returns
    -------
    dict
        Contains 'data' (parsed JSON object), 'formatted_text' (pretty-printed JSON), and 'filename'.
    """
    import json
    
    data = _load_resource(project_manifest_path, input_filename)
    
    if not isinstance(data, (dict, list)):
        raise ValueError(f"Expected JSON data (dict/list), got {type(data).__name__}")
    
    formatted_text = json.dumps(data, indent=2, ensure_ascii=False)
    
    return {
        "filename": input_filename,
        "data": data,
        "formatted_text": formatted_text,
        "type": type(data).__name__
    }


def get_all_dataset_tools():
    """Return a list of all dataset manipulation tools."""
    return [
        import_csv_from_path,
        import_csv_from_text,
        get_dataset_head,
        get_dataset_full,
        get_dataset_summary,
        inspect_dataset_rows,
        drop_from_dataset,
        subset_dataset,
        drop_duplicate_rows,
        drop_empty_rows,
        drop_columns,
        keep_columns,
        transform_column,
        scramble_column,
        combine_datasets_vertical,
        combine_datasets_horizontal,
        merge_datasets_on_smiles,
        read_txt,
        read_json
    ]
