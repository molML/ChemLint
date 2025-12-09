"""
Client-facing dataset manipulation tools.

These functions provide MCP-accessible operations for working with tabular datasets,
including loading, inspecting, filtering, and manipulating dataset rows and columns.
"""

from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable
from typing import Any, Dict, List, Optional



@loggable
def store_csv_as_dataset(file_path: str, project_manifest_path: str, filename: str, explanation: str) -> dict:
    """
    Store a CSV file from a local file path provided by the MCP client.
    
    This tool reads the CSV into a tabular dataset, stores it as an internal
    resource, and returns a new `resource_id` together with basic dataset
    information. No assumptions are made about the meaning of columns.

    Parameters
    ----------
    file_path : str
        Path to a CSV file supplied by the client (e.g. from a drag-and-drop
        upload). The file is read exactly as provided.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of what this dataset contains.

    Returns
    -------
    dict
        {
            "output_filename": str,     # identifier for the stored dataset
            "n_rows": int,          # number of rows in the dataset
            "columns": list[str],   # all column names detected in the CSV
            "preview": list[dict],  # first 5 rows as records (for inspection)
        }

    Notes
    -----
    - This tool only loads and stores the uploaded file. It does not perform
      any cleaning, type inference, or column guessing.
    - Subsequent tools should operate using the returned `resource_id`.
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


@loggable
def store_csv_as_dataset_from_text(csv_content: str, project_manifest_path: str, filename: str, explanation: str) -> dict:
    """
    Store CSV data from content provided by the MCP client.
    
    Parameters
    ----------
    csv_content : str
        The actual CSV file content as a string. We will use StringIO, so the content should be formatted as a valid CSV.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of what this dataset contains.
    
    Returns
    -------
    dict
        Dataset metadata
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
    Get the first n rows of a dataset for quick inspection.
    
    This is useful for quickly viewing the top of a dataset without
    loading the entire content. Perfect for initial data exploration.

    Parameters
    ----------
    project_manifest_path : str
        Path to the project manifest file.
    input_filename : str
        Base filename of the dataset resource.
    n_rows : int, default=10
        Number of rows to return from the top of the dataset.

    Returns
    -------
    dict
        {
            "input_filename": str,    # original resource identifier
            "n_rows_returned": int,   # number of rows returned
            "n_rows_total": int,      # total rows in dataset
            "columns": list[str],     # column names
            "rows": list[dict],       # first n rows as records
        }

    Examples
    --------
    # Get first 10 rows (default)
    get_dataset_head(manifest_path, 'my_dataset')
    
    # Get first 20 rows
    get_dataset_head(manifest_path, 'my_dataset', n_rows=20)
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
    Get the entire dataset content.
    
    WARNING: This returns ALL rows in the dataset, which can be very large.
    Use with caution on large datasets. A safety limit of max_rows is enforced.

    Parameters
    ----------
    project_manifest_path : str
        Path to the project manifest file.
    input_filename : str
        Base filename of the dataset resource.
    max_rows : int, default=10000
        Maximum number of rows to return (safety limit to prevent overwhelming output).

    Returns
    -------
    dict
        {
            "input_filename": str,    # original resource identifier
            "n_rows_returned": int,   # number of rows returned
            "n_rows_total": int,      # total rows in dataset
            "columns": list[str],     # column names
            "rows": list[dict],       # all rows (or first max_rows) as records
            "truncated": bool,        # True if dataset was truncated
        }

    Examples
    --------
    # Get entire dataset (up to 10000 rows)
    get_dataset_full(manifest_path, 'my_dataset')
    
    # Get entire dataset with higher limit
    get_dataset_full(manifest_path, 'my_dataset', max_rows=50000)
    
    Notes
    -----
    For large datasets, consider using get_dataset_head() or inspect_dataset_rows()
    with filter_condition instead of loading the entire dataset.
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
    Get a comprehensive summary of a dataset, similar to R's summary() function.
    
    Provides statistics for each column based on its data type:
    - Numeric columns: min, max, mean, median, std, count, n_missing
    - Non-numeric columns: data type, unique count, most common value, n_missing

    Parameters
    ----------
    project_manifest_path : str
        Path to the project manifest file.
    input_filename : str
        Base filename of the dataset resource.
    columns : list[str] | None, optional
        List of specific column names to summarize. If None, all columns are summarized.
        This is useful for large dataframes with many columns where you only want to
        examine a subset of columns to reduce computation time and output size.

    Returns
    -------
    dict
        {
            "resource_id": str,           # original resource identifier
            "n_rows": int,                # total rows in dataset
            "n_columns": int,             # total columns in dataset
            "n_columns_summarized": int,  # number of columns included in summary
            "column_summaries": dict,     # summary for each column
        }
        
        Each column_summary contains:
        - For numeric columns:
            {
                "dtype": str,
                "count": int,      # non-null count
                "n_missing": int,  # null count
                "min": float,
                "max": float,
                "mean": float,
                "median": float,
                "std": float,
            }
        - For non-numeric columns:
            {
                "dtype": str,
                "count": int,           # non-null count
                "n_missing": int,       # null count
                "n_unique": int,        # number of unique values
                "top_value": any,       # most common value
                "top_freq": int,        # frequency of most common value
            }

    Examples
    --------
    # Get summary statistics for all columns
    get_dataset_summary(rid)
    
    # Get summary for specific columns only (useful for large dataframes)
    get_dataset_summary(rid, columns=['TPSA', 'MolWt', 'label'])
    
    Notes
    -----
    This function is useful for initial data exploration and understanding
    the distribution and types of data in each column. For large dataframes
    with many columns, use the `columns` parameter to summarize only the
    columns of interest, which improves performance and reduces output size.
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
    Inspect or filter rows from a dataset by index or complex conditions.
    
    This is the recommended tool for filtering datasets with numeric comparisons,
    complex conditions, or when you need to examine specific rows. It supports
    full pandas query syntax including >, <, >=, <=, ==, !=, and logical operators.
    
    Use this tool instead of keep_from_dataset/drop_from_dataset when you need:
    - Numeric comparisons (e.g., "TPSA > 20", "MolWt < 500")
    - Range filters (e.g., "200 <= MolWt <= 600")
    - Multiple conditions (e.g., "TPSA > 20 and MolWt < 400")
    - Null value checks (e.g., "column_name.isnull()")

    Parameters
    ----------
    project_manifest_path : str
        Path to the project manifest file.
    input_filename : str
        Base filename of the dataset resource.
    row_indices : list[int] | None
        List of row indices (0-based) to retrieve. If provided, filter_condition is ignored.
    filter_condition : str | None
        Pandas query string to filter rows. USE BACKTICKS AROUND COLUMN NAMES WITH SPACES OR SPECIAL CHARS.
        
        CRITICAL SYNTAX RULES:
        - Column names: Use backticks for spaces/special chars: `column name` or `comments_after_salt_removal`
        - String values: Use DOUBLE quotes inside the filter string: column == "value"
        - The entire filter_condition should be a single string
        - Operators: >, <, >=, <=, ==, != (use == for equality, not single =)
        - Logic: and, or, not (lowercase required)
        
        Supported patterns:
        - Numeric comparison: "TPSA > 20" or "MolWt <= 500.5"
        - Range: "200 <= MolWt <= 600" (chained comparison)
        - Equality: "label == 1" or 'status == "active"' (note quotes)
        - Null checks: "column_name.isnull()" or "column_name.notnull()"
        - Multiple conditions: "TPSA > 20 and MolWt < 400"
        - Negation: "not (TPSA > 100)" or "status != \"failed\""
        
        COMMON MISTAKES TO AVOID:
        - ❌ "column = value" → ✅ "column == value" (use double ==)
        - ❌ "column == 'value'" → ✅ 'column == "value"' (use double quotes for strings)
        - ❌ "TPSA > 20 AND MolWt < 500" → ✅ "TPSA > 20 and MolWt < 500" (lowercase 'and')
        - ❌ "column with spaces > 10" → ✅ "`column with spaces` > 10" (use backticks)
        
    max_rows : int, default=100
        Maximum number of rows to return (safety limit to prevent large outputs).

    Returns
    -------
    dict
        {
            "resource_id": str,          # original resource identifier
            "n_rows_returned": int,      # number of rows matching condition
            "n_rows_total": int,         # total rows in dataset
            "columns": list[str],        # column names
            "rows": list[dict],          # retrieved rows as records
            "selection_method": str,     # "indices" or "filter"
        }

    Examples
    --------
    # Numeric comparison (simple)
    inspect_dataset_rows(rid, filter_condition="TPSA > 20")
    inspect_dataset_rows(rid, filter_condition="MolWt <= 500.5")
    
    # Range filter (chained comparison)
    inspect_dataset_rows(rid, filter_condition="200 <= MolWt <= 600")
    
    # Multiple numeric conditions
    inspect_dataset_rows(rid, filter_condition="TPSA > 20 and MolLogP < 5")
    inspect_dataset_rows(rid, filter_condition="TPSA > 20 or MolWt > 500")
    
    # String value comparison (note the double quotes around "Passed")
    inspect_dataset_rows(rid, filter_condition='comments_after_canonicalization == "Passed"')
    inspect_dataset_rows(rid, filter_condition='status != "Failed"')
    
    # Column names with spaces or special characters (use backticks)
    inspect_dataset_rows(rid, filter_condition='`comments after cleaning` == "Passed"')
    
    # Null value checks
    inspect_dataset_rows(rid, filter_condition="smiles_after_canonicalization.isnull()")
    inspect_dataset_rows(rid, filter_condition="label.notnull()")
    
    # Complex conditions
    inspect_dataset_rows(rid, filter_condition="TPSA > 20 and MolWt < 400 and label == 1")
    inspect_dataset_rows(rid, filter_condition='(TPSA > 100 or MolWt > 500) and status == "active"')
    
    # Inspect specific rows by index (alternative to filter_condition)
    inspect_dataset_rows(rid, row_indices=[5, 10, 15, 20])
    
    Notes
    -----
    This tool returns matching rows for inspection but does NOT create a new
    filtered dataset. To create a new filtered dataset, you would need to use
    a different workflow or tool.
    
    The filter_condition uses pandas.DataFrame.query() syntax. For detailed
    documentation, see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html
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


@loggable
def drop_from_dataset(input_filename: str, column_name: str, condition: str, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Drop rows from a dataset based on SIMPLE conditions (exact match or null check).
    
    This tool only supports TWO condition types:
    1. Drop rows with null/missing values: condition="is None" (EXACT STRING)
    2. Drop rows matching EXACT string value: condition="Passed" or condition="Failed"
    
    IMPORTANT: This is NOT pandas query syntax. Do NOT include == or quotes for exact matches.
    Just provide the literal string value to match.
    
    For numeric comparisons (>, <, >=, <=) or complex conditions, use
    inspect_dataset_rows() with filter_condition instead.

    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    column_name : str
        Name of the column to check.
    condition : str
        EITHER:
        - "is None" (EXACT STRING, no quotes) - drops rows where column is null/missing
        - "exact_value" (just the value, no == or quotes) - drops rows matching this EXACT value
        
        CRITICAL: Do NOT use pandas query syntax here:
        - ❌ '== "Passed"' → ✅ "Passed"
        - ❌ "== Failed" → ✅ "Failed"
        - ❌ 'is None' (with quotes around None) → ✅ "is None" (literal string)
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of what filtering was applied.

    Returns
    -------
    dict
        {
            "resource_id": str,        # new dataset without dropped rows
            "n_rows": int,             # rows remaining after drop
            "columns": list[str],      # column names
            "preview": list[dict],     # first 5 rows
        }
    
    Examples
    --------
    # Drop rows with null/missing values (use exact string "is None")
    drop_from_dataset(rid, "smiles_after_canonicalization", "is None")
    
    # Drop rows where column value is exactly "Failed" (no == needed)
    drop_from_dataset(rid, "validation_status", "Failed")
    
    # Drop rows where comment is exactly "Failed: Invalid SMILES string"
    drop_from_dataset(rid, "comments", "Failed: Invalid SMILES string")
    
    # Drop rows where status is exactly "Passed"
    drop_from_dataset(rid, "pains_screening", "Passed")
    
    Notes
    -----
    - This tool does EXACT string matching, not pattern matching
    - For numeric filtering (e.g., "TPSA > 20"), use inspect_dataset_rows() instead
    - For complex conditions, use inspect_dataset_rows() with filter_condition
    - The condition parameter is NOT pandas query syntax - just provide the literal value
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


@loggable
def keep_from_dataset(input_filename: str, column_name: str, condition: str, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Keep only rows from a dataset based on SIMPLE conditions (exact match or null check).
    
    This tool only supports TWO condition types:
    1. Keep rows with null/missing values: condition="is None" (EXACT STRING)
    2. Keep rows matching EXACT string value: condition="Passed" or condition="active"
    
    IMPORTANT: This is NOT pandas query syntax. Do NOT include == or quotes for exact matches.
    Just provide the literal string value to match.
    
    For numeric comparisons (>, <, >=, <=) or complex conditions, use
    inspect_dataset_rows() with filter_condition instead.

    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    column_name : str
        Name of the column to check.
    condition : str
        EITHER:
        - "is None" (EXACT STRING, no quotes) - keeps rows where column is null/missing
        - "exact_value" (just the value, no == or quotes) - keeps rows matching this EXACT value
        
        CRITICAL: Do NOT use pandas query syntax here:
        - ❌ '== "Passed"' → ✅ "Passed"
        - ❌ "== active" → ✅ "active"
        - ❌ 'is None' (with quotes around None) → ✅ "is None" (literal string)
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of what filtering was applied.

    Returns
    -------
    dict
        {
            "resource_id": str,        # new dataset with only kept rows
            "n_rows": int,             # rows remaining after filter
            "columns": list[str],      # column names
            "preview": list[dict],     # first 5 rows
        }
    
    Examples
    --------
    # Keep only rows with successful canonicalization (use exact string "Passed")
    keep_from_dataset(rid, "comments_after_canonicalization", "Passed")
    
    # Keep only rows where status is exactly "active" (no == needed)
    keep_from_dataset(rid, "status", "active")
    
    # Keep only rows with null/missing labels for review
    keep_from_dataset(rid, "label", "is None")
    
    # Keep only PAINS-free molecules (where screening result is exactly "Passed")
    keep_from_dataset(rid, "pains_screening", "Passed")
    
    Notes
    -----
    - This tool does EXACT string matching, not pattern matching
    - For numeric filtering (e.g., "TPSA > 20"), use inspect_dataset_rows() instead
    - For complex conditions, use inspect_dataset_rows() with filter_condition
    - The condition parameter is NOT pandas query syntax - just provide the literal value
    """
    import pandas as pd
    
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    if condition == 'is None':
        df_filtered = df[df[column_name].isnull()]
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
        
        df_filtered = df[df[column_name] == condition_value]

    output_filename = _store_resource(df_filtered, project_manifest_path, output_filename, explanation, 'csv')
    return {
        "output_filename": output_filename,
        "n_rows": len(df_filtered),
        "columns": list(df_filtered.columns),
        "preview": df_filtered.head(5).to_dict(orient="records"),
    }


@loggable
def deduplicate_molecules_dataset(input_filename: str, molecule_id_column: str, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Remove duplicate entries from a dataset based on a specified molecule identifier column. This should be a unique identifier for each molecule.

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


@loggable
def drop_duplicate_rows(input_filename: str, subset_columns: list[str] | None, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Remove duplicate rows from a dataset based on specified subset of columns.

    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    subset_columns : list[str] | None
        List of column names to consider for identifying duplicates.
        If None, all columns are used to identify duplicates.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of the deduplication performed.

    Returns
    -------
    dict
        Updated dataset information after removing duplicate rows.
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


@loggable
def drop_empty_rows(input_filename: str, project_manifest_path: str, output_filename: str, explanation: str) -> dict:
    """
    Remove rows from a dataset that are completely empty (all columns are null).

    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of the cleaning performed.

    Returns
    -------
    dict
        Updated dataset information after removing empty rows.
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


@loggable
def drop_columns(
    input_filename: str,
    columns_to_drop: list[str],
    project_manifest_path: str,
    output_filename: str,
    explanation: str = 'Dropped specified columns from dataset'
) -> dict:
    """
    Drop specified columns from a dataset and save as a new CSV.
    
    Removes one or more columns from a dataset and creates a new resource with
    the remaining columns. This is useful for removing intermediate processing
    columns, comment columns, or any other columns that are no longer needed.

    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    columns_to_drop : list[str]
        List of column names to drop from the dataset.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of what columns were dropped and why.

    Returns
    -------
    dict
        {
            "output_filename": str,        # new dataset without dropped columns
            "n_rows": int,                 # number of rows (unchanged)
            "n_columns_before": int,       # number of columns before dropping
            "n_columns_after": int,        # number of columns after dropping
            "columns_dropped": list[str],  # columns that were dropped
            "columns_remaining": list[str],# columns that remain
            "preview": list[dict],         # first 5 rows
        }
    
    Raises
    ------
    ValueError
        If any specified column is not found in the dataset.
    
    Examples
    --------
    # Drop all comment columns after cleaning pipeline
    drop_columns(
        input_filename="cleaned_data_A3F2B1D4.csv",
        columns_to_drop=["comments_after_canonicalization", 
                        "comments_after_salt_removal"],
        project_manifest_path="/path/to/manifest.json",
        output_filename="cleaned_final",
        explanation="Removed intermediate comment columns"
    )
    
    # Drop original SMILES column after standardization
    drop_columns(
        input_filename="standardized_A3F2B1D4.csv",
        columns_to_drop=["smiles"],
        project_manifest_path="/path/to/manifest.json",
        output_filename="final_standardized",
        explanation="Removed original SMILES, keeping only standardized_smiles"
    )
    
    # Drop multiple intermediate columns
    drop_columns(
        input_filename="pipeline_output_A3F2B1D4.csv",
        columns_to_drop=["smiles_after_canonicalization",
                        "smiles_after_salt_removal",
                        "smiles_after_solvent_removal"],
        project_manifest_path="/path/to/manifest.json",
        output_filename="cleaned_minimal",
        explanation="Removed intermediate SMILES columns, keeping only final result"
    )
    
    Notes
    -----
    - All specified columns must exist in the dataset, or an error will be raised
    - At least one column must remain after dropping (cannot drop all columns)
    - The operation creates a new resource; the original dataset is unchanged
    - Row order and values are preserved exactly
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


@loggable
def keep_columns(
    input_filename: str,
    columns_to_keep: list[str],
    project_manifest_path: str,
    output_filename: str,
    explanation: str = 'Kept specified columns from dataset'
) -> dict:
    """
    Keep only specified columns from a dataset and save as a new CSV.
    
    Retains only the specified columns from a dataset and creates a new resource
    with just those columns. This is useful for creating a minimal dataset with
    only the essential columns needed for downstream analysis.

    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    columns_to_keep : list[str]
        List of column names to keep in the dataset. All other columns will be dropped.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource (without extension).
    explanation : str
        Brief description of what columns were kept and why.

    Returns
    -------
    dict
        {
            "output_filename": str,        # new dataset with only kept columns
            "n_rows": int,                 # number of rows (unchanged)
            "n_columns_before": int,       # number of columns before filtering
            "n_columns_after": int,        # number of columns after filtering
            "columns_kept": list[str],     # columns that were kept
            "columns_dropped": list[str],  # columns that were dropped
            "preview": list[dict],         # first 5 rows
        }
    
    Raises
    ------
    ValueError
        If any specified column is not found in the dataset, or if no columns
        are specified to keep.
    
    Examples
    --------
    # Keep only SMILES and label columns for ML training
    keep_columns(
        input_filename="full_dataset_A3F2B1D4.csv",
        columns_to_keep=["standardized_smiles", "label"],
        project_manifest_path="/path/to/manifest.json",
        output_filename="training_data",
        explanation="Kept only SMILES and label for model training"
    )
    
    # Keep only final standardized SMILES
    keep_columns(
        input_filename="pipeline_output_A3F2B1D4.csv",
        columns_to_keep=["standardized_smiles", "validation_status"],
        project_manifest_path="/path/to/manifest.json",
        output_filename="final_smiles",
        explanation="Kept only standardized SMILES and validation status"
    )
    
    # Keep specific identifier and result columns
    keep_columns(
        input_filename="analysis_results_A3F2B1D4.csv",
        columns_to_keep=["molecule_id", "smiles", "predicted_activity", "confidence"],
        project_manifest_path="/path/to/manifest.json",
        output_filename="predictions",
        explanation="Kept only ID, SMILES, and prediction results"
    )
    
    Notes
    -----
    - All specified columns must exist in the dataset, or an error will be raised
    - At least one column must be specified to keep
    - The operation creates a new resource; the original dataset is unchanged
    - Row order and values are preserved exactly
    - This is the inverse operation of drop_columns()
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
            high_variability = avg_cv > 0.5  # CV > 50% is very high
            extreme_range = max_range > avg_range * 3 if avg_range > 0 else False  # Outlier detection
            very_high_conflict_rate = label_analysis["conflict_rate"] > 80
            
            # Only recommend drop if variability is truly problematic
            should_drop = high_variability or extreme_range or very_high_conflict_rate
            
            strategies.append({
                "strategy": "mean",
                "description": "Average all values (good for measurements with random error)",
                "recommended": avg_cv < 0.2 and not should_drop
            })
            strategies.append({
                "strategy": "median", 
                "description": "Take median value (robust to outliers)",
                "recommended": 0.2 <= avg_cv < 0.5 and not should_drop
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
                    "recommended": label_analysis["conflict_rate"] > 30
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
                            "has_conflict": len(vals.unique()) > 1 and (vals.std() / vals.abs().mean() > 0.01 if vals.abs().mean() != 0 else vals.std() > 0)
                        }
                    else:
                        unique_vals = vals.unique()
                        entry["label_stats"] = {
                            "unique_values": unique_vals.tolist(),
                            "n_unique": len(unique_vals),
                            "has_conflict": len(unique_vals) > 1,
                            "most_common": vals.mode()[0] if len(vals.mode()) > 0 else None
                        }

            preview.append(entry)

        result["duplicate_preview"] = preview

    return result


def get_all_dataset_tools():
    """Return a list of all dataset manipulation tools."""
    return [
        store_csv_as_dataset,
        store_csv_as_dataset_from_text,
        get_dataset_head,
        get_dataset_full,
        get_dataset_summary,
        inspect_dataset_rows,
        inspect_duplicates_dataset,
        drop_from_dataset,
        keep_from_dataset,
        deduplicate_molecules_dataset,
        drop_duplicate_rows,
        drop_empty_rows,
        drop_columns,
        keep_columns
    ]
