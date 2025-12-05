"""
Client-facing dataset manipulation tools.

These functions provide MCP-accessible operations for working with tabular datasets,
including loading, inspecting, filtering, and manipulating dataset rows and columns.
"""

from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable


@loggable
def store_csv_as_dataset(file_path: str) -> dict:
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

    Returns
    -------
    dict
        {
            "resource_id": str,     # identifier for the stored dataset
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

    rid = _store_resource(df, "csv")

    return {
        "resource_id": rid,
        "n_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


@loggable
def store_csv_as_dataset_from_text(csv_content: str) -> dict:
    """
    Store CSV data from content provided by the MCP client.
    
    Parameters
    ----------
    csv_content : str
        The actual CSV file content as a string. We will use StringIO, so the content should be formatted as a valid CSV.
    filename : str
        Optional filename for reference/logging
    
    Returns
    -------
    dict
        Dataset metadata
    """
    import pandas as pd
    from io import StringIO
    
    # Read CSV from string content
    df = pd.read_csv(StringIO(csv_content))
    
    rid = _store_resource(df, "csv")
    
    return {
        "resource_id": rid,
        "n_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


def get_dataset_head(resource_id: str, n_rows: int = 10) -> dict:
    """
    Get the first n rows of a dataset for quick inspection.
    
    This is useful for quickly viewing the top of a dataset without
    loading the entire content. Perfect for initial data exploration.

    Parameters
    ----------
    resource_id : str
        Identifier for the dataset resource.
    n_rows : int, default=10
        Number of rows to return from the top of the dataset.

    Returns
    -------
    dict
        {
            "resource_id": str,       # original resource identifier
            "n_rows_returned": int,   # number of rows returned
            "n_rows_total": int,      # total rows in dataset
            "columns": list[str],     # column names
            "rows": list[dict],       # first n rows as records
        }

    Examples
    --------
    # Get first 10 rows (default)
    get_dataset_head(rid)
    
    # Get first 20 rows
    get_dataset_head(rid, n_rows=20)
    """
    import pandas as pd
    
    df = _load_resource(resource_id)
    n_total = len(df)
    
    # Get the head
    head_df = df.head(n_rows)
    
    return {
        "resource_id": resource_id,
        "n_rows_returned": len(head_df),
        "n_rows_total": n_total,
        "columns": list(df.columns),
        "rows": head_df.to_dict(orient="records"),
    }


def get_dataset_full(resource_id: str, max_rows: int = 10000) -> dict:
    """
    Get the entire dataset content.
    
    WARNING: This returns ALL rows in the dataset, which can be very large.
    Use with caution on large datasets. A safety limit of max_rows is enforced.

    Parameters
    ----------
    resource_id : str
        Identifier for the dataset resource.
    max_rows : int, default=10000
        Maximum number of rows to return (safety limit to prevent overwhelming output).

    Returns
    -------
    dict
        {
            "resource_id": str,       # original resource identifier
            "n_rows_returned": int,   # number of rows returned
            "n_rows_total": int,      # total rows in dataset
            "columns": list[str],     # column names
            "rows": list[dict],       # all rows (or first max_rows) as records
            "truncated": bool,        # True if dataset was truncated
        }

    Examples
    --------
    # Get entire dataset (up to 10000 rows)
    get_dataset_full(rid)
    
    # Get entire dataset with higher limit
    get_dataset_full(rid, max_rows=50000)
    
    Notes
    -----
    For large datasets, consider using get_dataset_head() or inspect_dataset_rows()
    with filter_condition instead of loading the entire dataset.
    """
    import pandas as pd
    
    df = _load_resource(resource_id)
    n_total = len(df)
    
    # Check if we need to truncate
    truncated = n_total > max_rows
    result_df = df.head(max_rows) if truncated else df
    
    return {
        "resource_id": resource_id,
        "n_rows_returned": len(result_df),
        "n_rows_total": n_total,
        "columns": list(df.columns),
        "rows": result_df.to_dict(orient="records"),
        "truncated": truncated,
    }


def get_dataset_summary(resource_id: str, columns: list[str] | None = None) -> dict:
    """
    Get a comprehensive summary of a dataset, similar to R's summary() function.
    
    Provides statistics for each column based on its data type:
    - Numeric columns: min, max, mean, median, std, count, n_missing
    - Non-numeric columns: data type, unique count, most common value, n_missing

    Parameters
    ----------
    resource_id : str
        Identifier for the dataset resource.
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
    
    df = _load_resource(resource_id)
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
        "resource_id": resource_id,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_columns_summarized": len(cols_to_summarize),
        "column_summaries": column_summaries,
    }


def inspect_dataset_rows(resource_id: str, row_indices: list[int] | None = None, 
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
    resource_id : str
        Identifier for the dataset resource.
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
    
    df = _load_resource(resource_id)
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
        "resource_id": resource_id,
        "n_rows_returned": len(selected_df),
        "n_rows_total": n_total,
        "columns": list(df.columns),
        "rows": selected_df.to_dict(orient="records"),
        "selection_method": selection_method,
    }


@loggable
def drop_from_dataset(resource_id: str, column_name: str, condition: str) -> dict:
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
    resource_id : str
        Identifier for the tabular dataset resource.
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
    
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    if condition == 'is None':
        df_cleaned = df[df[column_name].notnull()]
    else:
        df_cleaned = df[df[column_name] != condition]

    new_resource_id = _store_resource(df_cleaned, 'csv')
    return {
        "resource_id": new_resource_id,
        "n_rows": len(df_cleaned),
        "columns": list(df_cleaned.columns),
        "preview": df_cleaned.head(5).to_dict(orient="records"),
    }


@loggable
def keep_from_dataset(resource_id: str, column_name: str, condition: str) -> dict:
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
    resource_id : str
        Identifier for the tabular dataset resource.
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
    
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    if condition == 'is None':
        df_filtered = df[df[column_name].isnull()]
    else:
        df_filtered = df[df[column_name] == condition]

    new_resource_id = _store_resource(df_filtered, 'csv')
    return {
        "resource_id": new_resource_id,
        "n_rows": len(df_filtered),
        "columns": list(df_filtered.columns),
        "preview": df_filtered.head(5).to_dict(orient="records"),
    }


@loggable
def deduplicate_molecules_dataset(resource_id: str, molecule_id_column: str) -> dict:
    """
    Remove duplicate entries from a dataset based on a specified molecule identifier column. This should be a unique identifier for each molecule.

    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource.
    molecule_id_column : str
        Name of the column containing unique molecule identifiers.

    Returns
    -------
    dict
        Updated dataset information after removing duplicates.
    """
    import pandas as pd

    df = _load_resource(resource_id)
    n_rows_before = len(df)

    if molecule_id_column not in df.columns:
        raise ValueError(f"Column {molecule_id_column} not found in dataset.")

    df_deduplicated = df.drop_duplicates(subset=[molecule_id_column])

    new_resource_id = _store_resource(df_deduplicated, 'csv')
    return {
        "resource_id": new_resource_id,
        "n_rows_before": n_rows_before,
        "n_rows_after": len(df_deduplicated),
        "columns": list(df_deduplicated.columns),
        "preview": df_deduplicated.head(5).to_dict(orient="records"),
    }


@loggable
def drop_duplicate_rows(resource_id: str, subset_columns: list[str] | None = None) -> dict:
    """
    Remove duplicate rows from a dataset based on specified subset of columns.

    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource.
    subset_columns : list[str] | None
        List of column names to consider for identifying duplicates.
        If None, all columns are used to identify duplicates.

    Returns
    -------
    dict
        Updated dataset information after removing duplicate rows.
    """
    import pandas as pd

    df = _load_resource(resource_id)
    n_rows_before = len(df)

    if subset_columns is not None:
        for col in subset_columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in dataset.")

    df_deduplicated = df.drop_duplicates(subset=subset_columns)

    new_resource_id = _store_resource(df_deduplicated, 'csv')
    return {
        "resource_id": new_resource_id,
        "n_rows_before": n_rows_before,
        "n_rows_after": len(df_deduplicated),
        "columns": list(df_deduplicated.columns),
        "preview": df_deduplicated.head(5).to_dict(orient="records"),
    }


@loggable
def drop_empty_rows(resource_id: str) -> dict:
    """
    Remove rows from a dataset that are completely empty (all columns are null).

    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource.

    Returns
    -------
    dict
        Updated dataset information after removing empty rows.
    """
    import pandas as pd

    df = _load_resource(resource_id)
    n_rows_before = len(df)

    df_non_empty = df.dropna(how='all')

    new_resource_id = _store_resource(df_non_empty, 'csv')
    return {
        "resource_id": new_resource_id,
        "n_rows_before": n_rows_before,
        "n_rows_after": len(df_non_empty),
        "columns": list(df_non_empty.columns),
        "preview": df_non_empty.head(5).to_dict(orient="records"),
    }


def get_all_dataset_tools():
    """Return a list of all dataset manipulation tools."""
    return [
        store_csv_as_dataset,
        store_csv_as_dataset_from_text,
        get_dataset_head,
        get_dataset_full,
        get_dataset_summary,
        inspect_dataset_rows,
        drop_from_dataset,
        keep_from_dataset,
        deduplicate_molecules_dataset,
        drop_duplicate_rows,
        drop_empty_rows
    ]
