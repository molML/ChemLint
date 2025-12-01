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


@loggable
def inspect_dataset_rows(resource_id: str, row_indices: list[int] | None = None, 
                         filter_condition: str | None = None, max_rows: int = 100) -> dict:
    """
    Inspect specific rows from a dataset by index or filter condition.
    
    Useful for examining rows that failed validation, have null values, or meet
    specific criteria. Returns full row data for detailed inspection.

    Parameters
    ----------
    resource_id : str
        Identifier for the dataset resource.
    row_indices : list[int] | None
        List of row indices (0-based) to retrieve. If provided, filter_condition is ignored.
    filter_condition : str | None
        Pandas query string to filter rows (e.g., "smiles.isnull()" or "label < 0").
        Only used if row_indices is None.
    max_rows : int, default=100
        Maximum number of rows to return (safety limit to prevent large outputs).

    Returns
    -------
    dict
        {
            "resource_id": str,          # original resource identifier
            "n_rows_returned": int,      # number of rows in result
            "n_rows_total": int,         # total rows in dataset
            "columns": list[str],        # column names
            "rows": list[dict],          # retrieved rows as records
            "selection_method": str,     # "indices" or "filter"
        }

    Examples
    --------
    # Inspect specific rows by index
    inspect_dataset_rows(rid, row_indices=[5, 10, 15])
    
    # Find rows with null SMILES
    inspect_dataset_rows(rid, filter_condition="canonic_smiles.isnull()")
    
    # Find rows where conversion failed (assuming a 'valid' column)
    inspect_dataset_rows(rid, filter_condition="valid == False")
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
    Drop rows from a dataset based on a condition applied to a specified column.

    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource.
    column_name : str
        Name of the column to apply the condition on.
    condition : str
        Condition to evaluate for dropping rows (e.g., 'is None', '== "Failed: Invalid SMILES string"').

    Returns
    -------
    dict
        Updated dataset information after dropping specified rows.
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
    Keep only rows from a dataset based on a condition applied to a specified column.

    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource.
    column_name : str
        Name of the column to apply the condition on.
    condition : str
        Condition to evaluate for keeping rows (e.g., 'is None', '== "Passed"').

    Returns
    -------
    dict
        Updated dataset information after keeping specified rows.
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
        inspect_dataset_rows,
        drop_from_dataset,
        keep_from_dataset,
        deduplicate_molecules_dataset,
        drop_duplicate_rows,
        drop_empty_rows
    ]
