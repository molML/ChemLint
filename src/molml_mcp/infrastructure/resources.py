""" Logistics for storing and loading resources."""

import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from molml_mcp.infrastructure.logging import loggable
from molml_mcp.infrastructure.supported_resource_types import TYPE_REGISTRY
from molml_mcp.config import DATA_ROOT


def get_all_resource_functions():
    """Return a list of all resource-related functions available."""
    return [
        store_csv_as_dataset,
        store_csv_as_dataset_from_text,
        # Add other resource functions here as they are implemented
    ]

def _generate_id(type_tag: str) -> str:
    """Generate a unique resource ID with timestamp, type tag, and random ID component."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    rand = secrets.token_hex(4).upper()  # 8 hex chars

    ext = TYPE_REGISTRY[type_tag]["ext"]
    
    return f"{ts}_{type_tag}_{rand}{ext}"


def _store_resource(obj: Any, type_tag: str) -> str:
    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unsupported resource type: {type_tag}")
    
    rid = _generate_id(type_tag)

    path = DATA_ROOT / f"{rid}"
    save_fn: Callable[[Any, Path], None] = TYPE_REGISTRY[type_tag]['save']
    save_fn(obj, path)

    return rid


def _load_resource(resource_id: str) -> Any:
    # infer type from the ID. They are always in the format: TIMESTAMP_TYPE_RANDOM.ext
    try:
        type_tag = resource_id.split("_")[-2]
    except ValueError:
        raise ValueError(f"Invalid resource_id format: {resource_id}")

    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unknown resource type in id: {resource_id}")

    path = DATA_ROOT / f"{resource_id}"

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    load_fn: Callable[[Path], Any] = TYPE_REGISTRY[type_tag]["load"]

    return load_fn(path)


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




# def load_csv_dataset(file_path: str, smiles_column: str | None = None) -> dict:
#     """Load a CSV file into a dataset store; optionally mark a SMILES column; return {'resource_id', 'n_rows', 'columns', 'preview'}."""

# def load_sdf_dataset(file_path: str) -> dict:
#     """Load an SDF file into a dataset store, extracting structures and basic fields; return {'resource_id', 'n_rows', 'columns', 'preview'}."""

# def save_dataset_to_csv(resource_id: str, file_path: str, include_hidden: bool = False) -> dict:
#     """Save a dataset to a CSV file and return {'file_path', 'n_rows', 'n_columns'}."""

# def save_dataset_to_csv(resource_id: str, file_path: str, include_hidden: bool = False) -> dict:
#     """Save a dataset to a CSV file and return {'file_path', 'n_rows', 'n_columns'}."""

# def select_dataset_columns(resource_id: str, columns: list[str], inplace: bool = False) -> dict:
#     """Keep only specified columns; return {'resource_id', 'columns'} (new or same id depending on inplace)."""

# def drop_dataset_columns(resource_id: str, columns: list[str], inplace: bool = False) -> dict:
#     """Drop specified columns; return {'resource_id', 'dropped_columns', 'columns'}."""

# def filter_dataset_rows(resource_id: str, expression: str, inplace: bool = False) -> dict:
#     """Filter rows using a boolean expression (e.g. pandas-style); return {'resource_id', 'n_rows_before', 'n_rows_after'}."""

# def merge_datasets_on_key(
#     left_resource_id: str,
#     right_resource_id: str,
#     on: str,
#     how: str = "inner"
# ) -> dict:
#     """Merge two datasets on a key column; return {'resource_id', 'n_rows', 'n_columns', 'how'}."""

# def concatenate_datasets(resource_ids: list[str]) -> dict:
#     """Concatenate multiple datasets row-wise; return {'resource_id', 'n_rows', 'source_ids'}."""






# @loggable
# def function_A(test_input:int, *args, **kwargs):
#     """ Example function A to demonstrate logging. """

#     test_input=1234

#     func_in = log_info({'output_a': 'A'})

#     # saved_args = {**locals()} 

#     # print(f"Function name = '{func_name()}'")
#     # print(f'saved_args = {saved_args}')

#     print("Function A called")


# function_A(test_input=123, extra_param="hello")


