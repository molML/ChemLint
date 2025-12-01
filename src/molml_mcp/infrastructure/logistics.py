""" Logistics for storing and loading resources."""

import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import inspect
from functools import wraps

from molml_mcp.infrastructure.supported_resource_types import TYPE_REGISTRY
from molml_mcp.config import DATA_ROOT, LOG_PATH


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


def loggable(func):
    """
    Decorator that logs:
      - function name
      - first line of docstring
      - ORIGINAL inputs (before mutation)
      - full return value (outputs)

    Writes to `server.log` in a human-readable, multiline format.
    """

    def _fmt(val):
        """Compact representation to avoid huge log lines."""
        try:
            # Truncate long text fields
            if isinstance(val, str):
                if len(val) > 100:
                    return val[:97] + "..."
                return val
            
            # Represent array-like objects compactly
            if hasattr(val, "shape"):  # e.g. pandas DataFrame / numpy array
                return f"<Array-like shape={val.shape}>"
            
            # Represent large collections compactly
            if isinstance(val, (list, dict, tuple, set)) and len(val) > 30:
                return f"<{type(val).__name__} len={len(val)}>"
            return repr(val)
        except Exception:
            return "<unprintable>"

    sig = inspect.signature(func)
    doc = inspect.getdoc(func)


    @wraps(func)
    def wrapper(*args, **kwargs):
        # Capture ORIGINAL inputs as passed into the function
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        original_inputs = dict(bound.arguments)

        # Time the function execution
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        # Prepare docstring line
        if doc:
            docstring_first_line = doc.strip().split("\n")[0]
        else:
            docstring_first_line = "Description not available."

        # Format inputs and outputs
        inputs_str = {k: _fmt(v) for k, v in original_inputs.items()}

        if isinstance(result, dict):
            outputs_str = {k: _fmt(v) for k, v in result.items()}
        else:
            outputs_str = _fmt(result)

        # Build log entry in your original style
        entry = (
            datetime.now().strftime("\n%Y-%m-%d %H:%M:%S:\n")
            + f"\tFunction: {func.__name__}()\n"
            + f"\tDescription: {docstring_first_line}\n"
            + f"\tInputs: {inputs_str}\n"
            + f"\tOutputs: {outputs_str}\n"
            + f"\tExecution Time: {elapsed_time:.4f}s\n"
        )

        # Write to log file
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(entry)

        return result

    return wrapper


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


