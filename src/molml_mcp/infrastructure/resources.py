"""
Internal resource management infrastructure.

This module handles the low-level operations for storing and loading resources
(datasets, models, JSON files, etc.) from the filesystem. These are internal
functions used by client-facing tools but not directly exposed to MCP clients.
"""

import secrets
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from molml_mcp.infrastructure.supported_resource_types import TYPE_REGISTRY
from molml_mcp.config import DATA_ROOT


def get_supported_resource_types() -> list[str]:
    """Return a list of supported resource types."""
    return list(TYPE_REGISTRY.keys())


def _get_timestamp() -> str:
    """Generate ISO-format timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def _get_parent_info() -> dict:
    """Get information about the parent function that called _store_resource.
    
    Walks up the call stack to find the first function that is NOT _store_resource,
    and extracts useful metadata about it.
    
    Returns:
        dict with keys:
            - 'function_name': Name of the parent function (str or None)
            - 'function_inputs': Dictionary of argument names and values (dict or None)
            - 'docstring_first_line': First line of parent's docstring (str or None)
            - 'module': Module where parent function is defined (str or None)
    """
    try:
        # Get the current call stack
        stack = inspect.stack()
        
        # Walk up the stack to find the caller of _store_resource
        # stack[0] = _get_parent_info (this function)
        # stack[1] = _store_resource (the function calling us)
        # stack[2+] = the actual parent functions we want info about
        
        parent_frame = None
        for frame_info in stack[2:]:  # Skip _get_parent_info and _store_resource
            # Look for the first frame that's not in this resources.py module
            # This helps skip internal helper functions
            frame_filename = frame_info.filename
            if 'resources.py' not in frame_filename:
                parent_frame = frame_info
                break
        
        if parent_frame is None:
            # Fallback: if all frames are in resources.py, use stack[2]
            parent_frame = stack[2] if len(stack) > 2 else None
        
        if parent_frame is None:
            return {
                'function_name': 'unknown',
                'function_inputs': {},
                'docstring_first_line': 'unknown',
                'module': 'unknown'
            }
        
        # Extract function name
        function_name = parent_frame.function
        
        # Get the function object to access its docstring
        frame = parent_frame.frame
        func_obj = frame.f_globals.get(function_name)
        
        # Extract docstring first line
        docstring_first_line = None
        if func_obj and hasattr(func_obj, '__doc__') and func_obj.__doc__:
            docstring = func_obj.__doc__.strip()
            if docstring:
                # Get first non-empty line
                first_line = docstring.split('\n')[0].strip()
                docstring_first_line = first_line
        
        # Extract function inputs (local variables at the time of call)
        # Get the argument names and their values
        arginfo = inspect.getargvalues(frame)
        function_inputs = {}
        
        for arg_name in arginfo.args:
            arg_value = arginfo.locals.get(arg_name)
            # Convert to a string representation, handling complex types
            try:
                # For simple types, just use the value
                if isinstance(arg_value, (str, int, float, bool, type(None))):
                    function_inputs[arg_name] = arg_value
                # For lists/tuples, show type and length
                elif isinstance(arg_value, (list, tuple)):
                    function_inputs[arg_name] = f"<{type(arg_value).__name__} of length {len(arg_value)}>"
                # For dicts, show type and key count
                elif isinstance(arg_value, dict):
                    function_inputs[arg_name] = f"<dict with {len(arg_value)} keys>"
                # For other objects, show type and repr if short
                else:
                    repr_str = repr(arg_value)
                    if len(repr_str) < 100:
                        function_inputs[arg_name] = repr_str
                    else:
                        function_inputs[arg_name] = f"<{type(arg_value).__name__}>"
            except Exception:
                # If anything goes wrong, just store the type
                function_inputs[arg_name] = f"<{type(arg_value).__name__}>"
        
        # Extract module name
        module = inspect.getmodule(frame)
        module_name = module.__name__ if module else None
        
        return {
            'function_name': function_name,
            'function_inputs': function_inputs,
            'docstring_first_line': docstring_first_line,
            'module': module_name
        }
        
    except Exception as e:
        # If anything goes wrong with introspection, return 'unknown' values
        # This ensures _store_resource doesn't fail due to introspection issues
        return {
            'function_name': 'unknown',
            'function_inputs': {},
            'docstring_first_line': 'unknown',
            'module': 'unknown',
            'error': str(e)
        }


def _generate_id(type_tag: str) -> str:
    """Generate unique resource ID: {type_tag}_{8_hex_chars}{extension}."""
    rand = secrets.token_hex(4).upper()  # 8 hex chars

    ext = TYPE_REGISTRY[type_tag]["ext"]
    
    return f"{type_tag}_{rand}{ext}"


def _store_resource(obj: Any, project_manifest_path: str, filename: str, explaination: str, type_tag: str) -> str:
    """Internal: Store object to disk, auto-track in manifest with parent function metadata.
    
    Args:
        obj: Object to store (DataFrame, model, etc.)
        project_manifest_path: Path to project_manifest.json
        filename: Base filename (no extension)
        explaination: Brief description of resource
        type_tag: Type from TYPE_REGISTRY (csv, model, json, etc.)
        
    Returns:
        Path to stored file
    """
    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unsupported resource type: {type_tag}")
    
    # gerate unique resource ID so if the client gives the same filename again, we dont overwrite. 
    # The type tag is included in the RID so we can always read the file later
    rid = _generate_id(type_tag)

    # first get data directory from manifest path,
    data_dir = Path(project_manifest_path).parent
    path = data_dir / f"{filename}_{rid}"

    save_fn: Callable[[Any, Path], None] = TYPE_REGISTRY[type_tag]['save']
    save_fn(obj, path)
    ts = _get_timestamp()

    parent_info = _get_parent_info()

    # add entry to project manifest
    add_to_project_manifest(project_manifest_path=project_manifest_path, 
                            filename=filename, type_tag=type_tag, explaination=explaination, timestamp=ts, 
                            parent_function_name=parent_info['function_name'], 
                            parent_function_inputs=parent_info['function_inputs'], 
                            module_name=parent_info['module'])

    return path


def _load_resource(project_manifest_path: str, filename: str) -> Any:
    """Internal: Load resource from disk by resource_id, inferring type from ID format."""
    # infer type from the ID. They are always in the format: TIMESTAMP_TYPE_RANDOM.ext

    # get directory location from manifest path after verifying manifest exists
    _check_if_manifest_exists(project_manifest_path)
    data_dir = Path(project_manifest_path).parent

    try:
        type_tag = filename.split("_")[-2]
    except ValueError:
        raise ValueError(f"Invalid resource_id format: {filename}")

    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unknown resource type in id: {filename}")

    path = data_dir / f"{filename}"

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    load_fn: Callable[[Path], Any] = TYPE_REGISTRY[type_tag]["load"]

    return load_fn(path)


def create_project_manifest(path: str, project_name: str) -> dict:
    """Create a new project manifest to track resources in a directory.
    
    Creates a JSON manifest file at <path>/<project_name>_manifest.json to track all
    resources (datasets, models, feature vectors, etc.) created during analysis.
    The manifest records what was created, when, and which operations produced it.
    
    **IMPORTANT**: You should create a manifest BEFORE performing any operations that
    store resources. Ask the user to provide a project directory path and name.
    
    Args:
        path: Directory where manifest and data files will be stored
        project_name: Name for this project (used in manifest filename)
        
    Returns:
        dict with project_name, created_at timestamp, and empty resources list
        
    Raises:
        FileExistsError: If manifest already exists at this location
        
    Example:
        # User says: "I want to analyze some SMILES data"
        # You should ask: "Where would you like to store the project files?"
        # Then create manifest:
        create_project_manifest("/path/to/project", "smiles_analysis")
        
    Note:
        Creates the directory if it doesn't exist. All subsequent operations
        should reference this manifest path to track resources properly.
    """

    project_manifest_path = Path(path) / f"{project_name}_manifest.json"

    # if manifest already exists, return error
    if project_manifest_path.exists():
        raise FileExistsError(f"Project manifest already exists at {project_manifest_path}")
    
    # create data directory if needed
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)

    # create empty manifest structure
    manifest = {
        "project_name": project_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "resources": []
    }

    # save manifest to file
    import json
    with open(project_manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    return manifest

    
def read_project_manifest(project_manifest_path: str) -> dict:
    """Read and return the complete project manifest with all tracked resources.
    
    Returns the full manifest including project metadata and list of all resources
    that have been created and tracked during this project.
    
    Args:
        project_manifest_path: Full path to the project_manifest.json file
        
    Returns:
        dict containing:
            - project_name: Name of the project
            - created_at: When manifest was created
            - resources: List of all tracked resources with metadata
            
    Raises:
        FileNotFoundError: If no manifest exists at this path. Create one first
                          using create_project_manifest().
                          
    Example:
        manifest = read_project_manifest("/path/to/project/myproject_manifest.json")
        print(f"Project: {manifest['project_name']}")
        print(f"Resources tracked: {len(manifest['resources'])}")
    """
    _check_if_manifest_exists(project_manifest_path)
    import json
    with open(project_manifest_path, "r") as f:
        manifest = json.load(f)
    return manifest


def _check_if_manifest_exists(project_manifest_path: str) -> bool:
    """Internal: Check manifest exists, raise helpful error if not."""
    if Path(project_manifest_path).exists():
        return True
    else:
        raise FileNotFoundError(f"Project manifest not found at {project_manifest_path}. Please supply the correct path or create a new project manifest using the create_project_manifest function. The default data directory is located at {DATA_ROOT}")
    

def add_to_project_manifest(project_manifest_path: str, filename: str, type_tag: str, explaination: str | None = 'unknown', timestamp: str | None = None, 
                            parent_function_name: str | None = 'unknown', parent_function_inputs: str | None = 'unknown', module_name: str | None = 'unknown') -> None:
    """Add a new resource entry to the project manifest for tracking.
    
    Records metadata about a newly created resource (dataset, model, features, etc.)
    in the project manifest. This creates an audit trail of what was created and how.
    
    **IMPORTANT**: If no manifest exists, you MUST create one first using
    create_project_manifest(). Ask the user for a project directory and name.
    
    **REQUIRED**: You must provide a brief explanation of what this resource is.
    Examples: "Canonicalized SMILES dataset", "Morgan fingerprints (radius=2)",
    "Trained random forest model", "Tanimoto similarity matrix"
    
    Args:
        project_manifest_path: Full path to the project_manifest.json file
        filename: Name of the resource file being tracked
        type_tag: Resource type from TYPE_REGISTRY (csv, model, json, etc.)
        explaination: Brief 1-sentence description of what this resource contains
        timestamp: Optional timestamp (auto-generated if None)
        parent_function_name: Auto-captured from stack (usually auto-provided)
        parent_function_inputs: Auto-captured from stack (usually auto-provided)
        module_name: Auto-captured from stack (usually auto-provided)
        
    Raises:
        FileNotFoundError: If manifest doesn't exist. Create with create_project_manifest().
        
    Example:
        # After storing cleaned SMILES data
        add_to_project_manifest(
            project_manifest_path="/path/to/project/myproject_manifest.json",
            filename="cleaned_smiles_data",
            type_tag="csv",
            explaination="SMILES dataset after canonicalization and salt removal"
        )
        
    Note:
        The parent function details are usually auto-captured by _store_resource(),
        so you typically only need to provide: manifest_path, filename, type_tag, explaination.
    """

    _check_if_manifest_exists(project_manifest_path)

    # load existing manifest
    import json
    with open(project_manifest_path, "r") as f:
        manifest = json.load(f) 
    
    if timestamp is None:
        timestamp = _get_timestamp()
    
    # add new resource entry
    resource_entry = {
        "filename": filename,
        "type_tag": type_tag,
        "explaination": explaination,
        "timestamp": timestamp,
        "parent_function_name": parent_function_name,
        "parent_function_inputs": parent_function_inputs,
        "module_name": module_name
    }   

    manifest["resources"].append(resource_entry)

    # save updated manifest
    with open(project_manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    

def remove_from_project_manifest(project_manifest_path: str, resource_name: str, delete_file: bool = False) -> dict:
    """Remove a resource from the project manifest tracking (optionally delete file).
    
    Untracks a resource from the manifest. By default, keeps the file on disk but
    stops tracking it. Set delete_file=True to also delete the actual file.
    
    Args:
        project_manifest_path: Full path to the project_manifest.json file
        resource_name: Filename of the resource to untrack
        delete_file: If True, also delete the actual file from disk (default: False)
        
    Returns:
        dict containing the removed resource entry metadata, or None if not found
        
    Raises:
        FileNotFoundError: If manifest doesn't exist. Create with create_project_manifest().
        
    Example:
        # Untrack but keep file
        removed = remove_from_project_manifest(
            "/path/to/project/myproject_manifest.json",
            "old_dataset_csv_ABC123.csv"
        )
        
        # Untrack and delete file
        removed = remove_from_project_manifest(
            "/path/to/project/myproject_manifest.json",
            "temporary_features_csv_XYZ789.csv",
            delete_file=True
        )
    """
    _check_if_manifest_exists(project_manifest_path)
    import json
    with open(project_manifest_path, "r") as f:
        manifest = json.load(f)
    resources = manifest.get("resources", [])
    updated_resources = []
    removed_resource = None 
    for res in resources:
        if res["filename"] == resource_name:
            removed_resource = res
            if delete_file:
                # delete the actual file
                data_dir = Path(project_manifest_path).parent
                type_tag = res["type_tag"]
                rid = f"{type_tag}_{resource_name.split('_')[-1]}"
                file_path = data_dir / rid
                if file_path.exists():
                    file_path.unlink()
        else:
            updated_resources.append(res)
    manifest["resources"] = updated_resources  
    with open(project_manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    return removed_resource


def list_untracked_resources_in_project(project_manifest_path: str) -> list[str]:
    """List files in project directory that aren't tracked in the manifest.
    
    Scans the project directory for files that exist on disk but aren't recorded
    in the manifest. Useful for finding orphaned files or manually added data.
    
    Args:
        project_manifest_path: Full path to the project_manifest.json file
        
    Returns:
        List of filenames present in directory but not in manifest
        
    Raises:
        FileNotFoundError: If manifest doesn't exist. Create with create_project_manifest().
        
    Example:
        untracked = list_untracked_resources_in_project(
            "/path/to/project/myproject_manifest.json"
        )
        if untracked:
            print(f"Found {len(untracked)} untracked files: {untracked}")
    """
    _check_if_manifest_exists(project_manifest_path)
    import json
    with open(project_manifest_path, "r") as f:
        manifest = json.load(f)
    resources = manifest.get("resources", [])
    data_dir = Path(project_manifest_path).parent
    all_files = set(f.name for f in data_dir.iterdir() if f.is_file())
    tracked_files = set(res["filename"] for res in resources)
    untracked_files = list(all_files - tracked_files)
    return untracked_files


def get_all_resources_tools() -> list[Callable]:
    """Return list of all resource management tools for MCP server."""
    return [
        create_project_manifest,
        read_project_manifest,
        add_to_project_manifest,
        remove_from_project_manifest,
        list_untracked_resources_in_project,
        get_supported_resource_types
    ]