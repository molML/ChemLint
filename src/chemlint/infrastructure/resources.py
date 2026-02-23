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

from chemlint.infrastructure.supported_resource_types import TYPE_REGISTRY
from chemlint.config import DATA_ROOT


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
    """Generate unique resource ID suffix: _{8_hex_chars}{extension}."""
    rand = secrets.token_hex(4).upper()  # 8 hex chars
    ext = TYPE_REGISTRY[type_tag]["ext"]
    
    return f"_{rand}{ext}"


def _store_resource(obj: Any, project_manifest_path: str, filename: str, explaination: str, type_tag: str) -> str:
    """Internal: Store object to disk with unique ID, auto-track in manifest with metadata.
    
    Stores an object (DataFrame, model, JSON, etc.) to disk with a unique resource ID
    to prevent overwriting. The filename pattern is: {filename}_{8_hex_chars}{ext}
    Example: "cleaned_data_A3F2B1D4.csv"
    
    Automatically captures parent function metadata (name, inputs, module) and adds an
    entry to the project manifest for full audit trail tracking. The type information
    is stored in the manifest, not in the filename.
    
    Args:
        obj: Object to store (DataFrame, model, dict, etc.)
        project_manifest_path: Full path to project_manifest.json file
        filename: Base filename without extension (e.g., "cleaned_data")
        explaination: Brief description of what this resource contains
        type_tag: Resource type from TYPE_REGISTRY (csv, model, json, etc.)
        
    Returns:
        Full path to the stored file (Path object)
        
    Raises:
        ValueError: If type_tag is not supported (check TYPE_REGISTRY)
        FileNotFoundError: If project manifest doesn't exist at specified path
        
    Example:
        path = _store_resource(
            obj=dataframe,
            project_manifest_path="/path/to/project/manifest.json",
            filename="cleaned_smiles",
            explaination="SMILES after canonicalization and salt removal",
            type_tag="csv"
        )
        # Stores as: /path/to/project/cleaned_smiles_A3F2B1D4.csv
        # Returns: Path('/path/to/project/cleaned_smiles_A3F2B1D4.csv')
        
    Note:
        The unique ID prevents accidental overwrites. The type is stored in the manifest,
        not the filename, enforcing that resources must be loaded through the manifest.
    """
    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unsupported resource type: {type_tag}")
    
    # Generate unique resource ID to prevent overwrites if same filename is used again
    # Format: _{8_hex_chars}{ext} (e.g., "_A3F2B1D4.csv")
    # Type is NOT in filename - it's stored in manifest for traceability
    rid = _generate_id(type_tag)

    # Get data directory from manifest path (same directory as manifest)
    data_dir = Path(project_manifest_path).parent
    # Combine user's filename with unique resource ID
    output_filename = f"{filename}{rid}"
    path = data_dir / output_filename

    # Get the appropriate save function for this type and save the object
    save_fn: Callable[[Any, Path], None] = TYPE_REGISTRY[type_tag]['save']
    save_fn(obj, path)
    
    # Generate timestamp for manifest entry
    ts = _get_timestamp()

    # Capture parent function metadata (name, inputs, module) for audit trail
    parent_info = _get_parent_info()

    # Add entry to project manifest with all metadata
    add_to_project_manifest(
        project_manifest_path=project_manifest_path, 
        filename=output_filename, 
        type_tag=type_tag, 
        explaination=explaination, 
        timestamp=ts, 
        parent_function_name=parent_info['function_name'], 
        parent_function_inputs=parent_info['function_inputs'], 
        module_name=parent_info['module']
    )

    return output_filename


def _load_resource(project_manifest_path: str, filename: str) -> Any:
    """Internal: Load resource from disk by looking up type in manifest.
    
    Loads a resource by finding its entry in the project manifest to determine the
    correct type and file extension. The manifest is the source of truth for all
    tracked resources. This enforces traceability by requiring all resources to be
    tracked in the manifest before loading.
    
    Args:
        project_manifest_path: Full path to the project_manifest.json file
        filename: Base filename as stored in manifest (e.g., "cleaned_data")
                 DO NOT include the unique ID or extension.
        
    Returns:
        The loaded object (DataFrame, dict, model, etc.)
        
    Raises:
        FileNotFoundError: If manifest doesn't exist or resource file not found on disk
        ValueError: If resource is not tracked in manifest (must be added via _store_resource)
        
    Example:
        # Load by base filename only
        df = _load_resource(
            project_manifest_path="/path/to/project/manifest.json",
            filename="cleaned_data"
        )
        
    Note:
        The resource MUST be tracked in the manifest. You cannot load untracked files.
        This enforces proper resource management and traceability.
    """
    # Verify manifest exists
    _check_if_manifest_exists(project_manifest_path)
    
    # Load manifest to look up resource metadata
    import json
    with open(project_manifest_path, "r") as f:
        manifest = json.load(f)
    
    resources = manifest.get("resources", [])
    
    # Search for the resource in manifest by exact filename match only
    resource_entry = None
    for res in resources:
        if res["filename"] == filename:
            resource_entry = res
            break
    
    if resource_entry is None:
        raise ValueError(
            f"Resource '{filename}' not found in manifest at {project_manifest_path}. "
            f"Available resources: {[r['filename'] for r in resources]}. "
            f"Use add_to_project_manifest() to track this resource first."
        )
    
    # Extract type and extension from manifest
    type_tag = resource_entry["type_tag"]
    
    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unknown resource type '{type_tag}' in manifest for resource '{filename}'")
    
    # Get the data directory (same as manifest directory)
    data_dir = Path(project_manifest_path).parent
    
    # The filename in the manifest is the complete filename with unique ID
    # (e.g., "cleaned_data_A3F2B1D4.csv")
    full_filename = resource_entry["filename"]
    path = data_dir / full_filename
    
    # Verify the file exists on disk
    if not path.exists():
        raise FileNotFoundError(
            f"Resource file '{full_filename}' not found at expected location: {path}"
        )
    load_fn: Callable[[Path], Any] = TYPE_REGISTRY[type_tag]["load"]
    
    return load_fn(path)


def create_project_manifest(path: str, project_name: str) -> dict:
    """Create a new project manifest to track resources in a directory.
    
    Creates a JSON manifest file at <path>/<project_name>_manifest.json to track all
    resources (datasets, models, feature vectors, etc.) created during analysis.
    The manifest records what was created, when, and which operations produced it.

    The default data directory can be found using check_default_data_dir().
    
    **IMPORTANT**: You should create a manifest BEFORE performing any operations that
    store resources. Ask the user to provide a project directory path and name.

    **CRITICAL**: Interact with files through the manifest system. Do NOT attempt to access files 
    directly by path after importing or creating them. Always use the manifest to load resources, 
    as it is the source of truth for all tracked files and their metadata.
    
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


def check_default_data_dir() -> str:
    """Get the default data directory path for storing project manifests and resources.
    
    Returns the configured default data directory where project manifests and their
    associated resources are stored when no explicit path is provided. This directory
    can be customized via the CHEMLINT_DATA_DIR environment variable.
    
    Returns:
        str: Absolute path to the default data directory
        
    Example:
        >>> dir_path = check_default_data_dir()
        >>> print(f"Default data directory: {dir_path}")
        Default data directory: /Users/username/.chemlint
        
    Note:
        The default is ~/.chemlint unless overridden by CHEMLINT_DATA_DIR environment variable.
    """
    return str(DATA_ROOT)


def check_data_directory_content(data_directory_path: str) -> dict[str, Any]:
    """Check contents of a data directory and organize files by their project manifest.
    
    Scans a directory for manifest files and lists all tracked resources belonging to
    each project. Files not tracked in any manifest are not listed.
    
    Parameters
    ----------
    data_directory_path : str
        Path to directory to inspect
        
    Returns
    -------
    dict
        Contains:
        - n_projects: Number of project manifests found
        - projects: List of dicts with 'manifest_path', 'project_name', 'n_resources', and 'resources'
        - message: Summary message
    """
    import json
    from pathlib import Path
    
    data_dir = Path(data_directory_path)
    
    # Check if directory exists
    if not data_dir.exists():
        return {
            "n_projects": 0,
            "projects": [],
            "message": f"Directory does not exist: {data_directory_path}"
        }
    
    if not data_dir.is_dir():
        return {
            "n_projects": 0,
            "projects": [],
            "message": f"Path is not a directory: {data_directory_path}"
        }
    
    # Find all manifest files
    manifest_files = list(data_dir.glob("*_manifest.json"))
    
    if len(manifest_files) == 0:
        return {
            "n_projects": 0,
            "projects": [],
            "message": f"No project manifests found in {data_directory_path}"
        }
    
    # Process each manifest
    projects = []
    for manifest_path in manifest_files:
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            project_name = manifest.get("project_name", "Unknown")
            resources = manifest.get("resources", [])
            
            # List resource filenames
            resource_list = [res["filename"] for res in resources]
            
            projects.append({
                "manifest_path": str(manifest_path),
                "project_name": project_name,
                "n_resources": len(resources),
                "resources": resource_list
            })
        except Exception as e:
            # If manifest is corrupted, note it
            projects.append({
                "manifest_path": str(manifest_path),
                "project_name": "Error reading manifest",
                "n_resources": 0,
                "resources": [],
                "error": str(e)
            })
    
    # Generate summary message
    if len(projects) == 1:
        message = f"Found 1 project with {projects[0]['n_resources']} tracked resources"
    else:
        total_resources = sum(p['n_resources'] for p in projects)
        message = f"Found {len(projects)} projects with {total_resources} total tracked resources"
    
    return {
        "n_projects": len(projects),
        "projects": projects,
        "message": message
    }


def get_all_resources_tools() -> list[Callable]:
    """Return list of all resource management tools for MCP server."""
    return [
        create_project_manifest,
        read_project_manifest,
        add_to_project_manifest,
        remove_from_project_manifest,
        list_untracked_resources_in_project,
        get_supported_resource_types,
        check_default_data_dir,
        check_data_directory_content
    ]