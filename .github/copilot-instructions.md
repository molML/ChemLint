# ChemLint - AI Coding Agent Instructions

## Project Overview
This is an **MCP (Model Context Protocol) server** that enables LLMs to perform molecular machine learning tasks. Built with FastMCP, it exposes 150+ tools for molecular structure manipulation, dataset operations, and ML workflows.

## Architecture

### Core Components
- **`server.py`**: FastMCP server registration point - tools added via `mcp.add_tool()`
- **`tools/`**: 150+ domain-organized tool modules organized in subdirectories:
  - `cleaning/`: SMILES standardization, deduplication
  - `core/`: Dataset operations, statistics, filtering, outliers, dimensionality reduction
  - `core_mol/`: Scaffolds, similarity, complexity, activity cliffs, data splitting
  - `featurization/`: Descriptors (simple, complex), SMILES encoding
  - `ml/`: Training, evaluation, cross-validation, hyperparameter tuning
  - `plotting/`: Interactive visualizations with Dash
  - `reports/`: Quality reports, scaffold analysis, split quality
  - `clustering/`: DBSCAN, hierarchical, k-means, Butina
- **`infrastructure/`**: Resource management and type registry
  - `resources.py`: Manifest-based resource tracking
  - `supported_resource_types.py`: Handlers for CSV, models, JSON, PNG

### Resource Management System
The project uses a **manifest-based resource tracking system** for stateful operations:

- **ID Format**: `{filename}_{8_HEX_ID}.{ext}` (e.g., `cleaned_data_A3F2B1D4.csv`)
  - User provides meaningful `filename` prefix
  - System appends unique 8-character hex ID
  - Extension determined by resource type (csv, pkl, json, png)

- **Manifest Tracking**: Each project has a `manifest.json` that tracks all resources
  - Contains: filename, type, created_at timestamp, created_by function, explanation
  - Enables resource lookup by filename (no filesystem globbing)
  - Provides audit trail of all operations

- **Storage**: Resources stored in project directory (or `~/.molml_mcp/` default)
  - User specifies `project_manifest_path` pointing to manifest.json
  - Resources stored in same directory as manifest

- **Registry**: `TYPE_REGISTRY` in `supported_resource_types.py` defines save/load handlers
  - `csv`: pandas DataFrames (saved with `to_csv`)
  - `model`: scikit-learn models (saved with joblib)
  - `json`: dictionaries/lists (saved with json.dump)
  - `png`: matplotlib figures (saved with savefig)

- **Core Functions**:
  ```python
  # Store: creates resource with unique ID and registers in manifest
  _store_resource(
      obj,                      # Object to store (DataFrame, model, dict, etc.)
      project_manifest_path,    # Path to manifest.json
      output_filename,          # User-provided filename prefix
      explanation,              # Human-readable description
      resource_type            # Type: 'csv', 'model', 'json', 'png'
  ) -> str  # Returns: "output_filename_A3F2B1D4.ext"
  
  # Load: retrieves resource by filename from manifest
  _load_resource(
      project_manifest_path,    # Path to manifest.json
      filename                  # Full filename with unique ID
  ) -> Any  # Returns: Original object (DataFrame, model, dict, etc.)
  ```

- **Tool Pattern**: Tools accept `input_filename` and return `output_filename`
  ```python
  def process_dataset(
      input_filename: str,           # Input resource (e.g., "raw_data_12345678.csv")
      project_manifest_path: str,    # Path to manifest.json
      output_filename: str,          # Output name (e.g., "processed_data")
      explanation: str,              # Description of operation
      # ... other params
  ) -> dict:
      # Load input
      df = _load_resource(project_manifest_path, input_filename)
      
      # Process data
      df_processed = df.copy()
      # ... processing logic
      
      # Store output - returns "processed_data_A3F2B1D4.csv"
      output_id = _store_resource(
          df_processed, 
          project_manifest_path, 
          output_filename, 
          explanation, 
          'csv'
      )
      
      return {
          "output_filename": output_id,  # Full filename with unique ID
          "n_rows": len(df_processed),
          # ... other metadata
      }
  ```

### Tool Organization
Tools follow a **domain-based namespace pattern** with `get_all_*_tools()` functions for batch registration:
- Each tool subdirectory exports a `get_all_*_tools()` function
- Tools are registered in `server.py` using loops: `for tool_func in get_all_*_tools(): mcp.add_tool(tool_func)`
- Some critical tools are registered individually for clarity

## Key Patterns

### 1. MCP Tool Registration
Tools must be explicitly added to FastMCP in `server.py`:
```python
from molml_mcp.tools.cleaning import get_all_cleaning_tools
for tool_func in get_all_cleaning_tools():
    mcp.add_tool(tool_func)
```

### 2. Dataset Tool Return Pattern
Dataset manipulation tools follow a consistent return schema:
```python
return {
    "output_filename": str,    # Full filename with unique ID
    "n_rows": int,             # Row count
    "columns": list[str],      # Column names
    "preview": list[dict],     # First 5 rows as records
}
```

### 3. Immutable Operations for Traceability
All dataset mutations create **new resources** rather than modifying existing ones:
- Every operation creates a new file with a unique ID
- Original data is never modified, ensuring full traceability
- The manifest tracks the complete lineage of transformations
- This enables reproduction of the entire workflow and rollback to any state

## Development Workflow

### Adding New Tools
1. Create function in appropriate `tools/` subdirectory
2. Export from subdirectory's `__init__.py` via `get_all_*_tools()` function
3. Ensure tool is registered in `server.py` (either individually or via the get_all loop)
4. Follow manifest-based pattern if stateful (input_filename → output_filename)

### Adding New Resource Types
Edit `supported_resource_types.py`:
```python
TYPE_REGISTRY["newtype"] = {
    "ext": ".extension",
    "save": _save_newtype,  # (obj, path: Path) -> None
    "load": _load_newtype,  # (path: Path) -> Any
}
```

## Dependencies & Tech Stack
- **FastMCP**: MCP server framework
- **RDKit**: Molecular structure manipulation (SMILES, molecule objects)
- **pandas**: Dataset operations
- **scikit-learn**: ML models
- **Dash + Plotly**: Interactive visualizations
- **joblib**: Model serialization
- Python 3.13+ required
