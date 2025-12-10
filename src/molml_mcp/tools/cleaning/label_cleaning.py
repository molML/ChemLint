from molml_mcp.infrastructure.resources import _load_resource, _store_resource


# function to convert continous values to categorical labels based on thresholds in a dataset
def continuous_to_binary_labels_dataset(
    project_manifest_path: str,
    input_filename: str,
    value_column: str,
    threshold: float,
    label_below: int = 1,
    label_above: int = 0,
    output_filename: str = "labeled_dataset",
    explanation: str = "Dataset with continuous values converted to binary labels (1=below/active, 0=above/inactive).",
) -> dict:
    """Convert continuous values to binary integer labels (0 or 1) based on a threshold.
    
    Args:
        project_manifest_path: Path to the project manifest
        input_filename: Name of the input dataset file
        value_column: Name of the column with continuous values to be labeled
        threshold: Threshold value for binary categorization
        label_below: Integer label for values <= threshold (default: 1)
        label_above: Integer label for values > threshold (default: 0)
        output_filename: Name of the output dataset file (without extension)
        explanation: Explanation for the output resource. Include label definitions for traceability.
    
    Returns:
        dict with output_filename, n_rows, columns, n_below, n_above, label_column, and preview
    
    Examples:
        result = continuous_to_binary_labels(
            project_manifest_path='/path/to/manifest.json',
            input_filename='molecules_with_ic50',
            value_column='IC50_nM',
            threshold=100.0,
            explanation='Binary labels: 1=active (IC50<=100nM), 0=inactive (IC50>100nM)'
        )
    """
    
    # Load the dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in dataset.")
    
    # Apply binary labeling: <= threshold gets label_below, > threshold gets label_above
    df[f"{value_column}_binary"] = df[value_column].apply(
        lambda x: label_below if x <= threshold else label_above
    )
    
    # Count labels
    n_below = (df[value_column] <= threshold).sum()
    n_above = (df[value_column] > threshold).sum()
    
    # Store the new dataset
    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')
    
    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": df.columns.tolist(),
        "threshold": threshold,
        "label_below": label_below,
        "label_above": label_above,
        "n_below": int(n_below),
        "n_above": int(n_above),
        "label_column": f"{value_column}_binary",
        "preview": df.head().to_dict(orient='records'),
    }