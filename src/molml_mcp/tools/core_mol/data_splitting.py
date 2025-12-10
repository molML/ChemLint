
from sklearn.model_selection import train_test_split
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def random_split_dataset(
    project_manifest_path: str,
    input_filename: str,
    train_df_output_filename: str,
    test_df_output_filename: str,
    val_df_output_filename: str | None = None,
    explanation: str = "Random train/test/val split",
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
) -> dict:
    """
    Randomly split a dataset into train, test, and optionally validation sets.
    
    Parameters
    ----------
    project_manifest_path : str
        Path to the project manifest JSON file.
    input_filename : str
        Base filename of the input dataset resource.
    train_df_output_filename : str
        Base filename for the training set output.
    test_df_output_filename : str
        Base filename for the test set output.
    val_df_output_filename : str | None
        Base filename for the validation set output (required if val_size > 0).
    explanation : str
        Brief description of this split operation.
    test_size : float
        Proportion of dataset for test set (0.0 to 1.0). Default 0.2 (20%).
    val_size : float
        Proportion of dataset for validation set (0.0 to 1.0). Default 0.0 (none).
    random_state : int
        Random seed for reproducibility. Default 42.
    
    Returns
    -------
    dict
        {
            "train_df_output_filename": str,
            "n_train_rows": int,
            "test_df_output_filename": str,
            "n_test_rows": int,
            "val_df_output_filename": str | None,
            "n_val_rows": int
        }
    """
    import pandas as pd
    
    # Validate split sizes
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if not (0.0 <= val_size < 1.0):
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")
    if val_size > 0 and val_df_output_filename is None:
        raise ValueError("val_df_output_filename is required when val_size > 0")
    
    df = _load_resource(project_manifest_path, input_filename)

    df_train_val, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    if val_size > 0:
        val_relative_size = val_size / (1 - test_size)
        df_train, df_val = train_test_split(df_train_val, test_size=val_relative_size, random_state=random_state)
    else:
        df_train = df_train_val
        df_val = pd.DataFrame()  # empty dataframe
       
    train_df_output_filename = _store_resource(df_train, project_manifest_path, train_df_output_filename, explanation, "csv")
    test_df_output_filename = _store_resource(df_test, project_manifest_path, test_df_output_filename, explanation, "csv")
    val_df_output_filename = None
    if val_size > 0:
        val_df_output_filename = _store_resource(df_val, project_manifest_path, val_df_output_filename, explanation, "csv")     

    result = {
        "train_df_output_filename": train_df_output_filename,
        "n_train_rows": len(df_train),
        "test_df_output_filename": test_df_output_filename,
        "n_test_rows": len(df_test),
        "val_df_output_filename": val_df_output_filename,
        "n_val_rows": len(df_val) if val_size > 0 else 0,
    }  

    return result






