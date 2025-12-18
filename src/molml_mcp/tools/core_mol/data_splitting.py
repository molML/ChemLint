
import pandas as pd
import numpy as np
from collections import defaultdict
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


def stratified_split_dataset(
    input_filename: str,
    label_column: str,
    project_manifest_path: str,
    train_output_filename: str,
    test_output_filename: str,
    val_output_filename: str | None = None,
    explanation: str = "Stratified dataset split",
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    val_ratio: float = 0.0,
    n_bins: int = 5,
    random_state: int = 42
) -> dict:
    """
    Split dataset while preserving label distribution across train/test/val sets.
    
    Automatically detects whether the task is classification or regression:
    - Classification: Stratifies by class labels directly (e.g., 'active'/'inactive')
    - Regression: Bins continuous values into quantiles, then stratifies (e.g., IC50, pKi)
    
    This ensures each split has similar label distributions, which is critical for:
    - Imbalanced datasets (prevents train/test distribution mismatch)
    - Small datasets (ensures representative sampling)
    - Fair model evaluation (test set reflects real-world distribution)
    
    Can also stratify by molecular properties (e.g., molecular weight) by using
    that property column as label_column.
    
    Args:
        input_filename: Input CSV filename from manifest
        label_column: Column name to stratify by (labels or properties)
        project_manifest_path: Path to manifest.json
        train_output_filename: Output name for training set
        test_output_filename: Output name for test set
        val_output_filename: Output name for validation set (if val_ratio > 0)
        explanation: Description of the split operation
        train_ratio: Training set fraction (default: 0.8)
        test_ratio: Test set fraction (default: 0.2)
        val_ratio: Validation set fraction (default: 0.0)
        n_bins: Number of bins for regression stratification (default: 5)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        dict with keys:
            - train/test/val_output_filename: Output filenames
            - n_train/test/val_rows: Row counts
            - train/test/val_label_distribution: Label distributions (as percentages)
            - is_regression: Whether regression binning was applied
            - bin_edges: Bin boundaries (for regression only)
            - note: Summary of the split
    
    Examples:
        # Binary classification (80/20 split)
        result = stratified_split_dataset(
            input_filename="data_A1B2C3D4.csv",
            label_column="active",
            project_manifest_path="/path/to/manifest.json",
            train_output_filename="train",
            test_output_filename="test"
        )
        
        # Regression with 3-way split (70/20/10)
        result = stratified_split_dataset(
            input_filename="data_A1B2C3D4.csv",
            label_column="ic50_nm",
            project_manifest_path="/path/to/manifest.json",
            train_output_filename="train",
            test_output_filename="test",
            val_output_filename="val",
            train_ratio=0.7,
            test_ratio=0.2,
            val_ratio=0.1,
            n_bins=5
        )
    """
    from sklearn.model_selection import train_test_split
    
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio + val_ratio:.6f}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns)}")
    if df[label_column].isna().sum() > 0:
        raise ValueError(f"Found {df[label_column].isna().sum()} missing values in '{label_column}'")
    
    labels = df[label_column]
    
    # Auto-detect: classification vs regression
    is_regression = False
    bin_edges = None
    
    if pd.api.types.is_numeric_dtype(labels):
        n_unique = labels.nunique()
        is_integers = (labels % 1 == 0).all()
        
        if is_integers and n_unique < 20:
            # Few unique integers → classification
            stratify = labels.astype(str)
        else:
            # Many values → regression (bin it)
            is_regression = True
            n_bins_adj = max(2, min(n_bins, n_unique // 5))
            try:
                stratify, bin_edges = pd.qcut(labels, q=n_bins_adj, labels=False, retbins=True, duplicates='drop')
            except ValueError:
                stratify, bin_edges = pd.cut(labels, bins=n_bins_adj, labels=False, retbins=True)
    else:
        # Non-numeric → classification
        stratify = labels.astype(str)
    
    # Check minimum samples
    n_splits = 2 if val_ratio == 0 else 3
    if stratify.value_counts().min() < n_splits:
        raise ValueError(f"Smallest class has {stratify.value_counts().min()} samples, need at least {n_splits}")
    
    # Split
    df_train_val, df_test = train_test_split(df, test_size=test_ratio, stratify=stratify, random_state=random_state)
    
    if val_ratio > 0:
        val_size = val_ratio / (train_ratio + val_ratio)
        df_train, df_val = train_test_split(
            df_train_val, 
            test_size=val_size, 
            stratify=stratify.loc[df_train_val.index], 
            random_state=random_state
        )
    else:
        df_train, df_val = df_train_val, pd.DataFrame()
    
    # Get distributions
    def get_dist(split_df):
        if len(split_df) == 0:
            return {}
        if is_regression:
            dist = stratify.loc[split_df.index].value_counts(normalize=True).sort_index()
            return {f"bin_{k}": f"{v*100:.1f}%" for k, v in dist.items()}
        else:
            dist = split_df[label_column].value_counts(normalize=True)
            return {str(k): f"{v*100:.1f}%" for k, v in dist.items()}
    
    # Store
    train_out = _store_resource(df_train, project_manifest_path, train_output_filename, explanation, "csv")
    test_out = _store_resource(df_test, project_manifest_path, test_output_filename, explanation, "csv")
    val_out = _store_resource(df_val, project_manifest_path, val_output_filename, explanation, "csv") if val_ratio > 0 else None
    
    return {
        "train_output_filename": train_out,
        "test_output_filename": test_out,
        "val_output_filename": val_out,
        "n_train_rows": len(df_train),
        "n_test_rows": len(df_test),
        "n_val_rows": len(df_val),
        "train_label_distribution": get_dist(df_train),
        "test_label_distribution": get_dist(df_test),
        "val_label_distribution": get_dist(df_val),
        "is_regression": is_regression,
        "bin_edges": bin_edges.tolist() if bin_edges is not None else None,
        "note": f"Stratified split: {len(df)}→ train:{len(df_train)}, test:{len(df_test)}" + (f", val:{len(df_val)}" if val_ratio > 0 else "")
    }


def scaffold_split_dataset(
    input_filename: str,
    scaffold_column: str,
    project_manifest_path: str,
    train_output_filename: str,
    test_output_filename: str,
    val_output_filename: str | None = None,
    explanation: str = "Scaffold-based dataset split",
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    val_ratio: float = 0.0,
    method: str = 'balanced',
    random_state: int = 42,
    handle_no_scaffold: str = 'assign_to_train'
) -> dict:
    """
    Split dataset by molecular scaffolds to prevent data leakage.
    
    Groups molecules by scaffold and assigns entire scaffold groups to train/test/val
    splits, ensuring molecules with the same scaffold don't appear across splits.
    This prevents data leakage in molecular property prediction tasks.
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    scaffold_column : str
        Name of column containing scaffold identifiers (e.g., 'scaffold_bemis_murcko').
    project_manifest_path : str
        Path to the project manifest JSON file.
    train_output_filename : str
        Base filename for the training set output.
    test_output_filename : str
        Base filename for the test set output.
    val_output_filename : str | None
        Base filename for the validation set output (required if val_ratio > 0).
    explanation : str
        Brief description of this split operation.
    train_ratio : float
        Target proportion for training set (0.0 to 1.0). Default 0.8.
    test_ratio : float
        Target proportion for test set (0.0 to 1.0). Default 0.2.
    val_ratio : float
        Target proportion for validation set (0.0 to 1.0). Default 0.0.
    method : str
        Splitting method: 'random' or 'balanced'. Default 'balanced'.
        - 'random': Shuffle scaffolds randomly, assign sequentially
        - 'balanced': Greedy bin packing to balance split sizes
    random_state : int
        Random seed for reproducibility. Default 42.
    handle_no_scaffold : str
        How to handle molecules with no scaffold: 'assign_to_train' or 'random'.
        Default 'assign_to_train'.
    
    Returns
    -------
    dict
        Contains output filenames, row counts, scaffold counts, actual ratios,
        overlap check status, and statistics for each split.
    
    Examples
    --------
    Balanced scaffold split (80/20):
    
        result = scaffold_split_dataset(
            input_filename='molecules_AB12CD34.csv',
            scaffold_column='scaffold_bemis_murcko',
            project_manifest_path='/path/to/manifest.json',
            train_output_filename='train',
            test_output_filename='test',
            method='balanced'
        )
        
        print(f"Train: {result['n_train_rows']} molecules, {result['n_train_scaffolds']} scaffolds")
        print(f"Test: {result['n_test_rows']} molecules, {result['n_test_scaffolds']} scaffolds")
    
    Three-way split with validation set:
    
        result = scaffold_split_dataset(
            input_filename='molecules_AB12CD34.csv',
            scaffold_column='scaffold_bemis_murcko',
            project_manifest_path='/path/to/manifest.json',
            train_output_filename='train',
            test_output_filename='test',
            val_output_filename='val',
            train_ratio=0.7,
            test_ratio=0.2,
            val_ratio=0.1,
            method='balanced'
        )
    
    Random scaffold split:
    
        result = scaffold_split_dataset(
            input_filename='molecules_AB12CD34.csv',
            scaffold_column='scaffold_bemis_murcko',
            project_manifest_path='/path/to/manifest.json',
            train_output_filename='train',
            test_output_filename='test',
            method='random',
            random_state=42
        )
    """
    # Validate inputs
    if method not in ['random', 'balanced']:
        raise ValueError(f"method must be 'random' or 'balanced', got '{method}'")
    
    if handle_no_scaffold not in ['assign_to_train', 'random']:
        raise ValueError(f"handle_no_scaffold must be 'assign_to_train' or 'random', got '{handle_no_scaffold}'")
    
    # Validate ratios
    total_ratio = train_ratio + test_ratio + val_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.6f}")
    
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    
    if val_ratio > 0 and val_output_filename is None:
        raise ValueError("val_output_filename is required when val_ratio > 0")
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_total = len(df)
    
    # Validate scaffold column exists
    if scaffold_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"Scaffold column '{scaffold_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    no_scaffold_indices = []
    
    for idx, scaffold in enumerate(df[scaffold_column]):
        if pd.isna(scaffold) or scaffold == '' or scaffold == 'No scaffold' or scaffold is None:
            no_scaffold_indices.append(idx)
        else:
            scaffold_to_indices[scaffold].append(idx)
    
    n_no_scaffold = len(no_scaffold_indices)
    n_unique_scaffolds = len(scaffold_to_indices)
    
    # Prepare scaffold groups (scaffold -> list of indices)
    scaffolds = list(scaffold_to_indices.keys())
    scaffold_sizes = {s: len(indices) for s, indices in scaffold_to_indices.items()}
    
    # Set random seed
    np.random.seed(random_state)
    
    # Initialize splits
    train_indices = []
    test_indices = []
    val_indices = []
    
    if method == 'random':
        # Random method: shuffle scaffolds and assign sequentially
        np.random.shuffle(scaffolds)
        
        # Calculate target counts
        n_with_scaffold = sum(scaffold_sizes.values())
        train_target = int(n_with_scaffold * train_ratio)
        test_target = int(n_with_scaffold * test_ratio)
        
        current_train = 0
        current_test = 0
        current_val = 0
        
        for scaffold in scaffolds:
            indices = scaffold_to_indices[scaffold]
            size = len(indices)
            
            # Assign to split with most deficit
            if current_train < train_target:
                train_indices.extend(indices)
                current_train += size
            elif current_test < test_target:
                test_indices.extend(indices)
                current_test += size
            else:
                val_indices.extend(indices)
                current_val += size
    
    elif method == 'balanced':
        # Balanced method: greedy bin packing (largest scaffolds first)
        sorted_scaffolds = sorted(scaffolds, key=lambda s: scaffold_sizes[s], reverse=True)
        
        # Track current sizes of each split
        split_sizes = {'train': 0, 'test': 0, 'val': 0}
        split_indices = {'train': train_indices, 'test': test_indices, 'val': val_indices}
        split_ratios = {'train': train_ratio, 'test': test_ratio, 'val': val_ratio}
        
        # Remove val from consideration if not used
        if val_ratio == 0:
            del split_sizes['val']
            del split_indices['val']
            del split_ratios['val']
        
        for scaffold in sorted_scaffolds:
            indices = scaffold_to_indices[scaffold]
            size = len(indices)
            
            # Find split with largest deficit (furthest below target ratio)
            best_split = None
            best_deficit = -float('inf')
            
            for split_name in split_sizes.keys():
                current_ratio = split_sizes[split_name] / max(sum(split_sizes.values()), 1)
                target_ratio = split_ratios[split_name]
                deficit = target_ratio - current_ratio
                
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_split = split_name
            
            # Assign scaffold to best split
            split_indices[best_split].extend(indices)
            split_sizes[best_split] += size
    
    # Handle molecules with no scaffold
    if n_no_scaffold > 0:
        if handle_no_scaffold == 'assign_to_train':
            train_indices.extend(no_scaffold_indices)
        else:  # 'random'
            # Distribute randomly according to ratios
            if val_ratio > 0:
                train_no_scaffold, temp = train_test_split(
                    no_scaffold_indices, 
                    train_size=train_ratio, 
                    random_state=random_state
                )
                test_frac = test_ratio / (test_ratio + val_ratio)
                test_no_scaffold, val_no_scaffold = train_test_split(
                    temp, 
                    train_size=test_frac, 
                    random_state=random_state
                )
                train_indices.extend(train_no_scaffold)
                test_indices.extend(test_no_scaffold)
                val_indices.extend(val_no_scaffold)
            else:
                train_no_scaffold, test_no_scaffold = train_test_split(
                    no_scaffold_indices,
                    train_size=train_ratio,
                    random_state=random_state
                )
                train_indices.extend(train_no_scaffold)
                test_indices.extend(test_no_scaffold)
    
    # Create split dataframes
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    df_val = df.iloc[val_indices].copy() if val_ratio > 0 else pd.DataFrame()
    
    # Count unique scaffolds in each split (excluding NaN)
    def count_unique_scaffolds(split_df):
        scaffolds_in_split = split_df[scaffold_column]
        valid_scaffolds = scaffolds_in_split[
            ~scaffolds_in_split.isna() & 
            (scaffolds_in_split != '') & 
            (scaffolds_in_split != 'No scaffold')
        ]
        return len(set(valid_scaffolds))
    
    n_train_scaffolds = count_unique_scaffolds(df_train)
    n_test_scaffolds = count_unique_scaffolds(df_test)
    n_val_scaffolds = count_unique_scaffolds(df_val) if val_ratio > 0 else 0
    
    # Verify no scaffold overlap (data leakage check)
    train_scaffolds = set(df_train[scaffold_column].dropna())
    test_scaffolds = set(df_test[scaffold_column].dropna())
    val_scaffolds = set(df_val[scaffold_column].dropna()) if val_ratio > 0 else set()
    
    # Remove empty/invalid scaffolds from sets
    train_scaffolds.discard('')
    train_scaffolds.discard('No scaffold')
    test_scaffolds.discard('')
    test_scaffolds.discard('No scaffold')
    val_scaffolds.discard('')
    val_scaffolds.discard('No scaffold')
    
    train_test_overlap = train_scaffolds & test_scaffolds
    train_val_overlap = train_scaffolds & val_scaffolds
    test_val_overlap = test_scaffolds & val_scaffolds
    
    has_overlap = len(train_test_overlap) > 0 or len(train_val_overlap) > 0 or len(test_val_overlap) > 0
    
    overlap_info = {
        'train_test_overlap': len(train_test_overlap),
        'train_val_overlap': len(train_val_overlap),
        'test_val_overlap': len(test_val_overlap)
    }
    
    # Store outputs
    train_output = _store_resource(df_train, project_manifest_path, train_output_filename, explanation, "csv")
    test_output = _store_resource(df_test, project_manifest_path, test_output_filename, explanation, "csv")
    val_output = None
    if val_ratio > 0:
        val_output = _store_resource(df_val, project_manifest_path, val_output_filename, explanation, "csv")
    
    # Calculate actual ratios
    actual_train_ratio = len(df_train) / n_total if n_total > 0 else 0.0
    actual_test_ratio = len(df_test) / n_total if n_total > 0 else 0.0
    actual_val_ratio = len(df_val) / n_total if n_total > 0 and val_ratio > 0 else 0.0
    
    result = {
        "train_output_filename": train_output,
        "n_train_rows": len(df_train),
        "n_train_scaffolds": n_train_scaffolds,
        "actual_train_ratio": actual_train_ratio,
        
        "test_output_filename": test_output,
        "n_test_rows": len(df_test),
        "n_test_scaffolds": n_test_scaffolds,
        "actual_test_ratio": actual_test_ratio,
        
        "val_output_filename": val_output,
        "n_val_rows": len(df_val) if val_ratio > 0 else 0,
        "n_val_scaffolds": n_val_scaffolds,
        "actual_val_ratio": actual_val_ratio,
        
        "n_total_rows": n_total,
        "n_unique_scaffolds": n_unique_scaffolds,
        "n_molecules_no_scaffold": n_no_scaffold,
        
        "method": method,
        "handle_no_scaffold": handle_no_scaffold,
        "random_state": random_state,
        
        "overlap_detected": has_overlap,
        "overlap_info": overlap_info,
        
        "target_ratios": {
            "train": train_ratio,
            "test": test_ratio,
            "val": val_ratio
        },
        
        "warning": "⚠️ Scaffold-based splitting may produce unbalanced splits when scaffold sizes vary greatly. Actual ratios may differ from target ratios." if has_overlap else None,
        
        "note": (
            f"Scaffold split ({method} method): {n_total} molecules → "
            f"Train: {len(df_train)} ({actual_train_ratio:.1%}), "
            f"Test: {len(df_test)} ({actual_test_ratio:.1%})"
            f"{f', Val: {len(df_val)} ({actual_val_ratio:.1%})' if val_ratio > 0 else ''}. "
            f"Total scaffolds: {n_unique_scaffolds}, No scaffold: {n_no_scaffold}. "
            f"{'✓ No scaffold overlap detected.' if not has_overlap else '⚠ SCAFFOLD OVERLAP DETECTED!'}"
        )
    }
    
    return result






