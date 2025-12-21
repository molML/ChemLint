
# k-fold (random)
# stratified k-fold
# leave-p-out
# monte carlo (random train/val splits)
# group-based splitting (scaffold or cluster)


def cv_splits_kfold(k: int, smiles: list, val_size: float, random_state: int, shuffle: bool = True) -> list[dict]:
    """
    Split data into k folds for cross-validation.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    """
    from sklearn.model_selection import KFold
    import numpy as np
    
    smiles_array = np.array(smiles)
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    splits = []
    for train_idx, val_idx in kf.split(smiles_array):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits

def cv_splits_stratifiedkfold(k: int, smiles: list, y: list, val_size: float, random_state: int, shuffle: bool = True) -> list[dict]:
    """
    Split data into k stratified folds for cross-validation.
    
    Stratified splitting ensures each fold maintains the same class distribution as the original dataset.
    Important for imbalanced classification problems.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        y: List of labels (for stratification)
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    """
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    smiles_array = np.array(smiles)
    y_array = np.array(y)
    
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    splits = []
    for train_idx, val_idx in skf.split(smiles_array, y_array):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits

def cv_splits_leavepout(p: int, smiles: list, val_size: float, random_state: int, max_splits: int = None) -> list[dict]:
    """
    Split data using Leave-P-Out cross-validation.
    
    In Leave-P-Out CV, p samples are left out for validation in each fold, and the model is trained
    on the remaining samples. This generates C(n, p) splits where n is the total number of samples.
    
    WARNING: Can generate a very large number of splits! For n=100 and p=2, this creates 4,950 splits.
    Use max_splits to limit the number of splits for computational efficiency.
    
    Args:
        p: Number of samples to leave out in each fold
        smiles: List of SMILES strings
        val_size: Fraction of data to use for validation (not used, kept for API consistency)
        random_state: Random seed for reproducibility (used if max_splits is set)
        max_splits: Maximum number of splits to generate (if None, generates all possible splits)
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    """
    from sklearn.model_selection import LeavePOut
    import numpy as np
    
    smiles_array = np.array(smiles)
    lpo = LeavePOut(p=p)
    
    splits = []
    for i, (train_idx, val_idx) in enumerate(lpo.split(smiles_array)):
        if max_splits is not None and i >= max_splits:
            break
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    # If max_splits is set and we want random sampling, shuffle the splits
    if max_splits is not None and random_state is not None and len(splits) > max_splits:
        np.random.seed(random_state)
        indices = np.random.choice(len(splits), max_splits, replace=False)
        splits = [splits[i] for i in sorted(indices)]
    
    return splits

def cv_splits_montecarlo(n_splits: int, smiles: list, val_size: float, random_state: int) -> list[dict]:
    """
    Split data using Monte Carlo cross-validation (repeated random sub-sampling).
    
    Monte Carlo CV randomly splits the data into training and validation sets n_splits times.
    Unlike k-fold, samples may appear in validation multiple times or not at all across splits.
    Useful when you want to control the exact validation size and don't need exhaustive coverage.
    
    Args:
        n_splits: Number of random train/val splits to generate
        smiles: List of SMILES strings
        val_size: Fraction of data to use for validation (e.g., 0.2 for 20%)
        random_state: Random seed for reproducibility
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    """
    from sklearn.model_selection import ShuffleSplit
    import numpy as np
    
    smiles_array = np.array(smiles)
    ss = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in ss.split(smiles_array):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits

def cv_splits_groupbased():
    pass


def _cross_validate_and_eval(model, X, Y, cv_strategy: str, n_folds: int, random_state: int, task_type: str, metric: str) -> float:
    # internal function for cross validation in for hyperparam tuning. Returns the average metric across folds
    
    # create the CV splits

    # perform model training and evaluation for each fold using _train_ml_model() and _eval_single_ml_model()

    # return average metric across folds
    pass



