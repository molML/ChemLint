
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
    pass

def cv_splits_leavepout():
    pass

def cv_splits_montecarlo():
    pass

def cv_splits_groupbased():
    pass


def _cross_validate_and_eval(model, X, Y, cv_strategy: str, n_folds: int, random_state: int, task_type: str, metric: str) -> float:
    # internal function for cross validation in for hyperparam tuning. Returns the average metric across folds
    
    # create the CV splits

    # perform model training and evaluation for each fold using _train_ml_model() and _eval_single_ml_model()

    # return average metric across folds
    pass



