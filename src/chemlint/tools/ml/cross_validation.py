
# k-fold (random)
# stratified k-fold
# leave-p-out
# monte carlo (random train/val splits)
# group-based splitting (scaffold or cluster)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from chemlint.tools.core_mol.scaffolds import _get_scaffold
from chemlint.tools.ml.training import _train_ml_model
from chemlint.tools.ml.evaluation import _eval_single_ml_model


# ============================================================================
# CV Strategy Registry
# ============================================================================

CV_STRATEGY_REGISTRY = {}

def register_cv_strategy(name: str):
    """Decorator to register a CV strategy function."""
    def decorator(func):
        CV_STRATEGY_REGISTRY[name] = func
        return func
    return decorator

def get_cv_splits(
    strategy: str,
    smiles: list,
    n_folds: int,
    random_state: int,
    labels: list = None,
    clusters: list = None,
    val_size: float = None,
    scaffolds: list = None,
    shuffle: bool = True,
    p: int = 1,
    max_splits: int = None
) -> list[dict]:
    """
    Generic function to get CV splits for any registered strategy.
    
    Args:
        strategy: CV strategy name ('kfold', 'stratified', 'montecarlo', 'scaffold', 'cluster', 'leavepout')
        smiles: List of SMILES strings
        n_folds: Number of folds/splits
        random_state: Random seed
        labels: List of labels (required for 'stratified')
        clusters: List of cluster assignments (required for 'cluster')
        val_size: Validation size fraction (used for 'montecarlo', optional for others)
        scaffolds: List of scaffold SMILES (required for 'scaffold' if pre-computed)
        shuffle: Whether to shuffle (used by most strategies)
        p: Number of samples to leave out (for 'leavepout')
        max_splits: Maximum splits for 'leavepout'
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
    """
    if strategy not in CV_STRATEGY_REGISTRY:
        available = ', '.join(CV_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown CV strategy: {strategy}. Available: {available}")
    
    # Build kwargs dict based on strategy requirements
    kwargs = {
        'k': n_folds,
        'smiles': smiles,
        'val_size': val_size if val_size is not None else 1.0 / n_folds,
        'random_state': random_state,
        'shuffle': shuffle
    }
    
    # Add strategy-specific parameters
    if strategy == 'stratified':
        if labels is None:
            raise ValueError("Stratified CV requires 'labels' parameter")
        kwargs['y'] = labels
    
    elif strategy == 'cluster':
        if clusters is None:
            raise ValueError("Cluster-based CV requires 'clusters' parameter")
        kwargs['clusters'] = clusters
    
    elif strategy == 'scaffold':
        if scaffolds is None:
            raise ValueError("Scaffold-based CV requires 'scaffolds' parameter (list of scaffold SMILES)")
        kwargs['scaffolds'] = scaffolds
    
    elif strategy == 'montecarlo':
        kwargs = {
            'n_splits': n_folds,
            'smiles': smiles,
            'val_size': val_size if val_size is not None else 0.2,
            'random_state': random_state
        }
    
    elif strategy == 'leavepout':
        kwargs = {
            'p': p,
            'smiles': smiles,
            'val_size': 1.0 / len(smiles),
            'random_state': random_state,
            'max_splits': max_splits if max_splits is not None else n_folds
        }
    
    return CV_STRATEGY_REGISTRY[strategy](**kwargs)


# ============================================================================
# CV Strategy Implementations
# ============================================================================

@register_cv_strategy('kfold')
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


def _bin_continuous_labels(y_array: np.ndarray, k: int) -> np.ndarray:
    """
    Bin continuous regression labels for stratification.
    
    Converts continuous values into discrete bins to enable stratified splitting.
    Uses quantile-based binning (qcut) for equal-sized bins, with fallback to
    equal-width binning (cut) if quantiles coincide.
    
    Args:
        y_array: Array of continuous labels
        k: Number of folds for cross-validation (used to determine optimal bin count)
    
    Returns:
        Array of binned labels (integers)
    
    Note:
        - Number of bins is min(10, max(5, k*2)) to ensure at least 2 bins per fold
        - For small samples (n < k*2), uses max(2, k) bins to ensure feasibility
        - Uses quantile-based binning for balanced bin sizes
        - Falls back to equal-width binning if qcut fails
    """
    import numpy as np
    
    n_samples = len(y_array)
    
    # Determine number of bins: min(10, max(5, k*2))
    # For small samples, limit bins to avoid having fewer samples than bins
    # Ensure at least k bins (one per fold) but not more than n_samples/2
    n_bins = min(10, max(5, k * 2))
    n_bins = min(n_bins, max(2, n_samples // 2))  # At least 2 samples per bin on average
    n_bins = max(n_bins, k)  # At least k bins for k folds
    
    # If we have very few samples, just use k bins
    if n_samples < k * 2:
        n_bins = max(2, k)
    
    # Use qcut for quantile-based binning (equal-sized bins)
    # duplicates='drop' handles cases where quantiles coincide
    try:
        y_binned = pd.qcut(y_array, q=n_bins, labels=False, duplicates='drop')
    except (ValueError, TypeError):
        # If qcut fails (e.g., too many duplicate values or insufficient unique values)
        # Fall back to equal-width binning
        try:
            y_binned = pd.cut(y_array, bins=n_bins, labels=False, duplicates='drop')
        except (ValueError, TypeError):
            # If even cut fails, fall back to simple binning with fewer bins
            n_bins_fallback = min(n_bins, len(np.unique(y_array)))
            if n_bins_fallback < 2:
                # If only 1 unique value, just return zeros (all same bin)
                y_binned = np.zeros(len(y_array), dtype=int)
            else:
                y_binned = pd.cut(y_array, bins=n_bins_fallback, labels=False, duplicates='drop')
    
    return y_binned


@register_cv_strategy('stratified')
def cv_splits_stratifiedkfold(k: int, smiles: list, y: list, val_size: float, random_state: int, shuffle: bool = True) -> list[dict]:
    """
    Split data into k stratified folds for cross-validation.
    
    Stratified splitting ensures each fold maintains the same class distribution as the original dataset.
    For classification problems, stratifies on class labels directly. For regression problems,
    automatically bins continuous values into quantile-based bins to enable stratification.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        y: List of labels (for stratification). Can be discrete class labels or continuous values.
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    
    Note:
        Continuous regression values are automatically binned into quantile-based bins
        to enable stratification. This ensures each fold has a representative distribution
        of the target variable range.
    """
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    smiles_array = np.array(smiles)
    y_array = np.array(y)
    
    # Check if labels are continuous (regression) or discrete (classification)
    n_unique = len(np.unique(y_array))
    
    # If we have many unique values (likely regression), bin them for stratification
    # Rule of thumb: if more unique values than 2*k, treat as regression
    if n_unique > 2:
        y_stratify = _bin_continuous_labels(y_array, k)
    else:
        # Discrete labels - use directly
        y_stratify = y_array
    
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    splits = []
    for train_idx, val_idx in skf.split(smiles_array, y_stratify):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits

@register_cv_strategy('leavepout')
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

@register_cv_strategy('montecarlo')
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

@register_cv_strategy('cluster')
def cv_splits_cluster(k: int, smiles: list, clusters: list, val_size: float, random_state: int, shuffle: bool = True) -> list[dict]:
    """
    Split data into k folds based on pre-defined cluster assignments.
    
    Group-based splitting ensures that all molecules in the same cluster are kept together
    in either training or validation. This is critical for evaluating model generalization
    to new chemical scaffolds or structural clusters.
    
    Uses sklearn's GroupKFold under the hood, which ensures no cluster appears in both
    train and validation within the same fold.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        clusters: List of cluster assignments (one per SMILES). Can be integers, strings, or any hashable type.
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility (used when shuffle=True)
        shuffle: Whether to shuffle the groups before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    
    Note:
        - Fold sizes may be unequal if clusters have different sizes
        - All molecules in the same cluster will be in the same fold
        - Use for scaffold-based or structural cluster-based CV
    """
    from sklearn.model_selection import GroupKFold
    import numpy as np
    
    if len(smiles) != len(clusters):
        raise ValueError(f"Length mismatch: {len(smiles)} SMILES but {len(clusters)} cluster assignments")
    
    smiles_array = np.array(smiles)
    clusters_array = np.array(clusters)
    
    # GroupKFold doesn't support shuffle directly, so we shuffle the groups manually if needed
    if shuffle:
        unique_clusters = np.unique(clusters_array)
        np.random.seed(random_state)
        shuffled_clusters = np.random.permutation(unique_clusters)
        
        # Create a mapping from old cluster ID to new shuffled order
        cluster_map = {old: new for new, old in enumerate(shuffled_clusters)}
        clusters_array = np.array([cluster_map[c] for c in clusters_array])
    
    gkf = GroupKFold(n_splits=k)
    
    splits = []
    for train_idx, val_idx in gkf.split(smiles_array, groups=clusters_array):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits


@register_cv_strategy('scaffold')
def cv_splits_scaffold(k: int, smiles: list, scaffolds: list, val_size: float, random_state: int, shuffle: bool = True) -> list[dict]:
    """
    Split data into k folds based on pre-computed scaffolds.
    
    Takes a list of pre-computed scaffold SMILES and uses cluster-based splitting to ensure
    molecules with the same scaffold are kept together. Molecules without a scaffold (None or '')
    are assigned to a separate 'no_scaffold' cluster. This ensures models are evaluated on
    their ability to generalize to new chemical scaffolds.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        scaffolds: List of scaffold SMILES (one per molecule, pre-computed)
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle the scaffold groups before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    
    Note:
        - Molecules with the same scaffold will always be in the same fold
        - Molecules without a scaffold are grouped together in a 'no_scaffold' cluster
        - Uses cv_splits_cluster internally
        - Scaffolds should be pre-computed using get_scaffold() to avoid redundant computation
    """
    if len(smiles) != len(scaffolds):
        raise ValueError(f"Length mismatch: {len(smiles)} SMILES but {len(scaffolds)} scaffolds")
    
    scaffold_list = scaffolds
    
    # Create cluster assignments: map scaffold SMILES to cluster IDs
    # Molecules with None/no scaffold get their own cluster
    unique_scaffolds = {}
    cluster_id = 0
    
    # Reserve cluster 0 for molecules without scaffolds
    no_scaffold_cluster = 0
    cluster_id = 1
    
    clusters = []
    for scaffold_smi in scaffold_list:
        if scaffold_smi is None or scaffold_smi == '':
            # Assign to 'no_scaffold' cluster
            clusters.append(no_scaffold_cluster)
        else:
            # Assign to scaffold-specific cluster
            if scaffold_smi not in unique_scaffolds:
                unique_scaffolds[scaffold_smi] = cluster_id
                cluster_id += 1
            clusters.append(unique_scaffolds[scaffold_smi])
    
    # Check if we have enough unique scaffolds for k folds
    n_unique_clusters = len(set(clusters))
    if k > n_unique_clusters:
        raise ValueError(
            f"Cannot split into {k} folds with only {n_unique_clusters} unique scaffolds. "
            f"Reduce k to {n_unique_clusters} or fewer."
        )
    
    # Use cluster-based splitting
    return cv_splits_cluster(
        k=k,
        smiles=smiles,
        clusters=clusters,
        val_size=val_size,
        random_state=random_state,
        shuffle=shuffle
    )


def _cross_validate_and_eval(
    model_algorithm: str,
    dataset: pd.DataFrame,
    smiles_column: str,
    label_column: str,
    feature_vector_dict: dict,
    cv_strategy: str,
    n_folds: int,
    random_state: int,
    metric: str,
    hyperparameters: dict = None,
    cluster_column: str = None,
    val_size: float = None,
    scaffold_column: str = None,
    shuffle: bool = True,
    p: int = 1,
    max_splits: int = None
) -> list[float]:
    """
    Internal function for cross validation used in hyperparameter tuning.
    
    Args:
        model_algorithm: Name of the ML algorithm (e.g., 'random_forest')
        dataset: DataFrame with SMILES, labels, and optionally cluster columns
        smiles_column: Name of SMILES column in dataset
        label_column: Name of label column in dataset
        feature_vector_dict: Dictionary mapping SMILES to feature vectors {smiles: [features]}
        cv_strategy: CV strategy ('kfold', 'stratified', 'montecarlo', 'scaffold', 'cluster', 'leavepout')
        n_folds: Number of folds/splits
        random_state: Random seed
        metric: Metric to evaluate (e.g., 'accuracy', 'f1_score', 'mse', 'r2')
        hyperparameters: Dictionary of hyperparameters to pass to the model
        cluster_column: Name of cluster column (required for cluster-based CV)
        val_size: Validation size fraction (for montecarlo, optional for others)
        scaffold_column: Name of column with pre-computed scaffolds (required for scaffold-based CV)
        shuffle: Whether to shuffle data before splitting
        p: Number of samples to leave out for leavepout strategy
        max_splits: Maximum number of splits for leavepout strategy
    
    Returns:
        list of metric values 
    """
    from chemlint.tools.ml.training import _train_ml_model
    from chemlint.tools.ml.evaluation import _eval_single_ml_model
    import numpy as np
    
    if hyperparameters is None:
        hyperparameters = {}
    
    # Validate columns exist
    if smiles_column not in dataset.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in dataset")
    if label_column not in dataset.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    # Extract SMILES and labels from dataset
    smiles_list = dataset[smiles_column].tolist()
    labels = dataset[label_column].tolist()
    
    # Get clusters if needed for cluster-based CV
    clusters = None
    if cv_strategy == 'cluster':
        if cluster_column is None:
            raise ValueError("Cluster-based CV requires cluster_column parameter")
        if cluster_column not in dataset.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found in dataset")
        clusters = dataset[cluster_column].tolist()
    
    # Get scaffolds if needed for scaffold-based CV
    scaffolds = None
    if cv_strategy == 'scaffold':
        if scaffold_column is None:
            raise ValueError("Scaffold-based CV requires scaffold_column parameter with pre-computed scaffolds")
        if scaffold_column not in dataset.columns:
            raise ValueError(f"Scaffold column '{scaffold_column}' not found in dataset")
        scaffolds = dataset[scaffold_column].tolist()
    
    # Create CV splits using the generic function
    splits = get_cv_splits(
        strategy=cv_strategy,
        smiles=smiles_list,
        n_folds=n_folds,
        random_state=random_state,
        labels=labels,
        clusters=clusters,
        val_size=val_size,
        scaffolds=scaffolds,
        shuffle=shuffle,
        p=p,
        max_splits=max_splits
    )
    
    # Evaluate on each fold
    scores = []
    for fold_idx, split in enumerate(splits):
        # Get train and val SMILES
        train_smiles = split['train_smiles']
        val_smiles = split['val_smiles']
        
        # Get feature vectors for train and val SMILES
        X_train = np.array([feature_vector_dict[smi] for smi in train_smiles])
        X_val = np.array([feature_vector_dict[smi] for smi in val_smiles])
        
        # Get labels for train and val SMILES from dataset
        # Create SMILES -> label mapping from dataset
        smiles_to_label = dict(zip(dataset[smiles_column], dataset[label_column]))
        
        y_train = np.array([smiles_to_label[smi] for smi in train_smiles])
        y_val = np.array([smiles_to_label[smi] for smi in val_smiles])
        
        # Train model on this fold
        trained_model = _train_ml_model(
            X=X_train,
            y=y_train,
            model_algorithm=model_algorithm,
            hyperparameters=hyperparameters,
            random_state=random_state
        )
        
        # Evaluate on validation set
        fold_score = _eval_single_ml_model(
            model=trained_model,
            X=X_val,
            y=y_val,
            metric=metric
        )
        
        # Skip None scores (e.g., roc_auc for models without predict_proba)
        if fold_score is not None:
            scores.append(fold_score)
    
    # Return average metric across folds
    if len(scores) == 0:
        raise ValueError(f"No valid scores computed for metric '{metric}'")
    
    return scores



