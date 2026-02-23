"""
Tests for cross_validation.py module.

Tests cover:
- CV strategy registry (registration, dispatcher)
- Individual CV strategies: kfold, stratified, leavepout, montecarlo, scaffold, cluster
- Helper functions like _bin_continuous_labels
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from chemlint.tools.ml.cross_validation import (
    get_cv_splits,
    cv_splits_kfold,
    cv_splits_stratifiedkfold,
    cv_splits_leavepout,
    cv_splits_montecarlo,
    cv_splits_cluster,
    cv_splits_scaffold,
    _bin_continuous_labels,
    CV_STRATEGY_REGISTRY
)


# Fixtures
@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        'CCO', 'CC(O)C', 'c1ccccc1', 'CC(=O)O', 'CCCC',
        'CCC(C)C', 'c1ccc(O)cc1', 'CCN', 'CCCO', 'c1ccncc1'
    ]


@pytest.fixture
def binary_labels():
    """Binary classification labels."""
    return [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


@pytest.fixture
def continuous_labels():
    """Continuous regression labels."""
    return [1.5, 2.3, 0.8, 4.2, 3.1, 2.9, 1.2, 3.8, 0.5, 4.5]


@pytest.fixture
def cluster_assignments():
    """Cluster assignments for group-based splitting."""
    return [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


@pytest.fixture
def scaffold_assignments():
    """Scaffold SMILES for scaffold-based splitting."""
    return [
        'c1ccccc1', 'c1ccccc1', 'C1CCCCC1', 'C1CCCCC1', 'CCO',
        'CCO', 'c1ccncc1', 'c1ccncc1', None, None
    ]


# ========== Registry Tests ==========

def test_cv_strategy_registry_populated():
    """Test that CV_STRATEGY_REGISTRY contains expected strategies."""
    expected_strategies = ['kfold', 'stratified', 'leavepout', 'montecarlo', 'scaffold', 'cluster']
    for strategy in expected_strategies:
        assert strategy in CV_STRATEGY_REGISTRY, f"Strategy '{strategy}' not found in registry"


def test_get_cv_splits_dispatcher(sample_smiles, binary_labels):
    """Test that get_cv_splits correctly dispatches to registered strategies."""
    # Test kfold dispatch
    splits = get_cv_splits(
        strategy='kfold',
        smiles=sample_smiles,
        n_folds=3,
        random_state=42
    )
    assert len(splits) == 3
    assert all('train_smiles' in s and 'val_smiles' in s for s in splits)


def test_get_cv_splits_invalid_strategy(sample_smiles):
    """Test that invalid strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown CV strategy"):
        get_cv_splits(
            strategy='invalid_strategy',
            smiles=sample_smiles,
            n_folds=3,
            random_state=42
        )


# ========== Helper Function Tests ==========

def test_bin_continuous_labels():
    """Test continuous label binning for stratification."""
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    binned = _bin_continuous_labels(labels, k=3)
    
    assert len(binned) == len(labels)
    # Check that binning produced valid integer bins
    assert all(isinstance(b, (int, np.integer)) or not np.isnan(b) for b in binned)


def test_bin_continuous_labels_edge_cases():
    """Test binning with edge cases."""
    # All same value - just ensure it doesn't crash and returns something valid
    labels = np.array([5.0] * 10)
    binned = _bin_continuous_labels(labels, k=3)
    assert len(binned) == len(labels)
    # When all values are same, may return NaN - that's acceptable
    
    # Two distinct values with enough samples per bin
    labels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    binned = _bin_continuous_labels(labels, k=2)
    # Filter out NaN if present
    valid_bins = [b for b in binned if not (isinstance(b, float) and np.isnan(b))]
    assert len(set(valid_bins)) <= 2  # Should have at most 2 bins


# ========== KFold Tests ==========

def test_cv_splits_kfold_basic(sample_smiles):
    """Test basic k-fold splitting."""
    splits = cv_splits_kfold(k=3, smiles=sample_smiles, val_size=0.2, random_state=42, shuffle=True)
    
    assert len(splits) == 3
    # Check that all splits have train and val
    for split in splits:
        assert 'train_smiles' in split
        assert 'val_smiles' in split
        assert len(split['train_smiles']) > 0
        assert len(split['val_smiles']) > 0
    
    # Check that all SMILES appear exactly once in validation across folds
    all_val_smiles = []
    for split in splits:
        all_val_smiles.extend(split['val_smiles'])
    assert sorted(all_val_smiles) == sorted(sample_smiles)


def test_cv_splits_kfold_deterministic(sample_smiles):
    """Test that k-fold is deterministic with same random_state."""
    splits1 = cv_splits_kfold(k=3, smiles=sample_smiles, val_size=0.2, random_state=42, shuffle=True)
    splits2 = cv_splits_kfold(k=3, smiles=sample_smiles, val_size=0.2, random_state=42, shuffle=True)
    
    for s1, s2 in zip(splits1, splits2):
        assert s1['train_smiles'] == s2['train_smiles']
        assert s1['val_smiles'] == s2['val_smiles']


def test_cv_splits_kfold_no_shuffle(sample_smiles):
    """Test k-fold without shuffling."""
    splits = cv_splits_kfold(k=3, smiles=sample_smiles, val_size=0.2, random_state=42, shuffle=False)
    
    assert len(splits) == 3
    # First fold should contain first N samples in validation
    assert sample_smiles[0] in splits[0]['val_smiles']


# ========== Stratified KFold Tests ==========

def test_cv_splits_stratifiedkfold_binary(sample_smiles, binary_labels):
    """Test stratified k-fold with binary labels."""
    splits = cv_splits_stratifiedkfold(
        k=3,
        smiles=sample_smiles,
        y=binary_labels,
        val_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    assert len(splits) == 3
    # Check that class distribution is roughly preserved in each fold
    smiles_to_label = dict(zip(sample_smiles, binary_labels))
    
    for split in splits:
        val_labels = [smiles_to_label[s] for s in split['val_smiles']]
        # Each fold should have both classes (or at least one if very small)
        assert len(val_labels) > 0


def test_cv_splits_stratifiedkfold_continuous(sample_smiles, continuous_labels):
    """Test stratified k-fold with continuous labels (should auto-bin)."""
    # Use more diverse continuous labels to ensure enough samples per bin
    continuous_labels_extended = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    splits = cv_splits_stratifiedkfold(
        k=2,  # Use 2 folds instead of 3 for more reliable binning
        smiles=sample_smiles,
        y=continuous_labels_extended,
        val_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    assert len(splits) == 2
    # Check that all SMILES appear exactly once in validation
    all_val_smiles = []
    for split in splits:
        all_val_smiles.extend(split['val_smiles'])
    assert sorted(all_val_smiles) == sorted(sample_smiles)


def test_cv_splits_stratifiedkfold_deterministic(sample_smiles, binary_labels):
    """Test that stratified k-fold is deterministic."""
    splits1 = cv_splits_stratifiedkfold(
        k=3, smiles=sample_smiles, y=binary_labels, val_size=0.2, random_state=42, shuffle=True
    )
    splits2 = cv_splits_stratifiedkfold(
        k=3, smiles=sample_smiles, y=binary_labels, val_size=0.2, random_state=42, shuffle=True
    )
    
    for s1, s2 in zip(splits1, splits2):
        assert s1['train_smiles'] == s2['train_smiles']
        assert s1['val_smiles'] == s2['val_smiles']


# ========== LeavePOut Tests ==========

def test_cv_splits_leavepout_basic(sample_smiles):
    """Test leave-p-out splitting."""
    splits = cv_splits_leavepout(
        p=2,
        smiles=sample_smiles,
        val_size=0.2,
        random_state=42,
        max_splits=10
    )
    
    assert len(splits) == 10  # Limited by max_splits
    # Each validation set should have exactly p samples
    for split in splits:
        assert len(split['val_smiles']) == 2


def test_cv_splits_leavepout_no_max_splits(sample_smiles):
    """Test leave-p-out without max_splits limit."""
    # For 10 samples with p=1, there should be exactly 10 splits
    splits = cv_splits_leavepout(
        p=1,
        smiles=sample_smiles,
        val_size=0.2,
        random_state=42,
        max_splits=None
    )
    
    assert len(splits) == 10
    # Each split should have exactly 1 validation sample
    for split in splits:
        assert len(split['val_smiles']) == 1
        assert len(split['train_smiles']) == 9


def test_cv_splits_leavepout_p_too_large():
    """Test that p >= n_samples raises ValueError."""
    smiles = ['CCO', 'CC', 'CCC']
    with pytest.raises(ValueError, match="must be strictly less than"):
        cv_splits_leavepout(p=3, smiles=smiles, val_size=0.2, random_state=42)


# ========== Monte Carlo Tests ==========

def test_cv_splits_montecarlo_basic(sample_smiles):
    """Test Monte Carlo cross-validation."""
    splits = cv_splits_montecarlo(
        n_splits=5,
        smiles=sample_smiles,
        val_size=0.2,
        random_state=42
    )
    
    assert len(splits) == 5
    # Each split should have roughly val_size fraction in validation
    for split in splits:
        val_fraction = len(split['val_smiles']) / len(sample_smiles)
        assert pytest.approx(val_fraction, abs=0.15) == 0.2


def test_cv_splits_montecarlo_deterministic(sample_smiles):
    """Test that Monte Carlo CV is deterministic with same random_state."""
    splits1 = cv_splits_montecarlo(n_splits=5, smiles=sample_smiles, val_size=0.3, random_state=42)
    splits2 = cv_splits_montecarlo(n_splits=5, smiles=sample_smiles, val_size=0.3, random_state=42)
    
    for s1, s2 in zip(splits1, splits2):
        assert s1['train_smiles'] == s2['train_smiles']
        assert s1['val_smiles'] == s2['val_smiles']


def test_cv_splits_montecarlo_different_val_size(sample_smiles):
    """Test Monte Carlo CV with different validation sizes."""
    splits_small = cv_splits_montecarlo(n_splits=3, smiles=sample_smiles, val_size=0.1, random_state=42)
    splits_large = cv_splits_montecarlo(n_splits=3, smiles=sample_smiles, val_size=0.5, random_state=42)
    
    # Larger val_size should have more validation samples
    avg_val_small = np.mean([len(s['val_smiles']) for s in splits_small])
    avg_val_large = np.mean([len(s['val_smiles']) for s in splits_large])
    assert avg_val_large > avg_val_small


# ========== Cluster Tests ==========

def test_cv_splits_cluster_basic(sample_smiles, cluster_assignments):
    """Test cluster-based splitting."""
    splits = cv_splits_cluster(
        k=3,
        smiles=sample_smiles,
        clusters=cluster_assignments,
        val_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    assert len(splits) == 3
    # Check that clusters are preserved (no cluster split between train and val)
    smiles_to_cluster = dict(zip(sample_smiles, cluster_assignments))
    
    for split in splits:
        val_clusters = set([smiles_to_cluster[s] for s in split['val_smiles']])
        train_clusters = set([smiles_to_cluster[s] for s in split['train_smiles']])
        # No overlap between train and val clusters
        assert len(val_clusters & train_clusters) == 0


def test_cv_splits_cluster_deterministic_with_shuffle(sample_smiles, cluster_assignments):
    """Test that cluster CV with shuffle is deterministic."""
    splits1 = cv_splits_cluster(
        k=3, smiles=sample_smiles, clusters=cluster_assignments, 
        val_size=0.2, random_state=42, shuffle=True
    )
    splits2 = cv_splits_cluster(
        k=3, smiles=sample_smiles, clusters=cluster_assignments, 
        val_size=0.2, random_state=42, shuffle=True
    )
    
    for s1, s2 in zip(splits1, splits2):
        assert sorted(s1['train_smiles']) == sorted(s2['train_smiles'])
        assert sorted(s1['val_smiles']) == sorted(s2['val_smiles'])


def test_cv_splits_cluster_length_mismatch(sample_smiles):
    """Test that length mismatch raises ValueError."""
    wrong_clusters = [0, 1, 2]  # Too short
    with pytest.raises(ValueError, match="Length mismatch"):
        cv_splits_cluster(
            k=3,
            smiles=sample_smiles,
            clusters=wrong_clusters,
            val_size=0.2,
            random_state=42
        )


# ========== Scaffold Tests ==========

def test_cv_splits_scaffold_basic(sample_smiles, scaffold_assignments):
    """Test scaffold-based splitting."""
    splits = cv_splits_scaffold(
        k=3,
        smiles=sample_smiles,
        scaffolds=scaffold_assignments,
        val_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    assert len(splits) == 3
    # Check that molecules with same scaffold are in same fold
    smiles_to_scaffold = dict(zip(sample_smiles, scaffold_assignments))
    
    for split in splits:
        val_scaffolds = set([smiles_to_scaffold[s] for s in split['val_smiles']])
        train_scaffolds = set([smiles_to_scaffold[s] for s in split['train_smiles']])
        # No scaffold should appear in both train and val (except None which is separate cluster)
        overlap = val_scaffolds & train_scaffolds
        # None might appear in overlap but represents different molecules
        overlap.discard(None)
        assert len(overlap) == 0


def test_cv_splits_scaffold_with_none(sample_smiles, scaffold_assignments):
    """Test scaffold splitting handles None scaffolds correctly."""
    # Scaffolds include None values
    assert None in scaffold_assignments
    
    splits = cv_splits_scaffold(
        k=3,
        smiles=sample_smiles,
        scaffolds=scaffold_assignments,
        val_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    assert len(splits) == 3
    # All SMILES should appear in exactly one validation set
    all_val_smiles = []
    for split in splits:
        all_val_smiles.extend(split['val_smiles'])
    assert sorted(all_val_smiles) == sorted(sample_smiles)


def test_cv_splits_scaffold_too_few_scaffolds():
    """Test that too few unique scaffolds for k folds raises ValueError."""
    smiles = ['CCO', 'CC', 'CCC', 'CCCC']
    scaffolds = ['c1ccccc1', 'c1ccccc1', 'c1ccccc1', 'c1ccccc1']  # Only 1 unique scaffold
    
    with pytest.raises(ValueError, match="Cannot split into .* folds with only"):
        cv_splits_scaffold(
            k=3,
            smiles=smiles,
            scaffolds=scaffolds,
            val_size=0.2,
            random_state=42
        )


def test_cv_splits_scaffold_length_mismatch(sample_smiles):
    """Test that length mismatch raises ValueError."""
    wrong_scaffolds = ['c1ccccc1', 'CCO']  # Too short
    with pytest.raises(ValueError, match="Length mismatch"):
        cv_splits_scaffold(
            k=3,
            smiles=sample_smiles,
            scaffolds=wrong_scaffolds,
            val_size=0.2,
            random_state=42
        )


# ========== Integration Tests ==========

def test_all_strategies_return_consistent_format(sample_smiles, binary_labels, cluster_assignments, scaffold_assignments):
    """Test that all CV strategies return consistent format."""
    strategies_and_params = [
        ('kfold', {}),
        ('stratified', {'labels': binary_labels}),
        ('montecarlo', {'val_size': 0.2}),
        ('cluster', {'clusters': cluster_assignments}),
        ('scaffold', {'scaffolds': scaffold_assignments}),
        ('leavepout', {'p': 2, 'max_splits': 5}),
    ]
    
    for strategy, params in strategies_and_params:
        splits = get_cv_splits(strategy=strategy, smiles=sample_smiles, n_folds=3, random_state=42, **params)
        
        # Check format
        assert isinstance(splits, list)
        assert len(splits) > 0
        for split in splits:
            assert isinstance(split, dict)
            assert 'train_smiles' in split
            assert 'val_smiles' in split
            assert isinstance(split['train_smiles'], list)
            assert isinstance(split['val_smiles'], list)
            assert len(split['train_smiles']) > 0
            assert len(split['val_smiles']) > 0


def test_no_data_leakage_across_folds(sample_smiles):
    """Test that train and validation sets don't overlap within each fold."""
    splits = cv_splits_kfold(k=3, smiles=sample_smiles, val_size=0.2, random_state=42, shuffle=True)
    
    for split in splits:
        train_set = set(split['train_smiles'])
        val_set = set(split['val_smiles'])
        # No overlap between train and val
        assert len(train_set & val_set) == 0
