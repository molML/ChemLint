"""
Comprehensive test suite for _analyze_split_characteristics function.

Tests all aspects including:
- Size calculations
- Ratio computations
- Flag detection (empty, small, imbalanced)
- Classification vs regression detection
- Class distribution analysis
- Value distribution analysis
- Edge cases and boundary conditions
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from molml_mcp.tools.reports.data_splitting import _analyze_split_characteristics
from molml_mcp.infrastructure.resources import _store_resource

# Test data directory
TEST_DIR = os.path.join(os.path.dirname(__file__), 'data')
MANIFEST_PATH = os.path.join(TEST_DIR, 'test_manifest.json')

def create_test_datasets():
    """Create test datasets for all scenarios."""
    datasets = {}
    
    # 1. Balanced regression dataset (80/20 split)
    df_train = pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC'] * 20,  # 80 molecules
        'activity': np.random.uniform(1, 100, 80)
    })
    df_test = pd.DataFrame({
        'smiles': ['CC', 'C'] * 10,  # 20 molecules
        'activity': np.random.uniform(1, 100, 20)
    })
    datasets['balanced_regression'] = (df_train, df_test, None)
    
    # 2. Balanced classification dataset (80/20 split)
    df_train = pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC'] * 20,
        'label': [0, 1] * 40
    })
    df_test = pd.DataFrame({
        'smiles': ['CC', 'C'] * 10,
        'label': [0, 1] * 10
    })
    datasets['balanced_classification'] = (df_train, df_test, None)
    
    # 3. With validation split (70/20/10)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 70,
        'activity': np.random.uniform(1, 100, 70)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 20,
        'activity': np.random.uniform(1, 100, 20)
    })
    df_val = pd.DataFrame({
        'smiles': ['CCCC'] * 10,
        'activity': np.random.uniform(1, 100, 10)
    })
    datasets['with_validation'] = (df_train, df_test, df_val)
    
    # 4. Imbalanced splits (95/5)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 95,
        'activity': np.random.uniform(1, 100, 95)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 5,
        'activity': np.random.uniform(1, 100, 5)
    })
    datasets['imbalanced_95_5'] = (df_train, df_test, None)
    
    # 5. Small test set (train=200, test=30)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 200,
        'activity': np.random.uniform(1, 100, 200)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 30,
        'activity': np.random.uniform(1, 100, 30)
    })
    datasets['small_test'] = (df_train, df_test, None)
    
    # 6. Empty test set
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'activity': np.random.uniform(1, 100, 100)
    })
    df_test = pd.DataFrame({
        'smiles': [],
        'activity': []
    })
    datasets['empty_test'] = (df_train, df_test, None)
    
    # 7. Multi-class classification
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 90,
        'label': [0] * 30 + [1] * 30 + [2] * 30
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 30,
        'label': [0] * 10 + [1] * 10 + [2] * 10
    })
    datasets['multiclass'] = (df_train, df_test, None)
    
    # 8. Very small splits (below threshold)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 40,
        'activity': np.random.uniform(1, 100, 40)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 10,
        'activity': np.random.uniform(1, 100, 10)
    })
    datasets['below_threshold'] = (df_train, df_test, None)
    
    # 9. Extreme imbalance (train:test = 1:10)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 10,
        'activity': np.random.uniform(1, 100, 10)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 100,
        'activity': np.random.uniform(1, 100, 100)
    })
    datasets['train_smaller_than_test'] = (df_train, df_test, None)
    
    # 10. Binary classification with class imbalance
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'label': [0] * 80 + [1] * 20
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 100,
        'label': [0] * 50 + [1] * 50
    })
    datasets['imbalanced_classes'] = (df_train, df_test, None)
    
    # 11. Missing label column in one split
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 50,
        'activity': np.random.uniform(1, 100, 50)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'different_col': np.random.uniform(1, 100, 50)
    })
    datasets['missing_label_col'] = (df_train, df_test, None)
    
    # 12. NaN values in labels
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 50,
        'activity': [np.nan] * 10 + list(np.random.uniform(1, 100, 40))
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'activity': [np.nan] * 5 + list(np.random.uniform(1, 100, 45))
    })
    datasets['nan_labels'] = (df_train, df_test, None)
    
    # 13. Single molecule datasets
    df_train = pd.DataFrame({
        'smiles': ['CCO'],
        'activity': [50.5]  # Non-integer to ensure regression
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'],
        'activity': [60.7]  # Non-integer to ensure regression
    })
    datasets['single_molecule'] = (df_train, df_test, None)
    
    # 14. Large dataset (stress test)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 10000,
        'activity': np.random.uniform(1, 100, 10000)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 2500,
        'activity': np.random.uniform(1, 100, 2500)
    })
    datasets['large_dataset'] = (df_train, df_test, None)
    
    # 15. All validation split edge case
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 60,
        'activity': np.random.uniform(1, 100, 60)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 20,
        'activity': np.random.uniform(1, 100, 20)
    })
    df_val = pd.DataFrame({
        'smiles': ['CCCC'] * 20,
        'activity': np.random.uniform(1, 100, 20)
    })
    datasets['equal_test_val'] = (df_train, df_test, df_val)
    
    return datasets


def store_test_data(datasets):
    """Store all test datasets and return their filenames."""
    stored = {}
    
    for name, (df_train, df_test, df_val) in datasets.items():
        train_file = _store_resource(
            df_train, MANIFEST_PATH, f'train_{name}', 
            f'Training data for {name}', 'csv'
        )
        test_file = _store_resource(
            df_test, MANIFEST_PATH, f'test_{name}', 
            f'Test data for {name}', 'csv'
        )
        val_file = None
        if df_val is not None:
            val_file = _store_resource(
                df_val, MANIFEST_PATH, f'val_{name}', 
                f'Validation data for {name}', 'csv'
            )
        
        stored[name] = (train_file, test_file, val_file)
    
    return stored


def test_basic_functionality():
    """Test basic size calculations and structure."""
    print("\n=== BASIC FUNCTIONALITY ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    # Test balanced regression
    train_file, test_file, val_file = stored['balanced_regression']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    assert 'sizes' in result
    assert 'percentages' in result
    assert 'ratios' in result
    assert 'flags' in result
    assert 'task_type' in result
    
    assert result['sizes']['train'] == 80
    assert result['sizes']['test'] == 20
    assert result['sizes']['total'] == 100
    
    assert abs(result['percentages']['train'] - 80.0) < 0.1
    assert abs(result['percentages']['test'] - 20.0) < 0.1
    
    assert result['task_type'] == 'regression'
    assert 'value_distribution' in result
    
    print("✅ Basic structure correct")
    print("✅ Size calculations correct")
    print("✅ Task type detection works")


def test_classification_detection():
    """Test classification vs regression detection."""
    print("\n=== CLASSIFICATION DETECTION ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    # Binary classification
    train_file, test_file, val_file = stored['balanced_classification']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'label'
    )
    
    assert result['task_type'] == 'classification'
    assert 'class_distribution' in result
    assert 'train' in result['class_distribution']
    assert 'test' in result['class_distribution']
    
    train_dist = result['class_distribution']['train']
    assert train_dist['n_classes'] == 2
    assert train_dist['n_samples'] == 80
    assert 0 in train_dist['counts'] or '0' in train_dist['counts']
    
    print("✅ Binary classification detected")
    
    # Multi-class (should now be treated as regression since we only support binary)
    train_file, test_file, val_file = stored['multiclass']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'label'
    )
    
    # With binary-only restriction, multi-class is treated as regression
    assert result['task_type'] == 'regression'
    assert 'value_distribution' in result
    assert 'class_distribution' not in result
    
    print("✅ Multi-class correctly treated as regression (binary-only mode)")


def test_validation_split():
    """Test with validation split."""
    print("\n=== VALIDATION SPLIT ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    train_file, test_file, val_file = stored['with_validation']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    assert result['sizes']['val'] == 10
    assert result['percentages']['val'] is not None
    assert abs(result['percentages']['train'] - 70.0) < 0.1
    assert abs(result['percentages']['test'] - 20.0) < 0.1
    assert abs(result['percentages']['val'] - 10.0) < 0.1
    
    assert 'train_val' in result['ratios']
    assert 'test_val' in result['ratios']
    
    assert 'val' in result['value_distribution']
    
    print("✅ Validation split handled correctly")
    print("✅ Three-way split percentages correct")


def test_imbalance_detection():
    """Test detection of imbalanced splits."""
    print("\n=== IMBALANCE DETECTION ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    # 95/5 split
    train_file, test_file, val_file = stored['imbalanced_95_5']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    assert len(result['flags']['imbalanced_splits']) > 0
    assert result['ratios']['train_test'] == 19.0
    
    print("✅ 95/5 split flagged as imbalanced")
    
    # Train smaller than test
    train_file, test_file, val_file = stored['train_smaller_than_test']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    assert len(result['flags']['imbalanced_splits']) > 0
    imbalance_msgs = ' '.join(result['flags']['imbalanced_splits'])
    assert 'Train too small relative to test' in imbalance_msgs or 'only' in imbalance_msgs.lower()
    
    print("✅ Train<Test flagged correctly")


def test_small_split_detection():
    """Test detection of small splits."""
    print("\n=== SMALL SPLIT DETECTION ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    train_file, test_file, val_file = stored['below_threshold']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity', min_split_size=50
    )
    
    assert len(result['flags']['small_splits']) > 0
    small_flags = ' '.join(result['flags']['small_splits'])
    assert 'train' in small_flags.lower() or 'test' in small_flags.lower()
    
    print("✅ Small splits detected")


def test_empty_split():
    """Test empty split handling."""
    print("\n=== EMPTY SPLIT ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    train_file, test_file, val_file = stored['empty_test']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    assert 'test' in result['flags']['empty_splits']
    assert result['sizes']['test'] == 0
    assert result['percentages']['test'] == 0.0
    
    print("✅ Empty test split handled")


def test_ratio_calculations():
    """Test ratio calculations."""
    print("\n=== RATIO CALCULATIONS ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    # Balanced 80/20
    train_file, test_file, val_file = stored['balanced_regression']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    assert result['ratios']['train_test'] == 4.0
    assert result['ratios']['test_train'] == 0.25
    
    print("✅ 80/20 ratios correct")
    
    # With validation
    train_file, test_file, val_file = stored['with_validation']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    assert 'train_val' in result['ratios']
    assert 'val_train' in result['ratios']
    assert result['ratios']['train_val'] == 7.0
    
    print("✅ Validation ratios correct")


def test_class_distribution():
    """Test class distribution analysis."""
    print("\n=== CLASS DISTRIBUTION ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    train_file, test_file, val_file = stored['balanced_classification']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'label'
    )
    
    train_dist = result['class_distribution']['train']
    test_dist = result['class_distribution']['test']
    
    # Check structure
    assert 'counts' in train_dist
    assert 'proportions' in train_dist
    assert 'n_classes' in train_dist
    assert 'n_samples' in train_dist
    
    # Check counts
    total_count = sum(train_dist['counts'].values())
    assert total_count == 80
    
    # Check proportions sum to 1
    total_prop = sum(train_dist['proportions'].values())
    assert abs(total_prop - 1.0) < 0.01
    
    print("✅ Class distribution structure correct")
    print("✅ Counts and proportions accurate")


def test_value_distribution():
    """Test value distribution for regression."""
    print("\n=== VALUE DISTRIBUTION ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    train_file, test_file, val_file = stored['balanced_regression']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    train_dist = result['value_distribution']['train']
    
    # Check structure
    assert 'n_samples' in train_dist
    assert 'mean' in train_dist
    assert 'std' in train_dist
    assert 'min' in train_dist
    assert 'max' in train_dist
    assert 'median' in train_dist
    assert 'q25' in train_dist
    assert 'q75' in train_dist
    
    # Check reasonable values
    assert train_dist['n_samples'] == 80
    assert 1.0 <= train_dist['min'] <= 100.0
    assert 1.0 <= train_dist['max'] <= 100.0
    assert train_dist['min'] <= train_dist['median'] <= train_dist['max']
    assert train_dist['q25'] <= train_dist['median'] <= train_dist['q75']
    
    print("✅ Value distribution structure correct")
    print("✅ Statistics are reasonable")


def test_edge_cases():
    """Test edge cases."""
    print("\n=== EDGE CASES ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    # Single molecule
    train_file, test_file, val_file = stored['single_molecule']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    assert result['sizes']['train'] == 1
    assert result['sizes']['test'] == 1
    # With only 2 unique float values, should be regression (not enough for classification)
    assert result['task_type'] == 'regression'
    
    print("✅ Single molecule datasets handled")
    
    # Missing label column
    train_file, test_file, val_file = stored['missing_label_col']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    # Should handle gracefully
    assert 'value_distribution' in result
    assert 'error' in result['value_distribution']['test']
    
    print("✅ Missing label column handled gracefully")
    
    # NaN labels
    train_file, test_file, val_file = stored['nan_labels']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    # Should only count non-NaN values
    assert result['value_distribution']['train']['n_samples'] == 40
    assert result['value_distribution']['test']['n_samples'] == 45
    
    print("✅ NaN labels handled correctly")


def test_large_dataset():
    """Test with large dataset (performance check)."""
    print("\n=== LARGE DATASET ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    import time
    start = time.time()
    
    train_file, test_file, val_file = stored['large_dataset']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    elapsed = time.time() - start
    
    assert result['sizes']['train'] == 10000
    assert result['sizes']['test'] == 2500
    assert elapsed < 5.0  # Should complete in reasonable time
    
    print(f"✅ Large dataset (10k+2.5k) processed in {elapsed:.2f}s")


def test_consistency():
    """Test that results are consistent across multiple runs."""
    print("\n=== CONSISTENCY ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    train_file, test_file, val_file = stored['balanced_regression']
    
    result1 = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    result2 = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    
    # Check key fields are identical
    assert result1['sizes'] == result2['sizes']
    assert result1['percentages'] == result2['percentages']
    assert result1['ratios'] == result2['ratios']
    assert result1['task_type'] == result2['task_type']
    
    print("✅ Results are consistent across runs")


def test_all_flag_types():
    """Test that all flag types can be triggered."""
    print("\n=== ALL FLAG TYPES ===")
    
    datasets = create_test_datasets()
    stored = store_test_data(datasets)
    
    # Empty flag
    train_file, test_file, val_file = stored['empty_test']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    assert len(result['flags']['empty_splits']) > 0
    print("✅ Empty flag can be triggered")
    
    # Small flag
    train_file, test_file, val_file = stored['below_threshold']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity', min_split_size=50
    )
    assert len(result['flags']['small_splits']) > 0
    print("✅ Small flag can be triggered")
    
    # Imbalanced flag
    train_file, test_file, val_file = stored['imbalanced_95_5']
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH,
        'smiles', 'activity'
    )
    assert len(result['flags']['imbalanced_splits']) > 0
    print("✅ Imbalanced flag can be triggered")


if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE TEST SUITE: _analyze_split_characteristics")
    print("="*80)
    
    try:
        test_basic_functionality()
        test_classification_detection()
        test_validation_split()
        test_imbalance_detection()
        test_small_split_detection()
        test_empty_split()
        test_ratio_calculations()
        test_class_distribution()
        test_value_distribution()
        test_edge_cases()
        test_large_dataset()
        test_consistency()
        test_all_flag_types()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
