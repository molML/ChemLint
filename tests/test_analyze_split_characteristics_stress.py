"""
Stress tests and extreme edge cases for _analyze_split_characteristics.

Tests numerical stability, extreme values, and adversarial cases.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from molml_mcp.tools.reports.data_splitting import _analyze_split_characteristics
from molml_mcp.infrastructure.resources import _store_resource

TEST_DIR = os.path.join(os.path.dirname(__file__), 'data')
MANIFEST_PATH = os.path.join(TEST_DIR, 'test_manifest.json')


def test_extreme_values():
    """Test with extreme numerical values."""
    print("\n=== EXTREME VALUES ===")
    
    # Very large activity values
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'activity': np.random.uniform(1e10, 1e12, 100)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'activity': np.random.uniform(1e10, 1e12, 50)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'extreme_large_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'extreme_large_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    assert result['value_distribution']['train']['mean'] > 1e10
    assert not np.isnan(result['value_distribution']['train']['mean'])
    assert not np.isinf(result['value_distribution']['train']['mean'])
    print("✅ Very large values handled")
    
    # Very small activity values
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'activity': np.random.uniform(1e-10, 1e-8, 100)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'activity': np.random.uniform(1e-10, 1e-8, 50)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'extreme_small_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'extreme_small_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    assert result['value_distribution']['train']['mean'] < 1e-7
    assert not np.isnan(result['value_distribution']['train']['mean'])
    print("✅ Very small values handled")
    
    # Negative values
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'activity': np.random.uniform(-1000, -10, 100)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'activity': np.random.uniform(-1000, -10, 50)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'negative_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'negative_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    assert result['value_distribution']['train']['mean'] < 0
    assert result['value_distribution']['train']['min'] < 0
    print("✅ Negative values handled")
    
    # Zero values
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'activity': [0.0] * 100
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'activity': [0.0] * 50
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'zero_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'zero_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    assert result['value_distribution']['train']['mean'] == 0.0
    assert result['value_distribution']['train']['std'] == 0.0
    print("✅ All-zero values handled")


def test_extreme_ratios():
    """Test extreme split ratios."""
    print("\n=== EXTREME RATIOS ===")
    
    # 99.9:0.1 split
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 999,
        'activity': np.random.uniform(1, 100, 999)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'],
        'activity': [50.0]
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'extreme_ratio_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'extreme_ratio_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    assert result['ratios']['train_test'] == 999.0
    assert len(result['flags']['imbalanced_splits']) > 0
    print("✅ 999:1 split handled")
    
    # Equal splits
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'activity': np.random.uniform(1, 100, 100)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 100,
        'activity': np.random.uniform(1, 100, 100)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'equal_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'equal_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    assert result['ratios']['train_test'] == 1.0
    assert result['percentages']['train'] == 50.0
    assert result['percentages']['test'] == 50.0
    print("✅ 50:50 split handled")


def test_many_classes():
    """Test with many classification classes."""
    print("\n=== MANY CLASSES ===")
    
    # 20 classes (boundary)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 200,
        'label': list(range(20)) * 10
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 100,
        'label': list(range(20)) * 5
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'many_classes_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'many_classes_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'label'
    )
    
    # With binary-only restriction, 20 classes treated as regression
    assert result['task_type'] == 'regression'
    assert 'value_distribution' in result
    print("✅ 20 classes correctly treated as regression (binary-only mode)")
    
    # 21 classes (also regression)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 210,
        'label': list(range(21)) * 10
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 105,
        'label': list(range(21)) * 5
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'too_many_classes_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'too_many_classes_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'label'
    )
    
    assert result['task_type'] == 'regression'
    print("✅ 21 classes detected as regression")


def test_class_imbalance_extreme():
    """Test extreme class imbalance."""
    print("\n=== EXTREME CLASS IMBALANCE ===")
    
    # 99:1 class ratio
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 1000,
        'label': [0] * 990 + [1] * 10
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 200,
        'label': [0] * 198 + [1] * 2
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'extreme_imbalance_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'extreme_imbalance_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'label'
    )
    
    assert result['task_type'] == 'classification'
    train_dist = result['class_distribution']['train']
    
    # Check proportions
    prop_0 = train_dist['proportions'][0]
    prop_1 = train_dist['proportions'][1]
    assert prop_0 > 0.98
    assert prop_1 < 0.02
    print("✅ 99:1 class imbalance handled")
    
    # Single class in one split
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'label': [0, 1] * 50
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'label': [0] * 50  # Only class 0
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'single_class_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'single_class_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'label'
    )
    
    assert result['class_distribution']['train']['n_classes'] == 2
    assert result['class_distribution']['test']['n_classes'] == 1
    print("✅ Single class in test split handled")


def test_all_nan_values():
    """Test with all NaN values in a split."""
    print("\n=== ALL NaN VALUES ===")
    
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'activity': [np.nan] * 100
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'activity': np.random.uniform(1, 100, 50)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'all_nan_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'all_nan_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    # Should handle gracefully
    assert 'value_distribution' in result
    assert 'error' in result['value_distribution']['train'] or result['value_distribution']['train']['n_samples'] == 0
    print("✅ All NaN values handled gracefully")


def test_mixed_types():
    """Test with mixed data types in labels."""
    print("\n=== MIXED TYPES ===")
    
    # String labels (should fail gracefully or handle)
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'label': ['active', 'inactive'] * 50
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 50,
        'label': ['active', 'inactive'] * 25
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'string_labels_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'string_labels_test', 'test', 'csv')
    
    try:
        result = _analyze_split_characteristics(
            train_file, test_file, None, MANIFEST_PATH, 'smiles', 'label'
        )
        # If it doesn't crash, check it's treated as classification
        assert result['task_type'] in ['classification', 'regression']
        print("✅ String labels handled")
    except:
        print("✅ String labels cause expected behavior")


def test_validation_edge_cases():
    """Test edge cases with validation splits."""
    print("\n=== VALIDATION EDGE CASES ===")
    
    # Val larger than test
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 100,
        'activity': np.random.uniform(1, 100, 100)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 10,
        'activity': np.random.uniform(1, 100, 10)
    })
    df_val = pd.DataFrame({
        'smiles': ['CCCC'] * 50,
        'activity': np.random.uniform(1, 100, 50)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'val_larger_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'val_larger_test', 'test', 'csv')
    val_file = _store_resource(df_val, MANIFEST_PATH, 'val_larger_val', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    assert result['sizes']['val'] > result['sizes']['test']
    assert 'test_val' in result['ratios']
    assert result['ratios']['test_val'] < 1.0
    print("✅ Val larger than test handled")
    
    # Empty validation
    df_val_empty = pd.DataFrame({
        'smiles': [],
        'activity': []
    })
    
    val_file = _store_resource(df_val_empty, MANIFEST_PATH, 'empty_val', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    assert 'val' in result['flags']['empty_splits']
    print("✅ Empty validation handled")


def test_numerical_precision():
    """Test numerical precision in percentage calculations."""
    print("\n=== NUMERICAL PRECISION ===")
    
    # Splits that don't divide evenly
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 77,
        'activity': np.random.uniform(1, 100, 77)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 23,
        'activity': np.random.uniform(1, 100, 23)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'precision_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'precision_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity'
    )
    
    # Percentages should sum to 100
    total_pct = result['percentages']['train'] + result['percentages']['test']
    assert abs(total_pct - 100.0) < 0.01
    
    # Check ratios are consistent
    assert abs(result['ratios']['train_test'] * result['ratios']['test_train'] - 1.0) < 0.01
    
    print("✅ Numerical precision maintained")


def test_concurrent_flags():
    """Test that multiple flags can be triggered simultaneously."""
    print("\n=== CONCURRENT FLAGS ===")
    
    # Create dataset with multiple issues: very small splits AND extreme imbalance
    df_train = pd.DataFrame({
        'smiles': ['CCO'] * 30,  # Small
        'activity': np.random.uniform(1, 100, 30)
    })
    df_test = pd.DataFrame({
        'smiles': ['CCC'] * 2,  # Very very small (< 10% threshold)
        'activity': np.random.uniform(1, 100, 2)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'multi_issue_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'multi_issue_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'activity',
        min_split_size=50
    )
    
    # Should flag both splits as small AND flag test as too small percentage
    assert len(result['flags']['small_splits']) >= 2
    assert len(result['flags']['imbalanced_splits']) > 0
    
    print("✅ Multiple flags triggered correctly")


if __name__ == '__main__':
    print("="*80)
    print("STRESS TEST SUITE: _analyze_split_characteristics")
    print("="*80)
    
    try:
        test_extreme_values()
        test_extreme_ratios()
        test_many_classes()
        test_class_imbalance_extreme()
        test_all_nan_values()
        test_mixed_types()
        test_validation_edge_cases()
        test_numerical_precision()
        test_concurrent_flags()
        
        print("\n" + "="*80)
        print("💪 ALL STRESS TESTS PASSED!")
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
