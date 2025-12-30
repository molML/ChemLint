"""
Real-world integration test for _analyze_split_characteristics.

Demonstrates typical usage patterns and validates against realistic datasets.
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


def print_result(name, result):
    """Pretty print results."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name}")
    print('='*60)
    print(f"Task Type: {result['task_type']}")
    print(f"\nSizes: {result['sizes']}")
    print(f"Percentages: {result['percentages']}")
    print(f"Ratios: {result['ratios']}")
    print(f"\nFlags:")
    print(f"  Empty: {result['flags']['empty_splits']}")
    print(f"  Small: {result['flags']['small_splits']}")
    print(f"  Imbalanced: {result['flags']['imbalanced_splits']}")
    
    if result['task_type'] == 'classification':
        print(f"\nClass Distribution:")
        for split_name, dist in result['class_distribution'].items():
            if isinstance(dist, dict) and 'n_classes' in dist:
                print(f"  {split_name}: {dist['n_classes']} classes, {dist['n_samples']} samples")
                print(f"    Proportions: {dist['proportions']}")
    else:
        print(f"\nValue Distribution:")
        for split_name, dist in result['value_distribution'].items():
            if isinstance(dist, dict) and 'mean' in dist:
                print(f"  {split_name}: mean={dist['mean']:.2f}, std={dist['std']:.2f}, "
                      f"range=[{dist['min']:.2f}, {dist['max']:.2f}]")


def test_good_split():
    """Test a well-balanced, properly sized split (best practice)."""
    print("\n" + "="*80)
    print("TEST 1: WELL-BALANCED SPLIT (BEST PRACTICE)")
    print("="*80)
    
    np.random.seed(42)
    
    # Typical ML split: 80% train, 20% test, 100+ molecules
    df_train = pd.DataFrame({
        'smiles': [f'C{i}' for i in range(800)],
        'ic50': np.random.lognormal(mean=5, sigma=1.5, size=800)
    })
    
    df_test = pd.DataFrame({
        'smiles': [f'N{i}' for i in range(200)],
        'ic50': np.random.lognormal(mean=5, sigma=1.5, size=200)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'good_split_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'good_split_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'ic50'
    )
    
    print_result("Good Split (80/20, n=1000)", result)
    
    # Validate expectations
    assert result['sizes']['train'] == 800
    assert result['sizes']['test'] == 200
    assert result['task_type'] == 'regression'
    assert len(result['flags']['empty_splits']) == 0
    assert len(result['flags']['small_splits']) == 0
    print("\n✅ No issues detected - this is a well-balanced split!")


def test_problematic_split():
    """Test a problematic split with multiple issues."""
    print("\n" + "="*80)
    print("TEST 2: PROBLEMATIC SPLIT (MULTIPLE ISSUES)")
    print("="*80)
    
    np.random.seed(42)
    
    # Issues: 98/2 split, small test set, imbalanced classes
    df_train = pd.DataFrame({
        'smiles': [f'C{i}' for i in range(980)],
        'active': [1] * 784 + [0] * 196  # 80/20 class imbalance
    })
    
    df_test = pd.DataFrame({
        'smiles': [f'N{i}' for i in range(20)],
        'active': [1] * 10 + [0] * 10  # Balanced in test
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'bad_split_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'bad_split_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'active'
    )
    
    print_result("Problematic Split (98/2, small test)", result)
    
    # Validate issues are detected
    assert result['sizes']['train'] == 980
    assert result['sizes']['test'] == 20
    assert len(result['flags']['small_splits']) > 0  # Test set too small
    assert len(result['flags']['imbalanced_splits']) > 0  # Split ratio off
    print("\n⚠️  Multiple issues detected - this split needs attention!")


def test_three_way_split():
    """Test a typical train/val/test split."""
    print("\n" + "="*80)
    print("TEST 3: THREE-WAY SPLIT (TRAIN/VAL/TEST)")
    print("="*80)
    
    np.random.seed(42)
    
    # 70/15/15 split for binary classification
    df_train = pd.DataFrame({
        'smiles': [f'C{i}' for i in range(700)],
        'class': np.random.choice([0, 1], size=700)
    })
    
    df_val = pd.DataFrame({
        'smiles': [f'N{i}' for i in range(150)],
        'class': np.random.choice([0, 1], size=150)
    })
    
    df_test = pd.DataFrame({
        'smiles': [f'O{i}' for i in range(150)],
        'class': np.random.choice([0, 1], size=150)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'threeway_train', 'test', 'csv')
    val_file = _store_resource(df_val, MANIFEST_PATH, 'threeway_val', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'threeway_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, val_file, MANIFEST_PATH, 'smiles', 'class'
    )
    
    print_result("Three-way Split (70/15/15)", result)
    
    # Validate three-way split
    assert result['sizes']['train'] == 700
    assert result['sizes']['val'] == 150
    assert result['sizes']['test'] == 150
    assert 'train_val' in result['ratios']
    assert 'test_val' in result['ratios']
    assert result['task_type'] == 'classification'
    print("\n✅ Three-way split properly analyzed!")


def test_real_world_drug_discovery():
    """Simulate real drug discovery dataset characteristics."""
    print("\n" + "="*80)
    print("TEST 4: REALISTIC DRUG DISCOVERY DATASET")
    print("="*80)
    
    np.random.seed(42)
    
    # Realistic scenario: 
    # - ~1500 compounds total
    # - IC50 values in nanomolar range
    # - Log-normal distribution (typical for bioactivity)
    # - 75/25 split
    
    df_train = pd.DataFrame({
        'smiles': [f'CC(C)Oc1ccc{i}' for i in range(1125)],
        'ic50_nm': np.random.lognormal(mean=np.log(100), sigma=2, size=1125)
    })
    
    df_test = pd.DataFrame({
        'smiles': [f'CN(C)c1ccc{i}' for i in range(375)],
        'ic50_nm': np.random.lognormal(mean=np.log(100), sigma=2, size=375)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'drug_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'drug_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'ic50_nm'
    )
    
    print_result("Drug Discovery Dataset (n=1500, IC50)", result)
    
    # Validate realistic characteristics
    assert result['sizes']['total'] == 1500
    assert result['task_type'] == 'regression'
    assert result['value_distribution']['train']['mean'] > 0
    assert result['value_distribution']['test']['mean'] > 0
    print("\n✅ Realistic drug discovery dataset analyzed!")


def test_hts_binary_classification():
    """Simulate high-throughput screening binary classification."""
    print("\n" + "="*80)
    print("TEST 5: HIGH-THROUGHPUT SCREENING (HTS)")
    print("="*80)
    
    np.random.seed(42)
    
    # HTS: large dataset, binary outcome, typically imbalanced (few actives)
    n_train = 8000
    n_test = 2000
    hit_rate = 0.05  # 5% hit rate (typical for HTS)
    
    df_train = pd.DataFrame({
        'smiles': [f'Compound_{i:06d}' for i in range(n_train)],
        'active': np.random.choice([0, 1], size=n_train, p=[1-hit_rate, hit_rate])
    })
    
    df_test = pd.DataFrame({
        'smiles': [f'Compound_{i:06d}' for i in range(n_train, n_train + n_test)],
        'active': np.random.choice([0, 1], size=n_test, p=[1-hit_rate, hit_rate])
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'hts_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'hts_test', 'test', 'csv')
    
    result = _analyze_split_characteristics(
        train_file, test_file, None, MANIFEST_PATH, 'smiles', 'active'
    )
    
    print_result("HTS Dataset (n=10k, 5% hit rate)", result)
    
    # Validate HTS characteristics
    assert result['sizes']['total'] == 10000
    assert result['task_type'] == 'classification'
    assert result['class_distribution']['train']['n_classes'] == 2
    
    # Check class imbalance is captured
    train_props = result['class_distribution']['train']['proportions']
    assert min(train_props.values()) < 0.1  # Minority class < 10%
    print("\n✅ HTS imbalanced classification analyzed!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("REAL-WORLD INTEGRATION TESTS")
    print("="*80)
    
    try:
        test_good_split()
        test_problematic_split()
        test_three_way_split()
        test_real_world_drug_discovery()
        test_hts_binary_classification()
        
        print("\n" + "="*80)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("="*80)
        print("\n_analyze_split_characteristics() is ready for production use!")
        print("Tested on realistic drug discovery and HTS scenarios.")
        
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
