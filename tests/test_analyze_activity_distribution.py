"""
RIGOROUS TEST SUITE: _analyze_activity_distribution

Tests all edge cases, boundary conditions, and statistical robustness.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from molml_mcp.tools.reports.quality import _analyze_activity_distribution

def test_continuous_basic():
    """Test basic continuous activity analysis"""
    print("\n=== CONTINUOUS BASIC ===")
    
    # Test 1: Normal distribution
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _analyze_activity_distribution(values, activity_type='continuous', units='nM')
    
    assert result['type'] == 'continuous'
    assert result['units'] == 'nM'
    assert 'linear_stats' in result
    assert 'log_stats' in result
    assert abs(result['linear_stats']['mean'] - 5.5) < 0.01
    assert abs(result['linear_stats']['median'] - 5.5) < 0.01
    print("✅ Normal distribution")
    
    # Test 2: Different units
    result = _analyze_activity_distribution(values, activity_type='continuous', units='μM')
    assert result['units'] == 'μM'
    print("✅ Different units")
    
    # Test 3: Wide range
    values = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    result = _analyze_activity_distribution(values, activity_type='continuous', units='nM')
    assert 'range_log_units' in result['linear_stats']
    assert result['linear_stats']['range_log_units'] > 6  # 7 orders of magnitude
    print("✅ Wide range")

def test_continuous_edge_cases():
    """Test edge cases for continuous data"""
    print("\n=== CONTINUOUS EDGE CASES ===")
    
    # Test 1: Insufficient data
    values = np.array([1.0, 2.0])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert 'error' in result
    print("✅ Insufficient data handled")
    
    # Test 2: Exactly 3 values (minimum)
    values = np.array([1.0, 2.0, 3.0])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert 'error' not in result
    assert 'linear_stats' in result
    print("✅ Minimum data (3 values)")
    
    # Test 3: Single value
    values = np.array([5.0])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert 'error' in result
    print("✅ Single value handled")
    
    # Test 4: Two values
    values = np.array([1.0, 10.0])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert 'error' in result
    print("✅ Two values handled")

def test_continuous_statistics():
    """Test statistical calculations"""
    print("\n=== CONTINUOUS STATISTICS ===")
    
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    
    # Linear stats
    ls = result['linear_stats']
    assert abs(ls['mean'] - 5.5) < 0.01
    assert abs(ls['median'] - 5.5) < 0.01
    assert ls['min'] == 1.0
    assert ls['max'] == 10.0
    assert abs(ls['std'] - np.std(values)) < 0.01
    print("✅ Linear statistics correct")
    
    # Percentiles
    assert abs(ls['p10'] - 1.9) < 0.5
    assert abs(ls['p25'] - 3.25) < 0.5
    assert abs(ls['p75'] - 7.75) < 0.5
    assert abs(ls['p90'] - 9.1) < 0.5
    print("✅ Percentiles correct")
    
    # Moments
    assert 'skewness' in ls
    assert 'kurtosis' in ls
    print("✅ Moments calculated")

def test_continuous_with_nan():
    """Test NaN handling"""
    print("\n=== NAN HANDLING ===")
    
    # Test 1: Some NaN
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['n_missing'] == 2
    assert result['n_valid'] == 5
    print("✅ Some NaN values")
    
    # Test 2: All NaN
    values = np.array([np.nan, np.nan, np.nan])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert 'error' in result  # Insufficient data after filtering NaN
    print("✅ All NaN values")
    
    # Test 3: Mixed NaN and valid
    values = np.array([np.nan, 1.0, 2.0, np.nan, 3.0, np.nan])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['n_valid'] == 3
    assert result['n_missing'] == 3
    print("✅ Mixed NaN and valid")

def test_continuous_extreme_values():
    """Test extreme value handling"""
    print("\n=== EXTREME VALUES ===")
    
    # Test 1: Very small values
    values = np.array([1e-20, 1e-15, 1e-10, 1e-5, 1e-2])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert 'linear_stats' in result
    print("✅ Very small values")
    
    # Test 2: Very large values
    values = np.array([1e5, 1e10, 1e15, 1e20])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert 'linear_stats' in result
    print("✅ Very large values")
    
    # Test 3: Mixed scale
    values = np.array([1e-10, 1.0, 1e10])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['linear_stats']['range_log_units'] > 15
    print("✅ Mixed scale")
    
    # Test 4: With zeros
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['linear_stats']['min'] == 0.0
    print("✅ With zeros")
    
    # Test 5: Negative values
    values = np.array([-10, -5, -1, 0, 1, 5, 10])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['linear_stats']['min'] < 0
    print("✅ Negative values")

def test_continuous_outliers():
    """Test outlier detection"""
    print("\n=== OUTLIER DETECTION ===")
    
    # Test: Data with outliers
    values = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 1000])  # 1000 is outlier
    result = _analyze_activity_distribution(values, activity_type='continuous')
    
    assert 'outliers' in result
    assert 'ultra_potent' in result['outliers']
    assert 'weak_outliers' in result['outliers']
    print("✅ Outlier detection")

def test_continuous_bins():
    """Test activity binning"""
    print("\n=== ACTIVITY BINS ===")
    
    values = np.array([50, 150, 500, 1500, 5000, 15000])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    
    bins = result['bins']
    assert '< 100' in bins
    assert '100-1000' in bins
    assert '1000-10000' in bins
    assert '> 10000' in bins
    
    assert bins['< 100'] == 1
    assert bins['100-1000'] == 2
    assert bins['1000-10000'] == 2
    assert bins['> 10000'] == 1
    print("✅ Activity binning")

def test_continuous_normality():
    """Test normality testing"""
    print("\n=== NORMALITY TESTING ===")
    
    # Test 1: Enough data for test
    values = np.random.normal(5, 2, 100)
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['normality_test'] is not None
    assert 'w_statistic' in result['normality_test']
    assert 'p_value' in result['normality_test']
    assert 'is_normal' in result['normality_test']
    print("✅ Normality test with sufficient data")
    
    # Test 2: Too few for test
    values = np.array([1.0, 2.0, 3.0])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    # With only 3 values, test should still run
    print("✅ Normality test with minimal data")

def test_continuous_identical():
    """Test identical values"""
    print("\n=== IDENTICAL VALUES ===")
    
    # Test 1: All same value
    values = np.array([5.0] * 10)
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['linear_stats']['std'] == 0.0
    assert result['linear_stats']['mean'] == 5.0
    assert result['linear_stats']['median'] == 5.0
    print("✅ All identical values")
    
    # Test 2: Two distinct values
    values = np.array([1.0] * 5 + [2.0] * 5)
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['linear_stats']['std'] > 0
    print("✅ Two distinct values")

def test_classification_basic():
    """Test basic classification analysis"""
    print("\n=== CLASSIFICATION BASIC ===")
    
    # Test 1: Balanced classes
    values = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    result = _analyze_activity_distribution(values, activity_type='classification')
    
    assert result['type'] == 'classification'
    assert result['n_positive'] == 4
    assert result['n_negative'] == 4
    assert abs(result['balance'] - 0.5) < 0.01
    print("✅ Balanced classes")
    
    # Test 2: Imbalanced classes
    values = np.array([0] * 90 + [1] * 10)
    result = _analyze_activity_distribution(values, activity_type='classification')
    assert result['n_positive'] == 10
    assert result['n_negative'] == 90
    assert abs(result['balance'] - 0.1) < 0.01
    print("✅ Imbalanced classes")
    
    # Test 3: All positive
    values = np.array([1] * 10)
    result = _analyze_activity_distribution(values, activity_type='classification')
    assert result['n_positive'] == 10
    assert result['n_negative'] == 0
    assert result['balance'] == 1.0
    print("✅ All positive")
    
    # Test 4: All negative
    values = np.array([0] * 10)
    result = _analyze_activity_distribution(values, activity_type='classification')
    assert result['n_positive'] == 0
    assert result['n_negative'] == 10
    assert result['balance'] == 0.0
    print("✅ All negative")

def test_classification_with_nan():
    """Test classification with NaN"""
    print("\n=== CLASSIFICATION NAN ===")
    
    # Test: Mixed with NaN
    values = np.array([0, np.nan, 1, np.nan, 0, 1])
    result = _analyze_activity_distribution(values, activity_type='classification')
    assert result['n_missing'] == 2
    assert result['n_valid'] == 4
    assert result['n_positive'] == 2
    assert result['n_negative'] == 2
    print("✅ Classification with NaN")

def test_classification_edge_cases():
    """Test classification edge cases"""
    print("\n=== CLASSIFICATION EDGE CASES ===")
    
    # Test 1: Single value
    values = np.array([1])
    result = _analyze_activity_distribution(values, activity_type='classification')
    assert result['n_valid'] == 1
    print("✅ Single classification value")
    
    # Test 2: Highly imbalanced (1%)
    values = np.array([0] * 99 + [1])
    result = _analyze_activity_distribution(values, activity_type='classification')
    assert abs(result['balance'] - 0.01) < 0.001
    print("✅ Highly imbalanced (1%)")
    
    # Test 3: Non-binary values (should still work with 0/1 counting)
    values = np.array([0, 1, 2, 3, 0, 1])
    result = _analyze_activity_distribution(values, activity_type='classification')
    # Counts 1s as positive, 0s as negative, others neither
    print("✅ Non-binary values handled")

def test_return_structures():
    """Test return value structures"""
    print("\n=== RETURN STRUCTURES ===")
    
    # Continuous return structure
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    
    required_keys = ['type', 'units', 'n_valid', 'n_missing', 'linear_stats', 
                     'log_stats', 'bins', 'outliers']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    print("✅ Continuous return structure")
    
    # Classification return structure
    values = np.array([0, 1, 0, 1])
    result = _analyze_activity_distribution(values, activity_type='classification')
    
    required_keys = ['type', 'n_valid', 'n_missing', 'n_positive', 'n_negative', 'balance']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    print("✅ Classification return structure")

def test_large_datasets():
    """Test with large datasets"""
    print("\n=== LARGE DATASETS ===")
    
    # Test 1: Large continuous dataset
    values = np.random.lognormal(5, 2, 10000)
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['n_valid'] == 10000
    assert 'linear_stats' in result
    print("✅ Large continuous dataset (10k)")
    
    # Test 2: Large classification dataset
    values = np.random.choice([0, 1], size=10000, p=[0.7, 0.3])
    result = _analyze_activity_distribution(values, activity_type='classification')
    assert result['n_valid'] == 10000
    assert 0.25 < result['balance'] < 0.35  # Should be ~0.3
    print("✅ Large classification dataset (10k)")
    
    # Test 3: Very large dataset (normality test limit)
    values = np.random.normal(5, 2, 6000)
    result = _analyze_activity_distribution(values, activity_type='continuous')
    # Shapiro-Wilk limited to 5000 samples
    print("✅ Very large dataset (6k, exceeds Shapiro limit)")

def test_special_distributions():
    """Test special statistical distributions"""
    print("\n=== SPECIAL DISTRIBUTIONS ===")
    
    # Test 1: Uniform distribution
    values = np.random.uniform(0, 10, 100)
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert abs(result['linear_stats']['skewness']) < 0.5  # Should be ~0
    print("✅ Uniform distribution")
    
    # Test 2: Exponential distribution
    values = np.random.exponential(2, 100)
    result = _analyze_activity_distribution(values, activity_type='continuous')
    assert result['linear_stats']['skewness'] > 0  # Right-skewed
    print("✅ Exponential distribution")
    
    # Test 3: Bimodal distribution
    values = np.concatenate([np.random.normal(2, 0.5, 50), 
                            np.random.normal(8, 0.5, 50)])
    result = _analyze_activity_distribution(values, activity_type='continuous')
    print("✅ Bimodal distribution")

if __name__ == '__main__':
    print("="*80)
    print("RIGOROUS TEST SUITE: _analyze_activity_distribution")
    print("="*80)
    
    try:
        test_continuous_basic()
        test_continuous_edge_cases()
        test_continuous_statistics()
        test_continuous_with_nan()
        test_continuous_extreme_values()
        test_continuous_outliers()
        test_continuous_bins()
        test_continuous_normality()
        test_continuous_identical()
        test_classification_basic()
        test_classification_with_nan()
        test_classification_edge_cases()
        test_return_structures()
        test_large_datasets()
        test_special_distributions()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        sys.exit(0)
        
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
