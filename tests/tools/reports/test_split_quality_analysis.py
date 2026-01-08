"""Pytest tests for tools/reports/data_splitting.py

Focused tests for analyze_split_quality() function using existing dummy datasets.
Tests the main orchestrator function which wraps all helper functions.

Optimized for speed:
- Module-scoped fixtures that cache analysis results
- Combined tests to reduce redundant analysis runs
- Uses only random splits (faster, still comprehensive coverage)
"""

import pytest
import pandas as pd
from pathlib import Path

from molml_mcp.tools.reports.data_splitting import analyze_split_quality
from molml_mcp.infrastructure.resources import _store_resource


# Test data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RANDOM_TRAIN = DATA_DIR / "dummy_small_random_train_split.csv"
RANDOM_TEST = DATA_DIR / "dummy_small_random_test_split.csv"


# ============================================================================
# FIXTURES - Run analysis once and cache results
# ============================================================================

@pytest.fixture(scope="module")
def random_splits(session_workdir):
    """Load and store random splits."""
    manifest_path = str(session_workdir / "test_manifest.json")
    
    # Load data
    train_df = pd.read_csv(RANDOM_TRAIN)
    test_df = pd.read_csv(RANDOM_TEST)
    
    # Store in manifest
    train_file = _store_resource(
        train_df, 
        manifest_path, 
        "random_train", 
        "Random train split", 
        "csv"
    )
    test_file = _store_resource(
        test_df, 
        manifest_path, 
        "random_test", 
        "Random test split", 
        "csv"
    )
    
    return {
        'train': train_file,
        'test': test_file,
        'manifest': manifest_path,
        'train_df': train_df,
        'test_df': test_df
    }


@pytest.fixture(scope="module")
def random_regression_result(random_splits):
    """Run analysis once on random splits with regression labels - cache result."""
    return analyze_split_quality(
        train_path=random_splits['train'],
        test_path=random_splits['test'],
        val_path=None,
        project_manifest_path=random_splits['manifest'],
        smiles_col='smiles',
        label_col='exp_mean [nM]',
        output_filename='random_regression_cached',
        explanation='Cached random regression analysis'
    )


@pytest.fixture(scope="module")
def random_classification_result(random_splits):
    """Run analysis once on random splits with classification labels - cache result."""
    return analyze_split_quality(
        train_path=random_splits['train'],
        test_path=random_splits['test'],
        val_path=None,
        project_manifest_path=random_splits['manifest'],
        smiles_col='smiles',
        label_col='class',
        output_filename='random_classification_cached',
        explanation='Cached random classification analysis'
    )


# ============================================================================
# TESTS - Using cached results for speed
# ============================================================================

@pytest.mark.slow
def test_regression_comprehensive(random_regression_result):
    """Comprehensive test with regression labels (reuses cached result)."""
    result = random_regression_result
    
    # Basic structure
    assert 'output_filename' in result
    assert 'overall_severity' in result
    assert 'severity_summary' in result
    assert 'n_checks_performed' in result
    assert 'execution_time_seconds' in result
    assert 'issues_found' in result
    
    # Should perform 8 checks
    assert result['n_checks_performed'] == 8
    
    # Overall severity should be valid
    assert result['overall_severity'] in ['OK', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    # Severity summary
    severity_summary = result['severity_summary']
    assert all(k in severity_summary for k in ['OK', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
    assert all(isinstance(v, int) for v in severity_summary.values())
    total_count = sum(severity_summary.values())
    assert 7 <= total_count <= 8
    
    # Issues structure
    issues = result['issues_found']
    assert 'exact_duplicates' in issues
    assert 'high_similarity_pairs' in issues
    assert 'activity_cliffs' in issues
    assert 'scaffold_overlap_pct' in issues
    assert 'stereoisomer_pairs' in issues
    assert 'tautomer_pairs' in issues
    assert 'significant_property_diffs' in issues
    assert 'activity_distribution_different' in issues
    assert 'unique_functional_groups_test' in issues
    
    # Type validation
    assert isinstance(issues['exact_duplicates'], (int, float))
    assert isinstance(issues['high_similarity_pairs'], (int, float))
    assert isinstance(issues['scaffold_overlap_pct'], (int, float))
    assert isinstance(issues['activity_distribution_different'], bool)
    
    # Execution time
    assert result['execution_time_seconds'] > 0
    assert result['execution_time_seconds'] < 60
    
    # Output filename
    assert result['output_filename'].endswith('.json')
    assert '_' in result['output_filename']


@pytest.mark.slow
def test_classification_comprehensive(random_classification_result):
    """Comprehensive test with classification labels (reuses cached result)."""
    result = random_classification_result
    
    # Basic structure
    assert result['n_checks_performed'] == 8
    assert result['overall_severity'] in ['OK', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    # Issues present
    issues = result['issues_found']
    assert isinstance(issues['exact_duplicates'], (int, float))
    assert isinstance(issues['activity_distribution_different'], bool)


@pytest.mark.slow
def test_parameter_variations(random_splits):
    """Test various parameter combinations in a single analysis."""
    result = analyze_split_quality(
        train_path=random_splits['train'],
        test_path=random_splits['test'],
        val_path=None,
        project_manifest_path=random_splits['manifest'],
        smiles_col='smiles',
        label_col='exp_mean [nM]',
        output_filename='parameter_variations',
        explanation='Test parameter variations',
        similarity_threshold=0.85,
        min_split_size=20,
        imbalance_threshold=0.1,
        alpha=0.05
    )
    
    # Should complete with all checks
    assert result['n_checks_performed'] == 8
    assert result['overall_severity'] in ['OK', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']


