"""
Tests for scaffold report generation functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from chemlint.tools.reports.scaffold_analysis import (
    _calculate_gini_coefficient,
    _calculate_shannon_entropy,
    _calculate_scaffold_similarity,
    _perform_enrichment_analysis,
    scaffold_analysis
)
from chemlint.infrastructure.resources import _store_resource, _load_resource


# ============================================================================
# Helper Functions Tests
# ============================================================================

def test_calculate_diversity_metrics():
    """Test Gini coefficient and Shannon entropy calculations."""
    # Gini coefficient tests
    assert _calculate_gini_coefficient([10, 10, 10, 10]) <= 0.05  # Equal distribution
    assert _calculate_gini_coefficient([100, 0, 0, 0]) >= 0.7  # Extreme inequality
    assert 0.2 < _calculate_gini_coefficient([50, 30, 15, 5]) < 0.7  # Moderate
    assert _calculate_gini_coefficient([]) == 0.0  # Empty
    
    # Shannon entropy tests
    assert _calculate_shannon_entropy([10, 10, 10, 10]) == pytest.approx(2.0, abs=0.01)  # Uniform
    assert _calculate_shannon_entropy([100]) == 0.0  # Single category
    assert _calculate_shannon_entropy([90, 5, 3, 2]) < 2.0  # Skewed
    assert _calculate_shannon_entropy([]) == 0.0  # Empty


def test_calculate_scaffold_similarity():
    """Test scaffold similarity calculation with various cases."""
    scaffold = "c1ccccc1"  # Benzene
    all_scaffolds = ["c1ccccc1", "c1ccc2ccccc2c1", "C1CCCCC1"]
    
    # Valid SMILES should return similarity between 0 and 1
    sim = _calculate_scaffold_similarity(scaffold, all_scaffolds)
    assert 0.0 <= sim <= 1.0 and not np.isnan(sim)
    
    # Invalid SMILES should return NaN
    assert np.isnan(_calculate_scaffold_similarity("invalid", all_scaffolds))
    
    # Single scaffold (only self) should return NaN
    assert np.isnan(_calculate_scaffold_similarity(scaffold, [scaffold]))


# ============================================================================
# Enrichment Analysis Tests
# ============================================================================

def test_perform_enrichment_analysis_classification():
    """Test enrichment analysis for classification with privileged/inactive scaffolds."""
    # Create synthetic data with distinct scaffold activities
    df = pd.DataFrame({
        'scaffold': ['A'] * 20 + ['B'] * 20 + ['C'] * 10,
        'activity': [1] * 18 + [0] * 2 + [1] * 2 + [0] * 18 + [1] * 5 + [0] * 5
    })
    
    result = _perform_enrichment_analysis(df, 'scaffold', 'activity', 'classification')
    
    assert 'privileged_scaffolds' in result
    assert 'inactive_scaffolds' in result
    assert 'overall_stats' in result
    
    # Scaffold A (90% active) should be privileged, B (10% active) should be inactive
    priv_scaffolds = [s['scaffold'] for s in result['privileged_scaffolds']]
    inact_scaffolds = [s['scaffold'] for s in result['inactive_scaffolds']]
    assert 'A' in priv_scaffolds or len(result['privileged_scaffolds']) >= 1
    assert 'B' in inact_scaffolds or len(result['inactive_scaffolds']) >= 1


def test_perform_enrichment_analysis_regression():
    """Test enrichment analysis for regression with high/low activity scaffolds."""
    df = pd.DataFrame({
        'scaffold': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
        'activity': [8.0, 9.0, 8.5, 9.5, 8.0, 9.0, 8.5, 9.0, 8.0, 9.0] +
                   [3.0, 4.0, 3.5, 4.5, 3.0, 4.0, 3.5, 4.0, 3.0, 4.0] +
                   [1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.0, 1.0, 2.0]
    })
    
    result = _perform_enrichment_analysis(df, 'scaffold', 'activity', 'regression')
    
    assert 'overall_stats' in result
    assert 'mean' in result['overall_stats']
    
    # Scaffold A should be high-activity, C should be low-activity
    if len(result['privileged_scaffolds']) > 0:
        assert result['privileged_scaffolds'][0]['mean'] > 7.0
    if len(result['inactive_scaffolds']) > 0:
        assert result['inactive_scaffolds'][0]['mean'] < 3.0


def test_perform_enrichment_analysis_edge_cases():
    """Test enrichment analysis handles edge cases (NaN scaffolds, rare scaffolds)."""
    # Test with NaN scaffolds
    df_nan = pd.DataFrame({
        'scaffold': ['A'] * 10 + [np.nan] * 5 + ['B'] * 10,
        'activity': [1] * 10 + [0] * 5 + [1] * 10
    })
    result = _perform_enrichment_analysis(df_nan, 'scaffold', 'activity', 'classification')
    assert result['overall_stats']['total'] == 20  # Should exclude NaN
    
    # Test rare scaffolds are excluded (< 5 molecules)
    df_rare = pd.DataFrame({
        'scaffold': ['A'] * 3 + ['B'] * 10,
        'activity': [1] * 3 + [1] * 10
    })
    result = _perform_enrichment_analysis(df_rare, 'scaffold', 'activity', 'classification')
    all_scaffolds = ([s['scaffold'] for s in result['privileged_scaffolds']] +
                    [s['scaffold'] for s in result['inactive_scaffolds']])
    assert 'A' not in all_scaffolds  # Too rare


# ============================================================================
# Main Report Generation Tests
# ============================================================================

@pytest.fixture
def test_dataset(session_workdir):
    """Create a test dataset and store it in the manifest."""
    dummy_path = Path(__file__).parent.parent.parent / 'data' / 'dummy_data_raw_small.csv'
    df = pd.read_csv(dummy_path)
    
    manifest_path = session_workdir / 'test_manifest.json'
    filename = _store_resource(
        df, str(manifest_path), 'test_scaffold_dataset',
        'Test dataset for scaffold report', 'csv'
    )
    
    return filename, str(manifest_path)


@pytest.mark.slow
def test_scaffold_analysis_basic(test_dataset):
    """Test basic scaffold report generation and output structure."""
    filename, manifest_path = test_dataset
    
    result = scaffold_analysis(
        dataset_filename=filename,
        project_manifest_path=manifest_path,
        smiles_column='smiles',
        output_filename='test_scaffold_report',
        scaffold_type='bemis_murcko',
        explanation='Test scaffold report'
    )
    
    # Check return structure
    assert all(k in result for k in ['report_text_filename', 'report_json_filename', 
                                      'dataset_with_scaffolds_filename', 'n_molecules',
                                      'n_unique_scaffolds', 'diversity_ratio', 'summary', 'report'])
    
    # Check values are reasonable
    assert result['n_molecules'] > 0
    assert result['n_unique_scaffolds'] > 0
    assert 0 <= result['diversity_ratio'] <= 1
    assert 0 <= result['gini_coefficient'] <= 1
    assert result['shannon_entropy'] >= 0
    
    # Verify report content
    assert len(result['report']) > 100
    assert all(section in result['report'] for section in 
              ['SCAFFOLD ANALYSIS REPORT', 'OVERVIEW', 'DISTRIBUTION'])


@pytest.mark.slow
def test_scaffold_analysis_with_activity(test_dataset):
    """Test scaffold report with both classification and regression activity."""
    filename, manifest_path = test_dataset
    
    # Test with classification
    result_class = scaffold_analysis(
        dataset_filename=filename, project_manifest_path=manifest_path,
        smiles_column='smiles', output_filename='test_report_class',
        activity_column='class', activity_type='classification',
        explanation='Test with classification'
    )
    assert 'ACTIVITY ENRICHMENT' in result_class['report']
    
    json_report = _load_resource(manifest_path, result_class['report_json_filename'])
    assert json_report['activity_enrichment'] is not None
    assert 'privileged_scaffolds' in json_report['activity_enrichment']
    
    # Test with regression
    result_reg = scaffold_analysis(
        dataset_filename=filename, project_manifest_path=manifest_path,
        smiles_column='smiles', output_filename='test_report_reg',
        activity_column='exp_mean [nM]', activity_type='regression',
        explanation='Test with regression'
    )
    assert 'ACTIVITY ENRICHMENT' in result_reg['report']
    
    json_report_reg = _load_resource(manifest_path, result_reg['report_json_filename'])
    assert 'mean' in json_report_reg['activity_enrichment']['overall_stats']


@pytest.mark.slow
def test_scaffold_analysis_scaffold_types(test_dataset):
    """Test report generation with different scaffold types."""
    filename, manifest_path = test_dataset
    
    for scaffold_type in ['bemis_murcko', 'generic', 'cyclic_skeleton']:
        result = scaffold_analysis(
            dataset_filename=filename, project_manifest_path=manifest_path,
            smiles_column='smiles', output_filename=f'test_report_{scaffold_type}',
            scaffold_type=scaffold_type, explanation=f'Test {scaffold_type}'
        )
        assert result['n_unique_scaffolds'] > 0
        assert 'SCAFFOLD ANALYSIS REPORT' in result['report']


@pytest.mark.slow
def test_scaffold_analysis_outputs_and_structure(test_dataset):
    """Test report outputs (dataset, JSON structure, diversity metrics)."""
    filename, manifest_path = test_dataset
    
    result = scaffold_analysis(
        dataset_filename=filename, project_manifest_path=manifest_path,
        smiles_column='smiles', output_filename='test_report_structure',
        outlier_threshold=0.3, top_n=5, explanation='Test outputs'
    )
    
    # Check dataset with scaffolds
    df_with_scaffolds = _load_resource(manifest_path, result['dataset_with_scaffolds_filename'])
    assert 'scaffold' in df_with_scaffolds.columns
    assert 'scaffold_comment' in df_with_scaffolds.columns
    assert df_with_scaffolds['scaffold'].notna().sum() > 0
    
    # Check JSON structure
    json_report = _load_resource(manifest_path, result['report_json_filename'])
    assert 'overview' in json_report
    assert all(k in json_report['overview'] for k in 
              ['total_molecules', 'unique_scaffolds', 'diversity_ratio'])
    assert 'distribution' in json_report
    assert all(k in json_report['distribution'] for k in 
              ['gini_coefficient', 'shannon_entropy', 'singleton_scaffolds'])
    assert 'top_scaffolds' in json_report and isinstance(json_report['top_scaffolds'], list)
    assert len(json_report['top_scaffolds']) <= 5  # Respects top_n parameter
    
    # Check structural outliers
    if len(json_report['structural_outliers']) > 0:
        outlier = json_report['structural_outliers'][0]
        assert all(k in outlier for k in ['scaffold', 'count', 'avg_similarity'])


def test_scaffold_analysis_error_handling(test_dataset):
    """Test error handling for invalid inputs."""
    filename, manifest_path = test_dataset
    
    # Missing SMILES column
    with pytest.raises(ValueError, match="SMILES column.*not found"):
        scaffold_analysis(
            dataset_filename=filename, project_manifest_path=manifest_path,
            smiles_column='nonexistent', output_filename='test_error',
            explanation='Test error'
        )
    
    # Missing activity column
    with pytest.raises(ValueError, match="Activity column.*not found"):
        scaffold_analysis(
            dataset_filename=filename, project_manifest_path=manifest_path,
            smiles_column='smiles', output_filename='test_error',
            activity_column='nonexistent', activity_type='classification',
            explanation='Test error'
        )
    
    # Activity column without type
    with pytest.raises(ValueError, match="activity_type must be specified"):
        scaffold_analysis(
            dataset_filename=filename, project_manifest_path=manifest_path,
            smiles_column='smiles', output_filename='test_error',
            activity_column='class', explanation='Test error'
        )
    
    # Invalid activity type
    with pytest.raises(ValueError, match="activity_type must be"):
        scaffold_analysis(
            dataset_filename=filename, project_manifest_path=manifest_path,
            smiles_column='smiles', output_filename='test_error',
            activity_column='class', activity_type='invalid',
            explanation='Test error'
        )

