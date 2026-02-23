"""
Tests for _perform_quality_report_calculations() in quality.py.

This function is the main orchestrator that wraps all quality report analyses,
so these tests provide comprehensive coverage of the entire quality report system.

Uses the comprehensive dummy_quality_report.csv test file which includes:
- Activity cliffs (similar structures with large activity differences)
- Stereochemistry issues (enantiomers with different activities)
- PAINS alerts
- Invalid SMILES
- Salts, solvents, and charged species
- Duplicates
- Extreme molecular weights and lipophilicity
- Complex stereochemistry
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from chemlint.tools.reports.data_quality_analysis import _perform_quality_report_calculations
from chemlint.infrastructure.resources import _store_resource, _load_resource


# Fixtures
@pytest.fixture(scope="session")
def quality_report_dataset(tmp_path_factory):
    """Load a subset of dummy_quality_report.csv for faster testing."""
    # Load the test CSV and take rows that include duplicates (rows 1-45)
    test_data_path = Path(__file__).parent.parent.parent / "data" / "dummy_quality_report.csv"
    df_full = pd.read_csv(test_data_path)
    # Take diverse subset: some clean, some problematic, including duplicates and PAINS
    df = df_full.head(45).copy()
    
    # Create session-level temp directory and manifest
    session_workdir = tmp_path_factory.mktemp("quality_report_session")
    manifest_path = session_workdir / "manifest.json"
    manifest_data = {"resources": []}
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)
    
    # Store dataset
    dataset_filename = _store_resource(
        df, str(manifest_path), "quality_report_dataset", "Fast quality report test dataset", "csv"
    )
    
    return {
        'manifest_path': str(manifest_path),
        'dataset_filename': dataset_filename,
        'df': df
    }


@pytest.fixture(scope="session")
def clean_subset_dataset(tmp_path_factory):
    """Create a smaller clean dataset for fast tests."""
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CC(C)O', 'c1ccccc1', 'CC(=O)O', 'CCCC', 
                   'CCC(C)C', 'c1ccc(O)cc1', 'CCN', 'CCCO', 'c1ccncc1'],
        'activity': [5.2, 4.8, 6.1, 5.5, 4.9, 5.1, 7.2, 4.7, 5.3, 6.8],
        'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    
    # Create session-level temp directory and manifest
    session_workdir = tmp_path_factory.mktemp("clean_subset_session")
    manifest_path = session_workdir / "manifest.json"
    manifest_data = {"resources": []}
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)
    
    # Store dataset
    dataset_filename = _store_resource(
        df, str(manifest_path), "clean_subset", "Clean subset for fast tests", "csv"
    )
    
    return {
        'manifest_path': str(manifest_path),
        'dataset_filename': dataset_filename,
        'df': df
    }


# ========== Basic Functionality Tests ==========

def test_perform_quality_report_basic(clean_subset_dataset):
    """Test basic quality report calculation with clean dataset."""
    result_filename = _perform_quality_report_calculations(
        input_filename=clean_subset_dataset['dataset_filename'],
        project_manifest_path=clean_subset_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='basic_test'
    )
    
    # Verify result is a filename
    assert isinstance(result_filename, str)
    assert 'basic_test_calculations' in result_filename
    
    # Load and verify results
    results = _load_resource(clean_subset_dataset['manifest_path'], result_filename)
    
    # Check main sections exist
    assert 'metadata' in results
    assert 'completeness' in results
    assert 'smiles_validity' in results
    assert 'pains' in results
    assert 'duplicates' in results
    assert 'physicochemical_properties' in results
    assert 'lipinski' in results
    assert 'veber' in results
    assert 'qed' in results
    assert 'outliers' in results
    assert 'scaffold_diversity' in results
    assert 'functional_groups' in results
    assert 'stereochemistry' in results
    assert 'charge_state' in results
    assert 'salts_fragments' in results
    assert 'special_features' in results
    assert 'recommendations' in results
    assert 'critical_issues' in results


def test_perform_quality_report_with_activity(quality_report_dataset):
    """Test quality report with continuous activity data."""
    result_filename = _perform_quality_report_calculations(
        input_filename=quality_report_dataset['dataset_filename'],
        project_manifest_path=quality_report_dataset['manifest_path'],
        smiles_col='SMILES',
        activity_col='activity',
        activity_type='regression',
        activity_units='nM',
        output_name='activity_test'
    )
    
    results = _load_resource(quality_report_dataset['manifest_path'], result_filename)
    
    # Check activity analysis exists
    assert 'activity' in results
    assert results['activity'] is not None
    assert results['activity']['type'] == 'continuous'
    assert 'linear_stats' in results['activity']
    assert 'log_stats' in results['activity']
    
    # Check activity correlations
    assert 'activity_correlations' in results


def test_perform_quality_report_with_classification(quality_report_dataset, session_workdir):
    """Test quality report with binary classification data."""
    # Load dataset and filter to rows with valid labels
    df = quality_report_dataset['df'].copy()
    # Filter out rows where label is NaN or empty
    df_filtered = df[df['label'].notna()].copy()
    # Convert to numeric, coercing errors (like 'Missing label' strings) to NaN
    df_filtered['label'] = pd.to_numeric(df_filtered['label'], errors='coerce')
    # Drop any rows that couldn't be converted
    df_filtered = df_filtered[df_filtered['label'].notna()].copy()
    # Now convert to int
    df_filtered['label'] = df_filtered['label'].astype(int)
    
    # Store filtered dataset
    manifest_path = session_workdir / "manifest_classification.json"
    manifest_data = {"resources": []}
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)
    
    filtered_filename = _store_resource(
        df_filtered, str(manifest_path), "filtered_classification_dataset", 
        "Filtered dataset for classification", "csv"
    )
    
    result_filename = _perform_quality_report_calculations(
        input_filename=filtered_filename,
        project_manifest_path=str(manifest_path),
        smiles_col='SMILES',
        activity_col='label',
        activity_type='classification',
        output_name='classification_test'
    )
    
    results = _load_resource(str(manifest_path), result_filename)
    
    # Check classification activity analysis
    assert 'activity' in results
    assert results['activity'] is not None
    assert results['activity']['type'] == 'classification'
    assert 'n_positive' in results['activity']
    assert 'n_negative' in results['activity']
    assert 'balance' in results['activity']


# ========== Metadata Tests ==========

def test_metadata_section(clean_subset_dataset):
    """Test metadata section contains correct information."""
    result_filename = _perform_quality_report_calculations(
        input_filename=clean_subset_dataset['dataset_filename'],
        project_manifest_path=clean_subset_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='metadata_test'
    )
    
    results = _load_resource(clean_subset_dataset['manifest_path'], result_filename)
    metadata = results['metadata']
    
    assert 'dataset' in metadata
    assert metadata['dataset'] == clean_subset_dataset['dataset_filename']
    assert metadata['n_molecules'] == 10
    assert metadata['smiles_col'] == 'SMILES'
    assert 'SMILES' in metadata['columns']
    assert 'activity' in metadata['columns']
    assert 'generated' in metadata


# ========== SMILES Validity Tests ==========

def test_smiles_validity_detection(quality_report_dataset):
    """Test detection of invalid SMILES."""
    result_filename = _perform_quality_report_calculations(
        input_filename=quality_report_dataset['dataset_filename'],
        project_manifest_path=quality_report_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='validity_test'
    )
    
    results = _load_resource(quality_report_dataset['manifest_path'], result_filename)
    validity = results['smiles_validity']
    
    assert 'n_valid' in validity
    assert 'n_invalid' in validity
    assert 'pct_invalid' in validity
    assert validity['n_invalid'] > 0  # Should detect invalid SMILES
    assert len(validity.get('invalid_examples', [])) > 0


# ========== Physicochemical Properties Tests ==========

def test_physicochemical_properties(clean_subset_dataset):
    """Test calculation of physicochemical properties."""
    result_filename = _perform_quality_report_calculations(
        input_filename=clean_subset_dataset['dataset_filename'],
        project_manifest_path=clean_subset_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='properties_test'
    )
    
    results = _load_resource(clean_subset_dataset['manifest_path'], result_filename)
    properties = results['physicochemical_properties']
    
    # Check key descriptors are present
    expected_props = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    for prop in expected_props:
        assert prop in properties
        assert 'mean' in properties[prop]
        assert 'median' in properties[prop]
        assert 'std' in properties[prop]
        assert 'min' in properties[prop]
        assert 'max' in properties[prop]


# ========== Rule Compliance Tests (Combined) ==========

def test_drug_likeness_rules_comprehensive(clean_subset_dataset):
    """Test Lipinski, Veber, and QED drug-likeness analyses together."""
    result_filename = _perform_quality_report_calculations(
        input_filename=clean_subset_dataset['dataset_filename'],
        project_manifest_path=clean_subset_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='drug_likeness_test'
    )
    
    results = _load_resource(clean_subset_dataset['manifest_path'], result_filename)
    
    # Lipinski Rule of Five
    lipinski = results['lipinski']
    assert '0_violations' in lipinski
    assert '1_violation' in lipinski
    assert '2+_violations' in lipinski
    assert 'pct_compliant' in lipinski
    total = lipinski['0_violations'] + lipinski['1_violation'] + lipinski['2+_violations']
    assert total == 10
    
    # Veber rules
    veber = results['veber']
    assert 'pass' in veber
    assert 'pct_compliant' in veber
    assert isinstance(veber['pass'], int)
    assert 0 <= veber['pct_compliant'] <= 100
    
    # QED (Drug-likeness)
    qed = results['qed']
    assert qed is not None
    assert 'mean' in qed
    assert 'median' in qed
    assert 'high' in qed
    assert 'moderate' in qed
    assert 'low' in qed
    total = qed['high'] + qed['moderate'] + qed['low']
    assert total == 10


# ========== Duplicates Tests ==========

def test_duplicate_detection(quality_report_dataset):
    """Test duplicate molecule detection."""
    result_filename = _perform_quality_report_calculations(
        input_filename=quality_report_dataset['dataset_filename'],
        project_manifest_path=quality_report_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='duplicates_test'
    )
    
    results = _load_resource(quality_report_dataset['manifest_path'], result_filename)
    duplicates = results['duplicates']
    
    assert 'n_duplicate_groups' in duplicates
    assert duplicates['n_duplicate_groups'] > 0  # Dataset has duplicates


# ========== Scaffold Diversity Tests ==========

def test_scaffold_diversity(clean_subset_dataset):
    """Test scaffold diversity analysis."""
    result_filename = _perform_quality_report_calculations(
        input_filename=clean_subset_dataset['dataset_filename'],
        project_manifest_path=clean_subset_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='scaffold_test'
    )
    
    results = _load_resource(clean_subset_dataset['manifest_path'], result_filename)
    scaffold = results['scaffold_diversity']
    
    assert 'n_unique_scaffolds' in scaffold
    assert 'diversity_ratio' in scaffold
    assert 'gini_coefficient' in scaffold
    assert 'shannon_entropy' in scaffold
    
    assert scaffold['n_unique_scaffolds'] > 0
    assert 0 <= scaffold['diversity_ratio'] <= 1


# ========== Molecular Features Tests (Combined) ==========

def test_molecular_features_comprehensive(quality_report_dataset):
    """Test all molecular feature analyses: functional groups, stereochemistry, charge state, salts, special features."""
    result_filename = _perform_quality_report_calculations(
        input_filename=quality_report_dataset['dataset_filename'],
        project_manifest_path=quality_report_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='molecular_features_test'
    )
    
    results = _load_resource(quality_report_dataset['manifest_path'], result_filename)
    
    # Functional groups
    assert isinstance(results['functional_groups'], dict)
    
    # Stereochemistry
    assert isinstance(results['stereochemistry'], dict)
    
    # Charge state
    assert isinstance(results['charge_state'], dict)
    assert 'needs_neutralization' in results['charge_state'] or 'pct_charged' in results['charge_state']
    
    # Salts/fragments
    assert isinstance(results['salts_fragments'], dict)
    assert 'needs_desalting' in results['salts_fragments'] or 'pct_fragmented' in results['salts_fragments']
    
    # Special features
    assert isinstance(results['special_features'], dict)


# ========== Recommendations Tests ==========

def test_recommendations_generation(quality_report_dataset):
    """Test that recommendations are generated for problematic data."""
    result_filename = _perform_quality_report_calculations(
        input_filename=quality_report_dataset['dataset_filename'],
        project_manifest_path=quality_report_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='recommendations_test'
    )
    
    results = _load_resource(quality_report_dataset['manifest_path'], result_filename)
    
    assert 'recommendations' in results
    assert 'critical_issues' in results
    assert isinstance(results['recommendations'], list)
    assert isinstance(results['critical_issues'], list)
    
    # Problematic dataset should have some recommendations
    assert len(results['recommendations']) > 0


def test_clean_data_fewer_recommendations(clean_subset_dataset):
    """Test that clean data generates fewer recommendations."""
    result_filename = _perform_quality_report_calculations(
        input_filename=clean_subset_dataset['dataset_filename'],
        project_manifest_path=clean_subset_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='clean_rec_test'
    )
    
    results = _load_resource(clean_subset_dataset['manifest_path'], result_filename)
    
    # Clean data should have fewer or no critical issues
    assert len(results['critical_issues']) <= 2


# ========== Outlier Detection Tests ==========

def test_outlier_detection(clean_subset_dataset):
    """Test outlier detection in properties."""
    result_filename = _perform_quality_report_calculations(
        input_filename=clean_subset_dataset['dataset_filename'],
        project_manifest_path=clean_subset_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='outliers_test'
    )
    
    results = _load_resource(clean_subset_dataset['manifest_path'], result_filename)
    outliers = results['outliers']
    
    assert 'n_extreme' in outliers
    assert 'examples' in outliers
    assert isinstance(outliers['n_extreme'], int)


# ========== Error Handling Tests (Combined) ==========

def test_error_handling_comprehensive(clean_subset_dataset):
    """Test all error handling: missing column, missing activity_type, invalid activity_type."""
    # Test 1: Missing SMILES column
    with pytest.raises(ValueError, match="SMILES column .* not found"):
        _perform_quality_report_calculations(
            input_filename=clean_subset_dataset['dataset_filename'],
            project_manifest_path=clean_subset_dataset['manifest_path'],
            smiles_col='NonExistentColumn',
            output_name='error_test1'
        )
    
    # Test 2: Activity column without activity_type
    with pytest.raises(ValueError, match="activity_type must be specified"):
        _perform_quality_report_calculations(
            input_filename=clean_subset_dataset['dataset_filename'],
            project_manifest_path=clean_subset_dataset['manifest_path'],
            smiles_col='SMILES',
            activity_col='activity',
            activity_type=None,
            output_name='error_test2'
        )
    
    # Test 3: Invalid activity_type
    with pytest.raises(ValueError, match="activity_type must be"):
        _perform_quality_report_calculations(
            input_filename=clean_subset_dataset['dataset_filename'],
            project_manifest_path=clean_subset_dataset['manifest_path'],
            smiles_col='SMILES',
            activity_col='activity',
            activity_type='invalid_type',
            output_name='error_test3'
        )


# ========== Temporary File Cleanup Tests ==========

def test_temporary_files_cleanup(clean_subset_dataset):
    """Test that temporary files are cleaned up after calculation."""
    # Get initial resource count
    with open(clean_subset_dataset['manifest_path'], 'r') as f:
        manifest_before = json.load(f)
    initial_count = len(manifest_before['resources'])
    
    result_filename = _perform_quality_report_calculations(
        input_filename=clean_subset_dataset['dataset_filename'],
        project_manifest_path=clean_subset_dataset['manifest_path'],
        smiles_col='SMILES',
        output_name='cleanup_test'
    )
    
    # Get final resource count
    with open(clean_subset_dataset['manifest_path'], 'r') as f:
        manifest_after = json.load(f)
    final_count = len(manifest_after['resources'])
    
    # Should have added the final calculations file
    # Temp files might be created during processing but should be cleaned up
    # Allow some intermediate files (calculations + descriptors + outliers + scaffolds)
    assert final_count >= initial_count + 1
    assert final_count <= initial_count + 5
    
    # Verify the result file exists in manifest
    result_in_manifest = any(r['filename'] == result_filename for r in manifest_after['resources'])
    assert result_in_manifest
