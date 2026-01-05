"""
Test predict_ml_model with CV models (multiple folds).
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molml_mcp.tools.ml.training import train_ml_models_cv
from molml_mcp.tools.ml.evaluation import predict_ml_model
from molml_mcp.infrastructure.resources import _store_resource, _load_resource

TEST_MANIFEST = Path(__file__).parent / "data" / "test_manifest.json"


def test_predict_single_model():
    """Test prediction with a single model."""
    # Create simple dataset
    df = pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC', 'CC'],
        'label': [1, 0, 1, 0]
    })
    features = {smi: [float(len(smi))] for smi in df['smiles']}
    
    data_file = _store_resource(df, str(TEST_MANIFEST), 'pred_single_data', 'Data', 'csv')
    features_file = _store_resource(features, str(TEST_MANIFEST), 'pred_single_features', 'Features', 'json')
    
    # Train single model (montecarlo with 1 split = single model)
    train_result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='single_model',
        explanation='Single model',
        model_algorithm='random_forest_classifier',
        cv_strategy='montecarlo',
        n_folds=1,
        val_size=0.25,
        random_state=42
    )
    
    # Predict on same data
    pred_result = predict_ml_model(
        ml_model_filename=train_result['output_filename'],
        test_input_filename=data_file,
        test_feature_vectors_filename=features_file,
        test_smiles_column='smiles',
        predict_column_name='prediction',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='predictions_single',
        explanation='Single model predictions'
    )
    
    # Verify output structure
    assert pred_result['n_models'] == 1
    assert pred_result['n_predictions'] == 4
    assert 'prediction' in pred_result['columns']
    
    # Load and check predictions
    pred_df = _load_resource(str(TEST_MANIFEST), pred_result['output_filename'])
    assert 'prediction' in pred_df.columns
    assert len(pred_df) == 4


def test_predict_cv_models_classification():
    """Test prediction with multiple CV models (classification)."""
    # Create dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'smiles': [f'C{i}' for i in range(20)],
        'label': np.random.randint(0, 2, 20)
    })
    features = {smi: np.random.randn(5).tolist() for smi in df['smiles']}
    
    data_file = _store_resource(df, str(TEST_MANIFEST), 'pred_cv_data', 'Data', 'csv')
    features_file = _store_resource(features, str(TEST_MANIFEST), 'pred_cv_features', 'Features', 'json')
    
    # Train CV models (3 folds)
    train_result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='cv_models_classification',
        explanation='CV models',
        model_algorithm='random_forest_classifier',
        cv_strategy='kfold',
        n_folds=3,
        random_state=42
    )
    
    # Predict on same data
    pred_result = predict_ml_model(
        ml_model_filename=train_result['output_filename'],
        test_input_filename=data_file,
        test_feature_vectors_filename=features_file,
        test_smiles_column='smiles',
        predict_column_name='pred',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='predictions_cv_class',
        explanation='CV predictions'
    )
    
    # Verify output structure
    assert pred_result['n_models'] == 3
    assert pred_result['n_predictions'] == 20
    
    # Load and check predictions
    pred_df = _load_resource(str(TEST_MANIFEST), pred_result['output_filename'])
    
    # Check per-fold columns
    assert 'pred_1' in pred_df.columns
    assert 'pred_2' in pred_df.columns
    assert 'pred_3' in pred_df.columns
    
    # Check aggregate columns
    assert 'pred_mean' in pred_df.columns
    assert 'pred_std' in pred_df.columns
    assert 'pred_entropy' in pred_df.columns
    
    # Verify data types and ranges
    assert len(pred_df) == 20
    assert pred_df['pred_mean'].dtype in [np.float64, np.float32]
    assert pred_df['pred_std'].dtype in [np.float64, np.float32]
    assert pred_df['pred_entropy'].dtype in [np.float64, np.float32]
    
    # Entropy should be >= 0 and <= log2(n_folds)
    assert (pred_df['pred_entropy'] >= 0).all()
    assert (pred_df['pred_entropy'] <= np.log2(3) + 0.01).all()  # Small tolerance for float precision


def test_predict_cv_models_regression():
    """Test prediction with multiple CV models (regression)."""
    np.random.seed(42)
    df = pd.DataFrame({
        'smiles': [f'C{i}' for i in range(15)],
        'label': np.random.randn(15)
    })
    features = {smi: np.random.randn(3).tolist() for smi in df['smiles']}
    
    data_file = _store_resource(df, str(TEST_MANIFEST), 'pred_cv_reg_data', 'Data', 'csv')
    features_file = _store_resource(features, str(TEST_MANIFEST), 'pred_cv_reg_features', 'Features', 'json')
    
    # Train CV models (2 folds)
    train_result = train_ml_models_cv(
        input_filename=data_file,
        feature_vectors_filename=features_file,
        smiles_column='smiles',
        label_column='label',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='cv_models_regression',
        explanation='CV regression models',
        model_algorithm='random_forest_regressor',
        cv_strategy='kfold',
        n_folds=2,
        random_state=42
    )
    
    # Predict
    pred_result = predict_ml_model(
        ml_model_filename=train_result['output_filename'],
        test_input_filename=data_file,
        test_feature_vectors_filename=features_file,
        test_smiles_column='smiles',
        predict_column_name='value',
        project_manifest_path=str(TEST_MANIFEST),
        output_filename='predictions_cv_reg',
        explanation='CV regression predictions'
    )
    
    # Verify
    pred_df = _load_resource(str(TEST_MANIFEST), pred_result['output_filename'])
    
    # Per-fold columns
    assert 'value_1' in pred_df.columns
    assert 'value_2' in pred_df.columns
    
    # Aggregate columns (no entropy for regression)
    assert 'value_mean' in pred_df.columns
    assert 'value_std' in pred_df.columns
    assert 'value_entropy' not in pred_df.columns  # Regression shouldn't have entropy
    
    # Verify std is reasonable
    assert (pred_df['value_std'] >= 0).all()


if __name__ == "__main__":
    test_predict_single_model()
    print("✓ Single model prediction test passed")
    
    test_predict_cv_models_classification()
    print("✓ CV classification prediction test passed")
    
    test_predict_cv_models_regression()
    print("✓ CV regression prediction test passed")
    
    print("\n✓ All tests passed!")
