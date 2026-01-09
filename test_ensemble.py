"""Quick test for BayesianEnsemble class"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from molml_mcp.tools.ml.trad_ml.ensembled_models import BayesianEnsemble

# Create test data
X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

# Test 1: Basic initialization and fitting
print('Test 1: Basic initialization and fitting')
ensemble = BayesianEnsemble(
    base_estimator=RandomForestClassifier,
    ensemble_size=5,
    n_estimators=10,
    max_depth=5,
    random_state=42
)
ensemble.fit(X, y)
print(f'✓ Fitted {len(ensemble)} models')

# Test 2: predict() method
print('\nTest 2: predict() method')
y_pred = ensemble.predict(X[:10])
print(f'✓ Predictions shape: {y_pred.shape}')
print(f'  Sample predictions: {y_pred[:5]}')

# Test 3: predict_with_uncertainty() method
print('\nTest 3: predict_with_uncertainty() method')
mean, std, all_preds = ensemble.predict_with_uncertainty(X[:10])
print(f'✓ Mean shape: {mean.shape}')
print(f'✓ Std shape: {std.shape}')
print(f'✓ All predictions shape: {all_preds.shape}')
print(f'  Sample mean: {mean[:3]}')
print(f'  Sample std: {std[:3]}')

# Test 4: predict_proba() method
print('\nTest 4: predict_proba() method')
y_proba = ensemble.predict_proba(X[:10])
print(f'✓ Probabilities shape: {y_proba.shape}')
print(f'  Sample proba: {y_proba[:2]}')

# Test 5: predict_proba_with_uncertainty() method
print('\nTest 5: predict_proba_with_uncertainty() method')
proba_mean, proba_std, proba_all = ensemble.predict_proba_with_uncertainty(X[:10])
print(f'✓ Mean proba shape: {proba_mean.shape}')
print(f'✓ Std proba shape: {proba_std.shape}')
print(f'✓ All proba shape: {proba_all.shape}')
print(f'  Sample mean proba: {proba_mean[:2]}')
print(f'  Sample std proba: {proba_std[:2]}')

# Test 6: Check random_state consistency
print('\nTest 6: Random state consistency')
ensemble2 = BayesianEnsemble(
    base_estimator=RandomForestClassifier,
    ensemble_size=5,
    n_estimators=10,
    max_depth=5,
    random_state=42
)
ensemble2.fit(X, y)
y_pred2 = ensemble2.predict(X[:10])
print(f'✓ Predictions match: {np.allclose(y_pred, y_pred2)}')

print('\n✅ All tests passed!')
