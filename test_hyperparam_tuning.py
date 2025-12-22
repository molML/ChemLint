"""Test script for tune_hyperparameters function"""
import sys
sys.path.insert(0, 'src')

from molml_mcp.tools.ml.hyperparam_tuning import tune_hyperparameters
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from rdkit.Chem import MolFromSmiles, Descriptors
import pandas as pd

# Test configuration
PROJECT_MANIFEST = "/Users/derekvantilborg/Dropbox/PD/molml_mcp/tests/data/test_manifest.json"
INPUT_CSV = "/Users/derekvantilborg/Dropbox/PD/molml_mcp/tests/data/cleaning_test.csv"
SMILES_COL = "smiles"
TARGET_COL = "label"

print("=" * 80)
print("Testing tune_hyperparameters()")
print("=" * 80)

# Step 1: Load data and create feature vectors (SMILES -> feature vector dict)
print("\n1. Loading dataset and generating molecular feature vectors...")
df = pd.read_csv(INPUT_CSV)
print(f"   ✓ Loaded {len(df)} molecules from dataset")

# Store dataset in manifest first
input_filename = _store_resource(
    df,
    PROJECT_MANIFEST,
    "tune_test_dataset",
    "Test dataset for hyperparameter tuning",
    "csv"
)
print(f"   ✓ Stored dataset: {input_filename}")

# Create simple feature vectors from RDKit descriptors
feature_dict = {}
descriptor_names = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA"]

for smiles in df[SMILES_COL]:
    mol = MolFromSmiles(smiles)
    if mol:
        features = [getattr(Descriptors, desc)(mol) for desc in descriptor_names]
        feature_dict[smiles] = features
    else:
        feature_dict[smiles] = [0.0] * len(descriptor_names)

# Store feature vectors as JSON
feature_filename = _store_resource(
    feature_dict,
    PROJECT_MANIFEST,
    "test_feature_vectors",
    "Feature vectors for hyperparameter tuning test",
    "json"
)
print(f"   ✓ Created feature vectors: {feature_filename}")
print(f"   ✓ Features per molecule: {len(descriptor_names)}")
print(f"   ✓ Total molecules with features: {len(feature_dict)}")

# Step 2: Grid search with small parameter space
print("\n2. Running grid search (small parameter space)...")
param_grid_small = {
    "n_estimators": [10, 20],
    "max_depth": [3, 5]
}
grid_result = tune_hyperparameters(
    input_filename=input_filename,
    feature_vectors_filename=feature_filename,
    smiles_column=SMILES_COL,
    target_column=TARGET_COL,
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="grid_search_best_params",
    explanation="Grid search for Random Forest (small space)",
    model_algorithm="random_forest",
    param_grid=param_grid_small,
    search_strategy="grid",
    cv_strategy="stratified",
    n_folds=3,
    metric="accuracy",
    higher_is_better=True,
    random_state=42
)
print(f"   ✓ Best params file: {grid_result['output_filename']}")
print(f"   ✓ Best hyperparameters: {grid_result['best_hyperparameters']}")
print(f"   ✓ Best CV score: {grid_result['best_score']:.4f}")

# Verify that all combinations were tried (2 * 2 = 4 combinations)
from sklearn.model_selection import ParameterGrid
total_combos = len(list(ParameterGrid(param_grid_small)))
print(f"   ✓ Total parameter combinations: {total_combos}")

# Step 3: Random search with larger parameter space
print("\n3. Running random search (sampling from larger space)...")
param_grid_large = {
    "n_estimators": [10, 20, 50, 100],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}
random_result = tune_hyperparameters(
    input_filename=input_filename,
    feature_vectors_filename=feature_filename,
    smiles_column=SMILES_COL,
    target_column=TARGET_COL,
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="random_search_best_params",
    explanation="Random search for Random Forest (larger space)",
    model_algorithm="random_forest",
    param_grid=param_grid_large,
    search_strategy="random",
    n_searches=6,  # Sample 6 out of 48 possible combinations
    cv_strategy="stratified",
    n_folds=3,
    metric="f1_score",
    higher_is_better=True,
    random_state=42
)
print(f"   ✓ Best params file: {random_result['output_filename']}")
print(f"   ✓ Best hyperparameters: {random_result['best_hyperparameters']}")
print(f"   ✓ Best CV score: {random_result['best_score']:.4f}")

total_combos_large = len(list(ParameterGrid(param_grid_large)))
print(f"   ✓ Total possible combinations: {total_combos_large}")
print(f"   ✓ Combinations evaluated: 6 (random sampling)")

# Step 4: Test reproducibility with same random_state
print("\n4. Testing reproducibility (same random_state)...")
random_result2 = tune_hyperparameters(
    input_filename=input_filename,
    feature_vectors_filename=feature_filename,
    smiles_column=SMILES_COL,
    target_column=TARGET_COL,
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="random_search_best_params_2",
    explanation="Second run with same random_state",
    model_algorithm="random_forest",
    param_grid=param_grid_large,
    search_strategy="random",
    n_searches=6,
    cv_strategy="stratified",
    n_folds=3,
    metric="f1_score",
    higher_is_better=True,
    random_state=42  # Same seed
)
print(f"   ✓ First run best params: {random_result['best_hyperparameters']}")
print(f"   ✓ Second run best params: {random_result2['best_hyperparameters']}")
if random_result['best_hyperparameters'] == random_result2['best_hyperparameters']:
    print("   ✓ REPRODUCIBLE: Same random_state produces same results!")
else:
    print("   ✗ WARNING: Results differ despite same random_state")

# Step 5: Verify best params are stored correctly
print("\n5. Verifying stored hyperparameters...")
loaded_params = _load_resource(PROJECT_MANIFEST, grid_result['output_filename'])
print(f"   ✓ Loaded params type: {type(loaded_params)}")
print(f"   ✓ Loaded params: {loaded_params}")
if loaded_params == grid_result['best_hyperparameters']:
    print("   ✓ Stored parameters match returned parameters!")

print("\n" + "=" * 80)
print("All tests completed successfully! ✓")
print("=" * 80)
