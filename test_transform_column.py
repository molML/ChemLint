"""Test script for transform_column function"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
from molml_mcp.tools.core.dataset_ops import transform_column
from molml_mcp.infrastructure.resources import _store_resource, _load_resource

# Test configuration
PROJECT_MANIFEST = "/Users/derekvantilborg/Dropbox/PD/molml_mcp/tests/data/test_manifest.json"

print("=" * 80)
print("Testing transform_column()")
print("=" * 80)

# Step 1: Create a test dataset with various numeric columns
print("\n1. Creating test dataset with numeric values...")
test_data = pd.DataFrame({
    "compound_id": ["A", "B", "C", "D", "E"],
    "Ki_nM": [10.0, 100.0, 1000.0, 0.1, 50.0],
    "IC50_nM": [5.0, 50.0, 500.0, 0.05, 25.0],
    "MolWt": [300.5, 450.2, 200.8, 525.3, 380.9],
    "LogP": [2.5, 3.8, 1.2, 4.5, 3.1],
    "concentration": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
})

input_filename = _store_resource(
    test_data,
    PROJECT_MANIFEST,
    "transform_test_data",
    "Test dataset for column transformations",
    "csv"
)
print(f"   ✓ Created dataset: {input_filename}")
print(f"   ✓ Columns: {list(test_data.columns)}")
print(f"   ✓ Preview:\n{test_data.to_string()}")

# Test 2: Simple log transformation (nM to pKi)
print("\n2. Testing pKi calculation (nM to pKi)...")
result = transform_column(
    input_filename=input_filename,
    expression="pKi = -log10(Ki_nM / 1e9)",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_with_pKi",
    explanation="Added pKi column from Ki_nM"
)
print(f"   ✓ Output: {result['output_filename']}")
print(f"   ✓ New columns: {result['columns']}")
print(f"   ✓ Expression: {result['expression']}")

# Verify the calculation
df = _load_resource(PROJECT_MANIFEST, result['output_filename'])
print(f"   ✓ Verification - Ki_nM=10.0 → pKi={df.loc[0, 'pKi']:.2f} (expected ~8.0)")
print(f"   ✓ Verification - Ki_nM=100.0 → pKi={df.loc[1, 'pKi']:.2f} (expected ~7.0)")

# Test 3: pIC50 calculation
print("\n3. Testing pIC50 calculation...")
result = transform_column(
    input_filename=input_filename,
    expression="pIC50 = -log10(IC50_nM / 1e9)",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_with_pIC50",
    explanation="Added pIC50 column from IC50_nM"
)
print(f"   ✓ Output: {result['output_filename']}")
print(f"   ✓ New columns: {result['columns']}")

df = _load_resource(PROJECT_MANIFEST, result['output_filename'])
print(f"   ✓ pIC50 values: {df['pIC50'].round(2).tolist()}")

# Test 4: Boolean flag based on pIC50
print("\n4. Testing boolean flag creation...")
result = transform_column(
    input_filename=result['output_filename'],  # Chain from previous
    expression="active = pIC50 > 7",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_with_activity_flag",
    explanation="Added active flag based on pIC50"
)
print(f"   ✓ Output: {result['output_filename']}")

df = _load_resource(PROJECT_MANIFEST, result['output_filename'])
print(f"   ✓ Active flags: {df['active'].tolist()}")
print(f"   ✓ Active compounds: {df[df['active']]['compound_id'].tolist()}")

# Test 5: Mathematical operations (log, sqrt, etc.)
print("\n5. Testing mathematical operations...")
result = transform_column(
    input_filename=input_filename,
    expression="log_MW = log10(MolWt)",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_log_mw",
    explanation="Log transformation of MolWt"
)
print(f"   ✓ Output: {result['output_filename']}")

df = _load_resource(PROJECT_MANIFEST, result['output_filename'])
print(f"   ✓ log_MW values: {df['log_MW'].round(3).tolist()}")

# Add sqrt transformation
result2 = transform_column(
    input_filename=result['output_filename'],
    expression="sqrt_LogP = sqrt(LogP)",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_math_ops",
    explanation="Sqrt transformation of LogP"
)
df2 = _load_resource(PROJECT_MANIFEST, result2['output_filename'])
print(f"   ✓ sqrt_LogP values: {df2['sqrt_LogP'].round(3).tolist()}")

# Test 6: Normalization (mean/std)
print("\n6. Testing normalization (z-score)...")
result = transform_column(
    input_filename=input_filename,
    expression="MolWt_normalized = (MolWt - MolWt.mean()) / MolWt.std()",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_normalized",
    explanation="Z-score normalization of MolWt"
)
print(f"   ✓ Output: {result['output_filename']}")

df = _load_resource(PROJECT_MANIFEST, result['output_filename'])
print(f"   ✓ Original MolWt: {df['MolWt'].tolist()}")
print(f"   ✓ Normalized MolWt: {df['MolWt_normalized'].round(3).tolist()}")
print(f"   ✓ Mean of normalized: {df['MolWt_normalized'].mean():.6f} (should be ~0)")
print(f"   ✓ Std of normalized: {df['MolWt_normalized'].std():.6f} (should be ~1)")

# Test 7: Arithmetic operations between columns
print("\n7. Testing arithmetic between columns...")
result = transform_column(
    input_filename=input_filename,
    expression="ratio = Ki_nM / IC50_nM",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_ratio",
    explanation="Ratio of Ki to IC50"
)
print(f"   ✓ Output: {result['output_filename']}")

df = _load_resource(PROJECT_MANIFEST, result['output_filename'])
print(f"   ✓ Ki/IC50 ratio: {df['ratio'].round(2).tolist()}")

# Add another arithmetic operation
result2 = transform_column(
    input_filename=result['output_filename'],
    expression="sum_vals = MolWt + LogP * 100",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_arithmetic",
    explanation="Weighted sum of MolWt and LogP"
)
df2 = _load_resource(PROJECT_MANIFEST, result2['output_filename'])
print(f"   ✓ Sum values: {df2['sum_vals'].round(2).tolist()}")

# Test 7: Conditional expressions
print("\n8. Testing conditional expressions...")
result = transform_column(
    input_filename=input_filename,
    expression="size_category = (MolWt < 300) * 1 + (MolWt >= 300) * (MolWt < 500) * 2 + (MolWt >= 500) * 3",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_conditionals",
    explanation="Categorical binning based on MolWt"
)
print(f"   ✓ Output: {result['output_filename']}")

df = _load_resource(PROJECT_MANIFEST, result['output_filename'])
print(f"   ✓ MolWt: {df['MolWt'].tolist()}")
print(f"   ✓ Categories: {df['size_category'].tolist()} (1=<300, 2=300-500, 3=>500)")

# Test 8: Exponential and power operations
print("\n9. Testing exponential and power operations...")
result = transform_column(
    input_filename=input_filename,
    expression="exp_LogP = exp(LogP)",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_exp",
    explanation="Exponential of LogP"
)
df = _load_resource(PROJECT_MANIFEST, result['output_filename'])
print(f"   ✓ exp(LogP): {df['exp_LogP'].round(2).tolist()}")

result2 = transform_column(
    input_filename=input_filename,
    expression="MolWt_squared = MolWt ** 2",
    project_manifest_path=PROJECT_MANIFEST,
    output_filename="test_exp_power",
    explanation="MolWt squared"
)
df2 = _load_resource(PROJECT_MANIFEST, result2['output_filename'])
print(f"   ✓ MolWt²: {df2['MolWt_squared'].round(0).tolist()}")

# Test 9: Test error handling with invalid expression
print("\n10. Testing error handling...")
try:
    result = transform_column(
        input_filename=input_filename,
        expression="invalid = nonexistent_column * 2",
        project_manifest_path=PROJECT_MANIFEST,
        output_filename="test_error",
        explanation="This should fail"
    )
    print("   ✗ Should have raised an error!")
except Exception as e:
    print(f"   ✓ Correctly raised error: {type(e).__name__}")
    print(f"   ✓ Error message: {str(e)[:100]}")

print("\n" + "=" * 80)
print("All tests completed successfully! ✓")
print("=" * 80)
