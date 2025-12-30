#!/bin/bash
#
# Run all quality report analyze function tests
#

echo "================================================================================"
echo "RUNNING ALL QUALITY REPORT ANALYZE FUNCTION TESTS"
echo "================================================================================"
echo ""

PASSED=0
FAILED=0
FAILED_TESTS=()

# Test 1: SMILES Validity
echo "Test 1/5: _analyze_smiles_validity"
echo "--------------------------------------------------------------------------------"
if python tests/test_analyze_smiles_validity.py; then
    ((PASSED++))
    echo "✅ PASSED: _analyze_smiles_validity"
else
    ((FAILED++))
    FAILED_TESTS+=("_analyze_smiles_validity")
    echo "❌ FAILED: _analyze_smiles_validity"
fi
echo ""

# Test 2: Activity Distribution
echo "Test 2/5: _analyze_activity_distribution"
echo "--------------------------------------------------------------------------------"
if python tests/test_analyze_activity_distribution.py; then
    ((PASSED++))
    echo "✅ PASSED: _analyze_activity_distribution"
else
    ((FAILED++))
    FAILED_TESTS+=("_analyze_activity_distribution")
    echo "❌ FAILED: _analyze_activity_distribution"
fi
echo ""

# Test 3: Functional Groups
echo "Test 3/5: _analyze_functional_groups"
echo "--------------------------------------------------------------------------------"
if python tests/test_analyze_functional_groups.py; then
    ((PASSED++))
    echo "✅ PASSED: _analyze_functional_groups"
else
    ((FAILED++))
    FAILED_TESTS+=("_analyze_functional_groups")
    echo "❌ FAILED: _analyze_functional_groups"
fi
echo ""

# Test 4: Salts/Fragments/Solvents
echo "Test 4/5: _analyze_salts_fragments_solvents"
echo "--------------------------------------------------------------------------------"
if python tests/test_analyze_salts_fragments_solvents.py; then
    ((PASSED++))
    echo "✅ PASSED: _analyze_salts_fragments_solvents"
else
    ((FAILED++))
    FAILED_TESTS+=("_analyze_salts_fragments_solvents")
    echo "❌ FAILED: _analyze_salts_fragments_solvents"
fi
echo ""

# Summary
echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    echo "❌ SOME TESTS FAILED"
    exit 1
else
    echo ""
    echo "🎉 ALL TESTS PASSED!"
    exit 0
fi
