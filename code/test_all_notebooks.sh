#!/bin/bash
echo "Testing All Example Notebooks"
echo "=============================="
echo ""

pass=0
fail=0

examples=(
    "example_ConsLaborSeparableModel.md"
    "example_ConsLaborPortfolioModel.md"
    "example_ConsRetirementModel.md"
    "example_ConsPensionModel.md"
    "example_GaussianProcessRegression.md"
    "example_WarpedInterpolation.md"
)

for example in "${examples[@]}"; do
    printf "%-40s " "$example"
    if timeout 60 uv run jupytext --sync --execute examples/$example >/dev/null 2>&1; then
        echo "✓ PASS"
        ((pass++))
    else
        echo "✗ FAIL"
        ((fail++))
    fi
done

echo ""
echo "=============================="
echo "Results: $pass passed, $fail failed"
