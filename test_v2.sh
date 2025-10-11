#!/bin/bash

# Quick test script for v2 implementation
# Tests on small sample to verify everything works

echo "=========================================="
echo "Testing v2 Implementation"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
fi

echo "Testing v2 with small sample (5k train, 100 test)..."
echo "Command: python temp.py --per_unit --max_train 5000 --max_test 100"
echo ""

# Run test
python temp.py --per_unit --max_train 5000 --max_test 100

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! v2 ran without errors"
    echo "=========================================="
    echo ""
    echo "Check the output above for OOF SMAPE score."
    echo "If SMAPE < 15%, you're good to go!"
    echo ""
    echo "To run on full dataset:"
    echo "  python temp.py --per_unit"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ ERROR! Something went wrong"
    echo "=========================================="
    echo ""
    echo "Check the error messages above."
    echo "Common issues:"
    echo "  - Missing dependencies: pip install -r requirements.txt"
    echo "  - Memory issues: reduce --max_train"
    echo "  - Missing data: check dataset/ folder"
    echo ""
fi

