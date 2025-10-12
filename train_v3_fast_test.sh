#!/bin/bash
# Fast test run for V3 (smaller dataset, fewer improvements)
# Use this to quickly test if everything works before full training

echo "=========================================="
echo "V3 Fast Test"
echo "=========================================="
echo ""
echo "This is a quick test to verify the pipeline works."
echo "For actual submission, use train_v3_recommended.sh"
echo ""

python train_and_save_model_v3.py \
  --max_train 200 \
  --max_test 100 \
  --per_unit \
  --stack_cv \
  --char_tfidf_max 100000 \
  --ratio_calib \
  --prior_blend \
  --out_csv test_out_v3_fast.csv

echo ""
echo "Fast test complete! Output: test_out_v3_fast.csv"

