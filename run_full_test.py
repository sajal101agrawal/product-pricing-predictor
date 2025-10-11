#!/usr/bin/env python3
"""
Run full test prediction and evaluation with model saving/loading

This script:
1. Trains the model on specified training samples (if not already trained)
2. SAVES the trained model for reuse
3. Predicts on the COMPLETE test dataset using saved model
4. Runs evaluation if ground truth is available

Usage:
    # Train and test (saves model)
    python run_full_test.py --max_train 15000
    
    # Just test using existing model (no training)
    python run_full_test.py --use_existing_model models/trained_model
    
    # Train with custom parameters
    python run_full_test.py --max_train 20000 --pca_dim 256
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser(description="Train, save model, and evaluate on full test dataset")
    parser.add_argument("--max_train", type=int, default=15000, 
                        help="Max training samples to use (default: 15000)")
    parser.add_argument("--pca_dim", type=int, default=128, 
                        help="PCA dimensions (default: 128)")
    parser.add_argument("--max_tfidf", type=int, default=200000, 
                        help="Max TF-IDF features (default: 200000)")
    parser.add_argument("--folds", type=int, default=5, 
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--knn_neighbors", type=int, default=64, 
                        help="KNN neighbors (default: 64)")
    parser.add_argument("--knn_tau", type=float, default=1.0, 
                        help="KNN distance temperature (default: 1.0)")
    parser.add_argument("--per_unit", action="store_true", default=True,
                        help="Use per-unit target normalization (default: True)")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Use CPU only (no CUDA)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory to save/load model (default: models/model_TIMESTAMP)")
    parser.add_argument("--use_existing_model", type=str, default=None,
                        help="Use existing model (skip training)")
    
    args = parser.parse_args()
    
    # Determine model directory
    if args.use_existing_model:
        model_dir = Path(args.use_existing_model)
        skip_training = True
    else:
        if args.model_dir:
            model_dir = Path(args.model_dir)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_dir = Path(f"models/model_{timestamp}")
        skip_training = False
    
    print("=" * 80)
    print(" " * 20 + "Full Test Dataset Prediction & Evaluation")
    print("=" * 80)
    print()
    print("Configuration:")
    if skip_training:
        print(f"  Mode: PREDICTION ONLY (using existing model)")
        print(f"  Model directory: {model_dir}")
    else:
        print(f"  Mode: TRAIN + PREDICT")
        print(f"  Max training samples: {args.max_train}")
        print(f"  Model will be saved to: {model_dir}")
        print(f"  PCA dimensions: {args.pca_dim}")
        print(f"  TF-IDF features: {args.max_tfidf}")
        print(f"  CV folds: {args.folds}")
        print(f"  KNN neighbors: {args.knn_neighbors}")
        print(f"  Per-unit normalization: {args.per_unit}")
        print(f"  CPU only: {args.cpu_only}")
    print(f"  Test samples: ALL (complete test.csv)")
    print(f"  Output: test_out.csv")
    print()
    print("=" * 80)
    print()
    
    # ============================================================
    # STEP 1: Training (if needed)
    # ============================================================
    if not skip_training:
        print("\n" + "=" * 80)
        print("STEP 1: TRAINING MODEL")
        print("=" * 80)
        print()
        
        # Build training command
        cmd = [
            sys.executable,
            "train_and_save_model.py",
            "--max_train", str(args.max_train),
            "--pca_dim", str(args.pca_dim),
            "--max_tfidf", str(args.max_tfidf),
            "--folds", str(args.folds),
            "--knn_neighbors", str(args.knn_neighbors),
            "--knn_tau", str(args.knn_tau),
            "--save_model", str(model_dir),
        ]
        
        if args.per_unit:
            cmd.append("--per_unit")
        
        if args.cpu_only:
            cmd.append("--cpu_only")
        
        # Use smaller test set for training to save time
        cmd.extend(["--max_test", "100"])
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run training with log output
        log_file = Path("training_full_test.log")
        print(f"Training in progress... (logging to {log_file})")
        print()
        
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print and log simultaneously
            for line in process.stdout:
                print(line, end='')
                f.write(line)
            
            process.wait()
            return_code = process.returncode
        
        if return_code != 0:
            print()
            print("ERROR: Training failed!")
            print(f"Check {log_file} for details")
            sys.exit(1)
        
        print()
        print("=" * 80)
        print(f"✓ Training completed! Model saved to: {model_dir}")
        print("=" * 80)
        print()
    else:
        # Check if model exists
        if not model_dir.exists():
            print(f"ERROR: Model directory not found: {model_dir}")
            sys.exit(1)
        
        print(f"Using existing model from: {model_dir}")
        print()
    
    # ============================================================
    # STEP 2: Prediction on Full Test Dataset
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 2: PREDICTING ON FULL TEST DATASET")
    print("=" * 80)
    print()
    
    # Build prediction command
    pred_cmd = [
        sys.executable,
        "train_and_save_model.py",
        "--load_model", str(model_dir),
        "--predict_only",
    ]
    
    if args.cpu_only:
        pred_cmd.append("--cpu_only")
    
    print(f"Command: {' '.join(pred_cmd)}")
    print()
    
    # Run prediction
    pred_result = subprocess.run(
        pred_cmd,
        capture_output=False
    )
    
    if pred_result.returncode != 0:
        print()
        print("ERROR: Prediction failed!")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("Prediction completed successfully!")
    print("=" * 80)
    print()
    
    # Check if test_out.csv was created
    test_out = Path("test_out.csv")
    if not test_out.exists():
        print("ERROR: test_out.csv was not created!")
        sys.exit(1)
    
    # Count predictions
    with open(test_out) as f:
        num_predictions = sum(1 for _ in f) - 1  # Subtract header
    
    print(f"Generated predictions for {num_predictions:,} samples")
    print()
    
    # Run evaluation if not skipped and ground truth exists
    if not args.skip_eval:
        ground_truth = Path("dataset/sample_test_out.csv")
        if ground_truth.exists():
            print("=" * 80)
            print("Running Evaluation...")
            print("=" * 80)
            print()
            
            eval_result = subprocess.run(
                [sys.executable, "evaluate_predictions.py"],
                capture_output=False
            )
            
            if eval_result.returncode == 0:
                print()
                print("=" * 80)
                print("Evaluation completed!")
                print("See evaluation_details.csv for per-sample details")
                print("=" * 80)
            else:
                print()
                print("Warning: Evaluation failed or no matching samples found")
        else:
            print("=" * 80)
            print("Note: Ground truth file (dataset/sample_test_out.csv) not found")
            print("Skipping evaluation. Only predictions generated.")
            print("=" * 80)
    else:
        print("Skipping evaluation (--skip_eval flag)")
    
    print()
    print("=" * 80)
    print("All Done!")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  - test_out.csv: Predictions for {num_predictions:,} test samples")
    print(f"  - {model_dir}: Saved model (reusable)")
    if not skip_training:
        log_file = Path("training_full_test.log")
        if log_file.exists():
            print(f"  - {log_file}: Training logs")
    
    eval_details = Path("evaluation_details.csv")
    if eval_details.exists():
        print(f"  - evaluation_details.csv: Detailed evaluation results")
    
    print()
    print("=" * 80)
    print("REUSE MODEL:")
    print(f"  python run_full_test.py --use_existing_model {model_dir}")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()

