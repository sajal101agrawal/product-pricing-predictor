#!/usr/bin/env python3
"""
Evaluation script for price predictions using SMAPE metric
Compares predicted prices against ground truth and provides detailed statistics
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_smape(actual, predicted, eps=1e-8):
    """
    Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
    
    Formula: SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|)/2)
    
    Returns percentage (0-200%)
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    
    # Avoid division by zero
    denominator = np.maximum(denominator, eps)
    
    smape = np.mean(numerator / denominator) * 100.0
    return smape

def calculate_detailed_metrics(actual, predicted):
    """Calculate detailed error metrics"""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    
    # Basic errors
    errors = predicted - actual
    abs_errors = np.abs(errors)
    percentage_errors = (abs_errors / np.maximum(actual, 1e-8)) * 100
    
    # SMAPE per sample
    smape_per_sample = abs_errors / np.maximum((np.abs(actual) + np.abs(predicted)) / 2.0, 1e-8) * 100
    
    metrics = {
        'smape': calculate_smape(actual, predicted),
        'mae': np.mean(abs_errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mape': np.mean(percentage_errors),
        'median_ae': np.median(abs_errors),
        'max_error': np.max(abs_errors),
        'min_error': np.min(abs_errors),
        'mean_error': np.mean(errors),  # Bias
        'std_error': np.std(errors),
        'mean_actual': np.mean(actual),
        'mean_predicted': np.mean(predicted),
        'smape_per_sample': smape_per_sample
    }
    
    return metrics, errors, abs_errors, percentage_errors

def print_evaluation_report(metrics, actual, predicted, errors, sample_ids=None):
    """Print detailed evaluation report"""
    
    print("="*80)
    print(" " * 25 + "EVALUATION REPORT")
    print("="*80)
    
    print(f"\n{'='*80}")
    print(f"  PRIMARY METRIC (Competition Metric)")
    print(f"{'='*80}")
    print(f"  SMAPE Score:                     {metrics['smape']:>10.4f}%")
    print(f"  (Lower is better, range: 0-200%)")
    
    print(f"\n{'='*80}")
    print(f"  ADDITIONAL METRICS")
    print(f"{'='*80}")
    print(f"  Mean Absolute Error (MAE):       ${metrics['mae']:>10.2f}")
    print(f"  Root Mean Squared Error (RMSE):  ${metrics['rmse']:>10.2f}")
    print(f"  Mean Absolute % Error (MAPE):    {metrics['mape']:>10.2f}%")
    print(f"  Median Absolute Error:           ${metrics['median_ae']:>10.2f}")
    
    print(f"\n{'='*80}")
    print(f"  ERROR STATISTICS")
    print(f"{'='*80}")
    print(f"  Max Error:                       ${metrics['max_error']:>10.2f}")
    print(f"  Min Error:                       ${metrics['min_error']:>10.2f}")
    print(f"  Mean Error (Bias):               ${metrics['mean_error']:>10.2f}")
    print(f"  Std Dev of Errors:               ${metrics['std_error']:>10.2f}")
    
    print(f"\n{'='*80}")
    print(f"  PRICE STATISTICS")
    print(f"{'='*80}")
    print(f"  Mean Actual Price:               ${metrics['mean_actual']:>10.2f}")
    print(f"  Mean Predicted Price:            ${metrics['mean_predicted']:>10.2f}")
    print(f"  Total Samples:                   {len(actual):>11,}")
    
    # Error distribution
    print(f"\n{'='*80}")
    print(f"  ERROR DISTRIBUTION")
    print(f"{'='*80}")
    
    abs_errors = np.abs(errors)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(abs_errors, p)
        print(f"  {p}th percentile error:           ${val:>10.2f}")
    
    # SMAPE distribution
    print(f"\n{'='*80}")
    print(f"  SMAPE DISTRIBUTION (per sample)")
    print(f"{'='*80}")
    smape_samples = metrics['smape_per_sample']
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(smape_samples, p)
        print(f"  {p}th percentile SMAPE:          {val:>10.2f}%")
    
    # Accuracy buckets
    print(f"\n{'='*80}")
    print(f"  PREDICTION ACCURACY BUCKETS")
    print(f"{'='*80}")
    
    smape_buckets = [
        (0, 5, "Excellent"),
        (5, 10, "Very Good"),
        (10, 20, "Good"),
        (20, 30, "Fair"),
        (30, 50, "Poor"),
        (50, 200, "Very Poor")
    ]
    
    for low, high, label in smape_buckets:
        count = np.sum((smape_samples >= low) & (smape_samples < high))
        pct = (count / len(smape_samples)) * 100
        print(f"  {label:12} ({low:3}-{high:3}%):  {count:>6} samples ({pct:>5.1f}%)")
    
    # Worst predictions
    print(f"\n{'='*80}")
    print(f"  TOP 10 WORST PREDICTIONS")
    print(f"{'='*80}")
    print(f"  {'Rank':<6} {'Sample ID':<12} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'SMAPE%':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    worst_indices = np.argsort(smape_samples)[-10:][::-1]
    for rank, idx in enumerate(worst_indices, 1):
        sid = sample_ids[idx] if sample_ids is not None else idx
        act = actual[idx]
        pred = predicted[idx]
        err = errors[idx]
        smape_val = smape_samples[idx]
        print(f"  {rank:<6} {sid:<12} ${act:>9.2f} ${pred:>9.2f} ${err:>9.2f} {smape_val:>9.2f}%")
    
    # Best predictions
    print(f"\n{'='*80}")
    print(f"  TOP 10 BEST PREDICTIONS")
    print(f"{'='*80}")
    print(f"  {'Rank':<6} {'Sample ID':<12} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'SMAPE%':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    best_indices = np.argsort(smape_samples)[:10]
    for rank, idx in enumerate(best_indices, 1):
        sid = sample_ids[idx] if sample_ids is not None else idx
        act = actual[idx]
        pred = predicted[idx]
        err = errors[idx]
        smape_val = smape_samples[idx]
        print(f"  {rank:<6} {sid:<12} ${act:>9.2f} ${pred:>9.2f} ${err:>9.2f} {smape_val:>9.2f}%")
    
    print(f"\n{'='*80}\n")

def main():
    # File paths
    predicted_file = Path("test_out.csv")
    actual_file = Path("dataset/sample_test_out.csv")
    
    # Check if files exist
    if not predicted_file.exists():
        print(f"Error: Predicted file not found: {predicted_file}")
        print("Please run the prediction script first to generate test_out.csv")
        sys.exit(1)
    
    if not actual_file.exists():
        print(f"Error: Ground truth file not found: {actual_file}")
        print("Using sample_test_out.csv as ground truth for evaluation")
        sys.exit(1)
    
    # Load data
    print(f"\nLoading predictions from: {predicted_file}")
    print(f"Loading ground truth from: {actual_file}")
    
    predicted_df = pd.read_csv(predicted_file)
    actual_df = pd.read_csv(actual_file)
    
    # Merge on sample_id to ensure alignment
    merged = actual_df.merge(predicted_df, on='sample_id', suffixes=('_actual', '_predicted'))
    
    if len(merged) == 0:
        print("Error: No matching sample_ids found between files!")
        sys.exit(1)
    
    if len(merged) < len(predicted_df):
        print(f"Warning: Only {len(merged)} out of {len(predicted_df)} predictions have ground truth")
    
    # Extract values
    actual = merged['price_actual'].values
    predicted = merged['price_predicted'].values
    sample_ids = merged['sample_id'].values
    
    # Calculate metrics
    metrics, errors, abs_errors, percentage_errors = calculate_detailed_metrics(actual, predicted)
    
    # Print report
    print_evaluation_report(metrics, actual, predicted, errors, sample_ids)
    
    # Save detailed results to CSV
    output_file = Path("evaluation_details.csv")
    results_df = pd.DataFrame({
        'sample_id': sample_ids,
        'actual_price': actual,
        'predicted_price': predicted,
        'error': errors,
        'abs_error': abs_errors,
        'percentage_error': percentage_errors,
        'smape': metrics['smape_per_sample']
    })
    results_df = results_df.sort_values('smape', ascending=False)
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    # Return SMAPE score
    return metrics['smape']

if __name__ == "__main__":
    smape_score = main()
    sys.exit(0)

