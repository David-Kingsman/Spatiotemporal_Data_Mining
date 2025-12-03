"""Evaluate and compare all models"""
import numpy as np
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor
from lstinterp.metrics import compute_regression_metrics, compute_probabilistic_metrics

# Output directory
OUTPUT_DIR = Path("output")
(OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)


def load_results():
    """Load results for all models"""
    results = {}
    results_dir = OUTPUT_DIR / "results"
    
    # Load Tree model results
    try:
        with open(results_dir / "tree_results.json", "r") as f:
            results["Tree (XGBoost)"] = json.load(f)
    except FileNotFoundError:
        print("Warning: tree_results.json not found")
    
    # Load U-Net results
    try:
        with open(results_dir / "unet_results.json", "r") as f:
            results["U-Net"] = json.load(f)
    except FileNotFoundError:
        print("Warning: unet_results.json not found")
    
    # Load GP results
    try:
        with open(results_dir / "gp_results.json", "r") as f:
            results["GP (Sparse)"] = json.load(f)
    except FileNotFoundError:
        print("Warning: gp_results.json not found")
    
    return results


def print_comparison_table(results):
    """Print comparison table"""
    if not results:
        print("No results available")
        return
    
    # Define metrics to display (exclude experiment_info)
    metric_display = {
        'rmse': ('RMSE (K)', 'Lower is better'),
        'mae': ('MAE (K)', 'Lower is better'),
        'r2': ('R²', 'Higher is better'),
        'mape': ('MAPE (%)', 'Lower is better'),
        'crps': ('CRPS (K)', 'Lower is better'),
        'coverage_90': ('Coverage (90%)', 'Target: 0.90'),
        'interval_width_90': ('Interval Width (90%)', 'Lower is better'),
        'calibration_error': ('Calibration Error', 'Lower is better')
    }
    
    # Print table
    print("\n" + "=" * 100)
    print("  Model Comparison Results - Test Set Performance")
    print("=" * 100)
    
    # Header
    header = f"{'Metric':<35} {'Tree (XGBoost)':<25} {'U-Net':<25} {'GP (Sparse)':<25}"
    print(header)
    print("-" * 100)
    
    # Data rows - Regression metrics
    print("\n[Regression Metrics]")
    for metric, (name, note) in metric_display.items():
        if metric in ['rmse', 'mae', 'r2', 'mape']:
            row = f"  {name:<33}"
            for model_name in results.keys():
                value = results[model_name].get(metric, None)
                if value is not None and isinstance(value, (int, float)):
                    row += f"{value:<25.4f}"
                else:
                    row += f"{'N/A':<25}"
            print(row)
    
    # Data rows - Probabilistic metrics
    print("\n[Probabilistic Metrics]")
    for metric, (name, note) in metric_display.items():
        if metric in ['crps', 'coverage_90', 'interval_width_90', 'calibration_error']:
            row = f"  {name:<33}"
            for model_name in results.keys():
                value = results[model_name].get(metric, None)
                if value is not None and isinstance(value, (int, float)):
                    row += f"{value:<25.4f}"
                else:
                    row += f"{'N/A':<25}"
            print(row)
    
    # Training info summary
    print("\n[Training Info]")
    training_info = f"{'Training Time':<33}"
    for model_name in results.keys():
        exp_info = results[model_name].get('experiment_info', {})
        train_time = exp_info.get('training_time_seconds', 0)
        if train_time:
            train_time_str = f"{train_time:.1f} s ({train_time/60:.1f} min)"
        else:
            train_time_str = "N/A"
        training_info += f"{train_time_str:<25}"
    print(training_info)
    
    print("=" * 100)
    
    # Performance Ranking
    print("\n[Performance Ranking]")
    
    # R² Ranking (Higher is better)
    r2_ranking = sorted([(name, results[name].get('r2', -999)) for name in results.keys()], 
                        key=lambda x: x[1], reverse=True)
    print("  R² Ranking:")
    for i, (name, value) in enumerate(r2_ranking, 1):
        print(f"    {i}. {name}: {value:.4f}")
    
    # RMSE Ranking (Lower is better)
    rmse_ranking = sorted([(name, results[name].get('rmse', 999)) for name in results.keys()], 
                          key=lambda x: x[1])
    print("\n  RMSE Ranking (Lower is better):")
    for i, (name, value) in enumerate(rmse_ranking, 1):
        print(f"    {i}. {name}: {value:.4f} K")
    
    # CRPS Ranking (Lower is better)
    crps_ranking = sorted([(name, results[name].get('crps', 999)) for name in results.keys()], 
                          key=lambda x: x[1])
    print("\n  CRPS Ranking (Lower is better):")
    for i, (name, value) in enumerate(crps_ranking, 1):
        print(f"    {i}. {name}: {value:.4f} K")
    
    print("=" * 100)


def main():
    results = load_results()
    
    if not results:
        print("No result files found. Please run training scripts first.")
        return
    
    print_comparison_table(results)
    
    # Save comparison results
    comparison_path = OUTPUT_DIR / "results" / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparison results saved to {comparison_path}")


if __name__ == "__main__":
    main()
