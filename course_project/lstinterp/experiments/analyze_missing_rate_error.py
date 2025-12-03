"""
Analysis of Error vs Missing Rate

This script analyzes the relationship between model prediction error and the missing rate of the data.
It helps understand how robust the model is to missing data.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor

# Output directory
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "missing_rate_analysis"
RESULTS_DIR = OUTPUT_DIR / "results" / "missing_rate_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data path
DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def calculate_missing_rate(tensor):
    """
    Calculate missing rate for each pixel (across time) and each day (across space)
    """
    H, W, T = tensor.shape
    
    # Spatial missing rate (fraction of missing days for each pixel)
    spatial_missing_rate = np.sum(tensor == 0, axis=2) / T
    
    # Temporal missing rate (fraction of missing pixels for each day)
    temporal_missing_rate = np.sum(tensor == 0, axis=(0, 1)) / (H * W)
    
    return spatial_missing_rate, temporal_missing_rate


def load_predictions(model_name="unet"):
    """Load model prediction results"""
    pred_dir = OUTPUT_DIR / "figures" / "all_days"
    
    pred_mean_path = pred_dir / f"{model_name}_pred_mean.npy"
    pred_std_path = pred_dir / f"{model_name}_pred_std.npy"
    true_path = pred_dir / f"{model_name}_true.npy"
    
    if not all(p.exists() for p in [pred_mean_path, pred_std_path, true_path]):
        print(f"⚠️  Prediction data for {model_name} not found, skipping")
        return None
    
    pred_mean = np.load(pred_mean_path)
    pred_std = np.load(pred_std_path)
    true_values = np.load(true_path)
    
    return pred_mean, pred_std, true_values


def analyze_error_vs_missing_rate(pred_mean, true_values, training_tensor, model_name):
    """Analyze relationship between prediction error and missing rate"""
    print(f"\nAnalyzing {model_name}...")
    
    # Calculate spatial missing rate (based on training data)
    # We want to know: does the model perform worse on pixels that had fewer training observations?
    spatial_missing_rate, _ = calculate_missing_rate(training_tensor)
    
    # Calculate error at each pixel (RMSE across time)
    # Mask out test points that were missing in ground truth (should be none if test set is clean, but let's be safe)
    # Note: true_values comes from test_tensor usually
    errors = true_values - pred_mean
    squared_errors = errors ** 2
    
    # Calculate RMSE per pixel (ignoring NaNs)
    with np.errstate(divide='ignore', invalid='ignore'):
        pixel_mse = np.nanmean(squared_errors, axis=2)
        pixel_rmse = np.sqrt(pixel_mse)
    
    # Flatten for correlation analysis
    missing_rates_flat = spatial_missing_rate.flatten()
    rmses_flat = pixel_rmse.flatten()
    
    # Remove invalid values (NaNs in RMSE where no test data existed for that pixel)
    mask = ~np.isnan(rmses_flat)
    valid_missing_rates = missing_rates_flat[mask]
    valid_rmses = rmses_flat[mask]
    
    # 1. Scatter plot with density or bins
    plt.figure(figsize=(10, 6))
    
    # Bin the missing rates to make the plot clearer
    bins = np.linspace(0, 1, 21) # 0.05 steps
    bin_indices = np.digitize(valid_missing_rates, bins)
    
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if bin_mask.sum() > 0:
            vals = valid_rmses[bin_mask]
            bin_centers.append((bins[i-1] + bins[i]) / 2)
            bin_means.append(np.mean(vals))
            bin_stds.append(np.std(vals))
            bin_counts.append(len(vals))
    
    plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5, 
                 label='Mean RMSE ± STD', color='blue')
    
    # Add count info on secondary axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=0.04, alpha=0.2, color='gray', label='Pixel Count')
    ax2.set_ylabel('Number of Pixels')
    
    ax1.set_xlabel('Missing Rate in Training Data (0-1)')
    ax1.set_ylabel('RMSE on Test Data (K)')
    ax1.set_title(f'{model_name.upper()}: RMSE vs. Training Data Missing Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"{model_name}_error_vs_missing_rate.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {fig_path}")
    
    # Save statistics
    stats_df = pd.DataFrame({
        'Missing_Rate_Bin_Center': bin_centers,
        'Mean_RMSE': bin_means,
        'Std_RMSE': bin_stds,
        'Pixel_Count': bin_counts
    })
    csv_path = RESULTS_DIR / f"{model_name}_error_vs_missing_rate.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"✅ Saved stats: {csv_path}")

    return stats_df


def main():
    """Main function"""
    print("=" * 80)
    print("  Analysis of Error vs Missing Rate")
    print("=" * 80)
    
    # Load training tensor to calculate missing rates
    training_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    
    # Analyze U-Net
    unet_results = load_predictions("unet")
    if unet_results:
        analyze_error_vs_missing_rate(unet_results[0], unet_results[2], training_tensor, "unet")
        
    # Analyze Tree
    tree_results = load_predictions("tree")
    if tree_results:
        analyze_error_vs_missing_rate(tree_results[0], tree_results[2], training_tensor, "tree")
        
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
