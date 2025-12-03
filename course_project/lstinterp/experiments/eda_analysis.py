"""
Exploratory Data Analysis (EDA)

This script performs comprehensive exploratory analysis on MODIS data:
1. Statistical Summary Table (Mean, Std, Quantiles, Missing Rate)
2. Spatial Correlation Analysis (Semi-variogram, Spatial Autocorrelation)
3. Temporal Analysis (Autocorrelation, Trend, Seasonality)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor, MODISDataset

# Output directory
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "eda"
RESULTS_DIR = OUTPUT_DIR / "results" / "eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data path
DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def print_section_header(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def compute_statistical_summary(train_tensor, test_tensor):
    """Compute statistical summary table"""
    print_section_header("Statistical Summary Table")
    
    # Extract observed values (non-zero)
    train_mask = (train_tensor != 0)
    test_mask = (test_tensor != 0)
    
    train_values = train_tensor[train_mask]
    test_values = test_tensor[test_mask]
    
    # Calculate statistics
    stats_dict = {
        'Dataset': ['Training', 'Test'],
        'Total_Pixels': [train_tensor.size, test_tensor.size],
        'Observed_Pixels': [train_mask.sum(), test_mask.sum()],
        'Missing_Pixels': [(~train_mask).sum(), (~test_mask).sum()],
        'Missing_Rate(%)': [(~train_mask).sum() / train_tensor.size * 100, 
                            (~test_mask).sum() / test_tensor.size * 100],
        'Mean(K)': [train_values.mean(), test_values.mean()],
        'Std(K)': [train_values.std(), test_values.std()],
        'Min(K)': [train_values.min(), test_values.min()],
        'Max(K)': [train_values.max(), test_values.max()],
        'Median(K)': [np.median(train_values), np.median(test_values)],
        'Q25(K)': [np.percentile(train_values, 25), np.percentile(test_values, 25)],
        'Q75(K)': [np.percentile(train_values, 75), np.percentile(test_values, 75)],
        'Skewness': [stats.skew(train_values), stats.skew(test_values)],
        'Kurtosis': [stats.kurtosis(train_values), stats.kurtosis(test_values)]
    }
    
    df = pd.DataFrame(stats_dict)
    
    # Print table
    print("\nData Statistical Summary:")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Save as CSV and JSON
    csv_path = RESULTS_DIR / "statistical_summary.csv"
    json_path = RESULTS_DIR / "statistical_summary.json"
    
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Statistical summary table saved: {csv_path}")
    
    with open(json_path, 'w') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Statistical summary JSON saved: {json_path}")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training data distribution
    axes[0, 0].hist(train_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Training Data Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Temperature (K)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(train_values.mean(), color='red', linestyle='--', 
                       label=f'Mean: {train_values.mean():.2f} K')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test data distribution
    axes[0, 1].hist(test_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Test Data Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Temperature (K)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(test_values.mean(), color='red', linestyle='--', 
                       label=f'Mean: {test_values.mean():.2f} K')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Boxplot comparison
    axes[1, 0].boxplot([train_values, test_values], labels=['Training', 'Test'])
    axes[1, 0].set_title('Temperature Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Temperature (K)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(train_values, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Training Data)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "statistical_summary.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Statistical summary plot saved: {fig_path}")
    
    return stats_dict


def analyze_daily_statistics(tensor, dataset_name="Training"):
    """Analyze daily statistics"""
    print_section_header(f"{dataset_name} - Daily Statistical Analysis")
    
    H, W, T = tensor.shape
    
    # Daily stats
    daily_stats = {
        'Day': list(range(1, T + 1)),
        'Mean_Temp(K)': [],
        'Std_Temp(K)': [],
        'Min_Temp(K)': [],
        'Max_Temp(K)': [],
        'Missing_Rate(%)': [],
        'Observed_Pixels': []
    }
    
    for t in range(T):
        day_data = tensor[:, :, t]
        mask = (day_data != 0)
        if mask.sum() > 0:
            values = day_data[mask]
            daily_stats['Mean_Temp(K)'].append(values.mean())
            daily_stats['Std_Temp(K)'].append(values.std())
            daily_stats['Min_Temp(K)'].append(values.min())
            daily_stats['Max_Temp(K)'].append(values.max())
            daily_stats['Missing_Rate(%)'].append((~mask).sum() / mask.size * 100)
            daily_stats['Observed_Pixels'].append(mask.sum())
        else:
            daily_stats['Mean_Temp(K)'].append(np.nan)
            daily_stats['Std_Temp(K)'].append(np.nan)
            daily_stats['Min_Temp(K)'].append(np.nan)
            daily_stats['Max_Temp(K)'].append(np.nan)
            daily_stats['Missing_Rate(%)'].append(100.0)
            daily_stats['Observed_Pixels'].append(0)
    
    df = pd.DataFrame(daily_stats)
    
    # Save
    csv_path = RESULTS_DIR / f"{dataset_name.lower()}_daily_stats.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ Daily statistics saved: {csv_path}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Daily Mean Temperature
    axes[0, 0].plot(df['Day'], df['Mean_Temp(K)'], marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Daily Mean Temperature', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Mean Temperature (K)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Daily Missing Rate
    axes[0, 1].plot(df['Day'], df['Missing_Rate(%)'], marker='s', color='red', linewidth=2, markersize=4)
    axes[0, 1].set_title('Daily Missing Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Missing Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature Range
    axes[1, 0].fill_between(df['Day'], df['Min_Temp(K)'], df['Max_Temp(K)'], 
                            alpha=0.3, label='Temperature Range')
    axes[1, 0].plot(df['Day'], df['Mean_Temp(K)'], marker='o', linewidth=2, 
                    markersize=4, label='Mean', color='red')
    axes[1, 0].set_title('Temperature Range by Day', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Temperature (K)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Observed Pixels
    axes[1, 1].bar(df['Day'], df['Observed_Pixels'], alpha=0.7, color='green')
    axes[1, 1].set_title('Observed Pixels per Day', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Number of Observed Pixels')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"{dataset_name.lower()}_daily_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Daily analysis plot saved: {fig_path}")
    
    return df


def analyze_spatial_correlation(tensor, sample_size=5000):
    """Analyze Spatial Correlation (Semi-variogram)"""
    print_section_header("Spatial Correlation Analysis")
    
    # Select a time slice for analysis (Day 15)
    H, W, T = tensor.shape
    day_idx = T // 2  # Middle day
    day_data = tensor[:, :, day_idx]
    mask = (day_data != 0)
    
    if mask.sum() < 100:
        print(f"⚠️  Too few observed points on Day {day_idx+1}, skipping spatial correlation analysis")
        return None
    
    # Sample observed points (if too many)
    lat_indices, lon_indices = np.where(mask)
    if len(lat_indices) > sample_size:
        sample_indices = np.random.choice(len(lat_indices), sample_size, replace=False)
        lat_indices = lat_indices[sample_indices]
        lon_indices = lon_indices[sample_indices]
    
    values = day_data[lat_indices, lon_indices]
    coords = np.column_stack([lat_indices, lon_indices])
    
    # Calculate spatial distances
    distances = pdist(coords, metric='euclidean')
    
    # Calculate squared differences of values
    value_diffs = pdist(values.reshape(-1, 1), metric='euclidean') ** 2
    
    # Calculate semi-variogram (binning)
    max_dist = distances.max()
    n_bins = 30
    bins = np.linspace(0, max_dist, n_bins + 1)
    
    semivariances = []
    bin_centers = []
    
    for i in range(n_bins):
        bin_mask = (distances >= bins[i]) & (distances < bins[i+1])
        if bin_mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            semivariances.append(value_diffs[bin_mask].mean() / 2)
    
    # Visualize semi-variogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Semi-variogram
    axes[0].plot(bin_centers, semivariances, 'o-', linewidth=2, markersize=6, color='blue')
    axes[0].set_title('Semi-variogram (Day {})'.format(day_idx + 1), fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Distance (pixels)')
    axes[0].set_ylabel('Semivariance')
    axes[0].grid(True, alpha=0.3)
    
    # Spatial Temperature Distribution
    im = axes[1].imshow(day_data, aspect='auto', origin='lower', cmap='jet_r')
    axes[1].set_title('Spatial Temperature Distribution (Day {})'.format(day_idx + 1), 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude Index')
    axes[1].set_ylabel('Latitude Index')
    plt.colorbar(im, ax=axes[1], label='Temperature (K)')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "spatial_correlation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Spatial correlation plot saved: {fig_path}")
    
    # Calculate Moran's I (simplified, using nearest neighbors)
    from scipy.spatial.distance import cdist
    
    # Calculate k nearest neighbors
    k = min(10, len(coords) - 1)
    dist_matrix = cdist(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Find k nearest neighbors
    nearest_neighbors = np.argsort(dist_matrix, axis=1)[:, :k]
    
    # Calculate Moran's I
    mean_val = values.mean()
    centered_values = values - mean_val
    numerator = 0
    denominator = np.sum(centered_values ** 2)
    
    for i in range(len(values)):
        for j in nearest_neighbors[i]:
            numerator += centered_values[i] * centered_values[j]
    
    morans_i = numerator / (k * denominator) if denominator > 0 else 0
    
    print(f"\nSpatial Autocorrelation (Approx. Moran's I): {morans_i:.4f}")
    print(f"  Value near 1 indicates strong positive correlation")
    print(f"  Value near 0 indicates no spatial correlation")
    print(f"  Value near -1 indicates strong negative correlation")
    
    return {
        'bin_centers': bin_centers,
        'semivariances': semivariances,
        'morans_i': morans_i
    }


def analyze_temporal_correlation(tensor):
    """Analyze Temporal Correlation"""
    print_section_header("Time Series Analysis")
    
    H, W, T = tensor.shape
    
    # Select a spatial location (center point) for time series analysis
    center_lat, center_lon = H // 2, W // 2
    
    # If center point is missing, find nearest observed location
    for offset in range(min(H, W) // 2):
        for lat_offset in [-offset, offset]:
            for lon_offset in [-offset, offset]:
                lat_idx = center_lat + lat_offset
                lon_idx = center_lon + lon_offset
                if 0 <= lat_idx < H and 0 <= lon_idx < W:
                    time_series = tensor[lat_idx, lon_idx, :]
                    if (time_series != 0).sum() > T * 0.5:  # At least 50% observations
                        break
            else:
                continue
            break
        else:
            continue
        break
    
    time_series = tensor[lat_idx, lon_idx, :]
    valid_mask = (time_series != 0)
    
    if valid_mask.sum() < 10:
        print("⚠️  Too few valid time points, skipping time series analysis")
        return None
    
    valid_series = time_series[valid_mask]
    valid_days = np.where(valid_mask)[0] + 1
    
    # Calculate Autocorrelation
    from scipy.signal import correlate
    
    # Zero-mean
    centered_series = valid_series - valid_series.mean()
    
    # Calculate autocorrelation (max lag T//2)
    max_lag = min(len(centered_series) // 2, 15)
    autocorr = []
    lags = list(range(max_lag))
    
    for lag in lags:
        if lag == 0:
            autocorr.append(1.0)
        else:
            numerator = np.sum(centered_series[:-lag] * centered_series[lag:])
            denominator = np.sum(centered_series ** 2)
            if denominator > 0:
                autocorr.append(numerator / denominator)
            else:
                autocorr.append(0.0)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time Series
    axes[0, 0].plot(valid_days, valid_series, marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_title(f'Time Series at Location ({lat_idx}, {lon_idx})', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Temperature (K)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Autocorrelation Function
    axes[0, 1].stem(lags, autocorr, basefmt=' ', linefmt='blue', markerfmt='bo')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 1].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Lag (days)')
    axes[0, 1].set_ylabel('Autocorrelation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Trend Analysis (Linear Fit)
    if len(valid_days) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_days, valid_series)
        axes[1, 0].plot(valid_days, valid_series, marker='o', linewidth=2, markersize=4, 
                       label='Data', alpha=0.7)
        trend_line = slope * valid_days + intercept
        axes[1, 0].plot(valid_days, trend_line, 'r--', linewidth=2, 
                       label=f'Trend (slope={slope:.4f} K/day, R²={r_value**2:.4f})')
        axes[1, 0].set_title('Time Series with Linear Trend', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Temperature (K)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        print(f"\nLinear Trend Analysis:")
        print(f"  Slope: {slope:.6f} K/day")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.4f}")
    
    # Daily Difference Analysis
    if len(valid_series) > 1:
        daily_diffs = np.diff(valid_series)
        axes[1, 1].plot(valid_days[1:], daily_diffs, marker='o', linewidth=2, markersize=4, color='green')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[1, 1].set_title('Daily Temperature Changes', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Temperature Change (K)')
        axes[1, 1].grid(True, alpha=0.3)
        
        print(f"\nDaily Change Statistics:")
        print(f"  Mean Daily Change: {daily_diffs.mean():.4f} K")
        print(f"  Daily Change Std: {daily_diffs.std():.4f} K")
        print(f"  Max Daily Increase: {daily_diffs.max():.4f} K")
        print(f"  Max Daily Decrease: {daily_diffs.min():.4f} K")
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "temporal_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Time series analysis plot saved: {fig_path}")
    
    return {
        'lags': lags,
        'autocorr': autocorr,
        'slope': slope if len(valid_days) > 1 else None,
        'r_squared': r_value**2 if len(valid_days) > 1 else None
    }


def main():
    """Main function"""
    print("=" * 80)
    print("  Exploratory Data Analysis (EDA)")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    
    print(f"Training data shape: {train_tensor.shape}")
    print(f"Test data shape: {test_tensor.shape}")
    
    # 1. Statistical Summary
    stats_dict = compute_statistical_summary(train_tensor, test_tensor)
    
    # 2. Daily Statistics Analysis
    train_daily = analyze_daily_statistics(train_tensor, "Training")
    test_daily = analyze_daily_statistics(test_tensor, "Test")
    
    # 3. Spatial Correlation Analysis
    spatial_corr = analyze_spatial_correlation(train_tensor)
    
    # 4. Temporal Correlation Analysis
    temporal_corr = analyze_temporal_correlation(train_tensor)
    
    # Save summary results
    summary = {
        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'statistical_summary': stats_dict,
        'spatial_correlation': spatial_corr,
        'temporal_correlation': temporal_corr
    }
    
    summary_path = RESULTS_DIR / "eda_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ EDA summary results saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("  EDA Completed")
    print("=" * 80)
    print(f"\nAll results saved to:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Data: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
