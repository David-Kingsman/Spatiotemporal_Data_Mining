"""
Generate GP Global Uncertainty Map

This script generates global uncertainty maps for the GP model,
analyzing how prediction uncertainty varies spatially across the entire domain.

Key Features:
1. Global spatial uncertainty averaging across all time steps
2. Temporal uncertainty analysis for each day
3. Spatial uncertainty patterns related to missing data ratio
4. Comprehensive uncertainty analysis and visualization
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor, create_spatial_temporal_coords
from lstinterp.models import GPSTModel, GPSTConfig

# Set matplotlib style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# Output directories
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data dimensions
LAT_SIZE = 100
LON_SIZE = 200
TIME_SIZE = 31


def load_gp_model(model_path: str, device: torch.device):
    """Load trained GP model"""
    print(f"Loading GP model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get config and inducing points
    config = checkpoint['config']
    inducing_points = checkpoint['inducing_points'].to(device)
    
    # Create model
    model = GPSTModel(inducing_points, config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  - Kernel: {config.kernel_space} x {config.kernel_time}")
    print(f"  - Inducing points: {len(inducing_points)}")
    
    return model


def predict_global(model, device, batch_size=5000):
    """
    Predict mean and uncertainty for the entire spatial-temporal domain
    Returns: pred_mean (100, 200, 31), pred_std (100, 200, 31)
    """
    import gpytorch
    
    print("\nGenerating global predictions...")
    print(f"  - Grid size: {LAT_SIZE} x {LON_SIZE} x {TIME_SIZE} = {LAT_SIZE * LON_SIZE * TIME_SIZE:,} points")
    
    # Create coordinate grid for all points
    coords = create_spatial_temporal_coords(
        lat_size=LAT_SIZE,
        lon_size=LON_SIZE, 
        time_size=TIME_SIZE,
        normalize=True
    )
    X = torch.FloatTensor(coords).to(device)
    
    # Batch prediction
    pred_mean_list = []
    pred_std_list = []
    
    n_batches = (len(X) + batch_size - 1) // batch_size
    print(f"  - Predicting in {n_batches} batches...")
    
    model.eval()
    model.likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            output = model.gp(batch)
            pred_dist = model.likelihood(output)
            
            pred_mean_list.append(pred_dist.mean.cpu().numpy())
            pred_std_list.append(pred_dist.stddev.cpu().numpy())
            
            if ((i // batch_size) + 1) % 20 == 0:
                print(f"    Batch {(i // batch_size) + 1}/{n_batches} completed")
    
    pred_mean = np.concatenate(pred_mean_list)
    pred_std = np.concatenate(pred_std_list)
    
    # Reshape to (lat, lon, time)
    pred_mean = pred_mean.reshape(LAT_SIZE, LON_SIZE, TIME_SIZE)
    pred_std = pred_std.reshape(LAT_SIZE, LON_SIZE, TIME_SIZE)
    
    print(f"  - Prediction completed!")
    print(f"  - Mean range: [{pred_mean.min():.2f}, {pred_mean.max():.2f}] K")
    print(f"  - Std range: [{pred_std.min():.2f}, {pred_std.max():.2f}] K")
    
    return pred_mean, pred_std


def plot_global_uncertainty_map(pred_std, save_path=None):
    """
    Plot global uncertainty map (time-averaged)
    """
    # Average uncertainty across time
    mean_std = np.mean(pred_std, axis=2)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Use a perceptually uniform colormap
    im = ax.imshow(mean_std, aspect='auto', origin='lower', 
                   cmap='viridis', 
                   extent=[-115, -105, 35, 40])
    
    cbar = plt.colorbar(im, ax=ax, label='Average Uncertainty (Std Dev, K)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel('Longitude (°W)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('GP Global Uncertainty Map\n(Time-Averaged Predictive Standard Deviation)', 
                 fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text
    stats_text = (f"Min: {mean_std.min():.2f} K\n"
                  f"Max: {mean_std.max():.2f} K\n"
                  f"Mean: {mean_std.mean():.2f} K\n"
                  f"Median: {np.median(mean_std):.2f} K")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.close()
    
    return mean_std


def plot_uncertainty_histogram(pred_std, save_path=None):
    """
    Plot histogram of uncertainty values
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: All pixels histogram
    all_std = pred_std.flatten()
    axes[0].hist(all_std, bins=100, color='steelblue', edgecolor='darkblue', alpha=0.7)
    axes[0].axvline(np.mean(all_std), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(all_std):.2f} K')
    axes[0].axvline(np.median(all_std), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(all_std):.2f} K')
    axes[0].set_xlabel('Uncertainty (Std Dev, K)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of All Uncertainty Values', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Time-averaged histogram
    time_avg_std = np.mean(pred_std, axis=2).flatten()
    axes[1].hist(time_avg_std, bins=50, color='seagreen', edgecolor='darkgreen', alpha=0.7)
    axes[1].axvline(np.mean(time_avg_std), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(time_avg_std):.2f} K')
    axes[1].axvline(np.median(time_avg_std), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(time_avg_std):.2f} K')
    axes[1].set_xlabel('Uncertainty (Std Dev, K)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Time-Averaged Uncertainty', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.close()


def plot_uncertainty_by_day(pred_std, save_path=None):
    """
    Plot uncertainty statistics by day
    """
    daily_mean = np.mean(pred_std, axis=(0, 1))
    daily_min = np.min(pred_std, axis=(0, 1))
    daily_max = np.max(pred_std, axis=(0, 1))
    daily_std = np.std(pred_std, axis=(0, 1))
    
    days = np.arange(1, TIME_SIZE + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Mean uncertainty with error band
    axes[0].fill_between(days, daily_mean - daily_std, daily_mean + daily_std,
                         alpha=0.3, color='steelblue', label='±1 Std')
    axes[0].plot(days, daily_mean, 'o-', color='steelblue', linewidth=2, 
                markersize=5, label='Mean Uncertainty')
    axes[0].plot(days, daily_min, '--', color='green', linewidth=1.5, 
                alpha=0.7, label='Min Uncertainty')
    axes[0].plot(days, daily_max, '--', color='red', linewidth=1.5,
                alpha=0.7, label='Max Uncertainty')
    
    axes[0].set_xlabel('Day (August 2020)', fontsize=12)
    axes[0].set_ylabel('Uncertainty (Std Dev, K)', fontsize=12)
    axes[0].set_title('Daily Uncertainty Statistics', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(days[::2])
    
    # Bottom: Coefficient of Variation (CV)
    cv = daily_std / daily_mean * 100  # Coefficient of variation in %
    axes[1].bar(days, cv, color='coral', edgecolor='darkred', alpha=0.7)
    axes[1].axhline(np.mean(cv), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean CV: {np.mean(cv):.1f}%')
    axes[1].set_xlabel('Day (August 2020)', fontsize=12)
    axes[1].set_ylabel('Coefficient of Variation (%)', fontsize=12)
    axes[1].set_title('Daily Uncertainty Variability (CV = Std/Mean)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(days[::2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.close()


def plot_uncertainty_vs_missing(pred_std, train_tensor, save_path=None):
    """
    Analyze relationship between uncertainty and missing data ratio
    """
    # Calculate missing ratio per location (averaged over time)
    mask = (train_tensor == 0).astype(float)
    missing_ratio = np.mean(mask, axis=2)  # (100, 200)
    
    # Average uncertainty per location
    mean_std = np.mean(pred_std, axis=2)  # (100, 200)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left: Missing ratio map
    im1 = axes[0].imshow(missing_ratio, aspect='auto', origin='lower',
                         cmap='YlOrRd', extent=[-115, -105, 35, 40])
    cbar1 = plt.colorbar(im1, ax=axes[0], label='Missing Ratio', shrink=0.8)
    axes[0].set_xlabel('Longitude (°W)', fontsize=12)
    axes[0].set_ylabel('Latitude (°N)', fontsize=12)
    axes[0].set_title('Missing Data Ratio (Training Set)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Middle: Uncertainty map
    im2 = axes[1].imshow(mean_std, aspect='auto', origin='lower',
                         cmap='viridis', extent=[-115, -105, 35, 40])
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Uncertainty (K)', shrink=0.8)
    axes[1].set_xlabel('Longitude (°W)', fontsize=12)
    axes[1].set_ylabel('Latitude (°N)', fontsize=12)
    axes[1].set_title('GP Prediction Uncertainty', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Right: Scatter plot of correlation
    missing_flat = missing_ratio.flatten()
    std_flat = mean_std.flatten()
    
    # Hexbin for better visualization of dense data
    hb = axes[2].hexbin(missing_flat * 100, std_flat, gridsize=30, cmap='Blues', mincnt=1)
    cbar3 = plt.colorbar(hb, ax=axes[2], label='Count', shrink=0.8)
    
    # Fit linear regression
    from numpy.polynomial import polynomial as P
    coef = np.polyfit(missing_flat * 100, std_flat, 1)
    x_line = np.linspace(0, 100, 100)
    y_line = np.polyval(coef, x_line)
    axes[2].plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'Linear fit: y={coef[0]:.4f}x+{coef[1]:.2f}')
    
    # Calculate correlation
    corr = np.corrcoef(missing_flat, std_flat)[0, 1]
    
    axes[2].set_xlabel('Missing Ratio (%)', fontsize=12)
    axes[2].set_ylabel('Uncertainty (K)', fontsize=12)
    axes[2].set_title(f'Uncertainty vs Missing Ratio\n(Correlation: {corr:.3f})', 
                      fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.close()
    
    return corr


def plot_daily_uncertainty_maps(pred_std, days_to_plot=[1, 8, 15, 22, 30], save_path=None):
    """
    Plot uncertainty maps for selected days
    """
    n_days = len(days_to_plot)
    fig, axes = plt.subplots(1, n_days, figsize=(4*n_days, 4.5))
    
    # Find global colorbar range
    vmin, vmax = pred_std.min(), pred_std.max()
    
    for i, day in enumerate(days_to_plot):
        day_idx = day - 1
        std_map = pred_std[:, :, day_idx]
        
        im = axes[i].imshow(std_map, aspect='auto', origin='lower',
                           cmap='viridis', vmin=vmin, vmax=vmax,
                           extent=[-115, -105, 35, 40])
        axes[i].set_xlabel('Longitude (°W)', fontsize=10)
        if i == 0:
            axes[i].set_ylabel('Latitude (°N)', fontsize=10)
        axes[i].set_title(f'Day {day}', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.2, linestyle='--')
    
    # Add single colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Uncertainty (K)')
    
    fig.suptitle('GP Daily Uncertainty Maps (Selected Days)', fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.close()


def plot_spatial_uncertainty_profiles(pred_std, save_path=None):
    """
    Plot spatial profiles of uncertainty (latitudinal and longitudinal averages)
    """
    mean_std = np.mean(pred_std, axis=2)  # (100, 200)
    
    # Latitudinal average (average over longitude)
    lat_avg = np.mean(mean_std, axis=1)  # (100,)
    lat_std = np.std(mean_std, axis=1)
    
    # Longitudinal average (average over latitude)
    lon_avg = np.mean(mean_std, axis=0)  # (200,)
    lon_std = np.std(mean_std, axis=0)
    
    lat_coords = np.linspace(35, 40, LAT_SIZE)
    lon_coords = np.linspace(-115, -105, LON_SIZE)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Latitudinal profile
    axes[0].fill_between(lat_coords, lat_avg - lat_std, lat_avg + lat_std,
                         alpha=0.3, color='steelblue')
    axes[0].plot(lat_coords, lat_avg, '-', color='steelblue', linewidth=2)
    axes[0].set_xlabel('Latitude (°N)', fontsize=12)
    axes[0].set_ylabel('Mean Uncertainty (K)', fontsize=12)
    axes[0].set_title('Latitudinal Uncertainty Profile\n(Averaged over Longitude)', 
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Right: Longitudinal profile
    axes[1].fill_between(lon_coords, lon_avg - lon_std, lon_avg + lon_std,
                         alpha=0.3, color='coral')
    axes[1].plot(lon_coords, lon_avg, '-', color='coral', linewidth=2)
    axes[1].set_xlabel('Longitude (°W)', fontsize=12)
    axes[1].set_ylabel('Mean Uncertainty (K)', fontsize=12)
    axes[1].set_title('Longitudinal Uncertainty Profile\n(Averaged over Latitude)',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.close()


def generate_analysis_report(pred_std, train_tensor, corr_missing):
    """
    Generate and print analysis report
    """
    mean_std = np.mean(pred_std, axis=2)
    
    print("\n" + "="*80)
    print("  GP Global Uncertainty Analysis Report")
    print("="*80)
    
    print("\n[1. Overall Uncertainty Statistics]")
    print(f"  - Min uncertainty: {pred_std.min():.4f} K")
    print(f"  - Max uncertainty: {pred_std.max():.4f} K")
    print(f"  - Mean uncertainty: {pred_std.mean():.4f} K")
    print(f"  - Median uncertainty: {np.median(pred_std):.4f} K")
    print(f"  - Std of uncertainty: {pred_std.std():.4f} K")
    
    print("\n[2. Spatial Patterns (Time-averaged)]")
    print(f"  - Min spatial uncertainty: {mean_std.min():.4f} K")
    print(f"  - Max spatial uncertainty: {mean_std.max():.4f} K")
    print(f"  - Mean spatial uncertainty: {mean_std.mean():.4f} K")
    print(f"  - Spatial variability (Std): {mean_std.std():.4f} K")
    
    # Find highest/lowest uncertainty regions
    max_idx = np.unravel_index(np.argmax(mean_std), mean_std.shape)
    min_idx = np.unravel_index(np.argmin(mean_std), mean_std.shape)
    
    lat_max = 35 + max_idx[0] * 5 / LAT_SIZE
    lon_max = -115 + max_idx[1] * 10 / LON_SIZE
    lat_min = 35 + min_idx[0] * 5 / LAT_SIZE
    lon_min = -115 + min_idx[1] * 10 / LON_SIZE
    
    print(f"\n  - Highest uncertainty location: ({lat_max:.2f}°N, {lon_max:.2f}°W)")
    print(f"    Value: {mean_std[max_idx]:.4f} K")
    print(f"  - Lowest uncertainty location: ({lat_min:.2f}°N, {lon_min:.2f}°W)")
    print(f"    Value: {mean_std[min_idx]:.4f} K")
    
    print("\n[3. Temporal Patterns]")
    daily_mean = np.mean(pred_std, axis=(0, 1))
    daily_std = np.std(pred_std, axis=(0, 1))
    
    most_uncertain_day = np.argmax(daily_mean) + 1
    least_uncertain_day = np.argmin(daily_mean) + 1
    
    print(f"  - Daily mean uncertainty range: [{daily_mean.min():.4f}, {daily_mean.max():.4f}] K")
    print(f"  - Most uncertain day: Day {most_uncertain_day} ({daily_mean.max():.4f} K)")
    print(f"  - Least uncertain day: Day {least_uncertain_day} ({daily_mean.min():.4f} K)")
    print(f"  - Temporal variability: {np.std(daily_mean):.4f} K")
    
    print("\n[4. Correlation with Missing Data]")
    print(f"  - Correlation coefficient: {corr_missing:.4f}")
    if corr_missing > 0.3:
        print(f"  ✓ Positive correlation: Higher missing rate → Higher uncertainty")
        print(f"    This is expected behavior! GP correctly captures data scarcity.")
    elif corr_missing < -0.3:
        print(f"  ⚠ Negative correlation: Unexpected behavior")
    else:
        print(f"  - Weak correlation: Uncertainty is relatively uniform")
    
    print("\n[5. GP Characteristics Summary]")
    print("  ✓ Global prediction capability: GP provides predictions everywhere")
    print("  ✓ Uncertainty quantification: Full posterior uncertainty available")
    print("  ✓ Spatial coherence: Uncertainty varies smoothly due to kernel structure")
    
    # Calculate uncertainty coverage
    pct_low = np.sum(pred_std < 4.0) / pred_std.size * 100
    pct_med = np.sum((pred_std >= 4.0) & (pred_std < 5.0)) / pred_std.size * 100
    pct_high = np.sum(pred_std >= 5.0) / pred_std.size * 100
    
    print(f"\n[6. Uncertainty Distribution]")
    print(f"  - Low uncertainty (<4K): {pct_low:.1f}%")
    print(f"  - Medium uncertainty (4-5K): {pct_med:.1f}%")
    print(f"  - High uncertainty (≥5K): {pct_high:.1f}%")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    print("="*80)
    print("  GP Global Uncertainty Map Generation")
    print("="*80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load model
    model_path = OUTPUT_DIR / "models" / "gp_model.pth"
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please run train_gp.py first!")
        return
    
    model = load_gp_model(str(model_path), device)
    
    # Load training data (for missing ratio analysis)
    train_tensor = load_modis_tensor("modis_aug_data/MODIS_Aug.mat", "training_tensor")
    print(f"\nTraining data shape: {train_tensor.shape}")
    
    # Generate global predictions
    pred_mean, pred_std = predict_global(model, device)
    
    # Save predictions
    np.save(OUTPUT_DIR / "results" / "gp_global_mean.npy", pred_mean)
    np.save(OUTPUT_DIR / "results" / "gp_global_std.npy", pred_std)
    print(f"✅ Saved predictions to {OUTPUT_DIR / 'results'}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Main global uncertainty map
    mean_std = plot_global_uncertainty_map(
        pred_std, 
        save_path=str(FIGURES_DIR / "gp_global_uncertainty_map.png")
    )
    
    # 2. Uncertainty histogram
    plot_uncertainty_histogram(
        pred_std,
        save_path=str(FIGURES_DIR / "gp_uncertainty_histogram.png")
    )
    
    # 3. Daily uncertainty statistics
    plot_uncertainty_by_day(
        pred_std,
        save_path=str(FIGURES_DIR / "gp_uncertainty_by_day.png")
    )
    
    # 4. Uncertainty vs missing ratio
    corr_missing = plot_uncertainty_vs_missing(
        pred_std, train_tensor,
        save_path=str(FIGURES_DIR / "gp_uncertainty_vs_missing.png")
    )
    
    # 5. Daily uncertainty maps
    plot_daily_uncertainty_maps(
        pred_std,
        save_path=str(FIGURES_DIR / "gp_daily_uncertainty_maps.png")
    )
    
    # 6. Spatial profiles
    plot_spatial_uncertainty_profiles(
        pred_std,
        save_path=str(FIGURES_DIR / "gp_spatial_uncertainty_profiles.png")
    )
    
    # Generate analysis report
    generate_analysis_report(pred_std, train_tensor, corr_missing)
    
    # Save statistics to JSON
    stats = {
        "overall": {
            "min": float(pred_std.min()),
            "max": float(pred_std.max()),
            "mean": float(pred_std.mean()),
            "median": float(np.median(pred_std)),
            "std": float(pred_std.std())
        },
        "time_averaged": {
            "min": float(mean_std.min()),
            "max": float(mean_std.max()),
            "mean": float(mean_std.mean()),
            "std": float(mean_std.std())
        },
        "correlation_with_missing": float(corr_missing),
        "daily_statistics": {
            "mean_per_day": [float(x) for x in np.mean(pred_std, axis=(0, 1))],
            "std_per_day": [float(x) for x in np.std(pred_std, axis=(0, 1))]
        }
    }
    
    with open(OUTPUT_DIR / "results" / "gp_global_uncertainty_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ Statistics saved to {OUTPUT_DIR / 'results' / 'gp_global_uncertainty_stats.json'}")
    
    print("\n" + "="*80)
    print("  All done!")
    print("="*80)


if __name__ == "__main__":
    main()

