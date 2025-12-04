"""
Generate Combined GP Global Uncertainty Figure
Combines the global uncertainty map and uncertainty vs missing analysis into one figure
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor

# Set matplotlib style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# Output directories
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"

# Data dimensions
LAT_SIZE = 100
LON_SIZE = 200
TIME_SIZE = 31


def create_combined_figure(pred_std, train_tensor, save_path=None):
    """
    Create a combined figure with:
    - Top row: Global uncertainty map (large)
    - Bottom row: Missing ratio map, Uncertainty map, Correlation scatter
    """
    # Calculate statistics
    mean_std = np.mean(pred_std, axis=2)  # Time-averaged uncertainty
    mask = (train_tensor == 0).astype(float)
    missing_ratio = np.mean(mask, axis=2)  # Missing ratio per location
    
    # Calculate correlation
    missing_flat = missing_ratio.flatten()
    std_flat = mean_std.flatten()
    corr = np.corrcoef(missing_flat, std_flat)[0, 1]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid spec for custom layout
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], 
                          hspace=0.25, wspace=0.25,
                          left=0.06, right=0.94, top=0.92, bottom=0.08)
    
    # ============ Top Row: Global Uncertainty Map (spans all 3 columns) ============
    ax_main = fig.add_subplot(gs[0, :])
    
    im_main = ax_main.imshow(mean_std, aspect='auto', origin='lower',
                              cmap='viridis',
                              extent=[-115, -105, 35, 40])
    
    cbar_main = plt.colorbar(im_main, ax=ax_main, label='Uncertainty (Std Dev, K)', 
                             shrink=0.8, pad=0.02)
    cbar_main.ax.tick_params(labelsize=10)
    
    ax_main.set_xlabel('Longitude (°W)', fontsize=12)
    ax_main.set_ylabel('Latitude (°N)', fontsize=12)
    ax_main.set_title('(a) GP Global Uncertainty Map (Time-Averaged over 31 Days)', 
                      fontsize=13, fontweight='bold', pad=10)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics box
    stats_text = (f"Min: {mean_std.min():.2f} K\n"
                  f"Max: {mean_std.max():.2f} K\n"
                  f"Mean: {mean_std.mean():.2f} K")
    ax_main.text(0.02, 0.96, stats_text, transform=ax_main.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # ============ Bottom Row: Three Analysis Panels ============
    
    # Panel (b): Missing ratio map
    ax_missing = fig.add_subplot(gs[1, 0])
    im_missing = ax_missing.imshow(missing_ratio, aspect='auto', origin='lower',
                                   cmap='YlOrRd', extent=[-115, -105, 35, 40])
    cbar_missing = plt.colorbar(im_missing, ax=ax_missing, label='Missing Ratio', shrink=0.9)
    ax_missing.set_xlabel('Longitude (°W)', fontsize=11)
    ax_missing.set_ylabel('Latitude (°N)', fontsize=11)
    ax_missing.set_title('(b) Training Set Missing Ratio', fontsize=12, fontweight='bold')
    ax_missing.grid(True, alpha=0.2, linestyle='--')
    
    # Panel (c): Uncertainty map (same as top but smaller for comparison)
    ax_unc = fig.add_subplot(gs[1, 1])
    im_unc = ax_unc.imshow(mean_std, aspect='auto', origin='lower',
                           cmap='viridis', extent=[-115, -105, 35, 40])
    cbar_unc = plt.colorbar(im_unc, ax=ax_unc, label='Uncertainty (K)', shrink=0.9)
    ax_unc.set_xlabel('Longitude (°W)', fontsize=11)
    ax_unc.set_ylabel('Latitude (°N)', fontsize=11)
    ax_unc.set_title('(c) GP Predictive Uncertainty', fontsize=12, fontweight='bold')
    ax_unc.grid(True, alpha=0.2, linestyle='--')
    
    # Panel (d): Correlation scatter
    ax_corr = fig.add_subplot(gs[1, 2])
    
    # Use hexbin for density visualization
    hb = ax_corr.hexbin(missing_flat * 100, std_flat, gridsize=25, 
                        cmap='Blues', mincnt=1)
    cbar_corr = plt.colorbar(hb, ax=ax_corr, label='Count', shrink=0.9)
    
    # Add linear fit
    coef = np.polyfit(missing_flat * 100, std_flat, 1)
    x_line = np.linspace(0, 100, 100)
    y_line = np.polyval(coef, x_line)
    ax_corr.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'Linear fit (slope={coef[0]:.5f})')
    
    ax_corr.set_xlabel('Missing Ratio (%)', fontsize=11)
    ax_corr.set_ylabel('Uncertainty (K)', fontsize=11)
    ax_corr.set_title(f'(d) Correlation Analysis (r = {corr:.3f})', 
                      fontsize=12, fontweight='bold')
    ax_corr.legend(fontsize=9, loc='upper right')
    ax_corr.grid(True, alpha=0.3)
    
    # Main figure title
    fig.suptitle('GP Global Uncertainty Analysis: Spatial Distribution and Missing Data Relationship',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.close()
    
    return corr


def main():
    """Main function"""
    print("="*60)
    print("  Generating Combined GP Uncertainty Figure")
    print("="*60)
    
    # Load pre-computed predictions
    results_dir = OUTPUT_DIR / "results"
    
    pred_std_path = results_dir / "gp_global_std.npy"
    if not pred_std_path.exists():
        print(f"❌ Error: {pred_std_path} not found!")
        print("Please run generate_gp_global_uncertainty.py first.")
        return
    
    print("\nLoading data...")
    pred_std = np.load(pred_std_path)
    train_tensor = load_modis_tensor("modis_aug_data/MODIS_Aug.mat", "training_tensor")
    
    print(f"  - pred_std shape: {pred_std.shape}")
    print(f"  - train_tensor shape: {train_tensor.shape}")
    
    # Generate combined figure
    print("\nGenerating combined figure...")
    save_path = FIGURES_DIR / "gp_global_uncertainty_combined.png"
    
    corr = create_combined_figure(pred_std, train_tensor, save_path=str(save_path))
    
    print(f"\n✅ Done! Correlation: {corr:.4f}")
    print(f"📊 Output: {save_path}")


if __name__ == "__main__":
    main()

