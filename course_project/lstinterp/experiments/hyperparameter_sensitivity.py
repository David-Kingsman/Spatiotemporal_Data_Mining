"""
Hyperparameter Sensitivity Analysis

This script analyzes the impact of different hyperparameter settings on model performance:
1. U-Net: Learning rate, batch size, base channels
2. GP: Learning rate, number of inducing points, lengthscales
3. Tree: Number of estimators, max depth
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from itertools import product

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import ProbUNet, TreeBaseline
from lstinterp.config import UNetConfig, TreeConfig
from lstinterp.metrics import rmse, r2, crps_gaussian
from lstinterp.utils import set_seed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Output directory
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "hyperparameter_sensitivity"
RESULTS_DIR = OUTPUT_DIR / "results" / "hyperparameter_sensitivity"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data path
DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

set_seed(42)


def print_section_header(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_unet_hyperparameters():
    """Analyze U-Net hyperparameter sensitivity"""
    print_section_header("U-Net Hyperparameter Sensitivity Analysis")
    
    # Load data
    train_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    
    H, W, T = train_tensor.shape
    
    # Calculate normalization parameters
    train_mask = (train_tensor != 0)
    train_mean = train_tensor[train_mask].mean()
    train_std = train_tensor[train_mask].std()
    
    # Create dataset (use smaller subset for speed)
    train_indices = list(range(28))  # Use 28 days for training
    val_indices = list(range(28, 31))  # Use 3 days for validation
    test_indices = list(range(T))
    
    train_dataset = MODISDataset(train_tensor, mode="image", 
                                norm_mean=train_mean, norm_std=train_std)
    test_dataset = MODISDataset(test_tensor, mode="image",
                               norm_mean=train_mean, norm_std=train_std)
    
    # Hyperparameter search space (simplified, only test key parameters)
    hyperparameter_space = {
        'lr': [0.0001, 0.0005, 0.001],
        'batch_size': [2, 4, 8],
        'base_channels': [16, 32, 64]
    }
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Number of hyperparameter combinations: {len(list(product(*hyperparameter_space.values())))}")
    print("\nStarting hyperparameter search...")
    
    for lr, batch_size, base_channels in product(
        hyperparameter_space['lr'],
        hyperparameter_space['batch_size'],
        hyperparameter_space['base_channels']
    ):
        print(f"\nTesting: lr={lr}, batch_size={batch_size}, base_channels={base_channels}")
        
        try:
            # Create model
            config = UNetConfig(
                in_channels=2,
                base_channels=base_channels,
                lr=lr,
                num_epochs=10,  # Reduced epochs for speed
                batch_size=batch_size
            )
            # Add dropout attribute if missing
            if not hasattr(config, 'dropout'):
                config.dropout = 0.2
            
            model = ProbUNet(config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            
            # Train (simplified, only a few epochs)
            model.train()
            train_loader = DataLoader(
                [train_dataset[i] for i in train_indices],
                batch_size=batch_size,
                shuffle=True
            )
            
            train_losses = []
            for epoch in range(5):  # Only 5 epochs
                epoch_loss = 0
                n_batches = 0
                
                for img, mask, target in train_loader:
                    img = img.to(device)
                    mask = mask.to(device)
                    target = target.to(device)
                    x = torch.cat([img, mask], dim=1)
                    
                    optimizer.zero_grad()
                    mean, log_var = model(x)
                    
                    # Calculate loss
                    var = torch.exp(torch.clamp(log_var, -10, 10)) + 1e-6
                    nll = 0.5 * (torch.log(2 * np.pi * var) + (target - mean)**2 / var)
                    loss = (nll * mask).sum() / mask.sum().clamp_min(1.0)
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_batches += 1
                
                if n_batches > 0:
                    avg_loss = epoch_loss / n_batches
                    train_losses.append(avg_loss)
            
            # Evaluate (on a subset of test set)
            model.eval()
            test_subset = [test_dataset[i] for i in test_indices[:5]]  # Only 5 days
            
            pred_means = []
            pred_stds = []
            targets = []
            
            with torch.no_grad():
                for img, mask, target in test_subset:
                    img = img.unsqueeze(0).to(device)
                    mask_np = mask.squeeze().numpy()
                    x = torch.cat([img, mask.unsqueeze(0).to(device)], dim=1)
                    
                    mean, log_var = model(x)
                    std = torch.sqrt(torch.exp(torch.clamp(log_var, -10, 10)) + 1e-6)
                    
                    mean_np = mean.squeeze().cpu().numpy() * train_std + train_mean
                    std_np = std.squeeze().cpu().numpy() * train_std
                    target_np = target.squeeze().numpy() * train_std + train_mean
                    
                    # Evaluate only on observed points
                    mask_2d = mask_np > 0
                    pred_means.append(mean_np[mask_2d])
                    pred_stds.append(std_np[mask_2d])
                    targets.append(target_np[mask_2d])
            
            # Calculate metrics
            y_true = np.concatenate(targets)
            y_pred = np.concatenate(pred_means)
            y_std = np.concatenate(pred_stds)
            
            test_rmse = rmse(y_true, y_pred)
            test_r2 = r2(y_true, y_pred)
            test_crps = crps_gaussian(y_true, y_pred, y_std)
            
            results.append({
                'lr': lr,
                'batch_size': batch_size,
                'base_channels': base_channels,
                'rmse': test_rmse,
                'r2': test_r2,
                'crps': test_crps,
                'final_loss': train_losses[-1] if train_losses else np.nan
            })
            
            print(f"  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, CRPS: {test_crps:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "unet_hyperparameter_sensitivity.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Results saved: {csv_path}")
    
    # Visualization
    if len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Learning rate impact
        lr_df = df.groupby('lr').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        axes[0, 0].plot(lr_df['lr'], lr_df['rmse'], 'o-', label='RMSE', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('RMSE (K)')
        axes[0, 0].set_title('Effect of Learning Rate', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        
        # Batch size impact
        bs_df = df.groupby('batch_size').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        axes[0, 1].plot(bs_df['batch_size'], bs_df['rmse'], 's-', label='RMSE', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('RMSE (K)')
        axes[0, 1].set_title('Effect of Batch Size', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Base channels impact
        ch_df = df.groupby('base_channels').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        axes[1, 0].plot(ch_df['base_channels'], ch_df['rmse'], '^-', label='RMSE', linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xlabel('Base Channels')
        axes[1, 0].set_ylabel('RMSE (K)')
        axes[1, 0].set_title('Effect of Base Channels', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Heatmap: LR vs Batch Size
        pivot = df.pivot_table(values='rmse', index='lr', columns='batch_size', aggfunc='mean')
        im = axes[1, 1].imshow(pivot.values, aspect='auto', cmap='viridis_r', origin='lower')
        axes[1, 1].set_xticks(range(len(pivot.columns)))
        axes[1, 1].set_xticklabels(pivot.columns)
        axes[1, 1].set_yticks(range(len(pivot.index)))
        axes[1, 1].set_yticklabels([f"{x:.4f}" for x in pivot.index])
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('RMSE Heatmap: LR vs Batch Size', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1], label='RMSE (K)')
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "unet_hyperparameter_sensitivity.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Visualization saved: {fig_path}")
    
    return df


def analyze_tree_hyperparameters():
    """Analyze Tree hyperparameter sensitivity"""
    print_section_header("Tree Hyperparameter Sensitivity Analysis")
    
    # Load data
    train_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    
    # Create dataset
    train_dataset = MODISDataset(train_tensor, mode="point", normalize_coords=True)
    test_dataset = MODISDataset(test_tensor, mode="point", normalize_coords=True)
    
    # Sample for speed
    train_size = min(50000, len(train_dataset))
    test_size = min(10000, len(test_dataset))
    
    train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_size, replace=False)
    
    X_train = train_dataset.coords[train_indices]
    y_train = train_dataset.values[train_indices]
    X_test = test_dataset.coords[test_indices]
    y_test = test_dataset.values[test_indices]
    
    # Hyperparameter search space
    hyperparameter_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8, None]
    }
    
    results = []
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Number of hyperparameter combinations: {len(list(product(*hyperparameter_space.values())))}")
    print("\nStarting hyperparameter search...")
    
    for n_estimators, max_depth in product(
        hyperparameter_space['n_estimators'],
        hyperparameter_space['max_depth']
    ):
        print(f"\nTesting: n_estimators={n_estimators}, max_depth={max_depth}")
        
        try:
            config = TreeConfig(
                model_type="xgb",
                n_estimators=n_estimators,
                max_depth=max_depth,
                quantile_regression=True
            )
            # quantiles is automatically set in TreeBaseline.__init__
            
            model = TreeBaseline(config)
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred_mean, y_pred_std = model.predict_with_uncertainty(X_test)
            
            test_rmse = rmse(y_test, y_pred_mean)
            test_r2 = r2(y_test, y_pred_mean)
            test_crps = crps_gaussian(y_test, y_pred_mean, y_pred_std)
            
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth if max_depth else -1,  # -1 for None
                'rmse': test_rmse,
                'r2': test_r2,
                'crps': test_crps,
                'training_time': training_time
            })
            
            print(f"  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, CRPS: {test_crps:.4f}, Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "tree_hyperparameter_sensitivity.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Results saved: {csv_path}")
    
    # Visualization
    if len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Number of estimators impact
        n_est_df = df.groupby('n_estimators').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        axes[0, 0].plot(n_est_df['n_estimators'], n_est_df['rmse'], 'o-', label='RMSE', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Estimators')
        axes[0, 0].set_ylabel('RMSE (K)')
        axes[0, 0].set_title('Effect of Number of Estimators', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Max depth impact
        depth_df = df[df['max_depth'] > 0].groupby('max_depth').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        if len(depth_df) > 0:
            axes[0, 1].plot(depth_df['max_depth'], depth_df['rmse'], 's-', label='RMSE', linewidth=2, markersize=8, color='green')
            axes[0, 1].set_xlabel('Max Depth')
            axes[0, 1].set_ylabel('RMSE (K)')
            axes[0, 1].set_title('Effect of Max Depth', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Training Time vs Performance
        axes[1, 0].scatter(df['training_time'], df['rmse'], c=df['n_estimators'], 
                          cmap='viridis', s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Training Time (s)')
        axes[1, 0].set_ylabel('RMSE (K)')
        axes[1, 0].set_title('Training Time vs Performance', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
        cbar.set_label('n_estimators')
        
        # Heatmap: n_estimators vs max_depth
        pivot = df.pivot_table(values='rmse', index='n_estimators', columns='max_depth', aggfunc='mean')
        im = axes[1, 1].imshow(pivot.values, aspect='auto', cmap='viridis_r', origin='lower')
        axes[1, 1].set_xticks(range(len(pivot.columns)))
        axes[1, 1].set_xticklabels([str(x) if x > 0 else 'None' for x in pivot.columns])
        axes[1, 1].set_yticks(range(len(pivot.index)))
        axes[1, 1].set_yticklabels(pivot.index)
        axes[1, 1].set_xlabel('Max Depth')
        axes[1, 1].set_ylabel('Number of Estimators')
        axes[1, 1].set_title('RMSE Heatmap: n_estimators vs max_depth', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1], label='RMSE (K)')
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "tree_hyperparameter_sensitivity.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Visualization saved: {fig_path}")
    
    return df


def main():
    """Main function"""
    print("=" * 80)
    print("  Hyperparameter Sensitivity Analysis")
    print("=" * 80)
    
    print("\nNote: To speed up analysis, simplified training settings are used")
    print("      (U-Net: 5 epochs, Tree: sampled data)")
    
    # Analyze U-Net (if time permits)
    print("\n" + "=" * 80)
    print("  Starting U-Net Hyperparameter Analysis (may take a while)...")
    print("=" * 80)
    try:
        unet_df = analyze_unet_hyperparameters()
    except Exception as e:
        print(f"⚠️  Error in U-Net analysis: {e}")
        unet_df = None
    
    # Analyze Tree
    print("\n" + "=" * 80)
    print("  Starting Tree Hyperparameter Analysis...")
    print("=" * 80)
    try:
        tree_df = analyze_tree_hyperparameters()
    except Exception as e:
        print(f"⚠️  Error in Tree analysis: {e}")
        tree_df = None
    
    print("\n" + "=" * 80)
    print("  Hyperparameter Sensitivity Analysis Completed")
    print("=" * 80)
    print(f"\nAll results saved to:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Data: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
