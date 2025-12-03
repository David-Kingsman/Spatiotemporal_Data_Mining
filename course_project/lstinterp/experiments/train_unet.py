# -*- coding: utf-8 -*-
"""
Train Probabilistic U-Net Model

This script implements a Probabilistic U-Net model based on the U-Net architecture,
used for image-level interpolation and prediction of MODIS Land Surface Temperature (LST) data.

Key Features:
1. Probabilistic Output: Outputs mean and variance (log_var) for each pixel, providing uncertainty estimates
2. U-Net Architecture: Encoder-Decoder structure, suitable for image inpainting tasks
3. Batch Normalization: Improves training stability
4. Dropout Regularization: Prevents overfitting
5. Negative Log-Likelihood Loss: Probabilistic loss function based on Gaussian assumption

Data Format:
- Input: 3D tensor (H, W, T) = (100, 200, 31)
  - H: Latitude dimension (35°-40°N)
  - W: Longitude dimension (-115°--105°W)
  - T: Time dimension (31 days)
- Output: Temperature values (Unit: Kelvin)
- Missing values: Represented by 0

Evaluation Metrics:
- Regression Metrics: RMSE, MAE, R², MAPE
- Probabilistic Metrics: CRPS, 90% Prediction Interval Coverage, Calibration Error

Author: lstinterp team
Created: 2024
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import ProbUNet, UNetConfig, gaussian_nll_loss
from lstinterp.metrics import compute_regression_metrics, compute_probabilistic_metrics
from lstinterp.viz import plot_mean_map, plot_std_map, plot_error_map
from lstinterp.utils import set_seed

# Create output directories
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "results").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "models").mkdir(exist_ok=True)


def print_section_header(title, width=80):
    """Print section header"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    """Main function: Train and evaluate U-Net model"""
    start_time = time.time()
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_section_header("Experiment Configuration", width=80)
    print(f"Experiment Time: {experiment_time}")
    print(f"Random Seed: 42")
    
    set_seed(42)
    
    # Check dependencies
    print("\nDependency Check:")
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch not installed")
        return
    
    try:
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
    except ImportError:
        print("  ❌ NumPy not installed")
        return
    
    # Set device (after importing torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing Device: {device}")
    if device.type == "cuda":
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    print_section_header("Data Loading")
    data_path = "modis_aug_data/MODIS_Aug.mat"
    print(f"Data Path: {data_path}")
    
    print("\nLoading training data...")
    train_tensor = load_modis_tensor(data_path, "training_tensor")
    H, W, T = train_tensor.shape
    print(f"Training data dimensions: {H} x {W} x {T}")
    
    print("\nLoading test data...")
    test_tensor = load_modis_tensor(data_path, "test_tensor")
    print(f"Test data dimensions: {H} x {W} x {T}")
    
    # Create dataset
    print_section_header("Data Preprocessing")
    print("Converting to image data format (T, 1, H, W) -> (mean, log_var)")
    
    print("\nCreating training dataset (Image Mode)...")
    train_dataset = MODISDataset(train_tensor, mode="image")
    print(f"  - Training images: {len(train_dataset)} (1 per day)")
    print(f"  - Image size: {H} x {W} pixels")
    
    # Get normalization statistics from training set (for consistency during testing)
    train_mean = train_dataset.mean_val
    train_std = train_dataset.std_val
    print(f"\nData Normalization Statistics (Z-score):")
    print(f"  - Mean: {train_mean:.2f} K")
    print(f"  - Std: {train_std:.2f} K")
    print(f"  - Normalized range: Approx [{train_mean - 3*train_std:.2f}, {train_mean + 3*train_std:.2f}] K")
    
    # Data Loader (Improved config)
    print_section_header("Model Configuration")
    config = UNetConfig(
        batch_size=4,          # Batch size (adjust based on GPU memory)
        num_epochs=50,         # Number of epochs
        lr=5e-4,               # Learning rate
        dropout=0.2,           # Dropout rate (prevent overfitting)
        init_log_var=-1.0      # Initial log_var=-1, corresponds to std approx 0.37 (reasonable after normalization)
    )
    
    print("Model Hyperparameters:")
    print(f"  - Batch Size: {config.batch_size}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Learning Rate: {config.lr}")
    print(f"  - Dropout: {config.dropout}")
    print(f"  - Initial log_var: {config.init_log_var}")
    print(f"  - Input Channels: {config.in_channels} (Temp + Mask)")
    print(f"  - Base Channels: {config.base_channels}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatible
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create validation set (using a subset of training data)
    print("\nCreating Train/Val Split...")
    train_size = len(train_dataset)
    val_size = max(1, int(train_size * 0.1))  # 10% as validation set
    indices = np.random.RandomState(42).permutation(train_size)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    print(f"  - Training Set: {len(train_indices)} images ({len(train_indices)/train_size*100:.1f}%)")
    print(f"  - Validation Set: {len(val_indices)} images ({len(val_indices)/train_size*100:.1f}%)")
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    print("\nCreating Model...")
    model = ProbUNet(config).to(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    
    # Model structure description
    print("\nModel Structure:")
    print("  - Architecture: U-Net (Encoder-Decoder)")
    print("  - Encoder: Conv Layers + MaxPool")
    print("  - Decoder: Transpose Conv + Upsampling")
    print("  - Skip Connections: Connect encoder and decoder layers")
    print("  - Output: mean (B, 1, H, W) and log_var (B, 1, H, W)")
    
    # Optimizer (with LR scheduler)
    print("\nOptimizer Configuration:")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    print(f"  - Optimizer: Adam")
    print(f"  - Learning Rate: {config.lr}")
    print(f"  - Weight Decay: 1e-5 (L2 Regularization)")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    print(f"  - LR Scheduler: ReduceLROnPlateau")
    print(f"  - Factor: 0.5")
    print(f"  - Patience: 5 epochs")
    
    print(f"\nLoss Function:")
    print(f"  - Type: Gaussian Negative Log-Likelihood")
    print(f"  - Calculated only on observed points (mask > 0.5)")
    
    # Training (with validation monitoring)
    print_section_header("Model Training")
    best_loss = float('inf')
    best_epoch = 1
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    training_start_time = time.time()
    
    print(f"Starting Training ({config.num_epochs} epochs)...")
    print("-" * 100)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Best Val':<15} {'LR':<15} {'Time':<10}")
    print("-" * 100)
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        # Training Phase
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch_idx, (img, mask, target) in enumerate(train_loader):
            img = img.to(device)
            mask = mask.to(device)
            target = target.to(device)
            
            x = torch.cat([img, mask], dim=1)
            
            optimizer.zero_grad()
            mean, log_var = model(x)
            loss = gaussian_nll_loss(mean, log_var, target, mask)
            
            if torch.isnan(loss) or torch.isinf(loss):
                if batch_idx == 0:  # Print warning only for first batch
                    print(f"    ⚠️  Warning: Invalid loss detected (NaN/Inf), skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for img, mask, target in val_loader:
                img = img.to(device)
                mask = mask.to(device)
                target = target.to(device)
                x = torch.cat([img, mask], dim=1)
                
                mean, log_var = model(x)
                loss = gaussian_nll_loss(mean, log_var, target, mask)
                
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    n_val_batches += 1
        
        avg_val_loss = val_loss / max(n_val_batches, 1) if n_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # LR Scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_reduced = (new_lr < old_lr)
        
        # Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        status = "⭐" if avg_val_loss == best_loss else ("📉" if lr_reduced else " ")
        
        # Print every 5 epochs or last epoch or first epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs or (epoch + 1) == 1:
            print(f"{epoch+1:<8} {avg_train_loss:<15.4f} {avg_val_loss:<15.4f} {best_loss:<15.4f} {current_lr:<15.6f} {epoch_time:<10.2f}s {status}")
        
        # Early Stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (patience={patience}), restoring best model (Epoch {best_epoch})")
            model.load_state_dict(best_model_state)
            break
    
    training_time = time.time() - training_start_time
    print("-" * 100)
    print(f"Training Completed!")
    print(f"  - Total Training Time: {training_time:.2f} s ({training_time/60:.2f} min)")
    print(f"  - Best Validation Loss: {best_loss:.4f} (Epoch {best_epoch})")
    print(f"  - Final Training Loss: {avg_train_loss:.4f}")
    print(f"  - Final Validation Loss: {avg_val_loss:.4f}")
    print(f"  - Avg Time per Epoch: {training_time/(epoch+1):.2f} s")
    
    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
        print(f"\nLoaded Best Model (Epoch {best_epoch}, Val Loss={best_loss:.4f})")
    
    # Evaluation
    print_section_header("Model Evaluation")
    evaluation_start_time = time.time()
    model.eval()
    
    # Evaluate on test data (using training set normalization parameters)
    test_dataset = MODISDataset(test_tensor, mode="image", norm_mean=train_mean, norm_std=train_std)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_preds_mean = []
    all_preds_std = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for img, mask, target in test_loader:
            img = img.to(device)
            mask = mask.to(device)
            target = target.to(device)
            x = torch.cat([img, mask], dim=1)
            
            mean, log_var = model(x)
            std = torch.exp(0.5 * log_var)
            
            # Move to CPU then numpy
            all_preds_mean.append(mean.cpu().numpy())
            all_preds_std.append(std.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
    
    # Concatenate results
    pred_mean = np.concatenate(all_preds_mean, axis=0)[:, 0, :, :]  # (T, H, W)
    pred_std = np.concatenate(all_preds_std, axis=0)[:, 0, :, :]
    targets = np.concatenate(all_targets, axis=0)[:, 0, :, :]
    masks = np.concatenate(all_masks, axis=0)[:, 0, :, :]
    
    # Denormalize predictions (restore to original scale)
    # Use training set statistics (consistent with training)
    mean_val = train_mean
    std_val = train_std
    
    # IMPORTANT: For proper interpolation evaluation, we need to evaluate on held-out points
    # Strategy: Randomly mask out 20% of observed points in test set, then evaluate on those
    print("\n⚠️  WARNING: Previous evaluation was only on observed points (data leakage risk)")
    print("   Implementing proper interpolation evaluation: masking 20% of test observations...")
    
    # Create evaluation mask: randomly hold out 20% of observed points
    observed_mask = masks > 0.5
    n_observed = observed_mask.sum()
    n_holdout = int(n_observed * 0.2)  # Hold out 20% for evaluation
    
    # Get indices of all observed points
    observed_indices = np.where(observed_mask)
    holdout_indices = np.random.RandomState(42).choice(
        len(observed_indices[0]), size=n_holdout, replace=False
    )
    
    # Create holdout mask (points we will evaluate on)
    holdout_mask = np.zeros_like(observed_mask, dtype=bool)
    holdout_mask[observed_indices[0][holdout_indices], 
                 observed_indices[1][holdout_indices], 
                 observed_indices[2][holdout_indices]] = True
    
    # Create input mask (remove holdout points from input)
    input_mask = observed_mask.copy()
    input_mask[holdout_mask] = False
    
    # Re-predict with holdout points masked out
    print("   Re-predicting with holdout points masked...")
    all_preds_mean_holdout = []
    all_preds_std_holdout = []
    
    with torch.no_grad():
        for t in range(len(test_dataset)):
            img, mask, target = test_dataset[t]
            # img shape: (1, H, W) from dataset
            img = img.to(device)
            
            # Create modified input: mask out holdout points
            # input_mask shape is (T, H, W), we need (1, H, W) for this time step
            modified_mask_2d = input_mask[t, :, :].copy()  # (H, W)
            modified_mask_tensor = torch.from_numpy(modified_mask_2d).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # Update image: fill holdout points with normalized mean (0.0 after normalization)
            modified_img = img.clone()  # (1, H, W)
            holdout_mask_2d = holdout_mask[t, :, :]  # (H, W)
            modified_img[0, holdout_mask_2d] = 0.0  # Set holdout points to normalized mean
            
            # Add batch dimension: (1, H, W) -> (1, 1, H, W)
            modified_img_batch = modified_img.unsqueeze(0)  # (1, 1, H, W)
            x = torch.cat([modified_img_batch, modified_mask_tensor], dim=1)  # (1, 2, H, W)
            
            mean, log_var = model(x)
            std = torch.exp(0.5 * log_var)
            
            all_preds_mean_holdout.append(mean.cpu().numpy())
            all_preds_std_holdout.append(std.cpu().numpy())
    
    pred_mean_holdout = np.concatenate(all_preds_mean_holdout, axis=0)[:, 0, :, :]
    pred_std_holdout = np.concatenate(all_preds_std_holdout, axis=0)[:, 0, :, :]
    
    # Evaluate ONLY on holdout points (true interpolation evaluation)
    y_true_norm = targets[holdout_mask]  # Normalized true values at holdout points
    y_pred_norm = pred_mean_holdout[holdout_mask]  # Normalized predicted values
    y_std_norm = pred_std_holdout[holdout_mask]  # Normalized standard deviation
    
    # Denormalize
    y_true = y_true_norm * std_val + mean_val
    y_pred = y_pred_norm * std_val + mean_val
    y_std = y_std_norm * std_val
    
    evaluation_time = time.time() - evaluation_start_time
    print(f"Prediction Completed (Time: {evaluation_time:.2f} s)")
    print(f"  - Total Observed Points: {n_observed:,}")
    print(f"  - Holdout Points (for evaluation): {n_holdout:,} ({n_holdout/n_observed*100:.1f}%)")
    print(f"  - Input Points (visible to model): {n_observed - n_holdout:,}")
    print(f"  ✅ Proper interpolation evaluation on held-out points")
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    reg_metrics = compute_regression_metrics(y_true, y_pred)
    prob_metrics = compute_probabilistic_metrics(y_true, y_pred, y_std)
    
    all_metrics = {**reg_metrics, **prob_metrics}
    
    # Add experiment info
    all_metrics["experiment_info"] = {
        "experiment_time": experiment_time,
        "random_seed": 42,
        "device": str(device),
        "training_time_seconds": training_time,
        "evaluation_time_seconds": evaluation_time,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_loss),
        "final_train_loss": float(avg_train_loss),
        "final_val_loss": float(avg_val_loss),
        "model_config": {
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "lr": config.lr,
            "dropout": config.dropout,
            "init_log_var": config.init_log_var,
            "in_channels": config.in_channels,
            "base_channels": config.base_channels
        },
        "data_info": {
            "train_images": len(train_indices),
            "val_images": len(val_indices),
            "test_images": T,
            "image_size": f"{H}x{W}",
            "normalization": {
                "mean": float(train_mean),
                "std": float(train_std)
            },
            "evaluation_method": "holdout_interpolation",
            "total_observed_points": int(n_observed),
            "holdout_points": int(n_holdout),
            "holdout_ratio": float(n_holdout / n_observed),
            "valid_test_points": len(y_true),
            "note": "Evaluation performed on 20% randomly held-out observed points to test true interpolation capability"
        }
    }
    
    print("\n" + "=" * 80)
    print("  Evaluation Results (Holdout Interpolation Evaluation)")
    print("=" * 80)
    print("  ⚠️  IMPORTANT: This evaluation uses proper holdout methodology")
    print("      - 20% of observed points were randomly masked out")
    print("      - Model predicted these masked points using only 80% of observations")
    print("      - This tests true interpolation capability (not data leakage)")
    print("=" * 80)
    
    # Regression Metrics
    print("\n[Regression Metrics]")
    print(f"  {'Metric':<30} {'Value':<15} {'Description':<30}")
    print("-" * 75)
    print(f"  {'RMSE (Root Mean Squared Error)':<30} {reg_metrics['rmse']:<15.4f} {'Lower is better, Unit: Kelvin'}")
    print(f"  {'MAE (Mean Absolute Error)':<30} {reg_metrics['mae']:<15.4f} {'Lower is better, Unit: Kelvin'}")
    print(f"  {'R² (Coefficient of Determination)':<30} {reg_metrics['r2']:<15.4f} {'Higher is better, Range: (-inf, 1]'}")
    print(f"  {'MAPE (Mean Absolute Percentage Error)':<30} {reg_metrics['mape']:<15.4f} {'Lower is better, Unit: %'}")
    
    # Probabilistic Metrics
    print("\n[Probabilistic Metrics]")
    print(f"  {'Metric':<30} {'Value':<15} {'Description':<30}")
    print("-" * 75)
    print(f"  {'CRPS (Continuous Ranked Probability Score)':<30} {prob_metrics['crps']:<15.4f} {'Lower is better, Unit: Kelvin'}")
    print(f"  {'Coverage (90% Prediction Interval)':<30} {prob_metrics['coverage_90']:<15.4f} {'Target: 0.90'}")
    print(f"  {'Interval Width (90%)':<30} {prob_metrics['interval_width_90']:<15.4f} {'Lower is better, Unit: Kelvin'}")
    print(f"  {'Calibration Error':<30} {prob_metrics['calibration_error']:<15.4f} {'Lower is better, measures calibration'}")
    
    # Prediction Statistics
    print("\n[Prediction Statistics]")
    print(f"  Predicted Mean:")
    print(f"    - Range: [{y_pred.min():.2f}, {y_pred.max():.2f}] K")
    print(f"    - Mean: {y_pred.mean():.2f} K")
    print(f"    - Std: {y_pred.std():.2f} K")
    
    print(f"\n  True Values:")
    print(f"    - Range: [{y_true.min():.2f}, {y_true.max():.2f}] K")
    print(f"    - Mean: {y_true.mean():.2f} K")
    print(f"    - Std: {y_true.std():.2f} K")
    
    print(f"\n  Prediction Uncertainty (Std):")
    print(f"    - Range: [{y_std.min():.2f}, {y_std.max():.2f}] K")
    print(f"    - Mean: {y_std.mean():.2f} K")
    print(f"    - Median: {np.median(y_std):.2f} K")
    
    # Error Analysis
    errors = y_true - y_pred
    print(f"\n[Error Analysis]")
    print(f"  Residuals (True - Predicted):")
    print(f"    - Mean: {errors.mean():.2f} K (Near 0 indicates unbiased)")
    print(f"    - Std: {errors.std():.2f} K")
    print(f"    - Range: [{errors.min():.2f}, {errors.max():.2f}] K")
    print(f"    - Median: {np.median(errors):.2f} K")
    
    # Coverage Analysis
    coverage = prob_metrics['coverage_90']
    target_coverage = 0.90
    coverage_error = abs(coverage - target_coverage)
    print(f"\n[Uncertainty Calibration]")
    print(f"  90% Prediction Interval Coverage: {coverage:.4f} (Target: {target_coverage})")
    if coverage_error < 0.05:
        print(f"  ✅ Well Calibrated (Error < 5%)")
    elif coverage_error < 0.10:
        print(f"  ⚠️  Acceptable Calibration (Error < 10%)")
    else:
        print(f"  ❌ Poor Calibration (Error >= 10%)")
    
    # Save Results
    print_section_header("Save Results")
    results_path = OUTPUT_DIR / "results" / "unet_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Evaluation results saved: {results_path}")
    
    # Save Model
    model_path = OUTPUT_DIR / "models" / "unet_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'experiment_info': all_metrics["experiment_info"]
    }, model_path)
    print(f"✅ Model saved: {model_path}")
    print(f"  - Model Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save Training Loss Curve
    loss_curve_path = OUTPUT_DIR / "results" / "unet_training_losses.json"
    with open(loss_curve_path, "w") as f:
        json.dump({
            "epochs": list(range(1, len(train_losses) + 1)),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_loss)
        }, f, indent=2)
    print(f"✅ Training loss curve saved: {loss_curve_path}")
    
    # Denormalize data for visualization (restore to original scale)
    pred_mean_denorm = pred_mean * std_val + mean_val
    pred_std_denorm = pred_std * std_val
    targets_denorm = targets * std_val + mean_val
    
    # Visualization (Day 15)
    print("\nGenerating Visualizations...")
    day_idx = 14
    mean_path = OUTPUT_DIR / "figures" / "unet_mean_day15.png"
    std_path = OUTPUT_DIR / "figures" / "unet_std_day15.png"
    error_path = OUTPUT_DIR / "figures" / "unet_error_day15.png"
    
    plot_mean_map(
        pred_mean_denorm, day_idx=day_idx,
        title="U-Net Mean Prediction - Day 15",
        save_path=str(mean_path)
    )
    print(f"✅ Predicted Mean Map saved: {mean_path}")
    
    plot_std_map(
        pred_std_denorm, day_idx=day_idx,
        title="U-Net Prediction Uncertainty - Day 15",
        save_path=str(std_path)
    )
    print(f"✅ Prediction Uncertainty Map saved: {std_path}")
    
    plot_error_map(
        targets_denorm, pred_mean_denorm, day_idx=day_idx,
        title="U-Net Prediction Error - Day 15",
        save_path=str(error_path)
    )
    print(f"✅ Prediction Error Map saved: {error_path}")
    
    # Summary
    total_time = time.time() - start_time
    print_section_header("Experiment Completed")
    print(f"Total Time: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"  - Data Loading & Preprocessing: {training_start_time - start_time:.2f} s")
    print(f"  - Model Training: {training_time:.2f} s")
    print(f"  - Model Evaluation: {evaluation_time:.2f} s")
    
    print(f"\nMain Metrics Summary (Holdout Interpolation Evaluation):")
    print(f"  - R²: {reg_metrics['r2']:.4f}")
    print(f"  - RMSE: {reg_metrics['rmse']:.4f} K")
    print(f"  - CRPS: {prob_metrics['crps']:.4f} K")
    print(f"  - Coverage (90%): {prob_metrics['coverage_90']:.4f}")
    print(f"\n  📊 Evaluation Method: Holdout Interpolation (20% of observed points)")
    print(f"  📊 This RMSE reflects true interpolation capability")
    print(f"  📊 Compare with GP (4.91 K) and Tree (3.89 K) models")
    
    print(f"\nAll Result Files:")
    print(f"  📄 {results_path}")
    print(f"  📄 {loss_curve_path}")
    print(f"  💾 {model_path}")
    print(f"  📊 {mean_path}")
    print(f"  📊 {std_path}")
    print(f"  📊 {error_path}")


if __name__ == "__main__":
    main()
