"""
Generate Visualizations for All Days Predictions

This script generates the following for U-Net, GP, and Tree models for all 31 days:
- Predicted Mean Maps
- Prediction Uncertainty Maps
- Prediction Error Maps

Outputs are saved to the output/figures/all_days/ directory.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import ProbUNet, GPSTModel, TreeBaseline
from lstinterp.config import UNetConfig, GPSTConfig, TreeConfig
from lstinterp.viz.maps import plot_mean_map, plot_std_map, plot_error_map
from lstinterp.utils import set_seed

# Output directory
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "all_days"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed
set_seed(42)

# Data path
DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_test_data():
    """Load test data"""
    print("=" * 80)
    print("  Loading Test Data")
    print("=" * 80)
    
    # Load data
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    print(f"Test data shape: {test_tensor.shape} (lat, lon, time)")
    
    H, W, T = test_tensor.shape
    print(f"  - Spatial dimensions: {H} x {W}")
    print(f"  - Time dimensions: {T} days")
    
    return test_tensor, H, W, T


def generate_unet_predictions(H, W, T):
    """Generate predictions for U-Net model for all days"""
    print("\n" + "=" * 80)
    print("  U-Net Model Prediction")
    print("=" * 80)
    
    # Check model file
    model_path = OUTPUT_DIR / "models" / "unet_model.pth"
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Please run train_unet.py to train the model first")
        return None
    
    # Load model
    print("Loading U-Net model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = ProbUNet(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    
    # Calculate normalization parameters (from training data)
    train_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    train_mask = (train_tensor != 0)
    train_mean = train_tensor[train_mask].mean()
    train_std = train_tensor[train_mask].std()
    
    print(f"Normalization parameters: mean={train_mean:.2f}, std={train_std:.2f}")
    
    # Create test dataset
    test_dataset = MODISDataset(
        test_tensor,
        mode="image",
        norm_mean=train_mean,
        norm_std=train_std
    )
    
    # Store all predictions
    all_pred_mean = np.zeros((H, W, T))
    all_pred_std = np.zeros((H, W, T))
    all_true = np.zeros((H, W, T))
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for t in range(T):
            if (t + 1) % 5 == 0:
                print(f"  Processing Day {t+1}/{T}...")
            
            # Get data
            image, mask, target = test_dataset[t]
            # Concatenate image and mask as 2-channel input (consistent with training)
            image = torch.cat([image, mask], dim=0)  # (2, H, W)
            image = image.unsqueeze(0).to(device)  # (1, 2, H, W)
            
            # Predict
            pred_mean, pred_log_var = model(image)
            pred_mean = pred_mean.squeeze().cpu().numpy()
            pred_std = np.sqrt(np.exp(np.clip(pred_log_var.squeeze().cpu().numpy(), -10, 10)))
            
            # Denormalize
            all_pred_mean[:, :, t] = pred_mean * train_std + train_mean
            all_pred_std[:, :, t] = pred_std * train_std
            all_true[:, :, t] = target.numpy() * train_std + train_mean
            
            # Keep predictions only where observed
            mask_2d = mask.squeeze().numpy()
            all_pred_mean[:, :, t][mask_2d == 0] = np.nan
            all_pred_std[:, :, t][mask_2d == 0] = np.nan
            all_true[:, :, t][mask_2d == 0] = np.nan
    
    print("✅ U-Net Prediction Completed")
    return all_pred_mean, all_pred_std, all_true


def generate_gp_predictions(H, W, T):
    """Generate predictions for GP model for all days"""
    print("\n" + "=" * 80)
    print("  GP Model Prediction")
    print("=" * 80)
    
    # Check model file
    model_path = OUTPUT_DIR / "models" / "gp_model.pth"
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Please run train_gp.py to train the model first")
        return None, None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print("Loading GP model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        # Compatible with old checkpoint: if kernel_design attribute is missing, set default value "separable"
        if not hasattr(config, 'kernel_design'):
            # Set attribute directly (if config is a dataclass instance, this should be fine)
            config.kernel_design = "separable"
    else:
        # Use default configuration
        config = GPSTConfig(
            kernel_space="matern32",
            kernel_time="matern32",
            num_inducing=500,
            lr=0.01,
            num_epochs=50,
            kernel_design="separable"  # Default to separable design
        )
    
    # Create inducing points
    if 'inducing_points' in checkpoint:
        inducing_points = checkpoint['inducing_points'].to(device)
    else:
        # Create default inducing points
        from lstinterp.models.gp_st import create_inducing_points
        inducing_points = create_inducing_points(
            n_space=15,
            n_time=10,
            normalize=True
        ).to(device)
        if len(inducing_points) > config.num_inducing:
            indices = torch.randperm(len(inducing_points))[:config.num_inducing]
            inducing_points = inducing_points[indices]
    
    # Initialize model (use initialization parameters from checkpoint or default values)
    # For old checkpoints, these parameters may not exist, use defaults
    # Extract scalar value from tensor (if present)
    if 'lengthscale_space' in checkpoint:
        ls_space = checkpoint['lengthscale_space']
        if isinstance(ls_space, torch.Tensor):
            if ls_space.numel() > 1:
                lengthscale_space = float(ls_space.mean().item())
            else:
                lengthscale_space = float(ls_space.item())
        else:
            lengthscale_space = float(ls_space)
    else:
        lengthscale_space = 0.5
    
    if 'lengthscale_time' in checkpoint:
        ls_time = checkpoint['lengthscale_time']
        if isinstance(ls_time, torch.Tensor):
            if ls_time.numel() > 1:
                lengthscale_time = float(ls_time.mean().item())
            else:
                lengthscale_time = float(ls_time.item())
        else:
            lengthscale_time = float(ls_time)
    else:
        lengthscale_time = 0.3
    
    if 'outputscale' in checkpoint:
        os_val = checkpoint['outputscale']
        if isinstance(os_val, torch.Tensor):
            outputscale = float(os_val.item())
        else:
            outputscale = float(os_val)
    else:
        outputscale = 1.0
    
    if 'noise' in checkpoint:
        noise_val = checkpoint['noise']
        if isinstance(noise_val, torch.Tensor):
            noise = float(noise_val.item())
        else:
            noise = float(noise_val)
    else:
        noise = 0.2
    
    # Initialize model (use scalar values instead of tensors)
    try:
        model = GPSTModel(
            inducing_points, config,
            lengthscale_space=lengthscale_space,
            lengthscale_time=lengthscale_time,
            outputscale=outputscale,
            noise=noise
        ).to(device)
    except TypeError:
        # If GPSTModel does not accept these arguments, use simplified version
        model = GPSTModel(inducing_points, config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.likelihood.eval()
    
    print("✅ GP model loaded")
    
    # Load test data (point mode)
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    test_dataset = MODISDataset(test_tensor, mode="point", normalize_coords=True)
    
    # Reorganize into spatial grid
    all_pred_mean = np.full((H, W, T), np.nan)
    all_pred_std = np.full((H, W, T), np.nan)
    all_true = np.full((H, W, T), np.nan)
    
    print("\nGenerating predictions...")
    # Get original indices
    test_mask = (test_tensor != 0)
    lat_indices, lon_indices, t_indices = np.where(test_mask)
    
    # Batch prediction (GP models require batch processing)
    batch_size = 1000
    n_points = len(test_dataset)
    
    X_test_list = []
    indices_list = []
    
    for idx in range(n_points):
        x, y = test_dataset[idx]
        X_test_list.append(x.numpy())
        indices_list.append((lat_indices[idx], lon_indices[idx], t_indices[idx]))
    
    X_test = np.array(X_test_list)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    print(f"Test points: {n_points}")
    print("Starting batch prediction (may take some time)...")
    
    all_means = []
    all_stds = []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_end = min(i + batch_size, n_points)
            X_batch = X_test_tensor[i:batch_end]
            
            # GP prediction
            mean, std = model.predict(X_batch)
            
            all_means.append(mean.cpu().numpy())
            all_stds.append(std.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed: {batch_end}/{n_points} points ({batch_end/n_points*100:.1f}%)")
    
    pred_mean = np.concatenate(all_means)
    pred_std = np.concatenate(all_stds)
    
    # Fill grid
    for idx, (lat_idx, lon_idx, t_idx) in enumerate(indices_list):
        all_pred_mean[lat_idx, lon_idx, t_idx] = pred_mean[idx]
        all_pred_std[lat_idx, lon_idx, t_idx] = pred_std[idx]
        all_true[lat_idx, lon_idx, t_idx] = test_dataset.values[idx]
    
    print("✅ GP Prediction Completed")
    return all_pred_mean, all_pred_std, all_true


def generate_tree_predictions(H, W, T):
    """Generate predictions for Tree model for all days"""
    print("\n" + "=" * 80)
    print("  Tree Model Prediction")
    print("=" * 80)
    
    # Check model file
    model_path = OUTPUT_DIR / "models" / "tree_model_xgb.pkl"
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Please run train_tree.py to train the model first")
        return None
    
    import pickle
    
    # Load model
    print("Loading Tree model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Get model object from saved data
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
    else:
        # If not wrapped, use directly
        model = model_data
    
    print(f"Model Type: {type(model).__name__}")
    
    # Load test data (point mode)
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    test_dataset = MODISDataset(test_tensor, mode="point", normalize_coords=True)
    
    # Reorganize into spatial grid
    all_pred_mean = np.full((H, W, T), np.nan)
    all_pred_std = np.full((H, W, T), np.nan)
    all_true = np.full((H, W, T), np.nan)
    
    print("\nGenerating predictions...")
    # Batch prediction
    X_test = []
    indices_list = []  # Save original indices (lat_idx, lon_idx, t_idx)
    
    # Get original indices (from where)
    test_mask = (test_tensor != 0)
    lat_indices, lon_indices, t_indices = np.where(test_mask)
    
    for idx in range(len(test_dataset)):
        x, y = test_dataset[idx]
        X_test.append(x.numpy())
        # Save original indices
        indices_list.append((lat_indices[idx], lon_indices[idx], t_indices[idx]))
    
    X_test = np.array(X_test)
    print(f"Test points: {len(X_test)}")
    
    # Prediction
    # Check if it is a TreeBaseline object
    if isinstance(model, TreeBaseline):
        # Check if quantile regression is available
        if hasattr(model, 'config') and hasattr(model.config, 'quantile_regression') and model.config.quantile_regression:
            # Quantile regression, can calculate uncertainty
            pred_mean, pred_std = model.predict_with_uncertainty(X_test)
        elif hasattr(model, 'quantile_models') and len(model.quantile_models) > 0:
            # Has quantile models, can use uncertainty
            pred_mean, pred_std = model.predict_with_uncertainty(X_test)
        else:
            # Ordinary prediction
            pred_mean = model.predict(X_test)
            # Use simple heuristic estimation
            pred_std = np.abs(pred_mean - pred_mean.mean()) * 0.1 + 1.0
    else:
        # Call predict directly
        pred_mean = model.predict(X_test)
        # Use simple heuristic estimation
        pred_std = np.abs(pred_mean - pred_mean.mean()) * 0.1 + 1.0
    
    # Fill grid (using original indices)
    for idx, (lat_idx, lon_idx, t_idx) in enumerate(indices_list):
        all_pred_mean[lat_idx, lon_idx, t_idx] = pred_mean[idx]
        if isinstance(pred_std, np.ndarray):
            if len(pred_std.shape) == 1:
                all_pred_std[lat_idx, lon_idx, t_idx] = pred_std[idx]
            else:
                all_pred_std[lat_idx, lon_idx, t_idx] = pred_std[idx, 0]
        else:
            all_pred_std[lat_idx, lon_idx, t_idx] = pred_std  # scalar
        all_true[lat_idx, lon_idx, t_idx] = test_dataset.values[idx]
    
    print("✅ Tree Prediction Completed")
    return all_pred_mean, all_pred_std, all_true


def plot_all_days_grid(
    pred_mean, pred_std, true_values,
    model_name, H, W, T,
    save_prefix,
    vmin_mean=None, vmax_mean=None,
    vmin_std=None, vmax_std=None,
    vmax_error=None
):
    """Generate grid plots for all days (8 rows x 4 columns)
    
    Args:
        vmin_mean, vmax_mean: Color range for predicted mean (if None, use current data range)
        vmin_std, vmax_std: Color range for predicted std (if None, use current data range)
        vmax_error: Maximum absolute error (if None, use current data range)
    """
    print(f"\nGenerating {model_name} visualizations for all days ({T} days, 8x4 layout)...")
    
    # Calculate days per row
    n_rows = 8
    n_cols = 4
    
    # Create grids for each type of plot (reduce whitespace, remove top title)
    fig_mean = plt.figure(figsize=(20, 40))
    fig_std = plt.figure(figsize=(20, 40))
    fig_error = plt.figure(figsize=(20, 40))
    
    # Reduce hspace and wspace to minimize whitespace
    gs_mean = GridSpec(n_rows, n_cols, figure=fig_mean, hspace=0.1, wspace=0.1, 
                       left=0.02, right=0.98, top=0.98, bottom=0.02)
    gs_std = GridSpec(n_rows, n_cols, figure=fig_std, hspace=0.1, wspace=0.1,
                      left=0.02, right=0.98, top=0.98, bottom=0.02)
    gs_error = GridSpec(n_rows, n_cols, figure=fig_error, hspace=0.1, wspace=0.1,
                        left=0.02, right=0.98, top=0.98, bottom=0.02)
    
    # If unified range not provided, use current data range
    if vmin_mean is None:
        vmin_mean = np.nanmin(pred_mean)
    if vmax_mean is None:
        vmax_mean = np.nanmax(pred_mean)
    if vmin_std is None:
        vmin_std = np.nanmin(pred_std)
    if vmax_std is None:
        vmax_std = np.nanmax(pred_std)
    
    errors = true_values - pred_mean
    if vmax_error is None:
        vmax_error = np.nanmax(np.abs(errors))
    
    # Print color ranges used (for debugging)
    print(f"  {model_name} Color Ranges:")
    print(f"    Mean: [{vmin_mean:.2f}, {vmax_mean:.2f}] K")
    print(f"    Std:  [{vmin_std:.2f}, {vmax_std:.2f}] K")
    print(f"    Error: [-{vmax_error:.2f}, {vmax_error:.2f}] K")
    
    for t in range(min(T, n_rows * n_cols)):
        row = t // n_cols
        col = t % n_cols
        
        # Predicted Mean Plot
        ax_mean = fig_mean.add_subplot(gs_mean[row, col])
        im_mean = ax_mean.imshow(
            pred_mean[:, :, t], 
            aspect='auto', origin='lower',
            cmap='jet_r', vmin=vmin_mean, vmax=vmax_mean
        )
        ax_mean.set_title(f'Day {t+1}', fontsize=8, pad=2)  # Reduce font size and padding
        ax_mean.set_xlabel('Longitude Index', fontsize=7)
        ax_mean.set_ylabel('Latitude Index', fontsize=7)
        # Explicitly set limits for imshow to ensure consistent color range
        im_mean.set_clim(vmin=vmin_mean, vmax=vmax_mean)
        fig_mean.colorbar(im_mean, ax=ax_mean, label='Temperature (K)', shrink=0.8, aspect=20)
        
        # Prediction Uncertainty Plot
        ax_std = fig_std.add_subplot(gs_std[row, col])
        im_std = ax_std.imshow(
            pred_std[:, :, t],
            aspect='auto', origin='lower',
            cmap='viridis', vmin=vmin_std, vmax=vmax_std
        )
        ax_std.set_title(f'Day {t+1}', fontsize=8, pad=2)
        ax_std.set_xlabel('Longitude Index', fontsize=7)
        ax_std.set_ylabel('Latitude Index', fontsize=7)
        im_std.set_clim(vmin=vmin_std, vmax=vmax_std)
        fig_std.colorbar(im_std, ax=ax_std, label='Std Dev (K)', shrink=0.8, aspect=20)
        
        # Prediction Error Plot
        ax_error = fig_error.add_subplot(gs_error[row, col])
        error_t = errors[:, :, t]
        im_error = ax_error.imshow(
            error_t,
            aspect='auto', origin='lower',
            cmap='RdBu_r', vmin=-vmax_error, vmax=vmax_error
        )
        ax_error.set_title(f'Day {t+1}', fontsize=8, pad=2)
        ax_error.set_xlabel('Longitude Index', fontsize=7)
        ax_error.set_ylabel('Latitude Index', fontsize=7)
        im_error.set_clim(vmin=-vmax_error, vmax=vmax_error)
        fig_error.colorbar(im_error, ax=ax_error, label='Error (K)', shrink=0.8, aspect=20)
    
    # Save
    mean_path = FIGURES_DIR / f"{save_prefix}_mean_all_days.png"
    std_path = FIGURES_DIR / f"{save_prefix}_std_all_days.png"
    error_path = FIGURES_DIR / f"{save_prefix}_error_all_days.png"
    
    # Remove top title, save directly (reduce whitespace)
    fig_mean.savefig(mean_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_mean)
    print(f"✅ Mean Prediction Grid Saved: {mean_path}")
    
    fig_std.savefig(std_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_std)
    print(f"✅ Uncertainty Prediction Grid Saved: {std_path}")
    
    fig_error.savefig(error_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_error)
    print(f"✅ Error Prediction Grid Saved: {error_path}")


def main():
    """Main function"""
    print("=" * 80)
    print("  Generate Visualizations for All Days Predictions")
    print("=" * 80)
    
    # Load test data
    test_tensor, H, W, T = load_test_data()
    
    # Collect predictions from all models
    all_results = {}
    
    # U-Net model
    unet_results = generate_unet_predictions(H, W, T)
    if unet_results[0] is not None:
        all_results['unet'] = unet_results
        np.save(FIGURES_DIR / "unet_pred_mean.npy", unet_results[0])
        np.save(FIGURES_DIR / "unet_pred_std.npy", unet_results[1])
        np.save(FIGURES_DIR / "unet_true.npy", unet_results[2])
        print("✅ U-Net predictions saved")
    
    # Tree model
    tree_results = generate_tree_predictions(H, W, T)
    if tree_results[0] is not None:
        all_results['tree'] = tree_results
        np.save(FIGURES_DIR / "tree_pred_mean.npy", tree_results[0])
        np.save(FIGURES_DIR / "tree_pred_std.npy", tree_results[1])
        np.save(FIGURES_DIR / "tree_true.npy", tree_results[2])
        print("✅ Tree predictions saved")
    
    # GP model
    gp_results = generate_gp_predictions(H, W, T)
    if gp_results[0] is not None:
        all_results['gp'] = gp_results
        np.save(FIGURES_DIR / "gp_pred_mean.npy", gp_results[0])
        np.save(FIGURES_DIR / "gp_pred_std.npy", gp_results[1])
        np.save(FIGURES_DIR / "gp_true.npy", gp_results[2])
        print("✅ GP predictions saved")
    
    # Calculate unified color range (across all models)
    print("\nCalculating unified color range (across all models)...")
    all_means = []
    all_stds = []
    all_errors = []
    
    for model_name, (pred_mean, pred_std, true_values) in all_results.items():
        all_means.append(pred_mean)
        all_stds.append(pred_std)
        errors = true_values - pred_mean
        all_errors.append(np.abs(errors))
    
    if all_means:
        # Unified color range
        vmin_mean = min(np.nanmin(m) for m in all_means)
        vmax_mean = max(np.nanmax(m) for m in all_means)
        vmin_std = max(0, min(np.nanmin(s) for s in all_stds))  # Ensure std min is at least 0
        vmax_std = max(np.nanmax(s) for s in all_stds)
        vmax_error = max(np.nanmax(e) for e in all_errors)
        
        print(f"Unified Color Ranges:")
        print(f"  Mean: [{vmin_mean:.2f}, {vmax_mean:.2f}] K")
        print(f"  Std:  [{vmin_std:.2f}, {vmax_std:.2f}] K")
        print(f"  Error: [-{vmax_error:.2f}, {vmax_error:.2f}] K")
        print(f"\n⚠️  All models will use the above unified color ranges for plotting")
        
        # Plot all models using unified ranges
        if 'unet' in all_results:
            pred_mean, pred_std, true_values = all_results['unet']
            plot_all_days_grid(
                pred_mean, pred_std, true_values,
                "U-Net", H, W, T,
                "unet",
                vmin_mean=vmin_mean, vmax_mean=vmax_mean,
                vmin_std=vmin_std, vmax_std=vmax_std,
                vmax_error=vmax_error
            )
        
        if 'tree' in all_results:
            pred_mean, pred_std, true_values = all_results['tree']
            plot_all_days_grid(
                pred_mean, pred_std, true_values,
                "Tree (XGBoost)", H, W, T,
                "tree",
                vmin_mean=vmin_mean, vmax_mean=vmax_mean,
                vmin_std=vmin_std, vmax_std=vmax_std,
                vmax_error=vmax_error
            )
        
        if 'gp' in all_results:
            pred_mean, pred_std, true_values = all_results['gp']
            plot_all_days_grid(
                pred_mean, pred_std, true_values,
                "GP (Sparse)", H, W, T,
                "gp",
                vmin_mean=vmin_mean, vmax_mean=vmax_mean,
                vmin_std=vmin_std, vmax_std=vmax_std,
                vmax_error=vmax_error
            )
    
    print("\n" + "=" * 80)
    print("  Completed")
    print("=" * 80)
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("\nGenerated Files:")
    print("  - unet_mean_all_days.png: U-Net Mean Predictions for All Days")
    print("  - unet_std_all_days.png: U-Net Uncertainty for All Days")
    print("  - unet_error_all_days.png: U-Net Error for All Days")
    print("  - tree_mean_all_days.png: Tree Mean Predictions for All Days")
    print("  - tree_std_all_days.png: Tree Uncertainty for All Days")
    print("  - tree_error_all_days.png: Tree Error for All Days")


if __name__ == "__main__":
    main()
