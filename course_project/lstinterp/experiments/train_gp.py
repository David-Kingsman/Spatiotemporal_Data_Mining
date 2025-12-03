"""
Train Spatio-temporal Gaussian Process Model

This script implements a Sparse Gaussian Process model based on separable spatio-temporal kernels,
used for interpolation and prediction of MODIS Land Surface Temperature (LST) data.

Key Features:
1. Separable Spatio-Temporal Kernel: k(x, x') = k_space(lat, lon) * k_time(t)
   - Spatial Kernel: Matern 3/2 (captures spatial correlation)
   - Temporal Kernel: Matern 3/2 (captures temporal correlation)
2. Sparse GP: Uses inducing points to improve scalability
3. Variational Inference: Uses Variational ELBO for efficient training
4. Probabilistic Prediction: Provides predictive mean and uncertainty estimates

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
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import GPSTModel, GPSTConfig
from lstinterp.metrics import compute_regression_metrics, compute_probabilistic_metrics
from lstinterp.viz import plot_prediction_scatter, plot_residuals
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


def print_data_statistics(tensor, name, mode="point"):
    """Print detailed data statistics"""
    print_section_header(f"{name} Data Statistics")
    
    H, W, T = tensor.shape
    print(f"Data Dimensions: {H} x {W} x {T}")
    print(f"  - Latitude (H): {H} grid points, range: 35°N - 40°N")
    print(f"  - Longitude (W): {W} grid points, range: -115°W - -105°W")
    print(f"  - Time (T): {T} days (August 2020)")
    
    # Missing value statistics
    mask = (tensor != 0.0)
    total_points = H * W * T
    observed_points = mask.sum()
    missing_points = total_points - observed_points
    missing_ratio = missing_points / total_points * 100
    
    print(f"\nMissing Value Statistics:")
    print(f"  - Total grid points: {total_points:,}")
    print(f"  - Observed points: {observed_points:,} ({observed_points/total_points*100:.2f}%)")
    print(f"  - Missing points: {missing_points:,} ({missing_ratio:.2f}%)")
    
    # Temperature statistics
    observed_values = tensor[mask]
    print(f"\nTemperature Statistics (Kelvin):")
    print(f"  - Mean: {observed_values.mean():.2f} K")
    print(f"  - Std: {observed_values.std():.2f} K")
    print(f"  - Min: {observed_values.min():.2f} K")
    print(f"  - Max: {observed_values.max():.2f} K")
    print(f"  - Median: {np.median(observed_values):.2f} K")
    
    # Missing values per day
    missing_per_day = []
    for t in range(T):
        day_mask = (tensor[:, :, t] != 0.0)
        missing_per_day.append((H * W - day_mask.sum()) / (H * W) * 100)
    
    print(f"\nDaily Missing Value Ratios:")
    print(f"  - Average missing rate: {np.mean(missing_per_day):.2f}%")
    print(f"  - Min missing rate: {np.min(missing_per_day):.2f}% (Day {np.argmin(missing_per_day)+1})")
    print(f"  - Max missing rate: {np.max(missing_per_day):.2f}% (Day {np.argmax(missing_per_day)+1})")


def main():
    """Main function: Train and evaluate GP model"""
    start_time = time.time()
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_section_header("Experiment Configuration", width=80)
    print(f"Experiment Time: {experiment_time}")
    print(f"Random Seed: 42")
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing Device: {device}")
    if device.type == "cuda":
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Check dependencies
    print("\nDependency Check:")
    try:
        import gpytorch
        print(f"  ✅ GPyTorch: {gpytorch.__version__}")
    except ImportError:
        print("  ❌ Error: gpytorch is required")
        print("  Please run: pip install gpytorch")
        return
    
    try:
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
    except ImportError:
        print("  ❌ NumPy not installed")
        return
    
    # Load data
    print_section_header("Data Loading")
    data_path = "modis_aug_data/MODIS_Aug.mat"
    print(f"Data Path: {data_path}")
    
    print("\nLoading training data...")
    train_tensor = load_modis_tensor(data_path, "training_tensor")
    print_data_statistics(train_tensor, "Training Set")
    
    print("\nLoading test data...")
    test_tensor = load_modis_tensor(data_path, "test_tensor")
    print_data_statistics(test_tensor, "Test Set")
    
    # Create dataset (point mode)
    print_section_header("Data Preprocessing")
    print("Converting to point data format (lat, lon, time) -> temperature")
    
    print("\nCreating training dataset...")
    train_dataset = MODISDataset(train_tensor, mode="point")
    print(f"  - Training observed points: {len(train_dataset):,}")
    
    print("\nCreating test dataset...")
    test_dataset = MODISDataset(test_tensor, mode="point")
    print(f"  - Test observed points: {len(test_dataset):,}")
    
    # Prepare training data
    print("\nExtracting training data...")
    X_train = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
    y_train = np.array([train_dataset[i][1].numpy() for i in range(len(train_dataset))])
    
    print(f"  - Input feature dimensions: {X_train.shape}")
    print(f"    * Feature 1 (Lat): Range [{X_train[:, 0].min():.2f}, {X_train[:, 0].max():.2f}]")
    print(f"    * Feature 2 (Lon): Range [{X_train[:, 1].min():.2f}, {X_train[:, 1].max():.2f}]")
    print(f"    * Feature 3 (Time): Range [{X_train[:, 2].min():.0f}, {X_train[:, 2].max():.0f}] days")
    print(f"  - Target variable dimensions: {y_train.shape}")
    print(f"    * Temperature range: [{y_train.min():.2f}, {y_train.max():.2f}] K")
    print(f"    * Mean temperature: {y_train.mean():.2f} K")
    print(f"    * Temperature Std: {y_train.std():.2f} K")
    
    # Prepare test data
    print("\nExtracting test data...")
    X_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1].numpy() for i in range(len(test_dataset))])
    
    print(f"  - Input feature dimensions: {X_test.shape}")
    print(f"    * Feature 1 (Lat): Range [{X_test[:, 0].min():.2f}, {X_test[:, 0].max():.2f}]")
    print(f"    * Feature 2 (Lon): Range [{X_test[:, 1].min():.2f}, {X_test[:, 1].max():.2f}]")
    print(f"    * Feature 3 (Time): Range [{X_test[:, 2].min():.0f}, {X_test[:, 2].max():.0f}] days")
    print(f"  - Target variable dimensions: {y_test.shape}")
    print(f"    * Temperature range: [{y_test.min():.2f}, {y_test.max():.2f}] K")
    print(f"    * Mean temperature: {y_test.mean():.2f} K")
    print(f"    * Temperature Std: {y_test.std():.2f} K")
    
    # Convert to tensor
    print("\nConverting to PyTorch tensors...")
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test_np = y_test.copy()
    print(f"  - Data Type: {X_train.dtype}")
    print(f"  - Device: {device}")
    
    # Configure model
    print_section_header("Model Configuration")
    config = GPSTConfig(
        kernel_space="matern32",  # Spatial kernel: Matern 3/2
        kernel_time="matern32",   # Temporal kernel: Matern 3/2
        num_inducing=500,         # Number of inducing points (controls model complexity)
        lr=0.01,                  # Learning rate
        num_epochs=50,            # Number of epochs
        batch_size=1000           # Batch size
    )
    
    print("Model Hyperparameters:")
    print(f"  - Spatial Kernel: {config.kernel_space} (Matern 3/2)")
    print(f"  - Temporal Kernel: {config.kernel_time} (Matern 3/2)")
    print(f"  - Number of Inducing Points: {config.num_inducing}")
    print(f"  - Learning Rate: {config.lr}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch Size: {config.batch_size}")
    
    print("\nCreating inducing points...")
    # Create inducing points (using a subset of training data)
    from lstinterp.models.gp_st import create_inducing_points
    n_space = 15  # 15x15 = 225 spatial points
    n_time = 10   # 10 time points
    print(f"  - Spatial Grid: {n_space}x{n_space} = {n_space**2} points")
    print(f"  - Time Points: {n_time} points")
    print(f"  - Theoretical Total Inducing Points: {n_space**2 * n_time:,} points")
    
    inducing_points = create_inducing_points(
        n_space=n_space,
        n_time=n_time,
        normalize=True
    ).float().to(device)  # Convert to float32 to match training data
    
    print(f"  - Actual Inducing Points: {len(inducing_points):,}")
    
    # If inducing points exceed config, randomly sample
    if len(inducing_points) > config.num_inducing:
        print(f"  - Too many inducing points, random sampling to {config.num_inducing}")
        indices = torch.randperm(len(inducing_points))[:config.num_inducing]
        inducing_points = inducing_points[indices]
        print(f"  - Final Inducing Points: {len(inducing_points)}")
    else:
        print(f"  - Using all inducing points: {len(inducing_points)}")
    
    print("\nCreating model...")
    model = GPSTModel(inducing_points, config).to(device)
    model = model.float()  # Ensure model is also float32
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    
    # Model structure description
    print("\nModel Structure:")
    print("  - GP Type: Sparse Variational GP (SVGP)")
    print("  - Kernel: Separable Spatio-Temporal Kernel k(x, x') = k_space(lat, lon) * k_time(t)")
    print("  - Variational Distribution: CholeskyVariationalDistribution")
    print("  - Variational Strategy: VariationalStrategy (learn_inducing_locations=True)")
    print("  - Mean Function: ConstantMean")
    print("  - Likelihood: GaussianLikelihood")
    
    # Training
    print_section_header("Model Training")
    model.train()
    model.likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print(f"Optimizer: Adam")
    print(f"  - Learning Rate: {config.lr}")
    
    # Use marginal log likelihood as loss
    # VariationalELBO requires the GP object (model.gp), not the wrapper
    mll = gpytorch.mlls.VariationalELBO(
        model.likelihood, 
        model.gp,  # Use GP object instead of wrapper
        num_data=len(X_train)
    )
    print(f"Loss Function: Variational ELBO")
    print(f"  - Data Points: {len(X_train):,}")
    
    best_loss = float('inf')
    best_model_state = None
    best_epoch = 1  # Initialize to first epoch
    train_losses = []
    training_start_time = time.time()
    
    print(f"\nStarting Training ({config.num_epochs} epochs)...")
    print("-" * 80)
    print(f"{'Epoch':<8} {'Loss':<15} {'Best Loss':<15} {'Time':<10}")
    print("-" * 80)
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        model.train()
        model.likelihood.train()
        
        # Batch training (if data is large)
        epoch_loss = 0
        n_batches = 0
        
        if len(X_train) > config.batch_size:
            # Shuffle
            indices = torch.randperm(len(X_train))
            n_batches_total = (len(X_train) + config.batch_size - 1) // config.batch_size
            
            for i in range(0, len(X_train), config.batch_size):
                batch_indices = indices[i:i+config.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                optimizer.zero_grad()
                output = model.gp(X_batch)  # Use GP object directly
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
        else:
            optimizer.zero_grad()
            output = model.gp(X_train)  # Use GP object directly
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            
            epoch_loss = loss.item()
            n_batches = 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start_time
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        
        # Print every 10 epochs or last epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.num_epochs:
            status = "⭐" if avg_loss == best_loss else " "
            print(f"{epoch+1:<8} {avg_loss:<15.4f} {best_loss:<15.4f} {epoch_time:<10.2f}s {status}")
    
    training_time = time.time() - training_start_time
    print("-" * 80)
    print(f"Training Completed!")
    print(f"  - Total Training Time: {training_time:.2f} s ({training_time/60:.2f} min)")
    print(f"  - Best Loss: {best_loss:.4f} (Epoch {best_epoch})")
    print(f"  - Final Loss: {avg_loss:.4f}")
    print(f"  - Avg Time per Epoch: {training_time/config.num_epochs:.2f} s")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded Best Model (Epoch {best_epoch}, Loss={best_loss:.4f})")
    
    # Evaluation
    print_section_header("Model Evaluation")
    evaluation_start_time = time.time()
    
    model.eval()
    model.likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Batch prediction (if test data is large)
        pred_mean_list = []
        pred_std_list = []
        
        batch_size = 1000
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            output = model.gp(X_batch)  # Use GP object directly
            pred_dist = model.likelihood(output)
            
            pred_mean_list.append(pred_dist.mean.cpu().numpy())
            pred_std_list.append(pred_dist.stddev.cpu().numpy())
        
        y_pred_mean = np.concatenate(pred_mean_list)
        y_pred_std = np.concatenate(pred_std_list)
    
    evaluation_time = time.time() - evaluation_start_time
    print(f"Prediction Completed (Time: {evaluation_time:.2f} s)")
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    reg_metrics = compute_regression_metrics(y_test_np, y_pred_mean)
    prob_metrics = compute_probabilistic_metrics(y_test_np, y_pred_mean, y_pred_std)
    
    all_metrics = {**reg_metrics, **prob_metrics}
    
    # Add experiment info to results
    all_metrics["experiment_info"] = {
        "experiment_time": experiment_time,
        "random_seed": 42,
        "device": str(device),
        "training_time_seconds": training_time,
        "evaluation_time_seconds": evaluation_time,
        "best_epoch": best_epoch,
        "best_loss": float(best_loss),
        "final_loss": float(avg_loss),
        "model_config": {
            "kernel_space": config.kernel_space,
            "kernel_time": config.kernel_time,
            "num_inducing": config.num_inducing,
            "lr": config.lr,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size
        },
        "data_info": {
            "train_points": len(X_train),
            "test_points": len(X_test),
            "n_space_inducing": n_space,
            "n_time_inducing": n_time,
            "total_inducing_points": len(inducing_points)
        }
    }
    
    print("\n" + "=" * 80)
    print("  Evaluation Results")
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
    print(f"    - Range: [{y_pred_mean.min():.2f}, {y_pred_mean.max():.2f}] K")
    print(f"    - Mean: {y_pred_mean.mean():.2f} K")
    print(f"    - Std: {y_pred_mean.std():.2f} K")
    
    print(f"\n  True Values:")
    print(f"    - Range: [{y_test_np.min():.2f}, {y_test_np.max():.2f}] K")
    print(f"    - Mean: {y_test_np.mean():.2f} K")
    print(f"    - Std: {y_test_np.std():.2f} K")
    
    print(f"\n  Prediction Uncertainty (Std):")
    print(f"    - Range: [{y_pred_std.min():.2f}, {y_pred_std.max():.2f}] K")
    print(f"    - Mean: {y_pred_std.mean():.2f} K")
    print(f"    - Median: {np.median(y_pred_std):.2f} K")
    
    # Error Analysis
    errors = y_test_np - y_pred_mean
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
    results_path = OUTPUT_DIR / "results" / "gp_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Evaluation results saved: {results_path}")
    
    # Save Model
    model_path = OUTPUT_DIR / "models" / "gp_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'inducing_points': inducing_points.cpu(),
        'experiment_info': all_metrics["experiment_info"]
    }, model_path)
    print(f"✅ Model saved: {model_path}")
    print(f"  - Model Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save Training Loss Curve
    loss_curve_path = OUTPUT_DIR / "results" / "gp_training_losses.json"
    with open(loss_curve_path, "w") as f:
        json.dump({
            "epochs": list(range(1, len(train_losses) + 1)),
            "losses": train_losses,
            "best_epoch": best_epoch,
            "best_loss": float(best_loss)
        }, f, indent=2)
    print(f"✅ Training loss curve saved: {loss_curve_path}")
    
    # Visualization
    print("\nGenerating Visualizations...")
    scatter_path = OUTPUT_DIR / "figures" / "gp_scatter.png"
    residuals_path = OUTPUT_DIR / "figures" / "gp_residuals.png"
    
    plot_prediction_scatter(y_test_np, y_pred_mean, save_path=str(scatter_path))
    print(f"✅ Prediction scatter plot saved: {scatter_path}")
    
    plot_residuals(y_test_np, y_pred_mean, save_path=str(residuals_path))
    print(f"✅ Residuals plot saved: {residuals_path}")
    
    # Summary
    total_time = time.time() - start_time
    print_section_header("Experiment Completed")
    print(f"Total Time: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"  - Data Loading & Preprocessing: {training_start_time - start_time:.2f} s")
    print(f"  - Model Training: {training_time:.2f} s")
    print(f"  - Model Evaluation: {evaluation_time:.2f} s")
    print(f"  - Saving & Visualization: {total_time - evaluation_time - training_time - (training_start_time - start_time):.2f} s")
    
    print(f"\nMain Metrics Summary:")
    print(f"  - R²: {reg_metrics['r2']:.4f}")
    print(f"  - RMSE: {reg_metrics['rmse']:.4f} K")
    print(f"  - CRPS: {prob_metrics['crps']:.4f} K")
    print(f"  - Coverage (90%): {prob_metrics['coverage_90']:.4f}")
    
    print(f"\nAll Result Files:")
    print(f"  📄 {results_path}")
    print(f"  📄 {loss_curve_path}")
    print(f"  💾 {model_path}")
    print(f"  📊 {scatter_path}")
    print(f"  📊 {residuals_path}")


if __name__ == "__main__":
    main()
