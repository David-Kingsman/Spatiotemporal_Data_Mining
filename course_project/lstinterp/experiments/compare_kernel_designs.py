"""
GP Kernel Design Comparison Experiment

This script trains and compares three different Spatio-Temporal GP kernel designs:
1. Separable Kernel (Space x Time) - Recommended
2. Additive Kernel (Space + Time)
3. Non-Separable Kernel (Direct 3D Matern)

Note: To speed up comparison, this script uses a subset of the data (subsampling).
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check for GPyTorch
try:
    import gpytorch
except ImportError:
    print("❌ GPyTorch not installed. Please run: pip install gpytorch")
    sys.exit(1)

from lstinterp.data import load_modis_tensor, MODISDataset, MODISConfig
from lstinterp.models.gp_st import STSeparableGP, STAdditiveGP, STNonSeparableGP
from lstinterp.config import GPSTConfig
from lstinterp.metrics import rmse, mae, r2, crps_gaussian, coverage_probability, interval_width

# Output configuration
OUTPUT_DIR = project_root / "output"
RESULTS_DIR = OUTPUT_DIR / "results" / "kernel_comparison"
FIGURES_DIR = OUTPUT_DIR / "figures" / "kernel_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Training configuration (Fast comparison mode)
BATCH_SIZE = 2048
NUM_EPOCHS = 30  # Reduced epochs for comparison
LR = 0.02
NUM_INDUCING = 600
SUBSAMPLE_RATIO = 1  # Only use 20% of training data for speed

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_and_evaluate(kernel_design, train_loader, test_loader, test_values_np):
    """Train and evaluate a specific kernel design"""
    print(f"\n{'-'*20} Training Kernel Design: {kernel_design.upper()} {'-'*20}")
    
    # 1. Initialize Model
    config = GPSTConfig(
        kernel_design=kernel_design,
        num_inducing=NUM_INDUCING,
        lr=LR,
        num_epochs=NUM_EPOCHS
    )
    
    # Initialize inducing points (from first batch)
    # Note: In a real scenario, K-means initialization is better
    dummy_batch, _ = next(iter(train_loader))
    inducing_points = dummy_batch[:NUM_INDUCING].clone()
    
    # Instantiate model based on design
    if kernel_design == "separable":
        model = STSeparableGP(inducing_points, config).to(device)
    elif kernel_design == "additive":
        model = STAdditiveGP(inducing_points, config).to(device)
    elif kernel_design == "non_separable":
        model = STNonSeparableGP(inducing_points, config).to(device)
    else:
        raise ValueError(f"Unknown kernel design: {kernel_design}")
        
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    # 2. Train
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=config.lr)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    
    start_time = time.time()
    epoch_losses = []
    
    print(f"Starting training ({NUM_EPOCHS} epochs)...")
    
    # Normalize targets for training stability
    # Calculate mean and std from training data (approximation from batch)
    # Better to compute from dataset, but we can estimate from subsample
    # Note: MODISDataset returns raw Kelvin values.
    # GP works much better if targets are ~N(0,1)
    
    # Extract targets from train_loader to compute normalization stats
    all_y = []
    for _, y in train_loader:
        all_y.append(y)
    all_y = torch.cat(all_y)
    y_mean = all_y.mean()
    y_std = all_y.std()
    print(f"Target Normalization: Mean={y_mean.item():.2f}, Std={y_std.item():.2f}")
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Normalize batch targets
            batch_y_norm = (batch_y - y_mean.to(device)) / y_std.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = -mll(output, batch_y_norm)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")
            
    train_time = time.time() - start_time
    print(f"Training finished in {train_time:.2f}s")
    
    # 3. Evaluate
    print("Evaluating...")
    model.eval()
    likelihood.eval()
    
    preds_list = []
    std_list = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            dist = likelihood(output)
            
            # Denormalize predictions
            pred_mean_norm = dist.mean
            pred_std_norm = dist.stddev
            
            pred_mean = pred_mean_norm * y_std.to(device) + y_mean.to(device)
            pred_std = pred_std_norm * y_std.to(device)
            
            preds_list.append(pred_mean.cpu().numpy())
            std_list.append(pred_std.cpu().numpy())
            
    predictions = np.concatenate(preds_list)
    stds = np.concatenate(std_list)
    
    # 4. Calculate Metrics
    metrics = {
        "RMSE": float(rmse(test_values_np, predictions)),
        "MAE": float(mae(test_values_np, predictions)),
        "R2": float(r2(test_values_np, predictions)),
        "CRPS": float(np.mean(crps_gaussian(test_values_np, predictions, stds))),
        "Coverage": float(coverage_probability(test_values_np, predictions, stds)),
        "Interval_Width": float(interval_width(stds)),
        "Training_Time": train_time
    }
    
    print(f"Results for {kernel_design}:")
    print(json.dumps(metrics, indent=2))
    
    return metrics, epoch_losses


def main():
    """Main comparison loop"""
    print("=" * 60)
    print("  Comparing GP Kernel Designs")
    print("=" * 60)
    
    # Load Data
    data_path = project_root / "modis_aug_data" / "MODIS_Aug.mat"
    
    # Training Data
    train_tensor = load_modis_tensor(str(data_path), key="training_tensor")
    train_dataset = MODISDataset(train_tensor, mode="point")
    
    # Subsample training data for speed
    indices = np.random.choice(len(train_dataset), int(len(train_dataset) * SUBSAMPLE_RATIO), replace=False)
    
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training set size (subsampled): {len(train_subset)}")
    
    # Test Data (Full)
    test_tensor = load_modis_tensor(str(data_path), key="test_tensor")
    test_dataset = MODISDataset(test_tensor, mode="point")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    # Extract ground truth for evaluation
    test_values_np = test_dataset.values
    print(f"Test set size: {len(test_dataset)}")
    
    # Run Comparison
    designs = ["separable", "additive", "non_separable"]
    results = {}
    loss_curves = {}
    
    for design in designs:
        try:
            metrics, losses = train_and_evaluate(design, train_loader, test_loader, test_values_np)
            results[design] = metrics
            loss_curves[design] = losses
        except Exception as e:
            print(f"❌ Failed to run {design}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Results
    with open(RESULTS_DIR / "kernel_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate Comparison Table
    df = pd.DataFrame(results).T
    df.index.name = 'Kernel Design'
    csv_path = RESULTS_DIR / "kernel_comparison_table.csv"
    df.to_csv(csv_path)
    print(f"\nComparison Table saved to {csv_path}")
    print(df)
    
    # Visualize
    # 1. Metrics Comparison
    metrics_to_plot = ['RMSE', 'R2', 'CRPS', 'Coverage']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x=df.index, y=metric, data=df, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Comparison of {metric}')
        axes[i].grid(True, alpha=0.3, axis='y')
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=300)
    
    # 2. Training Loss Curves
    plt.figure(figsize=(10, 6))
    for design, losses in loss_curves.items():
        plt.plot(losses, label=f"{design} (Final: {losses[-1]:.3f})", linewidth=2)
    plt.title('Training Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (ELBO)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / "loss_convergence.png", dpi=300)
    
    print("\nComparison Completed!")


if __name__ == "__main__":
    main()
