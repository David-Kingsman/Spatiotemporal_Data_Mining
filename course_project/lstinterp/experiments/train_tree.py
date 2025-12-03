"""
Train Tree-based Baseline Models

This script implements tree-based baseline methods for interpolation and prediction of MODIS Land Surface Temperature (LST) data.

Supported Models:
1. XGBoost (Preferred): Gradient Boosting Tree, supports quantile regression
2. Random Forest (Fallback): Random Forest, does not support quantile regression (uses standard deviation for uncertainty estimation)

Key Features:
1. Quantile Regression (XGBoost): Provides prediction quantiles (10%, 50%, 90%) and uncertainty estimates
2. Standard Deviation Estimation (Random Forest): Uses standard deviation of individual tree predictions to estimate uncertainty
3. Fast Training and Prediction: Tree models are fast to train, suitable as baselines

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
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import TreeBaseline, TreeConfig
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


def main():
    """Main function: Train and evaluate tree models"""
    start_time = time.time()
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_section_header("Experiment Configuration", width=80)
    print(f"Experiment Time: {experiment_time}")
    print(f"Random Seed: 42")
    
    set_seed(42)
    
    # Check dependencies
    print("\nDependency Check:")
    try:
        import xgboost
        print(f"  ✅ XGBoost: {xgboost.__version__}")
        xgb_available = True
    except ImportError:
        print("  ⚠️  XGBoost not installed, will use Random Forest")
        xgb_available = False
    
    try:
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
    except ImportError:
        print("  ❌ NumPy not installed")
        return
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        import sklearn
        print(f"  ✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("  ❌ scikit-learn not installed")
        return
    
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
    
    # Train Model
    print_section_header("Model Configuration and Training")
    
    # Select model type
    if xgb_available:
        model_type = "xgb"
        print("✅ Using XGBoost model")
        print("  - Supports quantile regression")
        print("  - Provides uncertainty estimation")
    else:
        model_type = "rf"
        print("⚠️  Using Random Forest model (XGBoost not available)")
        print("  - Uses standard deviation for uncertainty estimation")
    
    config = TreeConfig(
        model_type=model_type,
        n_estimators=100,
        quantile_regression=(model_type != "rf"),  # RF does not support quantile regression
        quantiles=[0.1, 0.5, 0.9] if model_type != "rf" else None
    )
    
    print("\nModel Hyperparameters:")
    print(f"  - Model Type: {config.model_type}")
    print(f"  - Number of Trees: {config.n_estimators}")
    print(f"  - Quantile Regression: {config.quantile_regression}")
    if config.quantile_regression:
        print(f"  - Quantiles: {config.quantiles}")
    
    # Training
    print("\nStarting Training...")
    training_start_time = time.time()
    model = TreeBaseline(config)
    model.fit(X_train, y_train)
    training_time = time.time() - training_start_time
    print(f"✅ Training Completed (Time: {training_time:.2f} s)")
    
    # Prediction
    print_section_header("Model Prediction")
    prediction_start_time = time.time()
    print("Predicting...")
    y_pred_mean, y_pred_std = model.predict_with_uncertainty(X_test)
    prediction_time = time.time() - prediction_start_time
    print(f"✅ Prediction Completed (Time: {prediction_time:.2f} s)")
    print(f"  - Predicted Points: {len(y_pred_mean):,}")
    
    # Evaluation
    print_section_header("Model Evaluation")
    print("Calculating evaluation metrics...")
    reg_metrics = compute_regression_metrics(y_test, y_pred_mean)
    prob_metrics = compute_probabilistic_metrics(y_test, y_pred_mean, y_pred_std)
    
    all_metrics = {**reg_metrics, **prob_metrics}
    
    # Add experiment info
    all_metrics["experiment_info"] = {
        "experiment_time": experiment_time,
        "random_seed": 42,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "model_config": {
            "model_type": config.model_type,
            "n_estimators": config.n_estimators,
            "quantile_regression": config.quantile_regression,
            "quantiles": config.quantiles if config.quantile_regression else None
        },
        "data_info": {
            "train_points": len(X_train),
            "test_points": len(X_test)
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
    print(f"    - Range: [{y_test.min():.2f}, {y_test.max():.2f}] K")
    print(f"    - Mean: {y_test.mean():.2f} K")
    print(f"    - Std: {y_test.std():.2f} K")
    
    print(f"\n  Prediction Uncertainty (Std):")
    print(f"    - Range: [{y_pred_std.min():.2f}, {y_pred_std.max():.2f}] K")
    print(f"    - Mean: {y_pred_std.mean():.2f} K")
    print(f"    - Median: {np.median(y_pred_std):.2f} K")
    
    # Error Analysis
    errors = y_test - y_pred_mean
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
    results_path = OUTPUT_DIR / "results" / "tree_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Evaluation results saved: {results_path}")
    
    # Save Model
    try:
        import pickle
        model_path = OUTPUT_DIR / "models" / f"tree_model_{model_type}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Model saved: {model_path}")
        print(f"  - Model Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"⚠️  Model save failed: {e}")
    
    # Visualization
    print("\nGenerating Visualizations...")
    scatter_path = OUTPUT_DIR / "figures" / "tree_scatter.png"
    residuals_path = OUTPUT_DIR / "figures" / "tree_residuals.png"
    
    plot_prediction_scatter(y_test, y_pred_mean, save_path=str(scatter_path))
    print(f"✅ Prediction scatter plot saved: {scatter_path}")
    
    plot_residuals(y_test, y_pred_mean, save_path=str(residuals_path))
    print(f"✅ Residuals plot saved: {residuals_path}")
    
    # Summary
    total_time = time.time() - start_time
    print_section_header("Experiment Completed")
    print(f"Total Time: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"  - Data Loading & Preprocessing: {training_start_time - start_time:.2f} s")
    print(f"  - Model Training: {training_time:.2f} s")
    print(f"  - Model Prediction: {prediction_time:.2f} s")
    
    print(f"\nMain Metrics Summary:")
    print(f"  - R²: {reg_metrics['r2']:.4f}")
    print(f"  - RMSE: {reg_metrics['rmse']:.4f} K")
    print(f"  - CRPS: {prob_metrics['crps']:.4f} K")
    print(f"  - Coverage (90%): {prob_metrics['coverage_90']:.4f}")
    
    print(f"\nAll Result Files:")
    print(f"  📄 {results_path}")
    if 'model_path' in locals():
        print(f"  💾 {model_path}")
    print(f"  📊 {scatter_path}")
    print(f"  📊 {residuals_path}")


if __name__ == "__main__":
    main()
