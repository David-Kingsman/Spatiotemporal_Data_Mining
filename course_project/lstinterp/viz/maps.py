"""Visualization module: maps, error plots, etc."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import warnings

# Set matplotlib to handle fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_mean_map(
    mean: np.ndarray,
    lat_size: int = 100,
    lon_size: int = 200,
    day_idx: Optional[int] = None,
    title: str = "Predicted Mean",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet_r",
    save_path: Optional[str] = None
):
    """
    Plot predicted mean map
    
    Args:
    mean: Predicted mean, can be (lat, lon) or (lat, lon, time)
    day_idx: If mean is 3D, specify which day to plot
    """
    if mean.ndim == 3:
        if day_idx is None:
            day_idx = mean.shape[2] // 2
        mean_2d = mean[:, :, day_idx]
    else:
        mean_2d = mean
    
    plt.figure(figsize=(10, 5))
    im = plt.imshow(mean_2d, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Temperature (K)')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_std_map(
    std: np.ndarray,
    lat_size: int = 100,
    lon_size: int = 200,
    day_idx: Optional[int] = None,
    title: str = "Predictive Uncertainty",
    cmap: str = "viridis",
    save_path: Optional[str] = None
):
    """Plot predicted standard deviation map"""
    if std.ndim == 3:
        if day_idx is None:
            day_idx = std.shape[2] // 2
        std_2d = std[:, :, day_idx]
    else:
        std_2d = std
    
    plt.figure(figsize=(10, 5))
    im = plt.imshow(std_2d, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(im, label='Std Dev (K)')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lat_size: int = 100,
    lon_size: int = 200,
    day_idx: Optional[int] = None,
    title: str = "Prediction Error",
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None
):
    """Plot prediction error map"""
    if y_true.ndim == 3:
        if day_idx is None:
            day_idx = y_true.shape[2] // 2
        error_2d = y_true[:, :, day_idx] - y_pred[:, :, day_idx]
    else:
        error_2d = y_true - y_pred
    
    plt.figure(figsize=(10, 5))
    vmax = np.max(np.abs(error_2d))
    im = plt.imshow(error_2d, aspect='auto', origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label='Error (K)')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs True",
    save_path: Optional[str] = None
):
    """Plot predicted vs true scatter plot"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    plt.xlabel('True Temperature (K)')
    plt.ylabel('Predicted Temperature (K)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    save_path: Optional[str] = None
):
    """Plot residual plot"""
    residuals = y_pred - y_true
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Temperature (K)')
    plt.ylabel('Residuals (K)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    std: np.ndarray,
    title: str = "Uncertainty vs Error",
    save_path: Optional[str] = None
):
    """Plot uncertainty vs error scatter plot"""
    error = np.abs(y_pred - y_true)
    plt.figure(figsize=(8, 6))
    plt.scatter(std, error, alpha=0.5, s=10)
    plt.xlabel('Predicted Std Dev (K)')
    plt.ylabel('Absolute Error (K)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
