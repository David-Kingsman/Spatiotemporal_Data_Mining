"""Probabilistic evaluation metrics: CRPS, coverage, etc."""
import numpy as np
from typing import Union, Tuple
from scipy.stats import norm


def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS) for Gaussian distribution
    
    Args:
    y_true: True values
    mean: Predicted mean
    std: Predicted standard deviation
    
    Returns:
    CRPS value
    """
    std = np.maximum(std, 1e-10)  # Avoid division by zero
    z = (y_true - mean) / std
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    """
    Calculate CRPS from samples (sample version)
    
    Args:
    y_true: True values (N,)
    samples: Predicted samples (N, M) or (M, N)
    
    Returns:
    CRPS value
    """
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.shape[0] != len(y_true):
        samples = samples.T
    
    N, M = samples.shape
    crps = np.zeros(N)
    
    for i in range(N):
        s = np.sort(samples[i, :])
        crps[i] = np.mean(np.abs(s - y_true[i])) - np.mean(np.abs(np.diff(s))) * (M - 1) / (2 * M)
    
    return float(np.mean(crps))


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate prediction interval coverage
    
    Args:
    y_true: True values
    lower: Interval lower bound
    upper: Interval upper bound
    
    Returns:
    (coverage, average interval width)
    """
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    width = np.mean(upper - lower)
    return float(coverage), float(width)


def prediction_interval_from_gaussian(
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction interval from Gaussian distribution
    
    Args:
    mean: Predicted mean
    std: Predicted standard deviation
    alpha: Significance level (default 0.1 for 90% interval)
    
    Returns:
    (lower, upper)
    """
    z = norm.ppf(1 - alpha / 2)
    lower = mean - z * std
    upper = mean + z * std
    return lower, upper


def calibration_error(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Calibration Error
    
    Args:
    y_true: True values
    mean: Predicted mean
    std: Predicted standard deviation
    n_bins: Number of bins
    
    Returns:
    Calibration error
    """
    std = np.maximum(std, 1e-10)
    z_scores = (y_true - mean) / std
    
    # Calculate coverage for each bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    empirical_coverage = []
    expected_coverage = []
    
    for i in range(n_bins):
        lower_z = norm.ppf(bin_centers[i])
        upper_z = norm.ppf(bin_centers[i] + 1 / n_bins)
        
        mask = (z_scores >= lower_z) & (z_scores < upper_z)
        empirical_coverage.append(np.mean(mask))
        expected_coverage.append(1 / n_bins)
    
    return float(np.mean(np.abs(np.array(empirical_coverage) - np.array(expected_coverage))))


def compute_probabilistic_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float = 0.1
) -> dict:
    """Calculate all probabilistic metrics"""
    crps = crps_gaussian(y_true, mean, std)
    lower, upper = prediction_interval_from_gaussian(mean, std, alpha)
    coverage, width = prediction_interval_coverage(y_true, lower, upper)
    cal_error = calibration_error(y_true, mean, std)
    
    return {
        "crps": crps,
        f"coverage_{int((1-alpha)*100)}": coverage,
        f"interval_width_{int((1-alpha)*100)}": width,
        "calibration_error": cal_error
    }


# Add aliases for cleaner imports
coverage_probability = lambda y, mean, std, alpha=0.1: compute_probabilistic_metrics(y, mean, std, alpha)[f"coverage_{int((1-alpha)*100)}"]
interval_width = lambda std, alpha=0.1: compute_probabilistic_metrics(np.zeros_like(std), np.zeros_like(std), std, alpha)[f"interval_width_{int((1-alpha)*100)}"]
