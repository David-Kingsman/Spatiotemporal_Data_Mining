"""概率评估指标：CRPS、覆盖率等"""
import numpy as np
from typing import Union, Tuple
from scipy.stats import norm


def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """
    计算连续概率排序分数（CRPS）对于高斯分布
    
    参数:
    y_true: 真实值
    mean: 预测均值
    std: 预测标准差
    
    返回:
    CRPS值
    """
    std = np.maximum(std, 1e-10)  # 避免除零
    z = (y_true - mean) / std
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    """
    从样本计算CRPS（样本版本）
    
    参数:
    y_true: 真实值 (N,)
    samples: 预测样本 (N, M) 或 (M, N)
    
    返回:
    CRPS值
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
    计算预测区间覆盖率
    
    参数:
    y_true: 真实值
    lower: 区间下界
    upper: 区间上界
    
    返回:
    (覆盖率, 平均区间宽度)
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
    从高斯分布计算预测区间
    
    参数:
    mean: 预测均值
    std: 预测标准差
    alpha: 显著性水平（默认0.1表示90%区间）
    
    返回:
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
    计算校准误差（Calibration Error）
    
    参数:
    y_true: 真实值
    mean: 预测均值
    std: 预测标准差
    n_bins: 分箱数量
    
    返回:
    校准误差
    """
    std = np.maximum(std, 1e-10)
    z_scores = (y_true - mean) / std
    
    # 计算每个分箱的覆盖率
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
    """计算所有概率指标"""
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

