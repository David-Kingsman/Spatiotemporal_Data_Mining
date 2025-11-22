"""回归评估指标"""
import numpy as np
from typing import Union


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差"""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """平均绝对百分比误差"""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算所有回归指标"""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred)
    }


