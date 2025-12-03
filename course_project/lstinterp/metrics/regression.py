"""Regression evaluation metrics"""
import numpy as np
from typing import Union


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error"""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate all regression metrics"""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred)
    }
