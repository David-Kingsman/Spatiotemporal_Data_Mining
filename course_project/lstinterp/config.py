"""Configuration Data Classes"""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class MODISConfig:
    """MODIS Dataset Configuration"""
    data_path: str = "modis_aug_data/MODIS_Aug.mat"
    split: Literal["train", "test"] = "train"
    normalize: bool = True
    lat_range: tuple = (35, 40)  # Latitude range
    lon_range: tuple = (-115, -105)  # Longitude range


@dataclass
class GPSTConfig:
    """Spatio-Temporal Gaussian Process Configuration"""
    kernel_design: Literal["separable", "additive", "non_separable"] = "separable"
    kernel_space: Literal["matern32", "matern52", "rbf"] = "matern32"
    kernel_time: Literal["exp", "matern32", "rbf"] = "matern32"
    num_inducing: int = 800
    lr: float = 0.01
    num_epochs: int = 50
    batch_size: int = 1000


@dataclass
class UNetConfig:
    """U-Net Configuration"""
    in_channels: int = 2  # temp + mask
    base_channels: int = 32
    lr: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 8
    dropout: Optional[float] = None


@dataclass
class TreeConfig:
    """Tree Model Configuration"""
    model_type: Literal["rf", "xgb", "lgbm"] = "xgb"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    quantile_regression: bool = True  # Whether to use quantile regression
    quantiles: Optional[list] = None  # List of quantiles, e.g., [0.1, 0.5, 0.9]
