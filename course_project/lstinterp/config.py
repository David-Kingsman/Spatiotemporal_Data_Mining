"""配置数据类"""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class MODISConfig:
    """MODIS数据配置"""
    data_path: str = "modis_aug_data/MODIS_Aug.mat"
    split: Literal["train", "test"] = "train"
    normalize: bool = True
    lat_range: tuple = (35, 40)  # 纬度范围
    lon_range: tuple = (-115, -105)  # 经度范围


@dataclass
class GPSTConfig:
    """时空高斯过程配置"""
    kernel_design: Literal["separable", "additive", "non_separable"] = "separable"
    kernel_space: Literal["matern32", "matern52", "rbf"] = "matern32"
    kernel_time: Literal["exp", "matern32", "rbf"] = "matern32"
    num_inducing: int = 800
    lr: float = 0.01
    num_epochs: int = 50
    batch_size: int = 1000


@dataclass
class UNetConfig:
    """U-Net配置"""
    in_channels: int = 2  # temp + mask
    base_channels: int = 32
    lr: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 8


@dataclass
class TreeConfig:
    """树模型配置"""
    model_type: Literal["rf", "xgb", "lgbm"] = "xgb"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    quantile_regression: bool = True  # 是否使用分位数回归
    quantiles: Optional[list] = None  # 分位数列表，如 [0.1, 0.5, 0.9]

