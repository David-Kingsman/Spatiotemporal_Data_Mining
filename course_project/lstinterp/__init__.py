"""
LSTInterp: 时空数据插值与不确定性评估库

一个用于时空数据（如MODIS地表温度）插补和预测的Python库，
提供多种概率模型（高斯过程、U-Net、树模型）和完整的评估指标。
"""

__version__ = "0.1.0"
__author__ = "Course Project"

from . import data
from . import models
from . import metrics
from . import viz

__all__ = ["data", "models", "metrics", "viz"]

