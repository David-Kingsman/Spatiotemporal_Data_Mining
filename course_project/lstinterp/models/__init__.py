"""模型模块"""
from .tree_baselines import TreeBaseline, TreeConfig
from .unet import ProbUNet, UNetConfig, gaussian_nll_loss

try:
    from .gp_st import (
        STSeparableGP, STAdditiveGP, STNonSeparableGP,
        GPSTModel, GPSTConfig, create_inducing_points
    )
    # 保持向后兼容性：STInducingGP 作为 STSeparableGP 的别名
    STInducingGP = STSeparableGP
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    STInducingGP = None
    STSeparableGP = None
    STAdditiveGP = None
    STNonSeparableGP = None
    GPSTModel = None
    GPSTConfig = None
    create_inducing_points = None

__all__ = [
    "TreeBaseline", "TreeConfig",
    "ProbUNet", "UNetConfig", "gaussian_nll_loss"
]

if GP_AVAILABLE:
    __all__.extend([
        "STInducingGP",  # 向后兼容
        "STSeparableGP", "STAdditiveGP", "STNonSeparableGP",
        "GPSTModel", "GPSTConfig", "create_inducing_points"
    ])


