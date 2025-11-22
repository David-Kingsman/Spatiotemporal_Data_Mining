"""模型模块"""
from .tree_baselines import TreeBaseline, TreeConfig
from .unet import ProbUNet, UNetConfig, gaussian_nll_loss

try:
    from .gp_st import STInducingGP, GPSTModel, GPSTConfig, create_inducing_points
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    STInducingGP = None
    GPSTModel = None
    GPSTConfig = None
    create_inducing_points = None

__all__ = [
    "TreeBaseline", "TreeConfig",
    "ProbUNet", "UNetConfig", "gaussian_nll_loss"
]

if GP_AVAILABLE:
    __all__.extend([
        "STInducingGP", "GPSTModel", "GPSTConfig", "create_inducing_points"
    ])

