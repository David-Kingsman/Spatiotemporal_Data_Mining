"""Models module"""
from .gp_st import STSeparableGP, STAdditiveGP, STNonSeparableGP
from .unet import ProbUNet, UNetConfig, gaussian_nll_loss
from .tree_baselines import TreeBaseline

# Maintain backward compatibility: STInducingGP as an alias for STSeparableGP
STInducingGP = STSeparableGP

__all__ = [
    "STSeparableGP",
    "STAdditiveGP",
    "STNonSeparableGP",
    "STInducingGP",  # Backward compatibility
    "ProbUNet",
    "UNetConfig",
    "gaussian_nll_loss",
    "TreeBaseline"
]
