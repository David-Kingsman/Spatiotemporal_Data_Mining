"""Utilities module"""
import numpy as np
import torch
import random

def set_seed(seed: int = 42):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Note: Cross-validation tool has been removed as the current project does not use CV

# Hyperparameter tuning tool (kept for potential future use)
from .hyperparameter_tuning import grid_search, random_search, HyperparameterSpace

__all__ = ["set_seed", "grid_search", "random_search", "HyperparameterSpace"]
