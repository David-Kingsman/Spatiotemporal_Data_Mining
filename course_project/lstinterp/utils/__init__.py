"""工具模块"""
import random
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# 注意：CV相关工具函数已保留在 cross_validation.py 中，但当前项目不使用CV
# 如需使用，可以取消下面的注释导入

# 交叉验证工具（当前未使用）
# from .cross_validation import (
#     k_fold_cv_split,
#     time_block_cv_split,
#     space_block_cv_split,
#     time_space_block_cv_split,
#     extract_data_from_indices,
#     cv_to_point_data,
#     cv_to_image_data,
#     CVSplit
# )

# 超参数调优工具（保留，可能未来有用）
from .hyperparameter_tuning import (
    HyperparameterSpace,
    grid_search,
    random_search,
    save_search_results
)

__all__ = [
    "set_seed",
    # CV相关函数已注释，当前项目不使用CV
    # "k_fold_cv_split",
    # "time_block_cv_split",
    # "space_block_cv_split",
    # "time_space_block_cv_split",
    # "extract_data_from_indices",
    # "cv_to_point_data",
    # "cv_to_image_data",
    # "CVSplit",
    "HyperparameterSpace",
    "grid_search",
    "random_search",
    "save_search_results"
]
