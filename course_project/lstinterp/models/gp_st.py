"""时空高斯过程模型"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, Optional

try:
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
    from gpytorch.constraints import Positive, Interval
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    ApproximateGP = object
    gpytorch = None
    Positive = None
    Interval = None


@dataclass
class GPSTConfig:
    """时空高斯过程配置"""
    kernel_space: Literal["matern32", "matern52", "rbf"] = "matern32"
    kernel_time: Literal["exp", "matern32", "rbf"] = "matern32"
    num_inducing: int = 800
    lr: float = 0.01
    num_epochs: int = 50
    batch_size: int = 1000


class STInducingGP(ApproximateGP):
    """
    时空诱导点高斯过程
    
    使用可分核：k(x, x') = k_space(lat, lon) * k_time(t)
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig, 
                 lengthscale_space: float = 0.5, lengthscale_time: float = 0.3,
                 outputscale: float = 10.0):
        if not GPYTORCH_AVAILABLE:
            raise ImportError("需要安装 gpytorch")
        
        self.config = config
        
        # 变分分布
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # 变分策略（增加 jitter 以提高数值稳定性）
        # 使用更大的 jitter 值以应对数值不稳定性（提高到 1e-4）
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-4,  # 提高到 1e-4 以应对数值不稳定性
        )
        
        super().__init__(variational_strategy)
        
        # 均值模块
        self.mean_module = ConstantMean()
        
        # 空间核（作用于前2维：lat, lon）
        # 确保长度尺度不会太小（避免数值不稳定）
        lengthscale_space_val = max(lengthscale_space, 0.3)
        
        if config.kernel_space == "matern32":
            base_space = MaternKernel(nu=1.5, ard_num_dims=2)
        elif config.kernel_space == "matern52":
            base_space = MaternKernel(nu=2.5, ard_num_dims=2)
        else:  # rbf
            base_space = RBFKernel(ard_num_dims=2)
        
        # 添加长度尺度约束（确保为正且不会太小）
        base_space.lengthscale_constraint = Positive()
        # 初始化长度尺度（使用与 inducing_points 相同的 dtype）
        base_space.lengthscale = torch.tensor([[lengthscale_space_val, lengthscale_space_val]], 
                                              dtype=inducing_points.dtype, device=inducing_points.device)
        
        # 添加输出尺度约束（限制在合理范围内以避免数值不稳定）
        # 允许更大的输出尺度以适应不同的数据尺度
        outputscale_val = min(max(outputscale, 0.1), 50.0)  # 限制在 [0.1, 50.0]
        self.covar_space = ScaleKernel(base_space)
        self.covar_space.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        # 初始化输出尺度（基于数据方差的合理估计）
        self.covar_space.outputscale = torch.tensor(outputscale_val, dtype=inducing_points.dtype, device=inducing_points.device)
        
        # 时间核（作用于第3维：time）
        # 确保时间长度尺度不会太小
        lengthscale_time_val = max(lengthscale_time, 0.2)
        
        if config.kernel_time == "matern32":
            base_time = MaternKernel(nu=1.5, ard_num_dims=1)
        elif config.kernel_time == "rbf":
            base_time = RBFKernel(ard_num_dims=1)
        else:  # exp (相当于Matern 1/2)
            base_time = MaternKernel(nu=0.5, ard_num_dims=1)
        
        # 添加长度尺度约束
        base_time.lengthscale_constraint = Positive()
        # 初始化时间长度尺度
        base_time.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        
        self.covar_time = ScaleKernel(base_time)
        self.covar_time.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        # 初始化输出尺度（使用相同的约束）
        self.covar_time.outputscale = torch.tensor(outputscale_val, dtype=inducing_points.dtype, device=inducing_points.device)
    
    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        参数:
        x: (N, 3) - (lat, lon, time)
        
        返回:
        MultivariateNormal分布
        """
        # 分离空间和时间维度
        space_x = x[:, :2]  # (N, 2) - lat, lon
        time_x = x[:, 2:3]  # (N, 1) - time
        
        # 均值
        mean_x = self.mean_module(x)
        
        # 协方差：空间核 × 时间核（可分核）
        covar_x = self.covar_space(space_x) * self.covar_time(time_x)
        
        if gpytorch is None:
            raise ImportError("需要安装 gpytorch")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSTModel(nn.Module):
    """
    时空GP模型包装器
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig,
                 lengthscale_space: float = 0.5, lengthscale_time: float = 0.3,
                 outputscale: float = 10.0, noise: float = 1.0):
        super().__init__()
        if not GPYTORCH_AVAILABLE:
            raise ImportError("需要安装 gpytorch")
        self.config = config
        self.gp = STInducingGP(inducing_points, config, 
                              lengthscale_space=lengthscale_space,
                              lengthscale_time=lengthscale_time,
                              outputscale=outputscale)
        # 添加噪声约束（确保噪声为正且不会太小）
        noise_val = max(min(noise, 5.0), 0.01)  # 限制在 [0.01, 5.0]
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise_constraint = Interval(lower_bound=0.01, upper_bound=5.0)
        # 初始化噪声（基于数据方差的合理估计）
        self.likelihood.noise = torch.tensor(noise_val, dtype=inducing_points.dtype, device=inducing_points.device)
    
    def forward(self, x: torch.Tensor):
        """前向传播"""
        return self.gp(x)
    
    def predict(self, x: torch.Tensor) -> tuple:
        """
        预测
        
        返回:
        mean: 预测均值
        std: 预测标准差
        """
        self.eval()
        self.likelihood.eval()
        
        if gpytorch is None:
            raise ImportError("需要安装 gpytorch")
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.gp(x))
            mean = dist.mean
            std = dist.stddev
        
        return mean, std


def create_inducing_points(
    n_space: int = 20,
    n_time: int = 31,
    lat_range: tuple = (35, 40),
    lon_range: tuple = (-115, -105),
    normalize: bool = True
) -> torch.Tensor:
    """
    创建诱导点
    
    参数:
    n_space: 空间维度上的诱导点数量（会创建 n_space^2 个空间点）
    n_time: 时间维度上的诱导点数量
    
    返回:
    inducing_points: (n_space^2 * n_time, 3) - (lat, lon, time)
    """
    # 空间网格
    lat_coords = torch.linspace(lat_range[0], lat_range[1], n_space)
    lon_coords = torch.linspace(lon_range[0], lon_range[1], n_space)
    lat_grid, lon_grid = torch.meshgrid(lat_coords, lon_coords, indexing='ij')
    
    # 时间点
    time_coords = torch.linspace(0, n_time - 1, n_time)
    
    # 创建所有组合
    inducing_list = []
    for t in time_coords:
        for lat, lon in zip(lat_grid.ravel(), lon_grid.ravel()):
            inducing_list.append([lat.item(), lon.item(), t.item()])
    
    # 使用 float32 以匹配训练数据
    inducing_points = torch.tensor(inducing_list, dtype=torch.float32)
    
    if normalize:
        inducing_points[:, 0] = (inducing_points[:, 0] - lat_range[0]) / (lat_range[1] - lat_range[0])
        inducing_points[:, 1] = (inducing_points[:, 1] - lon_range[0]) / (lon_range[1] - lon_range[0])
        inducing_points[:, 2] = inducing_points[:, 2] / (n_time - 1)
    
    # 去重以防止重复诱导点导致奇异矩阵
    # 使用 torch.unique 去重
    inducing_points_unique, unique_indices = torch.unique(
        inducing_points, dim=0, return_inverse=True
    )
    # 去重信息已移除，避免不必要的输出
    
    return inducing_points_unique

