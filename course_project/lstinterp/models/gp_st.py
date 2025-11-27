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
    
    # 尝试导入高级核（可能在不同版本的GPyTorch中有不同的命名）
    try:
        from gpytorch.kernels import PeriodicKernel, LinearKernel, RQKernel
        PeriodicKernel_available = True
        LinearKernel_available = True
        RQKernel_available = True
    except ImportError:
        try:
            from gpytorch.kernels import PeriodicKernel, LinearKernel
            from gpytorch.kernels.kernels import RationalQuadraticKernel as RQKernel
            PeriodicKernel_available = True
            LinearKernel_available = True
            RQKernel_available = True
        except ImportError:
            PeriodicKernel_available = False
            LinearKernel_available = False
            RQKernel_available = False
            PeriodicKernel = None
            LinearKernel = None
            RQKernel = None
    
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    PeriodicKernel_available = False
    LinearKernel_available = False
    RQKernel_available = False
    ApproximateGP = object
    gpytorch = None
    Positive = None
    Interval = None
    PeriodicKernel = None
    LinearKernel = None
    RQKernel = None


@dataclass
class GPSTConfig:
    """时空高斯过程配置"""
    kernel_space: Literal["matern32", "matern52", "rbf"] = "matern32"
    kernel_time: Literal["exp", "matern32", "rbf"] = "matern32"
    num_inducing: int = 800
    lr: float = 0.01
    num_epochs: int = 50
    batch_size: int = 1000


class STSeparableGP(ApproximateGP):
    """
    时空诱导点高斯过程 - Design 1: 可分离核
    
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
        # 处理tensor类型（如果是tensor，取平均值或第一个值）
        if isinstance(lengthscale_space, torch.Tensor):
            if lengthscale_space.numel() > 1:
                lengthscale_space_val = float(lengthscale_space.mean().item())
            else:
                lengthscale_space_val = float(lengthscale_space.item())
        else:
            lengthscale_space_val = float(lengthscale_space)
        lengthscale_space_val = max(lengthscale_space_val, 0.3)
        
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
        # 处理tensor类型（如果是tensor，取平均值或第一个值）
        if isinstance(lengthscale_time, torch.Tensor):
            if lengthscale_time.numel() > 1:
                lengthscale_time_val = float(lengthscale_time.mean().item())
            else:
                lengthscale_time_val = float(lengthscale_time.item())
        else:
            lengthscale_time_val = float(lengthscale_time)
        lengthscale_time_val = max(lengthscale_time_val, 0.2)
        
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


class STAdditiveGP(ApproximateGP):
    """
    时空诱导点高斯过程 - Design 2: 加性核
    
    使用加性核：k(x, x') = k_RQ(space) + k_Periodic(time) + k_Linear(time)
    RQ (Rational Quadratic) 用于空间，Periodic + Linear 用于时间
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig,
                 lengthscale_space: float = 0.5, lengthscale_time: float = 0.3,
                 outputscale: float = 10.0, alpha: float = 1.0, period: float = 1.0):
        if not GPYTORCH_AVAILABLE:
            raise ImportError("需要安装 gpytorch")
        
        self.config = config
        
        # 变分分布
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # 变分策略
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-4,
        )
        
        super().__init__(variational_strategy)
        
        # 均值模块
        self.mean_module = ConstantMean()
        
        # 空间核：Rational Quadratic (RQ) 或 Matern - 用于捕获多尺度空间相关性
        space_x = inducing_points[:, :2]  # (M, 2)
        # 处理tensor类型（如果是tensor，取平均值或第一个值）
        if isinstance(lengthscale_space, torch.Tensor):
            if lengthscale_space.numel() > 1:
                lengthscale_space_val = float(lengthscale_space.mean().item())
            else:
                lengthscale_space_val = float(lengthscale_space.item())
        else:
            lengthscale_space_val = float(lengthscale_space)
        lengthscale_space_val = max(lengthscale_space_val, 0.3)
        
        if RQKernel_available:
            # 使用 RQ 核（如果可用）
            # RQ 核参数：alpha 控制平滑度，较小的 alpha 意味着更平滑
            alpha_val = max(min(alpha, 5.0), 0.1)
            try:
                base_space = RQKernel(ard_num_dims=2, alpha_constraint=Interval(lower_bound=0.1, upper_bound=5.0))
                base_space.alpha = torch.tensor(alpha_val, dtype=inducing_points.dtype, device=inducing_points.device)
            except:
                # 如果RQKernel初始化失败，使用MaternKernel作为替代
                base_space = MaternKernel(nu=1.5, ard_num_dims=2)
        else:
            # 如果RQKernel不可用，使用MaternKernel作为替代
            # Matern 3/2 可以近似捕获多尺度特征
            base_space = MaternKernel(nu=1.5, ard_num_dims=2)
        
        base_space.lengthscale_constraint = Positive()
        base_space.lengthscale = torch.tensor([[lengthscale_space_val, lengthscale_space_val]], 
                                              dtype=inducing_points.dtype, device=inducing_points.device)
        
        outputscale_val = min(max(outputscale, 0.1), 50.0)
        self.covar_space = ScaleKernel(base_space)
        self.covar_space.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        self.covar_space.outputscale = torch.tensor(outputscale_val, dtype=inducing_points.dtype, device=inducing_points.device)
        
        # 时间核1：Periodic - 用于捕获周期性时间模式
        # 如果PeriodicKernel不可用，使用MaternKernel作为替代
        time_x = inducing_points[:, 2:3]  # (M, 1)
        # 处理tensor类型（如果是tensor，取平均值或第一个值）
        if isinstance(lengthscale_time, torch.Tensor):
            if lengthscale_time.numel() > 1:
                lengthscale_time_val = float(lengthscale_time.mean().item())
            else:
                lengthscale_time_val = float(lengthscale_time.item())
        else:
            lengthscale_time_val = float(lengthscale_time)
        lengthscale_time_val = max(lengthscale_time_val, 0.2)
        period_val = max(period, 0.1)
        
        if PeriodicKernel_available:
            try:
                base_periodic = PeriodicKernel(ard_num_dims=1)
                base_periodic.period_length_constraint = Positive()
                base_periodic.period_length = torch.tensor([[period_val]], dtype=inducing_points.dtype, device=inducing_points.device)
                base_periodic.lengthscale_constraint = Positive()
                base_periodic.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
            except:
                # 如果PeriodicKernel初始化失败，使用RBFKernel作为替代
                base_periodic = RBFKernel(ard_num_dims=1)
                base_periodic.lengthscale_constraint = Positive()
                base_periodic.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        else:
            # 如果PeriodicKernel不可用，使用RBFKernel作为替代
            base_periodic = RBFKernel(ard_num_dims=1)
            base_periodic.lengthscale_constraint = Positive()
            base_periodic.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        
        self.covar_periodic = ScaleKernel(base_periodic)
        self.covar_periodic.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        self.covar_periodic.outputscale = torch.tensor(outputscale_val * 0.5, dtype=inducing_points.dtype, device=inducing_points.device)
        
        # 时间核2：Linear - 用于捕获线性时间趋势
        # 如果LinearKernel不可用，使用RBFKernel作为替代
        if LinearKernel_available:
            try:
                base_linear = LinearKernel(ard_num_dims=1)
                base_linear.variance_constraint = Positive()
                base_linear.variance = torch.tensor(1.0, dtype=inducing_points.dtype, device=inducing_points.device)
            except:
                # 如果LinearKernel初始化失败，使用RBFKernel作为替代
                base_linear = RBFKernel(ard_num_dims=1)
                base_linear.lengthscale_constraint = Positive()
                base_linear.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        else:
            # 如果LinearKernel不可用，使用RBFKernel作为替代
            base_linear = RBFKernel(ard_num_dims=1)
            base_linear.lengthscale_constraint = Positive()
            base_linear.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        
        self.covar_linear = ScaleKernel(base_linear)
        self.covar_linear.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        self.covar_linear.outputscale = torch.tensor(outputscale_val * 0.3, dtype=inducing_points.dtype, device=inducing_points.device)
    
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
        
        # 协方差：加性核 k_RQ(space) + k_Periodic(time) + k_Linear(time)
        covar_x = self.covar_space(space_x) + self.covar_periodic(time_x) + self.covar_linear(time_x)
        
        if gpytorch is None:
            raise ImportError("需要安装 gpytorch")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class STNonSeparableGP(ApproximateGP):
    """
    时空诱导点高斯过程 - Design 3: 非分离核
    
    使用非分离核：直接对整个3D输入 (lat, lon, time) 使用 Matern 核
    捕获最复杂的时空交互
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig,
                 lengthscale: float = 0.5, outputscale: float = 10.0):
        if not GPYTORCH_AVAILABLE:
            raise ImportError("需要安装 gpytorch")
        
        self.config = config
        
        # 变分分布
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # 变分策略
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-4,
        )
        
        super().__init__(variational_strategy)
        
        # 均值模块
        self.mean_module = ConstantMean()
        
        # 非分离核：直接对整个3D输入使用 Matern 核
        # 使用 ARD (Automatic Relevance Determination) 为每个维度学习不同的长度尺度
        # 处理tensor类型（如果是tensor，取平均值或第一个值）
        if isinstance(lengthscale, torch.Tensor):
            if lengthscale.numel() > 1:
                lengthscale_val = float(lengthscale.mean().item())
            else:
                lengthscale_val = float(lengthscale.item())
        else:
            lengthscale_val = float(lengthscale)
        lengthscale_val = max(lengthscale_val, 0.3)
        
        # Matern 3/2 用于捕获平滑但不过度平滑的时空相关性
        if config.kernel_space == "matern32":
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=3)
        elif config.kernel_space == "matern52":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=3)
        else:  # rbf
            base_kernel = RBFKernel(ard_num_dims=3)
        
        base_kernel.lengthscale_constraint = Positive()
        # 初始化：为 lat, lon, time 分别设置长度尺度
        base_kernel.lengthscale = torch.tensor([[lengthscale_val, lengthscale_val, lengthscale_val * 0.6]], 
                                               dtype=inducing_points.dtype, device=inducing_points.device)
        
        outputscale_val = min(max(outputscale, 0.1), 50.0)
        self.covar_module = ScaleKernel(base_kernel)
        self.covar_module.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        self.covar_module.outputscale = torch.tensor(outputscale_val, dtype=inducing_points.dtype, device=inducing_points.device)
    
    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        参数:
        x: (N, 3) - (lat, lon, time)
        
        返回:
        MultivariateNormal分布
        """
        # 均值
        mean_x = self.mean_module(x)
        
        # 协方差：直接对整个3D输入计算（非分离）
        covar_x = self.covar_module(x)
        
        if gpytorch is None:
            raise ImportError("需要安装 gpytorch")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSTModel(nn.Module):
    """
    时空GP模型包装器
    
    根据 kernel_design 配置选择不同的核函数设计：
    - "separable": Design 1 - 可分离核 k_space × k_time
    - "additive": Design 2 - 加性核 k_RQ(space) + k_Periodic(time) + k_Linear(time)
    - "non_separable": Design 3 - 非分离核 k_Matern(3D input)
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig,
                 lengthscale_space: float = 0.5, lengthscale_time: float = 0.3,
                 outputscale: float = 10.0, noise: float = 1.0,
                 alpha: float = 1.0, period: float = 1.0):
        super().__init__()
        if not GPYTORCH_AVAILABLE:
            raise ImportError("需要安装 gpytorch")
        self.config = config
        
        # 根据 kernel_design 选择不同的核函数设计
        if config.kernel_design == "separable":
            # Design 1: 可分离核
            self.gp = STSeparableGP(inducing_points, config, 
                                  lengthscale_space=lengthscale_space,
                                  lengthscale_time=lengthscale_time,
                                  outputscale=outputscale)
        elif config.kernel_design == "additive":
            # Design 2: 加性核
            self.gp = STAdditiveGP(inducing_points, config,
                                 lengthscale_space=lengthscale_space,
                                 lengthscale_time=lengthscale_time,
                                 outputscale=outputscale,
                                 alpha=alpha, period=period)
        elif config.kernel_design == "non_separable":
            # Design 3: 非分离核
            # 使用平均长度尺度作为初始值
            avg_lengthscale = (lengthscale_space + lengthscale_time) / 2.0
            self.gp = STNonSeparableGP(inducing_points, config,
                                     lengthscale=avg_lengthscale,
                                     outputscale=outputscale)
        else:
            raise ValueError(f"未知的核函数设计: {config.kernel_design}")
        
        # 添加噪声约束（确保噪声为正且不会太小）
        noise_val = max(min(noise, 5.0), 0.01)  # 限制在 [0.01, 5.0]
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise_constraint = Interval(lower_bound=0.01, upper_bound=5.0)
        # 初始化噪声（基于数据方差的合理估计）
        self.likelihood.noise = torch.tensor(noise_val, dtype=inducing_points.dtype, device=inducing_points.device)
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

