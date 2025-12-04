"""Spatio-Temporal Gaussian Process Models"""
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
    
    # Try importing advanced kernels (naming may vary across GPyTorch versions)
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
    """Spatio-Temporal Gaussian Process Configuration"""
    kernel_design: Literal["separable", "additive", "non_separable"] = "separable"
    kernel_space: Literal["matern32", "matern52", "rbf"] = "matern32"
    kernel_time: Literal["exp", "matern32", "rbf"] = "matern32"
    num_inducing: int = 800
    lr: float = 0.01
    num_epochs: int = 50
    batch_size: int = 1000


class STSeparableGP(ApproximateGP):
    """
    Spatio-Temporal Inducing Point Gaussian Process - Design 1: Separable Kernel
    
    Uses a separable kernel: k(x, x') = k_space(lat, lon) * k_time(t)
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig, 
                 lengthscale_space: float = 0.5, lengthscale_time: float = 0.3,
                 outputscale: float = 10.0):
        if not GPYTORCH_AVAILABLE:
            raise ImportError("gpytorch is required")
        
        self.config = config
        
        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # Variational strategy (increased jitter for numerical stability)
        # Use a larger jitter value (1e-4) to handle numerical instability
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-4,  # Increased to 1e-4
        )
        
        super().__init__(variational_strategy)
        
        # Mean module
        self.mean_module = ConstantMean()
        
        # Spatial kernel (acts on first 2 dims: lat, lon)
        # Ensure lengthscale is not too small (avoid numerical instability)
        # Handle tensor types (if tensor, take mean or item)
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
        
        # Add lengthscale constraint (ensure positive and not too small)
        base_space.lengthscale_constraint = Positive()
        # Initialize lengthscale (use same dtype as inducing_points)
        base_space.lengthscale = torch.tensor([[lengthscale_space_val, lengthscale_space_val]], 
                                              dtype=inducing_points.dtype, device=inducing_points.device)
        
        # Add outputscale constraint (limit within reasonable range to avoid numerical instability)
        # Allow larger outputscale to adapt to different data scales
        outputscale_val = min(max(outputscale, 0.1), 50.0)  # Limit to [0.1, 50.0]
        self.covar_space = ScaleKernel(base_space)
        self.covar_space.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        # Initialize outputscale (reasonable estimate based on data variance)
        self.covar_space.outputscale = torch.tensor(outputscale_val, dtype=inducing_points.dtype, device=inducing_points.device)
        
        # Temporal kernel (acts on 3rd dim: time)
        # Ensure temporal lengthscale is not too small
        # Handle tensor types
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
        else:  # exp (equivalent to Matern 1/2)
            base_time = MaternKernel(nu=0.5, ard_num_dims=1)
        
        # Add lengthscale constraint
        base_time.lengthscale_constraint = Positive()
        # Initialize temporal lengthscale
        base_time.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        
        self.covar_time = ScaleKernel(base_time)
        self.covar_time.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        # Initialize outputscale (use same constraint)
        self.covar_time.outputscale = torch.tensor(outputscale_val, dtype=inducing_points.dtype, device=inducing_points.device)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass
        
        Args:
        x: (N, 3) - (lat, lon, time)
        
        Returns:
        MultivariateNormal distribution
        """
        # Separate spatial and temporal dimensions
        space_x = x[:, :2]  # (N, 2) - lat, lon
        time_x = x[:, 2:3]  # (N, 1) - time
        
        # Mean
        mean_x = self.mean_module(x)
        
        # Covariance: space kernel * time kernel (separable kernel)
        covar_x = self.covar_space(space_x) * self.covar_time(time_x)
        
        if gpytorch is None:
            raise ImportError("gpytorch is required")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class STAdditiveGP(ApproximateGP):
    """
    Spatio-Temporal Inducing Point Gaussian Process - Design 2: Additive Kernel
    
    Uses additive kernel: k(x, x') = k_RQ(space) + k_Periodic(time) + k_Linear(time)
    RQ (Rational Quadratic) for space, Periodic + Linear for time
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig,
                 lengthscale_space: float = 0.5, lengthscale_time: float = 0.3,
                 outputscale: float = 10.0, alpha: float = 1.0, period: float = 1.0):
        if not GPYTORCH_AVAILABLE:
            raise ImportError("gpytorch is required")
        
        self.config = config
        
        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-3,  # Increased jitter
        )
        
        super().__init__(variational_strategy)
        
        # Mean module
        self.mean_module = ConstantMean()
        
        # Spatial kernel: Rational Quadratic (RQ) or Matern - captures multi-scale spatial correlations
        space_x = inducing_points[:, :2]  # (M, 2)
        # Handle tensor types
        if isinstance(lengthscale_space, torch.Tensor):
            if lengthscale_space.numel() > 1:
                lengthscale_space_val = float(lengthscale_space.mean().item())
            else:
                lengthscale_space_val = float(lengthscale_space.item())
        else:
            lengthscale_space_val = float(lengthscale_space)
        lengthscale_space_val = max(lengthscale_space_val, 0.3)
        
        if RQKernel_available:
            # Use RQ Kernel (if available)
            # RQ parameter: alpha controls smoothness, smaller alpha means smoother
            alpha_val = max(min(alpha, 5.0), 0.1)
            try:
                base_space = RQKernel(ard_num_dims=2, alpha_constraint=Interval(lower_bound=0.1, upper_bound=5.0))
                base_space.alpha = torch.tensor(alpha_val, dtype=inducing_points.dtype, device=inducing_points.device)
            except:
                # If RQKernel initialization fails, use MaternKernel as fallback
                base_space = MaternKernel(nu=1.5, ard_num_dims=2)
        else:
            # If RQKernel not available, use MaternKernel as fallback
            # Matern 3/2 can approximate multi-scale features
            base_space = MaternKernel(nu=1.5, ard_num_dims=2)
        
        base_space.lengthscale_constraint = Positive()
        base_space.lengthscale = torch.tensor([[lengthscale_space_val, lengthscale_space_val]], 
                                              dtype=inducing_points.dtype, device=inducing_points.device)
        
        outputscale_val = min(max(outputscale, 0.1), 50.0)
        self.covar_space = ScaleKernel(base_space)
        self.covar_space.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        self.covar_space.outputscale = torch.tensor(outputscale_val, dtype=inducing_points.dtype, device=inducing_points.device)
        
        # Temporal kernel 1: Periodic - captures periodic temporal patterns
        # If PeriodicKernel not available, use MaternKernel as fallback
        time_x = inducing_points[:, 2:3]  # (M, 1)
        # Handle tensor types
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
                # If PeriodicKernel initialization fails, use RBFKernel as fallback
                base_periodic = RBFKernel(ard_num_dims=1)
                base_periodic.lengthscale_constraint = Positive()
                base_periodic.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        else:
            # If PeriodicKernel not available, use RBFKernel as fallback
            base_periodic = RBFKernel(ard_num_dims=1)
            base_periodic.lengthscale_constraint = Positive()
            base_periodic.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        
        self.covar_periodic = ScaleKernel(base_periodic)
        self.covar_periodic.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        self.covar_periodic.outputscale = torch.tensor(outputscale_val * 0.5, dtype=inducing_points.dtype, device=inducing_points.device)
        
        # Temporal kernel 2: Linear - captures linear temporal trends
        # If LinearKernel not available, use RBFKernel as fallback
        if LinearKernel_available:
            try:
                base_linear = LinearKernel(ard_num_dims=1)
                base_linear.variance_constraint = Positive()
                base_linear.variance = torch.tensor(1.0, dtype=inducing_points.dtype, device=inducing_points.device)
            except:
                # If LinearKernel initialization fails, use RBFKernel as fallback
                base_linear = RBFKernel(ard_num_dims=1)
                base_linear.lengthscale_constraint = Positive()
                base_linear.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        else:
            # If LinearKernel not available, use RBFKernel as fallback
            base_linear = RBFKernel(ard_num_dims=1)
            base_linear.lengthscale_constraint = Positive()
            base_linear.lengthscale = torch.tensor([[lengthscale_time_val]], dtype=inducing_points.dtype, device=inducing_points.device)
        
        self.covar_linear = ScaleKernel(base_linear)
        self.covar_linear.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        self.covar_linear.outputscale = torch.tensor(outputscale_val * 0.3, dtype=inducing_points.dtype, device=inducing_points.device)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass
        
        Args:
        x: (N, 3) - (lat, lon, time)
        
        Returns:
        MultivariateNormal distribution
        """
        # Separate spatial and temporal dimensions
        space_x = x[:, :2]  # (N, 2) - lat, lon
        time_x = x[:, 2:3]  # (N, 1) - time
        
        # Mean
        mean_x = self.mean_module(x)
        
        # Covariance: Additive kernel k_RQ(space) + k_Periodic(time) + k_Linear(time)
        covar_x = self.covar_space(space_x) + self.covar_periodic(time_x) + self.covar_linear(time_x)
        
        if gpytorch is None:
            raise ImportError("gpytorch is required")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class STNonSeparableGP(ApproximateGP):
    """
    Spatio-Temporal Inducing Point Gaussian Process - Design 3: Non-Separable Kernel
    
    Uses non-separable kernel: Applies Matern kernel directly on the entire 3D input (lat, lon, time)
    Captures the most complex spatio-temporal interactions
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig,
                 lengthscale: float = 0.5, outputscale: float = 10.0):
        if not GPYTORCH_AVAILABLE:
            raise ImportError("gpytorch is required")
        
        self.config = config
        
        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-3,  # Increased jitter
        )
        
        super().__init__(variational_strategy)
        
        # Mean module
        self.mean_module = ConstantMean()
        
        # Non-separable kernel: Matern kernel directly on 3D input
        # Use ARD (Automatic Relevance Determination) to learn different lengthscales for each dimension
        # Handle tensor types
        if isinstance(lengthscale, torch.Tensor):
            if lengthscale.numel() > 1:
                lengthscale_val = float(lengthscale.mean().item())
            else:
                lengthscale_val = float(lengthscale.item())
        else:
            lengthscale_val = float(lengthscale)
        lengthscale_val = max(lengthscale_val, 0.3)
        
        # Matern 3/2 captures smooth but not overly smooth spatio-temporal correlations
        if config.kernel_space == "matern32":
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=3)
        elif config.kernel_space == "matern52":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=3)
        else:  # rbf
            base_kernel = RBFKernel(ard_num_dims=3)
        
        base_kernel.lengthscale_constraint = Positive()
        # Initialize: Set lengthscales for lat, lon, time respectively
        base_kernel.lengthscale = torch.tensor([[lengthscale_val, lengthscale_val, lengthscale_val * 0.6]], 
                                               dtype=inducing_points.dtype, device=inducing_points.device)
        
        outputscale_val = min(max(outputscale, 0.1), 50.0)
        self.covar_module = ScaleKernel(base_kernel)
        self.covar_module.outputscale_constraint = Interval(lower_bound=0.1, upper_bound=50.0)
        self.covar_module.outputscale = torch.tensor(outputscale_val, dtype=inducing_points.dtype, device=inducing_points.device)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass
        
        Args:
        x: (N, 3) - (lat, lon, time)
        
        Returns:
        MultivariateNormal distribution
        """
        # Mean
        mean_x = self.mean_module(x)
        
        # Covariance: Computed directly on the entire 3D input (non-separable)
        covar_x = self.covar_module(x)
        
        if gpytorch is None:
            raise ImportError("gpytorch is required")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSTModel(nn.Module):
    """
    Spatio-Temporal GP Model Wrapper
    
    Selects different kernel designs based on kernel_design configuration:
    - "separable": Design 1 - Separable kernel k_space x k_time
    - "additive": Design 2 - Additive kernel k_RQ(space) + k_Periodic(time) + k_Linear(time)
    - "non_separable": Design 3 - Non-separable kernel k_Matern(3D input)
    """
    
    def __init__(self, inducing_points: torch.Tensor, config: GPSTConfig,
                 lengthscale_space: float = 0.5, lengthscale_time: float = 0.3,
                 outputscale: float = 10.0, noise: float = 1.0,
                 alpha: float = 1.0, period: float = 1.0):
        super().__init__()
        if not GPYTORCH_AVAILABLE:
            raise ImportError("gpytorch is required")
        self.config = config
        
        # Select kernel design based on config
        if config.kernel_design == "separable":
            # Design 1: Separable Kernel
            self.gp = STSeparableGP(inducing_points, config, 
                                  lengthscale_space=lengthscale_space,
                                  lengthscale_time=lengthscale_time,
                                  outputscale=outputscale)
        elif config.kernel_design == "additive":
            # Design 2: Additive Kernel
            self.gp = STAdditiveGP(inducing_points, config,
                                 lengthscale_space=lengthscale_space,
                                 lengthscale_time=lengthscale_time,
                                 outputscale=outputscale,
                                 alpha=alpha, period=period)
        elif config.kernel_design == "non_separable":
            # Design 3: Non-Separable Kernel
            # Use average lengthscale as initial value
            avg_lengthscale = (lengthscale_space + lengthscale_time) / 2.0
            self.gp = STNonSeparableGP(inducing_points, config,
                                     lengthscale=avg_lengthscale,
                                     outputscale=outputscale)
        else:
            raise ValueError(f"Unknown kernel design: {config.kernel_design}")
        
        # Add noise constraint (ensure noise is positive and not too small)
        noise_val = max(min(noise, 5.0), 0.01)  # Limit to [0.01, 5.0]
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise_constraint = Interval(lower_bound=0.01, upper_bound=5.0)
        # Initialize noise (reasonable estimate based on data variance)
        self.likelihood.noise = torch.tensor(noise_val, dtype=inducing_points.dtype, device=inducing_points.device)
        
        # Re-initialize noise (duplicated in original code, keeping for structure consistency but cleaning up)
        # (The previous duplicate lines were removed)
    
    def forward(self, x: torch.Tensor):
        """Forward pass"""
        return self.gp(x)
    
    def predict(self, x: torch.Tensor) -> tuple:
        """
        Predict
        
        Returns:
        mean: Predictive mean
        std: Predictive standard deviation
        """
        self.eval()
        self.likelihood.eval()
        
        if gpytorch is None:
            raise ImportError("gpytorch is required")
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
    Create inducing points
    
    Args:
    n_space: Number of inducing points in spatial dimension (creates n_space^2 points)
    n_time: Number of inducing points in temporal dimension
    
    Returns:
    inducing_points: (n_space^2 * n_time, 3) - (lat, lon, time)
    """
    # Spatial grid
    lat_coords = torch.linspace(lat_range[0], lat_range[1], n_space)
    lon_coords = torch.linspace(lon_range[0], lon_range[1], n_space)
    lat_grid, lon_grid = torch.meshgrid(lat_coords, lon_coords, indexing='ij')
    
    # Time points
    time_coords = torch.linspace(0, n_time - 1, n_time)
    
    # Create all combinations
    inducing_list = []
    for t in time_coords:
        for lat, lon in zip(lat_grid.ravel(), lon_grid.ravel()):
            inducing_list.append([lat.item(), lon.item(), t.item()])
    
    # Use float32 to match training data
    inducing_points = torch.tensor(inducing_list, dtype=torch.float32)
    
    if normalize:
        inducing_points[:, 0] = (inducing_points[:, 0] - lat_range[0]) / (lat_range[1] - lat_range[0])
        inducing_points[:, 1] = (inducing_points[:, 1] - lon_range[0]) / (lon_range[1] - lon_range[0])
        inducing_points[:, 2] = inducing_points[:, 2] / (n_time - 1)
    
    # Remove duplicates to prevent singular matrices from duplicate inducing points
    # Use torch.unique for deduplication
    inducing_points_unique, unique_indices = torch.unique(
        inducing_points, dim=0, return_inverse=True
    )
    
    return inducing_points_unique
