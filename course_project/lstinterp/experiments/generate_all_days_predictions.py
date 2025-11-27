"""
生成所有天数的预测可视化

该脚本为U-Net、GP和Tree模型生成所有31天的：
- 预测均值图
- 预测不确定性图
- 预测误差图

输出保存到 output/figures/all_days/ 目录
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import ProbUNet, GPSTModel, TreeBaseline
from lstinterp.config import UNetConfig, GPSTConfig, TreeConfig
from lstinterp.viz.maps import plot_mean_map, plot_std_map, plot_error_map
from lstinterp.utils import set_seed

# 输出目录
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "all_days"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子
set_seed(42)

# 数据路径
DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_test_data():
    """加载测试数据"""
    print("=" * 80)
    print("  加载测试数据")
    print("=" * 80)
    
    # 加载数据
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    print(f"测试数据形状: {test_tensor.shape} (lat, lon, time)")
    
    H, W, T = test_tensor.shape
    print(f"  - 空间维度: {H} × {W}")
    print(f"  - 时间维度: {T} 天")
    
    return test_tensor, H, W, T


def generate_unet_predictions(H, W, T):
    """生成U-Net模型的所有天数预测"""
    print("\n" + "=" * 80)
    print("  U-Net模型预测")
    print("=" * 80)
    
    # 检查模型文件
    model_path = OUTPUT_DIR / "models" / "unet_model.pth"
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行 train_unet.py 训练模型")
        return None
    
    # 加载模型
    print("加载U-Net模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # 创建模型
    model = ProbUNet(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    
    # 计算归一化参数（从训练数据）
    train_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    train_mask = (train_tensor != 0)
    train_mean = train_tensor[train_mask].mean()
    train_std = train_tensor[train_mask].std()
    
    print(f"归一化参数: mean={train_mean:.2f}, std={train_std:.2f}")
    
    # 创建测试数据集
    test_dataset = MODISDataset(
        test_tensor,
        mode="image",
        norm_mean=train_mean,
        norm_std=train_std
    )
    
    # 存储所有预测
    all_pred_mean = np.zeros((H, W, T))
    all_pred_std = np.zeros((H, W, T))
    all_true = np.zeros((H, W, T))
    
    print("\n生成预测...")
    with torch.no_grad():
        for t in range(T):
            if (t + 1) % 5 == 0:
                print(f"  处理第 {t+1}/{T} 天...")
            
            # 获取数据
            image, mask, target = test_dataset[t]
            # 拼接image和mask作为2通道输入（与训练时一致）
            image = torch.cat([image, mask], dim=0)  # (2, H, W)
            image = image.unsqueeze(0).to(device)  # (1, 2, H, W)
            
            # 预测
            pred_mean, pred_log_var = model(image)
            pred_mean = pred_mean.squeeze().cpu().numpy()
            pred_std = np.sqrt(np.exp(np.clip(pred_log_var.squeeze().cpu().numpy(), -10, 10)))
            
            # 反归一化
            all_pred_mean[:, :, t] = pred_mean * train_std + train_mean
            all_pred_std[:, :, t] = pred_std * train_std
            all_true[:, :, t] = target.numpy() * train_std + train_mean
            
            # 只在有观测的地方保留预测
            mask_2d = mask.squeeze().numpy()
            all_pred_mean[:, :, t][mask_2d == 0] = np.nan
            all_pred_std[:, :, t][mask_2d == 0] = np.nan
            all_true[:, :, t][mask_2d == 0] = np.nan
    
    print("✅ U-Net预测完成")
    return all_pred_mean, all_pred_std, all_true


def generate_gp_predictions(H, W, T):
    """生成GP模型的所有天数预测"""
    print("\n" + "=" * 80)
    print("  GP模型预测")
    print("=" * 80)
    
    # 检查模型文件
    model_path = OUTPUT_DIR / "models" / "gp_model.pth"
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行 train_gp.py 训练模型")
        return None, None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型checkpoint
    print("加载GP模型...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        # 兼容旧checkpoint：如果没有kernel_design属性，设置为默认值"separable"
        if not hasattr(config, 'kernel_design'):
            # 直接设置属性（如果config是dataclass实例，这应该是可以的）
            config.kernel_design = "separable"
    else:
        # 使用默认配置
        config = GPSTConfig(
            kernel_space="matern32",
            kernel_time="matern32",
            num_inducing=500,
            lr=0.01,
            num_epochs=50,
            kernel_design="separable"  # 默认使用separable设计
        )
    
    # 创建诱导点
    if 'inducing_points' in checkpoint:
        inducing_points = checkpoint['inducing_points'].to(device)
    else:
        # 创建默认诱导点
        from lstinterp.models.gp_st import create_inducing_points
        inducing_points = create_inducing_points(
            n_space=15,
            n_time=10,
            normalize=True
        ).to(device)
        if len(inducing_points) > config.num_inducing:
            indices = torch.randperm(len(inducing_points))[:config.num_inducing]
            inducing_points = inducing_points[indices]
    
    # 初始化模型（使用checkpoint中的初始化参数或默认值）
    # 对于旧checkpoint，这些参数可能不存在，使用默认值
    # 从tensor中提取scalar值（如果存在）
    if 'lengthscale_space' in checkpoint:
        ls_space = checkpoint['lengthscale_space']
        if isinstance(ls_space, torch.Tensor):
            if ls_space.numel() > 1:
                lengthscale_space = float(ls_space.mean().item())
            else:
                lengthscale_space = float(ls_space.item())
        else:
            lengthscale_space = float(ls_space)
    else:
        lengthscale_space = 0.5
    
    if 'lengthscale_time' in checkpoint:
        ls_time = checkpoint['lengthscale_time']
        if isinstance(ls_time, torch.Tensor):
            if ls_time.numel() > 1:
                lengthscale_time = float(ls_time.mean().item())
            else:
                lengthscale_time = float(ls_time.item())
        else:
            lengthscale_time = float(ls_time)
    else:
        lengthscale_time = 0.3
    
    if 'outputscale' in checkpoint:
        os_val = checkpoint['outputscale']
        if isinstance(os_val, torch.Tensor):
            outputscale = float(os_val.item())
        else:
            outputscale = float(os_val)
    else:
        outputscale = 1.0
    
    if 'noise' in checkpoint:
        noise_val = checkpoint['noise']
        if isinstance(noise_val, torch.Tensor):
            noise = float(noise_val.item())
        else:
            noise = float(noise_val)
    else:
        noise = 0.2
    
    # 初始化模型（使用scalar值而不是tensor）
    try:
        model = GPSTModel(
            inducing_points, config,
            lengthscale_space=lengthscale_space,
            lengthscale_time=lengthscale_time,
            outputscale=outputscale,
            noise=noise
        ).to(device)
    except TypeError:
        # 如果GPSTModel不接受这些参数，使用简化版本
        model = GPSTModel(inducing_points, config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.likelihood.eval()
    
    print("✅ GP模型加载完成")
    
    # 加载测试数据（点模式）
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    test_dataset = MODISDataset(test_tensor, mode="point", normalize_coords=True)
    
    # 重新组织为空间网格
    all_pred_mean = np.full((H, W, T), np.nan)
    all_pred_std = np.full((H, W, T), np.nan)
    all_true = np.full((H, W, T), np.nan)
    
    print("\n生成预测...")
    # 获取原始索引
    test_mask = (test_tensor != 0)
    lat_indices, lon_indices, t_indices = np.where(test_mask)
    
    # 批量预测（GP模型需要批量处理）
    batch_size = 1000
    n_points = len(test_dataset)
    
    X_test_list = []
    indices_list = []
    
    for idx in range(n_points):
        x, y = test_dataset[idx]
        X_test_list.append(x.numpy())
        indices_list.append((lat_indices[idx], lon_indices[idx], t_indices[idx]))
    
    X_test = np.array(X_test_list)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    print(f"测试点数: {n_points}")
    print("开始批量预测（可能需要一些时间）...")
    
    all_means = []
    all_stds = []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_end = min(i + batch_size, n_points)
            X_batch = X_test_tensor[i:batch_end]
            
            # GP预测
            mean, std = model.predict(X_batch)
            
            all_means.append(mean.cpu().numpy())
            all_stds.append(std.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  已处理: {batch_end}/{n_points} 点 ({batch_end/n_points*100:.1f}%)")
    
    pred_mean = np.concatenate(all_means)
    pred_std = np.concatenate(all_stds)
    
    # 填充到网格
    for idx, (lat_idx, lon_idx, t_idx) in enumerate(indices_list):
        all_pred_mean[lat_idx, lon_idx, t_idx] = pred_mean[idx]
        all_pred_std[lat_idx, lon_idx, t_idx] = pred_std[idx]
        all_true[lat_idx, lon_idx, t_idx] = test_dataset.values[idx]
    
    print("✅ GP预测完成")
    return all_pred_mean, all_pred_std, all_true


def generate_tree_predictions(H, W, T):
    """生成Tree模型的所有天数预测"""
    print("\n" + "=" * 80)
    print("  Tree模型预测")
    print("=" * 80)
    
    # 检查模型文件
    model_path = OUTPUT_DIR / "models" / "tree_model_xgb.pkl"
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行 train_tree.py 训练模型")
        return None
    
    import pickle
    
    # 加载模型
    print("加载Tree模型...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # 从保存的数据中获取模型对象
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
    else:
        # 如果没有包装，直接使用
        model = model_data
    
    print(f"模型类型: {type(model).__name__}")
    
    # 加载测试数据（点模式）
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    test_dataset = MODISDataset(test_tensor, mode="point", normalize_coords=True)
    
    # 重新组织为空间网格
    all_pred_mean = np.full((H, W, T), np.nan)
    all_pred_std = np.full((H, W, T), np.nan)
    all_true = np.full((H, W, T), np.nan)
    
    print("\n生成预测...")
    # 批量预测
    X_test = []
    indices_list = []  # 保存原始索引（lat_idx, lon_idx, t_idx）
    
    # 获取原始索引（从where得到）
    test_mask = (test_tensor != 0)
    lat_indices, lon_indices, t_indices = np.where(test_mask)
    
    for idx in range(len(test_dataset)):
        x, y = test_dataset[idx]
        X_test.append(x.numpy())
        # 保存原始索引
        indices_list.append((lat_indices[idx], lon_indices[idx], t_indices[idx]))
    
    X_test = np.array(X_test)
    print(f"测试点数: {len(X_test)}")
    
    # 预测
    # 检查是否是TreeBaseline对象
    if isinstance(model, TreeBaseline):
        # 检查是否有分位数回归
        if hasattr(model, 'config') and hasattr(model.config, 'quantile_regression') and model.config.quantile_regression:
            # 分位数回归，可以计算不确定性
            pred_mean, pred_std = model.predict_with_uncertainty(X_test)
        elif hasattr(model, 'quantile_models') and len(model.quantile_models) > 0:
            # 有分位数模型，可以使用不确定性
            pred_mean, pred_std = model.predict_with_uncertainty(X_test)
        else:
            # 普通预测
            pred_mean = model.predict(X_test)
            # 使用简单的启发式估计
            pred_std = np.abs(pred_mean - pred_mean.mean()) * 0.1 + 1.0
    else:
        # 直接调用predict
        pred_mean = model.predict(X_test)
        # 使用简单的启发式估计
        pred_std = np.abs(pred_mean - pred_mean.mean()) * 0.1 + 1.0
    
    # 填充到网格（使用原始索引）
    for idx, (lat_idx, lon_idx, t_idx) in enumerate(indices_list):
        all_pred_mean[lat_idx, lon_idx, t_idx] = pred_mean[idx]
        if isinstance(pred_std, np.ndarray):
            if len(pred_std.shape) == 1:
                all_pred_std[lat_idx, lon_idx, t_idx] = pred_std[idx]
            else:
                all_pred_std[lat_idx, lon_idx, t_idx] = pred_std[idx, 0]
        else:
            all_pred_std[lat_idx, lon_idx, t_idx] = pred_std  # scalar
        all_true[lat_idx, lon_idx, t_idx] = test_dataset.values[idx]
    
    print("✅ Tree预测完成")
    return all_pred_mean, all_pred_std, all_true


def plot_all_days_grid(
    pred_mean, pred_std, true_values,
    model_name, H, W, T,
    save_prefix,
    vmin_mean=None, vmax_mean=None,
    vmin_std=None, vmax_std=None,
    vmax_error=None
):
    """生成所有天数的网格图（8行×4列）
    
    参数:
        vmin_mean, vmax_mean: 预测均值的颜色范围（如果为None则使用当前数据范围）
        vmin_std, vmax_std: 预测标准差的颜色范围（如果为None则使用当前数据范围）
        vmax_error: 误差的最大绝对值（如果为None则使用当前数据范围）
    """
    print(f"\n生成{model_name}所有天数可视化（{T}天，8行×4列布局）...")
    
    # 计算每行的天数
    n_rows = 8
    n_cols = 4
    
    # 为每种图创建网格（减少留白，去掉顶部标题）
    fig_mean = plt.figure(figsize=(20, 40))
    fig_std = plt.figure(figsize=(20, 40))
    fig_error = plt.figure(figsize=(20, 40))
    
    # 减少hspace和wspace以减少留白
    gs_mean = GridSpec(n_rows, n_cols, figure=fig_mean, hspace=0.1, wspace=0.1, 
                       left=0.02, right=0.98, top=0.98, bottom=0.02)
    gs_std = GridSpec(n_rows, n_cols, figure=fig_std, hspace=0.1, wspace=0.1,
                      left=0.02, right=0.98, top=0.98, bottom=0.02)
    gs_error = GridSpec(n_rows, n_cols, figure=fig_error, hspace=0.1, wspace=0.1,
                        left=0.02, right=0.98, top=0.98, bottom=0.02)
    
    # 如果没有提供统一范围，使用当前数据范围
    if vmin_mean is None:
        vmin_mean = np.nanmin(pred_mean)
    if vmax_mean is None:
        vmax_mean = np.nanmax(pred_mean)
    if vmin_std is None:
        vmin_std = np.nanmin(pred_std)
    if vmax_std is None:
        vmax_std = np.nanmax(pred_std)
    
    errors = true_values - pred_mean
    if vmax_error is None:
        vmax_error = np.nanmax(np.abs(errors))
    
    # 打印使用的颜色范围（用于调试）
    print(f"  {model_name} 使用颜色范围:")
    print(f"    Mean: [{vmin_mean:.2f}, {vmax_mean:.2f}] K")
    print(f"    Std:  [{vmin_std:.2f}, {vmax_std:.2f}] K")
    print(f"    Error: [-{vmax_error:.2f}, {vmax_error:.2f}] K")
    
    for t in range(min(T, n_rows * n_cols)):
        row = t // n_cols
        col = t % n_cols
        
        # 预测均值图
        ax_mean = fig_mean.add_subplot(gs_mean[row, col])
        im_mean = ax_mean.imshow(
            pred_mean[:, :, t], 
            aspect='auto', origin='lower',
            cmap='jet_r', vmin=vmin_mean, vmax=vmax_mean
        )
        ax_mean.set_title(f'Day {t+1}', fontsize=8, pad=2)  # 减小字体和padding
        ax_mean.set_xlabel('Longitude Index', fontsize=7)
        ax_mean.set_ylabel('Latitude Index', fontsize=7)
        # 显式设置imshow的范围，确保颜色范围一致
        im_mean.set_clim(vmin=vmin_mean, vmax=vmax_mean)
        fig_mean.colorbar(im_mean, ax=ax_mean, label='Temperature (K)', shrink=0.8, aspect=20)
        
        # 预测不确定性图
        ax_std = fig_std.add_subplot(gs_std[row, col])
        im_std = ax_std.imshow(
            pred_std[:, :, t],
            aspect='auto', origin='lower',
            cmap='viridis', vmin=vmin_std, vmax=vmax_std
        )
        ax_std.set_title(f'Day {t+1}', fontsize=8, pad=2)  # 减小字体和padding
        ax_std.set_xlabel('Longitude Index', fontsize=7)
        ax_std.set_ylabel('Latitude Index', fontsize=7)
        # 显式设置imshow的范围，确保颜色范围一致
        im_std.set_clim(vmin=vmin_std, vmax=vmax_std)
        fig_std.colorbar(im_std, ax=ax_std, label='Std Dev (K)', shrink=0.8, aspect=20)
        
        # 预测误差图
        ax_error = fig_error.add_subplot(gs_error[row, col])
        error_t = errors[:, :, t]
        im_error = ax_error.imshow(
            error_t,
            aspect='auto', origin='lower',
            cmap='RdBu_r', vmin=-vmax_error, vmax=vmax_error
        )
        ax_error.set_title(f'Day {t+1}', fontsize=8, pad=2)  # 减小字体和padding
        ax_error.set_xlabel('Longitude Index', fontsize=7)
        ax_error.set_ylabel('Latitude Index', fontsize=7)
        # 显式设置imshow的范围，确保颜色范围一致
        im_error.set_clim(vmin=-vmax_error, vmax=vmax_error)
        fig_error.colorbar(im_error, ax=ax_error, label='Error (K)', shrink=0.8, aspect=20)
    
    # 保存
    mean_path = FIGURES_DIR / f"{save_prefix}_mean_all_days.png"
    std_path = FIGURES_DIR / f"{save_prefix}_std_all_days.png"
    error_path = FIGURES_DIR / f"{save_prefix}_error_all_days.png"
    
    # 移除顶部标题，直接保存（减少留白）
    fig_mean.savefig(mean_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_mean)
    print(f"✅ 预测均值网格图已保存: {mean_path}")
    
    fig_std.savefig(std_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_std)
    print(f"✅ 预测不确定性网格图已保存: {std_path}")
    
    fig_error.savefig(error_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_error)
    print(f"✅ 预测误差网格图已保存: {error_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("  生成所有天数的预测可视化")
    print("=" * 80)
    
    # 加载测试数据
    test_tensor, H, W, T = load_test_data()
    
    # 收集所有模型的预测结果
    all_results = {}
    
    # U-Net模型
    unet_results = generate_unet_predictions(H, W, T)
    if unet_results[0] is not None:
        all_results['unet'] = unet_results
        np.save(FIGURES_DIR / "unet_pred_mean.npy", unet_results[0])
        np.save(FIGURES_DIR / "unet_pred_std.npy", unet_results[1])
        np.save(FIGURES_DIR / "unet_true.npy", unet_results[2])
        print("✅ U-Net预测数据已保存")
    
    # Tree模型
    tree_results = generate_tree_predictions(H, W, T)
    if tree_results[0] is not None:
        all_results['tree'] = tree_results
        np.save(FIGURES_DIR / "tree_pred_mean.npy", tree_results[0])
        np.save(FIGURES_DIR / "tree_pred_std.npy", tree_results[1])
        np.save(FIGURES_DIR / "tree_true.npy", tree_results[2])
        print("✅ Tree预测数据已保存")
    
    # GP模型
    gp_results = generate_gp_predictions(H, W, T)
    if gp_results[0] is not None:
        all_results['gp'] = gp_results
        np.save(FIGURES_DIR / "gp_pred_mean.npy", gp_results[0])
        np.save(FIGURES_DIR / "gp_pred_std.npy", gp_results[1])
        np.save(FIGURES_DIR / "gp_true.npy", gp_results[2])
        print("✅ GP预测数据已保存")
    
    # 计算统一的颜色范围（跨所有模型）
    print("\n计算统一的颜色范围（跨所有模型）...")
    all_means = []
    all_stds = []
    all_errors = []
    
    for model_name, (pred_mean, pred_std, true_values) in all_results.items():
        all_means.append(pred_mean)
        all_stds.append(pred_std)
        errors = true_values - pred_mean
        all_errors.append(np.abs(errors))
    
    if all_means:
        # 统一的颜色范围
        vmin_mean = min(np.nanmin(m) for m in all_means)
        vmax_mean = max(np.nanmax(m) for m in all_means)
        vmin_std = max(0, min(np.nanmin(s) for s in all_stds))  # 确保std最小值至少为0
        vmax_std = max(np.nanmax(s) for s in all_stds)
        vmax_error = max(np.nanmax(e) for e in all_errors)
        
        print(f"统一颜色范围:")
        print(f"  Mean: [{vmin_mean:.2f}, {vmax_mean:.2f}] K")
        print(f"  Std:  [{vmin_std:.2f}, {vmax_std:.2f}] K")
        print(f"  Error: [-{vmax_error:.2f}, {vmax_error:.2f}] K")
        print(f"\n⚠️  所有模型将使用上述统一颜色范围绘制")
        
        # 使用统一范围绘制所有模型的图片
        if 'unet' in all_results:
            pred_mean, pred_std, true_values = all_results['unet']
            plot_all_days_grid(
                pred_mean, pred_std, true_values,
                "U-Net", H, W, T,
                "unet",
                vmin_mean=vmin_mean, vmax_mean=vmax_mean,
                vmin_std=vmin_std, vmax_std=vmax_std,
                vmax_error=vmax_error
            )
        
        if 'tree' in all_results:
            pred_mean, pred_std, true_values = all_results['tree']
            plot_all_days_grid(
                pred_mean, pred_std, true_values,
                "Tree (XGBoost)", H, W, T,
                "tree",
                vmin_mean=vmin_mean, vmax_mean=vmax_mean,
                vmin_std=vmin_std, vmax_std=vmax_std,
                vmax_error=vmax_error
            )
        
        if 'gp' in all_results:
            pred_mean, pred_std, true_values = all_results['gp']
            plot_all_days_grid(
                pred_mean, pred_std, true_values,
                "GP (Sparse)", H, W, T,
                "gp",
                vmin_mean=vmin_mean, vmax_mean=vmax_mean,
                vmin_std=vmin_std, vmax_std=vmax_std,
                vmax_error=vmax_error
            )
    
    print("\n" + "=" * 80)
    print("  完成")
    print("=" * 80)
    print(f"\n所有图表已保存到: {FIGURES_DIR}")
    print("\n生成的文件:")
    print("  - unet_mean_all_days.png: U-Net所有天数的预测均值")
    print("  - unet_std_all_days.png: U-Net所有天数的不确定性")
    print("  - unet_error_all_days.png: U-Net所有天数的误差")
    print("  - tree_mean_all_days.png: Tree所有天数的预测均值")
    print("  - tree_std_all_days.png: Tree所有天数的不确定性")
    print("  - tree_error_all_days.png: Tree所有天数的误差")


if __name__ == "__main__":
    main()

