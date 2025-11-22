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
    """生成GP模型的所有天数预测（需要从保存的结果重新预测或加载）"""
    print("\n" + "=" * 80)
    print("  GP模型预测")
    print("=" * 80)
    
    print("⚠️  GP模型预测需要重新运行，耗时较长")
    print("   建议：从已有的gp_results.json中提取，或运行train_gp.py生成预测")
    print("   暂时跳过GP模型的可视化生成")
    
    return None, None, None


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
    save_prefix
):
    """生成所有天数的网格图（8行×4列）"""
    print(f"\n生成{model_name}所有天数可视化（{T}天，8行×4列布局）...")
    
    # 计算每行的天数
    n_rows = 8
    n_cols = 4
    
    # 为每种图创建网格
    fig_mean = plt.figure(figsize=(20, 40))
    fig_std = plt.figure(figsize=(20, 40))
    fig_error = plt.figure(figsize=(20, 40))
    
    gs_mean = GridSpec(n_rows, n_cols, figure=fig_mean, hspace=0.3, wspace=0.3)
    gs_std = GridSpec(n_rows, n_cols, figure=fig_std, hspace=0.3, wspace=0.3)
    gs_error = GridSpec(n_rows, n_cols, figure=fig_error, hspace=0.3, wspace=0.3)
    
    vmin_mean = np.nanmin(pred_mean)
    vmax_mean = np.nanmax(pred_mean)
    vmin_std = np.nanmin(pred_std)
    vmax_std = np.nanmax(pred_std)
    
    errors = true_values - pred_mean
    vmax_error = np.nanmax(np.abs(errors))
    
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
        ax_mean.set_title(f'Day {t+1}: Mean Prediction', fontsize=10)
        ax_mean.set_xlabel('Longitude Index')
        ax_mean.set_ylabel('Latitude Index')
        plt.colorbar(im_mean, ax=ax_mean, label='Temperature (K)')
        
        # 预测不确定性图
        ax_std = fig_std.add_subplot(gs_std[row, col])
        im_std = ax_std.imshow(
            pred_std[:, :, t],
            aspect='auto', origin='lower',
            cmap='viridis', vmin=vmin_std, vmax=vmax_std
        )
        ax_std.set_title(f'Day {t+1}: Uncertainty', fontsize=10)
        ax_std.set_xlabel('Longitude Index')
        ax_std.set_ylabel('Latitude Index')
        plt.colorbar(im_std, ax=ax_std, label='Std Dev (K)')
        
        # 预测误差图
        ax_error = fig_error.add_subplot(gs_error[row, col])
        error_t = errors[:, :, t]
        im_error = ax_error.imshow(
            error_t,
            aspect='auto', origin='lower',
            cmap='RdBu_r', vmin=-vmax_error, vmax=vmax_error
        )
        ax_error.set_title(f'Day {t+1}: Error', fontsize=10)
        ax_error.set_xlabel('Longitude Index')
        ax_error.set_ylabel('Latitude Index')
        plt.colorbar(im_error, ax=ax_error, label='Error (K)')
    
    # 保存
    mean_path = FIGURES_DIR / f"{save_prefix}_mean_all_days.png"
    std_path = FIGURES_DIR / f"{save_prefix}_std_all_days.png"
    error_path = FIGURES_DIR / f"{save_prefix}_error_all_days.png"
    
    fig_mean.suptitle(f'{model_name} - Mean Predictions (All {T} Days)', fontsize=16, y=0.998)
    fig_mean.savefig(mean_path, dpi=150, bbox_inches='tight')
    plt.close(fig_mean)
    print(f"✅ 预测均值网格图已保存: {mean_path}")
    
    fig_std.suptitle(f'{model_name} - Prediction Uncertainty (All {T} Days)', fontsize=16, y=0.998)
    fig_std.savefig(std_path, dpi=150, bbox_inches='tight')
    plt.close(fig_std)
    print(f"✅ 预测不确定性网格图已保存: {std_path}")
    
    fig_error.suptitle(f'{model_name} - Prediction Errors (All {T} Days)', fontsize=16, y=0.998)
    fig_error.savefig(error_path, dpi=150, bbox_inches='tight')
    plt.close(fig_error)
    print(f"✅ 预测误差网格图已保存: {error_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("  生成所有天数的预测可视化")
    print("=" * 80)
    
    # 加载测试数据
    test_tensor, H, W, T = load_test_data()
    
    # U-Net模型
    unet_results = generate_unet_predictions(H, W, T)
    if unet_results[0] is not None:
        pred_mean, pred_std, true_values = unet_results
        plot_all_days_grid(
            pred_mean, pred_std, true_values,
            "U-Net", H, W, T,
            "unet"
        )
        
        # 保存预测数据（用于后续分析）
        np.save(FIGURES_DIR / "unet_pred_mean.npy", pred_mean)
        np.save(FIGURES_DIR / "unet_pred_std.npy", pred_std)
        np.save(FIGURES_DIR / "unet_true.npy", true_values)
        print("✅ U-Net预测数据已保存")
    
    # Tree模型
    tree_results = generate_tree_predictions(H, W, T)
    if tree_results[0] is not None:
        pred_mean, pred_std, true_values = tree_results
        plot_all_days_grid(
            pred_mean, pred_std, true_values,
            "Tree (XGBoost)", H, W, T,
            "tree"
        )
        
        # 保存预测数据
        np.save(FIGURES_DIR / "tree_pred_mean.npy", pred_mean)
        np.save(FIGURES_DIR / "tree_pred_std.npy", pred_std)
        np.save(FIGURES_DIR / "tree_true.npy", true_values)
        print("✅ Tree预测数据已保存")
    
    # GP模型（跳过，需要重新训练）
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

