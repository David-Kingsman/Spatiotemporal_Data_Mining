"""
按缺失率分析误差

该脚本分析不同缺失率下的预测性能，将测试集按缺失率分组（高/中/低缺失率），
分析不同缺失率下的预测性能，并可视化缺失率与误差的关系。
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor
from lstinterp.metrics import rmse, mae, r2

# 输出目录
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "missing_rate_analysis"
RESULTS_DIR = OUTPUT_DIR / "results" / "missing_rate_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 数据路径
DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def print_section_header(title):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def compute_missing_rate_per_pixel(tensor):
    """计算每个像素的缺失率（31天中有多少天缺失）"""
    H, W, T = tensor.shape
    missing_rate = np.zeros((H, W))
    
    for i in range(H):
        for j in range(W):
            missing_count = (tensor[i, j, :] == 0).sum()
            missing_rate[i, j] = missing_count / T
    
    return missing_rate


def analyze_by_missing_rate(pred_mean, true_values, missing_rate, model_name, H, W, T):
    """按缺失率分析误差"""
    print_section_header(f"{model_name.upper()} - 按缺失率分析误差")
    
    # 计算误差
    errors = true_values - pred_mean
    
    # 将缺失率分为三个组：低（0-33%）、中（33-67%）、高（67-100%）
    # 但首先需要找到每个预测点的缺失率
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    
    # 创建缺失率映射（按像素）
    pixel_missing_rate = compute_missing_rate_per_pixel(test_tensor)
    
    # 提取有效预测点的缺失率
    mask = ~np.isnan(true_values) & ~np.isnan(pred_mean)
    
    # 获取有效点的坐标和值
    lat_indices, lon_indices, time_indices = np.where(mask)
    
    if len(lat_indices) == 0:
        print("⚠️  没有有效的预测点，跳过分析")
        return None
    
    # 获取每个点的缺失率
    point_missing_rates = pixel_missing_rate[lat_indices, lon_indices]
    point_errors = errors[mask]
    point_true = true_values[mask]
    point_pred = pred_mean[mask]
    
    # 按缺失率分组
    low_mask = point_missing_rates < 0.33
    mid_mask = (point_missing_rates >= 0.33) & (point_missing_rates < 0.67)
    high_mask = point_missing_rates >= 0.67
    
    # 统计各组
    groups_stats = {
        'Missing_Rate_Range': ['Low (0-33%)', 'Medium (33-67%)', 'High (67-100%)'],
        'Mean_Missing_Rate': [
            point_missing_rates[low_mask].mean() if low_mask.sum() > 0 else np.nan,
            point_missing_rates[mid_mask].mean() if mid_mask.sum() > 0 else np.nan,
            point_missing_rates[high_mask].mean() if high_mask.sum() > 0 else np.nan
        ],
        'RMSE': [
            np.sqrt(np.mean(point_errors[low_mask]**2)) if low_mask.sum() > 0 else np.nan,
            np.sqrt(np.mean(point_errors[mid_mask]**2)) if mid_mask.sum() > 0 else np.nan,
            np.sqrt(np.mean(point_errors[high_mask]**2)) if high_mask.sum() > 0 else np.nan
        ],
        'MAE': [
            np.mean(np.abs(point_errors[low_mask])) if low_mask.sum() > 0 else np.nan,
            np.mean(np.abs(point_errors[mid_mask])) if mid_mask.sum() > 0 else np.nan,
            np.mean(np.abs(point_errors[high_mask])) if high_mask.sum() > 0 else np.nan
        ],
        'R2': [
            1 - np.sum(point_errors[low_mask]**2) / np.sum((point_true[low_mask] - point_true[low_mask].mean())**2) if low_mask.sum() > 0 else np.nan,
            1 - np.sum(point_errors[mid_mask]**2) / np.sum((point_true[mid_mask] - point_true[mid_mask].mean())**2) if mid_mask.sum() > 0 else np.nan,
            1 - np.sum(point_errors[high_mask]**2) / np.sum((point_true[high_mask] - point_true[high_mask].mean())**2) if high_mask.sum() > 0 else np.nan
        ],
        'Mean_Error': [
            point_errors[low_mask].mean() if low_mask.sum() > 0 else np.nan,
            point_errors[mid_mask].mean() if mid_mask.sum() > 0 else np.nan,
            point_errors[high_mask].mean() if high_mask.sum() > 0 else np.nan
        ],
        'Num_Samples': [
            low_mask.sum(),
            mid_mask.sum(),
            high_mask.sum()
        ]
    }
    
    df = pd.DataFrame(groups_stats)
    
    print("\n按缺失率分组的性能统计:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 保存统计表
    csv_path = RESULTS_DIR / f"{model_name}_missing_rate_analysis.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ 统计表已保存: {csv_path}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE vs 缺失率
    axes[0, 0].bar(df['Missing_Rate_Range'], df['RMSE'], 
                   color=['green', 'orange', 'red'], alpha=0.7)
    axes[0, 0].set_title('RMSE by Missing Rate Range', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('RMSE (K)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # R² vs 缺失率
    axes[0, 1].bar(df['Missing_Rate_Range'], df['R2'], 
                   color=['green', 'orange', 'red'], alpha=0.7)
    axes[0, 1].set_title('R² by Missing Rate Range', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 缺失率 vs 误差散点图
    # 采样以加快可视化
    sample_size = min(10000, len(point_missing_rates))
    sample_indices = np.random.choice(len(point_missing_rates), sample_size, replace=False)
    
    axes[1, 0].scatter(point_missing_rates[sample_indices], 
                      np.abs(point_errors[sample_indices]), 
                      alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Missing Rate')
    axes[1, 0].set_ylabel('Absolute Error (K)')
    axes[1, 0].set_title('Missing Rate vs Absolute Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(point_missing_rates) > 10:
        from scipy import stats
        z = np.polyfit(point_missing_rates[sample_indices], 
                      np.abs(point_errors[sample_indices]), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(point_missing_rates.min(), point_missing_rates.max(), 100)
        axes[1, 0].plot(x_trend, p(x_trend), "r--", linewidth=2, 
                       label=f'Trend (slope={z[0]:.2f})')
        axes[1, 0].legend()
    
    # 误差分布（按缺失率组）
    axes[1, 1].hist(np.abs(point_errors[low_mask]), bins=50, alpha=0.5, 
                   label='Low Missing Rate', color='green', density=True)
    axes[1, 1].hist(np.abs(point_errors[mid_mask]), bins=50, alpha=0.5, 
                   label='Medium Missing Rate', color='orange', density=True)
    axes[1, 1].hist(np.abs(point_errors[high_mask]), bins=50, alpha=0.5, 
                   label='High Missing Rate', color='red', density=True)
    axes[1, 1].set_xlabel('Absolute Error (K)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Error Distribution by Missing Rate Range', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"{model_name}_missing_rate_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化已保存: {fig_path}")
    
    return df


def analyze_missing_rate_pattern(test_tensor):
    """分析缺失率的空间模式"""
    print_section_header("缺失率空间模式分析")
    
    H, W, T = test_tensor.shape
    
    # 计算每个像素的缺失率
    pixel_missing_rate = compute_missing_rate_per_pixel(test_tensor)
    
    # 统计
    print(f"\n缺失率统计:")
    print(f"  均值: {pixel_missing_rate.mean():.4f} ({pixel_missing_rate.mean()*100:.2f}%)")
    print(f"  标准差: {pixel_missing_rate.std():.4f}")
    print(f"  最小值: {pixel_missing_rate.min():.4f} ({pixel_missing_rate.min()*100:.2f}%)")
    print(f"  最大值: {pixel_missing_rate.max():.4f} ({pixel_missing_rate.max()*100:.2f}%)")
    
    # 可视化空间模式
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 缺失率热图
    im1 = axes[0].imshow(pixel_missing_rate, aspect='auto', origin='lower', 
                        cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title('Missing Rate Spatial Pattern', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude Index')
    axes[0].set_ylabel('Latitude Index')
    plt.colorbar(im1, ax=axes[0], label='Missing Rate')
    
    # 缺失率分布直方图
    axes[1].hist(pixel_missing_rate.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_title('Missing Rate Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Missing Rate')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(pixel_missing_rate.mean(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {pixel_missing_rate.mean():.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "missing_rate_spatial_pattern.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 缺失率空间模式图已保存: {fig_path}")
    
    # 保存统计
    stats = {
        'mean': float(pixel_missing_rate.mean()),
        'std': float(pixel_missing_rate.std()),
        'min': float(pixel_missing_rate.min()),
        'max': float(pixel_missing_rate.max()),
        'median': float(np.median(pixel_missing_rate)),
        'q25': float(np.percentile(pixel_missing_rate, 25)),
        'q75': float(np.percentile(pixel_missing_rate, 75))
    }
    
    json_path = RESULTS_DIR / "missing_rate_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ 缺失率统计已保存: {json_path}")
    
    return pixel_missing_rate, stats


def main():
    """主函数"""
    print("=" * 80)
    print("  按缺失率分析误差")
    print("=" * 80)
    
    # 加载测试数据
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    H, W, T = test_tensor.shape
    print(f"\n测试数据维度: {H} × {W} × {T}")
    
    # 分析缺失率空间模式
    pixel_missing_rate, missing_stats = analyze_missing_rate_pattern(test_tensor)
    
    # 分析U-Net模型
    pred_dir = OUTPUT_DIR / "figures" / "all_days"
    unet_pred_path = pred_dir / "unet_pred_mean.npy"
    unet_true_path = pred_dir / "unet_true.npy"
    
    if unet_pred_path.exists() and unet_true_path.exists():
        pred_mean = np.load(unet_pred_path)
        true_values = np.load(unet_true_path)
        print(f"\n加载U-Net预测数据: {pred_mean.shape}")
        
        unet_df = analyze_by_missing_rate(
            pred_mean, true_values, pixel_missing_rate, "unet", H, W, T
        )
    
    # 分析Tree模型
    tree_pred_path = pred_dir / "tree_pred_mean.npy"
    tree_true_path = pred_dir / "tree_true.npy"
    
    if tree_pred_path.exists() and tree_true_path.exists():
        pred_mean = np.load(tree_pred_path)
        true_values = np.load(tree_true_path)
        print(f"\n加载Tree预测数据: {pred_mean.shape}")
        
        tree_df = analyze_by_missing_rate(
            pred_mean, true_values, pixel_missing_rate, "tree", H, W, T
        )
    
    print("\n" + "=" * 80)
    print("  缺失率分析完成")
    print("=" * 80)
    print(f"\n所有结果已保存到:")
    print(f"  - 图表: {FIGURES_DIR}")
    print(f"  - 数据: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

