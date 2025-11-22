"""
结果深度分析

该脚本对模型预测结果进行深度分析：
1. 误差空间分布图（按区域分析误差）
2. 时间序列预测对比（选定区域的时间序列）
3. 极端值分析（高温/低温区域的预测表现）
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.metrics import rmse, mae, r2

# 输出目录
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "deep_analysis"
RESULTS_DIR = OUTPUT_DIR / "results" / "deep_analysis"
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


def load_predictions(model_name="unet"):
    """加载模型预测结果"""
    pred_dir = OUTPUT_DIR / "figures" / "all_days"
    
    pred_mean_path = pred_dir / f"{model_name}_pred_mean.npy"
    pred_std_path = pred_dir / f"{model_name}_pred_std.npy"
    true_path = pred_dir / f"{model_name}_true.npy"
    
    if not all(p.exists() for p in [pred_mean_path, pred_std_path, true_path]):
        print(f"⚠️  {model_name}模型的预测数据不存在，跳过")
        return None
    
    pred_mean = np.load(pred_mean_path)
    pred_std = np.load(pred_std_path)
    true_values = np.load(true_path)
    
    return pred_mean, pred_std, true_values


def analyze_spatial_error_distribution(pred_mean, true_values, model_name, H, W, T):
    """分析误差的空间分布"""
    print_section_header(f"{model_name.upper()} - 误差空间分布分析")
    
    # 计算误差
    errors = true_values - pred_mean
    
    # 按区域分析（将空间分为9个区域：3×3）
    n_regions_lat = 3
    n_regions_lon = 3
    
    lat_size_per_region = H // n_regions_lat
    lon_size_per_region = W // n_regions_lon
    
    region_stats = {
        'Region': [],
        'Lat_Range': [],
        'Lon_Range': [],
        'RMSE': [],
        'MAE': [],
        'R2': [],
        'Mean_Error': [],
        'Std_Error': [],
        'Num_Pixels': []
    }
    
    fig, axes = plt.subplots(n_regions_lat, n_regions_lon, figsize=(18, 15))
    
    for i in range(n_regions_lat):
        for j in range(n_regions_lon):
            lat_start = i * lat_size_per_region
            lat_end = (i + 1) * lat_size_per_region if i < n_regions_lat - 1 else H
            lon_start = j * lon_size_per_region
            lon_end = (j + 1) * lon_size_per_region if j < n_regions_lon - 1 else W
            
            # 提取区域数据
            region_true = true_values[lat_start:lat_end, lon_start:lon_end, :]
            region_pred = pred_mean[lat_start:lat_end, lon_start:lon_end, :]
            region_errors = errors[lat_start:lat_end, lon_start:lon_end, :]
            
            # 只考虑有效值（非NaN）
            mask = ~np.isnan(region_true) & ~np.isnan(region_pred)
            if mask.sum() > 0:
                region_true_flat = region_true[mask]
                region_pred_flat = region_pred[mask]
                region_errors_flat = region_errors[mask]
                
                # 计算统计量
                region_rmse = np.sqrt(np.mean(region_errors_flat**2))
                region_mae = np.mean(np.abs(region_errors_flat))
                region_r2 = 1 - np.sum(region_errors_flat**2) / np.sum((region_true_flat - region_true_flat.mean())**2)
                region_mean_error = region_errors_flat.mean()
                region_std_error = region_errors_flat.std()
                
                region_stats['Region'].append(f"R{i*n_regions_lon+j+1}")
                region_stats['Lat_Range'].append(f"[{lat_start}, {lat_end})")
                region_stats['Lon_Range'].append(f"[{lon_start}, {lon_end})")
                region_stats['RMSE'].append(region_rmse)
                region_stats['MAE'].append(region_mae)
                region_stats['R2'].append(region_r2)
                region_stats['Mean_Error'].append(region_mean_error)
                region_stats['Std_Error'].append(region_std_error)
                region_stats['Num_Pixels'].append(mask.sum())
                
                # 绘制该区域的误差分布
                ax = axes[i, j]
                region_error_2d = np.nanmean(np.abs(region_errors), axis=2)  # 平均绝对误差
                im = ax.imshow(region_error_2d, aspect='auto', origin='lower', 
                              cmap='Reds', vmin=0, vmax=np.nanmax(np.abs(errors)))
                ax.set_title(f'Region {i*n_regions_lon+j+1}\nRMSE={region_rmse:.2f}K, R²={region_r2:.3f}', 
                           fontsize=10, fontweight='bold')
                ax.set_xlabel('Lon Index')
                ax.set_ylabel('Lat Index')
                plt.colorbar(im, ax=ax, label='Abs Error (K)')
            else:
                ax = axes[i, j]
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Region {i*n_regions_lon+j+1}', fontsize=10)
    
    plt.suptitle(f'{model_name.upper()} - Spatial Error Distribution (3×3 Regions)', 
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"{model_name}_spatial_error_regions.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 空间误差分布图已保存: {fig_path}")
    
    # 保存统计表
    df = pd.DataFrame(region_stats)
    csv_path = RESULTS_DIR / f"{model_name}_spatial_error_regions.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ 区域误差统计已保存: {csv_path}")
    
    # 打印摘要
    print("\n区域误差统计摘要:")
    print(df[['Region', 'RMSE', 'MAE', 'R2', 'Mean_Error']].to_string(index=False, float_format='%.3f'))
    
    return df


def analyze_timeseries_comparison(pred_mean, pred_std, true_values, model_name, H, W, T):
    """分析选定区域的时间序列预测对比"""
    print_section_header(f"{model_name.upper()} - 时间序列预测对比")
    
    # 选择几个代表性的空间位置
    locations = [
        (H//4, W//4, "Northwest"),
        (H//4, 3*W//4, "Northeast"),
        (3*H//4, W//4, "Southwest"),
        (3*H//4, 3*W//4, "Southeast"),
        (H//2, W//2, "Center")
    ]
    
    fig, axes = plt.subplots(len(locations), 1, figsize=(14, 3*len(locations)))
    if len(locations) == 1:
        axes = [axes]
    
    location_stats = {
        'Location': [],
        'RMSE': [],
        'MAE': [],
        'R2': [],
        'Mean_Error': []
    }
    
    for idx, (lat_idx, lon_idx, name) in enumerate(locations):
        # 提取该位置的时间序列
        true_ts = true_values[lat_idx, lon_idx, :]
        pred_ts = pred_mean[lat_idx, lon_idx, :]
        pred_std_ts = pred_std[lat_idx, lon_idx, :]
        
        # 只考虑有效值
        mask = ~np.isnan(true_ts) & ~np.isnan(pred_ts)
        if mask.sum() > 5:
            valid_days = np.where(mask)[0] + 1
            true_ts_valid = true_ts[mask]
            pred_ts_valid = pred_ts[mask]
            pred_std_ts_valid = pred_std_ts[mask]
            
            # 计算统计量
            errors = true_ts_valid - pred_ts_valid
            ts_rmse = np.sqrt(np.mean(errors**2))
            ts_mae = np.mean(np.abs(errors))
            ts_r2 = 1 - np.sum(errors**2) / np.sum((true_ts_valid - true_ts_valid.mean())**2)
            ts_mean_error = errors.mean()
            
            location_stats['Location'].append(name)
            location_stats['RMSE'].append(ts_rmse)
            location_stats['MAE'].append(ts_mae)
            location_stats['R2'].append(ts_r2)
            location_stats['Mean_Error'].append(ts_mean_error)
            
            # 绘制时间序列
            ax = axes[idx]
            ax.plot(valid_days, true_ts_valid, 'o-', label='True', linewidth=2, 
                   markersize=5, color='blue', alpha=0.7)
            ax.plot(valid_days, pred_ts_valid, 's-', label='Predicted', linewidth=2, 
                   markersize=4, color='red', alpha=0.7)
            
            # 绘制不确定性区间（90%）
            upper_bound = pred_ts_valid + 1.645 * pred_std_ts_valid
            lower_bound = pred_ts_valid - 1.645 * pred_std_ts_valid
            ax.fill_between(valid_days, lower_bound, upper_bound, alpha=0.2, 
                           color='red', label='90% Prediction Interval')
            
            ax.set_title(f'{name} (Lat={lat_idx}, Lon={lon_idx}) - RMSE={ts_rmse:.2f}K, R²={ts_r2:.3f}', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Day')
            ax.set_ylabel('Temperature (K)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax = axes[idx]
            ax.text(0.5, 0.5, f'Insufficient Data\n({name})', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(name, fontsize=11)
    
    plt.suptitle(f'{model_name.upper()} - Time Series Predictions at Different Locations', 
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"{model_name}_timeseries_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 时间序列对比图已保存: {fig_path}")
    
    # 保存统计表
    df = pd.DataFrame(location_stats)
    csv_path = RESULTS_DIR / f"{model_name}_timeseries_stats.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ 时间序列统计已保存: {csv_path}")
    
    return df


def analyze_extreme_values(pred_mean, pred_std, true_values, model_name, H, W, T):
    """分析极端值（高温/低温）的预测表现"""
    print_section_header(f"{model_name.upper()} - 极端值分析")
    
    # 提取所有有效值
    mask = ~np.isnan(true_values) & ~np.isnan(pred_mean)
    true_flat = true_values[mask]
    pred_flat = pred_mean[mask]
    pred_std_flat = pred_std[mask]
    errors_flat = true_flat - pred_flat
    
    # 定义极端值（高温和低温）
    temp_threshold_low = np.percentile(true_flat, 10)  # 最低10%
    temp_threshold_high = np.percentile(true_flat, 90)  # 最高10%
    
    low_temp_mask = true_flat <= temp_threshold_low
    high_temp_mask = true_flat >= temp_threshold_high
    normal_temp_mask = ~low_temp_mask & ~high_temp_mask
    
    # 统计各温度段的性能
    extreme_stats = {
        'Temperature_Range': ['Low (Bottom 10%)', 'Normal (10%-90%)', 'High (Top 10%)'],
        'Temperature_Mean(K)': [
            true_flat[low_temp_mask].mean() if low_temp_mask.sum() > 0 else np.nan,
            true_flat[normal_temp_mask].mean() if normal_temp_mask.sum() > 0 else np.nan,
            true_flat[high_temp_mask].mean() if high_temp_mask.sum() > 0 else np.nan
        ],
        'RMSE': [
            np.sqrt(np.mean(errors_flat[low_temp_mask]**2)) if low_temp_mask.sum() > 0 else np.nan,
            np.sqrt(np.mean(errors_flat[normal_temp_mask]**2)) if normal_temp_mask.sum() > 0 else np.nan,
            np.sqrt(np.mean(errors_flat[high_temp_mask]**2)) if high_temp_mask.sum() > 0 else np.nan
        ],
        'MAE': [
            np.mean(np.abs(errors_flat[low_temp_mask])) if low_temp_mask.sum() > 0 else np.nan,
            np.mean(np.abs(errors_flat[normal_temp_mask])) if normal_temp_mask.sum() > 0 else np.nan,
            np.mean(np.abs(errors_flat[high_temp_mask])) if high_temp_mask.sum() > 0 else np.nan
        ],
        'Mean_Error': [
            errors_flat[low_temp_mask].mean() if low_temp_mask.sum() > 0 else np.nan,
            errors_flat[normal_temp_mask].mean() if normal_temp_mask.sum() > 0 else np.nan,
            errors_flat[high_temp_mask].mean() if high_temp_mask.sum() > 0 else np.nan
        ],
        'Num_Samples': [
            low_temp_mask.sum(),
            normal_temp_mask.sum(),
            high_temp_mask.sum()
        ]
    }
    
    df = pd.DataFrame(extreme_stats)
    print("\n极端值预测性能:")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 误差分布（按温度段）
    axes[0, 0].hist(errors_flat[low_temp_mask], bins=30, alpha=0.5, label='Low Temp', color='blue')
    axes[0, 0].hist(errors_flat[normal_temp_mask], bins=30, alpha=0.5, label='Normal Temp', color='green')
    axes[0, 0].hist(errors_flat[high_temp_mask], bins=30, alpha=0.5, label='High Temp', color='red')
    axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Error Distribution by Temperature Range', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Error (K)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 预测值 vs 真实值（按温度段）
    if low_temp_mask.sum() > 0:
        axes[0, 1].scatter(true_flat[low_temp_mask], pred_flat[low_temp_mask], 
                          alpha=0.3, label='Low Temp', s=10, color='blue')
    if normal_temp_mask.sum() > 0:
        axes[0, 1].scatter(true_flat[normal_temp_mask], pred_flat[normal_temp_mask], 
                          alpha=0.3, label='Normal Temp', s=10, color='green')
    if high_temp_mask.sum() > 0:
        axes[0, 1].scatter(true_flat[high_temp_mask], pred_flat[high_temp_mask], 
                          alpha=0.3, label='High Temp', s=10, color='red')
    
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='y=x')
    axes[0, 1].set_title('Predicted vs True by Temperature Range', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('True Temperature (K)')
    axes[0, 1].set_ylabel('Predicted Temperature (K)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE对比
    axes[1, 0].bar(df['Temperature_Range'], df['RMSE'], color=['blue', 'green', 'red'], alpha=0.7)
    axes[1, 0].set_title('RMSE by Temperature Range', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('RMSE (K)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 平均误差对比
    axes[1, 1].bar(df['Temperature_Range'], df['Mean_Error'], color=['blue', 'green', 'red'], alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Mean Error by Temperature Range', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Mean Error (K)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"{model_name}_extreme_values.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 极端值分析图已保存: {fig_path}")
    
    # 保存统计表
    csv_path = RESULTS_DIR / f"{model_name}_extreme_values.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ 极端值统计已保存: {csv_path}")
    
    return df


def main():
    """主函数"""
    print("=" * 80)
    print("  结果深度分析")
    print("=" * 80)
    
    # 加载测试数据获取维度
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    H, W, T = test_tensor.shape
    print(f"\n数据维度: {H} × {W} × {T}")
    
    # 分析U-Net模型
    unet_results = load_predictions("unet")
    if unet_results is not None:
        pred_mean, pred_std, true_values = unet_results
        print(f"\n加载U-Net预测数据: {pred_mean.shape}")
        
        # 1. 误差空间分布
        spatial_df = analyze_spatial_error_distribution(pred_mean, true_values, "unet", H, W, T)
        
        # 2. 时间序列对比
        timeseries_df = analyze_timeseries_comparison(pred_mean, pred_std, true_values, "unet", H, W, T)
        
        # 3. 极端值分析
        extreme_df = analyze_extreme_values(pred_mean, pred_std, true_values, "unet", H, W, T)
    
    # 分析Tree模型
    tree_results = load_predictions("tree")
    if tree_results is not None:
        pred_mean, pred_std, true_values = tree_results
        print(f"\n加载Tree预测数据: {pred_mean.shape}")
        
        # 1. 误差空间分布
        spatial_df = analyze_spatial_error_distribution(pred_mean, true_values, "tree", H, W, T)
        
        # 2. 时间序列对比
        timeseries_df = analyze_timeseries_comparison(pred_mean, pred_std, true_values, "tree", H, W, T)
        
        # 3. 极端值分析
        extreme_df = analyze_extreme_values(pred_mean, pred_std, true_values, "tree", H, W, T)
    
    print("\n" + "=" * 80)
    print("  深度分析完成")
    print("=" * 80)
    print(f"\n所有结果已保存到:")
    print(f"  - 图表: {FIGURES_DIR}")
    print(f"  - 数据: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

