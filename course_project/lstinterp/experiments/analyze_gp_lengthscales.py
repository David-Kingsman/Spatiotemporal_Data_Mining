"""
GP模型长度尺度分析

该脚本分析GP模型的空间和时间长度尺度学习结果，可视化长度尺度分布，
并解释长度尺度的物理意义。
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import gpytorch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor
from lstinterp.models import GPSTModel
from lstinterp.config import GPSTConfig

# 输出目录
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "gp_lengthscales"
RESULTS_DIR = OUTPUT_DIR / "results" / "gp_lengthscales"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径
MODEL_PATH = OUTPUT_DIR / "models" / "gp_model.pth"
GP_RESULTS_PATH = OUTPUT_DIR / "results" / "gp_results.json"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def print_section_header(title):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_gp_lengthscales():
    """分析GP模型的长度尺度"""
    print_section_header("GP模型长度尺度分析")
    
    if not MODEL_PATH.exists():
        print(f"⚠️  模型文件不存在: {MODEL_PATH}")
        print("请先运行 train_gp.py 训练模型")
        return
    
    # 加载模型配置
    if GP_RESULTS_PATH.exists():
        with open(GP_RESULTS_PATH, 'r') as f:
            results = json.load(f)
        config_dict = results.get('experiment_info', {}).get('model_config', {})
        print(f"\n模型配置: {config_dict}")
    else:
        print("⚠️  无法找到模型配置文件，使用默认配置")
        config_dict = {}
    
    # 创建配置
    config = GPSTConfig(
        kernel_space=config_dict.get('kernel_space', 'matern32'),
        kernel_time=config_dict.get('kernel_time', 'matern32'),
        num_inducing=config_dict.get('num_inducing', 500),
        lr=config_dict.get('lr', 0.01),
        num_epochs=config_dict.get('num_epochs', 50)
    )
    
    # 加载训练数据（用于获取数据范围）
    DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"
    train_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    H, W, T = train_tensor.shape
    
    print(f"\n数据维度: {H} × {W} × {T}")
    
    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    
    # 从checkpoint加载诱导点（如果存在）
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # 尝试从checkpoint获取诱导点
    if 'inducing_points' in checkpoint:
        inducing_points = checkpoint['inducing_points']
    else:
        # 如果没有保存，使用默认创建方法
        # 需要创建简单的诱导点网格
        n_space_grid = int(np.sqrt(config.num_inducing // 10))  # 假设10个时间点
        n_time = 10
        
        # 创建空间网格
        lat_coords = np.linspace(0, 1, n_space_grid)
        lon_coords = np.linspace(0, 1, n_space_grid)
        time_coords = np.linspace(0, 1, n_time)
        
        # 组合所有点
        inducing_list = []
        for t in time_coords:
            for lat in lat_coords:
                for lon in lon_coords:
                    inducing_list.append([lat, lon, t])
        
        inducing_points = torch.tensor(inducing_list, dtype=torch.float32)
    
    # 初始化模型
    model = GPSTModel(
        inducing_points=inducing_points,
        config=config,
        lengthscale_space=torch.tensor([0.5], dtype=torch.float32),
        lengthscale_time=torch.tensor([0.3], dtype=torch.float32),
        outputscale=torch.tensor([1.0], dtype=torch.float32),
        noise=torch.tensor([0.2], dtype=torch.float32)
    )
    
    # 加载模型权重（已在上面加载checkpoint）
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ 模型加载成功")
    
    # 提取长度尺度
    with torch.no_grad():
        # 获取空间核的长度尺度
        if hasattr(model.gp.covar_space, 'base_kernel'):
            base_space = model.gp.covar_space.base_kernel
        else:
            base_space = model.gp.covar_space
        
        if hasattr(base_space, 'lengthscale'):
            lengthscale_space = base_space.lengthscale.clone().cpu().numpy()
        print(f"\n空间长度尺度 (lengthscale_space):")
        print(f"  形状: {lengthscale_space.shape}")
        print(f"  值: {lengthscale_space}")
        
        if lengthscale_space.ndim > 0:
            if lengthscale_space.shape[-1] == 2:  # ARD: [lat, lon]
                print(f"  - 纬度方向: {float(lengthscale_space.flat[0]):.4f}")
                print(f"  - 经度方向: {float(lengthscale_space.flat[1]):.4f}")
            else:
                print(f"  - 统一长度尺度: {float(lengthscale_space.flat[0]):.4f}")
        
        # 获取时间核的长度尺度
        if hasattr(model.gp.covar_time, 'base_kernel'):
            base_time = model.gp.covar_time.base_kernel
        else:
            base_time = model.gp.covar_time
        
        if hasattr(base_time, 'lengthscale'):
            lengthscale_time = base_time.lengthscale.clone().cpu().numpy()
            print(f"\n时间长度尺度 (lengthscale_time):")
            print(f"  形状: {lengthscale_time.shape}")
            print(f"  值: {lengthscale_time}")
            print(f"  - 时间方向: {float(lengthscale_time.flat[0]):.4f}")
        
        # 获取输出尺度
        if hasattr(model.gp.covar_space, 'outputscale'):
            outputscale_space = model.gp.covar_space.outputscale.clone().cpu().item()
            print(f"\n空间输出尺度 (outputscale_space): {outputscale_space:.4f}")
        
        if hasattr(model.gp.covar_time, 'outputscale'):
            outputscale_time = model.gp.covar_time.outputscale.clone().cpu().item()
            print(f"\n时间输出尺度 (outputscale_time): {outputscale_time:.4f}")
        
        # 获取噪声参数
        if hasattr(model.likelihood, 'noise'):
            noise = model.likelihood.noise.clone().cpu().item()
            print(f"\n观测噪声 (noise): {noise:.4f}")
    
    # 计算物理意义
    print("\n" + "-" * 80)
    print("  长度尺度的物理意义")
    print("-" * 80)
    
    # 数据归一化到[0,1]，所以长度尺度也在[0,1]范围内
    # 空间范围：H=100, W=200，经纬度范围约为5度×10度
    # 时间范围：T=31天
    
    if lengthscale_space.ndim > 0 and lengthscale_space.shape[-1] >= 2:
        ls_lat = float(lengthscale_space.flat[0])
        ls_lon = float(lengthscale_space.flat[1])
        
        # 归一化空间的物理范围（假设经纬度范围：lat 35-40, lon -115--105）
        lat_range = 5.0  # 度
        lon_range = 10.0  # 度
        
        physical_ls_lat = ls_lat * lat_range  # 度
        physical_ls_lon = ls_lon * lon_range  # 度
        
        print(f"\n空间相关性半径（物理单位）:")
        print(f"  - 纬度方向: {physical_ls_lat:.2f} 度 ≈ {physical_ls_lat * 111:.0f} 公里")
        print(f"  - 经度方向: {physical_ls_lon:.2f} 度 ≈ {physical_ls_lon * 111 * np.cos(np.radians(37.5)):.0f} 公里")
        
        print(f"\n解释:")
        print(f"  - 纬度方向: 相距 {physical_ls_lat:.2f} 度内的温度高度相关")
        print(f"  - 经度方向: 相距 {physical_ls_lon:.2f} 度内的温度高度相关")
    else:
        ls_space = float(lengthscale_space.flat[0]) if lengthscale_space.ndim > 0 else float(lengthscale_space)
        
        # 假设空间是归一化的
        spatial_scale = np.sqrt(H**2 + W**2)  # 归一化空间的对角线长度
        physical_ls_space = ls_space * spatial_scale
        
        print(f"\n空间相关性半径:")
        print(f"  - 归一化单位: {ls_space:.4f}")
        print(f"  - 解释: 在归一化空间中相距 {ls_space:.4f} 的点高度相关")
    
    if lengthscale_time.ndim > 0:
        ls_time = float(lengthscale_time.flat[0])
    else:
        ls_time = float(lengthscale_time)
    
    # 时间归一化到[0,1]，T=31天
    physical_ls_time = ls_time * 31  # 天
    
    print(f"\n时间相关性半径（物理单位）:")
    print(f"  - 归一化单位: {ls_time:.4f}")
    print(f"  - 物理单位: {physical_ls_time:.2f} 天")
    print(f"  - 解释: 相距 {physical_ls_time:.2f} 天的温度高度相关")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 长度尺度对比
    if lengthscale_space.ndim > 0 and lengthscale_space.shape[-1] >= 2:
        labels = ['Latitude', 'Longitude']
        values = [float(lengthscale_space.flat[0]), float(lengthscale_space.flat[1])]
        axes[0, 0].bar(labels, values, color=['blue', 'green'], alpha=0.7)
        axes[0, 0].set_title('Spatial Lengthscales (ARD)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Lengthscale (normalized)')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 物理单位
        if 'physical_ls_lon' in locals():
            ax2 = axes[0, 0].twinx()
            physical_values = [physical_ls_lon, physical_ls_lat]
            ax2.set_ylabel('Lengthscale (degrees)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            # 不绘制，只设置刻度
    else:
        ls_space_val = float(lengthscale_space.flat[0]) if lengthscale_space.ndim > 0 else float(lengthscale_space)
        axes[0, 0].bar(['Spatial'], [ls_space_val], color='blue', alpha=0.7)
        axes[0, 0].set_title('Spatial Lengthscale', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Lengthscale (normalized)')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 时间长度尺度
    axes[0, 1].bar(['Time'], [ls_time], color='orange', alpha=0.7)
    axes[0, 1].set_title('Temporal Lengthscale', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Lengthscale (normalized)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加物理单位
    ax2 = axes[0, 1].twinx()
    ax2.set_ylabel(f'Lengthscale ({physical_ls_time:.2f} days)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 输出尺度
    outputscales = []
    labels = []
    if 'outputscale_space' in locals():
        outputscales.append(outputscale_space)
        labels.append('Spatial')
    if 'outputscale_time' in locals():
        outputscales.append(outputscale_time)
        labels.append('Time')
    
    if len(outputscales) > 0:
        axes[1, 0].bar(labels, outputscales, color=['blue', 'orange'], alpha=0.7)
        axes[1, 0].set_title('Outputscales', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Outputscale')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 噪声参数
    if 'noise' in locals():
        axes[1, 1].bar(['Noise'], [noise], color='red', alpha=0.7)
        axes[1, 1].set_title('Observation Noise', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Noise')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "gp_lengthscales_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 可视化已保存: {fig_path}")
    
    # 保存结果
    results = {
        'lengthscale_space': lengthscale_space.tolist() if isinstance(lengthscale_space, np.ndarray) else float(lengthscale_space),
        'lengthscale_time': lengthscale_time.tolist() if isinstance(lengthscale_time, np.ndarray) else float(lengthscale_time),
        'physical_lengthscale_space_lat_deg': float(physical_ls_lat) if 'physical_ls_lat' in locals() else None,
        'physical_lengthscale_space_lon_deg': float(physical_ls_lon) if 'physical_ls_lon' in locals() else None,
        'physical_lengthscale_time_days': float(physical_ls_time),
        'outputscale_space': float(outputscale_space) if 'outputscale_space' in locals() else None,
        'outputscale_time': float(outputscale_time) if 'outputscale_time' in locals() else None,
        'noise': float(noise) if 'noise' in locals() else None,
        'kernel_space': config.kernel_space,
        'kernel_time': config.kernel_time
    }
    
    json_path = RESULTS_DIR / "gp_lengthscales_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 结果已保存: {json_path}")
    
    return results


def main():
    """主函数"""
    print("=" * 80)
    print("  GP模型长度尺度分析")
    print("=" * 80)
    
    try:
        results = analyze_gp_lengthscales()
        
        print("\n" + "=" * 80)
        print("  长度尺度分析完成")
        print("=" * 80)
        print(f"\n所有结果已保存到:")
        print(f"  - 图表: {FIGURES_DIR}")
        print(f"  - 数据: {RESULTS_DIR}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

