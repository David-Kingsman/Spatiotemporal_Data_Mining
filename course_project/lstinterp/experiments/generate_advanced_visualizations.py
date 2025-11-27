"""
高级可视化生成

该脚本生成：
1. 时间序列动画（GIF）
2. 3D可视化（空间×时间）
3. 交互式图表（可选，使用plotly）
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor

# 输出目录
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "advanced"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 数据路径
DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_timeseries_animation(model_name="unet"):
    """生成时间序列动画（GIF）"""
    print(f"\n生成{model_name.upper()}时间序列动画...")
    
    # 加载预测数据
    pred_dir = OUTPUT_DIR / "figures" / "all_days"
    pred_mean_path = pred_dir / f"{model_name}_pred_mean.npy"
    true_path = pred_dir / f"{model_name}_true.npy"
    
    if not all(p.exists() for p in [pred_mean_path, true_path]):
        print(f"⚠️  {model_name}模型的预测数据不存在，跳过")
        return None
    
    pred_mean = np.load(pred_mean_path)
    true_values = np.load(true_path)
    H, W, T = pred_mean.shape
    
    # 创建动画
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 确定颜色范围
    vmin = np.nanmin([np.nanmin(pred_mean), np.nanmin(true_values)])
    vmax = np.nanmax([np.nanmax(pred_mean), np.nanmax(true_values)])
    
    # 初始化图像
    im1 = axes[0].imshow(pred_mean[:, :, 0], aspect='auto', origin='lower', 
                        cmap='jet_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Predicted Temperature', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude Index')
    axes[0].set_ylabel('Latitude Index')
    plt.colorbar(im1, ax=axes[0], label='Temperature (K)')
    
    im2 = axes[1].imshow(true_values[:, :, 0], aspect='auto', origin='lower', 
                        cmap='jet_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('True Temperature', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude Index')
    axes[1].set_ylabel('Latitude Index')
    plt.colorbar(im2, ax=axes[1], label='Temperature (K)')
    
    def animate(frame):
        day = frame + 1
        im1.set_array(pred_mean[:, :, frame])
        im2.set_array(true_values[:, :, frame])
        fig.suptitle(f'Day {day}/{T} - {model_name.upper()} Predictions', 
                    fontsize=16, fontweight='bold')
        return [im1, im2]
    
    # 创建动画（每帧间隔200ms）
    anim = FuncAnimation(fig, animate, frames=T, interval=200, blit=True, repeat=True)
    
    # 保存为GIF
    gif_path = FIGURES_DIR / f"{model_name}_timeseries_animation.gif"
    print(f"  保存动画到: {gif_path}")
    anim.save(str(gif_path), writer=PillowWriter(fps=5), dpi=100)
    plt.close()
    print(f"✅ 时间序列动画已保存: {gif_path}")
    
    return gif_path


def generate_3d_visualization(model_name="unet"):
    """生成3D可视化（空间×时间）"""
    print(f"\n生成{model_name.upper()} 3D可视化...")
    
    # 加载预测数据
    pred_dir = OUTPUT_DIR / "figures" / "all_days"
    pred_mean_path = pred_dir / f"{model_name}_pred_mean.npy"
    
    if not pred_mean_path.exists():
        print(f"⚠️  {model_name}模型的预测数据不存在，跳过")
        return None
    
    pred_mean = np.load(pred_mean_path)
    H, W, T = pred_mean.shape
    
    # 降采样以提高可视化速度（每5个点采样一次）
    step = 5
    lat_indices = np.arange(0, H, step)
    lon_indices = np.arange(0, W, step)
    time_indices = np.arange(0, T, 2)  # 每2天采样一次
    
    # 创建网格
    lat_grid, lon_grid, time_grid = np.meshgrid(
        lat_indices, lon_indices, time_indices, indexing='ij'
    )
    
    # 提取数据
    values = pred_mean[lat_grid, lon_grid, time_grid]
    
    # 创建3D图
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D散点图（颜色表示温度）
    scatter = ax.scatter(lat_grid.flatten(), lon_grid.flatten(), time_grid.flatten(),
                        c=values.flatten(), cmap='jet_r', s=1, alpha=0.6)
    
    ax.set_xlabel('Latitude Index', fontsize=12)
    ax.set_ylabel('Longitude Index', fontsize=12)
    ax.set_zlabel('Time (Day)', fontsize=12)
    ax.set_title(f'{model_name.upper()} - 3D Temperature Distribution (Spatial × Temporal)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.colorbar(scatter, ax=ax, label='Temperature (K)', shrink=0.8)
    
    # 保存
    fig_path = FIGURES_DIR / f"{model_name}_3d_visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 3D可视化已保存: {fig_path}")
    
    # 创建时间切片3D图（所有31天，8行×4列布局）
    print(f"  生成所有{T}天的3D表面图（8行×4列布局）...")
    n_rows = 8
    n_cols = 4
    fig = plt.figure(figsize=(20, 40))
    
    # 确定统一的颜色范围
    vmin = np.nanmin(pred_mean)
    vmax = np.nanmax(pred_mean)
    
    # 降采样空间维度以提高渲染速度（每5个点采样一次）
    step = 5
    lat_sampled = np.arange(0, H, step)
    lon_sampled = np.arange(0, W, step)
    
    for day in range(min(T, n_rows * n_cols)):
        row = day // n_cols
        col = day % n_cols
        
        ax = fig.add_subplot(n_rows, n_cols, day+1, projection='3d')
        
        # 创建降采样后的空间网格
        lat_2d, lon_2d = np.meshgrid(lat_sampled, lon_sampled, indexing='ij')
        values_2d = pred_mean[lat_2d, lon_2d, day]
        
        # 绘制3D表面
        surf = ax.plot_surface(lat_2d, lon_2d, values_2d, cmap='jet_r', 
                              vmin=vmin, vmax=vmax,
                              alpha=0.8, linewidth=0, antialiased=True)
        
        ax.set_xlabel('Latitude', fontsize=7)
        ax.set_ylabel('Longitude', fontsize=7)
        ax.set_zlabel('Temp (K)', fontsize=7)
        ax.set_title(f'Day {day+1}', fontsize=8, fontweight='bold', pad=5)
        ax.view_init(elev=30, azim=45)
        # 减小刻度字体大小
        ax.tick_params(labelsize=6)
    
    # 移除顶部标题，减少留白
    plt.tight_layout(pad=0.5)
    
    fig_path = FIGURES_DIR / f"{model_name}_3d_surfaces.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✅ 3D表面图（所有{T}天）已保存: {fig_path}")
    
    return fig_path


def generate_interactive_plot(model_name="unet"):
    """生成交互式图表（使用plotly，如果可用）"""
    print(f"\n生成{model_name.upper()}交互式图表...")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        print("⚠️  Plotly未安装，跳过交互式图表生成")
        print("   安装命令: pip install plotly")
        return None
    
    # 加载预测数据
    pred_dir = OUTPUT_DIR / "figures" / "all_days"
    pred_mean_path = pred_dir / f"{model_name}_pred_mean.npy"
    true_path = pred_dir / f"{model_name}_true.npy"
    
    if not all(p.exists() for p in [pred_mean_path, true_path]):
        print(f"⚠️  {model_name}模型的预测数据不存在，跳过")
        return None
    
    pred_mean = np.load(pred_mean_path)
    true_values = np.load(true_path)
    H, W, T = pred_mean.shape
    
    # 创建交互式热图（选择第15天）
    day_idx = 14
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Predicted Temperature', 'True Temperature'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    # 预测值热图
    fig.add_trace(
        go.Heatmap(
            z=pred_mean[:, :, day_idx],
            colorscale='Jet',
            colorbar=dict(title="Temperature (K)", x=0.45),
            name='Predicted'
        ),
        row=1, col=1
    )
    
    # 真实值热图
    fig.add_trace(
        go.Heatmap(
            z=true_values[:, :, day_idx],
            colorscale='Jet',
            colorbar=dict(title="Temperature (K)", x=1.02),
            name='True'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'{model_name.upper()} - Interactive Heatmaps (Day {day_idx+1})',
        height=500,
        width=1400
    )
    
    # 保存为HTML
    html_path = FIGURES_DIR / f"{model_name}_interactive_heatmap.html"
    fig.write_html(str(html_path))
    print(f"✅ 交互式热图已保存: {html_path}")
    
    # 创建时间序列交互图（选择中心点）
    center_lat, center_lon = H//2, W//2
    true_ts = true_values[center_lat, center_lon, :]
    pred_ts = pred_mean[center_lat, center_lon, :]
    
    mask = ~np.isnan(true_ts) & ~np.isnan(pred_ts)
    if mask.sum() > 0:
        valid_days = np.where(mask)[0] + 1
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=valid_days,
            y=true_ts[mask],
            mode='lines+markers',
            name='True',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=valid_days,
            y=pred_ts[mask],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'{model_name.upper()} - Interactive Time Series (Center Point)',
            xaxis_title='Day',
            yaxis_title='Temperature (K)',
            hovermode='x unified',
            height=500,
            width=1000
        )
        
        html_path = FIGURES_DIR / f"{model_name}_interactive_timeseries.html"
        fig.write_html(str(html_path))
        print(f"✅ 交互式时间序列已保存: {html_path}")
    
    return html_path


def main():
    """主函数"""
    print("=" * 80)
    print("  高级可视化生成")
    print("=" * 80)
    
    models = ["unet", "tree", "gp"]
    
    for model_name in models:
        print(f"\n处理{model_name.upper()}模型...")
        
        # 1. 时间序列动画
        generate_timeseries_animation(model_name)
        
        # 2. 3D可视化
        generate_3d_visualization(model_name)
        
        # 3. 交互式图表
        generate_interactive_plot(model_name)
    
    print("\n" + "=" * 80)
    print("  高级可视化生成完成")
    print("=" * 80)
    print(f"\n所有文件已保存到: {FIGURES_DIR}")


if __name__ == "__main__":
    main()


