"""从已保存的npy文件生成all_days图片（使用统一颜色范围）"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# 路径设置
FIGURES_DIR = Path(__file__).parent.parent.parent / "output" / "figures" / "all_days"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_all_days_from_npy(model_name, prefix, vmin_mean, vmax_mean, vmin_std, vmax_std, vmax_error):
    """从npy文件生成all_days图片"""
    print(f"\n生成{model_name}所有天数可视化...")
    
    # 加载数据
    pred_mean = np.load(FIGURES_DIR / f"{prefix}_pred_mean.npy")
    pred_std = np.load(FIGURES_DIR / f"{prefix}_pred_std.npy")
    true_values = np.load(FIGURES_DIR / f"{prefix}_true.npy")
    
    H, W, T = pred_mean.shape
    n_rows = 8
    n_cols = 4
    
    print(f"  数据形状: {pred_mean.shape}")
    print(f"  使用统一颜色范围:")
    print(f"    Mean: [{vmin_mean:.2f}, {vmax_mean:.2f}] K")
    print(f"    Std:  [{vmin_std:.2f}, {vmax_std:.2f}] K")
    print(f"    Error: [-{vmax_error:.2f}, {vmax_error:.2f}] K")
    
    errors = true_values - pred_mean
    
    # 创建图片
    fig_mean = plt.figure(figsize=(20, 40))
    fig_std = plt.figure(figsize=(20, 40))
    fig_error = plt.figure(figsize=(20, 40))
    
    gs_mean = GridSpec(n_rows, n_cols, figure=fig_mean, hspace=0.1, wspace=0.1, 
                       left=0.02, right=0.98, top=0.98, bottom=0.02)
    gs_std = GridSpec(n_rows, n_cols, figure=fig_std, hspace=0.1, wspace=0.1,
                      left=0.02, right=0.98, top=0.98, bottom=0.02)
    gs_error = GridSpec(n_rows, n_cols, figure=fig_error, hspace=0.1, wspace=0.1,
                        left=0.02, right=0.98, top=0.98, bottom=0.02)
    
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
        ax_mean.set_title(f'Day {t+1}', fontsize=8, pad=2)
        ax_mean.set_xlabel('Longitude Index', fontsize=7)
        ax_mean.set_ylabel('Latitude Index', fontsize=7)
        # 显式设置colorbar范围，确保刻度一致
        cbar_mean = fig_mean.colorbar(im_mean, ax=ax_mean, label='Temperature (K)', shrink=0.8, aspect=20)
        cbar_mean.mappable.set_clim(vmin=vmin_mean, vmax=vmax_mean)
        # 设置colorbar刻度范围
        cbar_mean.set_ticks([vmin_mean, (vmin_mean+vmax_mean)/2, vmax_mean])
        
        # 预测不确定性图
        ax_std = fig_std.add_subplot(gs_std[row, col])
        im_std = ax_std.imshow(
            pred_std[:, :, t],
            aspect='auto', origin='lower',
            cmap='viridis', vmin=vmin_std, vmax=vmax_std
        )
        ax_std.set_title(f'Day {t+1}', fontsize=8, pad=2)
        ax_std.set_xlabel('Longitude Index', fontsize=7)
        ax_std.set_ylabel('Latitude Index', fontsize=7)
        # 显式设置colorbar范围，确保刻度一致
        cbar_std = fig_std.colorbar(im_std, ax=ax_std, label='Std Dev (K)', shrink=0.8, aspect=20)
        cbar_std.mappable.set_clim(vmin=vmin_std, vmax=vmax_std)
        # 设置colorbar刻度范围
        cbar_std.set_ticks([vmin_std, (vmin_std+vmax_std)/2, vmax_std])
        
        # 预测误差图
        ax_error = fig_error.add_subplot(gs_error[row, col])
        error_t = errors[:, :, t]
        im_error = ax_error.imshow(
            error_t,
            aspect='auto', origin='lower',
            cmap='RdBu_r', vmin=-vmax_error, vmax=vmax_error
        )
        ax_error.set_title(f'Day {t+1}', fontsize=8, pad=2)
        ax_error.set_xlabel('Longitude Index', fontsize=7)
        ax_error.set_ylabel('Latitude Index', fontsize=7)
        # 显式设置colorbar范围，确保刻度一致
        cbar_error = fig_error.colorbar(im_error, ax=ax_error, label='Error (K)', shrink=0.8, aspect=20)
        cbar_error.mappable.set_clim(vmin=-vmax_error, vmax=vmax_error)
        # 设置colorbar刻度范围（对称）
        cbar_error.set_ticks([-vmax_error, 0, vmax_error])
        
        if (t + 1) % 10 == 0:
            print(f"  已生成: {t+1}/{min(T, n_rows * n_cols)} 天")
    
    # 保存
    mean_path = FIGURES_DIR / f"{prefix}_mean_all_days.png"
    std_path = FIGURES_DIR / f"{prefix}_std_all_days.png"
    error_path = FIGURES_DIR / f"{prefix}_error_all_days.png"
    
    fig_mean.savefig(mean_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_mean)
    print(f"✅ {mean_path}")
    
    fig_std.savefig(std_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_std)
    print(f"✅ {std_path}")
    
    fig_error.savefig(error_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_error)
    print(f"✅ {error_path}")


def main():
    """主函数：计算统一颜色范围并生成所有图片"""
    print("=" * 80)
    print("  从npy文件生成all_days图片（统一颜色范围）")
    print("=" * 80)
    
    # 加载所有模型的数据
    models = {
        'unet': ('U-Net', 'unet'),
        'tree': ('Tree (XGBoost)', 'tree'),
        'gp': ('GP (Sparse)', 'gp')
    }
    
    all_means = []
    all_stds = []
    all_errors = []
    
    for key, (name, prefix) in models.items():
        mean_file = FIGURES_DIR / f"{prefix}_pred_mean.npy"
        std_file = FIGURES_DIR / f"{prefix}_pred_std.npy"
        true_file = FIGURES_DIR / f"{prefix}_true.npy"
        
        if all(f.exists() for f in [mean_file, std_file, true_file]):
            pred_mean = np.load(mean_file)
            pred_std = np.load(std_file)
            true_values = np.load(true_file)
            
            all_means.append(pred_mean)
            all_stds.append(pred_std)
            errors = true_values - pred_mean
            all_errors.append(np.abs(errors))
            print(f"✅ 已加载 {name} 数据")
        else:
            print(f"⚠️  {name} 数据文件不完整，跳过")
    
    if not all_means:
        print("❌ 没有找到任何模型数据文件")
        return
    
    # 计算统一颜色范围
    vmin_mean = min(np.nanmin(m) for m in all_means)
    vmax_mean = max(np.nanmax(m) for m in all_means)
    vmin_std = max(0, min(np.nanmin(s) for s in all_stds))
    vmax_std = max(np.nanmax(s) for s in all_stds)
    vmax_error = max(np.nanmax(e) for e in all_errors)
    
    print(f"\n统一颜色范围:")
    print(f"  Mean: [{vmin_mean:.2f}, {vmax_mean:.2f}] K")
    print(f"  Std:  [{vmin_std:.2f}, {vmax_std:.2f}] K")
    print(f"  Error: [-{vmax_error:.2f}, {vmax_error:.2f}] K")
    
    # 生成所有模型的图片
    for key, (name, prefix) in models.items():
        mean_file = FIGURES_DIR / f"{prefix}_pred_mean.npy"
        if mean_file.exists():
            try:
                plot_all_days_from_npy(name, prefix, vmin_mean, vmax_mean, 
                                      vmin_std, vmax_std, vmax_error)
            except Exception as e:
                print(f"❌ 生成 {name} 图片时出错: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("  完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

