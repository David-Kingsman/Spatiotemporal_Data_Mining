"""
训练概率U-Net模型（Probabilistic U-Net）

本脚本实现了基于U-Net架构的概率深度学习模型，用于MODIS地表温度数据的图像级插值和预测。

主要特点：
1. 概率输出：对每个像素输出均值和方差（log_var），提供不确定性估计
2. U-Net架构：编码器-解码器结构，适合图像inpainting任务
3. 批量归一化：提高训练稳定性
4. Dropout正则化：防止过拟合
5. 负对数似然损失：基于高斯假设的概率损失函数

数据格式：
- 输入：3维张量 (H, W, T) = (100, 200, 31)
  - H: 纬度维度（35°-40°N）
  - W: 经度维度（-115°--105°W）
  - T: 时间维度（31天）
- 输出：温度值（单位：Kelvin）
- 缺失值：用0表示

评估指标：
- 回归指标：RMSE, MAE, R², MAPE
- 概率指标：CRPS, 90%预测区间覆盖率, 校准误差

作者：lstinterp团队
创建时间：2024年
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import ProbUNet, UNetConfig, gaussian_nll_loss
from lstinterp.metrics import compute_regression_metrics, compute_probabilistic_metrics
from lstinterp.viz import plot_mean_map, plot_std_map, plot_error_map
from lstinterp.utils import set_seed

# 创建输出目录
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "results").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "models").mkdir(exist_ok=True)


def print_section_header(title, width=80):
    """打印章节标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    """主函数：训练和评估U-Net模型"""
    start_time = time.time()
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_section_header("实验配置", width=80)
    print(f"实验时间: {experiment_time}")
    print(f"随机种子: 42")
    
    set_seed(42)
    
    # 检查依赖库
    print("\n依赖库检查:")
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch未安装")
        return
    
    try:
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
    except ImportError:
        print("  ❌ NumPy未安装")
        return
    
    # 设置设备（在导入torch之后）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")
    if device.type == "cuda":
        print(f"  - GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载数据
    print_section_header("数据加载")
    data_path = "modis_aug_data/MODIS_Aug.mat"
    print(f"数据路径: {data_path}")
    
    print("\n加载训练数据...")
    train_tensor = load_modis_tensor(data_path, "training_tensor")
    H, W, T = train_tensor.shape
    print(f"训练数据维度: {H} × {W} × {T}")
    
    print("\n加载测试数据...")
    test_tensor = load_modis_tensor(data_path, "test_tensor")
    print(f"测试数据维度: {H} × {W} × {T}")
    
    # 创建数据集
    print_section_header("数据预处理")
    print("转换为图像数据格式 (T, 1, H, W) → (mean, log_var)")
    
    print("\n创建训练数据集（图像模式）...")
    train_dataset = MODISDataset(train_tensor, mode="image")
    print(f"  - 训练图像数量: {len(train_dataset)} 张 (每天1张)")
    print(f"  - 图像尺寸: {H} × {W} 像素")
    
    # 获取训练集的归一化统计量（用于测试时保持一致）
    train_mean = train_dataset.mean_val
    train_std = train_dataset.std_val
    print(f"\n数据归一化统计 (Z-score):")
    print(f"  - 均值: {train_mean:.2f} K")
    print(f"  - 标准差: {train_std:.2f} K")
    print(f"  - 归一化范围: 约 [{train_mean - 3*train_std:.2f}, {train_mean + 3*train_std:.2f}] K")
    
    # 数据加载器（改进配置）
    print_section_header("模型配置")
    config = UNetConfig(
        batch_size=4,          # 批大小（根据GPU内存调整）
        num_epochs=50,         # 训练轮数
        lr=5e-4,               # 学习率
        dropout=0.2,           # Dropout比率（防止过拟合）
        init_log_var=-1.0      # 初始log_var=-1，对应标准差≈0.37（归一化后合理）
    )
    
    print("模型超参数:")
    print(f"  - 批大小: {config.batch_size}")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - Dropout: {config.dropout}")
    print(f"  - 初始log_var: {config.init_log_var}")
    print(f"  - 输入通道数: {config.in_channels} (温度图 + mask)")
    print(f"  - 基础通道数: {config.base_channels}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows兼容
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建验证集（使用训练数据的一部分）
    print("\n创建训练/验证划分...")
    train_size = len(train_dataset)
    val_size = max(1, int(train_size * 0.1))  # 10%作为验证集
    indices = np.random.RandomState(42).permutation(train_size)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    print(f"  - 训练集: {len(train_indices)} 张图像 ({len(train_indices)/train_size*100:.1f}%)")
    print(f"  - 验证集: {len(val_indices)} 张图像 ({len(val_indices)/train_size*100:.1f}%)")
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
    
    # 创建模型
    print("\n创建模型...")
    model = ProbUNet(config).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 模型参数总数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    # 模型结构说明
    print("\n模型结构:")
    print("  - 架构: U-Net (Encoder-Decoder)")
    print("  - 编码器: 卷积层 + 最大池化")
    print("  - 解码器: 转置卷积 + 上采样")
    print("  - 跳跃连接: 连接编码器和解码器的对应层")
    print("  - 输出: mean (B, 1, H, W) 和 log_var (B, 1, H, W)")
    
    # 优化器（使用学习率调度器）
    print("\n优化器配置:")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    print(f"  - 优化器: Adam")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 权重衰减: 1e-5 (L2正则化)")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    print(f"  - 学习率调度器: ReduceLROnPlateau")
    print(f"  - 降低因子: 0.5")
    print(f"  - 耐心值: 5 epochs")
    
    print(f"\n损失函数:")
    print(f"  - 类型: Gaussian Negative Log-Likelihood")
    print(f"  - 仅在观测点上计算（mask > 0.5）")
    
    # 训练（带验证集监控）
    print_section_header("模型训练")
    best_loss = float('inf')
    best_epoch = 1
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    training_start_time = time.time()
    
    print(f"开始训练 ({config.num_epochs} 个epoch)...")
    print("-" * 100)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Best Val':<15} {'LR':<15} {'时间':<10}")
    print("-" * 100)
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch_idx, (img, mask, target) in enumerate(train_loader):
            img = img.to(device)
            mask = mask.to(device)
            target = target.to(device)
            
            x = torch.cat([img, mask], dim=1)
            
            optimizer.zero_grad()
            mean, log_var = model(x)
            loss = gaussian_nll_loss(mean, log_var, target, mask)
            
            if torch.isnan(loss) or torch.isinf(loss):
                if batch_idx == 0:  # 只打印第一次警告
                    print(f"    ⚠️  警告: 检测到无效loss (NaN/Inf)，跳过此batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for img, mask, target in val_loader:
                img = img.to(device)
                mask = mask.to(device)
                target = target.to(device)
                x = torch.cat([img, mask], dim=1)
                
                mean, log_var = model(x)
                loss = gaussian_nll_loss(mean, log_var, target, mask)
                
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    n_val_batches += 1
        
        avg_val_loss = val_loss / max(n_val_batches, 1) if n_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_reduced = (new_lr < old_lr)
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        status = "⭐" if avg_val_loss == best_loss else ("📉" if lr_reduced else " ")
        
        # 每5个epoch或最后一个epoch打印一次
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs or (epoch + 1) == 1:
            print(f"{epoch+1:<8} {avg_train_loss:<15.4f} {avg_val_loss:<15.4f} {best_loss:<15.4f} {current_lr:<15.6f} {epoch_time:<10.2f}s {status}")
        
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发（patience={patience}），恢复最佳模型 (Epoch {best_epoch})")
            model.load_state_dict(best_model_state)
            break
    
    training_time = time.time() - training_start_time
    print("-" * 100)
    print(f"训练完成！")
    print(f"  - 总训练时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
    print(f"  - 最佳验证Loss: {best_loss:.4f} (Epoch {best_epoch})")
    print(f"  - 最终训练Loss: {avg_train_loss:.4f}")
    print(f"  - 最终验证Loss: {avg_val_loss:.4f}")
    print(f"  - 平均每epoch时间: {training_time/(epoch+1):.2f} 秒")
    
    # 加载最佳模型
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
        print(f"\n已加载最佳模型 (Epoch {best_epoch}, 验证Loss={best_loss:.4f})")
    
    # 评估
    print_section_header("模型评估")
    evaluation_start_time = time.time()
    model.eval()
    
    # 在测试数据上评估（使用训练集的归一化参数）
    test_dataset = MODISDataset(test_tensor, mode="image", norm_mean=train_mean, norm_std=train_std)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_preds_mean = []
    all_preds_std = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for img, mask, target in test_loader:
            img = img.to(device)
            mask = mask.to(device)
            target = target.to(device)
            x = torch.cat([img, mask], dim=1)
            
            mean, log_var = model(x)
            std = torch.exp(0.5 * log_var)
            
            # 转移到CPU再转numpy
            all_preds_mean.append(mean.cpu().numpy())
            all_preds_std.append(std.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
    
    # 合并结果
    pred_mean = np.concatenate(all_preds_mean, axis=0)[:, 0, :, :]  # (T, H, W)
    pred_std = np.concatenate(all_preds_std, axis=0)[:, 0, :, :]
    targets = np.concatenate(all_targets, axis=0)[:, 0, :, :]
    masks = np.concatenate(all_masks, axis=0)[:, 0, :, :]
    
    # 反归一化预测结果（恢复到原始尺度）
    # 使用训练集的统计量（与训练时一致）
    mean_val = train_mean
    std_val = train_std
    
    # 只在有观测的点上评估
    valid_mask = masks > 0.5
    y_true_norm = targets[valid_mask]  # 归一化的真实值
    y_pred_norm = pred_mean[valid_mask]  # 归一化的预测值
    y_std_norm = pred_std[valid_mask]  # 归一化的标准差
    
    # 反归一化
    y_true = y_true_norm * std_val + mean_val
    y_pred = y_pred_norm * std_val + mean_val
    y_std = y_std_norm * std_val
    
    evaluation_time = time.time() - evaluation_start_time
    print(f"预测完成 (耗时: {evaluation_time:.2f} 秒)")
    print(f"  - 有效预测点数: {len(y_true):,} (仅在观测点上评估)")
    
    # 计算指标
    print("\n计算评估指标...")
    reg_metrics = compute_regression_metrics(y_true, y_pred)
    prob_metrics = compute_probabilistic_metrics(y_true, y_pred, y_std)
    
    all_metrics = {**reg_metrics, **prob_metrics}
    
    # 添加实验信息
    all_metrics["experiment_info"] = {
        "experiment_time": experiment_time,
        "random_seed": 42,
        "device": str(device),
        "training_time_seconds": training_time,
        "evaluation_time_seconds": evaluation_time,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_loss),
        "final_train_loss": float(avg_train_loss),
        "final_val_loss": float(avg_val_loss),
        "model_config": {
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "lr": config.lr,
            "dropout": config.dropout,
            "init_log_var": config.init_log_var,
            "in_channels": config.in_channels,
            "base_channels": config.base_channels
        },
        "data_info": {
            "train_images": len(train_indices),
            "val_images": len(val_indices),
            "test_images": T,
            "image_size": f"{H}×{W}",
            "normalization": {
                "mean": float(train_mean),
                "std": float(train_std)
            },
            "valid_test_points": len(y_true)
        }
    }
    
    print("\n" + "=" * 80)
    print("  评估结果")
    print("=" * 80)
    
    # 回归指标
    print("\n【回归指标】")
    print(f"  {'指标':<30} {'值':<15} {'说明':<30}")
    print("-" * 75)
    print(f"  {'RMSE (Root Mean Squared Error)':<30} {reg_metrics['rmse']:<15.4f} {'越小越好，单位: Kelvin'}")
    print(f"  {'MAE (Mean Absolute Error)':<30} {reg_metrics['mae']:<15.4f} {'越小越好，单位: Kelvin'}")
    print(f"  {'R² (Coefficient of Determination)':<30} {reg_metrics['r2']:<15.4f} {'越大越好，范围: (-∞, 1]'}")
    print(f"  {'MAPE (Mean Absolute Percentage Error)':<30} {reg_metrics['mape']:<15.4f} {'越小越好，单位: %'}")
    
    # 概率指标
    print("\n【概率预测指标】")
    print(f"  {'指标':<30} {'值':<15} {'说明':<30}")
    print("-" * 75)
    print(f"  {'CRPS (Continuous Ranked Probability Score)':<30} {prob_metrics['crps']:<15.4f} {'越小越好，单位: Kelvin'}")
    print(f"  {'Coverage (90% Prediction Interval)':<30} {prob_metrics['coverage_90']:<15.4f} {'目标: 0.90'}")
    print(f"  {'Interval Width (90%)':<30} {prob_metrics['interval_width_90']:<15.4f} {'越小越好，单位: Kelvin'}")
    print(f"  {'Calibration Error':<30} {prob_metrics['calibration_error']:<15.4f} {'越小越好，衡量校准度'}")
    
    # 预测统计
    print("\n【预测统计】")
    print(f"  预测均值:")
    print(f"    - 范围: [{y_pred.min():.2f}, {y_pred.max():.2f}] K")
    print(f"    - 均值: {y_pred.mean():.2f} K")
    print(f"    - 标准差: {y_pred.std():.2f} K")
    
    print(f"\n  真实值:")
    print(f"    - 范围: [{y_true.min():.2f}, {y_true.max():.2f}] K")
    print(f"    - 均值: {y_true.mean():.2f} K")
    print(f"    - 标准差: {y_true.std():.2f} K")
    
    print(f"\n  预测不确定性 (标准差):")
    print(f"    - 范围: [{y_std.min():.2f}, {y_std.max():.2f}] K")
    print(f"    - 均值: {y_std.mean():.2f} K")
    print(f"    - 中位数: {np.median(y_std):.2f} K")
    
    # 误差分析
    errors = y_true - y_pred
    print(f"\n【误差分析】")
    print(f"  残差 (真实值 - 预测值):")
    print(f"    - 均值: {errors.mean():.2f} K (接近0表示无偏)")
    print(f"    - 标准差: {errors.std():.2f} K")
    print(f"    - 范围: [{errors.min():.2f}, {errors.max():.2f}] K")
    print(f"    - 中位数: {np.median(errors):.2f} K")
    
    # 覆盖率分析
    coverage = prob_metrics['coverage_90']
    target_coverage = 0.90
    coverage_error = abs(coverage - target_coverage)
    print(f"\n【不确定性校准】")
    print(f"  90%预测区间覆盖率: {coverage:.4f} (目标: {target_coverage})")
    if coverage_error < 0.05:
        print(f"  ✅ 校准良好 (误差 < 5%)")
    elif coverage_error < 0.10:
        print(f"  ⚠️  校准尚可 (误差 < 10%)")
    else:
        print(f"  ❌ 校准较差 (误差 >= 10%)")
    
    # 保存结果
    print_section_header("保存结果")
    results_path = OUTPUT_DIR / "results" / "unet_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ 评估结果已保存: {results_path}")
    
    # 保存模型
    model_path = OUTPUT_DIR / "models" / "unet_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'experiment_info': all_metrics["experiment_info"]
    }, model_path)
    print(f"✅ 模型已保存: {model_path}")
    print(f"  - 模型大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 保存训练损失曲线
    loss_curve_path = OUTPUT_DIR / "results" / "unet_training_losses.json"
    with open(loss_curve_path, "w") as f:
        json.dump({
            "epochs": list(range(1, len(train_losses) + 1)),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_loss)
        }, f, indent=2)
    print(f"✅ 训练损失曲线已保存: {loss_curve_path}")
    
    # 反归一化可视化用的数据（恢复到原始尺度）
    pred_mean_denorm = pred_mean * std_val + mean_val
    pred_std_denorm = pred_std * std_val
    targets_denorm = targets * std_val + mean_val
    
    # 可视化（第15天）
    print("\n生成可视化图表...")
    day_idx = 14
    mean_path = OUTPUT_DIR / "figures" / "unet_mean_day15.png"
    std_path = OUTPUT_DIR / "figures" / "unet_std_day15.png"
    error_path = OUTPUT_DIR / "figures" / "unet_error_day15.png"
    
    plot_mean_map(
        pred_mean_denorm, day_idx=day_idx,
        title="U-Net Mean Prediction - Day 15",
        save_path=str(mean_path)
    )
    print(f"✅ 预测均值图已保存: {mean_path}")
    
    plot_std_map(
        pred_std_denorm, day_idx=day_idx,
        title="U-Net Prediction Uncertainty - Day 15",
        save_path=str(std_path)
    )
    print(f"✅ 预测不确定性图已保存: {std_path}")
    
    plot_error_map(
        targets_denorm, pred_mean_denorm, day_idx=day_idx,
        title="U-Net Prediction Error - Day 15",
        save_path=str(error_path)
    )
    print(f"✅ 预测误差图已保存: {error_path}")
    
    # 总结
    total_time = time.time() - start_time
    print_section_header("实验完成")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"  - 数据加载和预处理: {training_start_time - start_time:.2f} 秒")
    print(f"  - 模型训练: {training_time:.2f} 秒")
    print(f"  - 模型评估: {evaluation_time:.2f} 秒")
    
    print(f"\n主要指标总结:")
    print(f"  - R²: {reg_metrics['r2']:.4f}")
    print(f"  - RMSE: {reg_metrics['rmse']:.4f} K")
    print(f"  - CRPS: {prob_metrics['crps']:.4f} K")
    print(f"  - 覆盖率(90%): {prob_metrics['coverage_90']:.4f}")
    
    print(f"\n所有结果文件:")
    print(f"  📄 {results_path}")
    print(f"  📄 {loss_curve_path}")
    print(f"  💾 {model_path}")
    print(f"  📊 {mean_path}")
    print(f"  📊 {std_path}")
    print(f"  📊 {error_path}")


if __name__ == "__main__":
    main()

