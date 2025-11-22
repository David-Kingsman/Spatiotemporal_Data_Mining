"""
训练时空高斯过程模型（Spatio-temporal Gaussian Process）

本脚本实现了基于时空可分核（separable spatio-temporal kernel）的稀疏高斯过程模型，
用于MODIS地表温度数据的插值和预测。

主要特点：
1. 时空可分核：k(x, x') = k_space(lat, lon) × k_time(t)
   - 空间核：Matern 3/2（捕获空间相关性）
   - 时间核：Matern 3/2（捕获时间相关性）
2. 稀疏GP：使用诱导点（inducing points）提高可扩展性
3. 变分推理：使用Variational ELBO进行高效训练
4. 概率预测：提供预测均值和不确定性估计

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
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import GPSTModel, GPSTConfig
from lstinterp.metrics import compute_regression_metrics, compute_probabilistic_metrics
from lstinterp.viz import plot_prediction_scatter, plot_residuals
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


def print_data_statistics(tensor, name, mode="point"):
    """打印详细的数据统计信息"""
    print_section_header(f"{name} 数据统计")
    
    H, W, T = tensor.shape
    print(f"数据维度: {H} × {W} × {T}")
    print(f"  - 纬度维度 (H): {H} 个网格点，范围: 35°N - 40°N")
    print(f"  - 经度维度 (W): {W} 个网格点，范围: -115°W - -105°W")
    print(f"  - 时间维度 (T): {T} 天（2020年8月）")
    
    # 缺失值统计
    mask = (tensor != 0.0)
    total_points = H * W * T
    observed_points = mask.sum()
    missing_points = total_points - observed_points
    missing_ratio = missing_points / total_points * 100
    
    print(f"\n缺失值统计:")
    print(f"  - 总网格点数: {total_points:,}")
    print(f"  - 观测点数: {observed_points:,} ({observed_points/total_points*100:.2f}%)")
    print(f"  - 缺失点数: {missing_points:,} ({missing_ratio:.2f}%)")
    
    # 温度统计
    observed_values = tensor[mask]
    print(f"\n温度统计 (Kelvin):")
    print(f"  - 均值: {observed_values.mean():.2f} K")
    print(f"  - 标准差: {observed_values.std():.2f} K")
    print(f"  - 最小值: {observed_values.min():.2f} K")
    print(f"  - 最大值: {observed_values.max():.2f} K")
    print(f"  - 中位数: {np.median(observed_values):.2f} K")
    
    # 每天缺失值统计
    missing_per_day = []
    for t in range(T):
        day_mask = (tensor[:, :, t] != 0.0)
        missing_per_day.append((H * W - day_mask.sum()) / (H * W) * 100)
    
    print(f"\n每日缺失值比率:")
    print(f"  - 平均缺失率: {np.mean(missing_per_day):.2f}%")
    print(f"  - 最小缺失率: {np.min(missing_per_day):.2f}% (第{np.argmin(missing_per_day)+1}天)")
    print(f"  - 最大缺失率: {np.max(missing_per_day):.2f}% (第{np.argmax(missing_per_day)+1}天)")


def main():
    """主函数：训练和评估GP模型"""
    start_time = time.time()
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_section_header("实验配置", width=80)
    print(f"实验时间: {experiment_time}")
    print(f"随机种子: 42")
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")
    if device.type == "cuda":
        print(f"  - GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 检查依赖库
    print("\n依赖库检查:")
    try:
        import gpytorch
        print(f"  ✅ GPyTorch: {gpytorch.__version__}")
    except ImportError:
        print("  ❌ 错误: 需要安装 gpytorch")
        print("  请运行: pip install gpytorch")
        return
    
    try:
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
    except ImportError:
        print("  ❌ NumPy未安装")
        return
    
    # 加载数据
    print_section_header("数据加载")
    data_path = "modis_aug_data/MODIS_Aug.mat"
    print(f"数据路径: {data_path}")
    
    print("\n加载训练数据...")
    train_tensor = load_modis_tensor(data_path, "training_tensor")
    print_data_statistics(train_tensor, "训练集")
    
    print("\n加载测试数据...")
    test_tensor = load_modis_tensor(data_path, "test_tensor")
    print_data_statistics(test_tensor, "测试集")
    
    # 创建数据集（point模式）
    print_section_header("数据预处理")
    print("转换为点数据格式 (lat, lon, time) → temperature")
    
    print("\n创建训练数据集...")
    train_dataset = MODISDataset(train_tensor, mode="point")
    print(f"  - 训练观测点数: {len(train_dataset):,}")
    
    print("\n创建测试数据集...")
    test_dataset = MODISDataset(test_tensor, mode="point")
    print(f"  - 测试观测点数: {len(test_dataset):,}")
    
    # 准备训练数据
    print("\n提取训练数据...")
    X_train = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
    y_train = np.array([train_dataset[i][1].numpy() for i in range(len(train_dataset))])
    
    print(f"  - 输入特征维度: {X_train.shape}")
    print(f"    * 特征1 (纬度): 范围 [{X_train[:, 0].min():.2f}, {X_train[:, 0].max():.2f}]")
    print(f"    * 特征2 (经度): 范围 [{X_train[:, 1].min():.2f}, {X_train[:, 1].max():.2f}]")
    print(f"    * 特征3 (时间): 范围 [{X_train[:, 2].min():.0f}, {X_train[:, 2].max():.0f}] 天")
    print(f"  - 目标变量维度: {y_train.shape}")
    print(f"    * 温度范围: [{y_train.min():.2f}, {y_train.max():.2f}] K")
    print(f"    * 温度均值: {y_train.mean():.2f} K")
    print(f"    * 温度标准差: {y_train.std():.2f} K")
    
    # 准备测试数据
    print("\n提取测试数据...")
    X_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1].numpy() for i in range(len(test_dataset))])
    
    print(f"  - 输入特征维度: {X_test.shape}")
    print(f"    * 特征1 (纬度): 范围 [{X_test[:, 0].min():.2f}, {X_test[:, 0].max():.2f}]")
    print(f"    * 特征2 (经度): 范围 [{X_test[:, 1].min():.2f}, {X_test[:, 1].max():.2f}]")
    print(f"    * 特征3 (时间): 范围 [{X_test[:, 2].min():.0f}, {X_test[:, 2].max():.0f}] 天")
    print(f"  - 目标变量维度: {y_test.shape}")
    print(f"    * 温度范围: [{y_test.min():.2f}, {y_test.max():.2f}] K")
    print(f"    * 温度均值: {y_test.mean():.2f} K")
    print(f"    * 温度标准差: {y_test.std():.2f} K")
    
    # 转换为tensor
    print("\n转换为PyTorch张量...")
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test_np = y_test.copy()
    print(f"  - 数据类型: {X_train.dtype}")
    print(f"  - 设备: {device}")
    
    # 配置模型
    print_section_header("模型配置")
    config = GPSTConfig(
        kernel_space="matern32",  # 空间核：Matern 3/2
        kernel_time="matern32",   # 时间核：Matern 3/2
        num_inducing=500,         # 诱导点数量（控制模型复杂度）
        lr=0.01,                  # 学习率
        num_epochs=50,            # 训练轮数
        batch_size=1000           # 批大小
    )
    
    print("模型超参数:")
    print(f"  - 空间核函数: {config.kernel_space} (Matern 3/2)")
    print(f"  - 时间核函数: {config.kernel_time} (Matern 3/2)")
    print(f"  - 诱导点数量: {config.num_inducing}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 批大小: {config.batch_size}")
    
    print("\n创建诱导点...")
    # 创建诱导点（使用训练数据的一个子集）
    from lstinterp.models.gp_st import create_inducing_points
    n_space = 15  # 15×15 = 225 个空间点
    n_time = 10   # 10 个时间点
    print(f"  - 空间网格: {n_space}×{n_space} = {n_space**2} 个点")
    print(f"  - 时间点: {n_time} 个点")
    print(f"  - 理论诱导点总数: {n_space**2 * n_time:,} 个点")
    
    inducing_points = create_inducing_points(
        n_space=n_space,
        n_time=n_time,
        normalize=True
    ).float().to(device)  # 转换为float32以匹配训练数据
    
    print(f"  - 实际诱导点数量: {len(inducing_points):,}")
    
    # 如果诱导点数量超过配置，使用随机采样
    if len(inducing_points) > config.num_inducing:
        print(f"  - 诱导点过多，随机采样至 {config.num_inducing} 个")
        indices = torch.randperm(len(inducing_points))[:config.num_inducing]
        inducing_points = inducing_points[indices]
        print(f"  - 最终诱导点数量: {len(inducing_points)}")
    else:
        print(f"  - 使用全部诱导点: {len(inducing_points)}")
    
    print("\n创建模型...")
    model = GPSTModel(inducing_points, config).to(device)
    model = model.float()  # 确保模型也是float32
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 模型参数总数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    # 模型结构说明
    print("\n模型结构:")
    print("  - GP类型: Sparse Variational GP (SVGP)")
    print("  - 核函数: 时空可分核 k(x, x') = k_space(lat, lon) × k_time(t)")
    print("  - 变分分布: CholeskyVariationalDistribution")
    print("  - 变分策略: VariationalStrategy (learn_inducing_locations=True)")
    print("  - 均值函数: ConstantMean")
    print("  - 似然函数: GaussianLikelihood")
    
    # 训练
    print_section_header("模型训练")
    model.train()
    model.likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print(f"优化器: Adam")
    print(f"  - 学习率: {config.lr}")
    
    # 使用marginal log likelihood作为损失
    # VariationalELBO需要接收GP对象（model.gp），而不是包装器
    mll = gpytorch.mlls.VariationalELBO(
        model.likelihood, 
        model.gp,  # 使用GP对象而不是包装器
        num_data=len(X_train)
    )
    print(f"损失函数: Variational ELBO")
    print(f"  - 数据量: {len(X_train):,} 个点")
    
    best_loss = float('inf')
    best_model_state = None
    best_epoch = 1  # 初始化为第一个epoch
    train_losses = []
    training_start_time = time.time()
    
    print(f"\n开始训练 ({config.num_epochs} 个epoch)...")
    print("-" * 80)
    print(f"{'Epoch':<8} {'Loss':<15} {'最佳Loss':<15} {'时间':<10}")
    print("-" * 80)
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        model.train()
        model.likelihood.train()
        
        # 批量训练（如果数据量大）
        epoch_loss = 0
        n_batches = 0
        
        if len(X_train) > config.batch_size:
            # 随机打乱
            indices = torch.randperm(len(X_train))
            n_batches_total = (len(X_train) + config.batch_size - 1) // config.batch_size
            
            for i in range(0, len(X_train), config.batch_size):
                batch_indices = indices[i:i+config.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                optimizer.zero_grad()
                output = model.gp(X_batch)  # 直接使用GP对象
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
        else:
            optimizer.zero_grad()
            output = model.gp(X_train)  # 直接使用GP对象
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            
            epoch_loss = loss.item()
            n_batches = 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start_time
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        
        # 每10个epoch或最后一个epoch打印一次
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.num_epochs:
            status = "⭐" if avg_loss == best_loss else " "
            print(f"{epoch+1:<8} {avg_loss:<15.4f} {best_loss:<15.4f} {epoch_time:<10.2f}s {status}")
    
    training_time = time.time() - training_start_time
    print("-" * 80)
    print(f"训练完成！")
    print(f"  - 总训练时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
    print(f"  - 最佳Loss: {best_loss:.4f} (Epoch {best_epoch})")
    print(f"  - 最终Loss: {avg_loss:.4f}")
    print(f"  - 平均每epoch时间: {training_time/config.num_epochs:.2f} 秒")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n已加载最佳模型 (Epoch {best_epoch}, Loss={best_loss:.4f})")
    
    # 评估
    print_section_header("模型评估")
    evaluation_start_time = time.time()
    
    model.eval()
    model.likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # 分批预测（如果测试数据量大）
        pred_mean_list = []
        pred_std_list = []
        
        batch_size = 1000
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            output = model.gp(X_batch)  # 直接使用GP对象
            pred_dist = model.likelihood(output)
            
            pred_mean_list.append(pred_dist.mean.cpu().numpy())
            pred_std_list.append(pred_dist.stddev.cpu().numpy())
        
        y_pred_mean = np.concatenate(pred_mean_list)
        y_pred_std = np.concatenate(pred_std_list)
    
    evaluation_time = time.time() - evaluation_start_time
    print(f"预测完成 (耗时: {evaluation_time:.2f} 秒)")
    
    # 计算指标
    print("\n计算评估指标...")
    reg_metrics = compute_regression_metrics(y_test_np, y_pred_mean)
    prob_metrics = compute_probabilistic_metrics(y_test_np, y_pred_mean, y_pred_std)
    
    all_metrics = {**reg_metrics, **prob_metrics}
    
    # 添加训练信息到结果
    all_metrics["experiment_info"] = {
        "experiment_time": experiment_time,
        "random_seed": 42,
        "device": str(device),
        "training_time_seconds": training_time,
        "evaluation_time_seconds": evaluation_time,
        "best_epoch": best_epoch,
        "best_loss": float(best_loss),
        "final_loss": float(avg_loss),
        "model_config": {
            "kernel_space": config.kernel_space,
            "kernel_time": config.kernel_time,
            "num_inducing": config.num_inducing,
            "lr": config.lr,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size
        },
        "data_info": {
            "train_points": len(X_train),
            "test_points": len(X_test),
            "n_space_inducing": n_space,
            "n_time_inducing": n_time,
            "total_inducing_points": len(inducing_points)
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
    print(f"    - 范围: [{y_pred_mean.min():.2f}, {y_pred_mean.max():.2f}] K")
    print(f"    - 均值: {y_pred_mean.mean():.2f} K")
    print(f"    - 标准差: {y_pred_mean.std():.2f} K")
    
    print(f"\n  真实值:")
    print(f"    - 范围: [{y_test_np.min():.2f}, {y_test_np.max():.2f}] K")
    print(f"    - 均值: {y_test_np.mean():.2f} K")
    print(f"    - 标准差: {y_test_np.std():.2f} K")
    
    print(f"\n  预测不确定性 (标准差):")
    print(f"    - 范围: [{y_pred_std.min():.2f}, {y_pred_std.max():.2f}] K")
    print(f"    - 均值: {y_pred_std.mean():.2f} K")
    print(f"    - 中位数: {np.median(y_pred_std):.2f} K")
    
    # 误差分析
    errors = y_test_np - y_pred_mean
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
    results_path = OUTPUT_DIR / "results" / "gp_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ 评估结果已保存: {results_path}")
    
    # 保存模型
    model_path = OUTPUT_DIR / "models" / "gp_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'inducing_points': inducing_points.cpu(),
        'experiment_info': all_metrics["experiment_info"]
    }, model_path)
    print(f"✅ 模型已保存: {model_path}")
    print(f"  - 模型大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 保存训练损失曲线
    loss_curve_path = OUTPUT_DIR / "results" / "gp_training_losses.json"
    with open(loss_curve_path, "w") as f:
        json.dump({
            "epochs": list(range(1, len(train_losses) + 1)),
            "losses": train_losses,
            "best_epoch": best_epoch,
            "best_loss": float(best_loss)
        }, f, indent=2)
    print(f"✅ 训练损失曲线已保存: {loss_curve_path}")
    
    # 可视化
    print("\n生成可视化图表...")
    scatter_path = OUTPUT_DIR / "figures" / "gp_scatter.png"
    residuals_path = OUTPUT_DIR / "figures" / "gp_residuals.png"
    
    plot_prediction_scatter(y_test_np, y_pred_mean, save_path=str(scatter_path))
    print(f"✅ 预测散点图已保存: {scatter_path}")
    
    plot_residuals(y_test_np, y_pred_mean, save_path=str(residuals_path))
    print(f"✅ 残差图已保存: {residuals_path}")
    
    # 总结
    total_time = time.time() - start_time
    print_section_header("实验完成")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"  - 数据加载和预处理: {training_start_time - start_time:.2f} 秒")
    print(f"  - 模型训练: {training_time:.2f} 秒")
    print(f"  - 模型评估: {evaluation_time:.2f} 秒")
    print(f"  - 结果保存和可视化: {total_time - evaluation_time - training_time - (training_start_time - start_time):.2f} 秒")
    
    print(f"\n主要指标总结:")
    print(f"  - R²: {reg_metrics['r2']:.4f}")
    print(f"  - RMSE: {reg_metrics['rmse']:.4f} K")
    print(f"  - CRPS: {prob_metrics['crps']:.4f} K")
    print(f"  - 覆盖率(90%): {prob_metrics['coverage_90']:.4f}")
    
    print(f"\n所有结果文件:")
    print(f"  📄 {results_path}")
    print(f"  📄 {loss_curve_path}")
    print(f"  💾 {model_path}")
    print(f"  📊 {scatter_path}")
    print(f"  📊 {residuals_path}")


if __name__ == "__main__":
    main()

