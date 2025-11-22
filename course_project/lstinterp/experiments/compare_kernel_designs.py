"""
核函数设计对比实验

根据 Gemini 的建议，对比三种不同复杂度的时空核函数设计：
- Design 1: 可分离核 k_space × k_time
- Design 2: 加性核 k_RQ(space) + k_Periodic(time) + k_Linear(time)
- Design 3: 非分离核 k_Matern(3D input)

这个脚本将训练三种不同的设计，并在相同的数据上评估它们的性能。
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 检查并导入 gpytorch
try:
    import gpytorch
    from gpytorch.mlls import VariationalELBO
    GPYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️  警告: 需要安装 gpytorch 才能运行此脚本")
    print("   请运行: pip install gpytorch")
    GPYTORCH_AVAILABLE = False
    gpytorch = None
    VariationalELBO = None

from lstinterp.data.modis import load_modis_tensor, MODISDataset
from lstinterp.models.gp_st import GPSTModel, GPSTConfig, create_inducing_points
from lstinterp.metrics.probabilistic import crps_gaussian
from lstinterp.utils import set_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_section_header(title: str):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def evaluate_model(model, test_dataset, device='cpu'):
    """
    评估模型性能
    
    返回:
    metrics: dict，包含各种评估指标
    """
    model.eval()
    model.likelihood.eval()
    
    # 收集所有测试数据
    test_values = []
    test_coords = []
    
    for i in range(len(test_dataset)):
        coords, value = test_dataset[i]
        test_coords.append(coords.numpy())
        test_values.append(value.item())
    
    test_coords = torch.tensor(np.array(test_coords), dtype=torch.float32, device=device)
    test_values = np.array(test_values)
    
    # 批量预测（避免内存溢出）
    batch_size = 1000
    means = []
    stds = []
    
    with torch.no_grad():
        for i in range(0, len(test_coords), batch_size):
            batch_coords = test_coords[i:i+batch_size]
            mean_batch, std_batch = model.predict(batch_coords)
            means.append(mean_batch.cpu().numpy())
            stds.append(std_batch.cpu().numpy())
    
    mean = np.concatenate(means)
    std = np.concatenate(stds)
    
    # 计算评估指标
    rmse = np.sqrt(np.mean((test_values - mean) ** 2))
    mae = np.mean(np.abs(test_values - mean))
    mape = np.mean(np.abs((test_values - mean) / (test_values + 1e-8))) * 100
    
    ss_res = np.sum((test_values - mean) ** 2)
    ss_tot = np.sum((test_values - np.mean(test_values)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # CRPS
    crps = np.mean(crps_gaussian(test_values, mean, std))
    
    # 预测区间覆盖率（90%）
    lower = mean - 1.645 * std  # 5%分位数
    upper = mean + 1.645 * std  # 95%分位数
    coverage = np.mean((test_values >= lower) & (test_values <= upper))
    interval_width = np.mean(upper - lower)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'crps': crps,
        'coverage': coverage,
        'interval_width': interval_width,
        'mean': mean,
        'std': std,
        'true_values': test_values
    }


def train_model(config, train_dataset, device='cpu', verbose=True):
    """
    训练 GP 模型
    
    返回:
    model: 训练好的模型
    train_time: 训练时间（秒）
    """
    if verbose:
        print(f"\n训练 {config.kernel_design} 设计...")
    
    start_time = time.time()
    
    # 创建诱导点
    n_space = int(np.sqrt(config.num_inducing // 10))  # 假设10个时间点
    n_time = min(10, 31)  # 限制时间点数量
    
    inducing_points = create_inducing_points(
        n_space=n_space,
        n_time=n_time,
        normalize=True
    ).to(device)
    
    # 创建模型
    model = GPSTModel(
        inducing_points=inducing_points,
        config=config,
        lengthscale_space=0.5,
        lengthscale_time=0.3,
        outputscale=10.0,
        noise=1.0,
        alpha=1.0,  # RQ 核参数（仅用于 additive 设计）
        period=1.0  # Periodic 核参数（仅用于 additive 设计）
    ).to(device)
    
    # 优化器
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # 训练循环（简化版，使用较少的数据以加快速度）
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    
    # 限制训练数据量（用于对比实验）
    max_train_samples = 5000  # 使用前5000个样本进行快速训练
    train_samples = min(len(train_dataset), max_train_samples)
    
    if not GPYTORCH_AVAILABLE:
        raise ImportError("需要安装 gpytorch 才能训练模型")
    
    mll = VariationalELBO(
        model.likelihood, model.gp, num_data=train_samples
    )
    
    for epoch in range(min(config.num_epochs, 20)):  # 限制最大训练轮数
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, (coords, values) in enumerate(train_loader):
            if batch_idx * config.batch_size >= max_train_samples:
                break
            
            coords = coords.to(device)
            values = values.to(device)
            
            optimizer.zero_grad()
            output = model(coords)
            loss = -mll(output, values)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{config.num_epochs}, Loss: {epoch_loss / n_batches:.4f}")
    
    train_time = time.time() - start_time
    
    return model, train_time


def compare_kernel_designs(data_path: str = None, output_dir: str = None):
    """
    对比三种核函数设计
    
    参数:
    data_path: MODIS数据路径
    output_dir: 输出目录
    """
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    if data_path is None:
        data_path = project_root / "modis_aug_data" / "MODIS_Aug.mat"
    if output_dir is None:
        output_dir = project_root / "output" / "kernel_comparison"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print_section_header("核函数设计对比实验")
    
    # 加载数据
    print("\n加载数据...")
    training_tensor = load_modis_tensor(str(data_path), key="training_tensor")
    test_tensor = load_modis_tensor(str(data_path), key="test_tensor")
    
    print(f"训练数据形状: {training_tensor.shape}")
    print(f"测试数据形状: {test_tensor.shape}")
    
    # 创建数据集（使用点模式）
    train_dataset = MODISDataset(training_tensor, mode="point")
    test_dataset = MODISDataset(test_tensor, mode="point")
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    # 三种核函数设计配置
    designs = [
        {
            'name': 'Design 1: 可分离核',
            'config': GPSTConfig(
                kernel_design="separable",
                kernel_space="matern32",
                kernel_time="matern32",
                num_inducing=800,
                lr=0.01,
                num_epochs=30,
                batch_size=1000
            )
        },
        {
            'name': 'Design 2: 加性核',
            'config': GPSTConfig(
                kernel_design="additive",
                kernel_space="matern32",  # 用于 fallback
                kernel_time="matern32",   # 用于 fallback
                num_inducing=800,
                lr=0.01,
                num_epochs=30,
                batch_size=1000
            )
        },
        {
            'name': 'Design 3: 非分离核',
            'config': GPSTConfig(
                kernel_design="non_separable",
                kernel_space="matern32",
                kernel_time="matern32",
                num_inducing=800,
                lr=0.01,
                num_epochs=30,
                batch_size=1000
            )
        }
    ]
    
    # 存储结果
    results = {}
    
    print_section_header("训练和评估所有设计")
    
    # 对每种设计进行训练和评估
    for design in designs:
        print(f"\n{'='*80}")
        print(f"  处理: {design['name']}")
        print(f"{'='*80}")
        
        try:
            # 训练模型
            model, train_time = train_model(
                design['config'], 
                train_dataset, 
                device=device,
                verbose=True
            )
            
            print(f"\n训练完成，耗时: {train_time:.2f}秒")
            
            # 评估模型
            print("\n评估模型性能...")
            metrics = evaluate_model(model, test_dataset, device=device)
            
            # 存储结果
            results[design['name']] = {
                'config': design['config'],
                'train_time': train_time,
                'metrics': metrics,
                'model': model  # 保存模型（可选）
            }
            
            # 打印结果
            print(f"\n{design['name']} 评估结果:")
            print(f"  RMSE: {metrics['rmse']:.4f} K")
            print(f"  MAE: {metrics['mae']:.4f} K")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.4f} %")
            print(f"  CRPS: {metrics['crps']:.4f} K")
            print(f"  90% 预测区间覆盖率: {metrics['coverage']:.4f}")
            print(f"  平均区间宽度: {metrics['interval_width']:.4f} K")
            print(f"  训练时间: {train_time:.2f} 秒")
            
        except Exception as e:
            print(f"\n⚠️  设计 {design['name']} 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告
    print_section_header("生成对比报告")
    
    # 创建对比表格
    print("\n📊 性能对比表:")
    print("-" * 120)
    print(f"{'设计':<30} {'RMSE ↓':<12} {'MAE ↓':<12} {'R² ↑':<12} {'CRPS ↓':<12} {'Coverage':<12} {'训练时间':<12}")
    print("-" * 120)
    
    for design_name, result in results.items():
        m = result['metrics']
        print(f"{design_name:<30} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f} "
              f"{m['crps']:<12.4f} {m['coverage']:<12.4f} {result['train_time']:<12.2f}")
    
    print("-" * 120)
    
    # 保存结果到文件
    results_file = output_dir / "kernel_comparison_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("核函数设计对比实验结果\n")
        f.write("=" * 80 + "\n\n")
        
        for design_name, result in results.items():
            f.write(f"{design_name}\n")
            f.write("-" * 80 + "\n")
            m = result['metrics']
            f.write(f"RMSE: {m['rmse']:.4f} K\n")
            f.write(f"MAE: {m['mae']:.4f} K\n")
            f.write(f"R²: {m['r2']:.4f}\n")
            f.write(f"MAPE: {m['mape']:.4f} %\n")
            f.write(f"CRPS: {m['crps']:.4f} K\n")
            f.write(f"90% 预测区间覆盖率: {m['coverage']:.4f}\n")
            f.write(f"平均区间宽度: {m['interval_width']:.4f} K\n")
            f.write(f"训练时间: {result['train_time']:.2f} 秒\n\n")
    
    print(f"\n✅ 结果已保存到: {results_file}")
    
    # 生成对比可视化
    if len(results) > 0:
        print("\n生成对比可视化...")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            design_names = list(results.keys())
            rmse_values = [results[d]['metrics']['rmse'] for d in design_names]
            mae_values = [results[d]['metrics']['mae'] for d in design_names]
            r2_values = [results[d]['metrics']['r2'] for d in design_names]
            crps_values = [results[d]['metrics']['crps'] for d in design_names]
            
            # RMSE对比
            axes[0, 0].bar(design_names, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[0, 0].set_title('RMSE 对比 (越小越好)', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('RMSE (K)')
            axes[0, 0].tick_params(axis='x', rotation=15)
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # MAE对比
            axes[0, 1].bar(design_names, mae_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[0, 1].set_title('MAE 对比 (越小越好)', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('MAE (K)')
            axes[0, 1].tick_params(axis='x', rotation=15)
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            # R²对比
            axes[1, 0].bar(design_names, r2_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[1, 0].set_title('R² 对比 (越大越好)', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].tick_params(axis='x', rotation=15)
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # CRPS对比
            axes[1, 1].bar(design_names, crps_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[1, 1].set_title('CRPS 对比 (越小越好)', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('CRPS (K)')
            axes[1, 1].tick_params(axis='x', rotation=15)
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(output_dir / "kernel_designs_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 对比图已保存到: {output_dir / 'kernel_designs_comparison.png'}")
        except Exception as e:
            print(f"⚠️  生成可视化时出错: {str(e)}")
    
    print_section_header("实验完成")
    print("\n✅ 核函数设计对比实验已完成！")
    print(f"\n📁 结果保存在: {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="核函数设计对比实验")
    parser.add_argument("--data_path", type=str, default=None, help="MODIS数据路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    
    # 确保导入gpytorch（如果可用）
    try:
        import gpytorch
        compare_kernel_designs(
            data_path=args.data_path,
            output_dir=args.output_dir
        )
    except ImportError:
        print("⚠️  警告: 需要安装 gpytorch 才能运行此脚本")
        print("   请运行: pip install gpytorch")
        sys.exit(1)

