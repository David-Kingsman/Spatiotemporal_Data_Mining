"""
超参数敏感性分析

该脚本分析不同超参数设置对模型性能的影响：
1. U-Net: 学习率、批量大小、基础通道数
2. GP: 学习率、诱导点数、长度尺度
3. Tree: 树的数量、最大深度
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from itertools import product

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import ProbUNet, TreeBaseline
from lstinterp.config import UNetConfig, TreeConfig
from lstinterp.metrics import rmse, r2, crps_gaussian
from lstinterp.utils import set_seed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 输出目录
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "hyperparameter_sensitivity"
RESULTS_DIR = OUTPUT_DIR / "results" / "hyperparameter_sensitivity"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 数据路径
DATA_PATH = project_root / "modis_aug_data" / "MODIS_Aug.mat"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

set_seed(42)


def print_section_header(title):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_unet_hyperparameters():
    """分析U-Net超参数敏感性"""
    print_section_header("U-Net超参数敏感性分析")
    
    # 加载数据
    train_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    
    H, W, T = train_tensor.shape
    
    # 计算归一化参数
    train_mask = (train_tensor != 0)
    train_mean = train_tensor[train_mask].mean()
    train_std = train_tensor[train_mask].std()
    
    # 创建数据集（使用较小的子集以加快速度）
    train_indices = list(range(28))  # 使用28天训练
    val_indices = list(range(28, 31))  # 使用3天验证
    test_indices = list(range(T))
    
    train_dataset = MODISDataset(train_tensor, mode="image", 
                                norm_mean=train_mean, norm_std=train_std)
    test_dataset = MODISDataset(test_tensor, mode="image",
                               norm_mean=train_mean, norm_std=train_std)
    
    # 超参数搜索空间（简化版，只测试关键参数）
    hyperparameter_space = {
        'lr': [0.0001, 0.0005, 0.001],
        'batch_size': [2, 4, 8],
        'base_channels': [16, 32, 64]
    }
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"超参数组合数: {len(list(product(*hyperparameter_space.values())))}")
    print("\n开始超参数搜索...")
    
    for lr, batch_size, base_channels in product(
        hyperparameter_space['lr'],
        hyperparameter_space['batch_size'],
        hyperparameter_space['base_channels']
    ):
        print(f"\n测试: lr={lr}, batch_size={batch_size}, base_channels={base_channels}")
        
        try:
            # 创建模型
            config = UNetConfig(
                in_channels=2,
                base_channels=base_channels,
                lr=lr,
                num_epochs=10,  # 减少epoch数以加快速度
                batch_size=batch_size
            )
            # 添加dropout属性（如果不存在）
            if not hasattr(config, 'dropout'):
                config.dropout = 0.2
            
            model = ProbUNet(config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            
            # 训练（简化版，只训练几个epoch）
            model.train()
            train_loader = DataLoader(
                [train_dataset[i] for i in train_indices],
                batch_size=batch_size,
                shuffle=True
            )
            
            train_losses = []
            for epoch in range(5):  # 只训练5个epoch
                epoch_loss = 0
                n_batches = 0
                
                for img, mask, target in train_loader:
                    img = img.to(device)
                    mask = mask.to(device)
                    target = target.to(device)
                    x = torch.cat([img, mask], dim=1)
                    
                    optimizer.zero_grad()
                    mean, log_var = model(x)
                    
                    # 计算损失
                    var = torch.exp(torch.clamp(log_var, -10, 10)) + 1e-6
                    nll = 0.5 * (torch.log(2 * np.pi * var) + (target - mean)**2 / var)
                    loss = (nll * mask).sum() / mask.sum().clamp_min(1.0)
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_batches += 1
                
                if n_batches > 0:
                    avg_loss = epoch_loss / n_batches
                    train_losses.append(avg_loss)
            
            # 评估（在测试集的一个子集上）
            model.eval()
            test_subset = [test_dataset[i] for i in test_indices[:5]]  # 只用5天
            
            pred_means = []
            pred_stds = []
            targets = []
            
            with torch.no_grad():
                for img, mask, target in test_subset:
                    img = img.unsqueeze(0).to(device)
                    mask_np = mask.squeeze().numpy()
                    x = torch.cat([img, mask.unsqueeze(0).to(device)], dim=1)
                    
                    mean, log_var = model(x)
                    std = torch.sqrt(torch.exp(torch.clamp(log_var, -10, 10)) + 1e-6)
                    
                    mean_np = mean.squeeze().cpu().numpy() * train_std + train_mean
                    std_np = std.squeeze().cpu().numpy() * train_std
                    target_np = target.squeeze().numpy() * train_std + train_mean
                    
                    # 只在观测点评估
                    mask_2d = mask_np > 0
                    pred_means.append(mean_np[mask_2d])
                    pred_stds.append(std_np[mask_2d])
                    targets.append(target_np[mask_2d])
            
            # 计算指标
            y_true = np.concatenate(targets)
            y_pred = np.concatenate(pred_means)
            y_std = np.concatenate(pred_stds)
            
            test_rmse = rmse(y_true, y_pred)
            test_r2 = r2(y_true, y_pred)
            test_crps = crps_gaussian(y_true, y_pred, y_std)
            
            results.append({
                'lr': lr,
                'batch_size': batch_size,
                'base_channels': base_channels,
                'rmse': test_rmse,
                'r2': test_r2,
                'crps': test_crps,
                'final_loss': train_losses[-1] if train_losses else np.nan
            })
            
            print(f"  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, CRPS: {test_crps:.4f}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            continue
    
    # 保存结果
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "unet_hyperparameter_sensitivity.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ 结果已保存: {csv_path}")
    
    # 可视化
    if len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 学习率影响
        lr_df = df.groupby('lr').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        axes[0, 0].plot(lr_df['lr'], lr_df['rmse'], 'o-', label='RMSE', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('RMSE (K)')
        axes[0, 0].set_title('Effect of Learning Rate', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        
        # 批量大小影响
        bs_df = df.groupby('batch_size').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        axes[0, 1].plot(bs_df['batch_size'], bs_df['rmse'], 's-', label='RMSE', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('RMSE (K)')
        axes[0, 1].set_title('Effect of Batch Size', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 基础通道数影响
        ch_df = df.groupby('base_channels').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        axes[1, 0].plot(ch_df['base_channels'], ch_df['rmse'], '^-', label='RMSE', linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xlabel('Base Channels')
        axes[1, 0].set_ylabel('RMSE (K)')
        axes[1, 0].set_title('Effect of Base Channels', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 热图：学习率 vs 批量大小
        pivot = df.pivot_table(values='rmse', index='lr', columns='batch_size', aggfunc='mean')
        im = axes[1, 1].imshow(pivot.values, aspect='auto', cmap='viridis_r', origin='lower')
        axes[1, 1].set_xticks(range(len(pivot.columns)))
        axes[1, 1].set_xticklabels(pivot.columns)
        axes[1, 1].set_yticks(range(len(pivot.index)))
        axes[1, 1].set_yticklabels([f"{x:.4f}" for x in pivot.index])
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('RMSE Heatmap: LR vs Batch Size', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1], label='RMSE (K)')
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "unet_hyperparameter_sensitivity.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 可视化已保存: {fig_path}")
    
    return df


def analyze_tree_hyperparameters():
    """分析Tree超参数敏感性"""
    print_section_header("Tree超参数敏感性分析")
    
    # 加载数据
    train_tensor = load_modis_tensor(str(DATA_PATH), key="training_tensor")
    test_tensor = load_modis_tensor(str(DATA_PATH), key="test_tensor")
    
    # 创建数据集
    train_dataset = MODISDataset(train_tensor, mode="point", normalize_coords=True)
    test_dataset = MODISDataset(test_tensor, mode="point", normalize_coords=True)
    
    # 采样以加快速度
    train_size = min(50000, len(train_dataset))
    test_size = min(10000, len(test_dataset))
    
    train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_size, replace=False)
    
    X_train = train_dataset.coords[train_indices]
    y_train = train_dataset.values[train_indices]
    X_test = test_dataset.coords[test_indices]
    y_test = test_dataset.values[test_indices]
    
    # 超参数搜索空间
    hyperparameter_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8, None]
    }
    
    results = []
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"超参数组合数: {len(list(product(*hyperparameter_space.values())))}")
    print("\n开始超参数搜索...")
    
    for n_estimators, max_depth in product(
        hyperparameter_space['n_estimators'],
        hyperparameter_space['max_depth']
    ):
        print(f"\n测试: n_estimators={n_estimators}, max_depth={max_depth}")
        
        try:
            config = TreeConfig(
                model_type="xgb",
                n_estimators=n_estimators,
                max_depth=max_depth,
                quantile_regression=True
            )
            # quantiles会在TreeBaseline.__init__中自动设置
            
            model = TreeBaseline(config)
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred_mean, y_pred_std = model.predict_with_uncertainty(X_test)
            
            test_rmse = rmse(y_test, y_pred_mean)
            test_r2 = r2(y_test, y_pred_mean)
            test_crps = crps_gaussian(y_test, y_pred_mean, y_pred_std)
            
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth if max_depth else -1,  # -1表示None
                'rmse': test_rmse,
                'r2': test_r2,
                'crps': test_crps,
                'training_time': training_time
            })
            
            print(f"  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, CRPS: {test_crps:.4f}, Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            continue
    
    # 保存结果
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "tree_hyperparameter_sensitivity.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ 结果已保存: {csv_path}")
    
    # 可视化
    if len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 树数量影响
        n_est_df = df.groupby('n_estimators').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        axes[0, 0].plot(n_est_df['n_estimators'], n_est_df['rmse'], 'o-', label='RMSE', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Estimators')
        axes[0, 0].set_ylabel('RMSE (K)')
        axes[0, 0].set_title('Effect of Number of Estimators', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 最大深度影响
        depth_df = df[df['max_depth'] > 0].groupby('max_depth').agg({'rmse': 'mean', 'r2': 'mean', 'crps': 'mean'}).reset_index()
        if len(depth_df) > 0:
            axes[0, 1].plot(depth_df['max_depth'], depth_df['rmse'], 's-', label='RMSE', linewidth=2, markersize=8, color='green')
            axes[0, 1].set_xlabel('Max Depth')
            axes[0, 1].set_ylabel('RMSE (K)')
            axes[0, 1].set_title('Effect of Max Depth', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 训练时间 vs 性能
        axes[1, 0].scatter(df['training_time'], df['rmse'], c=df['n_estimators'], 
                          cmap='viridis', s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Training Time (s)')
        axes[1, 0].set_ylabel('RMSE (K)')
        axes[1, 0].set_title('Training Time vs Performance', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
        cbar.set_label('n_estimators')
        
        # 热图：n_estimators vs max_depth
        pivot = df.pivot_table(values='rmse', index='n_estimators', columns='max_depth', aggfunc='mean')
        im = axes[1, 1].imshow(pivot.values, aspect='auto', cmap='viridis_r', origin='lower')
        axes[1, 1].set_xticks(range(len(pivot.columns)))
        axes[1, 1].set_xticklabels([str(x) if x > 0 else 'None' for x in pivot.columns])
        axes[1, 1].set_yticks(range(len(pivot.index)))
        axes[1, 1].set_yticklabels(pivot.index)
        axes[1, 1].set_xlabel('Max Depth')
        axes[1, 1].set_ylabel('Number of Estimators')
        axes[1, 1].set_title('RMSE Heatmap: n_estimators vs max_depth', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1], label='RMSE (K)')
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "tree_hyperparameter_sensitivity.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 可视化已保存: {fig_path}")
    
    return df


def main():
    """主函数"""
    print("=" * 80)
    print("  超参数敏感性分析")
    print("=" * 80)
    
    print("\n注意: 为了加快分析速度，使用简化的训练设置")
    print("      (U-Net: 5 epochs, Tree: 采样数据)")
    
    # 分析U-Net（如果时间允许）
    print("\n" + "=" * 80)
    print("  开始U-Net超参数分析（可能需要较长时间）...")
    print("=" * 80)
    try:
        unet_df = analyze_unet_hyperparameters()
    except Exception as e:
        print(f"⚠️  U-Net分析出错: {e}")
        unet_df = None
    
    # 分析Tree
    print("\n" + "=" * 80)
    print("  开始Tree超参数分析...")
    print("=" * 80)
    try:
        tree_df = analyze_tree_hyperparameters()
    except Exception as e:
        print(f"⚠️  Tree分析出错: {e}")
        tree_df = None
    
    print("\n" + "=" * 80)
    print("  超参数敏感性分析完成")
    print("=" * 80)
    print(f"\n所有结果已保存到:")
    print(f"  - 图表: {FIGURES_DIR}")
    print(f"  - 数据: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

