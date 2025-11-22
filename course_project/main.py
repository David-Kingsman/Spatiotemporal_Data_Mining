"""
主运行脚本 - 地表温度数据概率插值模型
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from itertools import product

from src.data_loader import MODISDataLoader
from src.gp_model import create_model
from src.trainer import GPTrainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer
from src.cross_validation import CrossValidator
from src.hyperparameter_tuner import HyperparameterTuner, BayesianOptimizer
from src.utils import load_config, ensure_dir


def train_cmd(args):
    """训练命令"""
    config = load_config(args.config)
    print("="*60)
    print("开始训练模型")
    print("="*60)
    print(f"使用配置: {args.config}")
    
    ensure_dir(Path(args.model_save_path).parent)
    
    print("\n1. 加载数据...")
    data_loader = MODISDataLoader(
        data_path=config['data']['data_path'],
        lat_range=tuple(config['data']['lat_range']),
        lon_range=tuple(config['data']['lon_range'])
    )
    data_loader.load_data()
    data_loader.get_tensor_stats(data_loader.training_tensor, "训练数据")
    
    train_data = data_loader.get_training_data()
    train_coords = train_data['coords']
    train_values = train_data['values']
    
    print(f"训练数据点数量: {len(train_values)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    train_x = torch.from_numpy(train_coords).float().to(device)
    train_y = torch.from_numpy(train_values).float().to(device)
    
    print("\n2. 创建模型...")
    inducing_points = None
    if config['model']['use_sparse']:
        num_inducing = config['model']['num_inducing']
        indices = torch.randperm(train_x.shape[0])[:num_inducing]
        inducing_points = train_x[indices].clone()
        print(f"使用稀疏GP，诱导点数量: {num_inducing}")
    
    model, likelihood = create_model(
        model_type="sparse" if config['model']['use_sparse'] else "exact",
        train_x=train_x,
        train_y=train_y,
        inducing_points=inducing_points,
        kernel_type=config['model']['kernel_type'],
        mean_type=config['model'].get('mean_type', 'constant'),
        use_sparse=config['model']['use_sparse'],
        num_inducing=config['model'].get('num_inducing', 500)
    )
    
    print(f"核函数类型: {config['model']['kernel_type']}")
    
    print("\n3. 开始训练...")
    trainer = GPTrainer(model, likelihood, device=device)
    trainer.train(
        train_x=train_x.cpu().numpy(),
        train_y=train_y.cpu().numpy(),
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        batch_size=config['training'].get('batch_size', None),
        early_stopping_patience=config['training'].get('early_stopping_patience', None),
        verbose=True
    )
    
    print("\n4. 保存模型...")
    trainer.save_model(args.model_save_path)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


def evaluate_cmd(args):
    """评估命令"""
    config = load_config(args.config)
    print("="*60)
    print("开始评估模型")
    print("="*60)
    
    print("\n1. 加载数据...")
    data_loader = MODISDataLoader(
        data_path=config['data']['data_path'],
        lat_range=tuple(config['data']['lat_range']),
        lon_range=tuple(config['data']['lon_range'])
    )
    data_loader.load_data()
    
    test_data = data_loader.get_test_data()
    test_coords = test_data['coords']
    test_values = test_data['values']
    
    print(f"测试数据点数量: {len(test_values)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_data = data_loader.get_training_data()
    train_coords = train_data['coords']
    train_values = train_data['values']
    
    train_x = torch.from_numpy(train_coords).float().to(device)
    train_y = torch.from_numpy(train_values).float().to(device)
    
    inducing_points = None
    if config['model']['use_sparse']:
        num_inducing = config['model']['num_inducing']
        indices = torch.randperm(train_x.shape[0])[:num_inducing]
        inducing_points = train_x[indices].clone()
    
    model, likelihood = create_model(
        model_type="sparse" if config['model']['use_sparse'] else "exact",
        train_x=train_x,
        train_y=train_y,
        inducing_points=inducing_points,
        kernel_type=config['model']['kernel_type'],
        mean_type=config['model'].get('mean_type', 'constant'),
        use_sparse=config['model']['use_sparse'],
        num_inducing=config['model'].get('num_inducing', 500)
    )
    
    print("\n2. 加载模型...")
    trainer = GPTrainer(model, likelihood, device=device)
    trainer.load_model(args.model_path)
    
    print("\n3. 评估模型...")
    evaluator = Evaluator(model, likelihood, device=device)
    
    compute_additional = config['evaluation'].get('compute_additional', True)
    results = evaluator.evaluate(
        test_coords, 
        test_values, 
        batch_size=config['training'].get('batch_size', 1000),
        compute_additional=compute_additional
    )
    
    evaluator.print_results(results)
    
    ensure_dir(Path(args.save_results).parent)
    results_to_save = {
        'rmse': float(results['rmse']),
        'r2': float(results['r2']),
        'crps': float(results['crps'])
    }
    
    with open(args.save_results, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存到: {args.save_results}")


def visualize_cmd(args):
    """可视化命令"""
    config = load_config(args.config)
    print("="*60)
    print("开始可视化结果")
    print("="*60)
    
    print("\n1. 加载数据...")
    data_loader = MODISDataLoader(
        data_path=config['data']['data_path'],
        lat_range=tuple(config['data']['lat_range']),
        lon_range=tuple(config['data']['lon_range'])
    )
    data_loader.load_data()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_data = data_loader.get_training_data()
    train_coords = train_data['coords']
    train_values = train_data['values']
    
    train_x = torch.from_numpy(train_coords).float().to(device)
    train_y = torch.from_numpy(train_values).float().to(device)
    
    inducing_points = None
    if config['model']['use_sparse']:
        num_inducing = config['model']['num_inducing']
        indices = torch.randperm(train_x.shape[0])[:num_inducing]
        inducing_points = train_x[indices].clone()
    
    model, likelihood = create_model(
        model_type="sparse" if config['model']['use_sparse'] else "exact",
        train_x=train_x,
        train_y=train_y,
        inducing_points=inducing_points,
        kernel_type=config['model']['kernel_type'],
        mean_type=config['model'].get('mean_type', 'constant'),
        use_sparse=config['model']['use_sparse'],
        num_inducing=config['model'].get('num_inducing', 500)
    )
    
    print("\n2. 加载模型...")
    trainer = GPTrainer(model, likelihood, device=device)
    trainer.load_model(args.model_path)
    
    print("\n3. 绘制训练曲线...")
    visualizer = Visualizer(save_dir=config['visualization']['save_dir'])
    visualizer.plot_training_curve(trainer.training_losses)
    
    print("\n4. 在测试数据上进行预测...")
    evaluator = Evaluator(model, likelihood, device=device)
    
    test_data = data_loader.get_test_data()
    test_coords = test_data['coords']
    test_values = test_data['values']
    
    y_mean, y_var, y_std = evaluator.predict(
        test_coords, 
        batch_size=config['training'].get('batch_size', 1000)
    )
    
    print("\n5. 绘制评估图表...")
    visualizer.plot_scatter_comparison(test_values, y_mean)
    visualizer.plot_residuals(test_values, y_mean)
    
    print("\n6. 绘制2D预测图...")
    for day_idx in args.days:
        if day_idx < 31:
            day_mask = (test_coords[:, 2] == day_idx + 1)
            if np.sum(day_mask) > 0:
                day_test_coords = test_coords[day_mask]
                day_test_values = test_values[day_mask]
                day_pred = y_mean[day_mask]
                day_std = y_std[day_mask]
                
                visualizer.plot_predictions_2d(
                    day_test_coords, day_test_values, day_pred, day_std,
                    day_idx=day_idx,
                    save_name=f"predictions_day_{day_idx+1}.png"
                )
                
                visualizer.plot_prediction_std(
                    day_test_coords, day_std,
                    day_idx=day_idx,
                    save_name=f"prediction_std_day_{day_idx+1}.png"
                )
    
    print("\n" + "="*60)
    print("可视化完成！")
    print("="*60)


def run_experiments_cmd(args):
    """运行实验命令"""
    config = load_config(args.config)
    ensure_dir(args.output_dir)
    
    print("="*60)
    print("开始运行完整实验")
    print("="*60)
    
    print("\n加载数据...")
    data_loader = MODISDataLoader(
        data_path=config['data']['data_path'],
        lat_range=tuple(config['data']['lat_range']),
        lon_range=tuple(config['data']['lon_range'])
    )
    data_loader.load_data()
    
    train_data = data_loader.get_training_data()
    test_data = data_loader.get_test_data()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    train_x = torch.from_numpy(train_data['coords']).float().to(device)
    train_y = torch.from_numpy(train_data['values']).float().to(device)
    test_x = torch.from_numpy(test_data['coords']).float().to(device)
    test_y = torch.from_numpy(test_data['values']).float().to(device)
    
    kernel_types = ['rbf', 'rq', 'periodic', 'matern', 'additive', 'product', 'composite', 'matern_additive']
    num_inducings = [300, 500, 800, 1000] if config['model']['use_sparse'] else [None]
    
    all_results = []
    best_result = None
    best_score = float('inf')
    
    def train_and_evaluate(kernel_type, num_inducing):
        print(f"\n{'='*60}")
        print(f"核类型: {kernel_type}, 诱导点数量: {num_inducing}")
        print(f"{'='*60}")
        
        inducing_points = None
        if config['model']['use_sparse']:
            indices = torch.randperm(train_x.shape[0])[:num_inducing]
            inducing_points = train_x[indices].clone()
        
        model, likelihood = create_model(
            model_type="sparse" if config['model']['use_sparse'] else "exact",
            train_x=train_x,
            train_y=train_y,
            inducing_points=inducing_points,
            kernel_type=kernel_type,
            use_sparse=config['model']['use_sparse'],
            num_inducing=num_inducing
        )
        
        trainer = GPTrainer(model, likelihood, device=device)
        trainer.train(
            train_x=train_x.cpu().numpy(),
            train_y=train_y.cpu().numpy(),
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            batch_size=config['training'].get('batch_size', None),
            early_stopping_patience=config['training'].get('early_stopping_patience', None),
            verbose=False
        )
        
        evaluator = Evaluator(model, likelihood, device=device)
        results = evaluator.evaluate(
            test_x.cpu().numpy(),
            test_y.cpu().numpy(),
            batch_size=config['training'].get('batch_size', 1000),
            compute_additional=True,
            extract_hyperparams=True  # 启用超参数提取
        )
        
        # 准备返回结果（包含超参数）
        result_dict = {
            'kernel_type': kernel_type,
            'num_inducing': num_inducing,
            **{k: float(v) for k, v in results.items() if isinstance(v, (int, float, np.number))},
            'trainer': trainer,
            'model': model,
            'likelihood': likelihood
        }
        
        # 添加超参数（如果提取了）
        if 'raw_hyperparams' in results:
            result_dict['raw_hyperparams'] = results['raw_hyperparams']
        
        return result_dict
    
    for kernel_type, num_inducing in product(kernel_types, num_inducings):
        if num_inducing is None and config['model']['use_sparse']:
            continue
            
        try:
            result = train_and_evaluate(
                kernel_type, 
                num_inducing or config['model']['num_inducing']
            )
            
            all_results.append({k: v for k, v in result.items() if k not in ['trainer', 'model', 'likelihood']})
            
            if result['rmse'] < best_score:
                best_score = result['rmse']
                best_result = result
                model_path = Path(args.output_dir) / 'best_model.pth'
                ensure_dir(model_path.parent)
                result['trainer'].save_model(str(model_path))
                
        except Exception as e:
            print(f"错误: {kernel_type}, {num_inducing} - {str(e)}")
            continue
    
    results_path = Path(args.output_dir) / 'all_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    kernel_best_results = {}
    for r in all_results:
        kernel = r['kernel_type']
        if kernel not in kernel_best_results or r['rmse'] < kernel_best_results[kernel]['rmse']:
            kernel_best_results[kernel] = r
    
    print("\n" + "="*80)
    print("实验结果总结 - Kernel Function Comparison")
    print("="*80)
    print(f"{'Scenario':<10} {'Kernel Type':<20} {'Num Inducing':<15} {'R²':<12} {'RMSE (K)':<12} {'CRPS (K)':<12}")
    print("-"*80)
    
    sorted_results = sorted(kernel_best_results.values(), key=lambda x: x['rmse'])
    for idx, r in enumerate(sorted_results, 1):
        num_ind = r['num_inducing'] if r['num_inducing'] is not None else 'Exact'
        print(f"{idx:<10} {r['kernel_type']:<20} {str(num_ind):<15} "
              f"{r['r2']:<12.4f} {r['rmse']:<12.4f} {r['crps']:<12.4f}")
    print("="*80)
    
    if best_result:
        print(f"\n最佳模型: {best_result['kernel_type']}, "
              f"诱导点数: {best_result['num_inducing']}")
        print(f"RMSE: {best_result['rmse']:.4f}, "
              f"R²: {best_result['r2']:.4f}, "
              f"CRPS: {best_result['crps']:.4f}")


def tune_cmd(args):
    """超参数优化命令"""
    config = load_config(args.config)
    ensure_dir(args.output_dir)
    
    print("="*60)
    print("超参数优化")
    print("="*60)
    print(f"优化方法: {args.method}")
    
    print("\n1. 加载数据...")
    data_loader = MODISDataLoader(
        data_path=config['data']['data_path'],
        lat_range=tuple(config['data']['lat_range']),
        lon_range=tuple(config['data']['lon_range'])
    )
    data_loader.load_data()
    
    train_data = data_loader.get_training_data()
    test_data = data_loader.get_test_data()
    
    train_x = train_data['coords']
    train_y = train_data['values']
    test_x = test_data['coords']
    test_y = test_data['values']
    
    print(f"训练数据: {len(train_y)} 个样本")
    print(f"测试数据: {len(test_y)} 个样本")
    
    # 清理GPU内存（如果有之前的进程残留）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\nGPU内存状态:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  已保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  总容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 配置说明（根据GPU内存调整）：
    # - use_cv=False: 使用验证集（推荐11GB GPU，内存占用减少80%，速度提升5倍）
    #   * 仍然可靠：有独立验证集(20%)和测试集，配合早停机制
    #   * 内存友好：每次评估只需1次训练，而不是5次
    # - use_cv=True: 使用交叉验证（需要>24GB GPU，最可靠但成本高）
    #   * 最可靠：5折CV提供更稳健的评估
    #   * 内存消耗：每次评估需要5次训练，内存占用约5GB
    # 详细分析请参考：docs/GPU_MEMORY_ANALYSIS.md
    use_cv = False  # 推荐：使用验证集（适配11GB GPU）
    cv_folds = 5  # 如果use_cv=True时使用
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    
    if args.method == 'grid':
        print("\n2. 网格搜索...")
        param_grid = {
            'num_inducing': [300, 500, 800, 1000],  # 增加1000选项，A100可以支持
            'mean_type': ['constant', 'linear']
        }
        # 测试所有核函数（包括可能不稳定的，全面评估）
        kernel_types = ['rbf', 'rq', 'periodic', 'matern', 'additive', 'product', 'composite', 'matern_additive']
        
        tuner = HyperparameterTuner(
            train_x, train_y, test_x, test_y,
            validation_split=0.2,
            use_cv=use_cv,
            cv_folds=cv_folds
        )
        best_params, best_score, all_results = tuner.grid_search(
            param_grid=param_grid,
            kernel_types=kernel_types,
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            use_sparse=config['model']['use_sparse'],
            metric='rmse',
            early_stopping_patience=early_stopping_patience,
            use_cv=use_cv,
            cv_folds=cv_folds,
            verbose=True
        )
    else:
        print("\n2. 贝叶斯优化...")
        # 优化参数范围（根据GPU内存调整，本地GPU约11GB，但已有系统进程占用约1GB）
        # 可用内存约10GB，考虑到内存碎片，保守设置为(300, 500)
        param_ranges = {
            'num_inducing': (300, 500),  # 进一步降低上限，适配实际可用内存
            'learning_rate': (0.001, 0.1)
        }
        # 测试所有核函数（包括可能不稳定的，全面评估）
        kernel_types = ['rbf', 'rq', 'periodic', 'matern', 'additive', 'product', 'composite', 'matern_additive']
        
        optimizer = BayesianOptimizer(
            train_x, train_y, test_x, test_y,
            validation_split=0.2,
            use_cv=use_cv,
            cv_folds=cv_folds
        )
        best_params, best_score, all_results = optimizer.optimize(
            param_ranges=param_ranges,
            kernel_types=kernel_types,
            n_iter=50,  # 贝叶斯优化迭代次数（50次获得更优结果，A100可以承受）
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            metric='rmse',
            early_stopping_patience=early_stopping_patience,
            use_cv=use_cv,
            cv_folds=cv_folds,
            verbose=True
        )
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    output_path = Path(args.output_dir) / f'{args.method}_results.json'
    results_to_save = {
        'best_params': convert_to_serializable(best_params),
        'best_score': float(best_score),
        'all_results': convert_to_serializable(all_results)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    print("\n" + "="*60)
    print("优化完成！")
    print("="*60)


def cross_validate_cmd(args):
    """交叉验证命令"""
    config = load_config(args.config)
    ensure_dir(args.output_dir)
    
    print("="*60)
    print("交叉验证")
    print("="*60)
    print(f"交叉验证类型: {args.cv_type}")
    print(f"折数: {args.n_splits}")
    
    print("\n1. 加载数据...")
    data_loader = MODISDataLoader(
        data_path=config['data']['data_path'],
        lat_range=tuple(config['data']['lat_range']),
        lon_range=tuple(config['data']['lon_range'])
    )
    data_loader.load_data()
    
    train_data = data_loader.get_training_data()
    X = train_data['coords']
    y = train_data['values']
    
    print(f"数据总量: {len(y)} 个样本")
    
    print(f"\n2. 执行{args.cv_type}交叉验证...")
    cv = CrossValidator(X, y)
    
    if args.cv_type == 'kfold':
        cv_results = cv.k_fold_cv(
            n_splits=args.n_splits,
            kernel_type=config['model']['kernel_type'],
            num_inducing=config['model']['num_inducing'],
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            use_sparse=config['model']['use_sparse'],
            verbose=True
        )
    else:
        cv_results = cv.time_series_cv(
            n_splits=args.n_splits,
            kernel_type=config['model']['kernel_type'],
            num_inducing=config['model']['num_inducing'],
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            use_sparse=config['model']['use_sparse'],
            verbose=True
        )
    
    output_path = Path(args.output_dir) / f'{args.cv_type}_cv_results.json'
    
    results_to_save = {
        'mean_rmse': float(cv_results['mean_rmse']),
        'std_rmse': float(cv_results['std_rmse']),
        'mean_r2': float(cv_results['mean_r2']),
        'std_r2': float(cv_results['std_r2']),
        'mean_crps': float(cv_results['mean_crps']),
        'std_crps': float(cv_results['std_crps']),
        'fold_results': []
    }
    
    if 'mean_mae' in cv_results:
        results_to_save['mean_mae'] = float(cv_results['mean_mae'])
        results_to_save['std_mae'] = float(cv_results['std_mae'])
    
    for fold_result in cv_results['fold_results']:
        results_to_save['fold_results'].append({
            'fold': fold_result['fold'],
            'train_size': fold_result['train_size'],
            'val_size': fold_result['val_size'],
            'rmse': float(fold_result['results']['rmse']),
            'r2': float(fold_result['results']['r2']),
            'crps': float(fold_result['results']['crps'])
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    print("\n" + "="*60)
    print("交叉验证完成！")
    print("="*60)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description='地表温度数据概率插值模型',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, default='config/config.yaml')
    train_parser.add_argument('--model_save_path', type=str, default='models/best_model.pth')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--config', type=str, default='config/config.yaml')
    eval_parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    eval_parser.add_argument('--save_results', type=str, default='results/evaluation_results.json')
    
    # 可视化命令
    vis_parser = subparsers.add_parser('visualize', help='可视化结果')
    vis_parser.add_argument('--config', type=str, default='config/config.yaml')
    vis_parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    vis_parser.add_argument('--days', type=int, nargs='+', default=[0, 10, 20])
    
    # 实验命令
    exp_parser = subparsers.add_parser('run_experiments', help='运行完整实验')
    exp_parser.add_argument('--config', type=str, default='config/config.yaml')
    exp_parser.add_argument('--output_dir', type=str, default='results/experiments')
    
    # 超参数优化命令
    tune_parser = subparsers.add_parser('tune', help='超参数优化（推荐使用bayesian方法）')
    tune_parser.add_argument('--config', type=str, default='config/config.yaml')
    tune_parser.add_argument('--method', type=str, default='bayesian', choices=['grid', 'bayesian'],
                            help='优化方法: grid=网格搜索(完整但慢), bayesian=贝叶斯优化(推荐,更高效)')
    tune_parser.add_argument('--output_dir', type=str, default='results/hyperparameter_tuning')
    
    # 交叉验证命令
    cv_parser = subparsers.add_parser('cross_validate', help='交叉验证')
    cv_parser.add_argument('--config', type=str, default='config/config.yaml')
    cv_parser.add_argument('--cv_type', type=str, default='kfold', choices=['kfold', 'timeseries'])
    cv_parser.add_argument('--n_splits', type=int, default=5)
    cv_parser.add_argument('--output_dir', type=str, default='results/cross_validation')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_cmd(args)
    elif args.command == 'evaluate':
        evaluate_cmd(args)
    elif args.command == 'visualize':
        visualize_cmd(args)
    elif args.command == 'run_experiments':
        run_experiments_cmd(args)
    elif args.command == 'tune':
        tune_cmd(args)
    elif args.command == 'cross_validate':
        cross_validate_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
