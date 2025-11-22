"""超参数调优工具"""
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path


@dataclass
class HyperparameterSpace:
    """超参数空间定义"""
    params: Dict[str, List[Any]] = field(default_factory=dict)
    
    def add_param(self, name: str, values: List[Any]):
        """添加超参数"""
        self.params[name] = values
    
    def generate_grid(self) -> List[Dict[str, Any]]:
        """生成网格搜索的组合"""
        if not self.params:
            return [{}]
        
        keys = list(self.params.keys())
        values = list(self.params.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations


def grid_search(
    model_factory: Callable,
    train_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_space: HyperparameterSpace,
    metric_fn: Callable,
    metric_higher_better: bool = False,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float, List[Dict]]:
    """
    网格搜索超参数
    
    参数:
    model_factory: 函数，接受参数字典，返回模型实例
    train_fn: 函数，接受(model, X_train, y_train)，返回训练好的模型
    X_train, y_train: 训练数据
    X_val, y_val: 验证数据
    param_space: 超参数空间
    metric_fn: 评估函数，接受(y_true, y_pred)，返回分数
    metric_higher_better: 指标是否越大越好（默认False，越小越好）
    verbose: 是否打印进度
    
    返回:
    best_params: 最佳参数
    best_score: 最佳分数
    all_results: 所有组合的结果列表
    """
    combinations = param_space.generate_grid()
    n_combinations = len(combinations)
    
    best_score = float('-inf') if metric_higher_better else float('inf')
    best_params = None
    all_results = []
    
    if verbose:
        print(f"网格搜索: {n_combinations} 个组合")
        print("=" * 60)
    
    for i, params in enumerate(combinations, 1):
        if verbose:
            print(f"\n组合 {i}/{n_combinations}: {params}")
        
        try:
            # 创建模型
            model = model_factory(params)
            
            # 训练模型
            model = train_fn(model, X_train, y_train)
            
            # 预测
            if hasattr(model, 'predict_with_uncertainty'):
                y_pred, _ = model.predict_with_uncertainty(X_val)
            elif hasattr(model, 'predict'):
                y_pred = model.predict(X_val)
            else:
                # 对于PyTorch模型
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_val)
                    if hasattr(model, 'likelihood'):
                        # GP模型
                        output = model.gp(X_tensor)
                        pred_dist = model.likelihood(output)
                        y_pred = pred_dist.mean.cpu().numpy()
                    else:
                        # U-Net等其他模型
                        output = model(X_tensor)
                        if isinstance(output, tuple):
                            y_pred = output[0].cpu().numpy()
                        else:
                            y_pred = output.cpu().numpy()
            
            # 评估
            score = metric_fn(y_val, y_pred)
            
            # 更新最佳结果
            is_better = (score > best_score) if metric_higher_better else (score < best_score)
            if is_better:
                best_score = score
                best_params = params.copy()
            
            result = {
                'params': params.copy(),
                'score': float(score)
            }
            all_results.append(result)
            
            if verbose:
                metric_name = getattr(metric_fn, '__name__', 'metric')
                print(f"  分数 ({metric_name}): {score:.4f}")
                if is_better:
                    print(f"  ✅ 新的最佳分数!")
        
        except Exception as e:
            if verbose:
                print(f"  ❌ 错误: {e}")
            all_results.append({
                'params': params.copy(),
                'score': None,
                'error': str(e)
            })
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"最佳参数: {best_params}")
        print(f"最佳分数: {best_score:.4f}")
    
    return best_params, best_score, all_results


def random_search(
    model_factory: Callable,
    train_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_space: HyperparameterSpace,
    metric_fn: Callable,
    n_iter: int = 20,
    metric_higher_better: bool = False,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float, List[Dict]]:
    """
    随机搜索超参数
    
    参数:
    n_iter: 随机搜索的迭代次数
    random_state: 随机种子
    其他参数同grid_search
    
    返回:
    best_params: 最佳参数
    best_score: 最佳分数
    all_results: 所有尝试的结果列表
    """
    rng = np.random.RandomState(random_state)
    
    best_score = float('-inf') if metric_higher_better else float('inf')
    best_params = None
    all_results = []
    
    if verbose:
        print(f"随机搜索: {n_iter} 次迭代")
        print("=" * 60)
    
    for i in range(n_iter):
        # 随机选择参数
        params = {}
        for param_name, param_values in param_space.params.items():
            params[param_name] = rng.choice(param_values)
        
        if verbose:
            print(f"\n迭代 {i+1}/{n_iter}: {params}")
        
        try:
            # 创建和训练模型
            model = model_factory(params)
            model = train_fn(model, X_train, y_train)
            
            # 预测
            if hasattr(model, 'predict_with_uncertainty'):
                y_pred, _ = model.predict_with_uncertainty(X_val)
            elif hasattr(model, 'predict'):
                y_pred = model.predict(X_val)
            else:
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_val)
                    if hasattr(model, 'likelihood'):
                        output = model.gp(X_tensor)
                        pred_dist = model.likelihood(output)
                        y_pred = pred_dist.mean.cpu().numpy()
                    else:
                        output = model(X_tensor)
                        if isinstance(output, tuple):
                            y_pred = output[0].cpu().numpy()
                        else:
                            y_pred = output.cpu().numpy()
            
            # 评估
            score = metric_fn(y_val, y_pred)
            
            # 更新最佳结果
            is_better = (score > best_score) if metric_higher_better else (score < best_score)
            if is_better:
                best_score = score
                best_params = params.copy()
            
            result = {
                'params': params.copy(),
                'score': float(score),
                'iteration': i + 1
            }
            all_results.append(result)
            
            if verbose:
                metric_name = getattr(metric_fn, '__name__', 'metric')
                print(f"  分数 ({metric_name}): {score:.4f}")
                if is_better:
                    print(f"  ✅ 新的最佳分数!")
        
        except Exception as e:
            if verbose:
                print(f"  ❌ 错误: {e}")
            all_results.append({
                'params': params.copy(),
                'score': None,
                'error': str(e),
                'iteration': i + 1
            })
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"最佳参数: {best_params}")
        print(f"最佳分数: {best_score:.4f}")
    
    return best_params, best_score, all_results


def save_search_results(
    results: List[Dict],
    best_params: Dict[str, Any],
    best_score: float,
    save_path: str
):
    """保存搜索结果到JSON文件"""
    output = {
        'best_params': best_params,
        'best_score': float(best_score),
        'all_results': results,
        'n_combinations': len(results)
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n搜索结果已保存到: {save_path}")

