"""Hyperparameter tuning utilities"""
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path


@dataclass
class HyperparameterSpace:
    """Hyperparameter space definition"""
    params: Dict[str, List[Any]] = field(default_factory=dict)
    
    def add_param(self, name: str, values: List[Any]):
        """Add hyperparameter"""
        self.params[name] = values
    
    def generate_grid(self) -> List[Dict[str, Any]]:
        """Generate grid search combinations"""
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
    Grid search for hyperparameters
    
    Args:
    model_factory: Function accepting param dict, returning model instance
    train_fn: Function accepting (model, X_train, y_train), returning trained model
    X_train, y_train: Training data
    X_val, y_val: Validation data
    param_space: Hyperparameter space
    metric_fn: Evaluation function accepting (y_true, y_pred), returning score
    metric_higher_better: Whether higher metric is better (default False, lower is better)
    verbose: Whether to print progress
    
    Returns:
    best_params: Best parameters
    best_score: Best score
    all_results: List of results for all combinations
    """
    combinations = param_space.generate_grid()
    n_combinations = len(combinations)
    
    best_score = float('-inf') if metric_higher_better else float('inf')
    best_params = None
    all_results = []
    
    if verbose:
        print(f"Grid Search: {n_combinations} combinations")
        print("=" * 60)
    
    for i, params in enumerate(combinations, 1):
        if verbose:
            print(f"\nCombination {i}/{n_combinations}: {params}")
        
        try:
            # Create model
            model = model_factory(params)
            
            # Train model
            model = train_fn(model, X_train, y_train)
            
            # Predict
            if hasattr(model, 'predict_with_uncertainty'):
                y_pred, _ = model.predict_with_uncertainty(X_val)
            elif hasattr(model, 'predict'):
                y_pred = model.predict(X_val)
            else:
                # For PyTorch models
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_val)
                    if hasattr(model, 'likelihood'):
                        # GP model
                        output = model.gp(X_tensor)
                        pred_dist = model.likelihood(output)
                        y_pred = pred_dist.mean.cpu().numpy()
                    else:
                        # U-Net or other models
                        output = model(X_tensor)
                        if isinstance(output, tuple):
                            y_pred = output[0].cpu().numpy()
                        else:
                            y_pred = output.cpu().numpy()
            
            # Evaluate
            score = metric_fn(y_val, y_pred)
            
            # Update best result
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
                print(f"  Score ({metric_name}): {score:.4f}")
                if is_better:
                    print(f"  ✅ New best score!")
        
        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {e}")
            all_results.append({
                'params': params.copy(),
                'score': None,
                'error': str(e)
            })
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Best params: {best_params}")
        print(f"Best score: {best_score:.4f}")
    
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
    Random search for hyperparameters
    
    Args:
    n_iter: Number of iterations for random search
    random_state: Random seed
    Other parameters same as grid_search
    
    Returns:
    best_params: Best parameters
    best_score: Best score
    all_results: List of results for all attempts
    """
    rng = np.random.RandomState(random_state)
    
    best_score = float('-inf') if metric_higher_better else float('inf')
    best_params = None
    all_results = []
    
    if verbose:
        print(f"Random Search: {n_iter} iterations")
        print("=" * 60)
    
    for i in range(n_iter):
        # Randomly select parameters
        params = {}
        for param_name, param_values in param_space.params.items():
            params[param_name] = rng.choice(param_values)
        
        if verbose:
            print(f"\nIteration {i+1}/{n_iter}: {params}")
        
        try:
            # Create and train model
            model = model_factory(params)
            model = train_fn(model, X_train, y_train)
            
            # Predict
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
            
            # Evaluate
            score = metric_fn(y_val, y_pred)
            
            # Update best result
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
                print(f"  Score ({metric_name}): {score:.4f}")
                if is_better:
                    print(f"  ✅ New best score!")
        
        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {e}")
            all_results.append({
                'params': params.copy(),
                'score': None,
                'error': str(e),
                'iteration': i + 1
            })
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Best params: {best_params}")
        print(f"Best score: {best_score:.4f}")
    
    return best_params, best_score, all_results


def save_search_results(
    results: List[Dict],
    best_params: Dict[str, Any],
    best_score: float,
    save_path: str
):
    """Save search results to JSON file"""
    output = {
        'best_params': best_params,
        'best_score': float(best_score),
        'all_results': results,
        'n_combinations': len(results)
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSearch results saved to: {save_path}")
