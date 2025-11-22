"""
模型评估模块 - 扩展版本，包含更多评估指标
"""

import numpy as np
import torch
import gpytorch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from properscoring import crps_gaussian
from .utils import to_numpy, extract_hyperparameters


class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, likelihood, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化评估器
        
        参数:
            model: 训练好的GP模型
            likelihood: 似然函数
            device: 计算设备
        """
        self.model = model
        self.likelihood = likelihood
        self.device = device
        
        # 设置为评估模式
        self.model.eval()
        self.likelihood.eval()
        
        # 设置全局默认jitter值（提高数值稳定性）
        # 这会影响所有GP操作，确保在模型forward时也使用更大的jitter
        gpytorch.settings.cholesky_jitter._global_float_value = 1e-3
    
    def predict(self, test_x, batch_size=1000):
        """
        进行预测
        
        参数:
            test_x: 测试输入 (n, 3)
            batch_size: 批处理大小（用于大数据的预测）
            
        返回:
            mean: 预测均值 (n,)
            var: 预测方差 (n,)
            std: 预测标准差 (n,)
        """
        test_x = torch.from_numpy(test_x).float().to(self.device)
        
        with torch.no_grad():
            # 使用更大的jitter值提高数值稳定性（增加到1e-3以提高稳定性）
            with gpytorch.settings.cholesky_jitter(float_value=1e-3):
                if batch_size is None or batch_size >= test_x.shape[0]:
                    # 一次性预测
                    observed_pred = self.likelihood(self.model(test_x))
                    mean = observed_pred.mean
                    var = observed_pred.variance
                else:
                    # 批处理预测
                    n_samples = test_x.shape[0]
                    means = []
                    vars = []
                    
                    for i in range(0, n_samples, batch_size):
                        end_idx = min(i + batch_size, n_samples)
                        batch_x = test_x[i:end_idx]
                        
                        observed_pred = self.likelihood(self.model(batch_x))
                        means.append(observed_pred.mean)
                        vars.append(observed_pred.variance)
                    
                    mean = torch.cat(means, dim=0)
                    var = torch.cat(vars, dim=0)
            
            std = torch.sqrt(var)
        
        # 转换为numpy
        mean = to_numpy(mean)
        var = to_numpy(var)
        std = to_numpy(std)
        
        return mean, var, std
    
    def compute_rmse(self, y_true, y_pred):
        """计算RMSE"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse
    
    def compute_r2(self, y_true, y_pred):
        """计算R²"""
        r2 = r2_score(y_true, y_pred)
        return r2
    
    def compute_mae(self, y_true, y_pred):
        """计算平均绝对误差 (MAE)"""
        mae = mean_absolute_error(y_true, y_pred)
        return mae
    
    def compute_mape(self, y_true, y_pred):
        """计算平均绝对百分比误差 (MAPE)"""
        # 避免除以0
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    def compute_coverage(self, y_true, y_mean, y_std, confidence=0.95):
        """
        计算预测区间覆盖率
        
        参数:
            y_true: 真实值
            y_mean: 预测均值
            y_std: 预测标准差
            confidence: 置信水平 (0.95表示95%置信区间)
        """
        from scipy.stats import norm
        alpha = 1 - confidence
        z_score = norm.ppf(1 - alpha / 2)
        
        lower = y_mean - z_score * y_std
        upper = y_mean + z_score * y_std
        
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage
    
    def compute_picp(self, y_true, y_mean, y_std, coverage_level=0.95):
        """计算预测区间覆盖率 (PICP - Prediction Interval Coverage Probability)"""
        return self.compute_coverage(y_true, y_mean, y_std, confidence=coverage_level)
    
    def compute_crps(self, y_true, y_mean, y_std):
        """
        计算连续排名概率分数 (CRPS)
        
        参数:
            y_true: 真实值
            y_mean: 预测均值
            y_std: 预测标准差
        """
        # CRPS假设预测分布为正态分布
        crps = crps_gaussian(y_true, y_mean, y_std)
        return np.mean(crps)
    
    def evaluate(self, test_x, test_y, batch_size=1000, compute_additional=True, 
                 extract_hyperparams=False):
        """
        全面评估模型
        
        参数:
            test_x: 测试输入
            test_y: 测试真实值
            batch_size: 批处理大小
            compute_additional: 是否计算额外指标
            extract_hyperparams: 是否提取超参数
            
        返回:
            dict包含各种评估指标
        """
        # 进行预测
        y_mean, y_var, y_std = self.predict(test_x, batch_size)
        
        # 基本评估指标
        rmse = self.compute_rmse(test_y, y_mean)
        r2 = self.compute_r2(test_y, y_mean)
        crps = self.compute_crps(test_y, y_mean, y_std)
        
        results = {
            'rmse': rmse,
            'r2': r2,
            'crps': crps,
            'predictions': y_mean,
            'variances': y_var,
            'std': y_std
        }
        
        # 额外评估指标
        if compute_additional:
            mae = self.compute_mae(test_y, y_mean)
            mape = self.compute_mape(test_y, y_mean)
            coverage_95 = self.compute_coverage(test_y, y_mean, y_std, confidence=0.95)
            coverage_90 = self.compute_coverage(test_y, y_mean, y_std, confidence=0.90)
            
            results.update({
                'mae': mae,
                'mape': mape,
                'coverage_95': coverage_95,
                'coverage_90': coverage_90,
                'picp_95': coverage_95,
                'picp_90': coverage_90
            })
        
        # 提取超参数（如果需要）
        if extract_hyperparams:
            raw_hyperparams = extract_hyperparameters(self.model, self.likelihood)
            results['raw_hyperparams'] = raw_hyperparams
        
        return results
    
    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*60)
        print("模型评估结果")
        print("="*60)
        print(f"RMSE:  {results['rmse']:.4f} K")
        print(f"MAE:   {results.get('mae', 'N/A'):.4f} K" if 'mae' in results else "MAE:   N/A")
        print(f"MAPE:  {results.get('mape', 'N/A'):.4f} %" if 'mape' in results else "MAPE:  N/A")
        print(f"R²:    {results['r2']:.4f}")
        print(f"CRPS:  {results['crps']:.4f} K")
        if 'coverage_95' in results:
            print(f"95%覆盖率: {results['coverage_95']:.4f}")
        if 'coverage_90' in results:
            print(f"90%覆盖率: {results['coverage_90']:.4f}")
        print("="*60 + "\n")

