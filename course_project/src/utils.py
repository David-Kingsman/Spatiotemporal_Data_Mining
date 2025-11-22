"""
utility functions module
"""
import numpy as np
import torch
import yaml
from pathlib import Path


def load_config(config_path):
    """load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(dir_path):
    """ensure directory exists"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def to_numpy(tensor):
    """convert torch tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def to_tensor(array, device='cpu'):
    """convert numpy array to torch tensor"""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).float().to(device)
    return array


def normalize_data(data, mean=None, std=None):
    """normalize data"""
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
        std[std == 0] = 1.0  # 避免除以0
    
    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_data(normalized, mean, std):
    """denormalize data"""
    return normalized * std + mean


def extract_hyperparameters(model, likelihood):
    """
    提取模型超参数（简化方法，直接遍历所有参数）
    
    这是一个简化的超参数提取方法，不依赖于具体的kernel结构，
    直接遍历模型和似然函数的所有参数，查找包含关键词的参数。
    
    参数:
        model: GP模型
        likelihood: 似然函数
        
    返回:
        dict: 包含所有提取的超参数
            - lengthscale: 长度尺度参数（可能是标量或向量）
            - outputscale: 输出尺度参数
            - noise: 噪声参数
    """
    raw_hyperparams = {}
    
    # 从模型参数中提取
    for name, param in model.named_parameters():
        param_value = param.data.cpu().numpy()
        
        if 'lengthscale' in name.lower():
            if param_value.ndim == 0:
                raw_hyperparams[name] = float(param_value)
            else:
                raw_hyperparams[name] = param_value.tolist()
        elif 'outputscale' in name.lower():
            if param_value.ndim == 0:
                raw_hyperparams[name] = float(param_value)
            else:
                raw_hyperparams[name] = param_value.tolist()
    
    # 从似然函数参数中提取
    for name, param in likelihood.named_parameters():
        if 'noise' in name.lower():
            param_value = param.data.cpu().numpy()
            if param_value.ndim == 0:
                raw_hyperparams[name] = float(param_value)
            else:
                raw_hyperparams[name] = param_value.tolist()
    
    return raw_hyperparams

