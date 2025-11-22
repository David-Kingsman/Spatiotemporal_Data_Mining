"""交叉验证工具"""
import numpy as np
from typing import List, Tuple, Literal
from dataclasses import dataclass


@dataclass
class CVSplit:
    """交叉验证分割结果"""
    train_indices: np.ndarray
    test_indices: np.ndarray


def k_fold_cv_split(
    tensor: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> List[CVSplit]:
    """
    标准k折交叉验证（k-fold Cross-Validation）
    
    随机划分数据点，每个点都有同等机会被分配到任一折。
    这是最常用的交叉验证方法。
    
    参数:
    tensor: (H, W, T) - 3D张量
    n_splits: 交叉验证折数
    random_state: 随机种子
    
    返回:
    List[CVSplit]: 每折的训练/测试索引列表
    """
    H, W, T = tensor.shape
    
    # 找到所有观测点的3D索引
    mask = (tensor != 0.0)
    all_indices_3d = np.where(mask)
    n_samples = len(all_indices_3d[0])
    
    # 随机打乱索引
    rng = np.random.RandomState(random_state)
    shuffled_indices = np.arange(n_samples)
    rng.shuffle(shuffled_indices)
    
    # 将数据分成n_splits折
    fold_size = n_samples // n_splits
    splits = []
    
    for i in range(n_splits):
        # 计算当前折的范围
        test_start = i * fold_size
        if i == n_splits - 1:
            # 最后一折包含所有剩余的数据
            test_end = n_samples
        else:
            test_end = (i + 1) * fold_size
        
        # 测试集索引（当前折）
        test_indices_shuffled = shuffled_indices[test_start:test_end]
        
        # 训练集索引（其他所有折）
        train_indices_shuffled = np.concatenate([
            shuffled_indices[:test_start],
            shuffled_indices[test_end:]
        ])
        
        # 转换为原始3D索引
        test_indices_3d = tuple(
            arr[test_indices_shuffled] for arr in all_indices_3d
        )
        train_indices_3d = tuple(
            arr[train_indices_shuffled] for arr in all_indices_3d
        )
        
        splits.append(CVSplit(
            train_indices=train_indices_3d,
            test_indices=test_indices_3d
        ))
    
    return splits


def time_block_cv_split(
    tensor: np.ndarray,
    n_splits: int = 5,
    test_size: int = 3
) -> List[CVSplit]:
    """
    时间块交叉验证（Time-block CV）
    
    将时间维度分成多个块，每次留出一个块作为测试集。
    适用于评估模型在"新的一天"上的插值能力。
    
    参数:
    tensor: (H, W, T) - 3D张量
    n_splits: 交叉验证折数
    test_size: 每个测试块包含的天数
    
    返回:
    List[CVSplit]: 每折的训练/测试索引列表
    """
    H, W, T = tensor.shape
    
    # 将31天分成n_splits个块
    days_per_split = T // n_splits
    remaining_days = T % n_splits
    
    splits = []
    
    for i in range(n_splits):
        # 计算测试集的时间范围
        test_start = i * days_per_split
        test_end = test_start + test_size
        
        # 如果最后一个折，包含剩余的天数
        if i == n_splits - 1:
            test_end = T
        
        # 确保不超出范围
        test_end = min(test_end, T)
        
        # 测试集的时间索引
        test_time_indices = np.arange(test_start, test_end)
        
        # 训练集的时间索引（所有其他天）
        train_time_indices = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, T)
        ])
        
        # 提取训练集和测试集的索引
        # 对于点模式：找到所有观测点的3D索引
        train_mask = np.zeros((H, W, T), dtype=bool)
        test_mask = np.zeros((H, W, T), dtype=bool)
        
        for t in train_time_indices:
            train_mask[:, :, t] = (tensor[:, :, t] != 0.0)
        for t in test_time_indices:
            test_mask[:, :, t] = (tensor[:, :, t] != 0.0)
        
        # 转换为线性索引（用于点模式）
        train_indices_3d = np.where(train_mask)
        test_indices_3d = np.where(test_mask)
        
        # 如果是点模式，需要展平索引
        # 这里返回3D索引，让调用者决定如何使用
        splits.append(CVSplit(
            train_indices=train_indices_3d,
            test_indices=test_indices_3d
        ))
    
    return splits


def space_block_cv_split(
    tensor: np.ndarray,
    n_splits: int = 5,
    block_size: Tuple[int, int] = None,
    strategy: str = "grid"
) -> List[CVSplit]:
    """
    空间块交叉验证（Space-block CV）
    
    将空间维度分成多个块，每次留出一个块作为测试集。
    适用于评估模型在"新区域"上的插值能力。
    
    参数:
    tensor: (H, W, T) - 3D张量
    n_splits: 交叉验证折数
    block_size: (h_block, w_block) - 每个块的空间大小，如果为None则自动计算
    strategy: "grid" (网格分割) 或 "horizontal" (水平分割) 或 "vertical" (垂直分割)
    
    返回:
    List[CVSplit]: 每折的训练/测试索引列表
    """
    H, W, T = tensor.shape
    
    splits = []
    
    if strategy == "horizontal":
        # 水平分割：将空间按纬度（H）方向分成n_splits块
        h_block = H // n_splits
        remaining_h = H % n_splits
        
        for i in range(n_splits):
            h_start = i * h_block
            if i == n_splits - 1:
                # 最后一个块包含剩余的行
                h_end = H
            else:
                h_end = h_start + h_block
            
            # 测试集的空间范围（整个宽度）
            test_spatial_mask = np.zeros((H, W), dtype=bool)
            test_spatial_mask[h_start:h_end, :] = True
            
            # 训练集的空间范围（其他所有区域）
            train_spatial_mask = ~test_spatial_mask
            
            # 提取训练集和测试集的索引（所有时间）
            train_mask = np.zeros((H, W, T), dtype=bool)
            test_mask = np.zeros((H, W, T), dtype=bool)
            
            for t in range(T):
                train_mask[:, :, t] = train_spatial_mask & (tensor[:, :, t] != 0.0)
                test_mask[:, :, t] = test_spatial_mask & (tensor[:, :, t] != 0.0)
            
            # 转换为3D索引
            train_indices_3d = np.where(train_mask)
            test_indices_3d = np.where(test_mask)
            
            splits.append(CVSplit(
                train_indices=train_indices_3d,
                test_indices=test_indices_3d
            ))
    
    elif strategy == "vertical":
        # 垂直分割：将空间按经度（W）方向分成n_splits块
        w_block = W // n_splits
        remaining_w = W % n_splits
        
        for i in range(n_splits):
            w_start = i * w_block
            if i == n_splits - 1:
                # 最后一个块包含剩余的列
                w_end = W
            else:
                w_end = w_start + w_block
            
            # 测试集的空间范围（整个高度）
            test_spatial_mask = np.zeros((H, W), dtype=bool)
            test_spatial_mask[:, w_start:w_end] = True
            
            # 训练集的空间范围（其他所有区域）
            train_spatial_mask = ~test_spatial_mask
            
            # 提取训练集和测试集的索引（所有时间）
            train_mask = np.zeros((H, W, T), dtype=bool)
            test_mask = np.zeros((H, W, T), dtype=bool)
            
            for t in range(T):
                train_mask[:, :, t] = train_spatial_mask & (tensor[:, :, t] != 0.0)
                test_mask[:, :, t] = test_spatial_mask & (tensor[:, :, t] != 0.0)
            
            # 转换为3D索引
            train_indices_3d = np.where(train_mask)
            test_indices_3d = np.where(test_mask)
            
            splits.append(CVSplit(
                train_indices=train_indices_3d,
                test_indices=test_indices_3d
            ))
    
    else:  # strategy == "grid"
        # 网格分割：尝试分割成接近n_splits的网格
        # 计算最接近的网格布局
        sqrt_n = int(np.sqrt(n_splits))
        
        # 尝试不同的网格布局
        best_layout = None
        best_diff = float('inf')
        
        for n_h in range(1, n_splits + 1):
            n_w = (n_splits + n_h - 1) // n_h  # 向上取整
            total = n_h * n_w
            diff = abs(total - n_splits)
            
            if diff < best_diff and n_h <= H and n_w <= W:
                best_diff = diff
                best_layout = (n_h, n_w)
        
        if best_layout is None:
            # 如果找不到合适的网格，使用水平分割
            n_h, n_w = n_splits, 1
        else:
            n_h, n_w = best_layout
        
        # 计算块大小
        if block_size is None:
            h_block = H // n_h
            w_block = W // n_w
        else:
            h_block, w_block = block_size
        
        # 计算实际块的数量
        n_h_blocks = H // h_block if h_block > 0 else 1
        n_w_blocks = W // w_block if w_block > 0 else 1
        total_blocks = n_h_blocks * n_w_blocks
        
        # 确保不超过n_splits
        actual_splits = min(n_splits, total_blocks)
        
        for block_idx in range(actual_splits):
            # 计算当前块的空间范围
            h_idx = block_idx // n_w_blocks
            w_idx = block_idx % n_w_blocks
            
            h_start = h_idx * h_block
            h_end = min(h_start + h_block, H)
            w_start = w_idx * w_block
            w_end = min(w_start + w_block, W)
            
            # 测试集的空间范围
            test_spatial_mask = np.zeros((H, W), dtype=bool)
            test_spatial_mask[h_start:h_end, w_start:w_end] = True
            
            # 训练集的空间范围（其他所有区域）
            train_spatial_mask = ~test_spatial_mask
            
            # 提取训练集和测试集的索引（所有时间）
            train_mask = np.zeros((H, W, T), dtype=bool)
            test_mask = np.zeros((H, W, T), dtype=bool)
            
            for t in range(T):
                train_mask[:, :, t] = train_spatial_mask & (tensor[:, :, t] != 0.0)
                test_mask[:, :, t] = test_spatial_mask & (tensor[:, :, t] != 0.0)
            
            # 转换为3D索引
            train_indices_3d = np.where(train_mask)
            test_indices_3d = np.where(test_mask)
            
            splits.append(CVSplit(
                train_indices=train_indices_3d,
                test_indices=test_indices_3d
            ))
    
    return splits


def time_space_block_cv_split(
    tensor: np.ndarray,
    time_n_splits: int = 5,
    space_n_splits: int = 5
) -> List[CVSplit]:
    """
    时空块交叉验证（Time-Space-block CV）
    
    同时留出时间和空间块作为测试集。
    
    参数:
    tensor: (H, W, T) - 3D张量
    time_n_splits: 时间维度分割数
    space_n_splits: 空间维度分割数
    
    返回:
    List[CVSplit]: 每折的训练/测试索引列表
    """
    H, W, T = tensor.shape
    
    splits = []
    
    # 时间和空间的块大小
    time_block_size = T // time_n_splits
    h_block = H // int(np.sqrt(space_n_splits))
    w_block = W // int(np.sqrt(space_n_splits))
    n_h_blocks = H // h_block
    n_w_blocks = W // w_block
    
    for t_split in range(time_n_splits):
        for s_split in range(min(space_n_splits, n_h_blocks * n_w_blocks)):
            # 时间范围
            t_start = t_split * time_block_size
            t_end = min(t_start + time_block_size, T)
            
            # 空间范围
            h_idx = s_split // n_w_blocks
            w_idx = s_split % n_w_blocks
            h_start = h_idx * h_block
            h_end = min(h_start + h_block, H)
            w_start = w_idx * w_block
            w_end = min(w_start + w_block, W)
            
            # 测试集掩码
            test_mask = np.zeros((H, W, T), dtype=bool)
            test_mask[h_start:h_end, w_start:w_end, t_start:t_end] = True
            test_mask = test_mask & (tensor != 0.0)
            
            # 训练集掩码（其他所有区域和时间）
            train_mask = (tensor != 0.0) & ~test_mask
            
            # 转换为3D索引
            train_indices_3d = np.where(train_mask)
            test_indices_3d = np.where(test_mask)
            
            splits.append(CVSplit(
                train_indices=train_indices_3d,
                test_indices=test_indices_3d
            ))
    
    return splits


def extract_data_from_indices(
    tensor: np.ndarray,
    indices_3d: Tuple[np.ndarray, np.ndarray, np.ndarray],
    mode: Literal["point", "image"] = "point"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从3D索引提取数据
    
    参数:
    tensor: (H, W, T) - 3D张量
    indices_3d: (lat_indices, lon_indices, time_indices) - 3D索引元组
    mode: "point" 或 "image"
    
    返回:
    X: 特征矩阵 (N, 3) 或图像 (N, H, W)
    y: 目标值 (N,)
    """
    lat_indices, lon_indices, time_indices = indices_3d
    n_samples = len(lat_indices)
    
    if mode == "point":
        # 点模式：提取坐标和值
        X = np.stack([
            lon_indices,
            lat_indices,
            time_indices
        ], axis=1).astype(np.float32)
        y = tensor[lat_indices, lon_indices, time_indices].astype(np.float32)
        return X, y
    elif mode == "image":
        # 图像模式：提取图像
        # 这里需要按时间分组
        images = []
        targets = []
        masks = []
        
        unique_times = np.unique(time_indices)
        for t in unique_times:
            t_mask = (time_indices == t)
            if t_mask.sum() > 0:
                # 创建该时间的完整图像
                img = tensor[:, :, t].copy()
                mask = (img != 0.0).astype(np.float32)
                images.append(img)
                masks.append(mask)
                targets.append(img)  # 目标就是图像本身
        
        if len(images) == 0:
            return np.array([]), np.array([])
        
        X_images = np.stack(images, axis=0)  # (T', H, W)
        X_masks = np.stack(masks, axis=0)
        X = np.stack([X_images, X_masks], axis=1)  # (T', 2, H, W)
        y = np.stack(targets, axis=0)  # (T', H, W)
        
        return X, y
    else:
        raise ValueError(f"Unknown mode: {mode}")


def cv_to_point_data(
    tensor: np.ndarray,
    cv_splits: List[CVSplit]
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    将CV分割转换为点数据格式
    
    返回:
    List[(X_train, y_train, X_test, y_test)]
    """
    results = []
    for split in cv_splits:
        X_train, y_train = extract_data_from_indices(
            tensor, split.train_indices, mode="point"
        )
        X_test, y_test = extract_data_from_indices(
            tensor, split.test_indices, mode="point"
        )
        results.append((X_train, y_train, X_test, y_test))
    return results


def cv_to_image_data(
    tensor: np.ndarray,
    cv_splits: List[CVSplit]
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    将CV分割转换为图像数据格式
    
    返回:
    List[(X_train, y_train, X_test, y_test)]
    """
    results = []
    for split in cv_splits:
        X_train, y_train = extract_data_from_indices(
            tensor, split.train_indices, mode="image"
        )
        X_test, y_test = extract_data_from_indices(
            tensor, split.test_indices, mode="image"
        )
        results.append((X_train, y_train, X_test, y_test))
    return results

