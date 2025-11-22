# LSTInterp: 时空数据插值与不确定性评估库

## 项目概述

LSTInterp 是一个用于时空数据（如MODIS地表温度）插补和预测的Python库，提供多种概率模型和完整的评估指标。

### 核心特点

1. **真正的时空建模**：使用时空可分核（space × time），显式建模空间和时间相关性
2. **概率预测**：所有模型输出均值和不确定性（标准差），支持CRPS、覆盖率等概率指标
3. **多种模型**：
   - 时空稀疏高斯过程（Sparse GP）
   - 概率U-Net（用于图像插值）
   - 树模型baseline（XGBoost/LightGBM，支持分位数回归）
4. **完整的评估体系**：RMSE、R²、CRPS、预测区间覆盖率、校准误差等

## 项目结构

```
lstinterp/
├── __init__.py
├── config.py              # 配置数据类
├── data/
│   ├── __init__.py
│   └── modis.py           # MODIS数据加载
├── models/
│   ├── __init__.py
│   ├── gp_st.py           # 时空GP模型
│   ├── unet.py            # U-Net模型
│   └── tree_baselines.py  # 树模型baseline
├── metrics/
│   ├── __init__.py
│   ├── regression.py      # 回归指标（RMSE、R²等）
│   └── probabilistic.py   # 概率指标（CRPS、覆盖率等）
├── viz/
│   ├── __init__.py
│   └── maps.py            # 可视化函数
├── experiments/
│   ├── __init__.py
│   ├── train_tree.py      # 训练树模型
│   ├── train_unet.py      # 训练U-Net
│   └── eval_all.py        # 评估所有模型
└── utils/
    └── __init__.py        # 工具函数
```

## 安装

### 依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- numpy, scipy
- torch, gpytorch
- scikit-learn
- xgboost, lightgbm (可选)
- matplotlib, seaborn
- tqdm

### 可选依赖

- `tensorly`: 用于张量分解（如果需要）
- `gpytorch`: 用于高斯过程（推荐）

## 快速开始

### 1. 训练树模型baseline

```bash
cd course_project
python -m lstinterp.experiments.train_tree
```

### 2. 训练U-Net模型

```bash
python -m lstinterp.experiments.train_unet
```

### 3. 评估所有模型

```bash
python -m lstinterp.experiments.eval_all
```

## 使用示例

### 数据加载

```python
from lstinterp.data import load_modis_tensor, MODISDataset

# 加载数据
train_tensor = load_modis_tensor("modis_aug_data/MODIS_Aug.mat", "training_tensor")

# 创建数据集（点模式，用于GP/树模型）
dataset = MODISDataset(train_tensor, mode="point")

# 或图像模式（用于U-Net）
dataset = MODISDataset(train_tensor, mode="image")
```

### 训练模型

```python
from lstinterp.models import TreeBaseline, TreeConfig

# 配置
config = TreeConfig(
    model_type="xgb",
    n_estimators=100,
    quantile_regression=True
)

# 训练
model = TreeBaseline(config)
model.fit(X_train, y_train)

# 预测（带不确定性）
mean, std = model.predict_with_uncertainty(X_test)
```

### 评估指标

```python
from lstinterp.metrics import (
    compute_regression_metrics,
    compute_probabilistic_metrics
)

# 回归指标
reg_metrics = compute_regression_metrics(y_true, y_pred)
# {'rmse': 2.34, 'mae': 1.89, 'r2': 0.95, ...}

# 概率指标
prob_metrics = compute_probabilistic_metrics(y_true, mean, std)
# {'crps': 1.23, 'coverage_90': 0.89, 'interval_width_90': 4.56, ...}
```

### 可视化

```python
from lstinterp.viz import plot_mean_map, plot_std_map

plot_mean_map(pred_mean, day_idx=14, save_path="mean.png")
plot_std_map(pred_std, day_idx=14, save_path="std.png")
```

## 模型说明

### 1. 时空高斯过程（Sparse GP）

- **核结构**：可分时空核 `k(x, x') = k_space(lat, lon) × k_time(t)`
- **空间核**：Matern 3/2 或 5/2
- **时间核**：Matern 3/2 或 RBF
- **优势**：真正的概率模型，不确定性校准良好

### 2. 概率U-Net

- **架构**：简化版U-Net，输出均值和方差
- **输入**：温度图 + mask
- **优势**：快速训练，适合大规模数据，利用图像局部结构

### 3. 树模型Baseline

- **模型**：XGBoost / LightGBM / Random Forest
- **特点**：支持分位数回归，输出不确定性区间
- **用途**：作为baseline对比

## 评估指标

### 回归指标
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **MAPE**: 平均绝对百分比误差

### 概率指标
- **CRPS**: 连续概率排序分数（越小越好）
- **覆盖率**: 预测区间的实际覆盖率（应接近名义覆盖率）
- **区间宽度**: 预测区间的平均宽度（在保证覆盖率的前提下越小越好）
- **校准误差**: 预测不确定性的校准程度

## 与示例论文的改进

1. **时空建模**：
   - 示例：day作为类别变量
   - 本库：真正的时空可分核，显式建模时间相关性

2. **概率预测**：
   - 示例：点预测 + 启发式不确定性
   - 本库：真正的概率模型，输出校准良好的不确定性

3. **评估体系**：
   - 示例：主要关注RMSE
   - 本库：完整的概率评估指标（CRPS、覆盖率、校准误差）

4. **工程实现**：
   - 示例：单次实验脚本
   - 本库：可复用的Python库，统一API

## 输出文件

训练和评估会生成：

- `*_results.json`: 各模型的评估结果
- `model_comparison.json`: 所有模型的对比结果
- `*.png`: 可视化图片（预测图、误差图等）

## 许可证

本项目仅用于学术和教育目的。

## 作者

课程项目

