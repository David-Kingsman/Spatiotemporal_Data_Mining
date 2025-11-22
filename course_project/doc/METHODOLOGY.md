# 方法论

本文档详细说明了项目中使用的三种模型的方法论，包括数学公式推导、模型架构和超参数设置依据。

## 1. 时空高斯过程模型（GP）

### 1.1 高斯过程基础

高斯过程（Gaussian Process, GP）是一种非参数贝叶斯方法，通过定义在输入空间上的随机函数来建模不确定性。

**定义**：一个高斯过程完全由其均值函数 $m(\mathbf{x})$ 和协方差函数（核函数）$k(\mathbf{x}, \mathbf{x}')$ 确定：

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

其中 $\mathbf{x}$ 是输入特征向量（在本项目中为 $(\text{lat}, \text{lon}, t)$）。

### 1.2 可分离时空核函数

本项目采用**可分离时空核**（separable space-time kernel），将空间和时间相关性分开建模：

$$k((\mathbf{s}, t), (\mathbf{s}', t')) = k_{\text{space}}(\mathbf{s}, \mathbf{s}') \times k_{\text{time}}(t, t')$$

其中：
- $\mathbf{s} = (\text{lat}, \text{lon})$ 是空间坐标
- $t$ 是时间索引
- $k_{\text{space}}$ 是空间核函数
- $k_{\text{time}}$ 是时间核函数

**优势**：可分离核显著降低了计算复杂度（从 $O(n^3)$ 降低到 $O(n^2)$），同时保持了时空相关性的显式建模。

### 1.3 空间核函数：Matern核

空间相关性使用**Matern 3/2核**（$\nu = 1.5$）：

$$k_{\text{space}}(\mathbf{s}, \mathbf{s}') = \sigma^2_{\text{space}} \left(1 + \frac{\sqrt{3}r}{l_{\text{space}}}\right) \exp\left(-\frac{\sqrt{3}r}{l_{\text{space}}}\right)$$

其中：
- $r = ||\mathbf{s} - \mathbf{s}'||$ 是空间距离（欧氏距离）
- $l_{\text{space}}$ 是空间长度尺度（lengthscale）参数
- $\sigma^2_{\text{space}}$ 是空间输出尺度（outputscale）参数

**选择理由**：
- Matern核比RBF核（Squared Exponential）更粗糙（less smooth），更适合处理具有非平滑特征的地球物理数据
- Matern 3/2 核只需一次可微，计算效率高，同时保持了足够的平滑性

### 1.4 时间核函数：Matern核

时间相关性也使用**Matern 3/2核**：

$$k_{\text{time}}(t, t') = \sigma^2_{\text{time}} \left(1 + \frac{\sqrt{3}|t-t'|}{l_{\text{time}}}\right) \exp\left(-\frac{\sqrt{3}|t-t'|}{l_{\text{time}}}\right)$$

其中：
- $l_{\text{time}}$ 是时间长度尺度参数
- $\sigma^2_{\text{time}}$ 是时间输出尺度参数

**选择理由**：
- 与空间核保持一致，便于超参数优化
- Matern核能更好地捕捉时间序列中的短期相关性

### 1.5 变分推断与诱导点

为了处理大规模数据（~50万个观测点），我们使用**稀疏变分高斯过程**（Sparse Variational Gaussian Process, SVGP）：

**诱导点（Inducing Points）**：在空间和时间维度上选择代表性点：
- 空间诱导点：在空间网格上均匀采样（15×10 = 150个点）
- 时间诱导点：均匀采样时间维度（10个时间点）
- 总诱导点数：500个（15×10×10的笛卡尔积）

**变分下界（ELBO）**：

$$\mathcal{L}_{\text{ELBO}} = \sum_{i=1}^n \mathbb{E}_{q(f_i)}[\log p(y_i|f_i)] - \text{KL}(q(\mathbf{u})||p(\mathbf{u}))$$

其中：
- $\mathbf{u}$ 是诱导点处的函数值
- $q(\mathbf{u})$ 是变分后验分布
- $p(\mathbf{u})$ 是先验分布
- KL项确保变分分布接近真实后验

**优势**：
- 计算复杂度从 $O(n^3)$ 降低到 $O(m^2n)$，其中 $m \ll n$ 是诱导点数
- 可以通过批量训练处理大规模数据

### 1.6 超参数设置

| 超参数 | 值 | 设置依据 |
|--------|-----|----------|
| **空间核** | Matern 3/2 | 适合非平滑地理数据 |
| **时间核** | Matern 3/2 | 与空间核保持一致 |
| **诱导点数** | 500 | 平衡计算效率和精度 |
| **空间诱导网格** | 15×10 | 覆盖主要空间变化 |
| **时间诱导数** | 10 | 捕捉时间动态 |
| **学习率** | 0.01 | Adam优化器标准设置 |
| **训练轮数** | 50 | 早停策略（patience=10） |
| **批量大小** | 1000 | 内存和计算效率平衡 |
| **长度尺度初始化** | 0.5 | 对于[0,1]归一化数据合理 |
| **输出尺度初始化** | 1.0 | 对于归一化目标合理 |
| **噪声初始化** | 0.2 | 观察噪声的标准估计 |

**参数约束**：
- 长度尺度：$l \in [0.1, 50.0]$（确保正值且不过大）
- 输出尺度：$\sigma \in [0.1, 50.0]$（限制方差范围）
- 噪声：$\sigma_n \in [0.01, 5.0]$（合理噪声范围）

---

## 2. 概率U-Net模型

### 2.1 U-Net架构

U-Net是一种**编码器-解码器**（Encoder-Decoder）架构，最初用于医学图像分割，在图像修复（inpainting）任务中也表现优异。

**架构特点**：
- **编码器（下采样）**：逐步提取高级特征，减少空间维度
- **跳跃连接（Skip Connections）**：保留低级空间细节
- **解码器（上采样）**：逐步恢复空间分辨率并生成预测

### 2.2 网络结构

**编码器路径**：
```
输入: (B, 2, 100, 200)  # [温度图, mask]
    ↓ ConvBlock(2→32) + MaxPool2d(2)  # (B, 32, 50, 100)
    ↓ ConvBlock(32→64) + MaxPool2d(2)  # (B, 64, 25, 50)
    ↓ ConvBlock(64→128)                # (B, 128, 25, 50)
```

**解码器路径**：
```
    ↓ ConvTranspose2d(128→64, stride=2)  # (B, 64, 50, 100)
    ↓ Concatenate with encoder feature   # (B, 128, 50, 100)
    ↓ ConvBlock(128→64)                  # (B, 64, 50, 100)
    ↓ ConvTranspose2d(64→32, stride=2)   # (B, 32, 100, 200)
    ↓ Concatenate with encoder feature   # (B, 64, 100, 200)
    ↓ ConvBlock(64→32)                   # (B, 32, 100, 200)
    ↓ Separate heads:
        - Mean head: Conv(32→1)          # (B, 1, 100, 200)
        - LogVar head: Conv(32→1)        # (B, 1, 100, 200)
```

**ConvBlock结构**：
```python
Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU
```

### 2.3 概率输出与损失函数

U-Net输出每个像素的**均值** $\mu_{ij}$ 和**对数方差** $\log \sigma^2_{ij}$，假设观测值服从高斯分布：

$$p(y_{ij} | \mathbf{x}_{ij}) = \mathcal{N}(y_{ij} | \mu_{ij}, \sigma^2_{ij})$$

**负对数似然损失**（仅在观测点计算）：

$$\mathcal{L}_{\text{NLL}} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} \left[\frac{1}{2}\log(2\pi\sigma^2_{ij}) + \frac{(y_{ij} - \mu_{ij})^2}{2\sigma^2_{ij}}\right]$$

其中 $\Omega$ 是观测点的集合，$|\Omega|$ 是观测点数量。

**数值稳定性处理**：
- 限制 $\log \sigma^2_{ij} \in [-10, 10]$（避免数值溢出）
- 添加小的常数 $\epsilon = 1e-6$ 到方差：$\sigma^2 = \exp(\log \sigma^2) + \epsilon$
- 裁剪异常大的损失值

### 2.4 超参数设置

| 超参数 | 值 | 设置依据 |
|--------|-----|----------|
| **输入通道数** | 2 | [温度图, mask] |
| **基础通道数** | 32 | 平衡模型容量和计算效率 |
| **Dropout** | 0.2 | 标准正则化设置 |
| **学习率** | 0.0005 | 较小学习率确保稳定训练 |
| **优化器** | Adam | 自适应学习率，适合图像任务 |
| **权重衰减** | 1e-5 | L2正则化 |
| **梯度裁剪** | max_norm=1.0 | 防止梯度爆炸 |
| **训练轮数** | 50 | 早停策略（patience=10） |
| **批量大小** | 4 | 内存限制下的合理值 |
| **学习率调度** | ReduceLROnPlateau | 验证损失不降时降低学习率 |
| **$\log \sigma^2$ 初始化** | -1.0 | 对应 $\sigma \approx 0.6$，合理初始不确定性 |

**正则化策略**：
- **BatchNorm2d**：每层后添加，加速训练并提高稳定性
- **Dropout**：在编码器和解码器中添加，防止过拟合
- **早停**：监控验证集损失，防止过拟合

---

## 3. 树模型基线（Tree Baseline）

### 3.1 XGBoost模型

XGBoost（Extreme Gradient Boosting）是一种**梯度提升树**（Gradient Boosting Tree）算法。

**目标函数**：

$$\mathcal{L} = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

其中：
- $l(y_i, \hat{y}_i)$ 是损失函数（回归任务中使用平方误差）
- $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda ||w||^2$ 是正则化项
  - $T$ 是叶子节点数
  - $w$ 是叶子节点权重
  - $\gamma, \lambda$ 是正则化参数

**预测公式**：

$$\hat{y}_i = \sum_{k=1}^K f_k(\mathbf{x}_i)$$

其中 $f_k$ 是第 $k$ 棵树。

### 3.2 分位数回归

为了提供不确定性估计，我们使用**分位数回归**（Quantile Regression）：

**分位数损失**：

$$l_{\tau}(y, \hat{y}) = \begin{cases}
\tau (y - \hat{y}), & \text{if } y \geq \hat{y} \\
(1-\tau) (\hat{y} - y), & \text{if } y < \hat{y}
\end{cases}$$

**分位数**：$\tau \in \{0.1, 0.5, 0.9\}$
- $\tau = 0.1$：10%分位数（下界）
- $\tau = 0.5$：50%分位数（中位数，作为预测均值）
- $\tau = 0.9$：90%分位数（上界）

**不确定性估计**：

$$\sigma_{\text{pred}} = \frac{q_{0.9} - q_{0.1}}{2 \times \Phi^{-1}(0.9)}$$

其中 $\Phi^{-1}(0.9) \approx 1.28$ 是标准正态分布的90%分位数。

**假设**：90%预测区间为 $[q_{0.1}, q_{0.9}]$，且近似服从正态分布。

### 3.3 超参数设置

| 超参数 | 值 | 设置依据 |
|--------|-----|----------|
| **模型类型** | XGBoost | 性能优异，支持分位数回归 |
| **树的数量** | 100 | 标准设置，平衡性能和计算 |
| **最大深度** | 6 | 默认值，防止过拟合 |
| **分位数回归** | True | 提供不确定性估计 |
| **分位数列表** | [0.1, 0.5, 0.9] | 计算90%预测区间 |
| **随机种子** | 42 | 确保可重复性 |

**特征**：
- 输入：归一化的空间坐标 $(\text{lat}, \text{lon})$ 和时间索引 $t$
- 输出：分位数预测（用于计算均值和不确定性）

**优势**：
- 训练速度快（~12秒）
- 可解释性强（特征重要性）
- 对超参数不敏感

---

## 4. 数据预处理

### 4.1 归一化

**输入特征归一化**（GP和Tree模型）：
- **空间坐标**：Min-Max归一化到 [0, 1]
  - $\text{lat}_{norm} = \frac{\text{lat} - \text{lat}_{min}}{\text{lat}_{max} - \text{lat}_{min}}$
  - $\text{lon}_{norm} = \frac{\text{lon} - \text{lon}_{min}}{\text{lon}_{max} - \text{lon}_{min}}$
  - $t_{norm} = \frac{t}{T-1}$（$T$ 是总天数）

**图像归一化**（U-Net模型）：
- **Z-score归一化**：$\mathbf{x}_{norm} = \frac{\mathbf{x} - \mu}{\sigma}$
  - $\mu$ 和 $\sigma$ 从训练集计算
  - 测试集使用相同的 $\mu$ 和 $\sigma$（重要！）

**目标变量归一化**（仅GP模型）：
- GP模型中，目标变量也进行Z-score归一化，以匹配归一化输入的尺度

### 4.2 缺失值处理

- **U-Net**：缺失值用均值填充（用于输入），mask标记观测位置
- **GP/Tree**：仅使用有观测的点进行训练和预测

---

## 5. 评估指标

### 5.1 回归指标

**RMSE**（均方根误差）：
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

**MAE**（平均绝对误差）：
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**R²**（决定系数）：
$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

### 5.2 概率预测指标

**CRPS**（连续排序概率分数）：
对于高斯预测分布 $\mathcal{N}(\mu, \sigma^2)$：
$$\text{CRPS} = \sigma \left[\frac{1}{\sqrt{\pi}} - 2\phi\left(\frac{y-\mu}{\sigma}\right) - \frac{y-\mu}{\sigma}\left(2\Phi\left(\frac{y-\mu}{\sigma}\right) - 1\right)\right]$$

其中 $\phi$ 和 $\Phi$ 是标准正态分布的PDF和CDF。

**覆盖率**（Coverage）：
$$\text{Coverage}_{90\%} = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[y_i \in [\mu_i - 1.645\sigma_i, \mu_i + 1.645\sigma_i]]$$

目标值：0.90（对于90%预测区间）

**校准误差**（Calibration Error）：
$$\text{Calibration Error} = |\text{Coverage} - 0.90|$$

---

## 6. 训练策略

### 6.1 数据划分

- **训练集**：`training_tensor`（严格用于训练和交叉验证）
- **测试集**：`test_tensor`（仅用于最终评估，不参与任何训练过程）

### 6.2 训练流程

1. **数据加载与预处理**
2. **模型初始化**（使用合理的初始参数）
3. **训练循环**：
   - 前向传播
   - 损失计算
   - 反向传播
   - 参数更新
4. **验证集监控**（U-Net和GP）
5. **早停**（如果验证损失不再下降）
6. **模型保存**（保存最佳模型）

### 6.3 优化器设置

- **GP模型**：Adam优化器，学习率0.01
- **U-Net模型**：Adam优化器，学习率0.0005，权重衰减1e-5
- **Tree模型**：XGBoost内置优化（二阶梯度信息）

---

## 7. 模型选择与创新点

### 7.1 模型选择理由

1. **GP模型**：
   - 提供理论保证和不确定性量化
   - 显式建模时空相关性（可分离核）
   - 适合小到中等规模数据

2. **U-Net模型**：
   - 利用图像的局部结构（卷积操作）
   - 训练速度快，可扩展到更大数据
   - 概率输出提供不确定性

3. **Tree模型**：
   - 快速baseline，易于解释
   - 分位数回归提供不确定性
   - 对超参数不敏感

### 7.2 本项目创新点

1. **时空可分核**：显式建模空间和时间相关性，而非将时间作为类别变量
2. **概率预测**：所有模型都提供不确定性量化，不仅关注点预测
3. **综合评估**：不仅使用RMSE/R²，还使用CRPS、覆盖率等概率指标
4. **工程实现**：模块化、可复用的Python库，统一的数据处理和评估接口

---

**最后更新**: 2025-11-16


