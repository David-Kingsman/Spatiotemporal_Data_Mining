# 报告准备索引

本文档为最终项目报告提供索引和结构指导。

## 📋 文档结构

```
doc/
├── REPORT_INDEX.md              # 本文件 - 报告索引
├── EXPERIMENTAL_RESULTS.md      # 详细实验结果汇总
├── MODEL_COMPARISON.md          # 模型对比分析
├── METHODOLOGY.md               # 方法论说明
└── README_LSTINTERP.md          # 项目技术文档
```

## 📊 实验数据位置

### 1. 测试集结果

所有模型在独立测试集上的性能评估：

| 模型 | JSON文件路径 | 可视化 |
|------|-------------|--------|
| Tree (XGBoost) | `output/results/tree_results.json` | `output/figures/tree_*.png` |
| U-Net | `output/results/unet_results.json` | `output/figures/unet_*.png` |
| GP (Sparse) | `output/results/gp_results.json` | `output/figures/gp_*.png` |
| 模型对比 | `output/results/model_comparison.json` | - |

### 2. 交叉验证结果

所有交叉验证实验的结果：

| CV类型 | Tree结果 | GP结果 | U-Net结果 |
|--------|---------|--------|-----------|
| **空间块CV** | `output/results/cv/tree_space_cv.json` | `output/results/cv/gp_space_cv.json` | (待完成) |
| **时间块CV** | `output/results/cv/tree_time_cv.json` | (待完成) | (待完成) |
| **K折CV** | `output/results/cv/tree_kfold_cv.json` | (待完成) | (待完成) |

### 3. 可视化结果

所有图表保存在 `output/figures/` 目录：

- **预测散点图**: `*_scatter.png`
- **残差分布**: `*_residuals.png`
- **空间预测图**: `unet_mean_day15.png`, `unet_std_day15.png`, `unet_error_day15.png`
- **数据可视化**: `modis_aug_data/data_visualization/*.png`

## 📝 报告章节建议

### 1. Introduction (引言)

- **参考文档**: `README_LSTINTERP.md` (项目概述)
- **要点**:
  - 问题背景（MODIS LST数据插值）
  - 研究目标（概率插值 + 不确定性评估）
  - 项目创新点（时空可分核、概率模型、工程实现）

### 2. Methodology (方法论)

- **参考文档**: `README_LSTINTERP.md` (模型说明)
- **要点**:
  - 三种模型架构（Tree, U-Net, GP）
  - 数据预处理
  - 训练策略

### 3. Experimental Setup (实验设置)

- **参考文档**: `EXPERIMENTAL_RESULTS.md` (实验配置)
- **要点**:
  - 数据集描述
  - 评估指标
  - 交叉验证策略
  - 超参数设置

### 4. Results (结果)

- **参考文档**: `EXPERIMENTAL_RESULTS.md`, `MODEL_COMPARISON.md`
- **要点**:
  - 测试集性能对比（主要结果）
  - 交叉验证结果（泛化能力）
  - 可视化分析
  - 统计显著性分析（如果有）

### 5. Discussion (讨论)

- **参考文档**: `MODEL_COMPARISON.md` (模型对比)
- **要点**:
  - 模型优势与劣势
  - 不同场景下的表现
  - 方法论洞察
  - 改进方向

### 6. Conclusion (结论)

- **参考文档**: `EXPERIMENTAL_RESULTS.md` (结论)
- **要点**:
  - 主要发现
  - 贡献总结
  - 未来工作

## 🔍 关键数据快速查找

### 最佳结果速查表

| 指标 | 最佳模型 | 数值 | 来源 |
|------|----------|------|------|
| **最低RMSE** | U-Net | 1.10 K | `tree_results.json` vs `unet_results.json` |
| **最高R²** | U-Net | 0.983 | `unet_results.json` |
| **最低CRPS** | U-Net | 0.74 | `unet_results.json` |
| **最佳Coverage** | U-Net | 0.994 | `unet_results.json` |
| **最快训练** | Tree | ~5分钟 | 实验记录 |
| **最好可解释性** | Tree | 特征重要性 | 模型特性 |

### 交叉验证汇总

| CV类型 | 模型 | RMSE均值 | R²均值 | 说明 |
|--------|------|----------|--------|------|
| 空间块 | Tree | 6.84 K | 0.327 | 空间泛化能力中等 |
| 空间块 | GP | 14.14 K | -3.29 | 空间泛化能力差 |
| 时间块 | Tree | 5.29 K | 0.569 | 时间泛化能力较好 |

## 📈 图表建议

### 必须包含的图表

1. **模型性能对比表**
   - 来源: `MODEL_COMPARISON.md` Section 1.1-1.2
   - 建议: 三列对比表格

2. **预测散点图**
   - 来源: `output/figures/*_scatter.png`
   - 建议: 三张图并列（Tree, U-Net, GP）

3. **残差分布图**
   - 来源: `output/figures/*_residuals.png`
   - 建议: 三张图并列

4. **空间预测可视化**
   - 来源: `output/figures/unet_mean_day15.png` 等
   - 建议: Mean, Std, Error 三张图

5. **交叉验证结果箱线图**
   - 来源: `EXPERIMENTAL_RESULTS.md` Section 2
   - 建议: 自制（显示不同fold的分布）

### 可选图表

1. **训练损失曲线**
   - 来源: 训练日志
   - 建议: 三个模型的loss对比

2. **不确定性可视化**
   - 来源: `output/figures/unet_std_day15.png`
   - 建议: 展示模型对不确定区域的识别

3. **特征重要性（Tree模型）**
   - 来源: XGBoost feature importance
   - 建议: 条形图或热力图

## 🎯 报告重点强调

### 1. 创新点

根据 `README_LSTINTERP.md` 中的创新点：

1. **时空结构建模**:
   - 不同于example论文将day作为类别变量
   - 本项采用时空可分核（space × time）显式建模时间相关性

2. **概率建模**:
   - 不仅关注RMSE、R²
   - 还关注CRPS、预测区间覆盖率
   - 所有模型都提供不确定性量化

3. **工程实现**:
   - 封装为通用Python库 `lstinterp`
   - 统一的数据加载、模型API、评估工具
   - 便于复现和复用

### 2. 实验设计优势

根据 `EXPERIMENTAL_RESULTS.md` 和 `METHODOLOGY.md`:

1. **严格的数据划分**:
   - 只用 `training_tensor` 进行训练和CV
   - 最终只在 `test_tensor` 上评估一次

3. **全面的评估指标**:
   - 回归指标（RMSE, R²）
   - 概率指标（CRPS, Coverage, Calibration Error）

### 3. 主要发现

根据 `MODEL_COMPARISON.md`:

1. **U-Net表现最佳**，在所有指标上都优于其他模型
2. **空间泛化比时间泛化更困难**
3. **GP模型需要进一步优化**才能发挥潜力

## 📚 引用建议

### 内部文档引用

- 实验结果: 见 `doc/EXPERIMENTAL_RESULTS.md`
- 模型对比: 见 `doc/MODEL_COMPARISON.md`
- 方法论: 见 `doc/METHODOLOGY.md`
- 技术细节: 见 `doc/README_LSTINTERP.md`

### 代码引用

- 模型实现: `lstinterp/models/`
- 评估脚本: `lstinterp/experiments/`
- 数据处理: `lstinterp/data/`

## ✅ 报告检查清单

### 内容完整性

- [ ] Introduction：问题背景、目标、创新点
- [ ] Methodology：三种模型详细说明
- [ ] Experimental Setup：数据、评估、CV策略
- [ ] Results：测试集结果、CV结果、可视化
- [ ] Discussion：模型对比、优劣分析、发现
- [ ] Conclusion：总结、贡献、未来工作

### 数据准确性

- [ ] 所有数值与JSON文件一致
- [ ] 图表与结果对应
- [ ] 引用来源清晰

### 实验严谨性

- [ ] 数据划分说明清楚
- [ ] 评估指标定义明确
- [ ] 交叉验证方法合理

### 可读性

- [ ] 图表清晰美观
- [ ] 表格格式统一
- [ ] 术语解释清楚

---

*文档创建时间: 2024年*
*最后更新: 2024年*
*维护者: 项目团队*


