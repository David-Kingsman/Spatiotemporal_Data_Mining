# 实验文档索引

本文档目录包含了所有实验结果和对比分析，为最终项目报告做准备。

## 📚 文档结构

### 🆕 报告初稿

**🎯 [最终报告初稿](./FINAL_REPORT_DRAFT.md)** - 完整的学术报告初稿（英文）
- Abstract, Introduction, Methodology, Results, Discussion, Conclusion
- 包含所有关键结果、表格和图表引用
- 约3,800字，符合学术报告规范

### 核心文档

1. **[REPORT_INDEX.md](./REPORT_INDEX.md)** - 报告准备索引
   - 报告章节建议
   - 关键数据快速查找
   - 图表建议
   - 报告检查清单

2. **[EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md)** - 实验结果汇总
   - 测试集评估结果
   - 交叉验证结果
   - 模型对比总结
   - 实验配置
   - 可视化结果
   - 结论与建议

3. **[MODEL_COMPARISON.md](./MODEL_COMPARISON.md)** - 模型对比分析
   - 测试集性能对比
   - 交叉验证性能对比
   - 模型特性对比
   - 优势与劣势总结
   - 实验发现与洞察
   - 改进方向

4. **[CROSS_VALIDATION_GUIDE.md](./CROSS_VALIDATION_GUIDE.md)** - 交叉验证方法说明
   - 交叉验证类型
   - 使用方法
   - 代码示例
   - 与example论文对比

5. **[README_LSTINTERP.md](./README_LSTINTERP.md)** - 项目技术文档
   - 项目概述
   - 架构设计
   - 模型说明
   - 使用方法

## 📊 实验结果快速参考

### 测试集结果汇总

| 模型 | RMSE ↓ | R² ↑ | CRPS ↓ | Coverage 90% | 排名 |
|------|--------|------|--------|--------------|------|
| **U-Net** | **1.10** | **0.983** | **0.74** | 0.994 | 🥇 1 |
| **Tree (XGBoost)** | 3.89 | 0.793 | 2.06 | 0.869 | 🥈 2 |
| **GP (Sparse)** | 5.20 | 0.629 | 2.91 | 0.837 | 🥉 3 |

**详细结果**: 见 [EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md#1-测试集评估结果)

### 交叉验证结果汇总

| CV类型 | 模型 | RMSE均值 | R²均值 | 说明 |
|--------|------|----------|--------|------|
| **空间块** | Tree | 6.84 K | 0.327 | 空间泛化能力中等 |
| **空间块** | GP | 14.14 K | -3.29 | 空间泛化能力差 |
| **时间块** | Tree | 5.29 K | 0.569 | 时间泛化能力较好 |

**详细结果**: 见 [EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md#2-交叉验证结果)

## 📁 实验数据位置

### 结果文件

- **测试集结果**: `output/results/*_results.json`
- **交叉验证结果**: `output/results/cv/*_cv.json`
- **模型对比**: `output/results/model_comparison.json`

### 可视化结果

- **预测图**: `output/figures/*_scatter.png`
- **残差图**: `output/figures/*_residuals.png`
- **空间预测**: `output/figures/unet_*.png`
- **数据可视化**: `modis_aug_data/data_visualization/*.png`

## 🎯 报告撰写指南

### 1. 快速开始

1. 阅读 [REPORT_INDEX.md](./REPORT_INDEX.md) 了解报告结构
2. 查看 [EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md) 获取实验数据
3. 参考 [MODEL_COMPARISON.md](./MODEL_COMPARISON.md) 进行对比分析

### 2. 关键数据提取

所有关键数据都在以下文档中：
- **测试集结果**: [EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md#1-测试集评估结果)
- **交叉验证结果**: [EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md#2-交叉验证结果)
- **性能对比**: [MODEL_COMPARISON.md](./MODEL_COMPARISON.md#1-测试集性能对比)

### 3. 图表建议

参见 [REPORT_INDEX.md](./REPORT_INDEX.md#-图表建议) 获取详细的图表建议。

## 📝 文档使用说明

### 对于报告撰写

1. **Introduction**: 参考 `README_LSTINTERP.md` 中的项目概述和创新点
2. **Methodology**: 参考 `README_LSTINTERP.md` 中的模型说明
3. **Experimental Setup**: 参考 `EXPERIMENTAL_RESULTS.md` 中的实验配置
4. **Results**: 参考 `EXPERIMENTAL_RESULTS.md` 和 `MODEL_COMPARISON.md`
5. **Discussion**: 参考 `MODEL_COMPARISON.md` 中的分析和洞察
6. **Conclusion**: 参考 `EXPERIMENTAL_RESULTS.md` 中的结论

### 对于数据分析

1. 查看 [EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md) 获取原始结果
2. 查看 [MODEL_COMPARISON.md](./MODEL_COMPARISON.md) 进行对比分析
3. 查看 [CROSS_VALIDATION_GUIDE.md](./CROSS_VALIDATION_GUIDE.md) 理解CV方法

## 🔄 更新日志

### 2024年

- ✅ 创建实验结果汇总文档
- ✅ 创建模型对比分析文档
- ✅ 创建报告准备索引
- ✅ 整理所有实验结果

## 📞 联系与维护

如有问题或需要更新，请参考各个文档中的具体章节。

---

*文档维护: 项目团队*
*最后更新: 2024年*

