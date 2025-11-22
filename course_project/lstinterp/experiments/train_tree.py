"""
训练树模型Baseline（Tree-based Baseline Models）

本脚本实现了基于树模型的baseline方法，用于MODIS地表温度数据的插值和预测。

支持的模型：
1. XGBoost（优先使用）：梯度提升树，支持分位数回归
2. Random Forest（备用）：随机森林，不支持分位数回归（使用标准差估计不确定性）

主要特点：
1. 分位数回归（XGBoost）：提供预测分位数（10%, 50%, 90%）和不确定性估计
2. 标准差估计（Random Forest）：使用个体树预测的标准差估计不确定性
3. 快速训练和预测：树模型训练速度快，适合作为baseline

数据格式：
- 输入：3维张量 (H, W, T) = (100, 200, 31)
  - H: 纬度维度（35°-40°N）
  - W: 经度维度（-115°--105°W）
  - T: 时间维度（31天）
- 输出：温度值（单位：Kelvin）
- 缺失值：用0表示

评估指标：
- 回归指标：RMSE, MAE, R², MAPE
- 概率指标：CRPS, 90%预测区间覆盖率, 校准误差

作者：lstinterp团队
创建时间：2024年
"""
import numpy as np
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor, MODISDataset
from lstinterp.models import TreeBaseline, TreeConfig
from lstinterp.metrics import compute_regression_metrics, compute_probabilistic_metrics
from lstinterp.viz import plot_prediction_scatter, plot_residuals
from lstinterp.utils import set_seed

# 创建输出目录
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "results").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "models").mkdir(exist_ok=True)


def print_section_header(title, width=80):
    """打印章节标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    """主函数：训练和评估树模型"""
    start_time = time.time()
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_section_header("实验配置", width=80)
    print(f"实验时间: {experiment_time}")
    print(f"随机种子: 42")
    
    set_seed(42)
    
    # 检查依赖库
    print("\n依赖库检查:")
    try:
        import xgboost
        print(f"  ✅ XGBoost: {xgboost.__version__}")
        xgb_available = True
    except ImportError:
        print("  ⚠️  XGBoost未安装，将使用Random Forest")
        xgb_available = False
    
    try:
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
    except ImportError:
        print("  ❌ NumPy未安装")
        return
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        import sklearn
        print(f"  ✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("  ❌ scikit-learn未安装")
        return
    # 加载数据
    print_section_header("数据加载")
    data_path = "modis_aug_data/MODIS_Aug.mat"
    print(f"数据路径: {data_path}")
    
    print("\n加载训练数据...")
    train_tensor = load_modis_tensor(data_path, "training_tensor")
    H, W, T = train_tensor.shape
    print(f"训练数据维度: {H} × {W} × {T}")
    
    print("\n加载测试数据...")
    test_tensor = load_modis_tensor(data_path, "test_tensor")
    print(f"测试数据维度: {H} × {W} × {T}")
    
    # 创建数据集
    print_section_header("数据预处理")
    print("转换为点数据格式 (lat, lon, time) → temperature")
    
    print("\n创建训练数据集...")
    train_dataset = MODISDataset(train_tensor, mode="point")
    print(f"  - 训练观测点数: {len(train_dataset):,}")
    
    print("\n创建测试数据集...")
    test_dataset = MODISDataset(test_tensor, mode="point")
    print(f"  - 测试观测点数: {len(test_dataset):,}")
    
    # 准备训练数据
    print("\n提取训练数据...")
    X_train = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
    y_train = np.array([train_dataset[i][1].numpy() for i in range(len(train_dataset))])
    
    print(f"  - 输入特征维度: {X_train.shape}")
    print(f"    * 特征1 (纬度): 范围 [{X_train[:, 0].min():.2f}, {X_train[:, 0].max():.2f}]")
    print(f"    * 特征2 (经度): 范围 [{X_train[:, 1].min():.2f}, {X_train[:, 1].max():.2f}]")
    print(f"    * 特征3 (时间): 范围 [{X_train[:, 2].min():.0f}, {X_train[:, 2].max():.0f}] 天")
    print(f"  - 目标变量维度: {y_train.shape}")
    print(f"    * 温度范围: [{y_train.min():.2f}, {y_train.max():.2f}] K")
    print(f"    * 温度均值: {y_train.mean():.2f} K")
    print(f"    * 温度标准差: {y_train.std():.2f} K")
    
    # 准备测试数据
    print("\n提取测试数据...")
    X_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1].numpy() for i in range(len(test_dataset))])
    
    print(f"  - 输入特征维度: {X_test.shape}")
    print(f"    * 特征1 (纬度): 范围 [{X_test[:, 0].min():.2f}, {X_test[:, 0].max():.2f}]")
    print(f"    * 特征2 (经度): 范围 [{X_test[:, 1].min():.2f}, {X_test[:, 1].max():.2f}]")
    print(f"    * 特征3 (时间): 范围 [{X_test[:, 2].min():.0f}, {X_test[:, 2].max():.0f}] 天")
    print(f"  - 目标变量维度: {y_test.shape}")
    print(f"    * 温度范围: [{y_test.min():.2f}, {y_test.max():.2f}] K")
    print(f"    * 温度均值: {y_test.mean():.2f} K")
    print(f"    * 温度标准差: {y_test.std():.2f} K")
    
    # 训练模型
    print_section_header("模型配置和训练")
    
    # 选择模型类型
    if xgb_available:
        model_type = "xgb"
        print("✅ 使用XGBoost模型")
        print("  - 支持分位数回归")
        print("  - 提供不确定性估计")
    else:
        model_type = "rf"
        print("⚠️  使用Random Forest模型（XGBoost不可用）")
        print("  - 使用标准差估计不确定性")
    
    config = TreeConfig(
        model_type=model_type,
        n_estimators=100,
        quantile_regression=(model_type != "rf"),  # RF不支持分位数回归
        quantiles=[0.1, 0.5, 0.9] if model_type != "rf" else None
    )
    
    print("\n模型超参数:")
    print(f"  - 模型类型: {config.model_type}")
    print(f"  - 树的数量: {config.n_estimators}")
    print(f"  - 分位数回归: {config.quantile_regression}")
    if config.quantile_regression:
        print(f"  - 分位数: {config.quantiles}")
    
    # 训练
    print("\n开始训练...")
    training_start_time = time.time()
    model = TreeBaseline(config)
    model.fit(X_train, y_train)
    training_time = time.time() - training_start_time
    print(f"✅ 训练完成 (耗时: {training_time:.2f} 秒)")
    
    # 预测
    print_section_header("模型预测")
    prediction_start_time = time.time()
    print("进行预测...")
    y_pred_mean, y_pred_std = model.predict_with_uncertainty(X_test)
    prediction_time = time.time() - prediction_start_time
    print(f"✅ 预测完成 (耗时: {prediction_time:.2f} 秒)")
    print(f"  - 预测点数: {len(y_pred_mean):,}")
    
    # 评估
    print_section_header("模型评估")
    print("计算评估指标...")
    reg_metrics = compute_regression_metrics(y_test, y_pred_mean)
    prob_metrics = compute_probabilistic_metrics(y_test, y_pred_mean, y_pred_std)
    
    all_metrics = {**reg_metrics, **prob_metrics}
    
    # 添加实验信息
    all_metrics["experiment_info"] = {
        "experiment_time": experiment_time,
        "random_seed": 42,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "model_config": {
            "model_type": config.model_type,
            "n_estimators": config.n_estimators,
            "quantile_regression": config.quantile_regression,
            "quantiles": config.quantiles if config.quantile_regression else None
        },
        "data_info": {
            "train_points": len(X_train),
            "test_points": len(X_test)
        }
    }
    
    print("\n" + "=" * 80)
    print("  评估结果")
    print("=" * 80)
    
    # 回归指标
    print("\n【回归指标】")
    print(f"  {'指标':<30} {'值':<15} {'说明':<30}")
    print("-" * 75)
    print(f"  {'RMSE (Root Mean Squared Error)':<30} {reg_metrics['rmse']:<15.4f} {'越小越好，单位: Kelvin'}")
    print(f"  {'MAE (Mean Absolute Error)':<30} {reg_metrics['mae']:<15.4f} {'越小越好，单位: Kelvin'}")
    print(f"  {'R² (Coefficient of Determination)':<30} {reg_metrics['r2']:<15.4f} {'越大越好，范围: (-∞, 1]'}")
    print(f"  {'MAPE (Mean Absolute Percentage Error)':<30} {reg_metrics['mape']:<15.4f} {'越小越好，单位: %'}")
    
    # 概率指标
    print("\n【概率预测指标】")
    print(f"  {'指标':<30} {'值':<15} {'说明':<30}")
    print("-" * 75)
    print(f"  {'CRPS (Continuous Ranked Probability Score)':<30} {prob_metrics['crps']:<15.4f} {'越小越好，单位: Kelvin'}")
    print(f"  {'Coverage (90% Prediction Interval)':<30} {prob_metrics['coverage_90']:<15.4f} {'目标: 0.90'}")
    print(f"  {'Interval Width (90%)':<30} {prob_metrics['interval_width_90']:<15.4f} {'越小越好，单位: Kelvin'}")
    print(f"  {'Calibration Error':<30} {prob_metrics['calibration_error']:<15.4f} {'越小越好，衡量校准度'}")
    
    # 预测统计
    print("\n【预测统计】")
    print(f"  预测均值:")
    print(f"    - 范围: [{y_pred_mean.min():.2f}, {y_pred_mean.max():.2f}] K")
    print(f"    - 均值: {y_pred_mean.mean():.2f} K")
    print(f"    - 标准差: {y_pred_mean.std():.2f} K")
    
    print(f"\n  真实值:")
    print(f"    - 范围: [{y_test.min():.2f}, {y_test.max():.2f}] K")
    print(f"    - 均值: {y_test.mean():.2f} K")
    print(f"    - 标准差: {y_test.std():.2f} K")
    
    print(f"\n  预测不确定性 (标准差):")
    print(f"    - 范围: [{y_pred_std.min():.2f}, {y_pred_std.max():.2f}] K")
    print(f"    - 均值: {y_pred_std.mean():.2f} K")
    print(f"    - 中位数: {np.median(y_pred_std):.2f} K")
    
    # 误差分析
    errors = y_test - y_pred_mean
    print(f"\n【误差分析】")
    print(f"  残差 (真实值 - 预测值):")
    print(f"    - 均值: {errors.mean():.2f} K (接近0表示无偏)")
    print(f"    - 标准差: {errors.std():.2f} K")
    print(f"    - 范围: [{errors.min():.2f}, {errors.max():.2f}] K")
    print(f"    - 中位数: {np.median(errors):.2f} K")
    
    # 覆盖率分析
    coverage = prob_metrics['coverage_90']
    target_coverage = 0.90
    coverage_error = abs(coverage - target_coverage)
    print(f"\n【不确定性校准】")
    print(f"  90%预测区间覆盖率: {coverage:.4f} (目标: {target_coverage})")
    if coverage_error < 0.05:
        print(f"  ✅ 校准良好 (误差 < 5%)")
    elif coverage_error < 0.10:
        print(f"  ⚠️  校准尚可 (误差 < 10%)")
    else:
        print(f"  ❌ 校准较差 (误差 >= 10%)")
    
    # 保存结果
    print_section_header("保存结果")
    results_path = OUTPUT_DIR / "results" / "tree_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ 评估结果已保存: {results_path}")
    
    # 保存模型
    try:
        import pickle
        model_path = OUTPUT_DIR / "models" / f"tree_model_{model_type}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ 模型已保存: {model_path}")
        print(f"  - 模型大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"⚠️  模型保存失败: {e}")
    
    # 可视化
    print("\n生成可视化图表...")
    scatter_path = OUTPUT_DIR / "figures" / "tree_scatter.png"
    residuals_path = OUTPUT_DIR / "figures" / "tree_residuals.png"
    
    plot_prediction_scatter(y_test, y_pred_mean, save_path=str(scatter_path))
    print(f"✅ 预测散点图已保存: {scatter_path}")
    
    plot_residuals(y_test, y_pred_mean, save_path=str(residuals_path))
    print(f"✅ 残差图已保存: {residuals_path}")
    
    # 总结
    total_time = time.time() - start_time
    print_section_header("实验完成")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"  - 数据加载和预处理: {training_start_time - start_time:.2f} 秒")
    print(f"  - 模型训练: {training_time:.2f} 秒")
    print(f"  - 模型预测: {prediction_time:.2f} 秒")
    
    print(f"\n主要指标总结:")
    print(f"  - R²: {reg_metrics['r2']:.4f}")
    print(f"  - RMSE: {reg_metrics['rmse']:.4f} K")
    print(f"  - CRPS: {prob_metrics['crps']:.4f} K")
    print(f"  - 覆盖率(90%): {prob_metrics['coverage_90']:.4f}")
    
    print(f"\n所有结果文件:")
    print(f"  📄 {results_path}")
    if 'model_path' in locals():
        print(f"  💾 {model_path}")
    print(f"  📊 {scatter_path}")
    print(f"  📊 {residuals_path}")


if __name__ == "__main__":
    main()

