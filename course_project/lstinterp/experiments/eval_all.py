"""评估所有模型并对比"""
import numpy as np
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lstinterp.data import load_modis_tensor
from lstinterp.metrics import compute_regression_metrics, compute_probabilistic_metrics

# 输出目录
OUTPUT_DIR = Path("output")
(OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)


def load_results():
    """加载所有模型的结果"""
    results = {}
    results_dir = OUTPUT_DIR / "results"
    
    # 加载树模型结果
    try:
        with open(results_dir / "tree_results.json", "r") as f:
            results["Tree (XGBoost)"] = json.load(f)
    except FileNotFoundError:
        print("警告: tree_results.json 未找到")
    
    # 加载U-Net结果
    try:
        with open(results_dir / "unet_results.json", "r") as f:
            results["U-Net"] = json.load(f)
    except FileNotFoundError:
        print("警告: unet_results.json 未找到")
    
    # 加载GP结果
    try:
        with open(results_dir / "gp_results.json", "r") as f:
            results["GP (Sparse)"] = json.load(f)
    except FileNotFoundError:
        print("警告: gp_results.json 未找到")
    
    return results


def print_comparison_table(results):
    """打印对比表格"""
    if not results:
        print("没有可用的结果")
        return
    
    # 定义要显示的指标（排除experiment_info）
    metric_display = {
        'rmse': ('RMSE (K)', '越小越好'),
        'mae': ('MAE (K)', '越小越好'),
        'r2': ('R²', '越大越好'),
        'mape': ('MAPE (%)', '越小越好'),
        'crps': ('CRPS (K)', '越小越好'),
        'coverage_90': ('Coverage (90%)', '目标: 0.90'),
        'interval_width_90': ('Interval Width (90%)', '越小越好'),
        'calibration_error': ('Calibration Error', '越小越好')
    }
    
    # 打印表格
    print("\n" + "=" * 100)
    print("  模型对比结果 - 测试集性能")
    print("=" * 100)
    
    # 表头
    header = f"{'指标':<35} {'Tree (XGBoost)':<25} {'U-Net':<25} {'GP (Sparse)':<25}"
    print(header)
    print("-" * 100)
    
    # 数据行 - 回归指标
    print("\n【回归指标】")
    for metric, (name, note) in metric_display.items():
        if metric in ['rmse', 'mae', 'r2', 'mape']:
            row = f"  {name:<33}"
            for model_name in results.keys():
                value = results[model_name].get(metric, None)
                if value is not None and isinstance(value, (int, float)):
                    row += f"{value:<25.4f}"
                else:
                    row += f"{'N/A':<25}"
            print(row)
    
    # 数据行 - 概率指标
    print("\n【概率预测指标】")
    for metric, (name, note) in metric_display.items():
        if metric in ['crps', 'coverage_90', 'interval_width_90', 'calibration_error']:
            row = f"  {name:<33}"
            for model_name in results.keys():
                value = results[model_name].get(metric, None)
                if value is not None and isinstance(value, (int, float)):
                    row += f"{value:<25.4f}"
                else:
                    row += f"{'N/A':<25}"
            print(row)
    
    # 训练信息总结
    print("\n【训练信息】")
    training_info = f"{'训练时间':<33}"
    for model_name in results.keys():
        exp_info = results[model_name].get('experiment_info', {})
        train_time = exp_info.get('training_time_seconds', 0)
        if train_time:
            train_time_str = f"{train_time:.1f} 秒 ({train_time/60:.1f} 分钟)"
        else:
            train_time_str = "N/A"
        training_info += f"{train_time_str:<25}"
    print(training_info)
    
    print("=" * 100)
    
    # 性能排名
    print("\n【性能排名】")
    
    # R²排名（越大越好）
    r2_ranking = sorted([(name, results[name].get('r2', -999)) for name in results.keys()], 
                        key=lambda x: x[1], reverse=True)
    print("  R²排名:")
    for i, (name, value) in enumerate(r2_ranking, 1):
        print(f"    {i}. {name}: {value:.4f}")
    
    # RMSE排名（越小越好）
    rmse_ranking = sorted([(name, results[name].get('rmse', 999)) for name in results.keys()], 
                          key=lambda x: x[1])
    print("\n  RMSE排名 (越小越好):")
    for i, (name, value) in enumerate(rmse_ranking, 1):
        print(f"    {i}. {name}: {value:.4f} K")
    
    # CRPS排名（越小越好）
    crps_ranking = sorted([(name, results[name].get('crps', 999)) for name in results.keys()], 
                          key=lambda x: x[1])
    print("\n  CRPS排名 (越小越好):")
    for i, (name, value) in enumerate(crps_ranking, 1):
        print(f"    {i}. {name}: {value:.4f} K")
    
    print("=" * 100)


def main():
    results = load_results()
    
    if not results:
        print("没有找到任何结果文件。请先运行训练脚本。")
        return
    
    print_comparison_table(results)
    
    # 保存对比结果
    comparison_path = OUTPUT_DIR / "results" / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n对比结果已保存到 {comparison_path}")


if __name__ == "__main__":
    main()

