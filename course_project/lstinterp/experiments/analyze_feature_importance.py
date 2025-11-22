"""
Tree模型特征重要性分析

该脚本提取XGBoost的特征重要性，可视化空间vs时间的重要性，
并分析哪些特征最重要。
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 输出目录
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "feature_importance"
RESULTS_DIR = OUTPUT_DIR / "results" / "feature_importance"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径
MODEL_PATH = OUTPUT_DIR / "models" / "tree_model_xgb.pkl"
TREE_RESULTS_PATH = OUTPUT_DIR / "results" / "tree_results.json"

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def print_section_header(title):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_feature_importance():
    """分析Tree模型的特征重要性"""
    print_section_header("Tree模型特征重要性分析")
    
    if not MODEL_PATH.exists():
        print(f"⚠️  模型文件不存在: {MODEL_PATH}")
        print("请先运行 train_tree.py 训练模型")
        return None
    
    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        tree_baseline = pickle.load(f)
    
    print(f"✅ 模型类型: {type(tree_baseline).__name__}")
    
    # 获取实际的模型（TreeBaseline包装器内部的模型）
    model = None
    
    # 先检查是否有分位数模型（优先级更高，因为这是XGBoost的主要模式）
    if hasattr(tree_baseline, 'quantile_models') and len(tree_baseline.quantile_models) > 0:
        # 如果有分位数模型，使用0.5分位数的模型（中位数）
        if 0.5 in tree_baseline.quantile_models:
            model = tree_baseline.quantile_models[0.5]
        else:
            # 使用第一个模型
            model = list(tree_baseline.quantile_models.values())[0]
    elif hasattr(tree_baseline, 'model') and tree_baseline.model is not None:
        model = tree_baseline.model
    
    if model is None:
        print("⚠️  无法找到内部模型")
        print(f"  - tree_baseline.model: {getattr(tree_baseline, 'model', 'N/A')}")
        print(f"  - tree_baseline.quantile_models: {getattr(tree_baseline, 'quantile_models', 'N/A')}")
        return None
    
    print(f"✅ 内部模型类型: {type(model).__name__}")
    
    # 检查是否是XGBoost
    if hasattr(model, 'get_booster'):
        # XGBoost模型
        booster = model.get_booster()
        feature_importance = booster.get_score(importance_type='gain')
        
        print(f"\n特征重要性类型: gain (信息增益)")
        print(f"特征数量: {len(feature_importance)}")
        
        # 转换为DataFrame
        df = pd.DataFrame([
            {'feature': feat, 'importance': imp} 
            for feat, imp in feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        # 特征名称映射
        feature_names = {
            'f0': 'Latitude',
            'f1': 'Longitude',
            'f2': 'Time'
        }
        
        df['feature_name'] = df['feature'].map(feature_names)
        df['feature_name'] = df['feature_name'].fillna(df['feature'])
        
    elif hasattr(model, 'feature_importances_'):
        # scikit-learn模型（RandomForest, GradientBoosting等）
        feature_importance = model.feature_importances_
        
        # 特征名称
        feature_names = ['Latitude', 'Longitude', 'Time']
        
        df = pd.DataFrame({
            'feature': [f'f{i}' for i in range(len(feature_importance))],
            'feature_name': feature_names[:len(feature_importance)],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n特征重要性类型: feature_importances_")
        print(f"特征数量: {len(feature_importance)}")
    
    else:
        print("⚠️  无法提取特征重要性（模型类型不支持）")
        return None
    
    print("\n特征重要性排名:")
    print(df.to_string(index=False, float_format='%.6f'))
    
    # 保存结果
    csv_path = RESULTS_DIR / "feature_importance.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n✅ 特征重要性表已保存: {csv_path}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 特征重要性条形图
    axes[0, 0].barh(df['feature_name'], df['importance'], 
                   color=['green', 'blue', 'orange'], alpha=0.7)
    axes[0, 0].set_xlabel('Importance (Gain)', fontsize=11)
    axes[0, 0].set_title('Feature Importance Ranking', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].invert_yaxis()  # 最重要的在顶部
    
    # 2. 特征重要性饼图
    axes[0, 1].pie(df['importance'], labels=df['feature_name'], autopct='%1.1f%%',
                   colors=['green', 'blue', 'orange'], startangle=90)
    axes[0, 1].set_title('Feature Importance Distribution', fontsize=12, fontweight='bold')
    
    # 3. 归一化重要性（百分比）
    df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
    
    axes[1, 0].bar(df['feature_name'], df['importance_pct'], 
                   color=['green', 'blue', 'orange'], alpha=0.7)
    axes[1, 0].set_ylabel('Importance (%)', fontsize=11)
    axes[1, 0].set_title('Normalized Feature Importance', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 添加百分比标签
    for i, v in enumerate(df['importance_pct']):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. 空间vs时间重要性对比
    spatial_features = df[df['feature_name'].isin(['Latitude', 'Longitude'])]
    temporal_features = df[df['feature_name'] == 'Time']
    
    spatial_importance = spatial_features['importance'].sum()
    temporal_importance = temporal_features['importance'].sum() if len(temporal_features) > 0 else 0
    
    categories = ['Spatial\n(Lat + Lon)', 'Temporal\n(Time)']
    values = [spatial_importance, temporal_importance]
    
    axes[1, 1].bar(categories, values, color=['purple', 'orange'], alpha=0.7)
    axes[1, 1].set_ylabel('Total Importance', fontsize=11)
    axes[1, 1].set_title('Spatial vs Temporal Importance', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + max(values) * 0.01, f'{v:.2f}', 
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "feature_importance_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化已保存: {fig_path}")
    
    # 分析结果
    print("\n" + "-" * 80)
    print("  分析结果")
    print("-" * 80)
    
    most_important = df.iloc[0]
    print(f"\n最重要的特征: {most_important['feature_name']} (重要性: {most_important['importance']:.6f})")
    
    if len(df) >= 2:
        second_important = df.iloc[1]
        print(f"第二重要的特征: {second_important['feature_name']} (重要性: {second_important['importance']:.6f})")
    
    if len(df) >= 3:
        third_important = df.iloc[2]
        print(f"第三重要的特征: {third_important['feature_name']} (重要性: {third_important['importance']:.6f})")
    
    print(f"\n空间特征总重要性: {spatial_importance:.6f} ({spatial_importance/(spatial_importance+temporal_importance)*100:.1f}%)")
    print(f"时间特征总重要性: {temporal_importance:.6f} ({temporal_importance/(spatial_importance+temporal_importance)*100:.1f}%)")
    
    print("\n解释:")
    print("  - 重要性越高，该特征对预测的贡献越大")
    print("  - 空间特征（纬度、经度）反映地理位置对温度的影响")
    print("  - 时间特征反映时间变化对温度的影响")
    
    # 保存分析结果
    analysis_results = {
        'most_important_feature': most_important['feature_name'],
        'most_important_value': float(most_important['importance']),
        'spatial_importance': float(spatial_importance),
        'temporal_importance': float(temporal_importance),
        'spatial_percentage': float(spatial_importance/(spatial_importance+temporal_importance)*100),
        'temporal_percentage': float(temporal_importance/(spatial_importance+temporal_importance)*100),
        'feature_importance': df.to_dict('records')
    }
    
    json_path = RESULTS_DIR / "feature_importance_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n✅ 分析结果已保存: {json_path}")
    
    return analysis_results


def main():
    """主函数"""
    print("=" * 80)
    print("  Tree模型特征重要性分析")
    print("=" * 80)
    
    try:
        results = analyze_feature_importance()
        
        if results:
            print("\n" + "=" * 80)
            print("  特征重要性分析完成")
            print("=" * 80)
            print(f"\n所有结果已保存到:")
            print(f"  - 图表: {FIGURES_DIR}")
            print(f"  - 数据: {RESULTS_DIR}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

