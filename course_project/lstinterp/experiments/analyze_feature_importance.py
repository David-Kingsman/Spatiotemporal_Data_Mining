"""
Analysis of Tree Model Feature Importance

This script loads a trained Tree model and analyzes feature importance.
This helps understand which factors (Space, Time) have a greater impact on LST prediction.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lstinterp.models.tree_baselines import TreeBaseline

# Output directory
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "feature_importance"
RESULTS_DIR = OUTPUT_DIR / "results" / "feature_importance"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def analyze_feature_importance(model_path):
    """Analyze feature importance"""
    print(f"\nLoading model: {model_path}")
    
    try:
        # Load the wrapper class
        tree_model = joblib.load(model_path)
        
        # Get the internal model (sklearn or xgboost)
        # Check for quantile models first (often stored in quantile_models dict)
        if hasattr(tree_model, 'quantile_models') and tree_model.quantile_models:
            # Use the median model (0.5) if available
            if 0.5 in tree_model.quantile_models:
                inner_model = tree_model.quantile_models[0.5]
                print("✅ Using 0.5 quantile model for feature importance.")
            else:
                # Use the first available model
                first_q = list(tree_model.quantile_models.keys())[0]
                inner_model = tree_model.quantile_models[first_q]
                print(f"ℹ️  Using {first_q} quantile model.")
        elif hasattr(tree_model, 'model'):
            inner_model = tree_model.model
            print("✅ Using base model.")
        else:
            print(f"⚠️  Cannot find internal model instance. Structure: {dir(tree_model)}")
            return

        print(f"✅ Internal model type: {type(inner_model)}")

        # Extract feature importance
        feature_names = ['Longitude', 'Latitude', 'Time']
        importance = None
        
        # 1. XGBoost / LightGBM / RandomForest
        if hasattr(inner_model, 'feature_importances_'):
            importance = inner_model.feature_importances_
        # 2. XGBoost (native API)
        elif hasattr(inner_model, 'get_score'): 
            # get_score returns a dict, need to map to feature names
            # Assuming features are f0, f1, f2 matching input order
            importance_dict = inner_model.get_score(importance_type='gain')
            # Map f0->Lon, f1->Lat, f2->Time if keys are f0...
            # Or if numpy array was passed, keys might be indices
            print(f"Debug - XGBoost scores: {importance_dict}")
            importance = np.zeros(3)
            for k, v in importance_dict.items():
                if k == 'f0': importance[0] = v
                elif k == 'f1': importance[1] = v
                elif k == 'f2': importance[2] = v
        else:
            print("⚠️  Model type does not support feature importance extraction directly.")
            return

        if importance is not None:
            # Normalize to sum to 100%
            importance = importance / importance.sum() * 100
            
            # Create DataFrame
            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            df = df.sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            print(df)
            
            # Visualize
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
            plt.title('Feature Importance (Tree Model)', fontsize=14, fontweight='bold')
            plt.xlabel('Importance (%)')
            plt.tight_layout()
            
            fig_path = FIGURES_DIR / "tree_feature_importance.png"
            plt.savefig(fig_path, dpi=300)
            print(f"✅ Plot saved: {fig_path}")
            
            # Save data
            csv_path = RESULTS_DIR / "tree_feature_importance.csv"
            df.to_csv(csv_path, index=False)
            
    except Exception as e:
        print(f"❌ Error analyzing features: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("=" * 80)
    print("  Tree Model Feature Importance Analysis")
    print("=" * 80)
    
    # Find model file
    model_dir = OUTPUT_DIR / "checkpoints" / "tree"
    model_path = model_dir / "tree_model.pkl"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    analyze_feature_importance(model_path)


if __name__ == "__main__":
    main()
