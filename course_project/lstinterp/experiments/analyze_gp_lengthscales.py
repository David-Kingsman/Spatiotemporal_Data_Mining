"""
Analysis of GP Model Lengthscales

This script loads a trained GP model and analyzes its learned lengthscales.
This helps understand the spatial and temporal correlation structures captured by the model.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gpytorch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import configuration classes (needed for loading the model)
from lstinterp.config import GPSTConfig
from lstinterp.models.gp_st import STSeparableGP, STAdditiveGP, STNonSeparableGP

# Output directory
OUTPUT_DIR = project_root / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "gp_analysis"
RESULTS_DIR = OUTPUT_DIR / "results" / "gp_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def analyze_separable_gp(model_path):
    """Analyze Separable GP model"""
    print("\nAnalyzing Separable GP Model...")
    
    try:
        # Load checkpoint
        # Add weights_only=False to allow loading arbitrary objects (like Config)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config if not present
            print("Warning: Config not found in checkpoint, using default.")
            config = GPSTConfig(kernel_design='separable')
            
        # Create dummy inducing points to initialize model structure
        # We don't need exact inducing points to inspect kernels, but GPyTorch needs them for init
        dummy_inducing = torch.zeros(config.num_inducing, 3) 
        
        model = STSeparableGP(dummy_inducing, config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Extract lengthscales
        # Note: Parameter names depend on model structure
        print("\nExtracted Parameters:")
        
        # Space Kernel
        # Usually raw_lengthscale is stored, need to pass through constraint to get actual value
        # But model.covar_space.base_kernel.lengthscale should return the actual value
        lengthscale_space = model.covar_space.base_kernel.lengthscale.detach().numpy()
        print(f"  Spatial Lengthscale: {float(lengthscale_space.flat[0]):.4f} (normalized units)")
        
        # Time Kernel
        lengthscale_time = model.covar_time.base_kernel.lengthscale.detach().numpy()
        print(f"  Temporal Lengthscale: {float(lengthscale_time.flat[0]):.4f} (normalized units)")
        
        # Output Scale
        output_scale_space = model.covar_space.outputscale.detach().numpy()
        output_scale_time = model.covar_time.outputscale.detach().numpy()
        print(f"  Spatial Output Scale: {float(output_scale_space):.4f}")
        print(f"  Temporal Output Scale: {float(output_scale_time):.4f}")
        
        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of lengthscales
        scales = [float(lengthscale_space.flat[0]), float(lengthscale_time.flat[0])]
        labels = ['Spatial (Lat/Lon)', 'Temporal (Time)']
        
        ax[0].bar(labels, scales, color=['skyblue', 'lightgreen'])
        ax[0].set_title('Learned Lengthscales (Normalized Units)')
        ax[0].set_ylabel('Lengthscale')
        ax[0].grid(True, alpha=0.3, axis='y')
        
        # Visualize correlation decay
        x = np.linspace(0, 3, 100) # Distance in normalized units
        
        # Space correlation (Matern 3/2 or 5/2)
        # Simplified Matern correlation function visualization
        def matern32(d, l):
            return (1 + np.sqrt(3)*d/l) * np.exp(-np.sqrt(3)*d/l)
            
        y_space = matern32(x, float(lengthscale_space.flat[0]))
        y_time = matern32(x, float(lengthscale_time.flat[0]))
        
        ax[1].plot(x, y_space, label='Spatial Correlation', linewidth=2)
        ax[1].plot(x, y_time, label='Temporal Correlation', linewidth=2)
        ax[1].set_title('Correlation Decay vs. Distance')
        ax[1].set_xlabel('Distance (Normalized)')
        ax[1].set_ylabel('Correlation')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "separable_gp_kernels.png"
        plt.savefig(fig_path, dpi=300)
        print(f"✅ Visualization saved: {fig_path}")
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("=" * 80)
    print("  GP Model Parameter Analysis")
    print("=" * 80)
    
    # Find best model
    model_dir = OUTPUT_DIR / "checkpoints" / "gp"
    best_model_path = model_dir / "best_model.pth"
    
    if not best_model_path.exists():
        print(f"❌ Model not found: {best_model_path}")
        # Try finding by name pattern
        candidates = list(model_dir.glob("best_model_*.pth"))
        if candidates:
            best_model_path = candidates[0]
            print(f"ℹ️  Using alternative model: {best_model_path}")
        else:
            return

    analyze_separable_gp(best_model_path)

if __name__ == "__main__":
    main()
