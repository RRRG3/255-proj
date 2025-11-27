
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_comparison_chart(csv_path: Path, output_path: Path):
    # Load results
    df = pd.read_csv(csv_path)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² comparison
    models = df['model'].values
    r2_means = df['R2_CV_mean'].values
    r2_stds = df['R2_CV_std'].values
    
    colors = ['#2ecc71' if r2 > 0.85 else '#3498db' if r2 > 0.75 else '#e74c3c' 
              for r2 in r2_means]
    
    bars1 = ax1.bar(models, r2_means, yerr=r2_stds, capsize=5, 
                    color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance - R² Score', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val, std in zip(bars1, r2_means, r2_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{val:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # RMSE comparison
    rmse_values = df['RMSE_CV'].values
    
    colors2 = ['#2ecc71' if rmse < 140000 else '#3498db' if rmse < 180000 else '#e74c3c' 
               for rmse in rmse_values]
    
    bars2 = ax2.bar(models, rmse_values, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance - RMSE', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, rmse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Rotate x-axis labels if needed
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison chart saved to {output_path}")


if __name__ == "__main__":
    from houseprice.config import OUT_DIR
    
    csv_path = OUT_DIR / "model_cv_results.csv"
    output_path = OUT_DIR / "model_comparison_chart.png"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run run_cv.py first.")
        exit(1)
    
    generate_comparison_chart(csv_path, output_path)
