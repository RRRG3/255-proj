from pathlib import Path
import pandas as pd
from scipy import stats
import sys

# Add src to path just in case
sys.path.insert(0, str(Path.cwd().parent / 'src'))
from houseprice.config import OUT_DIR

def run_significance_test():
    folds_path = Path.cwd() / OUT_DIR / "model_cv_folds.csv"
    if not folds_path.exists():
        print(f"Error: {folds_path} not found. Run scripts/run_cv.py first.")
        return

    df = pd.read_csv(folds_path)
    
    # Get models
    models = df["model"].unique()
    print(f"Available models for testing: {models}")
    
    # We want to compare XGBoost vs RandomForest (or whatever is top 2)
    if "XGBoost" in models and "RandomForest" in models:
        xgb_scores = df[df["model"] == "XGBoost"].sort_values("fold")["r2"].values
        rf_scores = df[df["model"] == "RandomForest"].sort_values("fold")["r2"].values
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(xgb_scores, rf_scores)
        
        print("\nStatistical Significance Test (Paired t-test)")
        print("=============================================")
        print(f"Comparison: XGBoost vs RandomForest")
        print(f"H0: Mean R2 of XGBoost == Mean R2 of RandomForest")
        print(f"H1: Mean R2 of XGBoost != Mean R2 of RandomForest")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value:     {p_val:.6f}")
        
        alpha = 0.05
        if p_val < alpha:
            print(f"\nResult: REJECT H0 (p < {alpha})")
            print("Conclusion: There is a statistically significant difference between XGBoost and Random Forest.")
        else:
            print(f"\nResult: FAIL TO REJECT H0 (p >= {alpha})")
            print("Conclusion: The difference between models is NOT statistically significant.")
            
    # Also compare best vs Linear Regression
    if "LinearRegression" in models:
        # Find best non-linear model
        best_model = "XGBoost" if "XGBoost" in models else "RandomForest"
        if best_model in models:
            best_scores = df[df["model"] == best_model].sort_values("fold")["r2"].values
            lr_scores = df[df["model"] == "LinearRegression"].sort_values("fold")["r2"].values
            
            t_stat, p_val = stats.ttest_rel(best_scores, lr_scores)
            
            print(f"\nComparison: {best_model} vs Linear Regression")
            print(f"P-value:     {p_val:.6e}") # scientific notation for likely very small p
            if p_val < 0.05:
                 print("Conclusion: Significant improvement over baseline.")

if __name__ == "__main__":
    run_significance_test()
