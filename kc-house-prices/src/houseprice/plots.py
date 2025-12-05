from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.compose import ColumnTransformer

def plot_lr_residuals(model, X_test, y_test_log, outpath: Path):
    """Simple residual plot for linear regression"""
    y_pred_log = model.predict(X_test)
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)
    residuals = y_true - y_pred
    
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residual ($ actual - $ predicted)")
    plt.title("Linear Regression Residuals (test)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_lr_residuals_enhanced(model, X_test, y_test_log, outpath: Path):
    """Fancier residual plot with histogram"""
    y_pred_log = model.predict(X_test)
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # scatter with color coding
    colors = ['red' if r < 0 else 'blue' for r in residuals]
    ax1.scatter(y_pred, residuals, alpha=0.4, c=colors, s=10)
    ax1.axhline(0, linestyle="--", color='black', linewidth=2)
    ax1.set_xlabel("Predicted Price ($)")
    ax1.set_ylabel("Residual ($ actual - $ predicted)")
    ax1.set_title("Linear Regression Residuals")
    
    ax1.text(0.05, 0.95, 'Over-prediction (blue)', transform=ax1.transAxes,
             verticalalignment='top', color='blue', fontsize=9)
    ax1.text(0.05, 0.90, 'Under-prediction (red)', transform=ax1.transAxes,
             verticalalignment='top', color='red', fontsize=9)
    
    # histogram of residuals
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, linestyle="--", color='red', linewidth=2)
    ax2.set_xlabel("Residual ($)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Residuals")
    
    mean_res = residuals.mean()
    std_res = residuals.std()
    ax2.text(0.05, 0.95, f'Mean: ${mean_res:,.0f}\nStd: ${std_res:,.0f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()

def _feature_names(ct: ColumnTransformer):
    """Extract feature names from column transformer"""
    names = []
    for name, trans, cols in ct.transformers:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                names.extend(list(trans.get_feature_names_out(cols)))
            except Exception:
                names.extend(list(cols))
        else:
            names.extend(list(cols))
    return names

def plot_tree_importance(tree_pipe, outpath: Path, top_k: int = 20):
    """Plot feature importances from tree model"""
    prep = tree_pipe.named_steps["prep"]
    model = tree_pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    
    names = _feature_names(prep)
    importances = pd.Series(model.feature_importances_, index=names)
    top = importances.sort_values(ascending=False).head(top_k)[::-1]
    
    plt.figure()
    plt.barh(top.index, top.values)
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_feature_importance_comparison(rf_model, xgb_model, prep, outpath: Path, top_k: int = 15):
    """Side-by-side comparison of RF vs XGBoost feature importance"""
    names = _feature_names(prep)
    
    rf_imp = pd.Series(rf_model.named_steps["model"].feature_importances_, index=names)
    xgb_imp = pd.Series(xgb_model.named_steps["model"].feature_importances_, index=names)
    
    # get top features from both (union)
    top_rf = set(rf_imp.nlargest(top_k).index)
    top_xgb = set(xgb_imp.nlargest(top_k).index)
    top_features = list(top_rf | top_xgb)
    
    # sort by average importance
    avg_imp = (rf_imp[top_features] + xgb_imp[top_features]) / 2
    top_features = avg_imp.sort_values(ascending=True).index.tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # RF
    rf_top = rf_imp[top_features]
    ax1.barh(range(len(rf_top)), rf_top.values, color='steelblue')
    ax1.set_yticks(range(len(rf_top)))
    ax1.set_yticklabels(rf_top.index, fontsize=8)
    ax1.set_xlabel("Importance")
    ax1.set_title("Random Forest Feature Importance")
    
    # XGBoost
    xgb_top = xgb_imp[top_features]
    ax2.barh(range(len(xgb_top)), xgb_top.values, color='darkorange')
    ax2.set_yticks(range(len(xgb_top)))
    ax2.set_yticklabels(xgb_top.index, fontsize=8)
    ax2.set_xlabel("Importance")
    ax2.set_title("XGBoost Feature Importance")
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


def plot_shap_summary(model, X_train, outpath: Path):
    """SHAP summary plot for tree model"""
    prep = model.named_steps["prep"]
    tree_model = model.named_steps["model"]
    
    X_train_transformed = prep.transform(X_train)
    feature_names = _feature_names(prep)
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_train_transformed_df)

    plt.figure()
    shap.summary_plot(shap_values, X_train_transformed_df, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_linear_coefficients(lr_model, prep, outpath: Path, top_k: int = 15):
    """Plot linear regression coefficients"""
    names = _feature_names(prep)
    coefficients = lr_model.named_steps["model"].coef_
    
    coef_series = pd.Series(coefficients, index=names)
    
    # get top by absolute value
    top_coefs = coef_series.abs().nlargest(top_k)
    top_features = top_coefs.index
    
    plot_coefs = coef_series[top_features].sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red' if c < 0 else 'green' for c in plot_coefs.values]
    ax.barh(range(len(plot_coefs)), plot_coefs.values, color=colors)
    ax.set_yticks(range(len(plot_coefs)))
    ax.set_yticklabels(plot_coefs.index, fontsize=8)
    ax.set_xlabel("Coefficient Value")
    ax.set_title(f"Top {top_k} Linear Regression Coefficients")
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()
