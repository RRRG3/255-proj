"""
XGBoost Model Visualization Script

Generates 5 essential visualizations for XGBoost model analysis:
1. Predictions vs Actual scatter plot
2. Residuals plot
3. Partial dependence plots (top 3-4 features)
4. Prediction error distribution
5. Learning curves

Usage:
    python scripts/visualize_xgboost.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    data_path = Path("data/kc_house_data_xgboost_features.csv")
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    drop_cols = ["price", "log_price", "id", "date"]
    drop_cols_present = [col for col in drop_cols if col in df.columns]
    
    y = np.log1p(df["price"].values)
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    
    print(f"Dataset shape: {X.shape}")
    return X, y, feature_cols


def train_model(X_train, y_train):
    """Train XGBoost model with best parameters"""
    print("\nTraining XGBoost model...")
    
    # Best parameters from hyperparameter tuning
    best_params = {
        'colsample_bytree': 0.8,
        'gamma': 0,
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 7,
        'n_estimators': 600,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'subsample': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }
    
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    
    return model


def plot_predictions_vs_actual(y_test, y_pred, y_test_dollars, y_pred_dollars, output_dir):
    """1. Predictions vs Actual scatter plot"""
    print("\n1. Creating Predictions vs Actual plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Log space
    ax1.scatter(y_test, y_pred, alpha=0.3, s=10)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Log(Price)')
    ax1.set_ylabel('Predicted Log(Price)')
    ax1.set_title('Predictions vs Actual (Log Space)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dollar space
    ax2.scatter(y_test_dollars, y_pred_dollars, alpha=0.3, s=10)
    ax2.plot([y_test_dollars.min(), y_test_dollars.max()], 
             [y_test_dollars.min(), y_test_dollars.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.set_title('Predictions vs Actual (Dollar Space)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format dollar axis
    ax2.ticklabel_format(style='plain', axis='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: xgboost_predictions_vs_actual.png")
    plt.close()


def plot_residuals(y_test, y_pred, y_test_dollars, y_pred_dollars, output_dir):
    """2. Residuals plot"""
    print("\n2. Creating Residuals plot...")
    
    residuals_log = y_test - y_pred
    residuals_dollars = y_test_dollars - y_pred_dollars
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals vs Predicted (log space)
    axes[0, 0].scatter(y_pred, residuals_log, alpha=0.3, s=10)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Log(Price)')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted (Log Space)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted (dollar space)
    axes[0, 1].scatter(y_pred_dollars, residuals_dollars, alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residuals vs Predicted (Dollar Space)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='plain', axis='both')
    
    # Histogram of residuals (log space)
    axes[1, 0].hist(residuals_log, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals (Log Space)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals (Log Space)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals (dollar space)
    axes[1, 1].hist(residuals_dollars, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Residuals ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals (Dollar Space)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_residuals.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: xgboost_residuals.png")
    plt.close()


def plot_partial_dependence(model, X_train, feature_cols, output_dir):
    """3. Partial dependence plots for top features"""
    print("\n3. Creating Partial Dependence plots...")
    
    # Get top 4 features by importance
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:4]
    top_features = [feature_cols[i] for i in top_indices]
    
    print(f"   Top 4 features: {top_features}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    display = PartialDependenceDisplay.from_estimator(
        model, 
        X_train, 
        features=top_indices.tolist(),
        ax=axes,
        n_cols=2,
        grid_resolution=50
    )
    
    fig.suptitle('Partial Dependence Plots - Top 4 Features', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_partial_dependence.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: xgboost_partial_dependence.png")
    plt.close()


def plot_error_distribution(y_test_dollars, y_pred_dollars, output_dir):
    """4. Prediction error distribution"""
    print("\n4. Creating Error Distribution plot...")
    
    errors = y_pred_dollars - y_test_dollars
    percentage_errors = (errors / y_test_dollars) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Absolute errors histogram
    axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Prediction Error ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Prediction Errors')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].ticklabel_format(style='plain', axis='x')
    
    # Percentage errors histogram
    axes[0, 1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Percentage Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Percentage Errors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Absolute errors boxplot
    axes[1, 0].boxplot(errors, vert=True)
    axes[1, 0].set_ylabel('Prediction Error ($)')
    axes[1, 0].set_title('Boxplot of Prediction Errors')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    
    # Error statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Error Statistics:
    
    Mean Error: ${errors.mean():,.0f}
    Median Error: ${np.median(errors):,.0f}
    Std Dev: ${errors.std():,.0f}
    
    Mean Abs Error: ${np.abs(errors).mean():,.0f}
    Median Abs Error: ${np.median(np.abs(errors)):,.0f}
    
    Mean % Error: {percentage_errors.mean():.2f}%
    Median % Error: {np.median(percentage_errors):.2f}%
    
    Within ±10%: {(np.abs(percentage_errors) <= 10).sum() / len(percentage_errors) * 100:.1f}%
    Within ±20%: {(np.abs(percentage_errors) <= 20).sum() / len(percentage_errors) * 100:.1f}%
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: xgboost_error_distribution.png")
    plt.close()


def plot_learning_curves(model, X_train, y_train, output_dir):
    """5. Learning curves"""
    print("\n5. Creating Learning Curves...")
    print("   This may take a few minutes...")
    
    # Calculate learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Learning curve
    ax1.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                      alpha=0.2, color='blue')
    ax1.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
    ax1.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                      alpha=0.2, color='red')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Learning Curves')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Gap between training and validation
    gap = train_mean - val_mean
    ax2.plot(train_sizes_abs, gap, 'o-', color='purple')
    ax2.fill_between(train_sizes_abs, 0, gap, alpha=0.3, color='purple')
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Training - Validation Gap')
    ax2.set_title('Overfitting Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', lw=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: xgboost_learning_curves.png")
    plt.close()


def main():
    """Main execution function"""
    print("="*60)
    print("XGBoost Model Visualization Script")
    print("="*60)
    
    # Setup output directory
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, feature_cols = load_data()
    
    # Train/test split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    y_test_dollars = np.expm1(y_test)
    y_pred_dollars = np.expm1(y_pred)
    
    # Print metrics
    r2 = r2_score(y_test_dollars, y_pred_dollars)
    rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
    print(f"\nModel Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:,.0f}")
    
    # Generate all visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    plot_predictions_vs_actual(y_test, y_pred, y_test_dollars, y_pred_dollars, output_dir)
    plot_residuals(y_test, y_pred, y_test_dollars, y_pred_dollars, output_dir)
    plot_partial_dependence(model, X_train, feature_cols, output_dir)
    plot_error_distribution(y_test_dollars, y_pred_dollars, output_dir)
    plot_learning_curves(model, X_train, y_train, output_dir)
    
    print("\n" + "="*60)
    print("All visualizations completed successfully!")
    print(f"Saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
