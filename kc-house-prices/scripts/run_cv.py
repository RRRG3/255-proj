import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from houseprice.config import DATA_PATH, OUT_DIR, RANDOM_STATE, N_FOLDS, TEST_SIZE
from houseprice.data import load_data, clean_data
from houseprice.features import engineer_features
from houseprice.preprocess import split_columns, make_preprocessors
from houseprice.models import make_linear, make_random_forest, make_xgb
from houseprice.plots import (
    plot_lr_residuals, 
    plot_lr_residuals_enhanced, 
    plot_tree_importance,
    plot_feature_importance_comparison,
    plot_linear_coefficients,
    plot_shap_summary
)

def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def format_comparison_table(df: pd.DataFrame) -> str:
    """Generate formatted text table from results DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: model, R2_CV_mean, R2_CV_std, RMSE_CV
    
    Returns
    -------
    str
        Formatted table as string
    """
    lines = []
    lines.append("Model Comparison Table")
    lines.append("=" * 65)
    lines.append(f"{'Model':<20} | {'Mean R²':>8} | {'R² SD':>8} | {'RMSE':>12}")
    lines.append("-" * 65)
    
    for _, row in df.iterrows():
        model_name = row["model"]
        r2_mean = row["R2_CV_mean"]
        r2_std = row["R2_CV_std"]
        rmse_val = row["RMSE_CV"]
        
        lines.append(f"{model_name:<20} | {r2_mean:>8.4f} | {r2_std:>8.4f} | {rmse_val:>12,.2f}")
    
    return "\n".join(lines)

def main(data_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & FE
    df = engineer_features(clean_data(load_data(data_path)))
    y_log = np.log1p(df["price"].values)
    X = df.drop(columns=["price"])

    # Split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Preprocessors
    num, cat = split_columns(X_train)
    prep_lr, prep_trees = make_preprocessors(num, cat)

    # Models
    lr_pipe, lr_grid = make_linear(prep_lr)
    rf_pipe, rf_grid = make_random_forest(prep_trees, RANDOM_STATE)
    xgb_pipe, xgb_grid = make_xgb(prep_trees, RANDOM_STATE)

    candidates = [
        ("LinearRegression", lr_pipe, lr_grid),
        ("RandomForest", rf_pipe, rf_grid),
    ]
    if xgb_pipe is not None:
        candidates.append(("XGBoost", xgb_pipe, xgb_grid))

    # CV compare
    results = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_models = {}
    for name, pipe, grid in candidates:
        gs = GridSearchCV(
            pipe,
            param_grid=grid if grid else [{}],
            cv=kf,
            scoring="r2",
            n_jobs=-1,
        )
        gs.fit(X_train, y_train_log)
        best = gs.best_estimator_
        
        # Get fold-level scores using cross_validate
        cv_results = cross_validate(
            best, X_train, y_train_log, cv=kf, 
            scoring="r2", n_jobs=-1, return_train_score=False
        )
        fold_scores = cv_results["test_score"]
        
        # Also get predictions for RMSE calculation
        y_oof_log = cross_val_predict(best, X_train, y_train_log, cv=kf, n_jobs=-1)
        y_true = np.expm1(y_train_log)
        y_pred = np.expm1(y_oof_log)
        
        res = {
            "model": name,
            "best_params": gs.best_params_,
            "R2_CV_mean": float(fold_scores.mean()),
            "R2_CV_std": float(fold_scores.std()),
            "RMSE_CV": float(rmse(y_true, y_pred)),
            "fold_scores": fold_scores.tolist(),
        }
        results.append(res)
        best_models[name] = best
        print(name, "→", json.dumps(res, indent=2))

    cv_df = pd.DataFrame(results).sort_values("RMSE_CV")
    
    # Save CSV (without fold_scores for cleaner output)
    cv_df_export = cv_df.drop(columns=["fold_scores"]).copy()
    cv_path = out_dir / "model_cv_results.csv"
    cv_df_export.to_csv(cv_path, index=False)
    print(f"Saved CV table → {cv_path}")
    
    # Generate and save formatted table
    formatted_table = format_comparison_table(cv_df_export)
    table_path = out_dir / "comparison_table.txt"
    table_path.write_text(formatted_table)
    print(f"Saved formatted table → {table_path}")
    print("\n" + formatted_table)

    # Pick best tree for importance plot
    tree_name = None
    for n in ["XGBoost", "RandomForest"]:
        if n in best_models:
            if tree_name is None: tree_name = n
            else:
                # pick the one with lower CV RMSE
                r = cv_df.set_index("model")
                tree_name = (
                    n
                    if r.loc[n, "RMSE_CV"] < r.loc[tree_name, "RMSE_CV"]
                    else tree_name
                )

    # Final test evaluation and plotting
    # Linear residuals on test (both original and enhanced)
    if "LinearRegression" in best_models:
        lr_best = best_models["LinearRegression"]
        lr_best.fit(X_train, y_train_log)
        plot_lr_residuals(lr_best, X_test, y_test_log, out_dir / "lr_residuals.png")
        plot_lr_residuals_enhanced(lr_best, X_test, y_test_log, out_dir / "lr_residuals_enhanced.png")
        
        # Linear coefficients plot
        plot_linear_coefficients(lr_best, prep_lr, out_dir / "lr_coefficients.png")

    # Tree importance and SHAP
    if tree_name:
        tree_best = best_models[tree_name]
        tree_best.fit(X_train, y_train_log)
        plot_tree_importance(tree_best, out_dir / f"{tree_name.lower()}_feature_importance.png")
        plot_shap_summary(tree_best, X_train, out_dir / f"{tree_name.lower()}_shap_summary.png")
    
    # Feature importance comparison (if both RF and XGBoost available)
    if "RandomForest" in best_models and "XGBoost" in best_models:
        rf_best = best_models["RandomForest"]
        xgb_best = best_models["XGBoost"]
        rf_best.fit(X_train, y_train_log)
        xgb_best.fit(X_train, y_train_log)
        plot_feature_importance_comparison(
            rf_best, xgb_best, prep_trees, 
            out_dir / "feature_importance_comparison.png"
        )

    # Evaluate the global winner on test (lowest CV RMSE)
    winner = cv_df.iloc[0]["model"]
    model = best_models[winner].fit(X_train, y_train_log)
    y_pred_test_log = model.predict(X_test)
    y_test = np.expm1(y_test_log); y_pred_test = np.expm1(y_pred_test_log)
    test_r2, test_rmse = r2_score(y_test, y_pred_test), rmse(y_test, y_pred_test)

    metrics_path = out_dir / "test_metrics.txt"
    metrics_path.write_text(
        f"Winner: {winner}\nR2_test: {test_r2:.4f}\nRMSE_test: {test_rmse:,.2f}\n"
    )
    print(metrics_path.read_text())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()
    main(args.data, args.out)
