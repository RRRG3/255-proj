"""XGBoost optimization experiment runner."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from houseprice.config import DATA_PATH, OUT_DIR, RANDOM_STATE, N_FOLDS, TEST_SIZE
from houseprice.data import load_data, clean_data
from houseprice.features import engineer_features, engineer_advanced_features
from houseprice.preprocess import split_columns, make_preprocessors
from houseprice.models import make_xgb, make_xgb_optimized
from houseprice.outliers import detect_outliers, remove_outliers
from houseprice.transforms import TargetTransformer


def rmse(y_true, y_pred):
    """Calculate RMSE."""
    return root_mean_squared_error(y_true, y_pred)


def run_baseline_experiment(X_train, y_train, X_test, y_test, kf, random_state):
    """Run baseline experiment with current approach."""
    print("\n" + "="*60)
    print("BASELINE EXPERIMENT")
    print("="*60)
    
    # Use log1p transformation (current approach)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # Preprocessors
    num, cat = split_columns(X_train)
    _, prep_trees = make_preprocessors(num, cat)
    
    # Original XGBoost model
    xgb_pipe, xgb_grid = make_xgb(prep_trees, random_state)
    
    # Grid search
    gs = GridSearchCV(xgb_pipe, param_grid=xgb_grid, cv=kf, scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train_log)
    best = gs.best_estimator_
    
    # Cross-validation predictions
    y_oof_log = cross_val_predict(best, X_train, y_train_log, cv=kf, n_jobs=-1)
    y_true_cv = np.expm1(y_train_log)
    y_pred_cv = np.expm1(y_oof_log)
    
    # Test predictions
    y_pred_test_log = best.predict(X_test)
    y_pred_test = np.expm1(y_pred_test_log)
    
    results = {
        "strategy": "baseline",
        "r2_cv": float(r2_score(y_true_cv, y_pred_cv)),
        "rmse_cv": float(rmse(y_true_cv, y_pred_cv)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "rmse_test": float(rmse(y_test, y_pred_test)),
        "best_params": gs.best_params_
    }
    
    print(f"CV R²: {results['r2_cv']:.4f}")
    print(f"CV RMSE: ${results['rmse_cv']:,.2f}")
    print(f"Test R²: {results['r2_test']:.4f}")
    print(f"Test RMSE: ${results['rmse_test']:,.2f}")
    
    return results


def run_hyperparams_experiment(X_train, y_train, X_test, y_test, kf, random_state, search_mode="fast"):
    """Run experiment with expanded hyperparameters."""
    print("\n" + "="*60)
    print(f"HYPERPARAMETERS EXPERIMENT (mode={search_mode})")
    print("="*60)
    
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    num, cat = split_columns(X_train)
    _, prep_trees = make_preprocessors(num, cat)
    
    # Optimized XGBoost with expanded grid
    xgb_pipe, xgb_grid = make_xgb_optimized(prep_trees, random_state, search_mode=search_mode)
    
    print(f"Grid size: {np.prod([len(v) for v in xgb_grid.values()])} combinations")
    
    gs = GridSearchCV(xgb_pipe, param_grid=xgb_grid, cv=kf, scoring="r2", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train_log)
    best = gs.best_estimator_
    
    y_oof_log = cross_val_predict(best, X_train, y_train_log, cv=kf, n_jobs=-1)
    y_true_cv = np.expm1(y_train_log)
    y_pred_cv = np.expm1(y_oof_log)
    
    y_pred_test_log = best.predict(X_test)
    y_pred_test = np.expm1(y_pred_test_log)
    
    results = {
        "strategy": f"hyperparams_{search_mode}",
        "r2_cv": float(r2_score(y_true_cv, y_pred_cv)),
        "rmse_cv": float(rmse(y_true_cv, y_pred_cv)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "rmse_test": float(rmse(y_test, y_pred_test)),
        "best_params": gs.best_params_
    }
    
    print(f"Best params: {gs.best_params_}")
    print(f"CV R²: {results['r2_cv']:.4f}")
    print(f"CV RMSE: ${results['rmse_cv']:,.2f}")
    print(f"Test R²: {results['r2_test']:.4f}")
    print(f"Test RMSE: ${results['rmse_test']:,.2f}")
    
    return results


def run_features_experiment(X_train_raw, y_train, X_test_raw, y_test, kf, random_state):
    """Run experiment with advanced features."""
    print("\n" + "="*60)
    print("ADVANCED FEATURES EXPERIMENT")
    print("="*60)
    
    # Apply advanced feature engineering
    # Compute zipcode stats from training data only
    df_train_with_price = X_train_raw.copy()
    df_train_with_price["price"] = y_train
    X_train_adv = engineer_advanced_features(df_train_with_price, include_location_stats=True)
    
    # Extract zipcode stats for test set
    if "zipcode_median_price" in X_train_adv.columns:
        zipcode_stats = {
            "median": X_train_adv.groupby("zipcode")["zipcode_median_price"].first().to_dict(),
            "std": X_train_adv.groupby("zipcode")["zipcode_price_std"].first().to_dict() if "zipcode_price_std" in X_train_adv.columns else {}
        }
    else:
        zipcode_stats = None
    
    # Remove price column
    X_train_adv = X_train_adv.drop(columns=["price"])
    
    # Apply to test set with same stats
    df_test_with_price = X_test_raw.copy()
    df_test_with_price["price"] = y_test
    X_test_adv = engineer_advanced_features(df_test_with_price, include_location_stats=True, zipcode_stats=zipcode_stats)
    X_test_adv = X_test_adv.drop(columns=["price"])
    
    print(f"Features: {X_train_raw.shape[1]} → {X_train_adv.shape[1]} (+{X_train_adv.shape[1] - X_train_raw.shape[1]})")
    
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    num, cat = split_columns(X_train_adv)
    _, prep_trees = make_preprocessors(num, cat)
    
    xgb_pipe, xgb_grid = make_xgb(prep_trees, random_state)
    
    gs = GridSearchCV(xgb_pipe, param_grid=xgb_grid, cv=kf, scoring="r2", n_jobs=-1)
    gs.fit(X_train_adv, y_train_log)
    best = gs.best_estimator_
    
    y_oof_log = cross_val_predict(best, X_train_adv, y_train_log, cv=kf, n_jobs=-1)
    y_true_cv = np.expm1(y_train_log)
    y_pred_cv = np.expm1(y_oof_log)
    
    y_pred_test_log = best.predict(X_test_adv)
    y_pred_test = np.expm1(y_pred_test_log)
    
    results = {
        "strategy": "advanced_features",
        "r2_cv": float(r2_score(y_true_cv, y_pred_cv)),
        "rmse_cv": float(rmse(y_true_cv, y_pred_cv)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "rmse_test": float(rmse(y_test, y_pred_test)),
        "n_features": X_train_adv.shape[1]
    }
    
    print(f"CV R²: {results['r2_cv']:.4f}")
    print(f"CV RMSE: ${results['rmse_cv']:,.2f}")
    print(f"Test R²: {results['r2_test']:.4f}")
    print(f"Test RMSE: ${results['rmse_test']:,.2f}")
    
    return results


def run_outliers_experiment(X_train_raw, y_train, X_test, y_test, kf, random_state):
    """Run experiment with outlier removal."""
    print("\n" + "="*60)
    print("OUTLIER REMOVAL EXPERIMENT")
    print("="*60)
    
    # Detect and remove outliers from training set
    df_train = X_train_raw.copy()
    df_train["price"] = y_train
    
    outlier_mask = detect_outliers(df_train, columns=["price", "sqft_living"], n_std=3.0)
    n_outliers = outlier_mask.sum()
    print(f"Outliers detected: {n_outliers} ({100*n_outliers/len(df_train):.1f}%)")
    
    df_train_clean = remove_outliers(df_train, outlier_mask)
    X_train_clean = df_train_clean.drop(columns=["price"])
    y_train_clean = df_train_clean["price"].values
    
    print(f"Training samples: {len(X_train_raw)} → {len(X_train_clean)}")
    
    y_train_log = np.log1p(y_train_clean)
    y_test_log = np.log1p(y_test)
    
    num, cat = split_columns(X_train_clean)
    _, prep_trees = make_preprocessors(num, cat)
    
    xgb_pipe, xgb_grid = make_xgb(prep_trees, random_state)
    
    gs = GridSearchCV(xgb_pipe, param_grid=xgb_grid, cv=kf, scoring="r2", n_jobs=-1)
    gs.fit(X_train_clean, y_train_log)
    best = gs.best_estimator_
    
    y_oof_log = cross_val_predict(best, X_train_clean, y_train_log, cv=kf, n_jobs=-1)
    y_true_cv = np.expm1(y_train_log)
    y_pred_cv = np.expm1(y_oof_log)
    
    y_pred_test_log = best.predict(X_test)
    y_pred_test = np.expm1(y_pred_test_log)
    
    results = {
        "strategy": "outlier_removal",
        "r2_cv": float(r2_score(y_true_cv, y_pred_cv)),
        "rmse_cv": float(rmse(y_true_cv, y_pred_cv)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "rmse_test": float(rmse(y_test, y_pred_test)),
        "n_outliers_removed": int(n_outliers)
    }
    
    print(f"CV R²: {results['r2_cv']:.4f}")
    print(f"CV RMSE: ${results['rmse_cv']:,.2f}")
    print(f"Test R²: {results['r2_test']:.4f}")
    print(f"Test RMSE: ${results['rmse_test']:,.2f}")
    
    return results


def run_combined_experiment(X_train_raw, y_train, X_test_raw, y_test, kf, random_state, search_mode="fast"):
    """Run experiment with all optimizations combined."""
    print("\n" + "="*60)
    print(f"COMBINED OPTIMIZATIONS EXPERIMENT (mode={search_mode})")
    print("="*60)
    
    # 1. Remove outliers
    df_train = X_train_raw.copy()
    df_train["price"] = y_train
    outlier_mask = detect_outliers(df_train, columns=["price", "sqft_living"], n_std=3.0)
    df_train_clean = remove_outliers(df_train, outlier_mask)
    X_train_clean = df_train_clean.drop(columns=["price"])
    y_train_clean = df_train_clean["price"].values
    print(f"Outliers removed: {outlier_mask.sum()}")
    
    # 2. Advanced features
    df_train_fe = X_train_clean.copy()
    df_train_fe["price"] = y_train_clean
    X_train_adv = engineer_advanced_features(df_train_fe, include_location_stats=True)
    
    # Extract zipcode stats
    if "zipcode_median_price" in X_train_adv.columns:
        zipcode_stats = {
            "median": X_train_adv.groupby("zipcode")["zipcode_median_price"].first().to_dict(),
            "std": X_train_adv.groupby("zipcode")["zipcode_price_std"].first().to_dict() if "zipcode_price_std" in X_train_adv.columns else {}
        }
    else:
        zipcode_stats = None
    
    X_train_adv = X_train_adv.drop(columns=["price"])
    
    df_test_fe = X_test_raw.copy()
    df_test_fe["price"] = y_test
    X_test_adv = engineer_advanced_features(df_test_fe, include_location_stats=True, zipcode_stats=zipcode_stats)
    X_test_adv = X_test_adv.drop(columns=["price"])
    
    print(f"Features: {X_train_raw.shape[1]} → {X_train_adv.shape[1]}")
    
    # 3. Optimized hyperparameters
    y_train_log = np.log1p(y_train_clean)
    y_test_log = np.log1p(y_test)
    
    num, cat = split_columns(X_train_adv)
    _, prep_trees = make_preprocessors(num, cat)
    
    xgb_pipe, xgb_grid = make_xgb_optimized(prep_trees, random_state, search_mode=search_mode)
    
    print(f"Grid size: {np.prod([len(v) for v in xgb_grid.values()])} combinations")
    
    gs = GridSearchCV(xgb_pipe, param_grid=xgb_grid, cv=kf, scoring="r2", n_jobs=-1, verbose=1)
    gs.fit(X_train_adv, y_train_log)
    best = gs.best_estimator_
    
    y_oof_log = cross_val_predict(best, X_train_adv, y_train_log, cv=kf, n_jobs=-1)
    y_true_cv = np.expm1(y_train_log)
    y_pred_cv = np.expm1(y_oof_log)
    
    y_pred_test_log = best.predict(X_test_adv)
    y_pred_test = np.expm1(y_pred_test_log)
    
    results = {
        "strategy": f"combined_{search_mode}",
        "r2_cv": float(r2_score(y_true_cv, y_pred_cv)),
        "rmse_cv": float(rmse(y_true_cv, y_pred_cv)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "rmse_test": float(rmse(y_test, y_pred_test)),
        "best_params": gs.best_params_,
        "n_features": X_train_adv.shape[1]
    }
    
    print(f"Best params: {gs.best_params_}")
    print(f"CV R²: {results['r2_cv']:.4f}")
    print(f"CV RMSE: ${results['rmse_cv']:,.2f}")
    print(f"Test R²: {results['r2_test']:.4f}")
    print(f"Test RMSE: ${results['rmse_test']:,.2f}")
    
    return results


def main(data_path: Path, out_dir: Path, strategies: list, search_mode: str = "fast"):
    """Run optimization experiments."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = clean_data(load_data(data_path))
    
    # Basic feature engineering for baseline
    df_fe = engineer_features(df)
    
    # Save the engineered features for notebook use
    xgb_features_path = data_path.parent / "kc_house_data_xgboost_features.csv"
    df_fe.to_csv(xgb_features_path, index=False)
    print(f"Saved XGBoost features to {xgb_features_path}")
    
    y = df_fe["price"].values
    X = df_fe.drop(columns=["price"])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Cross-validation setup
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Run experiments
    all_results = []
    
    if "baseline" in strategies:
        results = run_baseline_experiment(X_train, y_train, X_test, y_test, kf, RANDOM_STATE)
        all_results.append(results)
    
    if "hyperparams" in strategies:
        results = run_hyperparams_experiment(X_train, y_train, X_test, y_test, kf, RANDOM_STATE, search_mode=search_mode)
        all_results.append(results)
    
    if "features" in strategies:
        results = run_features_experiment(X_train, y_train, X_test, y_test, kf, RANDOM_STATE)
        all_results.append(results)
    
    if "outliers" in strategies:
        results = run_outliers_experiment(X_train, y_train, X_test, y_test, kf, RANDOM_STATE)
        all_results.append(results)
    
    if "combined" in strategies:
        results = run_combined_experiment(X_train, y_train, X_test, y_test, kf, RANDOM_STATE, search_mode=search_mode)
        all_results.append(results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = out_dir / "optimization_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS COMPARISON")
    print("="*80)
    print(results_df[["strategy", "r2_cv", "rmse_cv", "r2_test", "rmse_test"]].to_string(index=False))
    
    # Calculate improvements over baseline
    if "baseline" in strategies:
        baseline_rmse = results_df[results_df["strategy"] == "baseline"]["rmse_test"].values[0]
        results_df["improvement_pct"] = (
            (baseline_rmse - results_df["rmse_test"]) / baseline_rmse * 100
        )
        print("\n" + "="*80)
        print("IMPROVEMENT OVER BASELINE")
        print("="*80)
        print(results_df[["strategy", "rmse_test", "improvement_pct"]].to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost optimization experiments")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to data file")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument(
        "--strategies", 
        nargs="+", 
        default=["baseline", "hyperparams", "features", "outliers", "combined"],
        choices=["baseline", "hyperparams", "features", "outliers", "combined"],
        help="Optimization strategies to run"
    )
    parser.add_argument(
        "--search-mode",
        choices=["fast", "full"],
        default="fast",
        help="Grid search mode (fast or full)"
    )
    
    args = parser.parse_args()
    main(args.data, args.out, args.strategies, args.search_mode)
