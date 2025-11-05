import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from houseprice.config import DATA_PATH, OUT_DIR, RANDOM_STATE, N_FOLDS, TEST_SIZE
from houseprice.data import load_data
from houseprice.features import engineer_features
from houseprice.preprocess import split_columns, make_preprocessors
from houseprice.models import make_linear, make_random_forest, make_xgb
from houseprice.plots import plot_lr_residuals, plot_tree_importance

def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

def main(data_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & FE
    df = engineer_features(load_data(data_path))
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
        y_oof_log = cross_val_predict(best, X_train, y_train_log, cv=kf, n_jobs=-1)
        y_true = np.expm1(y_train_log)
        y_pred = np.expm1(y_oof_log)
        res = {
            "model": name,
            "best_params": gs.best_params_,
            "R2_CV": float(r2_score(y_true, y_pred)),
            "RMSE_CV": float(rmse(y_true, y_pred)),
        }
        results.append(res)
        best_models[name] = best
        print(name, "→", json.dumps(res, indent=2))

    cv_df = pd.DataFrame(results).sort_values("RMSE_CV")
    cv_path = out_dir / "model_cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"Saved CV table → {cv_path}")

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

    # Final test evaluation
    # Linear residuals on test
    if "LinearRegression" in best_models:
        lr_best = best_models["LinearRegression"]
        lr_best.fit(X_train, y_train_log)
        plot_lr_residuals(lr_best, X_test, y_test_log, out_dir / "lr_residuals.png")

    # Tree importance
    if tree_name:
        tree_best = best_models[tree_name]
        tree_best.fit(X_train, y_train_log)
        plot_tree_importance(tree_best, out_dir / "tree_feature_importance.png")

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
