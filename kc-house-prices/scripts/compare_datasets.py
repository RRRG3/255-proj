"""
Compare XGBoost on two datasets:
1. kc_house_data_xgboost_features.csv
2. kc_house_data_encoded_tree_step3.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.2


def train_and_evaluate(df, dataset_name, target_col='log_price'):
    """Train and eval XGBoost."""
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    
    # check if log_price exists
    if target_col not in df.columns:
        if 'price' in df.columns:
            df[target_col] = np.log1p(df['price'])
            print(f"Created {target_col} from price")
        else:
            raise ValueError(f"No '{target_col}' or 'price' column")
    
    # prep features
    drop_cols = [target_col]
    if 'price' in df.columns:
        drop_cols.append('price')
    if 'id' in df.columns:
        drop_cols.append('id')
    if 'date' in df.columns:
        drop_cols.append('date')
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Shape: {df.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # xgboost with same hyperparams for fair comparison
    model = XGBRegressor(
        random_state=RANDOM_STATE,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        objective="reg:squarederror"
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # metrics (log scale)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse_log = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # metrics (dollars)
    y_test_dollars = np.expm1(y_test)
    y_test_pred_dollars = np.expm1(y_test_pred)
    test_rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, y_test_pred_dollars))
    
    print(f"\n{'-'*80}")
    print("RESULTS:")
    print(f"{'-'*80}")
    print(f"Train R²:         {train_r2:.6f}")
    print(f"Test R²:          {test_r2:.6f}")
    print(f"Test RMSE (log):  {test_rmse_log:.6f}")
    print(f"Test RMSE ($):    ${test_rmse_dollars:,.2f}")
    print(f"{'-'*80}")
    
    return {
        'dataset': dataset_name,
        'n_features': len(feature_cols),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse_log': test_rmse_log,
        'test_rmse_dollars': test_rmse_dollars,
        'feature_names': feature_cols
    }


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    
    print("="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    # dataset 1
    print("\n\n" + "="*80)
    print("DATASET 1: kc_house_data_xgboost_features.csv")
    print("="*80)
    df1 = pd.read_csv(data_dir / 'kc_house_data_xgboost_features.csv')
    results1 = train_and_evaluate(df1, 'xgboost_features')
    
    # dataset 2
    print("\n\n" + "="*80)
    print("DATASET 2: kc_house_data_encoded_tree_step3.csv")
    print("="*80)
    df2 = pd.read_csv(data_dir / 'kc_house_data_encoded_tree_step3.csv')
    results2 = train_and_evaluate(df2, 'encoded_tree_step3')
    
    # summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    comparison_df = pd.DataFrame([
        {
            'Dataset': results1['dataset'],
            'Features': results1['n_features'],
            'Train R²': f"{results1['train_r2']:.6f}",
            'Test R²': f"{results1['test_r2']:.6f}",
            'Test RMSE (log)': f"{results1['test_rmse_log']:.6f}",
            'Test RMSE ($)': f"${results1['test_rmse_dollars']:,.2f}"
        },
        {
            'Dataset': results2['dataset'],
            'Features': results2['n_features'],
            'Train R²': f"{results2['train_r2']:.6f}",
            'Test R²': f"{results2['test_r2']:.6f}",
            'Test RMSE (log)': f"{results2['test_rmse_log']:.6f}",
            'Test RMSE ($)': f"${results2['test_rmse_dollars']:,.2f}"
        }
    ])
    
    print("\n" + comparison_df.to_string(index=False))
    
    # feature diffs
    print("\n\n" + "="*80)
    print("FEATURE DIFFERENCES")
    print("="*80)
    
    features1 = set(results1['feature_names'])
    features2 = set(results2['feature_names'])
    
    only_in_1 = features1 - features2
    only_in_2 = features2 - features1
    common = features1 & features2
    
    print(f"\nCommon ({len(common)}): {sorted(common)}")
    print(f"\nOnly in xgboost_features ({len(only_in_1)}): {sorted(only_in_1)}")
    print(f"\nOnly in encoded_tree_step3 ({len(only_in_2)}): {sorted(only_in_2)}")
    
    # performance diff
    print("\n\n" + "="*80)
    print("PERFORMANCE DIFF")
    print("="*80)
    
    r2_diff = results2['test_r2'] - results1['test_r2']
    rmse_diff = results2['test_rmse_dollars'] - results1['test_rmse_dollars']
    
    print(f"\nTest R² diff: {r2_diff:+.6f}")
    print(f"  (encoded_tree_step3 - xgboost_features)")
    
    print(f"\nTest RMSE diff: ${rmse_diff:+,.2f}")
    print(f"  (encoded_tree_step3 - xgboost_features)")
    
    if abs(r2_diff) < 0.001:
        print("\nResults are very similar (R² diff < 0.001)")
    else:
        better = 'encoded_tree_step3' if r2_diff > 0 else 'xgboost_features'
        print(f"\n{better} performs better")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
