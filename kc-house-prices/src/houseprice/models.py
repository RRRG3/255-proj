from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def make_linear(prep):
    return Pipeline([("prep", prep), ("model", LinearRegression())]), {}

def make_random_forest(prep, random_state=42):
    pipe = Pipeline([("prep", prep),
                     ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1))])
    grid = {
        "model__n_estimators": [100],
        "model__max_depth": [10, 20],
        "model__min_samples_leaf": [1, 2],
    }
    return pipe, grid

def make_xgb(prep, random_state=42):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None, None  # xgboost not installed
        
    pipe = Pipeline([("prep", prep),
                     ("model", XGBRegressor(
                         objective="reg:squarederror", n_jobs=-1,
                         random_state=random_state
                     ))])
    
    # Reduced grid search for XGBoost (faster demonstration)
    grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.05, 0.1],
    }
    return pipe, grid


def make_xgb_optimized(prep, random_state=42, search_mode="full"):
    """
    Create XGBoost pipeline with expanded hyperparameter grid."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None, None  # xgboost not installed
    
    pipe = Pipeline([("prep", prep),
                     ("model", XGBRegressor(
                         objective="reg:squarederror", 
                         n_jobs=-1,
                         random_state=random_state
                     ))])
    
    if search_mode == "fast":
        # Reduced grid for faster iteration
        grid = {
            "model__n_estimators": [300],
            "model__max_depth": [6],
            "model__learning_rate": [0.05],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
            "model__min_child_weight": [3],
            "model__gamma": [0.1],
        }
    else:
        # Full expanded grid for comprehensive search
        grid = {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 6, 9],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__gamma": [0, 0.1, 0.2],
        }
    
    return pipe, grid
