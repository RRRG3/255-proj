from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def make_linear(prep):
    """Simple linear regression pipeline"""
    return Pipeline([("prep", prep), ("model", LinearRegression())]), {}

def make_random_forest(prep, random_state=42):
    """Random forest with basic grid search"""
    pipe = Pipeline([("prep", prep),
                     ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1))])
    grid = {
        "model__n_estimators": [100],
        "model__max_depth": [10, 20],
        "model__min_samples_leaf": [1, 2],
    }
    return pipe, grid

def make_xgb(prep, random_state=42):
    """XGBoost with small grid (faster)"""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None, None
        
    pipe = Pipeline([("prep", prep),
                     ("model", XGBRegressor(
                         objective="reg:squarederror", n_jobs=-1,
                         random_state=random_state
                     ))])
    
    grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.05, 0.1],
    }
    return pipe, grid


def make_xgb_optimized(prep, random_state=42, search_mode="full"):
    """XGBoost with bigger hyperparameter grid for optimization experiments"""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None, None
    
    pipe = Pipeline([("prep", prep),
                     ("model", XGBRegressor(
                         objective="reg:squarederror", 
                         n_jobs=-1,
                         random_state=random_state
                     ))])
    
    if search_mode == "fast":
        # quick grid for testing
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
        # full grid - takes a while
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
