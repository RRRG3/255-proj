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
