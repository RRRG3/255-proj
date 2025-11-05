from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def make_linear(prep):
    return Pipeline([("prep", prep), ("model", LinearRegression())]), {}

def make_random_forest(prep, random_state=42):
    pipe = Pipeline([("prep", prep),
                     ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1))])
    grid = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 3, 5],
    }
    return pipe, grid

def make_xgb(prep, random_state=42):
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None, None  # xgboost not installed
    pipe = Pipeline([("prep", prep),
                     ("model", XGBRegressor(
                         n_estimators=600, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         objective="reg:squarederror", n_jobs=-1,
                         random_state=random_state
                     ))])
    grid = {"model__max_depth": [4, 6, 8]}
    return pipe, grid
