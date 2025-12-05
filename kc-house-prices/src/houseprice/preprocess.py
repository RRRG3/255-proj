from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def split_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split into numeric and categorical columns"""
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical

def make_preprocessors(numeric, categorical):
    """Create preprocessing pipelines for linear and tree models"""
    
    # for linear regression - need to scale
    prep_lr = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
    ], remainder="drop")

    # for trees - no scaling needed
    prep_trees = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
    ], remainder="drop")

    return prep_lr, prep_trees
