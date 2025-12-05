"""Outlier detection and removal"""
import pandas as pd
import numpy as np
from typing import List


def detect_outliers(df: pd.DataFrame, 
                   columns: List[str], 
                   n_std: float = 3.0) -> pd.Series:
    """Find outliers using std deviation method"""
    if n_std <= 0:
        raise ValueError(f"n_std must be positive, got {n_std}")
    
    outlier_mask = pd.Series(False, index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        values = df[col]
        mean = values.mean()
        std = values.std()
        
        # flag anything beyond n_std standard deviations
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        col_outliers = (values < lower_bound) | (values > upper_bound)
        outlier_mask = outlier_mask | col_outliers
    
    return outlier_mask


def remove_outliers(df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
    """Drop outlier rows"""
    n_outliers = outlier_mask.sum()
    
    if n_outliers == len(df):
        raise ValueError(
            "All rows are outliers! Check your threshold."
        )
    
    return df[~outlier_mask].copy()


def cap_outliers(df: pd.DataFrame, 
                columns: List[str], 
                n_std: float = 3.0) -> pd.DataFrame:
    """Cap outliers instead of removing them"""
    if n_std <= 0:
        raise ValueError(f"n_std must be positive, got {n_std}")
    
    out = df.copy()
    
    for col in columns:
        if col not in out.columns:
            continue
            
        values = out[col]
        mean = values.mean()
        std = values.std()
        
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        # clip to bounds
        out[col] = values.clip(lower=lower_bound, upper=upper_bound)
    
    return out
