import pandas as pd
import numpy as np
from typing import Optional, Dict


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic features from raw data"""
    out = df.copy()
    
    # parse dates and extract temporal features
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["sale_year"] = out["date"].dt.year
        out["sale_month"] = out["date"].dt.month
        
    # calculate house age
    if "yr_built" in out.columns:
        ref = out["sale_year"].fillna(2015) if "sale_year" in out.columns else 2015
        out["house_age"] = ref - out["yr_built"]
        
    # was it renovated?
    if "yr_renovated" in out.columns:
        out["was_renovated"] = (out["yr_renovated"].fillna(0) > 0).astype(int)
    
    # some interaction terms that might be useful
    if "sqft_living" in out.columns and "grade" in out.columns:
        out["sqft_x_grade"] = out["sqft_living"] * out["grade"]
    
    if "house_age" in out.columns and "grade" in out.columns:
        out["age_x_grade"] = out["house_age"] * out["grade"]
        
    if "sqft_living" in out.columns and "sqft_lot" in out.columns:
        out["living_lot_ratio"] = out["sqft_living"] / out["sqft_lot"].replace(0, 1e-6)

    # don't need date anymore
    if "date" in out.columns:
        out = out.drop(columns=["date"])

    return out


def engineer_advanced_features(df: pd.DataFrame, 
                               include_location_stats: bool = True,
                               zipcode_stats: Optional[Dict] = None) -> pd.DataFrame:
    """
    More advanced feature engineering - adds density metrics, quality scores, etc.
    
    Args:
        df: input dataframe (needs price column if computing stats)
        include_location_stats: whether to add zipcode aggregations
        zipcode_stats: pre-computed zipcode stats for test set (dict with 'median', 'std' keys)
    
    Returns:
        dataframe with extra features
    """
    # start with basic features
    out = engineer_features(df)
    
    # total sqft
    if "sqft_above" in out.columns and "sqft_basement" in out.columns:
        out["total_sqft"] = out["sqft_above"] + out["sqft_basement"]
    elif "sqft_living" in out.columns:
        out["total_sqft"] = out["sqft_living"]
    
    # density features - how cramped is it?
    if "bathrooms" in out.columns and "sqft_living" in out.columns:
        out["bathroom_density"] = out["bathrooms"] / out["sqft_living"].replace(0, 1e-6)
    
    if "bedrooms" in out.columns and "sqft_living" in out.columns:
        out["bedroom_density"] = out["bedrooms"] / out["sqft_living"].replace(0, 1e-6)
    
    # bedroom/bathroom ratio
    if "bedrooms" in out.columns and "bathrooms" in out.columns:
        out["bedroom_bathroom_ratio"] = out["bedrooms"] / out["bathrooms"].replace(0, 1e-6)
    
    # overall quality metric
    if "grade" in out.columns and "condition" in out.columns:
        out["quality_score"] = out["grade"] * out["condition"]
    
    # how long since renovation?
    if "yr_renovated" in out.columns and "sale_year" in out.columns:
        renovated_mask = out["yr_renovated"].fillna(0) > 0
        out["renovation_age"] = 0
        out.loc[renovated_mask, "renovation_age"] = (
            out.loc[renovated_mask, "sale_year"] - out.loc[renovated_mask, "yr_renovated"]
        )
    
    # seasonal features
    if "sale_month" in out.columns:
        out["sale_quarter"] = ((out["sale_month"] - 1) // 3 + 1).astype(int)
        out["is_summer_sale"] = out["sale_month"].isin([6, 7, 8]).astype(int)
        out["is_winter_sale"] = out["sale_month"].isin([12, 1, 2]).astype(int)
    
    # zipcode-based features (location matters!)
    if include_location_stats and "zipcode" in out.columns:
        if zipcode_stats is None:
            # compute from training data
            if "price" in out.columns:
                zipcode_stats = {
                    "median": out.groupby("zipcode")["price"].median().to_dict(),
                    "std": out.groupby("zipcode")["price"].std().fillna(0).to_dict()
                }
            else:
                zipcode_stats = None
        
        if zipcode_stats is not None:
            # use global fallbacks for unseen zipcodes
            global_median = np.median(list(zipcode_stats["median"].values()))
            global_std = np.median(list(zipcode_stats["std"].values()))
            
            out["zipcode_median_price"] = out["zipcode"].map(
                zipcode_stats["median"]
            ).fillna(global_median)
            
            out["zipcode_price_std"] = out["zipcode"].map(
                zipcode_stats["std"]
            ).fillna(global_std)
    
    return out
