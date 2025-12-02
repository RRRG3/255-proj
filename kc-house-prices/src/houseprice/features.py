import pandas as pd
import numpy as np
from typing import Optional, Dict


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["sale_year"] = out["date"].dt.year
        out["sale_month"] = out["date"].dt.month
        
    # house_age (fallback 2015 if year missing)
    if "yr_built" in out.columns:
        ref = out["sale_year"].fillna(2015) if "sale_year" in out.columns else 2015
        out["house_age"] = ref - out["yr_built"]
        
    # renovation flag
    if "yr_renovated" in out.columns:
        out["was_renovated"] = (out["yr_renovated"].fillna(0) > 0).astype(int)
    
    # Interaction features
    if "sqft_living" in out.columns and "grade" in out.columns:
        out["sqft_x_grade"] = out["sqft_living"] * out["grade"]
    
    if "house_age" in out.columns and "grade" in out.columns:
        out["age_x_grade"] = out["house_age"] * out["grade"]
        
    if "sqft_living" in out.columns and "sqft_lot" in out.columns:
        out["living_lot_ratio"] = out["sqft_living"] / out["sqft_lot"].replace(0, 1e-6)

    # Drop date as it's no longer needed
    if "date" in out.columns:
        out = out.drop(columns=["date"])

    return out


def engineer_advanced_features(df: pd.DataFrame, 
                               include_location_stats: bool = True,
                               zipcode_stats: Optional[Dict] = None) -> pd.DataFrame:
    """
    Engineer advanced features for house price prediction.
    
    This function adds sophisticated features including density metrics,
    quality scores, temporal patterns, and location-based aggregations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw features (must include price column if computing stats)
    include_location_stats : bool, default=True
        Whether to compute zipcode aggregation features
    zipcode_stats : Optional[Dict], default=None
        Pre-computed zipcode statistics (for test set). If None, computes from df.
        Expected keys: 'median', 'std'
        
    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered features
    """
    # First apply basic feature engineering
    out = engineer_features(df)
    
    # Total square footage
    if "sqft_above" in out.columns and "sqft_basement" in out.columns:
        out["total_sqft"] = out["sqft_above"] + out["sqft_basement"]
    elif "sqft_living" in out.columns:
        out["total_sqft"] = out["sqft_living"]
    
    # Density features (rooms per square foot)
    if "bathrooms" in out.columns and "sqft_living" in out.columns:
        out["bathroom_density"] = out["bathrooms"] / out["sqft_living"].replace(0, 1e-6)
    
    if "bedrooms" in out.columns and "sqft_living" in out.columns:
        out["bedroom_density"] = out["bedrooms"] / out["sqft_living"].replace(0, 1e-6)
    
    # Bedroom to bathroom ratio
    if "bedrooms" in out.columns and "bathrooms" in out.columns:
        out["bedroom_bathroom_ratio"] = out["bedrooms"] / out["bathrooms"].replace(0, 1e-6)
    
    # Quality score (grade * condition)
    if "grade" in out.columns and "condition" in out.columns:
        out["quality_score"] = out["grade"] * out["condition"]
    
    # Renovation age (years since renovation)
    if "yr_renovated" in out.columns and "sale_year" in out.columns:
        # Only compute for renovated houses
        renovated_mask = out["yr_renovated"].fillna(0) > 0
        out["renovation_age"] = 0
        out.loc[renovated_mask, "renovation_age"] = (
            out.loc[renovated_mask, "sale_year"] - out.loc[renovated_mask, "yr_renovated"]
        )
    
    # Temporal features
    if "sale_month" in out.columns:
        # Quarter (1-4)
        out["sale_quarter"] = ((out["sale_month"] - 1) // 3 + 1).astype(int)
        
        # Seasonal indicators
        out["is_summer_sale"] = out["sale_month"].isin([6, 7, 8]).astype(int)
        out["is_winter_sale"] = out["sale_month"].isin([12, 1, 2]).astype(int)
    
    # Location-based aggregation features
    if include_location_stats and "zipcode" in out.columns:
        if zipcode_stats is None:
            # Compute from training data
            if "price" in out.columns:
                zipcode_stats = {
                    "median": out.groupby("zipcode")["price"].median().to_dict(),
                    "std": out.groupby("zipcode")["price"].std().fillna(0).to_dict()
                }
            else:
                # Can't compute without price column
                zipcode_stats = None
        
        if zipcode_stats is not None:
            # Global fallback values
            global_median = np.median(list(zipcode_stats["median"].values()))
            global_std = np.median(list(zipcode_stats["std"].values()))
            
            # Map zipcode to statistics
            out["zipcode_median_price"] = out["zipcode"].map(
                zipcode_stats["median"]
            ).fillna(global_median)
            
            out["zipcode_price_std"] = out["zipcode"].map(
                zipcode_stats["std"]
            ).fillna(global_std)
    
    return out
