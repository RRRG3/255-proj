import pandas as pd

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
