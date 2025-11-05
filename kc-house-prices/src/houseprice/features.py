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
    return out
