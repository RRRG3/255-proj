from pathlib import Path
import pandas as pd

def load_data(path: Path) -> pd.DataFrame:
    """Load dataset from CSV"""
    df = pd.read_csv(path)
    if "price" not in df.columns:
        raise ValueError("Expected 'price' column in dataset")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning - remove junk rows"""
    out = df.copy()
    
    # drop id column if it exists
    if "id" in out.columns:
        out = out.drop(columns=["id"])
        
    # filter out bad data
    if "price" in out.columns:
        out = out[out["price"] > 0]
    
    if "sqft_living" in out.columns:
        out = out[out["sqft_living"] > 0]
        
    # remove 0 bedroom entries (probably data errors or land sales)
    if "bedrooms" in out.columns:
        out = out[out["bedrooms"] > 0]
        
    return out
