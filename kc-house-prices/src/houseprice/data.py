from pathlib import Path
import pandas as pd

def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset from the specified path."""
    df = pd.read_csv(path)
    if "price" not in df.columns:
        raise ValueError("Expected 'price' column in dataset")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform initial data cleaning."""
    out = df.copy()
    
    # Drop id if exists
    if "id" in out.columns:
        out = out.drop(columns=["id"])
        
    # Sanity filters
    # Remove obviously broken rows
    if "price" in out.columns:
        out = out[out["price"] > 0]
    
    if "sqft_living" in out.columns:
        out = out[out["sqft_living"] > 0]
        
    # Optional: remove 0 bedrooms (unless studios are intended, but usually 0 is error or land)
    # The user instruction: "bedrooms == 0 (unless you want to keep studios explicitly)" -> assume remove for standard house model
    if "bedrooms" in out.columns:
        out = out[out["bedrooms"] > 0]
        
    return out
