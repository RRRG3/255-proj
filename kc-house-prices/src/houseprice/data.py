from pathlib import Path
import pandas as pd

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "price" not in df.columns:
        raise ValueError("Expected 'price' column in dataset")
    return df
