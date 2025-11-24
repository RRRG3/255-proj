from pathlib import Path

DATA_PATH = Path("data/kc_house_data.csv")
OUT_DIR = Path("reports")
RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.2

# EDA plot output filenames
EDA_PLOTS = {
    "price_dist": "eda_price_dist.png",
    "price_log_dist": "eda_price_log_dist.png",
    "correlation": "eda_correlation.png",
    "sqft_vs_price": "eda_sqft_vs_price.png",
    "geographic": "eda_geographic.png",
}
