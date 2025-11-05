# KC House Prices

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Preparing the Dataset

Copy the provided CSV into the repository's `data/` directory so the training
pipeline can find it. You can either move the file manually or run the helper
script:

```bash
python scripts/prepare_data.py --source /path/to/kc_house_dataset.csv
```

The script copies the dataset to `data/kc_house_data.csv` by default. Pass
`--destination` or `--force` if you need to customize the target location or
overwrite an existing file.

## Running the Pipeline

```bash
source .venv/bin/activate
python scripts/run_cv.py --data data/kc_house_data.csv --out reports/
```

## Project Structure

```
kc-house-prices/
├─ data/                      # raw dataset (ignored by git)
├─ notebooks/                 # exploratory notebooks
├─ reports/                   # generated tables, plots, and metrics
├─ scripts/                   # automation entry points
└─ src/houseprice/            # reusable library code
```

## Outputs

- `reports/model_cv_results.csv`: cross-validation metrics (R² and RMSE)
- `reports/test_metrics.txt`: best model summary with test scores
- `reports/lr_residuals.png`: residual plot for linear regression
- `reports/tree_feature_importance.png`: feature importance plot for the top tree model
