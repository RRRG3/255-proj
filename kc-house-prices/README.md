# KC House Prices

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

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
