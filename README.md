# King County House Price Prediction

Predicting house prices in King County, WA using machine learning. This project compares three different approaches—Linear Regression, Random Forest, and XGBoost—to see which one works best.

## Overview

We built this to explore how different ML models handle real estate pricing. The dataset has over 21,000 house sales from 2014-2015, and we wanted to see if tree-based models could beat a simple linear approach.

### What We Found

XGBoost came out on top with an R² of 0.905 and RMSE of $120,312. That's a 16% improvement over linear regression and about 2% better than random forest. On the test set, it held up well (R² = 0.900, RMSE = $116,874).

The most important features? Square footage, grade, and location—no surprises there. But the interaction features we engineered (like sqft × grade) also helped quite a bit.

Linear regression struggled with expensive homes, consistently under-predicting them. The residual plots made this pretty obvious. Tree-based models handled the non-linear relationships much better.

## Project Structure

```
kc-house-prices/
├── data/                          # Dataset files
│   ├── kc_house_data.csv          # Raw dataset
│   ├── kc_house_data_engineered.csv
│   ├── kc_house_data_engineered_step2.csv
│   ├── kc_house_data_encoded_linear_step3.csv
│   ├── kc_house_data_encoded_tree_step3.csv
│   └── kc_house_data_xgboost_features.csv
├── notebooks/                     # Jupyter notebooks
│   ├── analysis.ipynb             # Main analysis notebook
│   ├── 01_preprocessing.ipynb     # Data preprocessing
│   ├── lin_reg.ipynb              # Linear regression analysis
│   ├── random-forest.ipynb        # Random forest analysis
│   ├── xgboost.ipynb              # XGBoost analysis
│   ├── xgboost_improvement_comparison.ipynb
│   └── final_report.ipynb         # Final report
├── reports/                       # Generated outputs
│   ├── eda_*.png                  # EDA visualizations
│   ├── lr_*.png                   # Linear regression plots
│   ├── *_feature_importance.png   # Feature importance charts
│   ├── xgboost_shap_summary.png   # SHAP analysis
│   ├── model_cv_results.csv       # Cross-validation results
│   ├── comparison_table.txt       # Model comparison
│   └── test_metrics.txt           # Test set performance
├── scripts/                        # Automation scripts
│   ├── prepare_data.py            # Data preparation
│   ├── run_eda.py                 # Generate EDA plots
│   ├── run_cv.py                  # Run cross-validation
│   ├── optimize_xgboost.py        # Hyperparameter tuning
│   ├── test_significance.py       # Statistical tests
│   ├── compare_datasets.py        # Dataset comparison
│   └── generate_comparison_chart.py
├── src/houseprice/                 # Core library
│   ├── config.py                  # Configuration
│   ├── data.py                    # Data loading
│   ├── features.py                # Feature engineering
│   ├── feature_selection.py       # Feature selection
│   ├── preprocess.py              # Preprocessing pipelines
│   ├── transforms.py              # Data transformations
│   ├── models.py                  # Model definitions
│   ├── eda.py                     # EDA functions
│   ├── plots.py                   # Plotting utilities
│   └── outliers.py                # Outlier detection
├── tests/                         # Unit tests
│   ├── test_eda.py
│   ├── test_plots_enhanced.py
│   └── test_cv_stats.py
├── requirements.txt                # Dependencies
├── setup.py                        # Package setup
└── README.md
```

## Getting Started

### Setup

You'll need Python 3.12 (some dependencies like SHAP don't play nice with 3.14 yet).

```bash
git clone <your-repo-url>
cd kc-house-prices

python3.12 -m venv .venv312
source .venv312/bin/activate  # Windows: .venv312\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Get the Data

Grab the dataset from [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) and either run:

```bash
python scripts/prepare_data.py --source /path/to/kc_house_dataset.csv
```

Or just drop `kc_house_data.csv` into the `data/` folder.

### Run It

**Jupyter notebook** (recommended):
```bash
jupyter notebook notebooks/analysis.ipynb
```

**Command line scripts**:
```bash
python scripts/run_eda.py      # generates visualizations
python scripts/run_cv.py        # trains models and runs CV
pytest tests/ -v                # runs tests
```

## Results

### Model Performance

| Model              | Mean R²  | Std Dev | RMSE         |
|--------------------|----------|---------|--------------|
| **XGBoost**        | **0.905**| **0.003**| **$120,312** |
| Random Forest      | 0.886    | 0.004   | $133,642     |
| Linear Regression  | 0.780    | 0.007   | $177,014     |

### What This Means

XGBoost wins by a decent margin. It's 16% better than linear regression in terms of R², and cuts the error down by about $57k. Even compared to random forest, it's noticeably better.

The low standard deviation (0.003) means it's consistent across different data splits, which is what you want.

**Feature importance breakdown:**
- Living area (sqft_living) matters most—bigger houses cost more, obviously
- Grade and condition are huge factors (quality over quantity)
- Location (lat/long/zipcode) is critical—real estate 101
- House age and whether it's been renovated add predictive power
- Interaction features like sqft × grade helped capture relationships that individual features miss

Linear regression had a hard time with expensive homes. The residuals showed clear patterns—it kept under-predicting high-value properties. Tree models don't have this problem since they can handle non-linear relationships.

Test set performance stayed strong (R² = 0.900), so the model generalizes well to new data.

## What Gets Generated

Running the analysis creates a bunch of visualizations and reports in the `reports/` folder:

**EDA plots:**
- Price distributions (raw and log-transformed)
- Correlation heatmap
- Sqft vs price scatter
- Geographic price map

**Model diagnostics:**
- Residual plots for linear regression
- Feature importance charts for all models
- SHAP summary plot (shows how features impact predictions)

**Results:**
- `model_cv_results.csv` - full CV metrics
- `comparison_table.txt` - formatted comparison
- `test_metrics.txt` - final test performance

## How It Works

### Feature Engineering

We created some new features that helped a lot:
- `house_age` - calculated from sale year and build year
- `was_renovated` - binary flag
- `sale_month` - extracted from date
- Interaction features: `sqft_x_grade`, `age_x_grade`, `living_lot_ratio`

These interactions capture relationships like "a large house with high grade is worth more than the sum of its parts."

### Preprocessing

- Ordinal encoding for grade, condition, view
- One-hot encoding for zipcode
- StandardScaler for linear regression (tree models don't need it)
- Log-transformed the target (prices are right-skewed)

### Training

We used 5-fold cross-validation with GridSearchCV to tune hyperparameters. Evaluated on R² and RMSE (converted back to original price scale). Held out 20% for final testing.

**Models:**
- Linear Regression - baseline
- Random Forest - 100 trees, max_depth=20
- XGBoost - learning_rate=0.05, max_depth=6, n_estimators=300

## Tests

```bash
pytest tests/ -v
```

19 tests covering EDA functions, plotting utilities, and CV stats. 

## Dependencies

Python 3.12 recommended. Main packages: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap, jupyter, pytest.

See `requirements.txt` for full list.

## Dataset

[King County House Sales on Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

21,613 sales from May 2014 - May 2015. Includes price, physical attributes (bedrooms, bathrooms, sqft), quality indicators (grade, condition, view), location (lat/long/zipcode), and temporal data (build year, renovation year).

## Contributors

This project was developed as a collaborative effort to explore machine learning techniques for real estate price prediction.

