# King County House Price Prediction

A comprehensive data mining project comparing Linear Regression, Random Forest, and XGBoost models for predicting house prices in King County, Washington.

##  Project Overview

This project analyzes the King County House Sales dataset to predict house prices using machine learning techniques. We compare three different modeling approaches and provide in-depth analysis of their performance, feature importance, and predictive capabilities.

### Key Findings

- **Best Model**: XGBoost (R² = 0.895, RMSE = $129,500)
- **Random Forest**: R² = 0.885, RMSE = $134,278
- **Linear Regression**: R² = 0.772, RMSE = $187,588 (shows systematic residual patterns)
- **Key Features**: Living area (sqft_living), grade, location (lat/long), and house age, plus engineered interaction features.
- **Model Insights**: SHAP analysis reveals complex feature interactions driving predictions.

##  Project Structure

```
kc-house-prices/
├── data/                      # Raw dataset (ignored by git)
│   └── kc_house_data.csv
├── notebooks/                 # Jupyter notebooks for analysis
│   └── analysis.ipynb         # Complete end-to-end analysis
├── reports/                   # Generated visualizations and metrics
│   ├── eda_*.png             # Exploratory data analysis plots
│   ├── lr_*.png              # Linear regression diagnostics
│   ├── tree_*.png            # Tree model feature importance
│   ├── model_cv_results.csv  # Cross-validation results
│   └── comparison_table.txt  # Formatted model comparison
├── scripts/                   # Automation scripts
│   ├── prepare_data.py       # Dataset preparation utility
│   ├── run_eda.py           # Generate EDA visualizations
│   └── run_cv.py            # Run cross-validation pipeline
├── src/houseprice/           # Reusable library code
│   ├── config.py            # Configuration constants
│   ├── data.py              # Data loading utilities
│   ├── features.py          # Feature engineering
│   ├── preprocess.py        # Preprocessing pipelines
│   ├── models.py            # Model definitions
│   ├── eda.py               # EDA visualization functions
│   └── plots.py             # Diagnostic plotting functions
├── tests/                    # Unit tests
│   ├── test_eda.py
│   ├── test_plots_enhanced.py
│   └── test_cv_stats.py
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation
├── Dockerfile               # Docker container definition
└── README.md                # This file
```

##  Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd kc-house-prices

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare the Dataset

Download the King County House Sales dataset from [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) and place it in the project:

```bash
python scripts/prepare_data.py --source /path/to/kc_house_dataset.csv
```

Or manually copy the CSV to `data/kc_house_data.csv`.

### 3. Run the Analysis

#### Option A: Interactive Notebook (Recommended)

```bash
jupyter notebook notebooks/analysis.ipynb
```

This notebook provides a complete walkthrough with visualizations and interpretations.

#### Option B: Command Line Scripts

```bash
# Generate EDA visualizations
python scripts/run_eda.py

# Run model training and cross-validation
python scripts/run_cv.py

# Run tests
pytest tests/ -v

# Run statistical significance test
python scripts/test_significance.py
```

#### Option C: Docker (Easiest)

This method runs the entire analysis in a containerized environment, ensuring reproducibility.

```bash
# 1. Build the Docker image
docker build -t kc-house-prices .

# 2. Run the analysis
# This will mount the local 'reports' directory to the container's 'reports'
# so you can see the generated files on your machine.
docker run --rm -v "$(pwd)/reports:/app/reports" kc-house-prices
```

##  Results

### Model Comparison

| Model              | Mean R²  | R² SD   | RMSE         |
|--------------------|----------|---------|--------------|
| **XGBoost**        | **0.895**| **0.003**| **$129,500** |
| Random Forest      | 0.885    | 0.002   | $134,278     |
| Linear Regression  | 0.772    | 0.004   | $187,588     |

### Key Insights

1. **Random Forest Superiority**: Achieves 11% higher R² than Linear Regression, demonstrating the importance of capturing non-linear relationships in real estate pricing.

2. **Model Stability**: Random Forest shows lower R² standard deviation (0.002 vs 0.004), indicating more consistent performance across different data splits.

3. **Feature Importance**:
   - **Quantity**: Living area (sqft_living) is the strongest predictor
   - **Quality**: Grade and condition significantly impact price
   - **Location**: Geographic features (lat, long, zipcode) are critical
   - **Age**: House age and renovation status provide additional predictive power

4. **Linear Regression Limitations**: Residual analysis reveals systematic under-prediction of expensive homes, indicating the model cannot capture complex pricing patterns.

## Generated Outputs

### Exploratory Data Analysis (EDA)
- `eda_price_dist.png` - Distribution of raw house prices
- `eda_price_log_dist.png` - Log-transformed price distribution
- `eda_correlation.png` - Feature correlation heatmap
- `eda_sqft_vs_price.png` - Living area vs price scatter plot
- `eda_geographic.png` - Geographic price distribution

### Model Diagnostics
- `lr_residuals.png` - Basic residual plot for Linear Regression
- `lr_residuals_enhanced.png` - Enhanced residual analysis with error distribution
- `lr_coefficients.png` - Top Linear Regression coefficients
- `randomforest_feature_importance.png` - Feature importance for Random Forest
- `xgboost_feature_importance.png` - Feature importance for XGBoost
- `xgboost_shap_summary.png` - SHAP summary plot for XGBoost, showing feature impact on predictions

### Results Tables
- `model_cv_results.csv` - Detailed cross-validation metrics
- `comparison_table.txt` - Formatted model comparison table
- `test_metrics.txt` - Final test set performance

##  Methodology

### Data Preprocessing

1. **Feature Engineering**:
   - `house_age` = sale_year - yr_built
   - `was_renovated` = binary flag (0/1)
   - `sale_month` = extracted from date
   - **Interaction Features**:
     - `sqft_x_grade` = sqft_living * grade
     - `age_x_grade` = house_age * grade
     - `living_lot_ratio` = sqft_living / sqft_lot

2. **Encoding**:
   - Ordinal encoding: grade, condition, view
   - One-hot encoding: zipcode

3. **Scaling**:
   - StandardScaler for Linear Regression
   - No scaling for tree-based models

4. **Target Transformation**:
   - log(price + 1) to handle right-skewed distribution

### Model Training

- **Cross-Validation**: 5-fold stratified CV
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Evaluation Metrics**: R², RMSE on original price scale
- **Test Split**: 80/20 train-test split

### Models Evaluated

1. **Linear Regression**: Baseline model with StandardScaler preprocessing
2. **Random Forest**: Ensemble of 500 decision trees with max_depth=20
3. **XGBoost**: (Optional) Gradient boosting with optimized hyperparameters

##  Testing

Run the test suite to verify all functionality:

```bash
pytest tests/ -v
```

**Test Coverage**:
- EDA visualization functions (10 tests)
- Enhanced plotting functions (4 tests)
- CV statistics and formatting (5 tests)

All 19 tests passing 

##  Dependencies

- Python 3.9+
- pandas >= 2.2.2
- numpy >= 1.26.4
- scikit-learn >= 1.4.2
- matplotlib >= 3.8.4
- seaborn >= 0.13.0
- xgboost >= 2.0.3
- jupyter >= 1.0.0
- pytest >= 7.4.0
- shap

## Future Enhancements

1.  **Model Improvements**:
    - [x] ~~Implement XGBoost for comparison~~
    - [ ] Explore ensemble methods (stacking, blending)
    - [ ] Add neural network models (e.g., TabNet)

2.  **Feature Engineering**:
    - [x] ~~Create interaction terms (sqft × grade)~~
    - [ ] Add neighborhood clustering features (e.g., K-Means on lat/long)
    - [ ] Include temporal trends (e.g., seasonal effects on price)

3.  **Visualization & Explainability**:
    - [x] ~~SHAP values for model explainability~~
    - [ ] Interactive plots using Plotly or Bokeh
    - [ ] Automated PDF report generation

4.  **Deployment**:
    - [x] ~~Containerize with Docker~~
    - [ ] Create a REST API for predictions using FastAPI
    - [ ] Build a simple web interface with Streamlit or Gradio

##  Dataset Information

**Source**: [Kaggle - House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

**Features**: 21 columns including:
- Price (target variable)
- Physical attributes (bedrooms, bathrooms, sqft_living, sqft_lot)
- Quality indicators (grade, condition, view)
- Location (lat, long, zipcode)
- Temporal (date, yr_built, yr_renovated)

**Size**: 21,613 house sales records

**Time Period**: May 2014 - May 2015

