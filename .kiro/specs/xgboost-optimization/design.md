# Design Document: XGBoost Model Optimization

## Overview

This design outlines a systematic approach to improving the XGBoost regression model for King County house price prediction. The current baseline achieves R² = 0.9050 and RMSE = $120,311.95. We will implement multiple optimization strategies including expanded hyperparameter tuning, advanced feature engineering, outlier handling, alternative target transformations, and feature selection.

The design follows an experimental approach where each optimization strategy can be tested independently and in combination, allowing us to measure the incremental impact of each improvement.

## Architecture

The optimization system extends the existing pipeline architecture with the following components:

```
Data Loading → Outlier Handling → Feature Engineering → Target Transformation
                                                              ↓
                                                    Train/Test Split
                                                              ↓
                                            Preprocessing (Imputation/Encoding)
                                                              ↓
                                        XGBoost Training (Expanded Grid Search)
                                                              ↓
                                            Feature Selection (Optional)
                                                              ↓
                                        Model Retraining (If features removed)
                                                              ↓
                                            Evaluation & Comparison
```

### Key Design Principles

1. **Backward Compatibility**: All enhancements are additive and don't break existing functionality
2. **Modularity**: Each optimization can be enabled/disabled independently
3. **Reproducibility**: All random operations use configurable seeds
4. **Measurability**: Each optimization tracks its impact on performance metrics

## Components and Interfaces

### 1. Enhanced Feature Engineering Module

**Location**: `src/houseprice/features.py`

**New Function**: `engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame`

This function extends the existing `engineer_features` with additional transformations:

- **Price-per-sqft proxies**: Create ratios that capture value density
- **Total area features**: Combine related square footage measurements
- **Density features**: Rooms per square foot metrics
- **Quality scores**: Composite metrics from grade and condition
- **Temporal features**: Enhanced date-based features (quarter, season)
- **Location aggregates**: Zipcode-level statistics

**Interface**:
```python
def engineer_advanced_features(df: pd.DataFrame, 
                               include_location_stats: bool = True) -> pd.DataFrame:
    """
    Engineer advanced features for house price prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw features
    include_location_stats : bool
        Whether to compute zipcode aggregation features
        
    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered features
    """
```

### 2. Outlier Handling Module

**Location**: `src/houseprice/outliers.py` (new file)

**Functions**:
- `detect_outliers(df: pd.DataFrame, columns: List[str], n_std: float = 3.0) -> pd.Series`
- `remove_outliers(df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame`
- `cap_outliers(df: pd.DataFrame, columns: List[str], n_std: float = 3.0) -> pd.DataFrame`

**Interface**:
```python
def detect_outliers(df: pd.DataFrame, 
                   columns: List[str], 
                   n_std: float = 3.0) -> pd.Series:
    """
    Detect outliers using standard deviation method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to check for outliers
    n_std : float
        Number of standard deviations for threshold
        
    Returns
    -------
    pd.Series
        Boolean mask where True indicates outlier
    """
```

### 3. Target Transformation Module

**Location**: `src/houseprice/transforms.py` (new file)

**Class**: `TargetTransformer`

This class encapsulates different transformation strategies with a consistent interface:

```python
class TargetTransformer:
    """Handle target variable transformations."""
    
    def __init__(self, method: str = "log1p"):
        """
        Initialize transformer.
        
        Parameters
        ----------
        method : str
            One of: "log1p", "boxcox", "yeojohnson", "none"
        """
        
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit transformer and transform target."""
        
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Convert predictions back to original scale."""
```

### 4. Enhanced XGBoost Model Module

**Location**: `src/houseprice/models.py`

**Updated Function**: `make_xgb_optimized(prep, random_state=42)`

Expands the hyperparameter grid with regularization and tree structure parameters:

```python
def make_xgb_optimized(prep, random_state=42, search_mode="full"):
    """
    Create XGBoost pipeline with expanded hyperparameter grid.
    
    Parameters
    ----------
    prep : ColumnTransformer
        Preprocessing pipeline
    random_state : int
        Random seed
    search_mode : str
        "full" for comprehensive search, "fast" for quick search
        
    Returns
    -------
    tuple
        (pipeline, param_grid)
    """
```

**Expanded Grid**:
```python
grid = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [3, 6, 9],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma": [0, 0.1, 0.2],
}
```

### 5. Feature Selection Module

**Location**: `src/houseprice/feature_selection.py` (new file)

**Functions**:
- `get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame`
- `select_features_by_importance(X: pd.DataFrame, importance_df: pd.DataFrame, threshold: float) -> pd.DataFrame`

**Interface**:
```python
def get_feature_importance(model, 
                          feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Parameters
    ----------
    model : Pipeline
        Trained sklearn pipeline with XGBoost
    feature_names : List[str]
        Names of features after preprocessing
        
    Returns
    -------
    pd.DataFrame
        Dataframe with columns: feature, importance (sorted descending)
    """
```

### 6. Optimization Experiment Runner

**Location**: `scripts/optimize_xgboost.py` (new file)

Main script that orchestrates the optimization experiments:

```python
def run_optimization_experiment(
    data_path: Path,
    out_dir: Path,
    strategies: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run optimization experiments with different strategies.
    
    Parameters
    ----------
    data_path : Path
        Path to input data
    out_dir : Path
        Output directory for results
    strategies : List[str]
        List of strategies to test: 
        ["baseline", "hyperparams", "features", "outliers", 
         "transform", "feature_selection", "all"]
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each strategy
    """
```

## Data Models

### Feature Engineering Output Schema

After advanced feature engineering, the dataframe will contain:

**Original Features** (from dataset):
- bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade
- sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long
- sqft_living15, sqft_lot15

**Existing Engineered Features** (from current implementation):
- sale_year, sale_month, house_age, was_renovated
- sqft_x_grade, age_x_grade, living_lot_ratio

**New Advanced Features**:
- `total_sqft`: sqft_living + sqft_basement (if not already summed)
- `bathroom_density`: bathrooms / sqft_living
- `bedroom_density`: bedrooms / sqft_living
- `bedroom_bathroom_ratio`: bedrooms / bathrooms
- `quality_score`: grade * condition
- `renovation_age`: sale_year - yr_renovated (when renovated)
- `sale_quarter`: 1-4 based on sale_month
- `is_summer_sale`: 1 if sold in Jun-Aug, 0 otherwise
- `is_winter_sale`: 1 if sold in Dec-Feb, 0 otherwise
- `zipcode_median_price`: median price in that zipcode (computed from training data)
- `zipcode_price_std`: price std in that zipcode
- `price_vs_zipcode_median`: (house features suggest price) / zipcode_median_price

### Outlier Detection Schema

Outliers are identified based on:
- `price`: > mean + 3*std or < mean - 3*std
- `sqft_living`: > mean + 3*std
- `sqft_lot`: > mean + 3*std (if extremely large)

Returns a boolean mask indicating outlier rows.

### Transformation Metadata

The `TargetTransformer` stores:
- `method`: str - transformation type
- `lambda_`: float - Box-Cox lambda parameter (if applicable)
- `fitted_`: bool - whether transformer has been fitted

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Hyperparameter grid completeness
*For any* XGBoost model created with optimized settings, the hyperparameter grid should contain all specified regularization parameters (subsample, colsample_bytree, min_child_weight, gamma, n_estimators) with their defined value ranges, where subsample and colsample_bytree range from 0.6 to 1.0, min_child_weight includes [1, 3, 5], gamma includes [0, 0.1, 0.2], and n_estimators includes values up to 500.
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

### Property 2: Feature engineering preservation
*For any* input dataframe with valid house data, applying advanced feature engineering should preserve all original rows (same row count) and the price column values should remain unchanged.
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**

### Property 3: Feature count increase
*For any* input dataframe with required columns, the output of advanced feature engineering should have more columns than the input, specifically adding features for bathroom_density, bedroom_bathroom_ratio, quality_score, sale_quarter, and other engineered features.
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**

### Property 4: Outlier detection bounds
*For any* dataframe and outlier threshold, the number of outliers detected should be non-negative and less than or equal to the total number of rows.
**Validates: Requirements 3.1, 3.2**

### Property 5: Outlier removal integrity
*For any* dataframe with identified outliers, removing outliers should result in a dataframe with exactly (original_rows - outlier_count) rows, and all remaining rows should be identical to their original values.
**Validates: Requirements 3.4**

### Property 6: Transformation round-trip
*For any* positive price array and transformation method (log1p, boxcox, yeojohnson), applying the transformation followed by inverse transformation should return values approximately equal to the original within numerical precision (relative error < 1e-6).
**Validates: Requirements 4.5**

### Property 7: Feature importance completeness
*For any* trained XGBoost model with N features, the feature importance extraction should return exactly N importance scores, all values should be non-negative, and the sum of importance scores should be positive.
**Validates: Requirements 5.1**

### Property 8: Feature selection reduction
*For any* feature set and importance threshold > 0, applying feature selection should result in fewer or equal features compared to the original set, and all selected features should have importance >= threshold.
**Validates: Requirements 5.2**

### Property 9: Performance comparison completeness
*For any* set of optimization strategies tested, the comparison table should contain exactly one row per strategy, each row should have non-null RMSE and R² values, and all RMSE values should be positive.
**Validates: Requirements 7.1, 7.2, 7.3**

## Error Handling

### Feature Engineering Errors
- **Missing columns**: If expected columns are missing, skip that feature creation and log a warning
- **Division by zero**: Replace zero denominators with small epsilon (1e-6) to avoid inf/nan
- **Invalid dates**: Use `errors='coerce'` in date parsing, fill NaT with median year

### Outlier Handling Errors
- **Empty dataframe**: If all rows are outliers, raise ValueError with message
- **Invalid threshold**: If n_std <= 0, raise ValueError

### Transformation Errors
- **Negative values for Box-Cox**: Automatically fall back to Yeo-Johnson or log1p
- **Invalid method**: Raise ValueError with list of valid methods
- **Inverse transform before fit**: Raise RuntimeError

### Model Training Errors
- **Grid search failure**: Catch exceptions, log error, and fall back to default parameters
- **Memory errors**: Reduce grid size or use RandomizedSearchCV as fallback
- **Feature mismatch**: Ensure feature names are consistent between train and test

### Feature Selection Errors
- **No features meet threshold**: Log warning and keep all features
- **Model not fitted**: Raise RuntimeError with clear message

## Testing Strategy

### Unit Testing Framework
We will use `pytest` for unit testing with the following test modules:

**Test Files**:
- `tests/test_features_advanced.py`: Test advanced feature engineering
- `tests/test_outliers.py`: Test outlier detection and handling
- `tests/test_transforms.py`: Test target transformations
- `tests/test_feature_selection.py`: Test feature selection logic
- `tests/test_models_optimized.py`: Test optimized model creation

**Unit Test Coverage**:
- Test each feature engineering function with sample data
- Test outlier detection with known outlier cases
- Test transformation round-trips with various inputs
- Test feature importance extraction
- Test edge cases (empty data, single row, missing columns)

### Property-Based Testing Framework
We will use `hypothesis` for property-based testing to verify universal properties across random inputs.

**Configuration**: Each property test will run a minimum of 100 iterations.

**Property Test Tags**: Each test will include a comment in the format:
`# Feature: xgboost-optimization, Property {number}: {property_text}`

**Property Test Files**:
- `tests/test_properties_features.py`: Properties 2, 3
- `tests/test_properties_outliers.py`: Properties 4, 5
- `tests/test_properties_transforms.py`: Property 6
- `tests/test_properties_feature_selection.py`: Properties 7, 8

### Integration Testing
- Test the full optimization pipeline end-to-end
- Verify that each optimization strategy produces valid results
- Ensure comparison table generation works correctly
- Test with actual King County dataset

### Performance Testing
- Measure execution time for grid search with expanded parameters
- Verify that optimization completes within reasonable time (< 30 minutes)
- Test memory usage with full dataset

## Implementation Notes

### Hyperparameter Search Strategy
Given the expanded grid size (3^7 = 2187 combinations for full grid), we will:
1. Implement a "fast" mode with reduced grid for quick iterations
2. Use `n_jobs=-1` to parallelize across all CPU cores
3. Consider RandomizedSearchCV for very large grids

### Feature Engineering Order
Features must be created in dependency order:
1. Date features (year, month, quarter, season)
2. Age features (house_age, renovation_age)
3. Ratio features (require denominators to exist)
4. Aggregation features (require grouping)

### Zipcode Statistics
To avoid data leakage:
1. Compute zipcode statistics only on training data
2. Store the statistics dictionary
3. Apply same statistics to test data
4. Handle unseen zipcodes with global median

### Cross-Validation Consistency
All experiments must use the same:
- Random seed (42)
- Number of folds (5)
- Test set split (20%)
- Scoring metric (R² for optimization, RMSE for comparison)

This ensures fair comparison across strategies.
