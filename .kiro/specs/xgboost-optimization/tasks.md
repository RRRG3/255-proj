# Implementation Plan

- [x] 1. Create target transformation module
  - Implement `TargetTransformer` class with support for log1p, Box-Cox, and Yeo-Johnson transformations
  - Include fit_transform and inverse_transform methods
  - Handle edge cases (negative values, zero values)
  - _Requirements: 4.1, 4.2, 4.3, 4.5_

- [ ]* 1.1 Write property test for transformation round-trip
  - **Property 6: Transformation round-trip**
  - **Validates: Requirements 4.5**

- [x] 2. Create outlier handling module
  - Implement `detect_outliers` function using standard deviation method
  - Implement `remove_outliers` function to filter outlier rows
  - Implement `cap_outliers` function to limit extreme values
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ]* 2.1 Write property test for outlier detection bounds
  - **Property 4: Outlier detection bounds**
  - **Validates: Requirements 3.1, 3.2**

- [ ]* 2.2 Write property test for outlier removal integrity
  - **Property 5: Outlier removal integrity**
  - **Validates: Requirements 3.4**

- [x] 3. Implement advanced feature engineering
  - Create `engineer_advanced_features` function in features.py
  - Add price-per-sqft proxy features (bathroom_density, bedroom_density)
  - Add ratio features (bedroom_bathroom_ratio, living_lot_ratio enhancements)
  - Add quality score features (grade * condition)
  - Add temporal features (sale_quarter, seasonal indicators)
  - Add location aggregation features (zipcode statistics)
  - Ensure proper handling of missing values and division by zero
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [ ]* 3.1 Write property test for feature engineering preservation
  - **Property 2: Feature engineering preservation**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**

- [ ]* 3.2 Write property test for feature count increase
  - **Property 3: Feature count increase**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**

- [x] 4. Create optimized XGBoost model configuration
  - Update `make_xgb` function or create `make_xgb_optimized` in models.py
  - Define expanded hyperparameter grid with subsample, colsample_bytree, min_child_weight, gamma
  - Include n_estimators up to 500, max_depth up to 9, learning_rate options
  - Add search_mode parameter for "full" vs "fast" grid search
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ]* 4.1 Write unit test for hyperparameter grid completeness
  - **Property 1: Hyperparameter grid completeness**
  - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

- [x] 5. Create feature selection module
  - Implement `get_feature_importance` function to extract importance from trained model
  - Implement `select_features_by_importance` function to filter features by threshold
  - Handle feature name extraction from preprocessed pipeline
  - Return sorted DataFrame with feature names and importance scores
  - _Requirements: 5.1, 5.2_

- [ ]* 5.1 Write property test for feature importance completeness
  - **Property 7: Feature importance completeness**
  - **Validates: Requirements 5.1**

- [ ]* 5.2 Write property test for feature selection reduction
  - **Property 8: Feature selection reduction**
  - **Validates: Requirements 5.2**

- [x] 6. Create optimization experiment runner script
  - Create `scripts/optimize_xgboost.py` with main experiment orchestration
  - Implement baseline experiment (current approach)
  - Implement hyperparameter optimization experiment
  - Implement advanced features experiment
  - Implement outlier handling experiment
  - Implement target transformation comparison experiment
  - Implement feature selection experiment
  - Implement combined "all optimizations" experiment
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 6.1 Implement results comparison and reporting
  - Generate comparison table with RMSE, R², and improvement percentages
  - Save results to CSV and formatted text table
  - Create visualization comparing strategies
  - _Requirements: 7.3, 7.4, 7.5_

- [ ]* 6.2 Write property test for performance comparison completeness
  - **Property 9: Performance comparison completeness**
  - **Validates: Requirements 7.1, 7.2, 7.3**

- [x] 7. Integrate optimizations into existing pipeline
  - Update `run_cv.py` to optionally use advanced features
  - Add command-line flags for enabling different optimizations
  - Ensure backward compatibility with existing workflow
  - Update configuration file with new parameters
  - _Requirements: All_

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Run full optimization experiment
  - Execute optimization script with all strategies
  - Compare results against baseline
  - Document performance improvements
  - Generate final comparison report
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 10. Update documentation
  - Add docstrings to all new functions and classes
  - Update README with optimization instructions
  - Document hyperparameter tuning results
  - Create usage examples for new modules
  - _Requirements: All_
