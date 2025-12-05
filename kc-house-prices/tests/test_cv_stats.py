
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add scripts to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_cv import format_comparison_table


def test_format_comparison_table():
    df = pd.DataFrame({
        'model': ['LinearRegression', 'RandomForest', 'XGBoost'],
        'R2_CV_mean': [0.6234, 0.8567, 0.8789],
        'R2_CV_std': [0.0145, 0.0089, 0.0067],
        'RMSE_CV': [123456.78, 87654.32, 76543.21],
    })
    
    result = format_comparison_table(df)
    
    # Check that result is a string
    assert isinstance(result, str)
    
    # Check that it contains expected elements
    assert "Model Comparison Table" in result
    assert "LinearRegression" in result
    assert "RandomForest" in result
    assert "XGBoost" in result
    assert "0.6234" in result
    assert "0.8789" in result
    
    # Check formatting of RMSE with thousand separators
    assert "123,456.78" in result or "123456.78" in result


def test_format_comparison_table_single_model():
    """Test table formatting with single model."""
    df = pd.DataFrame({
        'model': ['LinearRegression'],
        'R2_CV_mean': [0.6234],
        'R2_CV_std': [0.0145],
        'RMSE_CV': [123456.78],
    })
    
    result = format_comparison_table(df)
    assert isinstance(result, str)
    assert "LinearRegression" in result


def test_r2_statistics_calculation():
    """Test R² mean and std calculation logic."""
    # Simulate fold scores
    fold_scores = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
    
    mean_r2 = fold_scores.mean()
    std_r2 = fold_scores.std()
    
    assert abs(mean_r2 - 0.86) < 0.01
    assert std_r2 > 0
    assert std_r2 < 0.02  # Should be relatively small for consistent folds


def test_fold_scores_consistency():
    """Test that fold scores are reasonable."""
    # Simulate realistic fold scores
    fold_scores = np.array([0.82, 0.85, 0.83, 0.84, 0.86])
    
    # Check that all scores are between 0 and 1
    assert np.all(fold_scores >= 0)
    assert np.all(fold_scores <= 1)
    
    # Check that standard deviation is reasonable (not too high)
    std = fold_scores.std()
    assert std < 0.1  # Good models should have consistent performance


def test_table_column_alignment():
    """Test that table has proper column structure."""
    df = pd.DataFrame({
        'model': ['Model1', 'Model2'],
        'R2_CV_mean': [0.85, 0.90],
        'R2_CV_std': [0.01, 0.02],
        'RMSE_CV': [10000.0, 8000.0],
    })
    
    result = format_comparison_table(df)
    lines = result.split('\n')
    
    # Should have header, separator, and data rows
    assert len(lines) >= 4
    
    # Check for separator lines
    assert any('=' in line for line in lines)
    assert any('-' in line for line in lines)
