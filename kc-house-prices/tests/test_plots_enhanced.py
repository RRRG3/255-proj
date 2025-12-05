import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from houseprice.plots import (
    plot_lr_residuals_enhanced,
    plot_feature_importance_comparison,
    plot_linear_coefficients
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
    })
    # Use absolute values to avoid negative inputs to log1p
    y = np.log1p(np.abs(X['feature1'] * 2 + X['feature2'] * 3) + np.abs(np.random.randn(n) * 0.1))
    return X, y


@pytest.fixture
def trained_lr_model(sample_data):
    """Create a trained linear regression model."""
    X, y = sample_data
    prep = ColumnTransformer([
        ("scaler", StandardScaler(), ['feature1', 'feature2', 'feature3'])
    ])
    model = Pipeline([
        ("prep", prep),
        ("model", LinearRegression())
    ])
    model.fit(X, y)
    return model, prep


@pytest.fixture
def trained_rf_model(sample_data):
    """Create a trained random forest model."""
    X, y = sample_data
    prep = ColumnTransformer([
        ("passthrough", "passthrough", ['feature1', 'feature2', 'feature3'])
    ])
    model = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(n_estimators=10, random_state=42))
    ])
    model.fit(X, y)
    return model


def test_plot_lr_residuals_enhanced(trained_lr_model, sample_data):
    """Test enhanced residual plot generation."""
    model, _ = trained_lr_model
    X, y = sample_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "residuals_enhanced.png"
        plot_lr_residuals_enhanced(model, X, y, outpath)
        assert outpath.exists()
        assert outpath.stat().st_size > 0


def test_plot_feature_importance_comparison(trained_rf_model, sample_data):
    """Test feature importance comparison plot."""
    rf_model = trained_rf_model
    
    # Create a second model (XGBoost-like)
    X, y = sample_data
    prep = ColumnTransformer([
        ("passthrough", "passthrough", ['feature1', 'feature2', 'feature3'])
    ])
    xgb_model = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(n_estimators=10, random_state=43))
    ])
    xgb_model.fit(X, y)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "importance_comparison.png"
        plot_feature_importance_comparison(rf_model, xgb_model, prep, outpath, top_k=3)
        assert outpath.exists()
        assert outpath.stat().st_size > 0


def test_plot_linear_coefficients(trained_lr_model):
    """Test linear coefficients plot."""
    model, prep = trained_lr_model
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "coefficients.png"
        plot_linear_coefficients(model, prep, outpath, top_k=3)
        assert outpath.exists()
        assert outpath.stat().st_size > 0


def test_plot_with_many_features():
    """Test plotting with many features (top_k selection)."""
    np.random.seed(42)
    n = 100
    n_features = 50
    
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = np.log1p(np.abs(np.random.randn(n)))
    
    prep = ColumnTransformer([
        ("passthrough", "passthrough", X.columns.tolist())
    ])
    
    model = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(n_estimators=10, random_state=42))
    ])
    model.fit(X, y)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "importance.png"
        # Should only plot top 15 features
        from houseprice.plots import plot_tree_importance
        plot_tree_importance(model, outpath, top_k=15)
        assert outpath.exists()
