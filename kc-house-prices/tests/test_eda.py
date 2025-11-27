
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from houseprice.eda import (
    plot_price_distribution,
    plot_log_price_distribution,
    plot_correlation_heatmap,
    plot_sqft_vs_price,
    plot_geographic_distribution
)


@pytest.fixture
def sample_df():
    """Create a minimal sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'price': np.random.lognormal(13, 0.5, n),
        'sqft_living': np.random.randint(1000, 5000, n),
        'sqft_lot': np.random.randint(5000, 20000, n),
        'bedrooms': np.random.randint(1, 6, n),
        'bathrooms': np.random.uniform(1, 4, n),
        'grade': np.random.randint(5, 12, n),
        'lat': np.random.uniform(47.1, 47.8, n),
        'long': np.random.uniform(-122.5, -121.5, n),
    })


def test_plot_price_distribution(sample_df):
    """Test price distribution plot generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "price_dist.png"
        plot_price_distribution(sample_df, outpath)
        assert outpath.exists()
        assert outpath.stat().st_size > 0


def test_plot_price_distribution_missing_column():
    """Test error handling for missing price column."""
    df = pd.DataFrame({'value': [1, 2, 3]})
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "test.png"
        with pytest.raises(ValueError, match="price"):
            plot_price_distribution(df, outpath)


def test_plot_log_price_distribution(sample_df):
    """Test log-transformed price distribution plot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "log_price_dist.png"
        plot_log_price_distribution(sample_df, outpath)
        assert outpath.exists()
        assert outpath.stat().st_size > 0


def test_plot_correlation_heatmap(sample_df):
    """Test correlation heatmap generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "correlation.png"
        plot_correlation_heatmap(sample_df, outpath, top_n=5)
        assert outpath.exists()
        assert outpath.stat().st_size > 0


def test_plot_sqft_vs_price(sample_df):
    """Test sqft vs price scatter plot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "sqft_vs_price.png"
        plot_sqft_vs_price(sample_df, outpath)
        assert outpath.exists()
        assert outpath.stat().st_size > 0


def test_plot_sqft_vs_price_without_grade(sample_df):
    """Test sqft vs price plot without grade column."""
    df = sample_df.drop(columns=['grade'])
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "sqft_vs_price.png"
        plot_sqft_vs_price(df, outpath)
        assert outpath.exists()


def test_plot_sqft_vs_price_missing_required():
    """Test error handling for missing required columns."""
    df = pd.DataFrame({'price': [1, 2, 3]})
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "test.png"
        with pytest.raises(ValueError, match="sqft_living"):
            plot_sqft_vs_price(df, outpath)


def test_plot_geographic_distribution(sample_df):
    """Test geographic distribution plot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "geographic.png"
        plot_geographic_distribution(sample_df, outpath)
        assert outpath.exists()
        assert outpath.stat().st_size > 0


def test_plot_geographic_distribution_large_dataset():
    """Test geographic plot with large dataset (sampling)."""
    np.random.seed(42)
    n = 60000  # Larger than 50k threshold
    df = pd.DataFrame({
        'price': np.random.lognormal(13, 0.5, n),
        'lat': np.random.uniform(47.1, 47.8, n),
        'long': np.random.uniform(-122.5, -121.5, n),
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "geographic.png"
        plot_geographic_distribution(df, outpath)
        assert outpath.exists()


def test_plot_with_nan_values(sample_df):
    """Test that plots handle NaN values gracefully."""
    df = sample_df.copy()
    df.loc[0:10, 'price'] = np.nan
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "test.png"
        plot_price_distribution(df, outpath)
        assert outpath.exists()
