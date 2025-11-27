from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_price_distribution(df: pd.DataFrame, outpath: Path) -> None:
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain 'price' column")
    
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    prices = df["price"].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(prices, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of House Prices")
    
    # Add statistics overlay
    mean_price = prices.mean()
    median_price = prices.median()
    ax.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:,.0f}')
    ax.axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_price:,.0f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


def plot_log_price_distribution(df: pd.DataFrame, outpath: Path) -> None:
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain 'price' column")
    
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    prices = df["price"].dropna()
    log_prices = np.log1p(prices)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_prices, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("log(Price + 1)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Log-Transformed House Prices")
    
    # Add statistics
    mean_log = log_prices.mean()
    median_log = log_prices.median()
    ax.axvline(mean_log, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_log:.2f}')
    ax.axvline(median_log, color='green', linestyle='--', linewidth=2, label=f'Median: {median_log:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, outpath: Path, top_n: int = 15) -> None:
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain 'price' column")
    
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    # Select numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlations with price
    correlations = numeric_df.corr()["price"].abs().sort_values(ascending=False)
    
    # Select top N features (including price itself)
    top_features = correlations.head(top_n).index.tolist()
    
    # Create correlation matrix for top features
    corr_matrix = numeric_df[top_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title(f"Correlation Heatmap - Top {top_n} Features")
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


def plot_sqft_vs_price(df: pd.DataFrame, outpath: Path) -> None:
    """Scatter plot of sqft_living vs price, colored by grade.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'sqft_living', 'price', and optionally 'grade' columns
    outpath : Path
        Output path for the PNG file
    """
    required_cols = ["sqft_living", "price"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns: {missing}")
    
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    # Drop rows with NaN in required columns
    plot_df = df[["sqft_living", "price"]].copy()
    if "grade" in df.columns:
        plot_df["grade"] = df["grade"]
    
    plot_df = plot_df.dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if "grade" in plot_df.columns:
        scatter = ax.scatter(plot_df["sqft_living"], plot_df["price"], 
                           c=plot_df["grade"], cmap='viridis', alpha=0.5, s=10)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Grade")
    else:
        ax.scatter(plot_df["sqft_living"], plot_df["price"], alpha=0.5, s=10)
    
    ax.set_xlabel("Living Area (sqft)")
    ax.set_ylabel("Price ($)")
    ax.set_title("House Price vs Living Area")
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


def plot_geographic_distribution(df: pd.DataFrame, outpath: Path) -> None:
    """Geographic scatter plot (lat/long) with price as color intensity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'lat', 'long', and 'price' columns
    outpath : Path
        Output path for the PNG file
    """
    required_cols = ["lat", "long", "price"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns: {missing}")
    
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    # Drop rows with NaN
    plot_df = df[required_cols].dropna()
    
    # Sample if dataset is very large (>50k points)
    if len(plot_df) > 50000:
        plot_df = plot_df.sample(n=50000, random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(plot_df["long"], plot_df["lat"], 
                        c=plot_df["price"], cmap='YlOrRd', 
                        alpha=0.6, s=5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Price ($)")
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of House Prices")
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()
