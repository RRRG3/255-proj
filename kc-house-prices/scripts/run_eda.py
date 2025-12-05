import argparse
from pathlib import Path

from houseprice.config import DATA_PATH, OUT_DIR, EDA_PLOTS
from houseprice.data import load_data
from houseprice.eda import (
    plot_price_distribution,
    plot_log_price_distribution,
    plot_correlation_heatmap,
    plot_sqft_vs_price,
    plot_geographic_distribution
)


def main(data_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} records")
    
    print("\nGenerating EDA visualizations...")
    
    # Price distribution
    print("  - Price distribution...")
    plot_price_distribution(df, out_dir / EDA_PLOTS["price_dist"])
    
    # Log price distribution
    print("  - Log-transformed price distribution...")
    plot_log_price_distribution(df, out_dir / EDA_PLOTS["price_log_dist"])
    
    # Correlation heatmap
    print("  - Correlation heatmap...")
    plot_correlation_heatmap(df, out_dir / EDA_PLOTS["correlation"])
    
    # Sqft vs price
    print("  - Living area vs price scatter plot...")
    plot_sqft_vs_price(df, out_dir / EDA_PLOTS["sqft_vs_price"])
    
    # Geographic distribution
    print("  - Geographic distribution...")
    plot_geographic_distribution(df, out_dir / EDA_PLOTS["geographic"])
    
    print(f"\n✓ All EDA plots saved to {out_dir}")
    print("\nGenerated files:")
    for name, filename in EDA_PLOTS.items():
        filepath = out_dir / filename
        if filepath.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (not found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH,
                       help="Path to the dataset CSV file")
    parser.add_argument("--out", type=Path, default=OUT_DIR,
                       help="Output directory for plots")
    
    args = parser.parse_args()
    main(args.data, args.out)
