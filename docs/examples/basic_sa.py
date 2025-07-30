"""Basic seasonal adjustment example."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.sa.tramoseats import TramoSeatsSpecification, TramoSeatsProcessor
from jdemetra_py.visualization import plot_series, plot_decomposition

def main():
    """Run basic seasonal adjustment example."""
    
    # Generate synthetic monthly data with seasonal pattern
    np.random.seed(42)
    n_years = 5
    n_obs = n_years * 12
    
    # Time index
    t = np.arange(n_obs)
    
    # Components
    trend = 100 + 0.5 * t  # Linear trend
    seasonal = 10 * np.sin(2 * np.pi * t / 12)  # Annual seasonal pattern
    irregular = np.random.randn(n_obs) * 2  # Random noise
    
    # Combine components
    values = trend + seasonal + irregular
    
    # Create time series
    start = TsPeriod.of(TsFrequency.MONTHLY, 2019, 0)  # January 2019
    ts = TsData.of(start, values)
    
    print("Original time series:")
    print(f"  Start: {ts.start()}")
    print(f"  End: {ts.end()}")
    print(f"  Length: {ts.length()}")
    print(f"  Mean: {ts.average():.2f}")
    print()
    
    # Perform seasonal adjustment using TRAMO/SEATS
    print("Performing seasonal adjustment...")
    spec = TramoSeatsSpecification.rsa5()  # Automatic specification
    processor = TramoSeatsProcessor(spec)
    results = processor.process(ts)
    
    # Extract components
    decomposition = results.decomposition
    sa = decomposition.seasonally_adjusted
    trend_component = decomposition.trend
    seasonal_component = decomposition.seasonal
    irregular_component = decomposition.irregular
    
    print("\nDecomposition results:")
    print(f"  SA series mean: {sa.average():.2f}")
    print(f"  Seasonal range: {np.ptp(seasonal_component.values):.2f}")
    print(f"  Irregular std: {np.std(irregular_component.values):.2f}")
    
    # Quality diagnostics
    from jdemetra_py.sa.diagnostics import QualityMeasures, SeasonalityTests
    
    quality = QualityMeasures(decomposition)
    print(f"\nQuality measures:")
    print(f"  M1: {quality.m1():.3f}")
    print(f"  M3: {quality.m3():.3f}")
    print(f"  M7: {quality.m7():.3f}")
    print(f"  Q: {quality.q():.3f}")
    
    # Test for residual seasonality
    seas_test = SeasonalityTests.qs_test(sa, ts.frequency())
    print(f"\nResidual seasonality test:")
    print(f"  QS statistic: {seas_test.statistic:.3f}")
    print(f"  p-value: {seas_test.pvalue:.4f}")
    print(f"  Seasonality present: {'Yes' if seas_test.pvalue < 0.05 else 'No'}")
    
    # Visualization
    print("\nGenerating plots...")
    
    # Plot original and SA series
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to pandas for easier plotting
    dates = pd.date_range(start='2019-01', periods=n_obs, freq='M')
    
    ax.plot(dates, ts.values, label='Original', linewidth=2)
    ax.plot(dates, sa.values, label='Seasonally Adjusted', linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Original vs Seasonally Adjusted Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_sa_comparison.png', dpi=300)
    print("  Saved: basic_sa_comparison.png")
    
    # Plot full decomposition
    fig = plot_decomposition(decomposition, title='TRAMO/SEATS Decomposition')
    fig.savefig('basic_sa_decomposition.png', dpi=300)
    print("  Saved: basic_sa_decomposition.png")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'date': dates,
        'original': ts.values,
        'seasonally_adjusted': sa.values,
        'trend': trend_component.values,
        'seasonal': seasonal_component.values,
        'irregular': irregular_component.values
    })
    
    results_df.to_csv('basic_sa_results.csv', index=False)
    print("  Saved: basic_sa_results.csv")
    
    print("\nExample completed successfully!")

if __name__ == '__main__':
    main()