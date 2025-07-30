"""Visualization examples for JDemetra+ Python."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.sa.tramoseats import TramoSeatsSpecification, TramoSeatsProcessor
from jdemetra_py.visualization import (
    TimeSeriesPlotter, DecompositionPlotter, DiagnosticPlotter,
    plot_series, plot_decomposition, plot_diagnostics
)
from jdemetra_py.sa.diagnostics import QualityMeasures

def create_sample_series() -> Dict[str, TsData]:
    """Create multiple sample series for visualization."""
    np.random.seed(42)
    
    series_dict = {}
    start = TsPeriod.of(TsFrequency.MONTHLY, 2018, 0)
    n_obs = 72  # 6 years
    t = np.arange(n_obs)
    
    # Series 1: Clean seasonal pattern
    trend1 = 100 + 0.5 * t
    seasonal1 = 10 * np.sin(2 * np.pi * t / 12)
    irregular1 = np.random.randn(n_obs) * 2
    values1 = trend1 + seasonal1 + irregular1
    series_dict['Clean'] = TsData.of(start, values1)
    
    # Series 2: With outliers
    values2 = values1.copy()
    values2[20] += 25  # Additive outlier
    values2[40:] += 10  # Level shift
    series_dict['With_Outliers'] = TsData.of(start, values2)
    
    # Series 3: Changing seasonal pattern
    seasonal3 = 10 * np.sin(2 * np.pi * t / 12) * (1 + 0.01 * t)
    values3 = trend1 + seasonal3 + irregular1
    series_dict['Changing_Seasonal'] = TsData.of(start, values3)
    
    # Series 4: Multiple frequencies
    seasonal4a = 10 * np.sin(2 * np.pi * t / 12)
    seasonal4b = 5 * np.sin(2 * np.pi * t / 6)
    values4 = trend1 + seasonal4a + seasonal4b + irregular1
    series_dict['Multiple_Frequencies'] = TsData.of(start, values4)
    
    return series_dict

def basic_plotting_examples(series_dict: Dict[str, TsData]):
    """Demonstrate basic plotting functionality."""
    print("\nBasic Plotting Examples")
    print("=" * 50)
    
    # 1. Single series plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ts = series_dict['Clean']
    dates = pd.date_range('2018-01', periods=ts.length(), freq='M')
    
    ax.plot(dates, ts.values, linewidth=2)
    ax.set_title('Single Time Series Plot', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.axhline(y=ts.average(), color='r', linestyle='--', alpha=0.5, label='Mean')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('viz_single_series.png', dpi=300)
    print("Saved: viz_single_series.png")
    
    # 2. Multiple series comparison
    fig = plot_series(
        list(series_dict.values()),
        labels=list(series_dict.keys()),
        title='Multiple Series Comparison',
        figsize=(14, 8)
    )
    plt.savefig('viz_multiple_series.png', dpi=300)
    print("Saved: viz_multiple_series.png")
    
    # 3. Series with confidence bands
    plotter = TimeSeriesPlotter()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate rolling mean and std
    window = 12
    ts_values = pd.Series(ts.values)
    rolling_mean = ts_values.rolling(window, center=True).mean()
    rolling_std = ts_values.rolling(window, center=True).std()
    
    # Plot with confidence bands
    ax.plot(dates, ts.values, 'b-', alpha=0.5, label='Original')
    ax.plot(dates, rolling_mean, 'r-', linewidth=2, label='12-month MA')
    ax.fill_between(dates, 
                    rolling_mean - 2*rolling_std,
                    rolling_mean + 2*rolling_std,
                    alpha=0.2, color='red', label='95% CI')
    
    ax.set_title('Time Series with Confidence Bands')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('viz_confidence_bands.png', dpi=300)
    print("Saved: viz_confidence_bands.png")

def decomposition_plotting_examples(series_dict: Dict[str, TsData]):
    """Demonstrate decomposition plotting."""
    print("\n\nDecomposition Plotting Examples")
    print("=" * 50)
    
    # Process series
    spec = TramoSeatsSpecification.rsa5()
    processor = TramoSeatsProcessor(spec)
    
    results = {}
    for name, ts in series_dict.items():
        try:
            results[name] = processor.process(ts)
        except Exception as e:
            print(f"Failed to process {name}: {e}")
    
    # 1. Standard decomposition plot
    if 'Clean' in results:
        fig = plot_decomposition(
            results['Clean'].decomposition,
            title='Standard Decomposition Plot'
        )
        plt.savefig('viz_decomposition_standard.png', dpi=300)
        print("Saved: viz_decomposition_standard.png")
    
    # 2. Custom decomposition plot
    if 'With_Outliers' in results:
        plotter = DecompositionPlotter()
        fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
        
        decomp = results['With_Outliers'].decomposition
        dates = pd.date_range('2018-01', periods=decomp.original.length(), freq='M')
        
        # Original
        axes[0].plot(dates, decomp.original.values, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Original')
        axes[0].set_title('Decomposition with Outlier Effects')
        axes[0].grid(True, alpha=0.3)
        
        # Seasonally adjusted
        axes[1].plot(dates, decomp.seasonally_adjusted.values, 'g-', linewidth=1.5)
        axes[1].set_ylabel('SA')
        axes[1].grid(True, alpha=0.3)
        
        # Trend
        axes[2].plot(dates, decomp.trend.values, 'r-', linewidth=2)
        axes[2].set_ylabel('Trend')
        axes[2].grid(True, alpha=0.3)
        
        # Seasonal
        axes[3].plot(dates, decomp.seasonal.values, 'm-', linewidth=1)
        axes[3].set_ylabel('Seasonal')
        axes[3].grid(True, alpha=0.3)
        
        # Irregular with outliers highlighted
        axes[4].plot(dates, decomp.irregular.values, 'k-', alpha=0.5)
        
        # Highlight outlier periods
        outlier_periods = [20, 40]  # Known outlier positions
        for period in outlier_periods:
            if period < len(dates):
                axes[4].axvline(x=dates[period], color='red', alpha=0.3, linewidth=2)
        
        axes[4].set_ylabel('Irregular')
        axes[4].set_xlabel('Date')
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('viz_decomposition_custom.png', dpi=300)
        print("Saved: viz_decomposition_custom.png")
    
    # 3. Decomposition comparison
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 4*len(results)))
    
    for i, (name, result) in enumerate(results.items()):
        decomp = result.decomposition
        dates = pd.date_range('2018-01', periods=decomp.original.length(), freq='M')
        
        # Original vs SA
        axes[i, 0].plot(dates, decomp.original.values, 'b-', alpha=0.5, label='Original')
        axes[i, 0].plot(dates, decomp.seasonally_adjusted.values, 'r-', linewidth=2, label='SA')
        axes[i, 0].set_title(f'{name}: Original vs SA')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Seasonal pattern
        seasonal_cycle = decomp.seasonal.values[:12]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[i, 1].bar(months, seasonal_cycle)
        axes[i, 1].set_title(f'{name}: Seasonal Pattern')
        axes[i, 1].tick_params(axis='x', rotation=45)
        axes[i, 1].grid(True, alpha=0.3, axis='y')
        
        # Irregular distribution
        axes[i, 2].hist(decomp.irregular.values, bins=30, density=True, alpha=0.7)
        axes[i, 2].set_title(f'{name}: Irregular Distribution')
        axes[i, 2].set_xlabel('Value')
        axes[i, 2].set_ylabel('Density')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('viz_decomposition_comparison.png', dpi=300)
    print("Saved: viz_decomposition_comparison.png")

def diagnostic_plotting_examples(results: Dict):
    """Demonstrate diagnostic plotting."""
    print("\n\nDiagnostic Plotting Examples")
    print("=" * 50)
    
    if not results:
        print("No results available for diagnostic plots")
        return
    
    # 1. Full diagnostic plot
    if 'Clean' in results:
        fig = plot_diagnostics(results['Clean'], figsize=(16, 12))
        plt.savefig('viz_diagnostics_full.png', dpi=300)
        print("Saved: viz_diagnostics_full.png")
    
    # 2. Custom diagnostic plots
    plotter = DiagnosticPlotter()
    
    # Quality measures comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    quality_data = []
    for name, result in results.items():
        quality = QualityMeasures(result.decomposition)
        quality_data.append({
            'Series': name,
            'M1': quality.m1(),
            'M2': quality.m2(),
            'M3': quality.m3(),
            'M4': quality.m4(),
            'M5': quality.m5(),
            'M6': quality.m6(),
            'M7': quality.m7(),
            'Q': quality.q()
        })
    
    quality_df = pd.DataFrame(quality_data)
    quality_df.set_index('Series', inplace=True)
    
    # Heatmap of quality measures
    sns.heatmap(quality_df.T, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=1.0, ax=ax, cbar_kws={'label': 'Value'})
    ax.set_title('Quality Measures Heatmap')
    
    plt.tight_layout()
    plt.savefig('viz_quality_heatmap.png', dpi=300)
    print("Saved: viz_quality_heatmap.png")
    
    # 3. Residual diagnostics grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if 'Clean' in results:
        residuals = results['Clean'].preprocessing.residuals
        
        # Residual plot
        axes[0, 0].plot(residuals, 'k-', alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram with normal overlay
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, color='blue')
        
        # Normal distribution overlay
        from scipy import stats
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
                       'r-', linewidth=2, label='Normal')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF of residuals
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=36, ax=axes[1, 0])
        axes[1, 0].set_title('Residual Autocorrelation')
        
        # Squared residuals ACF (for heteroscedasticity)
        plot_acf(residuals**2, lags=36, ax=axes[1, 1])
        axes[1, 1].set_title('Squared Residual Autocorrelation')
    
    plt.tight_layout()
    plt.savefig('viz_residual_diagnostics.png', dpi=300)
    print("Saved: viz_residual_diagnostics.png")

def create_dashboard_example(series_dict: Dict[str, TsData], results: Dict):
    """Create a comprehensive dashboard-style visualization."""
    print("\n\nDashboard Example")
    print("=" * 50)
    
    # Create figure with GridSpec for complex layout
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main time series plot (spans 2 columns)
    ax_main = fig.add_subplot(gs[0, :2])
    
    if 'Clean' in results:
        decomp = results['Clean'].decomposition
        dates = pd.date_range('2018-01', periods=decomp.original.length(), freq='M')
        
        ax_main.plot(dates, decomp.original.values, 'b-', alpha=0.5, 
                    label='Original', linewidth=1)
        ax_main.plot(dates, decomp.seasonally_adjusted.values, 'r-', 
                    label='Seasonally Adjusted', linewidth=2)
        ax_main.plot(dates, decomp.trend.values, 'g--', 
                    label='Trend', linewidth=2)
        
        ax_main.set_title('Time Series Decomposition', fontsize=16)
        ax_main.set_xlabel('Date')
        ax_main.set_ylabel('Value')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
    
    # Quality gauge chart
    ax_quality = fig.add_subplot(gs[0, 2])
    
    if 'Clean' in results:
        quality = QualityMeasures(results['Clean'].decomposition)
        q_stat = quality.q()
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Color zones
        colors = ['green', 'yellow', 'red']
        boundaries = [0, 1.0, 1.5, 3.0]
        
        for i in range(3):
            mask = (theta >= boundaries[i] * np.pi / 3) & (theta < boundaries[i+1] * np.pi / 3)
            ax_quality.fill_between(x[mask], 0, y[mask], color=colors[i], alpha=0.3)
        
        # Needle
        angle = min(q_stat * np.pi / 3, np.pi)
        ax_quality.plot([0, np.cos(angle)], [0, np.sin(angle)], 'k-', linewidth=3)
        ax_quality.plot(0, 0, 'ko', markersize=10)
        
        ax_quality.set_xlim(-1.2, 1.2)
        ax_quality.set_ylim(-0.2, 1.2)
        ax_quality.set_aspect('equal')
        ax_quality.set_title(f'Quality Gauge (Q={q_stat:.2f})', fontsize=14)
        ax_quality.axis('off')
    
    # Summary statistics
    ax_stats = fig.add_subplot(gs[0, 3])
    ax_stats.axis('off')
    
    if 'Clean' in results:
        decomp = results['Clean'].decomposition
        stats_text = f"""Summary Statistics
        
Original Mean: {decomp.original.average():.2f}
SA Mean: {decomp.seasonally_adjusted.average():.2f}
Trend Mean: {decomp.trend.average():.2f}

Seasonal Range: {np.ptp(decomp.seasonal.values):.2f}
Irregular Std: {np.std(decomp.irregular.values):.2f}

Series Length: {decomp.original.length()}
Frequency: Monthly"""
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Seasonal pattern
    ax_seasonal = fig.add_subplot(gs[1, :2])
    
    if 'Clean' in results:
        seasonal = results['Clean'].decomposition.seasonal.values
        
        # Show multiple years of seasonal pattern
        years = min(3, len(seasonal) // 12)
        for year in range(years):
            start_idx = year * 12
            end_idx = start_idx + 12
            if end_idx <= len(seasonal):
                months = np.arange(1, 13)
                ax_seasonal.plot(months, seasonal[start_idx:end_idx], 
                               'o-', alpha=0.7, label=f'Year {year+1}')
        
        ax_seasonal.set_title('Seasonal Pattern by Year', fontsize=14)
        ax_seasonal.set_xlabel('Month')
        ax_seasonal.set_ylabel('Seasonal Effect')
        ax_seasonal.set_xticks(range(1, 13))
        ax_seasonal.legend()
        ax_seasonal.grid(True, alpha=0.3)
    
    # Model comparison
    ax_comparison = fig.add_subplot(gs[1, 2:])
    
    # Compare different series
    comparison_data = []
    for name, result in results.items():
        quality = QualityMeasures(result.decomposition)
        comparison_data.append({
            'Series': name.replace('_', ' '),
            'Q-Stat': quality.q(),
            'M7': quality.m7()
        })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        x = np.arange(len(comp_df))
        width = 0.35
        
        bars1 = ax_comparison.bar(x - width/2, comp_df['Q-Stat'], width, label='Q-Stat')
        bars2 = ax_comparison.bar(x + width/2, comp_df['M7'], width, label='M7')
        
        ax_comparison.set_xlabel('Series')
        ax_comparison.set_ylabel('Value')
        ax_comparison.set_title('Quality Measures Comparison', fontsize=14)
        ax_comparison.set_xticks(x)
        ax_comparison.set_xticklabels(comp_df['Series'])
        ax_comparison.legend()
        ax_comparison.grid(True, alpha=0.3, axis='y')
        
        # Add reference lines
        ax_comparison.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    # Forecast plot
    ax_forecast = fig.add_subplot(gs[2, :])
    
    if 'Clean' in results:
        # Simulate forecast (in real implementation, use actual forecast)
        ts = series_dict['Clean']
        dates = pd.date_range('2018-01', periods=ts.length(), freq='M')
        
        # Last 2 years of data
        n_history = 24
        ax_forecast.plot(dates[-n_history:], ts.values[-n_history:], 
                        'b-', linewidth=2, label='Historical')
        
        # Simulated forecast
        n_forecast = 12
        forecast_dates = pd.date_range(dates[-1] + pd.DateOffset(months=1), 
                                     periods=n_forecast, freq='M')
        
        # Simple trend extrapolation for demo
        trend = np.polyfit(range(n_history), ts.values[-n_history:], 1)
        forecast_values = np.polyval(trend, range(n_history, n_history + n_forecast))
        
        # Add seasonal pattern
        seasonal = results['Clean'].decomposition.seasonal.values[-12:]
        forecast_values += seasonal
        
        # Add uncertainty
        forecast_std = np.std(results['Clean'].decomposition.irregular.values)
        
        ax_forecast.plot(forecast_dates, forecast_values, 'r--', linewidth=2, label='Forecast')
        ax_forecast.fill_between(forecast_dates,
                               forecast_values - 2*forecast_std,
                               forecast_values + 2*forecast_std,
                               alpha=0.2, color='red', label='95% CI')
        
        ax_forecast.set_title('Historical Data and Forecast', fontsize=16)
        ax_forecast.set_xlabel('Date')
        ax_forecast.set_ylabel('Value')
        ax_forecast.legend()
        ax_forecast.grid(True, alpha=0.3)
        
        # Vertical line at forecast start
        ax_forecast.axvline(x=dates[-1], color='gray', linestyle=':', alpha=0.5)
    
    plt.suptitle('JDemetra+ Python - Seasonal Adjustment Dashboard', fontsize=20)
    plt.savefig('viz_dashboard.png', dpi=300, bbox_inches='tight')
    print("Saved: viz_dashboard.png")

def main():
    """Run visualization examples."""
    print("Visualization Examples")
    print("=" * 50)
    
    # Create sample series
    print("\nCreating sample time series...")
    series_dict = create_sample_series()
    print(f"Created {len(series_dict)} series")
    
    # Basic plotting
    basic_plotting_examples(series_dict)
    
    # Process series for decomposition examples
    print("\nProcessing series for decomposition...")
    spec = TramoSeatsSpecification.rsa5()
    processor = TramoSeatsProcessor(spec)
    
    results = {}
    for name, ts in series_dict.items():
        try:
            print(f"  Processing {name}...")
            results[name] = processor.process(ts)
        except Exception as e:
            print(f"  Failed to process {name}: {e}")
    
    # Decomposition plotting
    decomposition_plotting_examples(series_dict)
    
    # Diagnostic plotting
    diagnostic_plotting_examples(results)
    
    # Dashboard example
    create_dashboard_example(series_dict, results)
    
    print("\nVisualization examples completed!")
    print("Generated files:")
    print("  - viz_single_series.png")
    print("  - viz_multiple_series.png")
    print("  - viz_confidence_bands.png")
    print("  - viz_decomposition_standard.png")
    print("  - viz_decomposition_custom.png")
    print("  - viz_decomposition_comparison.png")
    print("  - viz_diagnostics_full.png")
    print("  - viz_quality_heatmap.png")
    print("  - viz_residual_diagnostics.png")
    print("  - viz_dashboard.png")

if __name__ == '__main__':
    main()