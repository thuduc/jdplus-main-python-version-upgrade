"""Advanced features example demonstrating custom specifications and diagnostics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.toolkit.arima import SarimaOrder, ArimaEstimator
from jdemetra_py.sa.tramoseats import (
    TramoSeatsSpecification, TramoSeatsProcessor,
    TramoProcessor, SeatsDecomposer
)
from jdemetra_py.sa.diagnostics import (
    SeasonalityTests, QualityMeasures, ResidualsDiagnostics,
    OutOfSampleTest, SlidingSpansAnalysis
)
from jdemetra_py.visualization import DiagnosticPlotter

def create_complex_series() -> TsData:
    """Create a complex time series with multiple features."""
    np.random.seed(42)
    
    # 10 years of monthly data
    n_obs = 120
    t = np.arange(n_obs)
    
    # Complex trend with acceleration
    trend = 100 + 0.5 * t + 0.01 * t**2
    
    # Multiple seasonal patterns
    seasonal1 = 10 * np.sin(2 * np.pi * t / 12)  # Annual cycle
    seasonal2 = 3 * np.sin(2 * np.pi * t / 6)   # Semi-annual cycle
    
    # Calendar effects (simplified)
    calendar = 2 * np.sin(2 * np.pi * t / 12 + np.pi/4)
    
    # Irregular with changing variance
    irregular = np.random.randn(n_obs) * (1 + 0.01 * t)
    
    # Add outliers
    values = trend + seasonal1 + seasonal2 + calendar + irregular
    
    # Additive outlier
    values[30] += 20
    
    # Level shift
    values[60:] += 10
    
    # Transitory change
    tc_effect = 15
    for i in range(5):
        values[80 + i] += tc_effect * (0.7 ** i)
    
    # Create time series
    start = TsPeriod.of(TsFrequency.MONTHLY, 2014, 0)
    return TsData.of(start, values)

def custom_specification_example(ts: TsData) -> Dict:
    """Demonstrate custom specification settings."""
    print("\nCustom Specification Example")
    print("=" * 50)
    
    results = {}
    
    # 1. Basic automatic specification
    spec_auto = TramoSeatsSpecification.rsa5()
    processor = TramoSeatsProcessor(spec_auto)
    results['automatic'] = processor.process(ts)
    
    # 2. Custom ARIMA specification
    spec_custom_arima = TramoSeatsSpecification()
    spec_custom_arima.set_transform('log')
    spec_custom_arima.set_arima_specification({
        'p': 2,  # AR order
        'd': 1,  # Differencing
        'q': 1,  # MA order
        'P': 1,  # Seasonal AR
        'D': 1,  # Seasonal differencing
        'Q': 1,  # Seasonal MA
        'mean': True  # Include mean
    })
    results['custom_arima'] = processor.process(ts, spec_custom_arima)
    
    # 3. Custom outlier detection
    spec_outliers = TramoSeatsSpecification()
    spec_outliers.set_transform('auto')
    spec_outliers.set_outliers(['AO', 'LS', 'TC', 'SO'])  # All outlier types
    spec_outliers.set_outlier_critical_value(3.0)  # Lower threshold
    results['custom_outliers'] = processor.process(ts, spec_outliers)
    
    # 4. Custom regression variables
    spec_regression = TramoSeatsSpecification()
    spec_regression.set_transform('log')
    spec_regression.set_trading_days(True)
    spec_regression.set_easter(True, duration=8)  # 8-day Easter effect
    spec_regression.set_leap_year(True)
    results['custom_regression'] = processor.process(ts, spec_regression)
    
    # 5. Custom SEATS parameters
    spec_seats = TramoSeatsSpecification()
    spec_seats.set_transform('log')
    spec_seats.set_seats_parameters(
        xl=0.95,      # Lower bound for AR roots
        epsphi=0.05,  # Tolerance for identifying AR roots
        rmod=0.8,     # Threshold for model reduction
        smod='Unchanged'  # Keep seasonal component
    )
    results['custom_seats'] = processor.process(ts, spec_seats)
    
    return results

def compare_specifications(results: Dict) -> pd.DataFrame:
    """Compare results from different specifications."""
    comparison_data = []
    
    for name, result in results.items():
        if result is not None:
            decomp = result.decomposition
            quality = QualityMeasures(decomp)
            
            # Get preprocessing info
            preprocessing = result.preprocessing
            n_outliers = len(preprocessing.outliers) if preprocessing else 0
            
            comparison_data.append({
                'Specification': name,
                'Q-Stat': round(quality.q(), 3),
                'M1': round(quality.m1(), 3),
                'M7': round(quality.m7(), 3),
                'Outliers': n_outliers,
                'SA Mean': round(decomp.seasonally_adjusted.average(), 2),
                'Trend Mean': round(decomp.trend.average(), 2),
                'Seasonal Range': round(np.ptp(decomp.seasonal.values), 2)
            })
    
    return pd.DataFrame(comparison_data)

def advanced_diagnostics(ts: TsData, results: Dict):
    """Perform advanced diagnostic tests."""
    print("\nAdvanced Diagnostics")
    print("=" * 50)
    
    # Use the automatic results for diagnostics
    result = results['automatic']
    decomp = result.decomposition
    
    # 1. Comprehensive seasonality tests
    print("\n1. Seasonality Tests:")
    sa = decomp.seasonally_adjusted
    
    tests = {
        'QS Test': SeasonalityTests.qs_test(sa, 12),
        'Friedman Test': SeasonalityTests.friedman_test(sa, 12),
        'Kruskal-Wallis': SeasonalityTests.kruskal_wallis_test(sa, 12),
        'Combined Test': SeasonalityTests.combined_test(sa, 12)
    }
    
    for test_name, test_result in tests.items():
        print(f"  {test_name}: stat={test_result.statistic:.3f}, p-value={test_result.pvalue:.4f}")
    
    # 2. Residual diagnostics
    print("\n2. Residual Diagnostics:")
    residuals = result.preprocessing.residuals
    res_diag = ResidualsDiagnostics(residuals)
    
    # Ljung-Box test at different lags
    for lag in [12, 24, 36]:
        lb_test = res_diag.ljung_box_test(lag)
        print(f"  Ljung-Box (lag={lag}): stat={lb_test.statistic:.3f}, p-value={lb_test.pvalue:.4f}")
    
    # Other tests
    norm_test = res_diag.normality_test()
    print(f"  Normality: stat={norm_test.statistic:.3f}, p-value={norm_test.pvalue:.4f}")
    
    het_test = res_diag.heteroscedasticity_test()
    print(f"  Heteroscedasticity: stat={het_test.statistic:.3f}, p-value={het_test.pvalue:.4f}")
    
    runs_test = res_diag.runs_test()
    print(f"  Runs test: stat={runs_test.statistic:.3f}, p-value={runs_test.pvalue:.4f}")
    
    # 3. Out-of-sample test
    print("\n3. Out-of-Sample Test:")
    oos_test = OutOfSampleTest(n_forecasts=12)
    oos_results = oos_test.test(ts, result.specification, TramoSeatsProcessor())
    
    print(f"  RMSE: {oos_results['rmse']:.3f}")
    print(f"  MAE: {oos_results['mae']:.3f}")
    print(f"  MAPE: {oos_results['mape']:.2f}%")
    
    # 4. Sliding spans analysis
    print("\n4. Sliding Spans Analysis:")
    spans_analysis = SlidingSpansAnalysis(n_spans=4, span_length=84)
    spans_results = spans_analysis.analyze(ts, result.specification, TramoSeatsProcessor())
    
    print(f"  SA stability: {spans_results['sa_stability']:.1f}%")
    print(f"  Trend stability: {spans_results['trend_stability']:.1f}%")
    print(f"  Seasonal stability: {spans_results['seasonal_stability']:.1f}%")

def plot_advanced_diagnostics(ts: TsData, results: Dict):
    """Create advanced diagnostic plots."""
    fig = plt.figure(figsize=(16, 12))
    
    # Get automatic results
    result = results['automatic']
    decomp = result.decomposition
    residuals = result.preprocessing.residuals
    
    # 1. Spectral analysis
    ax1 = plt.subplot(3, 3, 1)
    plotter = DiagnosticPlotter()
    plotter._plot_spectrum_subplot(decomp.seasonally_adjusted.values, ax1)
    ax1.set_title('SA Series Spectrum')
    
    # 2. Residual ACF/PACF
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    ax2 = plt.subplot(3, 3, 2)
    plot_acf(residuals, lags=36, ax=ax2)
    ax2.set_title('Residual ACF')
    
    ax3 = plt.subplot(3, 3, 3)
    plot_pacf(residuals, lags=36, ax=ax3)
    ax3.set_title('Residual PACF')
    
    # 3. Outlier effects
    ax4 = plt.subplot(3, 3, 4)
    dates = pd.date_range('2014-01', periods=ts.length(), freq='M')
    
    outlier_effects = np.zeros(ts.length())
    if result.preprocessing and result.preprocessing.outliers:
        for outlier in result.preprocessing.outliers:
            # Simple visualization of outlier positions
            outlier_effects[outlier.position] = outlier.coefficient
    
    ax4.stem(dates, outlier_effects, basefmt=' ')
    ax4.set_title('Detected Outliers')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Effect')
    ax4.grid(True, alpha=0.3)
    
    # 4. Component comparison across specifications
    ax5 = plt.subplot(3, 3, 5)
    for name, result in list(results.items())[:3]:  # First 3 specs
        if result:
            ax5.plot(dates, result.decomposition.seasonally_adjusted.values, 
                    label=name, alpha=0.7)
    ax5.set_title('SA Series Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 5. Seasonal patterns comparison
    ax6 = plt.subplot(3, 3, 6)
    for name, result in list(results.items())[:3]:
        if result:
            seasonal = result.decomposition.seasonal.values
            ax6.plot(seasonal[:24], 'o-', label=name, alpha=0.7)
    ax6.set_title('Seasonal Pattern (First 2 Years)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 6. Quality measures comparison
    ax7 = plt.subplot(3, 3, 7)
    spec_names = []
    q_stats = []
    for name, result in results.items():
        if result:
            quality = QualityMeasures(result.decomposition)
            spec_names.append(name.replace('custom_', ''))
            q_stats.append(quality.q())
    
    bars = ax7.bar(spec_names, q_stats)
    ax7.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax7.set_title('Q-Statistics by Specification')
    ax7.set_ylabel('Q-Stat')
    ax7.set_xticklabels(spec_names, rotation=45, ha='right')
    
    # Color bars based on quality
    for bar, q in zip(bars, q_stats):
        if q < 1.0:
            bar.set_color('green')
        elif q < 1.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # 7. Residual Q-Q plot
    ax8 = plt.subplot(3, 3, 8)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax8)
    ax8.set_title('Residual Q-Q Plot')
    
    # 8. Time-varying seasonal amplitude
    ax9 = plt.subplot(3, 3, 9)
    seasonal = decomp.seasonal.values
    
    # Calculate rolling seasonal amplitude
    window = 24  # 2 years
    seasonal_amp = pd.Series(seasonal).rolling(window).apply(
        lambda x: np.ptp(x[:12]) if len(x) >= 12 else np.nan
    )
    
    ax9.plot(dates, seasonal_amp)
    ax9.set_title('Time-Varying Seasonal Amplitude')
    ax9.set_xlabel('Date')
    ax9.set_ylabel('Amplitude')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_diagnostics.png', dpi=300)
    print("\nSaved: advanced_diagnostics.png")

def main():
    """Run advanced features example."""
    print("Advanced Features Example")
    print("=" * 50)
    
    # Create complex series
    ts = create_complex_series()
    print(f"\nCreated complex series:")
    print(f"  Length: {ts.length()} observations")
    print(f"  Period: {ts.start()} to {ts.end()}")
    
    # Test different specifications
    results = custom_specification_example(ts)
    
    # Compare results
    print("\nSpecification Comparison:")
    comparison_df = compare_specifications(results)
    print(comparison_df)
    
    # Save comparison
    comparison_df.to_csv('specification_comparison.csv', index=False)
    print("\nSaved: specification_comparison.csv")
    
    # Advanced diagnostics
    advanced_diagnostics(ts, results)
    
    # Create diagnostic plots
    plot_advanced_diagnostics(ts, results)
    
    # Additional analysis: Model selection
    print("\n\nModel Selection Analysis")
    print("=" * 50)
    
    # Test different ARIMA orders
    orders_to_test = [
        SarimaOrder(1, 1, 1, 0, 1, 1, 12),
        SarimaOrder(0, 1, 1, 0, 1, 1, 12),
        SarimaOrder(2, 1, 0, 1, 1, 0, 12),
        SarimaOrder(1, 1, 1, 1, 1, 1, 12),
        SarimaOrder(2, 1, 2, 1, 1, 1, 12)
    ]
    
    estimator = ArimaEstimator()
    model_results = []
    
    for order in orders_to_test:
        try:
            model = estimator.estimate(ts, order)
            model_results.append({
                'Order': order.spec_str(),
                'Log-likelihood': round(model.log_likelihood, 2),
                'AIC': round(model.aic, 2),
                'BIC': round(model.bic, 2),
                'Stationary': model.is_stationary(),
                'Invertible': model.is_invertible()
            })
        except Exception as e:
            print(f"  Failed to estimate {order.spec_str()}: {e}")
    
    model_df = pd.DataFrame(model_results)
    print("\nARIMA Model Comparison:")
    print(model_df)
    
    # Find best model by AIC
    if model_results:
        best_idx = model_df['AIC'].idxmin()
        print(f"\nBest model by AIC: {model_df.loc[best_idx, 'Order']}")
    
    print("\nAdvanced features example completed!")

if __name__ == '__main__':
    main()