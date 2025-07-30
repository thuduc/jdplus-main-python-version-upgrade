# JDemetra+ Python User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Working with Time Series](#working-with-time-series)
5. [Seasonal Adjustment](#seasonal-adjustment)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

JDemetra+ Python is a comprehensive time series analysis and seasonal adjustment framework, providing Python users with the powerful capabilities of the JDemetra+ ecosystem. This guide will walk you through the main features and common workflows.

### Key Concepts

- **Time Series**: Ordered sequences of observations indexed by time
- **Seasonal Adjustment**: Process of removing seasonal patterns from time series
- **TRAMO/SEATS**: Time series Regression with ARIMA noise, Missing values and Outliers / Signal Extraction in ARIMA Time Series
- **X-13ARIMA-SEATS**: US Census Bureau's seasonal adjustment program

## Installation

### Standard Installation

```bash
pip install jdemetra-py
```

### Development Installation

```bash
git clone https://github.com/jdemetra/jdplus-python.git
cd jdplus-python
pip install -e .
```

### Optional Dependencies

For enhanced performance:
```bash
pip install jdemetra-py[performance]
```

For development:
```bash
pip install jdemetra-py[dev]
```

## Quick Start

### Your First Seasonal Adjustment

```python
import numpy as np
from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.sa.tramoseats import TramoSeatsSpecification, TramoSeatsProcessor

# Create sample monthly data
start = TsPeriod.of(TsFrequency.MONTHLY, 2019, 0)  # January 2019
values = 100 + 10 * np.sin(np.arange(60) * 2 * np.pi / 12) + np.random.randn(60)
ts = TsData.of(start, values)

# Perform seasonal adjustment
spec = TramoSeatsSpecification.rsa5()  # Automatic specification
processor = TramoSeatsProcessor(spec)
results = processor.process(ts)

# Access results
sa_series = results.decomposition.seasonally_adjusted
print(f"Original mean: {ts.average():.2f}")
print(f"SA mean: {sa_series.average():.2f}")

# Plot results
from jdemetra_py.visualization import plot_decomposition
fig = plot_decomposition(results.decomposition)
```

## Working with Time Series

### Creating Time Series

#### From Arrays
```python
# Monthly data starting January 2020
start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
values = np.array([100, 102, 98, 105, 110, 108])
ts = TsData.of(start, values)
```

#### From Pandas
```python
import pandas as pd

# Create from DataFrame
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=24, freq='M'),
    'value': np.random.randn(24) * 10 + 100
})

# Convert to TsData
start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
ts = TsData.of(start, df['value'].values)
```

#### From Files
```python
from jdemetra_py.io import CsvDataProvider

# Read from CSV
provider = CsvDataProvider()
collection = provider.read('data.csv', date_column='Date', value_columns=['Sales'])
ts = collection.get('Sales')
```

### Time Series Operations

#### Basic Operations
```python
# Lag/Lead
ts_lag1 = ts.lag(1)  # Lag by 1 period
ts_lead1 = ts.lead(1)  # Lead by 1 period

# Differencing
ts_diff = ts.delta(1)  # First difference
ts_diff12 = ts.delta(12)  # Seasonal difference (monthly)

# Transformations
ts_log = ts.log()  # Natural logarithm
ts_exp = ts.exp()  # Exponential
ts_sqrt = ts.fn(np.sqrt)  # Square root
```

#### Window Operations
```python
# Extract a window
start_window = TsPeriod.of(TsFrequency.MONTHLY, 2020, 6)  # July 2020
end_window = TsPeriod.of(TsFrequency.MONTHLY, 2021, 5)  # June 2021
ts_window = ts.window(start_window, end_window)

# Extend series
ts_extended = ts.extend(12, 12)  # Add 12 NaN before and after

# Clean extremities
ts_clean = ts.clean_extremities()  # Remove leading/trailing NaN
```

### Working with Collections

```python
from jdemetra_py.io import TsCollection, TsCollectionBuilder

# Create collection manually
collection = TsCollection("Economic Indicators")
collection.add("GDP", gdp_ts)
collection.add("CPI", cpi_ts)
collection.add("Unemployment", unemployment_ts)

# Or use builder
builder = TsCollectionBuilder()
builder.add_series("GDP", gdp_values, "2020-01")
builder.add_series("CPI", cpi_values, "2020-01")
collection = builder.build()

# Access series
gdp = collection.get("GDP")
all_names = collection.names()
```

## Seasonal Adjustment

### TRAMO/SEATS Method

#### Predefined Specifications

```python
from jdemetra_py.sa.tramoseats import TramoSeatsSpecification

# RSA0: No transformation, no outliers
spec0 = TramoSeatsSpecification.rsa0()

# RSA1: Log transformation, outlier detection
spec1 = TramoSeatsSpecification.rsa1()

# RSA3: Working days adjustment
spec3 = TramoSeatsSpecification.rsa3()

# RSA5: Full automatic (recommended)
spec5 = TramoSeatsSpecification.rsa5()
```

#### Custom Specifications

```python
# Start with a base specification
spec = TramoSeatsSpecification.rsa5()

# Customize transformation
spec.set_transform('log')  # 'log', 'none', or 'auto'

# Set outlier types
spec.set_outliers(['AO', 'LS', 'TC'])  # Additive, Level Shift, Transitory Change

# Calendar effects
spec.set_trading_days(True)
spec.set_easter(True)

# ARIMA model
spec.set_arima_specification(dict(p=1, d=1, q=1, P=1, D=1, Q=1))
```

#### Processing and Results

```python
# Process series
processor = TramoSeatsProcessor(spec)
results = processor.process(ts)

# Access components
original = results.decomposition.original
sa = results.decomposition.seasonally_adjusted
trend = results.decomposition.trend
seasonal = results.decomposition.seasonal
irregular = results.decomposition.irregular

# Calendar effects (if estimated)
if results.decomposition.calendar is not None:
    calendar = results.decomposition.calendar

# Get preprocessing results
preprocessing = results.preprocessing
if preprocessing:
    # Detected outliers
    outliers = preprocessing.outliers
    # ARIMA model
    arima_model = preprocessing.arima_model
```

### X-13ARIMA-SEATS Method

```python
from jdemetra_py.sa.x13 import X13Specification, X13ArimaSeatsProcessor

# Create specification
spec = X13Specification.rsa5c()

# Process (requires X-13 executable)
processor = X13ArimaSeatsProcessor()
results = processor.process(ts, spec)

# Results have same structure as TRAMO/SEATS
sa = results.decomposition.seasonally_adjusted
```

### Quality Assessment

#### Diagnostics

```python
from jdemetra_py.sa.diagnostics import SeasonalityTests, QualityMeasures

# Test for residual seasonality
seasonality_test = SeasonalityTests.combined_test(
    results.decomposition.seasonally_adjusted, 
    ts.frequency()
)
print(f"Residual seasonality p-value: {seasonality_test.pvalue:.4f}")

# M-statistics
quality = QualityMeasures(results.decomposition)
print(f"M1 (irregular contribution): {quality.m1():.3f}")
print(f"M7 (moving seasonality): {quality.m7():.3f}")
print(f"Overall Q: {quality.q():.3f}")

# Residual diagnostics
from jdemetra_py.sa.diagnostics import ResidualsDiagnostics
residuals = results.preprocessing.residuals
diag = ResidualsDiagnostics(residuals)

# Ljung-Box test
lb_test = diag.ljung_box_test(24)
print(f"Ljung-Box p-value: {lb_test.pvalue:.4f}")
```

#### Visual Diagnostics

```python
from jdemetra_py.visualization import plot_diagnostics, DiagnosticPlotter

# Comprehensive diagnostic plots
fig = plot_diagnostics(results)

# Or use the plotter class for more control
plotter = DiagnosticPlotter()
fig = plotter.plot_residuals(results.preprocessing.residuals)
fig = plotter.plot_spectrum(results.decomposition.seasonally_adjusted)
fig = plotter.plot_sliding_spans(results.decomposition)
```

## Advanced Features

### Batch Processing

```python
from jdemetra_py.workspace import Workspace, SAItem

# Create workspace
workspace = Workspace("Quarterly Analysis")

# Add multiple series
for name, series in [("GDP", gdp_ts), ("Investment", inv_ts), ("Consumption", cons_ts)]:
    item = SAItem(name, series, TramoSeatsSpecification.rsa5())
    workspace.add_sa_item(item)

# Process all
processor = TramoSeatsProcessor()
results = workspace.process_all(processor)

# Check results
for item_id, success in results.items():
    if success:
        item = workspace.get_sa_item(item_id)
        sa = item.results.decomposition.seasonally_adjusted
        print(f"{item.name}: SA series has {sa.length()} observations")
```

### Calendar Management

```python
from jdemetra_py.toolkit.calendars import NationalCalendar, FixedWeekDayHoliday, Easter
from datetime import date

# Create custom calendar
calendar = NationalCalendar(weekend_definition="SaturdaySunday")

# Add fixed holidays
calendar.add_holiday(FixedWeekDayHoliday("New Year", 1, 1))
calendar.add_holiday(FixedWeekDayHoliday("Christmas", 12, 25))

# Add moving holidays
calendar.add_holiday(Easter(offset=-2))  # Good Friday

# Check dates
is_holiday = calendar.is_holiday(date(2024, 12, 25))
is_weekend = calendar.is_weekend(date(2024, 1, 6))  # Saturday

# Generate calendar regressors
from jdemetra_py.toolkit.calendars import CalendarUtilities
td_regressors = CalendarUtilities.trading_days_regressors(
    start_period, end_period, calendar
)
```

### ARIMA Modeling

```python
from jdemetra_py.toolkit.arima import ArimaEstimator, SarimaOrder

# Specify model: SARIMA(1,1,1)(1,1,1)12
order = SarimaOrder(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)

# Estimate
estimator = ArimaEstimator(method='css-ml')
model = estimator.estimate(ts, order)

# Check diagnostics
print(f"Log-likelihood: {model.log_likelihood:.2f}")
print(f"AIC: {model.aic:.2f}")
print(f"Is stationary: {model.is_stationary()}")
print(f"Is invertible: {model.is_invertible()}")

# Forecast
forecasts = model.forecast(ts.values, n_ahead=12)

# Plot with forecast
from jdemetra_py.visualization import TimeSeriesPlotter
plotter = TimeSeriesPlotter()
fig = plotter.plot_with_forecast(ts, forecasts, n_history=36)
```

### Performance Optimization

#### Enable Numba JIT Compilation

```python
from jdemetra_py.optimization import enable_numba, benchmark_numba_performance

# Enable globally
enable_numba()

# Check performance improvement
speedup = benchmark_numba_performance()
print(f"Numba speedup: {speedup:.1f}x")
```

#### Parallel Processing

```python
from jdemetra_py.optimization import parallel_map, VectorizedTsOperations

# Process multiple series in parallel
def process_series(ts):
    spec = TramoSeatsSpecification.rsa5()
    processor = TramoSeatsProcessor(spec)
    return processor.process(ts)

# Parallel execution
series_list = [ts1, ts2, ts3, ts4, ts5]
results = parallel_map(process_series, series_list, n_workers=4)

# Vectorized operations
ops = VectorizedTsOperations()
smoothed_list = ops.batch_transform(
    series_list, 
    lambda ts: ts.fn(lambda x: pd.Series(x).rolling(12).mean().values)
)
```

#### Caching

```python
from jdemetra_py.optimization import memoize, CacheManager

# Cache expensive computations
@memoize(maxsize=100)
def expensive_seasonal_test(series_id, data):
    # Complex seasonal testing
    return SeasonalityTests.combined_test(data, 12)

# Clear caches when needed
CacheManager.clear_all_caches()

# Get cache statistics
stats = CacheManager.get_cache_info()
print(f"Cache sizes: {stats}")
```

## Best Practices

### 1. Data Preparation

- **Check data quality**: Ensure no gaps in time series
- **Handle missing values**: Use interpolation or let TRAMO handle them
- **Consider transformations**: Log transform for multiplicative series

```python
# Data quality check
def check_data_quality(ts):
    issues = []
    
    # Check for gaps
    if not ts.isRegular():
        issues.append("Irregular time series")
    
    # Check for missing values
    n_missing = np.isnan(ts.values).sum()
    if n_missing > 0:
        issues.append(f"{n_missing} missing values")
    
    # Check for outliers
    z_scores = np.abs((ts.values - np.nanmean(ts.values)) / np.nanstd(ts.values))
    n_outliers = np.sum(z_scores > 4)
    if n_outliers > 0:
        issues.append(f"{n_outliers} potential outliers")
    
    return issues
```

### 2. Specification Selection

- **Start with automatic specifications** (RSA5/RSA5c)
- **Validate results** before using custom specifications
- **Document choices** for reproducibility

```python
# Specification comparison
def compare_specifications(ts):
    specs = {
        'RSA1': TramoSeatsSpecification.rsa1(),
        'RSA3': TramoSeatsSpecification.rsa3(),
        'RSA5': TramoSeatsSpecification.rsa5()
    }
    
    processor = TramoSeatsProcessor()
    results = {}
    
    for name, spec in specs.items():
        try:
            result = processor.process(ts, spec)
            quality = QualityMeasures(result.decomposition)
            results[name] = {
                'q_stat': quality.q(),
                'residual_seasonality': SeasonalityTests.qs_test(
                    result.decomposition.seasonally_adjusted, 
                    ts.frequency()
                ).pvalue
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results
```

### 3. Production Workflows

```python
# Production-ready SA workflow
class ProductionSAWorkflow:
    def __init__(self, spec=None):
        self.spec = spec or TramoSeatsSpecification.rsa5()
        self.processor = TramoSeatsProcessor()
        self.results_cache = {}
    
    def process_with_validation(self, ts, series_name):
        # Pre-processing checks
        issues = check_data_quality(ts)
        if issues:
            print(f"Warning for {series_name}: {', '.join(issues)}")
        
        # Process
        try:
            results = self.processor.process(ts, self.spec)
        except Exception as e:
            print(f"Processing failed for {series_name}: {e}")
            return None
        
        # Validate results
        sa = results.decomposition.seasonally_adjusted
        
        # Check for residual seasonality
        seas_test = SeasonalityTests.combined_test(sa, ts.frequency())
        if seas_test.pvalue < 0.01:
            print(f"Warning: Residual seasonality in {series_name}")
        
        # Check quality
        quality = QualityMeasures(results.decomposition)
        if quality.q() > 1.0:
            print(f"Warning: Low quality adjustment for {series_name}")
        
        # Cache results
        self.results_cache[series_name] = results
        
        return results
    
    def generate_report(self, output_dir):
        # Generate reports for all processed series
        for name, results in self.results_cache.items():
            # Save plots
            fig = plot_decomposition(results.decomposition)
            fig.savefig(f"{output_dir}/{name}_decomposition.png")
            
            # Save diagnostics
            with open(f"{output_dir}/{name}_diagnostics.txt", 'w') as f:
                f.write(results.summary())
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Issue: ModuleNotFoundError
# Solution: Ensure proper installation
pip install --upgrade jdemetra-py

# For development
pip install -e .[dev]
```

#### 2. Processing Failures
```python
# Issue: ProcessingError during SA
# Common causes and solutions:

# Too short series
if ts.length() < 36:  # Less than 3 years monthly
    print("Series too short for seasonal adjustment")

# All missing values
if np.all(np.isnan(ts.values)):
    print("No valid data points")

# Constant series
if np.std(ts.values) < 1e-10:
    print("Series has no variation")
```

#### 3. Performance Issues
```python
# Enable optimizations
from jdemetra_py.optimization import enable_numba
enable_numba()

# Use appropriate data structures
# Bad: Processing one by one
for ts in huge_list:
    results.append(processor.process(ts))

# Good: Batch processing
results = parallel_map(processor.process, huge_list, n_workers=4)
```

#### 4. Memory Issues
```python
# Clear caches periodically
from jdemetra_py.optimization import CacheManager
CacheManager.clear_all_caches()

# Process in chunks
def process_large_dataset(series_list, chunk_size=100):
    all_results = []
    for i in range(0, len(series_list), chunk_size):
        chunk = series_list[i:i + chunk_size]
        results = parallel_map(processor.process, chunk)
        all_results.extend(results)
        # Clear caches between chunks
        CacheManager.clear_all_caches()
    return all_results
```

### Getting Help

1. **Check the API Reference**: Detailed documentation of all classes and methods
2. **Run the examples**: Located in the `examples/` directory
3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. **Report issues**: GitHub issues page with minimal reproducible example

### Debug Utilities

```python
# Enable detailed processing logs
class DebugProcessor(TramoSeatsProcessor):
    def process(self, series, spec=None):
        print(f"Processing series of length {series.length()}")
        print(f"Frequency: {series.frequency()}")
        print(f"Start: {series.start()}")
        
        try:
            results = super().process(series, spec)
            print("Processing successful")
            
            # Print component statistics
            dec = results.decomposition
            print(f"SA mean: {dec.seasonally_adjusted.average():.2f}")
            print(f"Seasonal range: {np.ptp(dec.seasonal.values):.2f}")
            
            return results
        except Exception as e:
            print(f"Processing failed: {e}")
            raise
```

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed documentation
- Check out example scripts in the `examples/` directory
- Read about [JDemetra+ methodology](https://jdemetradocumentation.github.io/)
- Join the community discussions