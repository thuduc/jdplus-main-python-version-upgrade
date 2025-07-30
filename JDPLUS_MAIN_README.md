# JDemetra+ Python

A Python implementation of the JDemetra+ seasonal adjustment and time series analysis framework.

## Overview

JDemetra+ Python provides a comprehensive toolkit for time series analysis and seasonal adjustment, compatible with the JDemetra+ ecosystem. It implements the TRAMO/SEATS and X-13ARIMA-SEATS methods along with extensive diagnostic and visualization tools.

## Features

- **Time Series Core**
  - Efficient time series data structures
  - Period and frequency handling
  - Missing value support
  - Comprehensive operations (lag, lead, differencing, etc.)

- **Seasonal Adjustment Methods**
  - TRAMO/SEATS implementation
  - X-13ARIMA-SEATS wrapper
  - Customizable specifications
  - Quality diagnostics

- **Statistical Models**
  - ARIMA/SARIMA estimation and forecasting
  - State space models and Kalman filtering
  - Regression with ARIMA errors

- **Utilities**
  - Calendar and holiday management
  - Data I/O (CSV, Excel, XML, JSON)
  - Workspace management
  - Visualization tools

## Installation

```bash
pip install jdemetra-py
```

### Dependencies

- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7
- statsmodels >= 0.12
- matplotlib >= 3.3
- numba >= 0.54 (optional, for performance)

## Quick Start

### Basic Time Series Operations

```python
import numpy as np
from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency

# Create a monthly time series
start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)  # January 2020
values = np.random.randn(60)  # 5 years of data
ts = TsData.of(start, values)

# Basic operations
ts_lag = ts.lag(1)
ts_diff = ts.delta(1)
ts_log = ts.fn(np.log)

# Statistics
print(f"Mean: {ts.average()}")
print(f"Std: {np.std(ts.values)}")
```

### Seasonal Adjustment

```python
from jdemetra_py.sa import TramoSeatsSpecification, TramoSeatsProcessor

# Create specification
spec = TramoSeatsSpecification.rsa5()  # Automatic with log transformation

# Process series
processor = TramoSeatsProcessor(spec)
results = processor.process(ts)

# Access components
sa = results.decomposition.seasonally_adjusted
trend = results.decomposition.trend
seasonal = results.decomposition.seasonal
irregular = results.decomposition.irregular

# View diagnostics
print(results.summary())
```

### Visualization

```python
from jdemetra_py.visualization import plot_decomposition, plot_diagnostics

# Plot decomposition
fig = plot_decomposition(results.decomposition)

# Plot diagnostics
fig = plot_diagnostics(results)
```

### Data I/O

```python
from jdemetra_py.io import CsvDataProvider, TsCollectionBuilder

# Read from CSV
provider = CsvDataProvider()
collection = provider.read("data.csv")

# Build collection programmatically
builder = TsCollectionBuilder()
builder.add_series("series1", values1, "2020-01")
builder.add_series("series2", values2, "2020-01")
collection = builder.build()

# Write to file
provider.write(collection, "output.csv")
```

## API Reference

### Core Modules

#### `toolkit.timeseries`
Core time series data structures and operations.

- `TsData`: Time series container
- `TsPeriod`: Time period representation
- `TsFrequency`: Frequency enumeration
- `TsDomain`: Time domain operations

#### `toolkit.arima`
ARIMA modeling and forecasting.

- `ArimaModel`: ARIMA model representation
- `ArimaEstimator`: Parameter estimation
- `ArimaForecaster`: Forecasting functionality

#### `toolkit.ssf`
State space framework.

- `StateSpaceModel`: SSM representation
- `KalmanFilter`: Filtering algorithm
- `KalmanSmoother`: Smoothing algorithm

### Seasonal Adjustment

#### `sa.base`
Base classes for seasonal adjustment.

- `SaSpecification`: Specification base class
- `SaProcessor`: Processor base class
- `SaResults`: Results container
- `SeriesDecomposition`: Decomposition storage

#### `sa.tramoseats`
TRAMO/SEATS implementation.

- `TramoSeatsSpecification`: Method specification
- `TramoSeatsProcessor`: Processing engine
- `TramoProcessor`: TRAMO pre-adjustment
- `SeatsDecomposer`: SEATS decomposition

#### `sa.x13`
X-13ARIMA-SEATS wrapper.

- `X13Specification`: Method specification
- `X13ArimaSeatsProcessor`: Processing wrapper

### Diagnostics

#### `sa.diagnostics`
Quality and diagnostic measures.

- `SeasonalityTests`: Tests for seasonality
- `QualityMeasures`: M-statistics and quality scores
- `ResidualsDiagnostics`: Residual analysis

### Utilities

#### `io`
Data input/output functionality.

- `CsvDataProvider`: CSV file I/O
- `ExcelDataProvider`: Excel file I/O
- `TsCollection`: Time series collection

#### `workspace`
Project management.

- `Workspace`: Container for SA projects
- `SAItem`: Seasonal adjustment item
- `WorkspacePersistence`: Save/load functionality

#### `visualization`
Plotting and visualization.

- `plot_series`: Time series plotting
- `plot_decomposition`: Component plots
- `plot_diagnostics`: Diagnostic plots

## Performance

JDemetra+ Python includes several optimization features:

- **Caching**: Automatic memoization of expensive operations
- **Vectorization**: NumPy-based array operations
- **Numba JIT**: Optional compilation for critical paths
- **Parallel Processing**: Multi-core support for batch operations

### Benchmarks

```python
from jdemetra_py.benchmarks import run_all_benchmarks

results = run_all_benchmarks()
```

## Examples

See the `examples/` directory for detailed examples:

- `basic_sa.py`: Basic seasonal adjustment workflow
- `advanced_features.py`: Advanced specifications and diagnostics
- `batch_processing.py`: Processing multiple series
- `custom_calendars.py`: Holiday and calendar effects
- `visualization_examples.py`: Plotting and reports

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Building Documentation

```bash
cd docs
make html
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the EUPL-1.2 license - see the LICENSE file for details.

## Acknowledgments

JDemetra+ Python is based on the JDemetra+ project developed by the National Bank of Belgium and Eurostat.

## Citation

If you use JDemetra+ Python in your research, please cite:

```bibtex
@software{jdemetra_python,
  title = {JDemetra+ Python: Time Series Analysis and Seasonal Adjustment},
  author = {JDemetra+ Team},
  year = {2024},
  url = {https://github.com/jdemetra/jdplus-python}
}
```