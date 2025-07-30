# Changelog

All notable changes to JDemetra+ Python will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-15

### Added

#### Core Infrastructure
- Time series data structures (`TsData`, `TsPeriod`, `TsFrequency`, `TsDomain`)
- Comprehensive time series operations (lag, lead, differencing, transformations)
- Automatic handling of missing values
- Support for multiple frequencies (yearly to daily)

#### ARIMA/SARIMA Models
- Complete ARIMA model implementation with estimation
- SARIMA support with seasonal components
- Maximum likelihood and CSS-ML estimation methods
- Model diagnostics and validation
- Forecasting capabilities

#### State Space Framework
- General state space model representation
- Kalman filter and smoother implementations
- Support for time-varying and time-invariant models
- Missing observation handling

#### Seasonal Adjustment Methods
- **TRAMO/SEATS**: Full implementation
  - Predefined specifications (RSA0-RSA5)
  - Custom specification support
  - TRAMO pre-adjustment (outliers, calendar effects)
  - SEATS decomposition
- **X-13ARIMA-SEATS**: Wrapper implementation
  - Integration with Census Bureau's X-13 executable
  - Compatible specification interface
  - Full diagnostic output

#### Diagnostics and Quality Measures
- Seasonality tests (QS, Friedman, Kruskal-Wallis)
- M-statistics (M1-M7) and Q-statistic
- Residual diagnostics (Ljung-Box, normality, heteroscedasticity)
- Out-of-sample testing
- Sliding spans analysis

#### Data I/O and Integration
- File format support: CSV, Excel, XML, JSON
- Pandas DataFrame integration
- Time series collection management
- Automatic format detection

#### Workspace Management
- Project-based organization
- Batch processing capabilities
- Persistence in multiple formats
- Metadata support

#### Calendar and Holiday Support
- National calendar definitions
- Fixed and moving holidays
- Easter calculation
- Trading day and working day effects
- Custom calendar creation

#### Visualization Tools
- Time series plotting with multiple series support
- Decomposition visualization
- Diagnostic plots (spectrum, ACF/PACF, residuals)
- Quality measure dashboards
- Customizable plotting options

#### Performance Optimization
- Numba JIT compilation for critical operations
- Vectorized operations using NumPy
- Parallel processing support
- Intelligent caching system
- Memory-efficient batch processing

#### Documentation
- Comprehensive API reference
- User guide with examples
- Code examples for common workflows
- Performance optimization guide

### Dependencies
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7
- statsmodels >= 0.12
- matplotlib >= 3.3
- seaborn >= 0.11
- numba >= 0.54 (optional)

### Known Limitations
- X-13ARIMA-SEATS requires external executable
- Some advanced SEATS options not yet exposed
- Limited support for daily/weekly frequencies in seasonal adjustment

## [Unreleased]

### Planned Features
- STL decomposition method
- Enhanced daily/weekly frequency support
- RESTful API interface
- Additional file format support (SDMX, TSV)
- GPU acceleration for large-scale processing
- Interactive visualization dashboard
- Extended regression variables
- Structural time series models

### Under Consideration
- Integration with other statistical packages
- Cloud deployment capabilities
- Real-time processing support
- Machine learning integration for outlier detection