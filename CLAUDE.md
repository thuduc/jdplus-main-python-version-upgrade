# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Requirements

- Python 3.12 or higher is required
- All dependencies are specified in pyproject.toml

## Common Development Commands

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_sa.py

# Run with coverage
python -m pytest --cov=jdemetra_py tests/

# Run tests in parallel
python -m pytest -n auto tests/
```

### Code Quality
```bash
# Format code with Black
black .

# Check linting with flake8
flake8 .

# Type checking with mypy
mypy .
```

### Building and Installation
```bash
# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[all]"

# Build package
python -m build
```

## Architecture Overview

JDemetra+ Python is a time series analysis and seasonal adjustment framework. The codebase follows a modular architecture:

### Core Components

1. **`toolkit/`** - Core mathematical and statistical foundations
   - `timeseries/` - Time series data structures (TsData, TsPeriod, TsFrequency)
   - `arima/` - ARIMA model estimation and forecasting
   - `ssf/` - State space models and Kalman filtering
   - `math/` - Linear algebra and polynomial operations
   - `stats/` - Statistical tests and distributions
   - `calendars/` - Holiday and calendar effect management

2. **`sa/`** - Seasonal adjustment implementations
   - `base/` - Abstract base classes defining the SA framework
   - `tramoseats/` - TRAMO/SEATS method implementation
   - `x13/` - X-13ARIMA-SEATS wrapper
   - `diagnostics/` - Quality measures and residual diagnostics
   - `benchmarking/` - Benchmarking methods (Denton, Cholette)

3. **`io/`** - Data input/output functionality
   - Providers for CSV, Excel, XML, JSON formats
   - TsCollection for managing multiple time series

4. **`visualization/`** - Plotting and reporting tools
   - Decomposition plots, diagnostic charts, custom styles

5. **`workspace/`** - Project and workspace management
   - Persistence of SA specifications and results

6. **`optimization/`** - Performance enhancements
   - Caching decorators, Numba JIT compilation, vectorization utilities

### Key Design Patterns

- **Specification/Processor Pattern**: Seasonal adjustment methods use separate specification objects (configuration) and processor objects (execution)
- **Abstract Base Classes**: Core interfaces defined in `sa/base/` ensure consistency across different SA methods
- **Immutable Data Structures**: TsData and related classes are designed to be immutable for thread safety
- **Decorator-based Optimization**: Performance features use decorators to avoid cluttering business logic

### Testing Structure

Tests mirror the source structure under `tests/`:
- Unit tests for each module
- Integration tests for SA workflows
- `conftest.py` contains shared fixtures

### Entry Points

The main user-facing APIs are:
- `TramoSeatsProcessor` and `TramoSeatsSpecification` for TRAMO/SEATS
- `X13ArimaSeatsProcessor` and `X13Specification` for X-13
- `TsData` for time series manipulation
- `plot_decomposition()` and `plot_diagnostics()` for visualization