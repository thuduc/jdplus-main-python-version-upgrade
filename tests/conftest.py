"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path

from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency


@pytest.fixture
def sample_monthly_series():
    """Generate sample monthly time series."""
    start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
    
    # Generate 5 years of data
    n = 60
    t = np.arange(n)
    
    # Trend + seasonal + irregular
    trend = 100 + 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    irregular = np.random.normal(0, 2, n)
    
    values = trend + seasonal + irregular
    
    return TsData.of(start, values)


@pytest.fixture
def sample_quarterly_series():
    """Generate sample quarterly time series."""
    start = TsPeriod.of(TsFrequency.QUARTERLY, 2020, 0)
    
    # Generate 5 years of data
    n = 20
    t = np.arange(n)
    
    # Trend + seasonal + irregular
    trend = 100 + 1.0 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 4)
    irregular = np.random.normal(0, 1.5, n)
    
    values = trend + seasonal + irregular
    
    return TsData.of(start, values)


@pytest.fixture
def sample_decomposition(sample_monthly_series):
    """Generate sample decomposition."""
    from jdemetra_py.sa.base import SeriesDecomposition, DecompositionMode
    
    decomp = SeriesDecomposition(mode=DecompositionMode.ADDITIVE)
    
    # Extract approximate components
    series = sample_monthly_series
    n = series.length
    t = np.arange(n)
    
    # Approximate trend (moving average)
    from scipy.ndimage import uniform_filter1d
    trend_values = uniform_filter1d(series.values, 13, mode='nearest')
    
    # Approximate seasonal (detrended and averaged by month)
    detrended = series.values - trend_values
    seasonal_values = np.zeros(n)
    for month in range(12):
        month_values = detrended[month::12]
        seasonal_values[month::12] = np.mean(month_values)
    
    # Irregular
    irregular_values = series.values - trend_values - seasonal_values
    
    # Set components
    decomp.series = series
    decomp.trend = TsData.of(series.start, trend_values)
    decomp.seasonal = TsData.of(series.start, seasonal_values)
    decomp.irregular = TsData.of(series.start, irregular_values)
    decomp.seasonally_adjusted = TsData.of(
        series.start, 
        trend_values + irregular_values
    )
    
    return decomp


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_monthly_series):
    """Create sample CSV file."""
    import pandas as pd
    
    # Convert to pandas
    dates = pd.date_range('2020-01-01', periods=sample_monthly_series.length, freq='MS')
    df = pd.DataFrame({
        'date': dates,
        'value': sample_monthly_series.values
    })
    
    # Save to CSV
    csv_path = temp_data_dir / "sample.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def sample_workspace():
    """Create sample workspace."""
    from jdemetra_py.workspace import Workspace
    
    ws = Workspace("Test Workspace")
    
    # Add some items
    from jdemetra_py.workspace import SAItem
    from jdemetra_py.sa import TramoSeatsSpecification
    
    # Create SA item
    start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
    series = TsData.of(start, np.random.randn(60))
    
    item = SAItem(
        name="Test Series",
        series=series,
        specification=TramoSeatsSpecification.rsa1(),
        method="tramoseats"
    )
    
    ws.add_item(item)
    
    return ws


# Test data generators

def generate_airline_series(n: int = 144, seed: int = 42) -> TsData:
    """Generate series following airline model."""
    np.random.seed(seed)
    
    # ARIMA (0,1,1)(0,1,1)12
    # (1-B)(1-B^12) y_t = (1-θB)(1-ΘB^12) ε_t
    
    theta = 0.3  # MA(1) parameter
    Theta = 0.5  # Seasonal MA(1) parameter
    
    # Generate innovations
    epsilon = np.random.normal(0, 1, n + 24)
    
    # Apply MA filters
    ma_series = np.zeros(n + 24)
    for t in range(n + 24):
        ma_series[t] = epsilon[t]
        if t >= 1:
            ma_series[t] -= theta * epsilon[t-1]
        if t >= 12:
            ma_series[t] -= Theta * epsilon[t-12]
        if t >= 13:
            ma_series[t] += theta * Theta * epsilon[t-13]
    
    # Apply differences
    y = np.zeros(n + 24)
    for t in range(13, n + 24):
        y[t] = y[t-1] + y[t-12] - y[t-13] + ma_series[t]
    
    # Add level
    y = y[24:] + 100
    
    start = TsPeriod.of(TsFrequency.MONTHLY, 2015, 0)
    return TsData.of(start, y)


def generate_seasonal_series(n: int = 60, 
                           trend_type: str = "linear",
                           seasonal_type: str = "stable",
                           seed: int = 42) -> TsData:
    """Generate series with specified characteristics."""
    np.random.seed(seed)
    
    t = np.arange(n)
    
    # Trend component
    if trend_type == "linear":
        trend = 100 + 0.5 * t
    elif trend_type == "quadratic":
        trend = 100 + 0.5 * t + 0.01 * t**2
    elif trend_type == "stochastic":
        trend = 100 + np.cumsum(np.random.normal(0.5, 0.2, n))
    else:
        trend = np.ones(n) * 100
    
    # Seasonal component
    if seasonal_type == "stable":
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
    elif seasonal_type == "evolving":
        amplitude = 10 + 0.1 * t
        seasonal = amplitude * np.sin(2 * np.pi * t / 12)
    elif seasonal_type == "changing":
        seasonal = np.zeros(n)
        for i in range(n):
            month = i % 12
            year = i // 12
            seasonal[i] = (5 + 0.5 * year) * np.sin(2 * np.pi * month / 12)
    else:
        seasonal = np.zeros(n)
    
    # Irregular component
    irregular = np.random.normal(0, 2, n)
    
    # Combine
    series = trend + seasonal + irregular
    
    start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
    return TsData.of(start, series)