"""Unit tests for ARIMA models."""

import pytest
import numpy as np

from jdemetra_py.toolkit.arima import (
    ArimaModel, SarimaModel, ArimaOrder, SarimaOrder,
    ArimaEstimator, EstimationMethod,
    ArimaForecaster
)
from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsUnit


class TestArimaModel:
    """Tests for ARIMA model."""
    
    def test_order_creation(self):
        order = ArimaOrder(1, 1, 1)
        assert order.p == 1
        assert order.d == 1
        assert order.q == 1
        assert str(order) == "ARIMA(1,1,1)"
        
        # Invalid orders
        with pytest.raises(ValueError):
            ArimaOrder(-1, 0, 0)
    
    def test_model_creation(self):
        order = ArimaOrder(2, 0, 1)
        ar_params = np.array([0.5, -0.3])
        ma_params = np.array([0.7])
        
        model = ArimaModel(order, ar_params, ma_params, variance=2.0)
        
        assert model.order == order
        np.testing.assert_array_equal(model.ar_params, ar_params)
        np.testing.assert_array_equal(model.ma_params, ma_params)
        assert model.variance == 2.0
    
    def test_polynomials(self):
        order = ArimaOrder(2, 0, 1)
        model = ArimaModel(order, [0.5, -0.3], [0.7])
        
        # AR polynomial: 1 - 0.5L + 0.3L^2
        ar_poly = model.ar_polynomial
        assert ar_poly.degree == 2
        np.testing.assert_array_equal(ar_poly.coefficients, [1, -0.5, 0.3])
        
        # MA polynomial: 1 + 0.7L
        ma_poly = model.ma_polynomial
        assert ma_poly.degree == 1
        np.testing.assert_array_equal(ma_poly.coefficients, [1, 0.7])
    
    def test_stationarity(self):
        # Stationary model
        order = ArimaOrder(1, 0, 0)
        model1 = ArimaModel(order, [0.5])
        assert model1.is_stationary
        
        # Non-stationary model (unit root)
        model2 = ArimaModel(order, [1.0])
        assert not model2.is_stationary
    
    def test_simulation(self):
        order = ArimaOrder(1, 0, 1)
        model = ArimaModel(order, [0.7], [0.3], variance=1.0)
        
        # Simulate series
        sim = model.simulate(100, random_state=42)
        
        assert len(sim) == 100
        assert np.isfinite(sim).all()


class TestArimaEstimation:
    """Tests for ARIMA estimation."""
    
    def test_basic_estimation(self):
        # Generate simple AR(1) data
        np.random.seed(42)
        n = 200
        ar_param = 0.7
        
        y = np.zeros(n)
        y[0] = np.random.randn()
        for t in range(1, n):
            y[t] = ar_param * y[t-1] + np.random.randn()
        
        # Create TsData
        start = TsPeriod(2020, 0, TsUnit.MONTH)
        data = TsData.of(start, y)
        
        # Estimate AR(1) model
        estimator = ArimaEstimator()
        order = ArimaOrder(1, 0, 0)
        model = estimator.estimate(data, order)
        
        # Check parameter is close to true value
        assert abs(model.ar_params[0] - ar_param) < 0.1
        assert model.variance > 0
    
    def test_airline_data(self):
        # Create synthetic airline-like data
        np.random.seed(42)
        t = np.arange(144)
        trend = 100 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.randn(144) * 5
        airline_data = trend + seasonal + noise
        
        # Only use subset for speed
        data_subset = airline_data[:100]
        
        start = TsPeriod(1949, 0, TsUnit.MONTH)
        ts_data = TsData.of(start, data_subset)
        
        # Estimate ARIMA(0,1,1)
        estimator = ArimaEstimator()
        order = ArimaOrder(0, 1, 1)
        model = estimator.estimate(ts_data, order)
        
        assert model.order == order
        assert len(model.ma_params) == 1
        assert hasattr(model, '_aic')
        assert hasattr(model, '_bic')
    
    def test_information_criteria(self):
        # Generate data
        np.random.seed(42)
        y = np.cumsum(np.random.randn(100))  # Random walk
        
        start = TsPeriod(2020, 0, TsUnit.MONTH)
        data = TsData.of(start, y)
        
        # Estimate model with parameters (ARIMA(1,1,0))
        estimator = ArimaEstimator()
        order = ArimaOrder(1, 1, 0)
        model = estimator.estimate(data, order)
        
        # Compute IC
        ic = estimator.compute_information_criteria(model, data)
        
        assert 'aic' in ic
        assert 'bic' in ic
        assert 'aicc' in ic
        assert ic['aic'] < ic['bic']  # AIC < BIC for simple models


class TestSarimaModel:
    """Tests for seasonal ARIMA model."""
    
    def test_sarima_order(self):
        order = ArimaOrder(1, 1, 1)
        seasonal_order = ArimaOrder(1, 1, 1)
        
        sarima_order = SarimaOrder(order, seasonal_order, 12)
        
        assert sarima_order.period == 12
        assert str(sarima_order) == "SARIMA(1,1,1)(1,1,1)[12]"
    
    def test_sarima_model(self):
        order = SarimaOrder(
            ArimaOrder(1, 0, 1),
            ArimaOrder(1, 0, 1),
            4
        )
        
        model = SarimaModel(
            order,
            ar_params=[0.5],
            ma_params=[0.3],
            seasonal_ar_params=[0.7],
            seasonal_ma_params=[0.4]
        )
        
        assert model.period == 4
        assert model.seasonal_ar_params[0] == 0.7
        
        # Check seasonal polynomials
        sar_poly = model.seasonal_ar_polynomial
        assert sar_poly.degree == 4  # Coefficient at lag 4
        assert sar_poly.get(4) == -0.7


class TestArimaForecasting:
    """Tests for ARIMA forecasting."""
    
    def test_basic_forecast(self):
        # Create simple model
        order = ArimaOrder(1, 0, 0)
        model = ArimaModel(order, [0.5], variance=1.0, mean=10.0)
        
        # Generate data from model
        np.random.seed(42)
        data = model.simulate(100)
        start = TsPeriod(2020, 0, TsUnit.MONTH)
        ts_data = TsData.of(start, data)
        
        # Forecast
        forecaster = ArimaForecaster(model)
        forecast = forecaster.forecast(ts_data, steps=10)
        
        assert len(forecast.point_forecast) == 10
        assert forecast.lower_bound is not None
        assert forecast.upper_bound is not None
        assert np.all(forecast.lower_bound < forecast.upper_bound)
    
    def test_forecast_to_tsdata(self):
        # Simple model
        order = ArimaOrder(0, 1, 1)
        model = ArimaModel(order, ma_params=[0.3])
        
        # Data
        start = TsPeriod(2020, 0, TsUnit.QUARTER)
        data = TsData.of(start, np.random.randn(20))
        
        # Forecast as TsData
        forecaster = ArimaForecaster(model)
        forecast_ts = forecaster.forecast_to_tsdata(data, steps=4)
        
        assert forecast_ts.length == 4
        assert forecast_ts.start.year == 2025
        assert forecast_ts.start.period == 0
    
    def test_simulation(self):
        # AR(1) model
        order = ArimaOrder(1, 0, 0)
        model = ArimaModel(order, [0.7], variance=1.0)
        
        # Historical data
        start = TsPeriod(2020, 0, TsUnit.MONTH)
        data = TsData.of(start, np.random.randn(50))
        
        # Simulate future paths
        forecaster = ArimaForecaster(model)
        sims = forecaster.simulate(steps=12, nsim=100, data=data, random_state=42)
        
        assert sims.shape == (100, 12)
        
        # Mean should converge to model mean
        sim_mean = np.mean(sims, axis=0)
        assert np.all(np.abs(sim_mean) < 2.0)  # Reasonable bounds