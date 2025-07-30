"""ARIMA forecasting utilities."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .models import ArimaModel, SarimaModel
from ..timeseries.data import TsData, TsPeriod


@dataclass
class Forecast:
    """Container for forecast results."""
    point_forecast: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    se: Optional[np.ndarray] = None
    confidence_level: float = 0.95
    
    @property
    def prediction_intervals(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get prediction intervals as tuple."""
        if self.lower_bound is not None and self.upper_bound is not None:
            return (self.lower_bound, self.upper_bound)
        return None


class ArimaForecaster:
    """ARIMA forecasting engine."""
    
    def __init__(self, model: ArimaModel):
        """Initialize forecaster with fitted model.
        
        Args:
            model: Fitted ARIMA or SARIMA model
        """
        self.model = model
    
    def forecast(self, data: TsData, steps: int,
                confidence_level: float = 0.95) -> Forecast:
        """Generate forecasts from fitted model.
        
        Args:
            data: Historical data used for fitting
            steps: Number of steps ahead to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Forecast object with point forecasts and intervals
        """
        # Convert to pandas for statsmodels
        series = data.to_pandas()
        
        # Recreate statsmodels model with same specification
        if isinstance(self.model, SarimaModel):
            sm_model = SARIMAX(
                series,
                order=(self.model.order.p, self.model.order.d, self.model.order.q),
                seasonal_order=(self.model.seasonal_order.p, 
                              self.model.seasonal_order.d,
                              self.model.seasonal_order.q, 
                              self.model.period)
            )
        else:
            sm_model = ARIMA(
                series,
                order=(self.model.order.p, self.model.order.d, self.model.order.q)
            )
        
        # Create parameter vector for statsmodels
        params = self._create_params_vector()
        
        # Apply parameters and generate forecast
        sm_result = sm_model.filter(params)
        forecast_result = sm_result.get_forecast(steps=steps)
        
        # Extract results
        point_forecast = forecast_result.predicted_mean.values
        
        # Get prediction intervals
        pred_int = forecast_result.conf_int(alpha=1 - confidence_level)
        lower_bound = pred_int.iloc[:, 0].values
        upper_bound = pred_int.iloc[:, 1].values
        
        # Standard errors
        se = forecast_result.se_mean.values
        
        return Forecast(
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            se=se,
            confidence_level=confidence_level
        )
    
    def forecast_to_tsdata(self, data: TsData, steps: int) -> TsData:
        """Generate forecasts and return as TsData.
        
        Args:
            data: Historical data
            steps: Forecast horizon
            
        Returns:
            TsData with forecasted values
        """
        forecast = self.forecast(data, steps)
        
        # Get start period for forecasts
        last_period = data.domain.end
        forecast_start = last_period.next()
        
        return TsData.of(forecast_start, forecast.point_forecast)
    
    def backcast(self, data: TsData, steps: int) -> np.ndarray:
        """Generate backcasts (forecasts in reverse time).
        
        Args:
            data: Historical data
            steps: Number of periods to backcast
            
        Returns:
            Array of backcasted values
        """
        # Reverse the series
        reversed_data = TsData.of(data.start, data.values[::-1])
        
        # Forecast on reversed series
        forecast = self.forecast(reversed_data, steps)
        
        # Reverse the forecasts
        return forecast.point_forecast[::-1]
    
    def in_sample_forecast(self, data: TsData, 
                          start: Optional[int] = None) -> np.ndarray:
        """Generate in-sample forecasts (one-step ahead).
        
        Args:
            data: Full dataset
            start: Starting index for forecasts (default: after initial values)
            
        Returns:
            Array of one-step ahead forecasts
        """
        if start is None:
            start = max(self.model.order.p, self.model.order.q)
        
        n = data.length
        forecasts = np.full(n, np.nan)
        
        # Generate one-step ahead forecasts
        for t in range(start, n):
            # Use data up to time t-1
            subset = data.select(data.domain.extend(0, -(n - t)))
            forecast = self.forecast(subset, steps=1)
            forecasts[t] = forecast.point_forecast[0]
        
        return forecasts
    
    def simulate(self, steps: int, nsim: int = 1000,
                data: Optional[TsData] = None,
                random_state: Optional[int] = None) -> np.ndarray:
        """Simulate future paths from model.
        
        Args:
            steps: Number of steps to simulate
            nsim: Number of simulation paths
            data: Historical data (for conditional simulation)
            random_state: Random seed
            
        Returns:
            Array of shape (nsim, steps) with simulated paths
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate innovation samples
        innovations = np.random.normal(
            0, np.sqrt(self.model.variance), 
            size=(nsim, steps)
        )
        
        # Initialize simulation array
        simulations = np.zeros((nsim, steps))
        
        # Get initial values from data if provided
        if data is not None:
            # Use last few observations as initial conditions
            p = self.model.order.p
            q = self.model.order.q
            
            initial_values = data.values[-p:] if p > 0 else []
            initial_innovations = np.zeros(q)  # Assume zero past innovations
        else:
            initial_values = []
            initial_innovations = []
        
        # Simulate each path
        for i in range(nsim):
            path = self._simulate_path(
                innovations[i], 
                initial_values,
                initial_innovations
            )
            simulations[i] = path
        
        return simulations
    
    def _create_params_vector(self) -> np.ndarray:
        """Create parameter vector for statsmodels."""
        params = []
        
        # AR parameters
        if self.model.order.p > 0:
            params.extend(self.model.ar_params)
        
        # MA parameters  
        if self.model.order.q > 0:
            params.extend(self.model.ma_params)
        
        # Seasonal parameters for SARIMA
        if isinstance(self.model, SarimaModel):
            if self.model.seasonal_order.p > 0:
                params.extend(self.model.seasonal_ar_params)
            if self.model.seasonal_order.q > 0:
                params.extend(self.model.seasonal_ma_params)
        
        # Mean/intercept
        if hasattr(self.model, 'mean') and self.model.mean != 0:
            params.insert(0, self.model.mean)
        
        # Variance
        params.append(self.model.variance)
        
        return np.array(params)
    
    def _simulate_path(self, innovations: np.ndarray,
                      initial_values: np.ndarray,
                      initial_innovations: np.ndarray) -> np.ndarray:
        """Simulate single path from ARIMA model."""
        n = len(innovations)
        y = np.zeros(n)
        
        # Apply ARMA structure
        for t in range(n):
            # AR part
            for j in range(min(t, self.model.order.p)):
                if t - j - 1 >= 0:
                    y[t] += self.model.ar_params[j] * y[t - j - 1]
                elif j < len(initial_values):
                    # Use initial values
                    idx = len(initial_values) - j - 1
                    y[t] += self.model.ar_params[j] * initial_values[idx]
            
            # MA part
            y[t] += innovations[t]
            for j in range(min(t, self.model.order.q)):
                if t - j - 1 >= 0:
                    y[t] += self.model.ma_params[j] * innovations[t - j - 1]
            
            # Add mean
            y[t] += self.model.mean
        
        return y