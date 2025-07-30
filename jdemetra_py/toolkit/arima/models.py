"""ARIMA model implementations matching JDemetra+."""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..math.polynomials import Polynomial
from ..timeseries.data import TsData


@dataclass
class ArimaOrder:
    """ARIMA model order specification."""
    p: int  # AR order
    d: int  # Differencing order
    q: int  # MA order
    
    def __post_init__(self):
        if self.p < 0 or self.d < 0 or self.q < 0:
            raise ValueError("Orders must be non-negative")
    
    @property
    def is_pure_ar(self) -> bool:
        """Check if model is pure AR."""
        return self.q == 0
    
    @property
    def is_pure_ma(self) -> bool:
        """Check if model is pure MA."""
        return self.p == 0
    
    @property
    def is_arma(self) -> bool:
        """Check if model is ARMA (no differencing)."""
        return self.d == 0
    
    def __str__(self) -> str:
        return f"ARIMA({self.p},{self.d},{self.q})"


@dataclass
class SarimaOrder:
    """Seasonal ARIMA model order specification."""
    order: ArimaOrder  # Non-seasonal order
    seasonal_order: ArimaOrder  # Seasonal order
    period: int  # Seasonal period
    
    def __post_init__(self):
        if self.period < 2:
            raise ValueError("Seasonal period must be at least 2")
    
    def __str__(self) -> str:
        return (f"SARIMA({self.order.p},{self.order.d},{self.order.q})"
                f"({self.seasonal_order.p},{self.seasonal_order.d},{self.seasonal_order.q})"
                f"[{self.period}]")


class ArimaModel:
    """ARIMA model representation matching JDemetra+.
    
    This class wraps statsmodels ARIMA but provides JDemetra+ compatible API.
    """
    
    def __init__(self, order: ArimaOrder, 
                 ar_params: Optional[np.ndarray] = None,
                 ma_params: Optional[np.ndarray] = None,
                 variance: float = 1.0,
                 mean: float = 0.0):
        """Initialize ARIMA model.
        
        Args:
            order: Model order (p, d, q)
            ar_params: AR coefficients (without leading 1)
            ma_params: MA coefficients (without leading 1)
            variance: Innovation variance
            mean: Mean parameter (for models with constant)
        """
        self.order = order
        self.variance = variance
        self.mean = mean
        
        # Initialize parameters
        if ar_params is not None:
            self.ar_params = np.asarray(ar_params)
            if len(self.ar_params) != order.p:
                raise ValueError(f"Expected {order.p} AR parameters")
        else:
            self.ar_params = np.zeros(order.p)
        
        if ma_params is not None:
            self.ma_params = np.asarray(ma_params)
            if len(self.ma_params) != order.q:
                raise ValueError(f"Expected {order.q} MA parameters")
        else:
            self.ma_params = np.zeros(order.q)
    
    @property
    def ar_polynomial(self) -> Polynomial:
        """Get AR polynomial (1 - φ₁L - φ₂L² - ...)."""
        coeffs = np.concatenate([[1], -self.ar_params])
        return Polynomial(coeffs)
    
    @property
    def ma_polynomial(self) -> Polynomial:
        """Get MA polynomial (1 + θ₁L + θ₂L² + ...)."""
        coeffs = np.concatenate([[1], self.ma_params])
        return Polynomial(coeffs)
    
    @property
    def is_stationary(self) -> bool:
        """Check if AR part is stationary."""
        if self.order.p == 0:
            return True
        
        # Check if all AR roots are outside unit circle
        roots = self.ar_polynomial.roots()
        return np.all(np.abs(roots) > 1.0)
    
    @property
    def is_invertible(self) -> bool:
        """Check if MA part is invertible."""
        if self.order.q == 0:
            return True
        
        # Check if all MA roots are outside unit circle
        roots = self.ma_polynomial.roots()
        return np.all(np.abs(roots) > 1.0)
    
    def fit(self, data: TsData, method: str = 'mle') -> 'ArimaModel':
        """Fit model to data using statsmodels.
        
        Args:
            data: Time series data
            method: Estimation method ('mle', 'css', 'css-mle')
            
        Returns:
            Fitted model
        """
        # Convert to pandas for statsmodels
        series = data.to_pandas()
        
        # Fit using statsmodels
        sm_model = ARIMA(series, order=(self.order.p, self.order.d, self.order.q))
        sm_result = sm_model.fit(method=method)
        
        # Extract parameters
        ar_params = sm_result.arparams if self.order.p > 0 else None
        ma_params = sm_result.maparams if self.order.q > 0 else None
        
        # Create fitted model
        return ArimaModel(
            order=self.order,
            ar_params=ar_params,
            ma_params=ma_params,
            variance=sm_result.sigma2,
            mean=sm_result.params.get('const', 0.0)
        )
    
    def forecast(self, steps: int, data: Optional[TsData] = None) -> np.ndarray:
        """Generate forecasts.
        
        Args:
            steps: Number of steps ahead
            data: Historical data (optional, for conditional forecasts)
            
        Returns:
            Array of forecasts
        """
        if data is None:
            # Unconditional forecast (from model only)
            # For now, return mean
            return np.full(steps, self.mean)
        
        # Use statsmodels for forecasting
        series = data.to_pandas()
        sm_model = ARIMA(series, order=(self.order.p, self.order.d, self.order.q))
        
        # Set parameters
        params = []
        if self.order.p > 0:
            params.extend(self.ar_params)
        if self.order.q > 0:
            params.extend(self.ma_params)
        params.append(self.variance)
        
        # Generate forecast
        # Note: This is simplified - proper implementation would use fitted model
        forecast = sm_model.filter(params).forecast(steps=steps)
        
        return forecast.values
    
    def simulate(self, nobs: int, random_state: Optional[int] = None) -> np.ndarray:
        """Simulate from model.
        
        Args:
            nobs: Number of observations
            random_state: Random seed
            
        Returns:
            Simulated series
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate innovations
        innovations = np.random.normal(0, np.sqrt(self.variance), nobs)
        
        # Initialize series
        y = np.zeros(nobs)
        
        # Apply MA part
        for t in range(nobs):
            y[t] = innovations[t]
            for j in range(min(t, self.order.q)):
                y[t] += self.ma_params[j] * innovations[t - j - 1]
        
        # Apply AR part (if stationary)
        if self.order.p > 0 and self.is_stationary:
            for t in range(self.order.p, nobs):
                for j in range(self.order.p):
                    y[t] += self.ar_params[j] * y[t - j - 1]
        
        # Add mean
        y += self.mean
        
        # Apply differencing
        for _ in range(self.order.d):
            y = np.diff(y)
        
        return y
    
    def __repr__(self) -> str:
        return f"ArimaModel({self.order})"


class SarimaModel(ArimaModel):
    """Seasonal ARIMA model."""
    
    def __init__(self, order: SarimaOrder,
                 ar_params: Optional[np.ndarray] = None,
                 ma_params: Optional[np.ndarray] = None,
                 seasonal_ar_params: Optional[np.ndarray] = None,
                 seasonal_ma_params: Optional[np.ndarray] = None,
                 variance: float = 1.0,
                 mean: float = 0.0):
        """Initialize SARIMA model.
        
        Args:
            order: Model order specification
            ar_params: Non-seasonal AR parameters
            ma_params: Non-seasonal MA parameters
            seasonal_ar_params: Seasonal AR parameters
            seasonal_ma_params: Seasonal MA parameters
            variance: Innovation variance
            mean: Mean parameter
        """
        # Initialize base ARIMA with non-seasonal part
        super().__init__(order.order, ar_params, ma_params, variance, mean)
        
        self.seasonal_order = order.seasonal_order
        self.period = order.period
        
        # Initialize seasonal parameters
        if seasonal_ar_params is not None:
            self.seasonal_ar_params = np.asarray(seasonal_ar_params)
            if len(self.seasonal_ar_params) != order.seasonal_order.p:
                raise ValueError(f"Expected {order.seasonal_order.p} seasonal AR parameters")
        else:
            self.seasonal_ar_params = np.zeros(order.seasonal_order.p)
        
        if seasonal_ma_params is not None:
            self.seasonal_ma_params = np.asarray(seasonal_ma_params)
            if len(self.seasonal_ma_params) != order.seasonal_order.q:
                raise ValueError(f"Expected {order.seasonal_order.q} seasonal MA parameters")
        else:
            self.seasonal_ma_params = np.zeros(order.seasonal_order.q)
    
    @property
    def seasonal_ar_polynomial(self) -> Polynomial:
        """Get seasonal AR polynomial."""
        # Create polynomial with coefficients at seasonal lags
        coeffs = np.zeros(self.seasonal_order.p * self.period + 1)
        coeffs[0] = 1
        for i, param in enumerate(self.seasonal_ar_params):
            coeffs[(i + 1) * self.period] = -param
        return Polynomial(coeffs)
    
    @property
    def seasonal_ma_polynomial(self) -> Polynomial:
        """Get seasonal MA polynomial."""
        # Create polynomial with coefficients at seasonal lags
        coeffs = np.zeros(self.seasonal_order.q * self.period + 1)
        coeffs[0] = 1
        for i, param in enumerate(self.seasonal_ma_params):
            coeffs[(i + 1) * self.period] = param
        return Polynomial(coeffs)
    
    def fit(self, data: TsData, method: str = 'mle') -> 'SarimaModel':
        """Fit SARIMA model to data.
        
        Args:
            data: Time series data
            method: Estimation method
            
        Returns:
            Fitted model
        """
        # Convert to pandas for statsmodels
        series = data.to_pandas()
        
        # Fit using statsmodels SARIMAX
        sm_model = SARIMAX(
            series,
            order=(self.order.p, self.order.d, self.order.q),
            seasonal_order=(self.seasonal_order.p, self.seasonal_order.d, 
                          self.seasonal_order.q, self.period)
        )
        sm_result = sm_model.fit(disp=False)
        
        # Extract parameters
        # Note: statsmodels parameter ordering needs careful handling
        params_dict = sm_result.params.to_dict()
        
        # Create fitted model with extracted parameters
        sarima_order = SarimaOrder(self.order, self.seasonal_order, self.period)
        
        return SarimaModel(
            order=sarima_order,
            ar_params=sm_result.arparams if self.order.p > 0 else None,
            ma_params=sm_result.maparams if self.order.q > 0 else None,
            seasonal_ar_params=sm_result.seasonalarparams if self.seasonal_order.p > 0 else None,
            seasonal_ma_params=sm_result.seasonalmaparams if self.seasonal_order.q > 0 else None,
            variance=sm_result.sigma2,
            mean=params_dict.get('const', 0.0)
        )
    
    def __repr__(self) -> str:
        return (f"SarimaModel({self.order.p},{self.order.d},{self.order.q})"
                f"({self.seasonal_order.p},{self.seasonal_order.d},{self.seasonal_order.q})"
                f"[{self.period}]")