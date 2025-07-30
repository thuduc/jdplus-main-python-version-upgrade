"""ARIMA model estimation."""

from enum import Enum
from typing import Optional, Dict, Any
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .models import ArimaModel, SarimaModel, ArimaOrder, SarimaOrder
from ..timeseries.data import TsData


class EstimationMethod(Enum):
    """Estimation methods for ARIMA models."""
    MLE = "mle"  # Maximum likelihood
    CSS = "css"  # Conditional sum of squares
    CSS_MLE = "css-mle"  # CSS followed by MLE
    HANNAN_RISSANEN = "hannan_rissanen"  # Hannan-Rissanen


class ArimaEstimator:
    """ARIMA model estimator matching JDemetra+ functionality."""
    
    def __init__(self, 
                 method: EstimationMethod = EstimationMethod.MLE,
                 maxiter: int = 1000,
                 tolerance: float = 1e-8,
                 use_exact_diffuse: bool = False):
        """Initialize estimator.
        
        Args:
            method: Estimation method
            maxiter: Maximum iterations for optimization
            tolerance: Convergence tolerance
            use_exact_diffuse: Use exact diffuse initialization
        """
        self.method = method
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.use_exact_diffuse = use_exact_diffuse
    
    def estimate(self, data: TsData, order: ArimaOrder, 
                 include_mean: bool = True,
                 fixed_params: Optional[Dict[str, float]] = None) -> ArimaModel:
        """Estimate ARIMA model.
        
        Args:
            data: Time series data
            order: Model order
            include_mean: Include mean/intercept term
            fixed_params: Fixed parameter values (optional)
            
        Returns:
            Estimated model
        """
        # Convert to pandas for statsmodels
        series = data.to_pandas()
        
        # Create statsmodels ARIMA
        # Handle trend parameter based on differencing
        if order.d > 0 or (isinstance(order, SarimaOrder) and order.D > 0):
            # With differencing, can't include constant, use 'n' (no trend)
            trend = 'n'
        else:
            trend = 'c' if include_mean else 'n'
        
        # Build seasonal order if needed
        if isinstance(order, SarimaOrder) and order.s > 1:
            seasonal_order = (order.P, order.D, order.Q, order.s)
        else:
            seasonal_order = (0, 0, 0, 0)  # Default non-seasonal
            
        sm_model = ARIMA(
            series, 
            order=(order.p, order.d, order.q),
            seasonal_order=seasonal_order,
            trend=trend
        )
        
        # Fit model - handle method parameter
        fit_kwargs = {}
        if self.method != EstimationMethod.MLE:
            # MLE is default, only specify if different
            fit_kwargs['method'] = self.method.value.lower()
        
        # Handle fixed parameters if provided
        if fixed_params:
            # TODO: Implement parameter fixing
            pass
        
        sm_result = sm_model.fit(**fit_kwargs)
        
        # Extract results
        ar_params = sm_result.arparams if order.p > 0 else None
        ma_params = sm_result.maparams if order.q > 0 else None
        
        # Create JDemetra+ style model
        model = ArimaModel(
            order=order,
            ar_params=ar_params,
            ma_params=ma_params,
            variance=sm_result.scale,  # ARIMA uses scale, not sigma2
            mean=sm_result.params.get('const', 0.0) if include_mean else 0.0
        )
        
        # Store additional diagnostics
        model._loglikelihood = sm_result.llf
        model._aic = sm_result.aic
        model._bic = sm_result.bic
        model._residuals = sm_result.resid.values
        
        return model
    
    def estimate_sarima(self, data: TsData, order: SarimaOrder,
                       include_mean: bool = True,
                       fixed_params: Optional[Dict[str, float]] = None) -> SarimaModel:
        """Estimate SARIMA model.
        
        Args:
            data: Time series data
            order: Model order specification
            include_mean: Include mean/intercept term
            fixed_params: Fixed parameter values (optional)
            
        Returns:
            Estimated model
        """
        # Convert to pandas for statsmodels
        series = data.to_pandas()
        
        # Create statsmodels SARIMAX
        sm_model = SARIMAX(
            series,
            order=(order.order.p, order.order.d, order.order.q),
            seasonal_order=(order.seasonal_order.p, order.seasonal_order.d,
                          order.seasonal_order.q, order.period),
            trend='c' if include_mean else 'n',
            initialization='approximate_diffuse' if self.use_exact_diffuse else None
        )
        
        # Fit model
        fit_kwargs = {
            'maxiter': self.maxiter,
            'tolerance': self.tolerance,
            'disp': False
        }
        
        # MLE is default for SARIMAX
        if self.method == EstimationMethod.CSS:
            fit_kwargs['method'] = 'css'
        
        sm_result = sm_model.fit(**fit_kwargs)
        
        # Extract parameters
        model = SarimaModel(
            order=order,
            ar_params=sm_result.polynomial_ar[1:] if order.order.p > 0 else None,
            ma_params=sm_result.polynomial_ma[1:] if order.order.q > 0 else None,
            seasonal_ar_params=sm_result.polynomial_seasonal_ar[1:] if order.seasonal_order.p > 0 else None,
            seasonal_ma_params=sm_result.polynomial_seasonal_ma[1:] if order.seasonal_order.q > 0 else None,
            variance=sm_result.scale,  # SARIMAX uses scale
            mean=sm_result.params.get('const', 0.0) if include_mean else 0.0
        )
        
        # Store diagnostics
        model._loglikelihood = sm_result.llf
        model._aic = sm_result.aic
        model._bic = sm_result.bic
        model._residuals = sm_result.resid.values
        
        return model
    
    def compute_information_criteria(self, model: ArimaModel, 
                                   data: TsData) -> Dict[str, float]:
        """Compute information criteria for model selection.
        
        Args:
            model: Fitted model
            data: Original data
            
        Returns:
            Dictionary with AIC, BIC, AICC values
        """
        n = data.length
        
        # Get log-likelihood
        if hasattr(model, '_loglikelihood'):
            llf = model._loglikelihood
        else:
            # Compute from residuals if not available
            residuals = getattr(model, '_residuals', None)
            if residuals is not None:
                llf = -0.5 * n * (np.log(2 * np.pi) + np.log(model.variance) + 1)
            else:
                llf = np.nan
        
        # Count parameters
        k = model.order.p + model.order.q
        if hasattr(model, 'mean') and model.mean != 0:
            k += 1
        if isinstance(model, SarimaModel):
            k += model.seasonal_order.p + model.seasonal_order.q
        
        # Compute criteria
        aic = -2 * llf + 2 * k
        bic = -2 * llf + k * np.log(n)
        
        # Corrected AIC for small samples
        if n - k - 1 > 0:
            aicc = aic + 2 * k * (k + 1) / (n - k - 1)
        else:
            aicc = np.nan
        
        return {
            'aic': aic,
            'bic': bic,
            'aicc': aicc,
            'loglikelihood': llf
        }
    
    def estimate_auto(self, data: TsData, 
                     max_p: int = 5,
                     max_q: int = 5,
                     max_d: int = 2,
                     seasonal: bool = False,
                     period: Optional[int] = None,
                     information_criterion: str = 'aic') -> ArimaModel:
        """Automatic ARIMA model selection.
        
        Args:
            data: Time series data
            max_p: Maximum AR order
            max_q: Maximum MA order  
            max_d: Maximum differencing order
            seasonal: Include seasonal terms
            period: Seasonal period (required if seasonal=True)
            information_criterion: Criterion for selection ('aic', 'bic', 'aicc')
            
        Returns:
            Best model according to criterion
        """
        # For full auto-ARIMA, we would use pmdarima
        # This is a simplified version
        from pmdarima import auto_arima
        
        series = data.to_pandas()
        
        # Use auto_arima for model selection
        auto_model = auto_arima(
            series,
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            seasonal=seasonal,
            m=period if seasonal else 1,
            information_criterion=information_criterion,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        # Extract order
        order_tuple = auto_model.order
        seasonal_order_tuple = auto_model.seasonal_order if seasonal else None
        
        # Convert to our model
        if seasonal and seasonal_order_tuple:
            sarima_order = SarimaOrder(
                ArimaOrder(*order_tuple),
                ArimaOrder(*seasonal_order_tuple[:3]),
                seasonal_order_tuple[3]
            )
            return self.estimate_sarima(data, sarima_order)
        else:
            arima_order = ArimaOrder(*order_tuple)
            return self.estimate(data, arima_order)