"""TRAMO pre-adjustment processor."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import logging

from ...toolkit.timeseries import TsData, TsPeriod
from ...toolkit.arima import (
    ArimaModel, ArimaEstimator, ArimaForecaster,
    EstimationMethod
)
from ...toolkit.regression import (
    TsVariable, TrendConstant, TradingDays, Easter,
    Outlier, OutlierType
)
from .specification import TramoSpec


@dataclass
class TramoResults:
    """Results from TRAMO pre-adjustment."""
    
    # Original and linearized series
    original: TsData
    linearized: TsData
    
    # Transformation
    log_transformed: bool
    transformation_adjustment: Optional[float] = None
    
    # ARIMA model
    arima_model: Optional[ArimaModel] = None
    
    # Regression effects
    regression_effects: Dict[str, TsData] = None
    regression_coefficients: Dict[str, float] = None
    
    # Outliers
    outliers: List[Outlier] = None
    
    # Residuals
    residuals: Optional[TsData] = None
    
    # Forecasts/Backcasts
    forecasts: Optional[TsData] = None
    backcasts: Optional[TsData] = None
    
    # Diagnostics
    diagnostics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.regression_effects is None:
            self.regression_effects = {}
        if self.regression_coefficients is None:
            self.regression_coefficients = {}
        if self.outliers is None:
            self.outliers = []
        if self.diagnostics is None:
            self.diagnostics = {}


class TramoProcessor:
    """TRAMO processor for pre-adjustment."""
    
    def __init__(self, specification: TramoSpec):
        """Initialize TRAMO processor.
        
        Args:
            specification: TRAMO specification
        """
        self.spec = specification
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(self, data: TsData) -> TramoResults:
        """Process series with TRAMO.
        
        Args:
            data: Input time series
            
        Returns:
            TRAMO results
        """
        # Initialize results
        results = TramoResults(
            original=data.copy(),
            linearized=data.copy(),
            log_transformed=False
        )
        
        # Apply transformation
        transformed = self._apply_transformation(data, results)
        
        # Build regression variables
        regression_vars = self._build_regression_variables(transformed)
        
        # Detect outliers if enabled
        if self.spec.outliers.enabled:
            outliers = self._detect_outliers(transformed, regression_vars)
            results.outliers = outliers
            
            # Add outliers to regression
            for outlier in outliers:
                regression_vars[outlier.name()] = outlier
        
        # Estimate RegARIMA model
        model_results = self._estimate_regarima(transformed, regression_vars)
        
        # Store results
        results.arima_model = model_results['model']
        results.regression_coefficients = model_results['coefficients']
        results.residuals = model_results['residuals']
        
        # Compute regression effects
        for name, var in regression_vars.items():
            effect = self._compute_effect(var, model_results['coefficients'].get(name, 0),
                                        transformed.domain)
            results.regression_effects[name] = effect
        
        # Linearize series
        results.linearized = self._linearize_series(transformed, results.regression_effects)
        
        # Generate forecasts/backcasts
        if self.spec.fcasts != 0:
            results.forecasts = self._generate_forecasts(results)
        if self.spec.bcasts != 0:
            results.backcasts = self._generate_backcasts(results)
        
        # Compute diagnostics
        results.diagnostics = self._compute_diagnostics(results)
        
        return results
    
    def _apply_transformation(self, data: TsData, results: TramoResults) -> TsData:
        """Apply transformation to series.
        
        Args:
            data: Original series
            results: Results object to update
            
        Returns:
            Transformed series
        """
        if self.spec.transform.function == "log" or \
           (self.spec.transform.function == "auto" and 
            self.spec.transform.should_use_log(data)):
            
            # Check for non-positive values
            min_val = np.min(data.values)
            if min_val <= 0:
                # Add constant
                adjustment = 1 - min_val
                adjusted = TsData.of(data.start, data.values + adjustment)
                results.transformation_adjustment = adjustment
                self.logger.warning(f"Added constant {adjustment} before log transformation")
            else:
                adjusted = data
                results.transformation_adjustment = None
            
            # Apply log
            transformed = adjusted.fn(np.log)
            results.log_transformed = True
            
            return transformed
        else:
            return data
    
    def _build_regression_variables(self, data: TsData) -> Dict[str, TsVariable]:
        """Build regression variables.
        
        Args:
            data: Time series
            
        Returns:
            Dictionary of regression variables
        """
        vars = {}
        
        # Mean/Intercept
        if self.spec.arima.mean:
            vars["const"] = TrendConstant()
        
        # Trading days
        if self.spec.trading_days is not None:
            td = TradingDays(
                contrast=True,
                include_length_of_month=(self.spec.trading_days.lp_type == "LengthOfPeriod"),
                include_leap_year=(self.spec.trading_days.lp_type == "LeapYear")
            )
            vars["td"] = td
        
        # Easter
        if self.spec.easter is not None:
            easter = Easter(duration=self.spec.easter.duration)
            vars["easter"] = easter
        
        return vars
    
    def _detect_outliers(self, data: TsData, 
                        regression_vars: Dict[str, TsVariable]) -> List[Outlier]:
        """Detect outliers in series.
        
        Args:
            data: Time series
            regression_vars: Current regression variables
            
        Returns:
            List of detected outliers
        """
        outliers = []
        
        # Simplified outlier detection
        # In practice, this would use iterative detection with t-statistics
        
        # Estimate initial model
        initial_results = self._estimate_regarima(data, regression_vars)
        residuals = initial_results['residuals'].values
        
        # Compute robust standard deviation
        mad = np.median(np.abs(residuals - np.median(residuals)))
        robust_std = 1.4826 * mad
        
        # Find outliers
        threshold = self.spec.outliers.critical_value * robust_std
        
        for i, res in enumerate(residuals):
            if np.abs(res) > threshold:
                period = data.domain.get(i)
                
                # Determine outlier type (simplified)
                if "AO" in self.spec.outliers.types:
                    outlier = Outlier(period, OutlierType.AO)
                    outliers.append(outlier)
        
        self.logger.info(f"Detected {len(outliers)} outliers")
        
        return outliers
    
    def _estimate_regarima(self, data: TsData,
                          regression_vars: Dict[str, TsVariable]) -> Dict[str, Any]:
        """Estimate RegARIMA model.
        
        Args:
            data: Time series
            regression_vars: Regression variables
            
        Returns:
            Estimation results
        """
        # Build regression matrix
        if regression_vars:
            X = self._build_regression_matrix(data.domain, regression_vars)
        else:
            X = None
        
        # Determine ARIMA order
        if self.spec.arima.auto_model:
            # Automatic model selection
            estimator = ArimaEstimator(method=EstimationMethod.MLE)
            model = estimator.estimate_auto(
                data,
                seasonal=data.domain.frequency.periods_per_year > 1,
                period=data.domain.frequency.periods_per_year
            )
        else:
            # Fixed order
            order = self.spec.arima.to_arima_order(data.domain.frequency.periods_per_year)
            if order is None:
                raise ValueError("ARIMA order not fully specified")
            
            estimator = ArimaEstimator(method=EstimationMethod.MLE)
            model = estimator.estimate(data, order)
        
        # Extract results
        results = {
            'model': model,
            'coefficients': {},
            'residuals': TsData.of(data.start, getattr(model, '_residuals', np.zeros(data.length)))
        }
        
        # Store regression coefficients (simplified)
        for i, (name, var) in enumerate(regression_vars.items()):
            results['coefficients'][name] = 0.0  # Placeholder
        
        return results
    
    def _compute_effect(self, variable: TsVariable, coefficient: float,
                       domain: 'TsDomain') -> TsData:
        """Compute regression effect.
        
        Args:
            variable: Regression variable
            coefficient: Estimated coefficient
            domain: Time domain
            
        Returns:
            Effect series
        """
        values = variable.get_values(domain)
        
        if values.ndim == 2:
            # Multiple columns - sum effects
            effect = np.sum(values * coefficient, axis=1)
        else:
            effect = values * coefficient
        
        return TsData.of(domain.start, effect)
    
    def _linearize_series(self, data: TsData,
                         regression_effects: Dict[str, TsData]) -> TsData:
        """Remove regression effects to get linearized series.
        
        Args:
            data: Original series
            regression_effects: Dictionary of effects
            
        Returns:
            Linearized series
        """
        linearized = data.values.copy()
        
        # Remove each effect
        for name, effect in regression_effects.items():
            if name not in ["const"]:  # Keep mean
                linearized -= effect.values
        
        return TsData.of(data.start, linearized)
    
    def _generate_forecasts(self, results: TramoResults) -> Optional[TsData]:
        """Generate forecasts.
        
        Args:
            results: TRAMO results
            
        Returns:
            Forecast series
        """
        if results.arima_model is None:
            return None
        
        # Determine forecast horizon
        if self.spec.fcasts == -1:
            # Automatic: 1 year
            n_fcasts = results.original.domain.frequency.periods_per_year
        else:
            n_fcasts = self.spec.fcasts
        
        if n_fcasts <= 0:
            return None
        
        # Generate forecasts
        forecaster = ArimaForecaster(results.arima_model)
        forecast_ts = forecaster.forecast_to_tsdata(results.linearized, n_fcasts)
        
        return forecast_ts
    
    def _generate_backcasts(self, results: TramoResults) -> Optional[TsData]:
        """Generate backcasts.
        
        Args:
            results: TRAMO results
            
        Returns:
            Backcast series
        """
        if results.arima_model is None or self.spec.bcasts <= 0:
            return None
        
        # Simplified: return empty for now
        return None
    
    def _compute_diagnostics(self, results: TramoResults) -> Dict[str, Any]:
        """Compute diagnostics.
        
        Args:
            results: TRAMO results
            
        Returns:
            Dictionary of diagnostics
        """
        diagnostics = {}
        
        if results.residuals is not None:
            from ..diagnostics.residuals import compute_residuals_diagnostics
            
            residual_diag = compute_residuals_diagnostics(results.residuals)
            diagnostics['residuals'] = residual_diag
        
        if results.arima_model is not None:
            diagnostics['model_order'] = str(results.arima_model.order)
        
        diagnostics['n_outliers'] = len(results.outliers)
        
        return diagnostics
    
    def _build_regression_matrix(self, domain: 'TsDomain',
                                variables: Dict[str, TsVariable]) -> np.ndarray:
        """Build regression matrix.
        
        Args:
            domain: Time domain
            variables: Regression variables
            
        Returns:
            Regression matrix
        """
        matrices = []
        
        for var in variables.values():
            values = var.get_values(domain)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            matrices.append(values)
        
        if matrices:
            return np.hstack(matrices)
        else:
            return None