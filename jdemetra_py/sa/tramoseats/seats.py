"""SEATS decomposition."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import logging

from ...toolkit.timeseries import TsData
from ...toolkit.arima import ArimaModel, SarimaModel
from ...toolkit.ssf import (
    CompositeSSM, LocalLinearTrend, SeasonalComponent,
    ArmaComponent, KalmanSmoother, Measurement
)
from ..base.results import SeriesDecomposition, DecompositionMode
from .specification import SeatsSpec
from .tramo import TramoResults


@dataclass
class SeatsResults:
    """Results from SEATS decomposition."""
    
    # Components
    trend: Optional[TsData] = None
    seasonal: Optional[TsData] = None
    irregular: Optional[TsData] = None
    
    # Trend-cycle
    trend_cycle: Optional[TsData] = None
    
    # Component models
    trend_model: Optional[ArimaModel] = None
    seasonal_model: Optional[ArimaModel] = None
    irregular_model: Optional[ArimaModel] = None
    
    # Variances
    trend_variance: Optional[float] = None
    seasonal_variance: Optional[float] = None
    irregular_variance: Optional[float] = None
    
    # Diagnostics
    diagnostics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.diagnostics is None:
            self.diagnostics = {}


class SeatsDecomposer:
    """SEATS decomposition engine."""
    
    def __init__(self, specification: SeatsSpec):
        """Initialize SEATS decomposer.
        
        Args:
            specification: SEATS specification
        """
        self.spec = specification
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def decompose(self, tramo_results: TramoResults) -> SeatsResults:
        """Perform SEATS decomposition.
        
        Args:
            tramo_results: Results from TRAMO pre-adjustment
            
        Returns:
            SEATS decomposition results
        """
        # Get linearized series and ARIMA model
        series = tramo_results.linearized
        model = tramo_results.arima_model
        
        if model is None:
            raise ValueError("No ARIMA model available from TRAMO")
        
        # Perform canonical decomposition
        component_models = self._canonical_decomposition(model)
        
        # Extract components using appropriate method
        if self.spec.method == "KalmanSmoother":
            components = self._extract_components_kalman(series, model, component_models)
        else:  # Burman
            components = self._extract_components_burman(series, model, component_models)
        
        # Create results
        results = SeatsResults(
            trend=components.get('trend'),
            seasonal=components.get('seasonal'),
            irregular=components.get('irregular'),
            trend_model=component_models.get('trend'),
            seasonal_model=component_models.get('seasonal'),
            irregular_model=component_models.get('irregular')
        )
        
        # Compute variances
        if results.trend is not None:
            results.trend_variance = np.var(results.trend.values[~np.isnan(results.trend.values)])
        if results.seasonal is not None:
            results.seasonal_variance = np.var(results.seasonal.values[~np.isnan(results.seasonal.values)])
        if results.irregular is not None:
            results.irregular_variance = np.var(results.irregular.values[~np.isnan(results.irregular.values)])
        
        # Apply bias correction if requested
        if self.spec.bias_correction:
            results = self._apply_bias_correction(results, series)
        
        return results
    
    def _canonical_decomposition(self, model: ArimaModel) -> Dict[str, ArimaModel]:
        """Perform canonical decomposition of ARIMA model.
        
        Args:
            model: ARIMA model
            
        Returns:
            Dictionary of component models
        """
        # Simplified canonical decomposition
        # In practice, this involves partial fraction decomposition
        
        components = {}
        
        # For now, use simple models
        from ...toolkit.arima import ArimaOrder
        
        # Trend: Random walk with drift
        components['trend'] = ArimaModel(
            ArimaOrder(0, 2, 2),
            ma_params=[0.5, 0.3],
            variance=0.1
        )
        
        # Seasonal (if seasonal model)
        if isinstance(model, SarimaModel) and model.seasonal_order.p + model.seasonal_order.q > 0:
            components['seasonal'] = ArimaModel(
                ArimaOrder(0, 0, model.period - 1),
                ma_params=np.ones(model.period - 1) / model.period,
                variance=0.05
            )
        
        # Irregular: White noise
        components['irregular'] = ArimaModel(
            ArimaOrder(0, 0, 0),
            variance=0.01
        )
        
        return components
    
    def _extract_components_kalman(self, series: TsData, model: ArimaModel,
                                  component_models: Dict[str, ArimaModel]) -> Dict[str, TsData]:
        """Extract components using Kalman smoother.
        
        Args:
            series: Linearized series
            model: Full ARIMA model
            component_models: Component models
            
        Returns:
            Dictionary of components
        """
        # Build state space model
        components = []
        
        # Trend component
        if 'trend' in component_models:
            trend_ssm = LocalLinearTrend(
                level_variance=component_models['trend'].variance,
                slope_variance=0.01
            )
            components.append(trend_ssm)
        
        # Seasonal component
        if 'seasonal' in component_models:
            period = series.domain.frequency.periods_per_year
            seasonal_ssm = SeasonalComponent(
                period=period,
                variance=component_models['seasonal'].variance
            )
            components.append(seasonal_ssm)
        
        # ARMA component for cycle/irregular
        if model.order.p > 0 or model.order.q > 0:
            arma_ssm = ArmaComponent(model)
            components.append(arma_ssm)
        
        # Composite model
        ssm = CompositeSSM(components)
        
        # Prepare observations
        measurements = []
        for val in series.values:
            if np.isnan(val):
                measurements.append(Measurement.missing())
            else:
                measurements.append(Measurement(val))
        
        # Run smoother
        smoother = KalmanSmoother(ssm)
        smoothed_states = smoother.smooth(measurements)
        
        # Extract components from states
        extracted = {}
        
        # Simplified extraction - would need proper state mapping
        smoothed_values = np.array([s.smoothed_state.values[0] for s in smoothed_states])
        
        # For now, use simple allocation
        extracted['trend'] = TsData.of(series.start, smoothed_values * 0.7)
        
        if 'seasonal' in component_models:
            seasonal_pattern = self._generate_seasonal_pattern(series)
            extracted['seasonal'] = seasonal_pattern
        
        # Irregular is residual
        irregular = series.values - extracted['trend'].values
        if 'seasonal' in extracted:
            irregular -= extracted['seasonal'].values
        extracted['irregular'] = TsData.of(series.start, irregular)
        
        return extracted
    
    def _extract_components_burman(self, series: TsData, model: ArimaModel,
                                  component_models: Dict[str, ArimaModel]) -> Dict[str, TsData]:
        """Extract components using Burman-Wilson method.
        
        Args:
            series: Linearized series
            model: Full ARIMA model
            component_models: Component models
            
        Returns:
            Dictionary of components
        """
        # Simplified Burman extraction
        # In practice uses Wiener-Kolmogorov filters
        
        extracted = {}
        
        # Apply filters to extract components
        n = series.length
        
        # Trend: Apply Henderson filter or similar
        trend = self._apply_trend_filter(series)
        extracted['trend'] = trend
        
        # Seasonal: Detrended series filtered
        if 'seasonal' in component_models:
            detrended = TsData.of(series.start, series.values - trend.values)
            seasonal = self._apply_seasonal_filter(detrended)
            extracted['seasonal'] = seasonal
        
        # Irregular: Residual
        irregular = series.values - trend.values
        if 'seasonal' in extracted:
            irregular -= extracted['seasonal'].values
        extracted['irregular'] = TsData.of(series.start, irregular)
        
        return extracted
    
    def _apply_trend_filter(self, series: TsData) -> TsData:
        """Apply trend extraction filter.
        
        Args:
            series: Input series
            
        Returns:
            Trend component
        """
        # Simple moving average for trend
        window = 13  # Henderson-like
        
        from scipy.ndimage import uniform_filter1d
        
        # Apply symmetric filter
        trend_values = uniform_filter1d(series.values, window, mode='nearest')
        
        return TsData.of(series.start, trend_values)
    
    def _apply_seasonal_filter(self, series: TsData) -> TsData:
        """Apply seasonal extraction filter.
        
        Args:
            series: Detrended series
            
        Returns:
            Seasonal component
        """
        period = series.domain.frequency.periods_per_year
        values = series.values
        n = len(values)
        
        # Compute seasonal means
        seasonal_means = np.zeros(period)
        counts = np.zeros(period)
        
        for i in range(n):
            season = i % period
            if not np.isnan(values[i]):
                seasonal_means[season] += values[i]
                counts[season] += 1
        
        # Average
        for s in range(period):
            if counts[s] > 0:
                seasonal_means[s] /= counts[s]
        
        # Center seasonal means
        seasonal_means -= np.mean(seasonal_means)
        
        # Replicate pattern
        seasonal_values = np.zeros(n)
        for i in range(n):
            seasonal_values[i] = seasonal_means[i % period]
        
        return TsData.of(series.start, seasonal_values)
    
    def _generate_seasonal_pattern(self, series: TsData) -> TsData:
        """Generate simple seasonal pattern.
        
        Args:
            series: Input series
            
        Returns:
            Seasonal pattern
        """
        period = series.domain.frequency.periods_per_year
        n = series.length
        
        # Simple sinusoidal pattern
        t = np.arange(n)
        seasonal = 0.1 * np.sin(2 * np.pi * t / period)
        
        return TsData.of(series.start, seasonal)
    
    def _apply_bias_correction(self, results: SeatsResults, series: TsData) -> SeatsResults:
        """Apply bias correction to ensure components sum to series.
        
        Args:
            results: Initial results
            series: Original series
            
        Returns:
            Bias-corrected results
        """
        # Compute sum of components
        component_sum = np.zeros(series.length)
        
        if results.trend is not None:
            component_sum += results.trend.values
        if results.seasonal is not None:
            component_sum += results.seasonal.values
        if results.irregular is not None:
            component_sum += results.irregular.values
        
        # Compute discrepancy
        discrepancy = series.values - component_sum
        
        # Allocate discrepancy to irregular
        if results.irregular is not None:
            corrected_irregular = results.irregular.values + discrepancy
            results.irregular = TsData.of(series.start, corrected_irregular)
        
        return results
    
    def create_decomposition(self, seats_results: SeatsResults,
                           tramo_results: TramoResults) -> SeriesDecomposition:
        """Create full decomposition combining SEATS and TRAMO results.
        
        Args:
            seats_results: SEATS decomposition
            tramo_results: TRAMO pre-adjustment
            
        Returns:
            Complete series decomposition
        """
        # Determine decomposition mode
        if tramo_results.log_transformed:
            mode = DecompositionMode.LOG_ADDITIVE
        else:
            mode = DecompositionMode.ADDITIVE
        
        # Create decomposition
        decomp = SeriesDecomposition(mode=mode)
        
        # Original series
        decomp.series = tramo_results.original
        
        # Components from SEATS
        decomp.trend = seats_results.trend
        decomp.seasonal = seats_results.seasonal
        decomp.irregular = seats_results.irregular
        
        # Seasonally adjusted = Trend + Irregular
        if decomp.trend is not None and decomp.irregular is not None:
            sa_values = decomp.trend.values + decomp.irregular.values
            decomp.seasonally_adjusted = TsData.of(decomp.trend.start, sa_values)
        
        # Add regression effects from TRAMO
        if 'td' in tramo_results.regression_effects:
            decomp.trading_days = tramo_results.regression_effects['td']
        if 'easter' in tramo_results.regression_effects:
            decomp.easter = tramo_results.regression_effects['easter']
        
        # Add outliers
        if tramo_results.outliers:
            outlier_sum = np.zeros(tramo_results.original.length)
            for outlier in tramo_results.outliers:
                effect = tramo_results.regression_effects.get(outlier.name())
                if effect is not None:
                    outlier_sum += effect.values
            decomp.outliers = TsData.of(tramo_results.original.start, outlier_sum)
        
        # Add forecasts
        decomp.forecast = tramo_results.forecasts
        decomp.backcast = tramo_results.backcasts
        
        return decomp