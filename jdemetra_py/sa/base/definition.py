"""Core definitions for seasonal adjustment."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from jdemetra_py.toolkit.timeseries import TsData, TsDomain, TsPeriod


class ComponentType(Enum):
    """Types of time series components."""
    
    # Main components
    SERIES = "y"
    SEASONALLY_ADJUSTED = "sa"
    TREND = "t"
    SEASONAL = "s"
    IRREGULAR = "i"
    
    # Additional components
    CALENDAR = "cal"
    TRADING_DAYS = "td"
    EASTER = "easter"
    OUTLIERS = "outliers"
    
    # Transformations
    LOG = "log"
    LEVEL = "level"
    
    # Forecasts
    FORECAST = "fcst"
    BACKCAST = "bcst"
    
    # Quality
    RESIDUALS = "residuals"
    
    def __str__(self):
        return self.value


class DecompositionMode(Enum):
    """Decomposition mode."""
    
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    LOG_ADDITIVE = "log_additive"
    PSEUDO_ADDITIVE = "pseudo_additive"
    
    def is_multiplicative(self) -> bool:
        """Check if mode involves multiplication."""
        return self in (DecompositionMode.MULTIPLICATIVE, 
                       DecompositionMode.PSEUDO_ADDITIVE)


@dataclass
class SaDefinition:
    """Seasonal adjustment definition.
    
    Specifies the basic parameters for seasonal adjustment.
    """
    
    domain: TsDomain
    log_transformation: bool = False
    decomposition_mode: DecompositionMode = DecompositionMode.MULTIPLICATIVE
    
    # Calendar effects
    trading_days: bool = True
    easter: bool = True
    
    # Outliers
    outliers: bool = True
    
    # ARIMA modeling
    arima_modeling: bool = True
    
    # Forecasting
    forecast_horizon: Optional[int] = None
    backcast_horizon: Optional[int] = None
    
    # Additional options
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.options is None:
            self.options = {}
        
        # Set default horizons based on frequency
        if self.forecast_horizon is None:
            periods_per_year = self.domain.frequency.periods_per_year
            if periods_per_year == 12:
                self.forecast_horizon = 12
            elif periods_per_year == 4:
                self.forecast_horizon = 4
            else:
                self.forecast_horizon = periods_per_year
        
        if self.backcast_horizon is None:
            self.backcast_horizon = 0
        
        # Validate mode with transformation
        if self.log_transformation and self.decomposition_mode == DecompositionMode.ADDITIVE:
            self.decomposition_mode = DecompositionMode.LOG_ADDITIVE
    
    def validate(self, data: TsData) -> bool:
        """Validate definition against data.
        
        Args:
            data: Time series data
            
        Returns:
            True if valid
        """
        # Check domain compatibility
        if not self.domain.contains(data.domain.start):
            return False
        
        # Check for negative values with log transformation
        if self.log_transformation:
            if np.any(data.values <= 0):
                return False
        
        # Check minimum length
        min_length = 3 * self.domain.frequency.periods_per_year
        if data.length < min_length:
            return False
        
        return True


@dataclass
class TransformationSpec:
    """Transformation specification."""
    
    function: str = "auto"  # "none", "log", "auto"
    fct: float = 1.0  # Forecast comparison threshold
    
    def should_use_log(self, data: TsData) -> bool:
        """Determine if log transformation should be used."""
        if self.function == "log":
            return True
        elif self.function == "none":
            return False
        else:  # auto
            # Simple heuristic based on data characteristics
            from ...toolkit.stats import DescriptiveStatistics
            
            stats = DescriptiveStatistics.compute(data.values)
            
            # Check for negative values
            if stats.min <= 0:
                return False
            
            # Check coefficient of variation
            cv = stats.cv
            if cv > 0.5:  # High variability suggests log
                return True
            
            # Check range ratio
            range_ratio = stats.max / stats.min
            if range_ratio > 10:
                return True
            
            return False


@dataclass 
class OutlierSpec:
    """Outlier detection specification."""
    
    enabled: bool = True
    types: list = None  # ["AO", "LS", "TC", "SO"]
    critical_value: float = 3.5
    
    # Date spans for outlier detection
    span_start: Optional[TsPeriod] = None
    span_end: Optional[TsPeriod] = None
    
    def __post_init__(self):
        if self.types is None:
            self.types = ["AO", "LS", "TC"]