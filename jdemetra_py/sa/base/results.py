"""Results from seasonal adjustment."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

from ...toolkit.timeseries import TsData
from .definition import ComponentType, DecompositionMode


@dataclass
class SeriesDecomposition:
    """Decomposition of time series into components."""
    
    mode: DecompositionMode
    
    # Main components
    series: Optional[TsData] = None
    seasonally_adjusted: Optional[TsData] = None
    trend: Optional[TsData] = None
    seasonal: Optional[TsData] = None
    irregular: Optional[TsData] = None
    
    # Additional components
    calendar: Optional[TsData] = None
    trading_days: Optional[TsData] = None
    easter: Optional[TsData] = None
    outliers: Optional[TsData] = None
    
    # Transformed series
    transformed: Optional[TsData] = None
    
    def get_components(self) -> Dict[ComponentType, TsData]:
        """Get all available components as a dictionary."""
        components = {}
        
        # Add main components
        if self.series is not None:
            components[ComponentType.SERIES] = self.series
        if self.seasonally_adjusted is not None:
            components[ComponentType.SEASONALLY_ADJUSTED] = self.seasonally_adjusted
        if self.trend is not None:
            components[ComponentType.TREND] = self.trend
        if self.seasonal is not None:
            components[ComponentType.SEASONAL] = self.seasonal
        if self.irregular is not None:
            components[ComponentType.IRREGULAR] = self.irregular
            
        # Add additional components
        if self.calendar is not None:
            components[ComponentType.CALENDAR] = self.calendar
        if self.trading_days is not None:
            components[ComponentType.TRADING_DAYS] = self.trading_days
        if self.easter is not None:
            components[ComponentType.EASTER] = self.easter
        if self.outliers is not None:
            components[ComponentType.OUTLIERS] = self.outliers
        if self.transformed is not None:
            components[ComponentType.LOG] = self.transformed
            
        return components
    
    # Forecasts/backcasts
    forecast: Optional[TsData] = None
    backcast: Optional[TsData] = None
    
    def get_component(self, component_type: ComponentType) -> Optional[TsData]:
        """Get component by type.
        
        Args:
            component_type: Type of component
            
        Returns:
            Component series or None
        """
        mapping = {
            ComponentType.SERIES: self.series,
            ComponentType.SEASONALLY_ADJUSTED: self.seasonally_adjusted,
            ComponentType.TREND: self.trend,
            ComponentType.SEASONAL: self.seasonal,
            ComponentType.IRREGULAR: self.irregular,
            ComponentType.CALENDAR: self.calendar,
            ComponentType.TRADING_DAYS: self.trading_days,
            ComponentType.EASTER: self.easter,
            ComponentType.OUTLIERS: self.outliers,
            ComponentType.FORECAST: self.forecast,
            ComponentType.BACKCAST: self.backcast,
        }
        return mapping.get(component_type)
    
    def validate_decomposition(self) -> bool:
        """Validate decomposition consistency."""
        if self.series is None:
            return False
        
        # Check main components exist
        if any(c is None for c in [self.seasonally_adjusted, self.seasonal]):
            return False
        
        # Check decomposition identity
        if self.mode == DecompositionMode.ADDITIVE:
            # Y = SA + S (simplified)
            reconstructed = self.seasonally_adjusted.values + self.seasonal.values
            
            # Add other components if present
            if self.calendar is not None:
                reconstructed += self.calendar.values
            if self.outliers is not None:
                reconstructed += self.outliers.values
            
            # Check identity holds (with tolerance)
            diff = np.abs(self.series.values - reconstructed)
            return np.max(diff) < 1e-6
        
        elif self.mode == DecompositionMode.MULTIPLICATIVE:
            # Y = SA * S (simplified)
            reconstructed = self.seasonally_adjusted.values * self.seasonal.values
            
            # Multiply other components if present
            if self.calendar is not None:
                reconstructed *= self.calendar.values
            if self.outliers is not None:
                reconstructed *= self.outliers.values
            
            # Check identity holds
            ratio = self.series.values / reconstructed
            return np.abs(np.max(ratio) - 1) < 1e-6 and np.abs(np.min(ratio) - 1) < 1e-6
        
        return True


@dataclass
class SaResults:
    """Results from seasonal adjustment processing."""
    
    # Decomposition
    decomposition: SeriesDecomposition
    
    # Pre-adjustment model (e.g., RegARIMA)
    pre_adjustment_model: Optional[Any] = None
    
    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Quality indicators
    quality: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing details
    specification_used: Optional[Any] = None
    processing_log: List[str] = field(default_factory=list)
    
    def add_diagnostic(self, name: str, value: Any):
        """Add diagnostic result.
        
        Args:
            name: Diagnostic name
            value: Diagnostic value
        """
        self.diagnostics[name] = value
    
    def add_quality_indicator(self, name: str, value: float):
        """Add quality indicator.
        
        Args:
            name: Indicator name
            value: Indicator value
        """
        self.quality[name] = value
    
    def get_main_results(self) -> Dict[ComponentType, TsData]:
        """Get main decomposition results.
        
        Returns:
            Dictionary of main components
        """
        results = {}
        
        for comp_type in [ComponentType.SERIES, 
                         ComponentType.SEASONALLY_ADJUSTED,
                         ComponentType.TREND,
                         ComponentType.SEASONAL,
                         ComponentType.IRREGULAR]:
            series = self.decomposition.get_component(comp_type)
            if series is not None:
                results[comp_type] = series
        
        return results
    
    def summary(self) -> str:
        """Get summary of results.
        
        Returns:
            Summary string
        """
        lines = ["Seasonal Adjustment Results", "=" * 30]
        
        # Decomposition mode
        lines.append(f"Decomposition: {self.decomposition.mode.value}")
        
        # Components available
        lines.append("\nComponents:")
        for comp_type in ComponentType:
            if self.decomposition.get_component(comp_type) is not None:
                lines.append(f"  - {comp_type.value}")
        
        # Diagnostics summary
        if self.diagnostics:
            lines.append("\nDiagnostics:")
            for name, value in self.diagnostics.items():
                if isinstance(value, float):
                    lines.append(f"  - {name}: {value:.4f}")
                else:
                    lines.append(f"  - {name}: {value}")
        
        # Quality indicators
        if self.quality:
            lines.append("\nQuality Indicators:")
            for name, value in self.quality.items():
                lines.append(f"  - {name}: {value:.4f}")
        
        return "\n".join(lines)