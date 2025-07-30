"""Base specification for seasonal adjustment methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .definition import SaDefinition, TransformationSpec, OutlierSpec


@dataclass
class SaSpecification(ABC):
    """Abstract base class for seasonal adjustment specifications."""
    
    # Basic options
    transformation: TransformationSpec = None
    outliers: OutlierSpec = None
    
    # Calendar effects
    trading_days_spec: Optional[Dict[str, Any]] = None
    easter_spec: Optional[Dict[str, Any]] = None
    
    # User-defined variables
    user_variables: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.transformation is None:
            self.transformation = TransformationSpec()
        if self.outliers is None:
            self.outliers = OutlierSpec()
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary."""
        pass
    
    @abstractmethod
    def from_dict(self, spec_dict: Dict[str, Any]) -> 'SaSpecification':
        """Create specification from dictionary."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate specification consistency."""
        pass
    
    def create_definition(self, domain: 'TsDomain') -> SaDefinition:
        """Create SA definition from specification.
        
        Args:
            domain: Time domain
            
        Returns:
            SA definition
        """
        return SaDefinition(
            domain=domain,
            log_transformation=self.transformation.function == "log",
            trading_days=self.trading_days_spec is not None,
            easter=self.easter_spec is not None,
            outliers=self.outliers.enabled
        )


@dataclass
class BasicSpec(SaSpecification):
    """Basic seasonal adjustment specification."""
    
    # Pre-adjustment
    pre_adjustment: bool = True
    
    # Decomposition
    decomposition_scheme: str = "multiplicative"  # or "additive"
    
    # Filters
    seasonal_filter: Optional[str] = None
    henderson_filter: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transformation": {
                "function": self.transformation.function,
                "fct": self.transformation.fct
            },
            "outliers": {
                "enabled": self.outliers.enabled,
                "types": self.outliers.types,
                "critical_value": self.outliers.critical_value
            },
            "pre_adjustment": self.pre_adjustment,
            "decomposition_scheme": self.decomposition_scheme,
            "seasonal_filter": self.seasonal_filter,
            "henderson_filter": self.henderson_filter
        }
    
    def from_dict(self, spec_dict: Dict[str, Any]) -> 'BasicSpec':
        """Create from dictionary."""
        self.transformation = TransformationSpec(
            function=spec_dict.get("transformation", {}).get("function", "auto"),
            fct=spec_dict.get("transformation", {}).get("fct", 1.0)
        )
        
        outlier_dict = spec_dict.get("outliers", {})
        self.outliers = OutlierSpec(
            enabled=outlier_dict.get("enabled", True),
            types=outlier_dict.get("types", ["AO", "LS", "TC"]),
            critical_value=outlier_dict.get("critical_value", 3.5)
        )
        
        self.pre_adjustment = spec_dict.get("pre_adjustment", True)
        self.decomposition_scheme = spec_dict.get("decomposition_scheme", "multiplicative")
        self.seasonal_filter = spec_dict.get("seasonal_filter")
        self.henderson_filter = spec_dict.get("henderson_filter")
        
        return self
    
    def validate(self) -> bool:
        """Validate specification."""
        # Check decomposition scheme
        if self.decomposition_scheme not in ["multiplicative", "additive"]:
            return False
        
        # Check Henderson filter length
        if self.henderson_filter is not None:
            if self.henderson_filter < 3 or self.henderson_filter % 2 == 0:
                return False
        
        return True