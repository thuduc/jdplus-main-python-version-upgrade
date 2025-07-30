"""TRAMO-SEATS specification."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from ..base.specification import SaSpecification, TransformationSpec, OutlierSpec
from ...toolkit.arima import ArimaOrder


@dataclass
class TramoSpec:
    """TRAMO (pre-adjustment) specification."""
    
    # ARIMA specification
    arima: Optional['ArimaSpec'] = None
    
    # Transformation
    transform: TransformationSpec = None
    
    # Calendar effects
    trading_days: Optional['TradingDaysSpec'] = None
    easter: Optional['EasterSpec'] = None
    
    # Outliers
    outliers: OutlierSpec = None
    
    # Estimation
    estimation_span: Optional['EstimationSpan'] = None
    
    # Forecasting
    fcasts: int = -1  # -1 means automatic
    bcasts: int = 0
    
    def __post_init__(self):
        if self.transform is None:
            self.transform = TransformationSpec()
        if self.outliers is None:
            self.outliers = OutlierSpec()
        if self.arima is None:
            self.arima = ArimaSpec()


@dataclass
class ArimaSpec:
    """ARIMA model specification for TRAMO."""
    
    # Model orders (None means automatic)
    p: Optional[int] = None
    d: Optional[int] = None
    q: Optional[int] = None
    bp: Optional[int] = None  # Seasonal P
    bd: Optional[int] = None  # Seasonal D
    bq: Optional[int] = None  # Seasonal Q
    
    # Mean correction
    mean: bool = True
    
    # Automatic model identification
    auto_model: bool = True
    
    # Model selection criteria
    ub1: float = 0.97  # Unit root boundary for AR
    ub2: float = 0.91  # Unit root boundary for MA
    cancel: float = 0.1  # Cancellation limit
    
    def to_arima_order(self, period: int) -> Optional[ArimaOrder]:
        """Convert to ARIMA order if fully specified."""
        if any(x is None for x in [self.p, self.d, self.q]):
            return None
        
        from ...toolkit.arima import SarimaOrder
        
        if any(x is not None for x in [self.bp, self.bd, self.bq]):
            # Seasonal model
            seasonal_order = ArimaOrder(
                self.bp or 0,
                self.bd or 0,
                self.bq or 0
            )
            return SarimaOrder(
                ArimaOrder(self.p, self.d, self.q),
                seasonal_order,
                period
            )
        else:
            # Non-seasonal
            return ArimaOrder(self.p, self.d, self.q)


@dataclass
class TradingDaysSpec:
    """Trading days specification."""
    
    td_type: str = "TradingDays"  # "TradingDays", "WorkingDays", "None"
    lp_type: str = "LeapYear"  # "LeapYear", "LengthOfPeriod", "None"
    
    # User-defined variables
    user_td: Optional[List[str]] = None
    
    # Test for trading days
    test: bool = True
    pftd: float = 0.01  # P-value threshold


@dataclass
class EasterSpec:
    """Easter specification."""
    
    duration: int = 6  # Duration in days
    test: bool = True  # Test for Easter effect
    
    # Julian Easter (Orthodox)
    julian: bool = False


@dataclass
class EstimationSpan:
    """Span for model estimation."""
    
    start: Optional[str] = None  # Date string or period
    end: Optional[str] = None
    
    # Exclude specific periods
    excludes: Optional[List[str]] = None


@dataclass
class SeatsSpec:
    """SEATS (decomposition) specification."""
    
    # Approximation mode
    approximation: str = "None"  # "None", "Legacy", "Noisy"
    
    # Trend/Cycle separation
    trend_boundary: float = 0.5
    seas_boundary: float = 0.8
    
    # Bias correction
    bias_correction: bool = True
    
    # Method
    method: str = "Burman"  # "Burman", "KalmanSmoother"


@dataclass
class TramoSeatsSpecification(SaSpecification):
    """Complete TRAMO-SEATS specification."""
    
    tramo: TramoSpec = None
    seats: SeatsSpec = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.tramo is None:
            self.tramo = TramoSpec()
        if self.seats is None:
            self.seats = SeatsSpec()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tramo": {
                "transform": {
                    "function": self.tramo.transform.function,
                    "fct": self.tramo.transform.fct
                },
                "arima": {
                    "p": self.tramo.arima.p,
                    "d": self.tramo.arima.d,
                    "q": self.tramo.arima.q,
                    "bp": self.tramo.arima.bp,
                    "bd": self.tramo.arima.bd,
                    "bq": self.tramo.arima.bq,
                    "mean": self.tramo.arima.mean,
                    "auto_model": self.tramo.arima.auto_model
                },
                "outliers": {
                    "enabled": self.tramo.outliers.enabled,
                    "types": self.tramo.outliers.types,
                    "critical_value": self.tramo.outliers.critical_value
                },
                "fcasts": self.tramo.fcasts,
                "bcasts": self.tramo.bcasts
            },
            "seats": {
                "approximation": self.seats.approximation,
                "trend_boundary": self.seats.trend_boundary,
                "seas_boundary": self.seats.seas_boundary,
                "bias_correction": self.seats.bias_correction,
                "method": self.seats.method
            }
        }
    
    def from_dict(self, spec_dict: Dict[str, Any]) -> 'TramoSeatsSpecification':
        """Create from dictionary."""
        tramo_dict = spec_dict.get("tramo", {})
        seats_dict = spec_dict.get("seats", {})
        
        # Parse TRAMO spec
        self.tramo = TramoSpec()
        
        transform_dict = tramo_dict.get("transform", {})
        self.tramo.transform = TransformationSpec(
            function=transform_dict.get("function", "auto"),
            fct=transform_dict.get("fct", 1.0)
        )
        
        arima_dict = tramo_dict.get("arima", {})
        self.tramo.arima = ArimaSpec(
            p=arima_dict.get("p"),
            d=arima_dict.get("d"),
            q=arima_dict.get("q"),
            bp=arima_dict.get("bp"),
            bd=arima_dict.get("bd"),
            bq=arima_dict.get("bq"),
            mean=arima_dict.get("mean", True),
            auto_model=arima_dict.get("auto_model", True)
        )
        
        outlier_dict = tramo_dict.get("outliers", {})
        self.tramo.outliers = OutlierSpec(
            enabled=outlier_dict.get("enabled", True),
            types=outlier_dict.get("types", ["AO", "LS", "TC"]),
            critical_value=outlier_dict.get("critical_value", 3.5)
        )
        
        self.tramo.fcasts = tramo_dict.get("fcasts", -1)
        self.tramo.bcasts = tramo_dict.get("bcasts", 0)
        
        # Parse SEATS spec
        self.seats = SeatsSpec(
            approximation=seats_dict.get("approximation", "None"),
            trend_boundary=seats_dict.get("trend_boundary", 0.5),
            seas_boundary=seats_dict.get("seas_boundary", 0.8),
            bias_correction=seats_dict.get("bias_correction", True),
            method=seats_dict.get("method", "Burman")
        )
        
        return self
    
    def validate(self) -> bool:
        """Validate specification."""
        # Check ARIMA orders
        if not self.tramo.arima.auto_model:
            if any(x is None for x in [self.tramo.arima.p, 
                                       self.tramo.arima.d,
                                       self.tramo.arima.q]):
                return False
        
        # Check boundaries
        if not (0 < self.seats.trend_boundary < 1):
            return False
        if not (0 < self.seats.seas_boundary < 1):
            return False
        
        return True
    
    @classmethod
    def rsa0(cls) -> 'TramoSeatsSpecification':
        """RSA0: No seasonal adjustment."""
        spec = cls()
        spec.tramo.arima.auto_model = False
        spec.tramo.arima.p = 0
        spec.tramo.arima.d = 1  
        spec.tramo.arima.q = 1
        spec.tramo.arima.bp = 0
        spec.tramo.arima.bd = 0
        spec.tramo.arima.bq = 0
        return spec
    
    @classmethod
    def rsa1(cls) -> 'TramoSeatsSpecification':
        """RSA1: Automatic seasonal adjustment."""
        return cls()  # Default is automatic
    
    @classmethod
    def rsa2(cls) -> 'TramoSeatsSpecification':
        """RSA2: Trading days and Easter."""
        spec = cls()
        spec.tramo.trading_days = TradingDaysSpec()
        spec.tramo.easter = EasterSpec()
        return spec
    
    @classmethod 
    def rsa3(cls) -> 'TramoSeatsSpecification':
        """RSA3: Outliers."""
        spec = cls()
        spec.tramo.outliers = OutlierSpec(
            enabled=True,
            types=["AO", "LS", "TC", "SO"]
        )
        return spec
    
    @classmethod
    def rsa4(cls) -> 'TramoSeatsSpecification':
        """RSA4: Trading days, Easter and Outliers."""
        spec = cls.rsa2()  # Trading days and Easter
        spec.tramo.outliers = OutlierSpec(
            enabled=True,
            types=["AO", "LS", "TC", "SO"]
        )
        return spec
    
    @classmethod
    def rsa5(cls) -> 'TramoSeatsSpecification':
        """RSA5: Trading days, Easter, Outliers and log transformation."""
        spec = cls.rsa4()
        spec.tramo.transform.function = "log"
        return spec