"""X-13ARIMA-SEATS specification."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from ..base.specification import SaSpecification, TransformationSpec, OutlierSpec
from ...toolkit.arima import ArimaOrder


@dataclass
class RegArimaSpec:
    """RegARIMA model specification for X-13."""
    
    # ARIMA orders
    arima: Optional['ArimaModelSpec'] = None
    
    # Regression variables
    variables: Optional['VariablesSpec'] = None
    
    # Transformation
    transform: TransformationSpec = None
    
    # Outliers
    outliers: OutlierSpec = None
    
    # Estimation
    estimate: Optional['EstimateSpec'] = None
    
    def __post_init__(self):
        if self.transform is None:
            self.transform = TransformationSpec()
        if self.outliers is None:
            self.outliers = OutlierSpec()


@dataclass
class ArimaModelSpec:
    """ARIMA model specification."""
    
    # Model specification
    model: Optional[str] = None  # e.g., "(0 1 1)(0 1 1)"
    
    # Individual orders (alternative to model string)
    p: Optional[int] = None
    d: Optional[int] = None
    q: Optional[int] = None
    P: Optional[int] = None  # Seasonal P
    D: Optional[int] = None  # Seasonal D
    Q: Optional[int] = None  # Seasonal Q
    
    def to_order(self, period: int) -> Optional[ArimaOrder]:
        """Convert to ARIMA order."""
        if self.model is not None:
            # Parse model string
            return self._parse_model_string(self.model, period)
        elif all(x is not None for x in [self.p, self.d, self.q]):
            from ...toolkit.arima import SarimaOrder
            if any(x is not None for x in [self.P, self.D, self.Q]):
                seasonal_order = ArimaOrder(
                    self.P or 0,
                    self.D or 0,
                    self.Q or 0
                )
                return SarimaOrder(
                    ArimaOrder(self.p, self.d, self.q),
                    seasonal_order,
                    period
                )
            else:
                return ArimaOrder(self.p, self.d, self.q)
        return None
    
    def _parse_model_string(self, model: str, period: int):
        """Parse X-13 model string."""
        # Simplified parsing - in practice would be more robust
        import re
        
        # Match patterns like (p d q)(P D Q)
        match = re.match(r'\((\d+)\s+(\d+)\s+(\d+)\)\((\d+)\s+(\d+)\s+(\d+)\)', model)
        if match:
            p, d, q, P, D, Q = map(int, match.groups())
            from ...toolkit.arima import SarimaOrder
            return SarimaOrder(
                ArimaOrder(p, d, q),
                ArimaOrder(P, D, Q),
                period
            )
        
        # Match non-seasonal pattern (p d q)
        match = re.match(r'\((\d+)\s+(\d+)\s+(\d+)\)', model)
        if match:
            p, d, q = map(int, match.groups())
            return ArimaOrder(p, d, q)
        
        return None


@dataclass
class VariablesSpec:
    """Regression variables specification."""
    
    # Trading days
    td: Optional[str] = None  # "td", "td1coef", etc.
    tdprior: Optional[List[float]] = None
    
    # Easter
    easter: Optional[int] = None  # Duration
    
    # Outliers (predefined)
    outlier: Optional[List[str]] = None  # e.g., ["AO2020.Mar", "LS2020.Apr"]
    
    # User-defined variables
    user: Optional[List[str]] = None
    usertype: Optional[List[str]] = None
    
    # Holiday
    holiday: Optional[str] = None  # "thanksgiving", "christmas", etc.


@dataclass
class EstimateSpec:
    """Estimation specification."""
    
    # Save estimates
    save: Optional[List[str]] = None  # Components to save
    
    # Maximum iterations
    maxiter: int = 1500
    
    # Tolerance
    tol: float = 1e-5
    
    # Exact ML
    exact: str = "ma"  # "ma", "arma", "none"


@dataclass
class X11Spec:
    """X-11 decomposition specification."""
    
    # Mode
    mode: str = "mult"  # "mult", "add", "logadd", "pseudoadd"
    
    # Seasonal MA
    seasonalma: Optional[List[str]] = None  # e.g., ["s3x5", "s3x9", ...]
    
    # Trend MA
    trendma: Optional[int] = None  # Henderson filter length
    
    # Sigma limits
    sigmalim: Optional[List[float]] = None  # Lower and upper sigma limits
    
    # Calendarsigma
    calendarsigma: Optional[str] = None  # "all", "signif", "select"
    
    # Save
    save: Optional[List[str]] = None  # Components to save


@dataclass
class SeatsSpec:
    """SEATS decomposition specification."""
    
    # Save
    save: Optional[List[str]] = None  # Components to save
    
    # Print
    print: Optional[List[str]] = None  # What to print
    
    # Finite sample
    finite: bool = True
    
    # Options
    noadmiss: bool = False  # Don't test for admissibility
    qmax: int = 50  # Max MA order for components


@dataclass
class X13Specification(SaSpecification):
    """Complete X-13ARIMA-SEATS specification."""
    
    # Series
    series: Optional['SeriesSpec'] = None
    
    # RegARIMA
    regression: Optional[RegArimaSpec] = None
    
    # X-11 or SEATS
    x11: Optional[X11Spec] = None
    seats: Optional[SeatsSpec] = None
    
    # Forecast
    forecast: Optional['ForecastSpec'] = None
    
    # Check
    check: Optional['CheckSpec'] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.regression is None:
            self.regression = RegArimaSpec()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        spec_dict = {}
        
        # RegARIMA section
        if self.regression:
            regarima = {}
            
            if self.regression.arima:
                arima = self.regression.arima
                if arima.model:
                    regarima['model'] = arima.model
                else:
                    regarima['arima'] = {
                        'p': arima.p,
                        'd': arima.d,
                        'q': arima.q,
                        'P': arima.P,
                        'D': arima.D,
                        'Q': arima.Q
                    }
            
            if self.regression.variables:
                regarima['variables'] = {
                    'td': self.regression.variables.td,
                    'easter': self.regression.variables.easter,
                    'outlier': self.regression.variables.outlier
                }
            
            if self.regression.transform:
                regarima['transform'] = {
                    'function': self.regression.transform.function
                }
            
            spec_dict['regression'] = regarima
        
        # X-11 section
        if self.x11:
            spec_dict['x11'] = {
                'mode': self.x11.mode,
                'seasonalma': self.x11.seasonalma,
                'trendma': self.x11.trendma
            }
        
        # SEATS section
        if self.seats:
            spec_dict['seats'] = {
                'finite': self.seats.finite,
                'noadmiss': self.seats.noadmiss
            }
        
        return spec_dict
    
    def validate(self) -> bool:
        """Validate specification."""
        # Check that either X-11 or SEATS is specified, not both
        if self.x11 is not None and self.seats is not None:
            return False
        
        # Check ARIMA specification if provided
        if self.regression and self.regression.arima:
            arima = self.regression.arima
            if arima.model is None:
                # Check individual orders
                if any(x is not None and x < 0 for x in 
                       [arima.p, arima.d, arima.q, arima.P, arima.D, arima.Q]):
                    return False
        
        return True
    
    @classmethod
    def rsa1(cls) -> 'X13Specification':
        """RSA1: Default automatic seasonal adjustment."""
        spec = cls()
        spec.regression = RegArimaSpec(
            arima=ArimaModelSpec(model="(0 1 1)(0 1 1)"),
            transform=TransformationSpec(function="auto"),
            outliers=OutlierSpec(enabled=True)
        )
        spec.x11 = X11Spec()
        return spec
    
    @classmethod
    def rsa2c(cls) -> 'X13Specification':
        """RSA2c: With calendar effects."""
        spec = cls.rsa1()
        spec.regression.variables = VariablesSpec(
            td="td",
            easter=8
        )
        return spec
    
    @classmethod
    def rsa3(cls) -> 'X13Specification':
        """RSA3: SEATS decomposition."""
        spec = cls()
        spec.regression = RegArimaSpec(
            arima=ArimaModelSpec(model="(0 1 1)(0 1 1)"),
            transform=TransformationSpec(function="auto"),
            outliers=OutlierSpec(enabled=True)
        )
        spec.seats = SeatsSpec()
        return spec
    
    @classmethod
    def rsa4c(cls) -> 'X13Specification':
        """RSA4c: SEATS with calendar effects."""
        spec = cls.rsa3()
        spec.regression.variables = VariablesSpec(
            td="td",
            easter=8
        )
        return spec
    
    @classmethod
    def rsa5c(cls) -> 'X13Specification':
        """RSA5c: Full automatic with calendar."""
        spec = cls()
        spec.regression = RegArimaSpec(
            transform=TransformationSpec(function="auto"),
            outliers=OutlierSpec(enabled=True),
            variables=VariablesSpec(td="td", easter=8)
        )
        spec.x11 = X11Spec()
        return spec


@dataclass
class SeriesSpec:
    """Series specification."""
    
    # Data
    data: Optional[List[float]] = None
    
    # Start date
    start: Optional[str] = None  # e.g., "1990.1" or "1990.jan"
    
    # Period
    period: Optional[int] = None
    
    # Title
    title: Optional[str] = None
    
    # Name
    name: Optional[str] = None


@dataclass
class ForecastSpec:
    """Forecast specification."""
    
    # Maximum forecast
    maxlead: Optional[int] = None
    
    # Maximum backcast
    maxback: Optional[int] = None
    
    # Save forecasts
    save: Optional[List[str]] = None
    
    # Probability
    probability: float = 0.95


@dataclass
class CheckSpec:
    """Check specification."""
    
    # Maximum lag
    maxlag: Optional[int] = None
    
    # Print checks
    print: Optional[List[str]] = None
    
    # Save checks
    save: Optional[List[str]] = None