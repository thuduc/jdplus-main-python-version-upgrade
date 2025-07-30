"""Regression variables for time series models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union
import numpy as np
import pandas as pd
from datetime import date

from ..timeseries.domain import TsDomain, TsPeriod
from ..timeseries.data import TsData


class TsVariable(ABC):
    """Abstract base class for time series regression variables."""
    
    @abstractmethod
    def name(self) -> str:
        """Get variable name."""
        pass
    
    @abstractmethod
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get variable values for given domain.
        
        Args:
            domain: Time domain
            
        Returns:
            Array of variable values
        """
        pass
    
    @abstractmethod
    def dim(self) -> int:
        """Get dimension of variable (number of columns)."""
        pass


class TrendConstant(TsVariable):
    """Constant (intercept) variable."""
    
    def name(self) -> str:
        return "const"
    
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get constant values (all ones)."""
        return np.ones((domain.length, 1))
    
    def dim(self) -> int:
        return 1


class Trend(TsVariable):
    """Linear trend variable."""
    
    def __init__(self, start_value: float = 0.0):
        """Initialize trend.
        
        Args:
            start_value: Starting value of trend
        """
        self.start_value = start_value
    
    def name(self) -> str:
        return "trend"
    
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get trend values."""
        values = np.arange(domain.length, dtype=float) + self.start_value
        return values.reshape(-1, 1)
    
    def dim(self) -> int:
        return 1


class Seasonal(TsVariable):
    """Seasonal dummy variables."""
    
    def __init__(self, period: int, contrast: bool = True):
        """Initialize seasonal dummies.
        
        Args:
            period: Seasonal period
            contrast: If True, use contrasts (period-1 dummies)
        """
        self.period = period
        self.contrast = contrast
    
    def name(self) -> str:
        return f"seasonal[{self.period}]"
    
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get seasonal dummy variables."""
        n = domain.length
        n_dummies = self.period - 1 if self.contrast else self.period
        
        dummies = np.zeros((n, n_dummies))
        
        for i, period in enumerate(domain):
            season = period.period % self.period
            if self.contrast:
                # Contrasts: season s-1 is reference (all -1)
                if season < self.period - 1:
                    dummies[i, season] = 1
                else:
                    dummies[i, :] = -1
            else:
                # Full dummies
                dummies[i, season] = 1
        
        return dummies
    
    def dim(self) -> int:
        return self.period - 1 if self.contrast else self.period


class TradingDays(TsVariable):
    """Trading days variables."""
    
    def __init__(self, contrast: bool = True, 
                 include_length_of_month: bool = False,
                 include_leap_year: bool = False):
        """Initialize trading days.
        
        Args:
            contrast: If True, use contrasts (6 variables)
            include_length_of_month: Include length of month/quarter variable
            include_leap_year: Include leap year variable
        """
        self.contrast = contrast
        self.include_lom = include_length_of_month
        self.include_ly = include_leap_year
    
    def name(self) -> str:
        return "td"
    
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get trading days variables."""
        n = domain.length
        
        # Basic TD: 6 or 7 variables for days of week
        n_td = 6 if self.contrast else 7
        n_vars = n_td
        
        if self.include_lom:
            n_vars += 1
        if self.include_ly:
            n_vars += 1
        
        values = np.zeros((n, n_vars))
        
        # Compute trading days for each period
        for i, period in enumerate(domain):
            # Get date range for period
            start_date = period.to_datetime()
            end_date = period.next().to_datetime()
            
            # Count days of week in period
            date_range = pd.date_range(start_date, end_date, inclusive='left')
            day_counts = np.zeros(7)
            for day in range(7):
                day_counts[day] = (date_range.dayofweek == day).sum()
            
            # Fill TD variables
            if self.contrast:
                # Contrasts: Mon-Sat vs Sunday
                for j in range(6):
                    values[i, j] = day_counts[j] - day_counts[6]
            else:
                # Full dummies
                for j in range(7):
                    values[i, j] = day_counts[j]
            
            # Length of month/quarter
            col_idx = n_td
            if self.include_lom:
                values[i, col_idx] = len(date_range)
                col_idx += 1
            
            # Leap year
            if self.include_ly:
                is_leap = start_date.year % 4 == 0 and (
                    start_date.year % 100 != 0 or start_date.year % 400 == 0
                )
                # Only affects February
                if start_date.month == 2:
                    values[i, col_idx] = 1 if is_leap else 0
        
        return values
    
    def dim(self) -> int:
        n = 6 if self.contrast else 7
        if self.include_lom:
            n += 1
        if self.include_ly:
            n += 1
        return n


class Easter(TsVariable):
    """Easter effect variable."""
    
    def __init__(self, duration: int = 6):
        """Initialize Easter variable.
        
        Args:
            duration: Duration of Easter effect in days before Easter Sunday
        """
        self.duration = duration
    
    def name(self) -> str:
        return f"easter[{self.duration}]"
    
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get Easter effect values."""
        n = domain.length
        values = np.zeros((n, 1))
        
        # Compute Easter dates for relevant years
        years = set()
        for period in domain:
            years.add(period.year)
            # Also check previous/next year for spillover
            years.add(period.year - 1)
            years.add(period.year + 1)
        
        easter_dates = {}
        for year in years:
            easter_dates[year] = self._compute_easter_date(year)
        
        # Compute Easter effect for each period
        for i, period in enumerate(domain):
            # Get date range for period
            start_date = period.to_datetime().date()
            end_date = period.next().to_datetime().date()
            
            # Check Easter dates that might affect this period
            for year in [period.year - 1, period.year, period.year + 1]:
                if year in easter_dates:
                    easter = easter_dates[year]
                    
                    # Easter effect window
                    effect_start = pd.Timestamp(easter) - pd.Timedelta(days=self.duration - 1)
                    effect_end = pd.Timestamp(easter) + pd.Timedelta(days=1)
                    
                    # Count days in period that fall in Easter window
                    period_start = pd.Timestamp(start_date)
                    period_end = pd.Timestamp(end_date)
                    
                    overlap_start = max(effect_start, period_start)
                    overlap_end = min(effect_end, period_end)
                    
                    if overlap_start < overlap_end:
                        overlap_days = (overlap_end - overlap_start).days
                        # Normalize by period length
                        period_days = (period_end - period_start).days
                        values[i, 0] += overlap_days / period_days
        
        return values
    
    def dim(self) -> int:
        return 1
    
    def _compute_easter_date(self, year: int) -> date:
        """Compute Easter date using Meeus algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        
        return date(year, month, day)


class OutlierType(Enum):
    """Types of outliers."""
    AO = "AO"  # Additive outlier
    LS = "LS"  # Level shift
    TC = "TC"  # Temporary change
    SO = "SO"  # Seasonal outlier


@dataclass
class Outlier(TsVariable):
    """Outlier variable."""
    
    position: TsPeriod
    type: OutlierType
    delta: float = 0.7  # For TC type
    
    def name(self) -> str:
        return f"{self.type.value}({self.position})"
    
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get outlier values."""
        n = domain.length
        values = np.zeros((n, 1))
        
        # Find position in domain
        idx = domain.index_of(self.position)
        if idx < 0:
            return values
        
        if self.type == OutlierType.AO:
            # Additive outlier: single spike
            values[idx, 0] = 1
            
        elif self.type == OutlierType.LS:
            # Level shift: step function
            values[idx:, 0] = 1
            
        elif self.type == OutlierType.TC:
            # Temporary change: exponential decay
            for i in range(idx, n):
                values[i, 0] = self.delta ** (i - idx)
                
        elif self.type == OutlierType.SO:
            # Seasonal outlier: affects same season in subsequent periods
            period = domain.frequency.periods_per_year
            season = self.position.period
            for i in range(idx, n):
                if domain.get(i).period == season:
                    values[i, 0] = 1
        
        return values
    
    def dim(self) -> int:
        return 1


@dataclass
class InterventionVariable(TsVariable):
    """General intervention variable."""
    
    start: Optional[TsPeriod] = None
    end: Optional[TsPeriod] = None
    delta: float = 1.0
    seasonal_delta: float = 1.0
    
    def name(self) -> str:
        return "intervention"
    
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get intervention values."""
        n = domain.length
        values = np.zeros((n, 1))
        
        # Determine affected periods
        start_idx = 0
        if self.start:
            idx = domain.index_of(self.start)
            if idx >= 0:
                start_idx = idx
        
        end_idx = n
        if self.end:
            idx = domain.index_of(self.end)
            if idx >= 0:
                end_idx = idx + 1
        
        # Apply intervention
        for i in range(start_idx, end_idx):
            values[i, 0] = self.delta ** (i - start_idx)
        
        return values
    
    def dim(self) -> int:
        return 1


class UserVariable(TsVariable):
    """User-defined regression variable."""
    
    def __init__(self, name: str, data: Union[TsData, np.ndarray]):
        """Initialize user variable.
        
        Args:
            name: Variable name
            data: Variable values (TsData or array)
        """
        self._name = name
        if isinstance(data, TsData):
            self.data = data
            self.is_ts = True
        else:
            self.data = np.asarray(data)
            self.is_ts = False
    
    def name(self) -> str:
        return self._name
    
    def get_values(self, domain: TsDomain) -> np.ndarray:
        """Get variable values."""
        if self.is_ts:
            # Extract values for domain
            subset = self.data.select(domain)
            values = subset.values
        else:
            # Use array directly
            if len(self.data) < domain.length:
                raise ValueError(f"Insufficient data for variable {self._name}")
            values = self.data[:domain.length]
        
        # Ensure 2D
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        
        return values
    
    def dim(self) -> int:
        if self.is_ts:
            return 1
        else:
            return 1 if self.data.ndim == 1 else self.data.shape[1]