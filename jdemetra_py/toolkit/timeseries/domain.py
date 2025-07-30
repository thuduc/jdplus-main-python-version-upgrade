"""Time series domain and period classes."""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Union, Optional, Iterator, Tuple
import pandas as pd
import numpy as np

from .frequency import TsUnit, TsFrequency


@dataclass(frozen=True)
class TsPeriod:
    """Represents a time period in a regular time series."""
    
    year: int
    period: int
    frequency: TsFrequency
    
    def __init__(self, year: int, period: int, frequency: Union[TsFrequency, TsUnit, int]):
        """Initialize time period.
        
        Args:
            year: Year of the period
            period: Period within the year (0-based)
            frequency: Time series frequency
        """
        if isinstance(frequency, (TsUnit, int)):
            frequency = TsFrequency(frequency)
        
        object.__setattr__(self, 'year', year)
        object.__setattr__(self, 'period', period)
        object.__setattr__(self, 'frequency', frequency)
        
        # Validate period
        if period < 0 or period >= frequency.periods_per_year:
            raise ValueError(f"Period {period} out of range for frequency {frequency}")
    
    @property
    def position(self) -> int:
        """Alias for period (position within year)."""
        return self.period
    
    def minus(self, other: 'TsPeriod') -> int:
        """Calculate the number of periods between two TsPeriods.
        
        Args:
            other: The other TsPeriod
            
        Returns:
            Number of periods from other to self
        """
        if self.frequency != other.frequency:
            raise ValueError("Cannot subtract periods with different frequencies")
        
        self_epoch = self.year * self.frequency.periods_per_year + self.period
        other_epoch = other.year * other.frequency.periods_per_year + other.period
        return self_epoch - other_epoch
    
    def start_date(self) -> date:
        """Get the start date of this period."""
        return self.to_datetime().date()
    
    def to_date(self) -> date:
        """Get date representation of period."""
        return self.to_datetime().date()
    
    def end_date(self) -> date:
        """Get end date of this period."""
        # Get next period start and subtract one day
        next_period = self.plus(1)
        next_start = next_period.to_datetime()
        end = next_start - pd.Timedelta(days=1)
        return end.date()
    
    def display(self) -> str:
        """Get display string for the period."""
        if self.frequency.unit == TsUnit.MONTH:
            return f"{self.year:04d}-{self.period + 1:02d}"
        elif self.frequency.unit == TsUnit.QUARTER:
            return f"{self.year:04d}-Q{self.period + 1}"
        elif self.frequency.unit == TsUnit.YEAR:
            return f"{self.year:04d}"
        else:
            return f"{self.year}-{self.period}"
    
    @classmethod
    def of(cls, frequency: Union[TsFrequency, TsUnit, int], year: int, position: int = None) -> 'TsPeriod':
        """Create TsPeriod from frequency, year and position.
        
        Args:
            frequency: Time series frequency
            year: Year
            position: Position within year (0-based). If None, treated as epoch_period.
        """
        if isinstance(frequency, (TsUnit, int)):
            frequency = TsFrequency(frequency)
        
        if position is None:
            # Legacy behavior - year is actually epoch_period
            epoch_period = year
            periods_per_year = frequency.periods_per_year
            year = epoch_period // periods_per_year
            position = epoch_period % periods_per_year
        
        return cls(year, position, frequency)
    
    @classmethod
    def from_epoch(cls, frequency: Union[TsFrequency, TsUnit, int], epoch_period: int) -> 'TsPeriod':
        """Create TsPeriod from epoch period (periods since epoch).
        
        Args:
            frequency: Time series frequency
            epoch_period: Number of periods since epoch (year 0)
        """
        if isinstance(frequency, (TsUnit, int)):
            frequency = TsFrequency(frequency)
        
        periods_per_year = frequency.periods_per_year
        year = epoch_period // periods_per_year
        period = epoch_period % periods_per_year
        
        return cls(year, period, frequency)
    
    @classmethod
    def from_datetime(cls, dt: Union[datetime, date, pd.Timestamp], 
                     frequency: Union[TsFrequency, TsUnit]) -> 'TsPeriod':
        """Create TsPeriod from datetime."""
        if isinstance(frequency, TsUnit):
            frequency = TsFrequency(frequency)
        
        year = dt.year
        
        if frequency.unit == TsUnit.YEAR:
            period = 0
        elif frequency.unit == TsUnit.QUARTER:
            period = (dt.month - 1) // 3
        elif frequency.unit == TsUnit.MONTH:
            period = dt.month - 1
        elif frequency.unit == TsUnit.WEEK:
            # ISO week number
            period = dt.isocalendar()[1] - 1
        elif frequency.unit == TsUnit.DAY:
            period = dt.timetuple().tm_yday - 1
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        return cls(year, period, frequency)
    
    @property
    def epoch_period(self) -> int:
        """Get epoch period (periods since year 0)."""
        return self.year * self.frequency.periods_per_year + self.period
    
    def to_datetime(self) -> pd.Timestamp:
        """Convert to pandas Timestamp."""
        if self.frequency.unit == TsUnit.YEAR:
            # For YE, return end of year
            return pd.Timestamp(year=self.year, month=12, day=31)
        elif self.frequency.unit == TsUnit.QUARTER:
            # For QE-DEC, need to return end of quarter dates
            quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
            month, day = quarter_ends[self.period]
            return pd.Timestamp(year=self.year, month=month, day=day)
        elif self.frequency.unit == TsUnit.MONTH:
            return pd.Timestamp(year=self.year, month=self.period + 1, day=1)
        elif self.frequency.unit == TsUnit.WEEK:
            # Approximate - first day of year plus weeks
            base = pd.Timestamp(year=self.year, month=1, day=1)
            return base + pd.Timedelta(weeks=self.period)
        elif self.frequency.unit == TsUnit.DAY:
            base = pd.Timestamp(year=self.year, month=1, day=1)
            return base + pd.Timedelta(days=self.period)
        else:
            raise ValueError(f"Cannot convert frequency {self.frequency} to datetime")
    
    def next(self) -> 'TsPeriod':
        """Get next period."""
        next_period = self.period + 1
        next_year = self.year
        
        if next_period >= self.frequency.periods_per_year:
            next_period = 0
            next_year += 1
        
        return TsPeriod(next_year, next_period, self.frequency)
    
    def previous(self) -> 'TsPeriod':
        """Get previous period."""
        prev_period = self.period - 1
        prev_year = self.year
        
        if prev_period < 0:
            prev_period = self.frequency.periods_per_year - 1
            prev_year -= 1
        
        return TsPeriod(prev_year, prev_period, self.frequency)
    
    def plus(self, n: int) -> 'TsPeriod':
        """Add n periods."""
        if n == 0:
            return self
        elif n > 0:
            result = self
            for _ in range(n):
                result = result.next()
            return result
        else:
            result = self
            for _ in range(-n):
                result = result.previous()
            return result
    
    def __str__(self) -> str:
        """String representation."""
        if self.frequency.unit == TsUnit.YEAR:
            return str(self.year)
        elif self.frequency.unit == TsUnit.QUARTER:
            return f"{self.year}Q{self.period + 1}"
        elif self.frequency.unit == TsUnit.MONTH:
            return f"{self.year}-{self.period + 1:02d}"
        else:
            return f"{self.year}P{self.period + 1}"
    
    def __repr__(self) -> str:
        return f"TsPeriod({self.year}, {self.period}, {self.frequency.unit.name})"


@dataclass(frozen=True)
class TsDomain:
    """Represents the time domain of a regular time series."""
    
    start: TsPeriod
    length: int
    
    @classmethod
    def of(cls, start: TsPeriod, length: int) -> 'TsDomain':
        """Create domain from start period and length."""
        return cls(start, length)
    
    @classmethod
    def range(cls, start: TsPeriod, end: TsPeriod) -> 'TsDomain':
        """Create time domain from start to end period (inclusive)."""
        if start.frequency != end.frequency:
            raise ValueError("Start and end must have same frequency")
        
        length = end.minus(start) + 1
        if length < 0:
            raise ValueError("End must be after or equal to start")
        
        return cls(start, length)
    
    @property
    def end(self) -> TsPeriod:
        """Get last period (inclusive)."""
        if self.length == 0:
            raise ValueError("Empty domain has no end")
        return self.start.plus(self.length - 1)
    
    @property
    def frequency(self) -> TsFrequency:
        """Get frequency of the domain."""
        return self.start.frequency
    
    def is_empty(self) -> bool:
        """Check if domain is empty."""
        return self.length == 0
    
    def contains(self, period: TsPeriod) -> bool:
        """Check if period is in domain."""
        if period.frequency != self.frequency:
            return False
        if self.is_empty():
            return False
        
        epoch = period.epoch_period
        start_epoch = self.start.epoch_period
        end_epoch = start_epoch + self.length
        
        return start_epoch <= epoch < end_epoch
    
    def index_of(self, period: TsPeriod) -> int:
        """Get index of period in domain (-1 if not found)."""
        if not self.contains(period):
            return -1
        return period.epoch_period - self.start.epoch_period
    
    def get(self, index: int) -> TsPeriod:
        """Get period at index."""
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} out of range [0, {self.length})")
        return self.start.plus(index)
    
    def intersection(self, other: 'TsDomain') -> Optional['TsDomain']:
        """Get intersection of two domains."""
        if self.frequency != other.frequency:
            return None
        
        start_epoch = max(self.start.epoch_period, other.start.epoch_period)
        end_epoch = min(self.start.epoch_period + self.length, 
                       other.start.epoch_period + other.length)
        
        if start_epoch >= end_epoch:
            return None
        
        new_start = TsPeriod.of(self.frequency, start_epoch)
        new_length = end_epoch - start_epoch
        
        return TsDomain.of(new_start, new_length)
    
    def extend(self, n_before: int, n_after: int) -> 'TsDomain':
        """Extend domain by n periods before and after."""
        if n_before < 0 or n_after < 0:
            raise ValueError("Extensions must be non-negative")
        
        new_start = self.start.plus(-n_before) if n_before > 0 else self.start
        new_length = self.length + n_before + n_after
        
        return TsDomain.of(new_start, new_length)
    
    def __iter__(self) -> Iterator[TsPeriod]:
        """Iterate over periods in domain."""
        current = self.start
        for _ in range(self.length):
            yield current
            current = current.next()
    
    def to_pandas_index(self) -> pd.DatetimeIndex:
        """Convert to pandas DatetimeIndex."""
        dates = [period.to_datetime() for period in self]
        return pd.DatetimeIndex(dates, freq=self.frequency.pandas_freq)
    
    def __repr__(self) -> str:
        if self.is_empty():
            return f"TsDomain(empty, {self.frequency.unit.name})"
        return f"TsDomain({self.start} to {self.end}, length={self.length})"