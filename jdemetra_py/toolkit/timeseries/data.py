"""Time series data structures."""

from dataclasses import dataclass
from typing import Optional, Union, List, Iterator, Tuple, Callable
import numpy as np
import pandas as pd
from enum import Enum

from .domain import TsDomain, TsPeriod
from .frequency import TsUnit, TsFrequency


class EmptyCause(Enum):
    """Reasons why a time series might be empty."""
    NO_DATA = "NO_DATA"
    UNDEFINED = "UNDEFINED"


@dataclass(frozen=True)
class TsObs:
    """Single time series observation."""
    period: TsPeriod
    value: float
    
    def is_missing(self) -> bool:
        """Check if value is missing (NaN)."""
        return np.isnan(self.value)


class TsData:
    """Time series data container matching JDemetra+ TsData.
    
    This class wraps a pandas Series internally but provides the JDemetra+ API.
    """
    
    def __init__(self, 
                 start: TsPeriod,
                 values: Union[np.ndarray, List[float], pd.Series],
                 empty_cause: Optional[EmptyCause] = None):
        """Initialize time series data.
        
        Args:
            start: Starting period
            values: Time series values
            empty_cause: Reason if series is empty
        """
        self._start = start
        self._empty_cause = empty_cause
        
        # Convert values to numpy array
        if isinstance(values, pd.Series):
            self._values = values.values
        elif isinstance(values, list):
            self._values = np.array(values, dtype=np.float64)
        else:
            self._values = np.asarray(values, dtype=np.float64)
        
        # Create domain
        self._domain = TsDomain.of(start, len(self._values))
        
        # Create internal pandas series for efficient operations
        if len(self._values) > 0:
            index = self._domain.to_pandas_index()
            self._series = pd.Series(self._values, index=index)
        else:
            self._series = pd.Series(dtype=np.float64)
    
    @classmethod
    def of(cls, start: TsPeriod, values: Union[np.ndarray, List[float]]) -> 'TsData':
        """Create TsData from start period and values."""
        return cls(start, values)
    
    @classmethod
    def empty(cls, cause: EmptyCause = EmptyCause.UNDEFINED) -> 'TsData':
        """Create empty TsData."""
        start = TsPeriod(1970, 0, TsFrequency.YEARLY)
        return cls(start, [], cause)
    
    @classmethod
    def from_pandas(cls, series: pd.Series, frequency: Union[TsFrequency, TsUnit]) -> 'TsData':
        """Create TsData from pandas Series with DatetimeIndex."""
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")
        
        if len(series) == 0:
            # Empty series
            start = TsPeriod(1970, 0, TsFrequency(frequency))
            return cls(start, [], EmptyCause.NO_DATA)
        
        # Get start period from first date
        start_date = series.index[0]
        start = TsPeriod.from_datetime(start_date, frequency)
        
        return cls(start, series.values)
    
    @property
    def frequency(self) -> TsFrequency:
        """Get time series frequency."""
        return self._domain.frequency
    
    def get_by_period(self, period: TsPeriod) -> float:
        """Get value at specific period."""
        if not self._domain.contains(period):
            raise ValueError(f"Period {period} not in domain")
        
        index = period.minus(self._domain.start)
        return self._values[index]
    
    def average(self) -> float:
        """Calculate average of non-missing values."""
        clean_values = self._values[~np.isnan(self._values)]
        if len(clean_values) == 0:
            return np.nan
        return np.mean(clean_values)
    
    def sum(self) -> float:
        """Calculate sum of non-missing values."""
        clean_values = self._values[~np.isnan(self._values)]
        if len(clean_values) == 0:
            return np.nan
        return np.sum(clean_values)
    
    def count_missing(self) -> int:
        """Count number of missing values."""
        return np.sum(np.isnan(self._values))
    
    def drop(self, n_begin: int, n_end: int) -> 'TsData':
        """Drop values from beginning and end."""
        if n_begin < 0 or n_end < 0:
            raise ValueError("Drop counts must be non-negative")
        
        if n_begin + n_end >= len(self._values):
            # Return empty series
            new_start = self._domain.start.plus(n_begin)
            return TsData(new_start, np.array([]))
        
        new_values = self._values[n_begin:len(self._values)-n_end if n_end > 0 else None]
        new_start = self._domain.start.plus(n_begin)
        return TsData(new_start, new_values)
    
    @classmethod
    def random(cls, frequency: Union[TsFrequency, TsUnit], seed: int = None) -> 'TsData':
        """Create random time series (matching JDemetra+ API)."""
        if seed is not None:
            np.random.seed(seed)
        
        # Random start between 1970-1990
        start_year = 1970 + np.random.randint(0, 20)
        start_period = 0
        
        # Random length up to 600
        length = np.random.randint(50, 600)
        
        # Generate random walk
        values = np.cumsum(np.random.randn(length) * 0.5) + 100
        
        start = TsPeriod(start_year, start_period, frequency)
        return cls(start, values)
    
    @property
    def domain(self) -> TsDomain:
        """Get time domain."""
        return self._domain
    
    @property
    def start(self) -> TsPeriod:
        """Get start period."""
        return self._start
    
    @property
    def values(self) -> np.ndarray:
        """Get values as numpy array."""
        return self._values.copy()
    
    @property
    def length(self) -> int:
        """Get number of observations."""
        return len(self._values)
    
    @property
    def empty_cause(self) -> Optional[EmptyCause]:
        """Get empty cause if series is empty."""
        return self._empty_cause
    
    def is_empty(self) -> bool:
        """Check if series is empty."""
        return len(self._values) == 0
    
    def get(self, index: int) -> float:
        """Get value at index."""
        if index < 0 or index >= len(self._values):
            raise IndexError(f"Index {index} out of range")
        return self._values[index]
    
    def get_obs(self, index: int) -> TsObs:
        """Get observation at index."""
        period = self._domain.get(index)
        value = self.get(index)
        return TsObs(period, value)
    
    def iterator(self) -> Iterator[TsObs]:
        """Iterate over observations."""
        for i, period in enumerate(self._domain):
            yield TsObs(period, self._values[i])
    
    def __iter__(self) -> Iterator[TsObs]:
        """Make TsData iterable."""
        return self.iterator()
    
    def __getitem__(self, index: int) -> float:
        """Get value at index."""
        return self.get(index)
    
    def __len__(self) -> int:
        """Get length."""
        return self.length
    
    def to_pandas(self) -> pd.Series:
        """Convert to pandas Series."""
        return self._series.copy()
    
    def select(self, selector: TsDomain) -> 'TsData':
        """Select subset by domain."""
        intersection = self._domain.intersection(selector)
        if intersection is None or intersection.is_empty():
            return TsData(selector.start, [], EmptyCause.NO_DATA)
        
        start_idx = self._domain.index_of(intersection.start)
        end_idx = start_idx + intersection.length
        
        new_values = self._values[start_idx:end_idx]
        return TsData(intersection.start, new_values)
    
    def extend(self, n_before: int, n_after: int) -> 'TsData':
        """Extend series with NaN values."""
        if n_before < 0 or n_after < 0:
            raise ValueError("Extensions must be non-negative")
        
        new_start = self._start.plus(-n_before) if n_before > 0 else self._start
        
        # Create extended values array
        new_length = n_before + len(self._values) + n_after
        new_values = np.full(new_length, np.nan)
        new_values[n_before:n_before + len(self._values)] = self._values
        
        return TsData(new_start, new_values)
    
    def drop_missing(self) -> 'TsData':
        """Remove missing values from start and end."""
        if self.is_empty():
            return self
        
        # Find first and last non-NaN
        non_nan_idx = np.where(~np.isnan(self._values))[0]
        
        if len(non_nan_idx) == 0:
            # All NaN
            return TsData(self._start, [], EmptyCause.NO_DATA)
        
        first_idx = non_nan_idx[0]
        last_idx = non_nan_idx[-1]
        
        new_start = self._start.plus(first_idx)
        new_values = self._values[first_idx:last_idx + 1]
        
        return TsData(new_start, new_values)
    
    def clean_extremities(self) -> 'TsData':
        """Alias for drop_missing (JDemetra+ compatibility)."""
        return self.drop_missing()
    
    def fn(self, fn: Callable[[float], float]) -> 'TsData':
        """Apply function to all values."""
        new_values = np.array([fn(v) for v in self._values])
        return TsData(self._start, new_values)
    
    def aggregate(self, target_frequency: Union[TsFrequency, TsUnit], 
                  agg_func: str = 'mean') -> 'TsData':
        """Aggregate to lower frequency."""
        # Convert to pandas for easy aggregation
        series = self.to_pandas()
        
        # Determine aggregation rule
        if isinstance(target_frequency, TsUnit):
            target_frequency = TsFrequency(target_frequency)
        
        freq_map = {
            TsUnit.YEAR: 'Y',
            TsUnit.QUARTER: 'Q',
            TsUnit.MONTH: 'M',
        }
        
        target_freq = freq_map.get(target_frequency.unit)
        if not target_freq:
            raise ValueError(f"Cannot aggregate to {target_frequency}")
        
        # Perform aggregation
        if agg_func == 'mean':
            aggregated = series.resample(target_freq).mean()
        elif agg_func == 'sum':
            aggregated = series.resample(target_freq).sum()
        elif agg_func == 'first':
            aggregated = series.resample(target_freq).first()
        elif agg_func == 'last':
            aggregated = series.resample(target_freq).last()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
        
        return TsData.from_pandas(aggregated, target_frequency)
    
    def lead(self, n: int) -> 'TsData':
        """Lead series by n periods (shift backwards in time)."""
        if n == 0:
            return TsData(self._start, self._values)
        elif n > 0:
            # Lead: add NaN at end, keep same start
            led_values = np.concatenate([self._values[n:], np.full(n, np.nan)])
            return TsData(self._start, led_values)
        else:
            # Negative lead is a lag
            return self.lag(-n)
    
    def lag(self, n: int) -> 'TsData':
        """Lag series by n periods (shift forward in time)."""
        if n == 0:
            return TsData(self._start, self._values)
        elif n > 0:
            # Lag: add NaN at beginning, keep same start
            lagged_values = np.concatenate([np.full(n, np.nan), self._values[:-n]])
            return TsData(self._start, lagged_values)
        else:
            # Negative lag is a lead
            return self.lead(-n)
    
    def delta(self, lag: int = 1) -> 'TsData':
        """Compute differences (y[t] - y[t-lag])."""
        if lag <= 0:
            raise ValueError("Lag must be positive")
        
        diff_values = self._values[lag:] - self._values[:-lag]
        new_start = self._start.plus(lag)
        
        return TsData(new_start, diff_values)
    
    def delta_log(self, lag: int = 1) -> 'TsData':
        """Compute log differences."""
        if lag <= 0:
            raise ValueError("Lag must be positive")
        
        # Take log first
        log_values = np.log(self._values)
        diff_values = log_values[lag:] - log_values[:-lag]
        new_start = self._start.plus(lag)
        
        return TsData(new_start, diff_values)
    
    def pct_change(self, lag: int = 1) -> 'TsData':
        """Compute percentage change."""
        if lag <= 0:
            raise ValueError("Lag must be positive")
        
        pct_values = (self._values[lag:] - self._values[:-lag]) / self._values[:-lag]
        new_start = self._start.plus(lag)
        
        return TsData(new_start, pct_values)
    
    def __repr__(self) -> str:
        if self.is_empty():
            return f"TsData(empty, cause={self._empty_cause})"
        return f"TsData({self._start} to {self._domain.end}, length={self.length})"
    
    def __str__(self) -> str:
        if self.is_empty():
            return "Empty time series"
        
        # Show first and last few values
        n_show = 5
        if self.length <= n_show * 2:
            values_str = ", ".join(f"{v:.4f}" for v in self._values)
        else:
            first = ", ".join(f"{v:.4f}" for v in self._values[:n_show])
            last = ", ".join(f"{v:.4f}" for v in self._values[-n_show:])
            values_str = f"{first}, ..., {last}"
        
        return f"TsData[{self._start} to {self._domain.end}]: [{values_str}]"