"""Time series frequency definitions."""

from enum import Enum
from typing import Optional, Union

class TsUnit(Enum):
    """Time series unit enumeration matching JDemetra+ TsUnit."""
    
    YEAR = 1
    HALF_YEAR = 2
    QUARTER = 4
    MONTH = 12
    WEEK = 52
    DAY = 365
    
    # Additional values
    QUARTER_3 = 3
    MONTH_2 = 6
    UNDEFINED = 0
    
    @property
    def frequency(self) -> int:
        """Get the numerical frequency value."""
        return self.value
    
    @property
    def pandas_freq(self) -> Optional[str]:
        """Get corresponding pandas frequency string."""
        mapping = {
            TsUnit.YEAR: 'YE',
            TsUnit.QUARTER: 'QE-DEC',  # Quarter end December
            TsUnit.MONTH: 'MS',  # Month start to avoid conflicts
            TsUnit.WEEK: 'W',
            TsUnit.DAY: 'D',
        }
        return mapping.get(self)
    
    @classmethod
    def from_frequency(cls, freq: int) -> 'TsUnit':
        """Create TsUnit from numerical frequency."""
        for unit in cls:
            if unit.value == freq:
                return unit
        raise ValueError(f"Unknown frequency: {freq}")
    
    @classmethod
    def from_pandas_freq(cls, freq: str) -> 'TsUnit':
        """Create TsUnit from pandas frequency string."""
        freq = freq.upper()
        mapping = {
            'Y': cls.YEAR,
            'A': cls.YEAR,
            'Q': cls.QUARTER,
            'M': cls.MONTH,
            'W': cls.WEEK,
            'D': cls.DAY,
        }
        if freq in mapping:
            return mapping[freq]
        raise ValueError(f"Unknown pandas frequency: {freq}")


class TsFrequency:
    """Enhanced frequency representation with period information."""
    
    # Common frequencies as class attributes
    YEARLY = None
    HALF_YEARLY = None  
    QUARTERLY = None
    MONTHLY = None
    WEEKLY = None
    DAILY = None
    UNDEFINED = None
    
    def __init__(self, unit: Union[TsUnit, int], periods_per_year: Optional[int] = None):
        """Initialize frequency.
        
        Args:
            unit: TsUnit enum or frequency integer
            periods_per_year: Override periods per year (optional)
        """
        if isinstance(unit, int):
            unit = TsUnit.from_frequency(unit)
        self.unit = unit
        self._periods_per_year = periods_per_year or unit.frequency
    
    @property
    def periods_per_year(self) -> int:
        """Number of periods in a year."""
        return self._periods_per_year
    
    @property
    def pandas_freq(self) -> Optional[str]:
        """Get pandas frequency string."""
        return self.unit.pandas_freq
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TsFrequency):
            return False
        return self.unit == other.unit and self._periods_per_year == other._periods_per_year
    
    def __repr__(self) -> str:
        return f"TsFrequency({self.unit.name}, periods_per_year={self._periods_per_year})"


# Initialize common frequencies
TsFrequency.YEARLY = TsFrequency(TsUnit.YEAR)
TsFrequency.HALF_YEARLY = TsFrequency(TsUnit.HALF_YEAR)
TsFrequency.QUARTERLY = TsFrequency(TsUnit.QUARTER)
TsFrequency.MONTHLY = TsFrequency(TsUnit.MONTH)
TsFrequency.WEEKLY = TsFrequency(TsUnit.WEEK)
TsFrequency.DAILY = TsFrequency(TsUnit.DAY)
TsFrequency.UNDEFINED = TsFrequency(TsUnit.UNDEFINED)