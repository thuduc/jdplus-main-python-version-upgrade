"""Calendar definitions for regression variables."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Set
import numpy as np
import pandas as pd


class Holiday(ABC):
    """Abstract base class for holidays."""
    
    def __init__(self, name: str, weight: float = 1.0, offset: int = 0):
        self.name = name
        self.weight = weight
        self.offset = offset  # Days offset from actual date
    
    @abstractmethod
    def get_dates(self, start_year: int, end_year: int) -> List[date]:
        """Get holiday dates for given year range.
        
        Args:
            start_year: First year
            end_year: Last year (inclusive)
            
        Returns:
            List of holiday dates
        """
        pass
    
    def get_effective_dates(self, start_year: int, end_year: int) -> List[date]:
        """Get effective dates (with offset applied)."""
        dates = self.get_dates(start_year, end_year)
        if self.offset != 0:
            dates = [d + timedelta(days=self.offset) for d in dates]
        return dates


class FixedHoliday(Holiday):
    """Fixed date holiday (e.g., Christmas)."""
    
    def __init__(self, name: str, month: int, day: int, weight: float = 1.0, offset: int = 0):
        super().__init__(name, weight, offset)
        self.month = month
        self.day = day
    
    def get_dates(self, start_year: int, end_year: int) -> List[date]:
        """Get fixed holiday dates."""
        dates = []
        for year in range(start_year, end_year + 1):
            try:
                dates.append(date(year, self.month, self.day))
            except ValueError:
                # Invalid date (e.g., Feb 29 in non-leap year)
                pass
        return dates


class EasterRelatedHoliday(Holiday):
    """Holiday related to Easter (e.g., Good Friday)."""
    
    def __init__(self, name: str, easter_offset: int, weight: float = 1.0, offset: int = 0):
        super().__init__(name, weight, offset)
        self.easter_offset = easter_offset  # Days from Easter Sunday
    
    def get_dates(self, start_year: int, end_year: int) -> List[date]:
        """Get Easter-related holiday dates."""
        dates = []
        for year in range(start_year, end_year + 1):
            easter = self._compute_easter_date(year)
            holiday = easter + timedelta(days=self.easter_offset)
            dates.append(holiday)
        return dates
    
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


class Calendar:
    """Calendar for computing working days and holidays."""
    
    def __init__(self, holidays: Optional[List[Holiday]] = None):
        """Initialize calendar.
        
        Args:
            holidays: List of holidays
        """
        self.holidays = holidays or []
    
    def get_holiday_dates(self, start_year: int, end_year: int) -> Set[date]:
        """Get all holiday dates in range."""
        all_dates = set()
        for holiday in self.holidays:
            dates = holiday.get_effective_dates(start_year, end_year)
            all_dates.update(dates)
        return all_dates
    
    def is_holiday(self, d: date) -> bool:
        """Check if date is a holiday."""
        # This is inefficient for many queries - in practice, cache holiday dates
        holiday_dates = self.get_holiday_dates(d.year, d.year)
        return d in holiday_dates
    
    def is_weekend(self, d: date) -> bool:
        """Check if date is weekend (Saturday or Sunday)."""
        return d.weekday() in (5, 6)
    
    def is_working_day(self, d: date) -> bool:
        """Check if date is a working day."""
        return not self.is_weekend(d) and not self.is_holiday(d)
    
    def count_working_days(self, start: date, end: date) -> int:
        """Count working days in date range [start, end)."""
        count = 0
        current = start
        while current < end:
            if self.is_working_day(current):
                count += 1
            current += timedelta(days=1)
        return count
    
    def get_working_days_pattern(self, start: date, end: date) -> np.ndarray:
        """Get daily pattern of working days (1) vs non-working days (0)."""
        days = (end - start).days
        pattern = np.zeros(days)
        
        for i in range(days):
            d = start + timedelta(days=i)
            pattern[i] = 1 if self.is_working_day(d) else 0
        
        return pattern


class NationalCalendar(Calendar):
    """Pre-defined national calendars."""
    
    @classmethod
    def usa(cls) -> 'NationalCalendar':
        """US calendar with federal holidays."""
        holidays = [
            FixedHoliday("New Year", 1, 1),
            FixedHoliday("Independence Day", 7, 4),
            FixedHoliday("Veterans Day", 11, 11),
            FixedHoliday("Christmas", 12, 25),
            # TODO: Add movable holidays (MLK Day, Presidents Day, etc.)
        ]
        return cls(holidays)
    
    @classmethod
    def target(cls) -> 'NationalCalendar':
        """TARGET (Trans-European Automated Real-time Gross settlement) calendar."""
        holidays = [
            FixedHoliday("New Year", 1, 1),
            EasterRelatedHoliday("Good Friday", -2),
            EasterRelatedHoliday("Easter Monday", 1),
            FixedHoliday("Labour Day", 5, 1),
            FixedHoliday("Christmas", 12, 25),
            FixedHoliday("Boxing Day", 12, 26),
        ]
        return cls(holidays)
    
    @classmethod
    def belgium(cls) -> 'NationalCalendar':
        """Belgian national calendar."""
        holidays = [
            FixedHoliday("New Year", 1, 1),
            EasterRelatedHoliday("Easter Monday", 1),
            FixedHoliday("Labour Day", 5, 1),
            EasterRelatedHoliday("Ascension", 39),
            EasterRelatedHoliday("Whit Monday", 50),
            FixedHoliday("National Day", 7, 21),
            FixedHoliday("Assumption", 8, 15),
            FixedHoliday("All Saints", 11, 1),
            FixedHoliday("Armistice", 11, 11),
            FixedHoliday("Christmas", 12, 25),
        ]
        return cls(holidays)