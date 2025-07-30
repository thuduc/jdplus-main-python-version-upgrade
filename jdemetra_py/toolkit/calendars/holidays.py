"""Holiday definitions for calendars."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from datetime import date, timedelta
from enum import Enum

from .calendar import DayOfWeek
from .easter import easter_date, EasterAlgorithm


class Holiday(ABC):
    """Abstract base class for holidays."""
    
    @abstractmethod
    def date_in_year(self, year: int) -> Optional[date]:
        """Get holiday date in given year.
        
        Args:
            year: Year
            
        Returns:
            Holiday date or None if not applicable
        """
        pass


@dataclass
class HolidayDefinition:
    """Base holiday definition."""
    
    name: str
    validity_start: Optional[int] = None  # First year holiday is valid
    validity_end: Optional[int] = None    # Last year holiday is valid
    
    def is_valid_year(self, year: int) -> bool:
        """Check if holiday is valid in given year.
        
        Args:
            year: Year to check
            
        Returns:
            True if valid
        """
        if self.validity_start and year < self.validity_start:
            return False
        if self.validity_end and year > self.validity_end:
            return False
        return True
    
    @abstractmethod
    def get_date(self, year: int) -> Optional[date]:
        """Get holiday date in given year.
        
        Args:
            year: Year
            
        Returns:
            Holiday date or None
        """
        pass


@dataclass
class FixedHoliday(HolidayDefinition):
    """Fixed date holiday (e.g., Christmas)."""
    
    month: int
    day: int
    observance_rule: str = "none"  # "none", "nearest_workday", "monday_if_weekend"
    
    def get_date(self, year: int) -> Optional[date]:
        """Get holiday date."""
        if not self.is_valid_year(year):
            return None
        
        try:
            return date(year, self.month, self.day)
        except ValueError:
            # Invalid date (e.g., Feb 30)
            return None


@dataclass
class EasterRelatedHoliday(HolidayDefinition):
    """Easter-related holiday (e.g., Good Friday)."""
    
    offset: int  # Days from Easter (negative for before)
    algorithm: EasterAlgorithm = EasterAlgorithm.GREGORIAN
    
    def get_date(self, year: int) -> Optional[date]:
        """Get holiday date."""
        if not self.is_valid_year(year):
            return None
        
        easter = easter_date(year, self.algorithm)
        return easter + timedelta(days=self.offset)


@dataclass
class FixedWeekHoliday(HolidayDefinition):
    """Fixed week holiday (e.g., Thanksgiving)."""
    
    month: int
    day_of_week: DayOfWeek
    week: int  # 1-5 for first through fifth, -1 for last
    
    def get_date(self, year: int) -> Optional[date]:
        """Get holiday date."""
        if not self.is_valid_year(year):
            return None
        
        if self.week > 0:
            # Find nth occurrence
            return self._find_nth_weekday(year, self.month, self.day_of_week, self.week)
        else:
            # Find last occurrence
            return self._find_last_weekday(year, self.month, self.day_of_week)
    
    def _find_nth_weekday(self, year: int, month: int, 
                         day_of_week: DayOfWeek, n: int) -> Optional[date]:
        """Find nth occurrence of weekday in month.
        
        Args:
            year: Year
            month: Month
            day_of_week: Day of week to find
            n: Which occurrence (1-5)
            
        Returns:
            Date or None if not found
        """
        # Start from first day of month
        current = date(year, month, 1)
        
        # Find first occurrence
        while current.weekday() != day_of_week:
            current += timedelta(days=1)
        
        # Advance to nth occurrence
        current += timedelta(days=7 * (n - 1))
        
        # Check if still in same month
        if current.month != month:
            return None
        
        return current
    
    def _find_last_weekday(self, year: int, month: int,
                          day_of_week: DayOfWeek) -> date:
        """Find last occurrence of weekday in month.
        
        Args:
            year: Year
            month: Month
            day_of_week: Day of week to find
            
        Returns:
            Date
        """
        # Start from last day of month
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        
        current = next_month - timedelta(days=1)
        
        # Find last occurrence
        while current.weekday() != day_of_week:
            current -= timedelta(days=1)
        
        return current


@dataclass
class ConditionalHoliday(HolidayDefinition):
    """Holiday with conditional logic."""
    
    base_holiday: HolidayDefinition
    condition: callable  # Function(year) -> bool
    fallback_holiday: Optional[HolidayDefinition] = None
    
    def get_date(self, year: int) -> Optional[date]:
        """Get holiday date."""
        if not self.is_valid_year(year):
            return None
        
        if self.condition(year):
            return self.base_holiday.get_date(year)
        elif self.fallback_holiday:
            return self.fallback_holiday.get_date(year)
        else:
            return None


@dataclass
class ChineseNewYear(HolidayDefinition):
    """Chinese New Year holiday."""
    
    def get_date(self, year: int) -> Optional[date]:
        """Get Chinese New Year date.
        
        Note: Simplified calculation - actual implementation
        would use lunar calendar calculations.
        """
        if not self.is_valid_year(year):
            return None
        
        # Simplified: Use approximate dates
        # Real implementation would use lunar calendar
        approx_dates = {
            2020: date(2020, 1, 25),
            2021: date(2021, 2, 12),
            2022: date(2022, 2, 1),
            2023: date(2023, 1, 22),
            2024: date(2024, 2, 10),
            2025: date(2025, 1, 29),
        }
        
        return approx_dates.get(year)


# Predefined holidays

def new_years_day(observance_rule: str = "none") -> FixedHoliday:
    """New Year's Day (January 1)."""
    return FixedHoliday("New Year's Day", 1, 1, observance_rule=observance_rule)


def christmas(observance_rule: str = "none") -> FixedHoliday:
    """Christmas Day (December 25)."""
    return FixedHoliday("Christmas", 12, 25, observance_rule=observance_rule)


def good_friday(algorithm: EasterAlgorithm = EasterAlgorithm.GREGORIAN) -> EasterRelatedHoliday:
    """Good Friday (2 days before Easter)."""
    return EasterRelatedHoliday("Good Friday", -2, algorithm)


def easter_monday(algorithm: EasterAlgorithm = EasterAlgorithm.GREGORIAN) -> EasterRelatedHoliday:
    """Easter Monday (1 day after Easter)."""
    return EasterRelatedHoliday("Easter Monday", 1, algorithm)


def thanksgiving_us() -> FixedWeekHoliday:
    """US Thanksgiving (4th Thursday in November)."""
    return FixedWeekHoliday("Thanksgiving", 11, DayOfWeek.THURSDAY, 4)


def labor_day_us() -> FixedWeekHoliday:
    """US Labor Day (1st Monday in September)."""
    return FixedWeekHoliday("Labor Day", 9, DayOfWeek.MONDAY, 1)


def memorial_day_us() -> FixedWeekHoliday:
    """US Memorial Day (last Monday in May)."""
    return FixedWeekHoliday("Memorial Day", 5, DayOfWeek.MONDAY, -1)