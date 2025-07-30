"""Calendar definitions and implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any
from datetime import date, datetime, timedelta
from enum import IntEnum

from .holidays import Holiday, HolidayDefinition


class DayOfWeek(IntEnum):
    """Day of week enumeration."""
    
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6
    
    @classmethod
    def from_date(cls, dt: date) -> 'DayOfWeek':
        """Get day of week from date."""
        return cls(dt.weekday())


@dataclass
class WeekendDefinition:
    """Definition of weekend days."""
    
    weekend_days: Set[DayOfWeek] = field(
        default_factory=lambda: {DayOfWeek.SATURDAY, DayOfWeek.SUNDAY}
    )
    
    def is_weekend(self, dt: date) -> bool:
        """Check if date is a weekend.
        
        Args:
            dt: Date to check
            
        Returns:
            True if weekend
        """
        return DayOfWeek.from_date(dt) in self.weekend_days
    
    @classmethod
    def western(cls) -> 'WeekendDefinition':
        """Western weekend (Saturday-Sunday)."""
        return cls({DayOfWeek.SATURDAY, DayOfWeek.SUNDAY})
    
    @classmethod
    def middle_eastern(cls) -> 'WeekendDefinition':
        """Middle Eastern weekend (Friday-Saturday)."""
        return cls({DayOfWeek.FRIDAY, DayOfWeek.SATURDAY})
    
    @classmethod
    def friday_only(cls) -> 'WeekendDefinition':
        """Friday only weekend."""
        return cls({DayOfWeek.FRIDAY})


class CalendarDefinition(ABC):
    """Abstract base class for calendar definitions."""
    
    def __init__(self, name: str = ""):
        """Initialize calendar.
        
        Args:
            name: Calendar name
        """
        self.name = name
        self._holiday_cache: Dict[int, Set[date]] = {}
    
    @abstractmethod
    def is_holiday(self, dt: date) -> bool:
        """Check if date is a holiday.
        
        Args:
            dt: Date to check
            
        Returns:
            True if holiday
        """
        pass
    
    @abstractmethod
    def is_weekend(self, dt: date) -> bool:
        """Check if date is a weekend.
        
        Args:
            dt: Date to check
            
        Returns:
            True if weekend
        """
        pass
    
    def is_working_day(self, dt: date) -> bool:
        """Check if date is a working day.
        
        Args:
            dt: Date to check
            
        Returns:
            True if working day
        """
        return not self.is_holiday(dt) and not self.is_weekend(dt)
    
    def next_working_day(self, dt: date) -> date:
        """Get next working day.
        
        Args:
            dt: Starting date
            
        Returns:
            Next working day
        """
        next_day = dt + timedelta(days=1)
        while not self.is_working_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def previous_working_day(self, dt: date) -> date:
        """Get previous working day.
        
        Args:
            dt: Starting date
            
        Returns:
            Previous working day
        """
        prev_day = dt - timedelta(days=1)
        while not self.is_working_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day
    
    def count_working_days(self, start: date, end: date) -> int:
        """Count working days in range.
        
        Args:
            start: Start date (inclusive)
            end: End date (exclusive)
            
        Returns:
            Number of working days
        """
        count = 0
        current = start
        while current < end:
            if self.is_working_day(current):
                count += 1
            current += timedelta(days=1)
        return count
    
    def get_holidays(self, year: int) -> List[date]:
        """Get all holidays in a year.
        
        Args:
            year: Year
            
        Returns:
            List of holiday dates
        """
        # Check cache
        if year in self._holiday_cache:
            return sorted(self._holiday_cache[year])
        
        # Generate holidays
        holidays = set()
        
        # Check each day of the year
        current = date(year, 1, 1)
        while current.year == year:
            if self.is_holiday(current) and not self.is_weekend(current):
                holidays.add(current)
            current += timedelta(days=1)
        
        # Cache results
        self._holiday_cache[year] = holidays
        
        return sorted(holidays)
    
    def clear_cache(self):
        """Clear holiday cache."""
        self._holiday_cache.clear()


@dataclass
class NationalCalendar(CalendarDefinition):
    """National calendar with holidays and weekends."""
    
    holidays: List[HolidayDefinition] = field(default_factory=list)
    weekend: WeekendDefinition = field(default_factory=WeekendDefinition.western)
    
    def __init__(self, name: str = "", 
                 holidays: Optional[List[HolidayDefinition]] = None,
                 weekend: Optional[WeekendDefinition] = None):
        """Initialize national calendar.
        
        Args:
            name: Calendar name
            holidays: List of holiday definitions
            weekend: Weekend definition
        """
        super().__init__(name)
        self.holidays = holidays or []
        self.weekend = weekend or WeekendDefinition.western()
        self._holiday_set_cache: Dict[int, Set[date]] = {}
    
    def add_holiday(self, holiday: HolidayDefinition):
        """Add holiday to calendar.
        
        Args:
            holiday: Holiday definition
        """
        self.holidays.append(holiday)
        self.clear_cache()
    
    def remove_holiday(self, holiday_name: str) -> bool:
        """Remove holiday by name.
        
        Args:
            holiday_name: Holiday name
            
        Returns:
            True if removed
        """
        original_len = len(self.holidays)
        self.holidays = [h for h in self.holidays if h.name != holiday_name]
        
        if len(self.holidays) < original_len:
            self.clear_cache()
            return True
        return False
    
    def is_holiday(self, dt: date) -> bool:
        """Check if date is a holiday."""
        year = dt.year
        
        # Get holiday set for year
        if year not in self._holiday_set_cache:
            holiday_set = set()
            for holiday in self.holidays:
                holiday_date = holiday.get_date(year)
                if holiday_date:
                    # Apply observance rules
                    observed = self._apply_observance_rules(holiday_date, holiday)
                    holiday_set.add(observed)
            self._holiday_set_cache[year] = holiday_set
        
        return dt in self._holiday_set_cache[year]
    
    def is_weekend(self, dt: date) -> bool:
        """Check if date is a weekend."""
        return self.weekend.is_weekend(dt)
    
    def _apply_observance_rules(self, holiday_date: date, 
                               holiday: HolidayDefinition) -> date:
        """Apply observance rules for holidays falling on weekends.
        
        Args:
            holiday_date: Original holiday date
            holiday: Holiday definition
            
        Returns:
            Observed date
        """
        if not hasattr(holiday, 'observance_rule'):
            return holiday_date
        
        rule = getattr(holiday, 'observance_rule', 'none')
        
        if rule == 'none':
            return holiday_date
        
        day_of_week = DayOfWeek.from_date(holiday_date)
        
        if rule == 'nearest_workday':
            if day_of_week == DayOfWeek.SATURDAY:
                return holiday_date - timedelta(days=1)
            elif day_of_week == DayOfWeek.SUNDAY:
                return holiday_date + timedelta(days=1)
        
        elif rule == 'monday_if_weekend':
            if day_of_week in {DayOfWeek.SATURDAY, DayOfWeek.SUNDAY}:
                days_to_monday = (DayOfWeek.MONDAY - day_of_week) % 7
                return holiday_date + timedelta(days=days_to_monday)
        
        elif rule == 'next_monday':
            if day_of_week == DayOfWeek.SUNDAY:
                return holiday_date + timedelta(days=1)
        
        return holiday_date
    
    def clear_cache(self):
        """Clear all caches."""
        super().clear_cache()
        self._holiday_set_cache.clear()
    
    @classmethod
    def united_states(cls) -> 'NationalCalendar':
        """US national calendar."""
        from .holidays import FixedHoliday, FixedWeekHoliday, EasterRelatedHoliday
        
        holidays = [
            FixedHoliday("New Year's Day", 1, 1, observance_rule="monday_if_weekend"),
            FixedWeekHoliday("Martin Luther King Jr. Day", 1, DayOfWeek.MONDAY, 3),
            FixedWeekHoliday("Presidents Day", 2, DayOfWeek.MONDAY, 3),
            EasterRelatedHoliday("Good Friday", -2),
            FixedWeekHoliday("Memorial Day", 5, DayOfWeek.MONDAY, -1),  # Last Monday
            FixedHoliday("Independence Day", 7, 4, observance_rule="nearest_workday"),
            FixedWeekHoliday("Labor Day", 9, DayOfWeek.MONDAY, 1),
            FixedWeekHoliday("Columbus Day", 10, DayOfWeek.MONDAY, 2),
            FixedHoliday("Veterans Day", 11, 11, observance_rule="monday_if_weekend"),
            FixedWeekHoliday("Thanksgiving", 11, DayOfWeek.THURSDAY, 4),
            FixedHoliday("Christmas", 12, 25, observance_rule="nearest_workday"),
        ]
        
        return cls("United States", holidays)
    
    @classmethod
    def european_central_bank(cls) -> 'NationalCalendar':
        """ECB TARGET calendar."""
        from .holidays import FixedHoliday, EasterRelatedHoliday
        
        holidays = [
            FixedHoliday("New Year's Day", 1, 1),
            EasterRelatedHoliday("Good Friday", -2),
            EasterRelatedHoliday("Easter Monday", 1),
            FixedHoliday("Labour Day", 5, 1),
            FixedHoliday("Christmas", 12, 25),
            FixedHoliday("St. Stephen's Day", 12, 26),
        ]
        
        return cls("ECB TARGET", holidays)


@dataclass
class CompositeCalendar(CalendarDefinition):
    """Composite calendar combining multiple calendars."""
    
    calendars: List[CalendarDefinition] = field(default_factory=list)
    mode: str = "union"  # "union" or "intersection"
    
    def __init__(self, name: str = "",
                 calendars: Optional[List[CalendarDefinition]] = None,
                 mode: str = "union"):
        """Initialize composite calendar.
        
        Args:
            name: Calendar name
            calendars: List of calendars
            mode: Combination mode
        """
        super().__init__(name)
        self.calendars = calendars or []
        self.mode = mode
    
    def add_calendar(self, calendar: CalendarDefinition):
        """Add calendar to composite.
        
        Args:
            calendar: Calendar to add
        """
        self.calendars.append(calendar)
        self.clear_cache()
    
    def is_holiday(self, dt: date) -> bool:
        """Check if date is a holiday."""
        if not self.calendars:
            return False
        
        if self.mode == "union":
            # Holiday in any calendar
            return any(cal.is_holiday(dt) for cal in self.calendars)
        else:
            # Holiday in all calendars
            return all(cal.is_holiday(dt) for cal in self.calendars)
    
    def is_weekend(self, dt: date) -> bool:
        """Check if date is a weekend."""
        if not self.calendars:
            return False
        
        if self.mode == "union":
            # Weekend in any calendar
            return any(cal.is_weekend(dt) for cal in self.calendars)
        else:
            # Weekend in all calendars
            return all(cal.is_weekend(dt) for cal in self.calendars)