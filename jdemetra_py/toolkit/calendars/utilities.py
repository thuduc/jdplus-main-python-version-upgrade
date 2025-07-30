"""Calendar utility functions."""

from typing import List, Optional, Set, Tuple
from datetime import date, timedelta
import numpy as np
import pandas as pd

from .calendar import CalendarDefinition, NationalCalendar
from ..timeseries import TsData, TsPeriod, TsFrequency


class CalendarUtilities:
    """Utility functions for calendar operations."""
    
    @staticmethod
    def create_calendar_variables(calendar: CalendarDefinition,
                                 start: date,
                                 end: date,
                                 frequency: TsFrequency) -> dict:
        """Create calendar regression variables.
        
        Args:
            calendar: Calendar definition
            start: Start date
            end: End date
            frequency: Time series frequency
            
        Returns:
            Dictionary of calendar variables
        """
        variables = {}
        
        # Generate dates for the period
        dates = CalendarUtilities._generate_dates(start, end, frequency)
        n_periods = len(dates)
        
        # Trading days variables
        td_vars = CalendarUtilities._create_trading_days_variables(
            calendar, dates, frequency
        )
        variables.update(td_vars)
        
        # Working days variable
        wd = CalendarUtilities._create_working_days_variable(
            calendar, dates, frequency
        )
        variables['working_days'] = wd
        
        # Holiday indicators
        holidays = CalendarUtilities._create_holiday_indicators(
            calendar, dates, frequency
        )
        variables.update(holidays)
        
        return variables
    
    @staticmethod
    def _generate_dates(start: date, end: date, 
                       frequency: TsFrequency) -> List[Tuple[date, date]]:
        """Generate date ranges for each period.
        
        Args:
            start: Start date
            end: End date
            frequency: Frequency
            
        Returns:
            List of (period_start, period_end) tuples
        """
        dates = []
        current = start
        
        while current < end:
            if frequency == TsFrequency.MONTHLY:
                # Month boundaries
                if current.month == 12:
                    next_date = date(current.year + 1, 1, 1)
                else:
                    next_date = date(current.year, current.month + 1, 1)
                dates.append((current, next_date - timedelta(days=1)))
                current = next_date
                
            elif frequency == TsFrequency.QUARTERLY:
                # Quarter boundaries
                quarter = (current.month - 1) // 3
                if quarter == 3:
                    next_date = date(current.year + 1, 1, 1)
                else:
                    next_date = date(current.year, (quarter + 1) * 3 + 1, 1)
                dates.append((current, next_date - timedelta(days=1)))
                current = next_date
                
            elif frequency == TsFrequency.YEARLY:
                # Year boundaries
                next_date = date(current.year + 1, 1, 1)
                dates.append((current, next_date - timedelta(days=1)))
                current = next_date
                
            else:
                # Daily or other - use single day
                dates.append((current, current))
                current += timedelta(days=1)
        
        return dates
    
    @staticmethod
    def _create_trading_days_variables(calendar: CalendarDefinition,
                                     dates: List[Tuple[date, date]],
                                     frequency: TsFrequency) -> dict:
        """Create trading days variables.
        
        Args:
            calendar: Calendar definition
            dates: Period date ranges
            frequency: Frequency
            
        Returns:
            Dictionary of TD variables
        """
        from .calendar import DayOfWeek
        
        n_periods = len(dates)
        
        # Create variables for each day of week
        td_vars = {}
        for day in DayOfWeek:
            td_vars[f'td_{day.name.lower()}'] = np.zeros(n_periods)
        
        # Count working days by day of week
        for i, (start, end) in enumerate(dates):
            current = start
            while current <= end:
                if calendar.is_working_day(current):
                    day = DayOfWeek.from_date(current)
                    td_vars[f'td_{day.name.lower()}'][i] += 1
                current += timedelta(days=1)
        
        # Create contrast variables (6 variables instead of 7)
        # Use Sunday as reference
        contrast_vars = {}
        sunday_values = td_vars['td_sunday']
        
        for day in [DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY,
                   DayOfWeek.THURSDAY, DayOfWeek.FRIDAY, DayOfWeek.SATURDAY]:
            key = f'td{day.value}'  # td1, td2, ..., td6
            contrast_vars[key] = td_vars[f'td_{day.name.lower()}'] - sunday_values
        
        return contrast_vars
    
    @staticmethod
    def _create_working_days_variable(calendar: CalendarDefinition,
                                    dates: List[Tuple[date, date]],
                                    frequency: TsFrequency) -> np.ndarray:
        """Create working days variable.
        
        Args:
            calendar: Calendar definition
            dates: Period date ranges
            frequency: Frequency
            
        Returns:
            Working days array
        """
        n_periods = len(dates)
        working_days = np.zeros(n_periods)
        
        for i, (start, end) in enumerate(dates):
            working_days[i] = calendar.count_working_days(
                start, end + timedelta(days=1)
            )
        
        return working_days
    
    @staticmethod
    def _create_holiday_indicators(calendar: CalendarDefinition,
                                 dates: List[Tuple[date, date]],
                                 frequency: TsFrequency) -> dict:
        """Create holiday indicator variables.
        
        Args:
            calendar: Calendar definition
            dates: Period date ranges
            frequency: Frequency
            
        Returns:
            Dictionary of holiday indicators
        """
        indicators = {}
        
        # Get unique years
        years = sorted(set(d[0].year for d in dates) | set(d[1].year for d in dates))
        
        # Get all holidays
        all_holidays = {}
        if isinstance(calendar, NationalCalendar):
            for holiday_def in calendar.holidays:
                holiday_dates = []
                for year in years:
                    hdate = holiday_def.get_date(year)
                    if hdate:
                        holiday_dates.append(hdate)
                if holiday_dates:
                    all_holidays[holiday_def.name] = holiday_dates
        
        # Create indicators
        for holiday_name, holiday_dates in all_holidays.items():
            indicator = np.zeros(len(dates))
            
            for i, (start, end) in enumerate(dates):
                # Check if any holiday falls in this period
                for hdate in holiday_dates:
                    if start <= hdate <= end:
                        indicator[i] = 1
                        break
            
            # Only include if holiday appears in data
            if np.sum(indicator) > 0:
                clean_name = holiday_name.lower().replace(' ', '_').replace("'", '')
                indicators[f'holiday_{clean_name}'] = indicator
        
        return indicators
    
    @staticmethod
    def adjust_for_calendar(series: TsData, 
                          calendar: CalendarDefinition,
                          method: str = 'ratio') -> TsData:
        """Adjust series for calendar effects.
        
        Args:
            series: Time series
            calendar: Calendar definition
            method: Adjustment method ('ratio' or 'difference')
            
        Returns:
            Calendar-adjusted series
        """
        # Get period dates
        dates = []
        start_period = series.start
        
        for i in range(series.length):
            period = start_period.plus(i)
            if series.frequency == TsFrequency.MONTHLY:
                pstart = date(period.year, period.position + 1, 1)
                if period.position == 11:
                    pend = date(period.year + 1, 1, 1) - timedelta(days=1)
                else:
                    pend = date(period.year, period.position + 2, 1) - timedelta(days=1)
            else:
                # Simplified for other frequencies
                pstart = pend = date(period.year, 1, 1)
            
            dates.append((pstart, pend))
        
        # Calculate working days
        working_days = np.array([
            calendar.count_working_days(start, end + timedelta(days=1))
            for start, end in dates
        ])
        
        # Calculate average working days
        avg_working_days = np.mean(working_days)
        
        # Adjust series
        if method == 'ratio':
            adjustment = avg_working_days / working_days
            adjusted_values = series.values * adjustment
        else:  # difference
            adjustment = avg_working_days - working_days
            adjusted_values = series.values + adjustment
        
        return TsData.of(series.start, adjusted_values)


# Convenience functions

def is_holiday(dt: date, calendar: CalendarDefinition) -> bool:
    """Check if date is a holiday.
    
    Args:
        dt: Date to check
        calendar: Calendar definition
        
    Returns:
        True if holiday
    """
    return calendar.is_holiday(dt)


def is_weekend(dt: date, calendar: CalendarDefinition) -> bool:
    """Check if date is a weekend.
    
    Args:
        dt: Date to check
        calendar: Calendar definition
        
    Returns:
        True if weekend
    """
    return calendar.is_weekend(dt)


def is_working_day(dt: date, calendar: CalendarDefinition) -> bool:
    """Check if date is a working day.
    
    Args:
        dt: Date to check
        calendar: Calendar definition
        
    Returns:
        True if working day
    """
    return calendar.is_working_day(dt)


def count_working_days(start: date, end: date, 
                      calendar: CalendarDefinition) -> int:
    """Count working days in range.
    
    Args:
        start: Start date (inclusive)
        end: End date (exclusive)
        calendar: Calendar definition
        
    Returns:
        Number of working days
    """
    return calendar.count_working_days(start, end)


def next_working_day(dt: date, calendar: CalendarDefinition) -> date:
    """Get next working day.
    
    Args:
        dt: Starting date
        calendar: Calendar definition
        
    Returns:
        Next working day
    """
    return calendar.next_working_day(dt)


def previous_working_day(dt: date, calendar: CalendarDefinition) -> date:
    """Get previous working day.
    
    Args:
        dt: Starting date
        calendar: Calendar definition
        
    Returns:
        Previous working day
    """
    return calendar.previous_working_day(dt)