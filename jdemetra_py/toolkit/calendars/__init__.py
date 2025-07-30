"""Calendar utilities for time series analysis."""

from .calendar import (
    CalendarDefinition,
    NationalCalendar,
    CompositeCalendar,
    WeekendDefinition,
    DayOfWeek
)
from .holidays import (
    Holiday,
    FixedHoliday,
    EasterRelatedHoliday,
    FixedWeekHoliday,
    HolidayDefinition
)
from .easter import (
    easter_date,
    EasterAlgorithm
)
from .utilities import (
    CalendarUtilities,
    is_holiday,
    is_weekend,
    is_working_day,
    count_working_days,
    next_working_day,
    previous_working_day
)

__all__ = [
    # Calendar
    "CalendarDefinition",
    "NationalCalendar",
    "CompositeCalendar",
    "WeekendDefinition",
    "DayOfWeek",
    # Holidays
    "Holiday",
    "FixedHoliday",
    "EasterRelatedHoliday",
    "FixedWeekHoliday",
    "HolidayDefinition",
    # Easter
    "easter_date",
    "EasterAlgorithm",
    # Utilities
    "CalendarUtilities",
    "is_holiday",
    "is_weekend",
    "is_working_day",
    "count_working_days",
    "next_working_day",
    "previous_working_day",
]