"""Custom calendar effects example."""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.toolkit.calendars import (
    NationalCalendar, CalendarDefinition, CalendarUtilities,
    FixedWeekDayHoliday, FixedDayHoliday, EasterRelatedHoliday,
    DayOfWeek
)
from jdemetra_py.sa.tramoseats import TramoSeatsSpecification, TramoSeatsProcessor

def create_us_calendar() -> NationalCalendar:
    """Create US national calendar with major holidays."""
    calendar = NationalCalendar("US", weekend_definition="SaturdaySunday")
    
    # Fixed holidays
    calendar.add_holiday(FixedDayHoliday("New Year", 1, 1))
    calendar.add_holiday(FixedDayHoliday("Independence Day", 7, 4))
    calendar.add_holiday(FixedDayHoliday("Veterans Day", 11, 11))
    calendar.add_holiday(FixedDayHoliday("Christmas", 12, 25))
    
    # Fixed weekday holidays
    calendar.add_holiday(FixedWeekDayHoliday("MLK Day", 1, DayOfWeek.MONDAY, 3))  # 3rd Monday in January
    calendar.add_holiday(FixedWeekDayHoliday("Presidents Day", 2, DayOfWeek.MONDAY, 3))  # 3rd Monday in February
    calendar.add_holiday(FixedWeekDayHoliday("Memorial Day", 5, DayOfWeek.MONDAY, -1))  # Last Monday in May
    calendar.add_holiday(FixedWeekDayHoliday("Labor Day", 9, DayOfWeek.MONDAY, 1))  # 1st Monday in September
    calendar.add_holiday(FixedWeekDayHoliday("Thanksgiving", 11, DayOfWeek.THURSDAY, 4))  # 4th Thursday in November
    
    # Easter-related holidays
    calendar.add_holiday(EasterRelatedHoliday("Good Friday", -2))  # 2 days before Easter
    
    return calendar

def create_european_calendar() -> NationalCalendar:
    """Create European calendar with common holidays."""
    calendar = NationalCalendar("Europe", weekend_definition="SaturdaySunday")
    
    # Fixed holidays
    calendar.add_holiday(FixedDayHoliday("New Year", 1, 1))
    calendar.add_holiday(FixedDayHoliday("Epiphany", 1, 6))
    calendar.add_holiday(FixedDayHoliday("Labour Day", 5, 1))
    calendar.add_holiday(FixedDayHoliday("Assumption", 8, 15))
    calendar.add_holiday(FixedDayHoliday("All Saints", 11, 1))
    calendar.add_holiday(FixedDayHoliday("Christmas", 12, 25))
    calendar.add_holiday(FixedDayHoliday("Boxing Day", 12, 26))
    
    # Easter-related holidays
    calendar.add_holiday(EasterRelatedHoliday("Good Friday", -2))
    calendar.add_holiday(EasterRelatedHoliday("Easter Monday", 1))
    calendar.add_holiday(EasterRelatedHoliday("Ascension", 39))
    calendar.add_holiday(EasterRelatedHoliday("Whit Monday", 50))
    
    return calendar

def analyze_calendar_effects(calendar: NationalCalendar, year: int):
    """Analyze calendar effects for a given year."""
    print(f"\nCalendar Analysis for {year}")
    print("=" * 40)
    
    # Count holidays by month
    holidays_by_month = {i: 0 for i in range(1, 13)}
    
    # Check each day of the year
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    current_date = start_date
    
    total_holidays = 0
    total_weekends = 0
    total_working_days = 0
    
    while current_date <= end_date:
        if calendar.is_holiday(current_date):
            holidays_by_month[current_date.month] += 1
            total_holidays += 1
        elif calendar.is_weekend(current_date):
            total_weekends += 1
        else:
            total_working_days += 1
        
        current_date += timedelta(days=1)
    
    print(f"Total days: {(end_date - start_date).days + 1}")
    print(f"Working days: {total_working_days}")
    print(f"Weekends: {total_weekends}")
    print(f"Holidays: {total_holidays}")
    
    print("\nHolidays by month:")
    for month in range(1, 13):
        if holidays_by_month[month] > 0:
            month_name = date(year, month, 1).strftime('%B')
            print(f"  {month_name}: {holidays_by_month[month]}")
    
    return holidays_by_month

def generate_calendar_regressors(start_period: TsPeriod, 
                               end_period: TsPeriod,
                               calendar: NationalCalendar) -> np.ndarray:
    """Generate calendar regression variables."""
    # Get trading days regressors
    td_regressors = CalendarUtilities.trading_days_regressors(
        start_period, end_period, calendar
    )
    
    # Get working days count
    wd_count = CalendarUtilities.working_days_count(
        start_period, end_period, calendar
    )
    
    # Get leap year regressor
    ly_regressor = CalendarUtilities.leap_year_regressor(
        start_period, end_period
    )
    
    return td_regressors, wd_count, ly_regressor

def simulate_series_with_calendar_effects(calendar: NationalCalendar,
                                        n_years: int = 5) -> TsData:
    """Simulate a time series with calendar effects."""
    np.random.seed(42)
    
    # Monthly series
    start = TsPeriod.of(TsFrequency.MONTHLY, 2019, 0)
    n_obs = n_years * 12
    
    # Base components
    t = np.arange(n_obs)
    trend = 100 + 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    
    # Add calendar effects
    calendar_effects = np.zeros(n_obs)
    
    for i in range(n_obs):
        # Get the period
        period = start.plus(i)
        year = period.year()
        month = period.position() + 1
        
        # Count working days in this month
        month_start = date(year, month, 1)
        if month == 12:
            month_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)
        
        working_days = 0
        current = month_start
        while current <= month_end:
            if not calendar.is_holiday(current) and not calendar.is_weekend(current):
                working_days += 1
            current += timedelta(days=1)
        
        # Calendar effect proportional to working days
        # Fewer working days = lower activity
        expected_working_days = 21  # Typical month
        calendar_effects[i] = 5 * (working_days - expected_working_days)
    
    # Combine all components
    irregular = np.random.randn(n_obs) * 2
    values = trend + seasonal + calendar_effects + irregular
    
    return TsData.of(start, values), calendar_effects

def main():
    """Run calendar effects example."""
    print("Calendar Effects Example")
    print("=" * 50)
    
    # Create calendars
    us_calendar = create_us_calendar()
    eu_calendar = create_european_calendar()
    
    # Analyze calendars
    analyze_calendar_effects(us_calendar, 2024)
    analyze_calendar_effects(eu_calendar, 2024)
    
    # Generate series with calendar effects
    print("\nGenerating time series with calendar effects...")
    ts_us, calendar_effects_us = simulate_series_with_calendar_effects(us_calendar)
    ts_eu, calendar_effects_eu = simulate_series_with_calendar_effects(eu_calendar)
    
    # Perform seasonal adjustment with calendar regressors
    print("\nPerforming seasonal adjustment...")
    
    # US series with US calendar
    spec_us = TramoSeatsSpecification.rsa5()
    spec_us.set_calendar(us_calendar)
    spec_us.set_trading_days(True)
    
    processor = TramoSeatsProcessor(spec_us)
    results_us = processor.process(ts_us)
    
    # EU series with EU calendar
    spec_eu = TramoSeatsSpecification.rsa5()
    spec_eu.set_calendar(eu_calendar)
    spec_eu.set_trading_days(True)
    
    results_eu = processor.process(ts_eu)
    
    # Extract calendar components
    calendar_component_us = results_us.decomposition.calendar
    calendar_component_eu = results_eu.decomposition.calendar
    
    # Visualization
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Dates for plotting
    dates = pd.date_range('2019-01', periods=ts_us.length(), freq='M')
    
    # US plots
    axes[0, 0].plot(dates, ts_us.values, label='Original')
    axes[0, 0].plot(dates, results_us.decomposition.seasonally_adjusted.values, 
                    label='SA', linewidth=2)
    axes[0, 0].set_title('US Series: Original vs SA')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(dates, calendar_effects_us, label='True effect', alpha=0.7)
    if calendar_component_us is not None:
        axes[1, 0].plot(dates, calendar_component_us.values, 
                       label='Estimated effect', linewidth=2)
    axes[1, 0].set_title('US Calendar Effects')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # EU plots
    axes[0, 1].plot(dates, ts_eu.values, label='Original')
    axes[0, 1].plot(dates, results_eu.decomposition.seasonally_adjusted.values, 
                    label='SA', linewidth=2)
    axes[0, 1].set_title('EU Series: Original vs SA')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(dates, calendar_effects_eu, label='True effect', alpha=0.7)
    if calendar_component_eu is not None:
        axes[1, 1].plot(dates, calendar_component_eu.values, 
                       label='Estimated effect', linewidth=2)
    axes[1, 1].set_title('EU Calendar Effects')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Working days comparison
    wd_us = []
    wd_eu = []
    
    for i in range(ts_us.length()):
        period = ts_us.start().plus(i)
        year = period.year()
        month = period.position() + 1
        
        # Count working days
        month_start = date(year, month, 1)
        if month == 12:
            month_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)
        
        # US working days
        wd_count_us = 0
        current = month_start
        while current <= month_end:
            if not us_calendar.is_holiday(current) and not us_calendar.is_weekend(current):
                wd_count_us += 1
            current += timedelta(days=1)
        wd_us.append(wd_count_us)
        
        # EU working days
        wd_count_eu = 0
        current = month_start
        while current <= month_end:
            if not eu_calendar.is_holiday(current) and not eu_calendar.is_weekend(current):
                wd_count_eu += 1
            current += timedelta(days=1)
        wd_eu.append(wd_count_eu)
    
    axes[2, 0].plot(dates, wd_us, 'o-', markersize=4)
    axes[2, 0].set_title('US Working Days per Month')
    axes[2, 0].set_ylabel('Working Days')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=21, color='r', linestyle='--', alpha=0.5, label='Average')
    
    axes[2, 1].plot(dates, wd_eu, 'o-', markersize=4)
    axes[2, 1].set_title('EU Working Days per Month')
    axes[2, 1].set_ylabel('Working Days')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=21, color='r', linestyle='--', alpha=0.5, label='Average')
    
    plt.tight_layout()
    plt.savefig('calendar_effects_comparison.png', dpi=300)
    print("Saved: calendar_effects_comparison.png")
    
    # Generate calendar regressors example
    print("\nGenerating calendar regressors...")
    td_regressors, wd_count, ly_regressor = generate_calendar_regressors(
        ts_us.start(), ts_us.end(), us_calendar
    )
    
    print(f"Trading days regressors shape: {td_regressors.shape}")
    print(f"Working days count shape: {wd_count.shape}")
    print(f"Leap year regressor shape: {ly_regressor.shape}")
    
    # Save example regressors
    regressors_df = pd.DataFrame({
        'date': dates,
        'working_days': wd_count,
        'leap_year': ly_regressor
    })
    
    # Add trading days regressors
    for i in range(td_regressors.shape[1]):
        regressors_df[f'td_{i+1}'] = td_regressors[:, i]
    
    regressors_df.to_csv('calendar_regressors.csv', index=False)
    print("\nSaved calendar regressors to: calendar_regressors.csv")
    
    print("\nCalendar effects example completed!")

if __name__ == '__main__':
    main()