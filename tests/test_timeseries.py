"""Tests for time series module."""

import unittest
import numpy as np
import pandas as pd
from datetime import date

from jdemetra_py.toolkit.timeseries import (
    TsData, TsPeriod, TsFrequency, TsDomain
)
from jdemetra_py.toolkit.timeseries.data import EmptyCause


class TestTsPeriod(unittest.TestCase):
    """Test TsPeriod class."""
    
    def test_creation(self):
        """Test period creation."""
        # Monthly
        period = TsPeriod.of(TsFrequency.MONTHLY, 2023, 0)
        self.assertEqual(period.frequency, TsFrequency.MONTHLY)
        self.assertEqual(period.year, 2023)
        self.assertEqual(period.position, 0)
        
        # Quarterly
        period = TsPeriod.of(TsFrequency.QUARTERLY, 2023, 2)
        self.assertEqual(period.frequency, TsFrequency.QUARTERLY)
        self.assertEqual(period.year, 2023)
        self.assertEqual(period.position, 2)
    
    def test_arithmetic(self):
        """Test period arithmetic."""
        period = TsPeriod.of(TsFrequency.MONTHLY, 2023, 5)
        
        # Plus
        next_period = period.plus(1)
        self.assertEqual(next_period.year, 2023)
        self.assertEqual(next_period.position, 6)
        
        # Plus with year rollover
        period_dec = TsPeriod.of(TsFrequency.MONTHLY, 2023, 11)
        period_jan = period_dec.plus(1)
        self.assertEqual(period_jan.year, 2024)
        self.assertEqual(period_jan.position, 0)
        
        # Minus
        prev_period = period.minus(period_jan)
        self.assertEqual(prev_period, -7)
    
    def test_date_conversion(self):
        """Test date conversions."""
        period = TsPeriod.of(TsFrequency.MONTHLY, 2023, 5)
        
        # Start date
        start = period.start_date()
        self.assertEqual(start, date(2023, 6, 1))
        
        # End date
        end = period.end_date()
        self.assertEqual(end, date(2023, 6, 30))
    
    def test_display(self):
        """Test display strings."""
        # Monthly
        period = TsPeriod.of(TsFrequency.MONTHLY, 2023, 0)
        self.assertEqual(period.display(), "2023-01")
        
        # Quarterly
        period = TsPeriod.of(TsFrequency.QUARTERLY, 2023, 2)
        self.assertEqual(period.display(), "2023-Q3")


class TestTsDomain(unittest.TestCase):
    """Test TsDomain class."""
    
    def test_creation(self):
        """Test domain creation."""
        start = TsPeriod.of(TsFrequency.MONTHLY, 2023, 0)
        domain = TsDomain.of(start, 12)
        
        self.assertEqual(domain.start, start)
        self.assertEqual(domain.length, 12)
        self.assertEqual(domain.frequency, TsFrequency.MONTHLY)
    
    def test_range(self):
        """Test domain range."""
        start = TsPeriod.of(TsFrequency.MONTHLY, 2023, 0)
        end = TsPeriod.of(TsFrequency.MONTHLY, 2023, 11)
        
        domain = TsDomain.range(start, end)
        self.assertEqual(domain.length, 12)
    
    def test_contains(self):
        """Test domain contains."""
        start = TsPeriod.of(TsFrequency.MONTHLY, 2023, 0)
        domain = TsDomain.of(start, 12)
        
        # Inside domain
        period = TsPeriod.of(TsFrequency.MONTHLY, 2023, 5)
        self.assertTrue(domain.contains(period))
        
        # Outside domain
        period = TsPeriod.of(TsFrequency.MONTHLY, 2024, 0)
        self.assertFalse(domain.contains(period))


class TestTsData(unittest.TestCase):
    """Test TsData class."""
    
    def setUp(self):
        """Set up test data."""
        self.start = TsPeriod.of(TsFrequency.MONTHLY, 2023, 0)
        self.values = np.array([100, 102, 105, 103, 107, 110])
    
    def test_creation(self):
        """Test series creation."""
        # From array
        ts = TsData.of(self.start, self.values)
        self.assertEqual(ts.length, 6)
        self.assertEqual(ts.frequency, TsFrequency.MONTHLY)
        np.testing.assert_array_equal(ts.values, self.values)
        
        # From list
        ts = TsData.of(self.start, list(self.values))
        self.assertEqual(ts.length, 6)
        
        # Empty series
        ts = TsData.empty(EmptyCause.UNDEFINED)
        self.assertTrue(ts.is_empty())
    
    def test_indexing(self):
        """Test series indexing."""
        ts = TsData.of(self.start, self.values)
        
        # Get value by index
        self.assertEqual(ts.get(0), 100)
        self.assertEqual(ts.get(5), 110)
        
        # Get value by period
        period = TsPeriod.of(TsFrequency.MONTHLY, 2023, 2)
        self.assertEqual(ts.get_by_period(period), 105)
    
    def test_operations(self):
        """Test series operations."""
        ts = TsData.of(self.start, self.values)
        
        # Function application
        log_ts = ts.fn(np.log)
        np.testing.assert_array_almost_equal(
            log_ts.values, np.log(self.values)
        )
        
        # Lead/lag
        lag_ts = ts.lag(1)
        self.assertTrue(np.isnan(lag_ts.values[0]))
        np.testing.assert_array_equal(lag_ts.values[1:], self.values[:-1])
        
        lead_ts = ts.lead(1)
        self.assertTrue(np.isnan(lead_ts.values[-1]))
        np.testing.assert_array_equal(lead_ts.values[:-1], self.values[1:])
    
    def test_window_operations(self):
        """Test window operations."""
        ts = TsData.of(self.start, self.values)
        
        # Drop first/last
        dropped = ts.drop(1, 1)
        self.assertEqual(dropped.length, 4)
        np.testing.assert_array_equal(dropped.values, self.values[1:-1])
        
        # Extend
        extended = ts.extend(0, 2)
        self.assertEqual(extended.length, 8)
        self.assertTrue(np.isnan(extended.values[-1]))
        self.assertTrue(np.isnan(extended.values[-2]))
    
    def test_statistics(self):
        """Test statistical operations."""
        ts = TsData.of(self.start, self.values)
        
        # Basic stats
        self.assertAlmostEqual(ts.average(), np.mean(self.values))
        self.assertAlmostEqual(ts.sum(), np.sum(self.values))
        
        # Missing values
        values_with_nan = self.values.astype(float)
        values_with_nan[2] = np.nan
        ts_nan = TsData.of(self.start, values_with_nan)
        
        self.assertEqual(ts_nan.count_missing(), 1)
        self.assertAlmostEqual(
            ts_nan.average(), 
            np.nanmean(values_with_nan)
        )


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_pandas_integration(self):
        """Test pandas integration."""
        # Create pandas series
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        values = np.random.randn(12)
        pd_series = pd.Series(values, index=dates)
        
        # Convert to TsData
        start = TsPeriod.of(TsFrequency.MONTHLY, 2023, 0)
        ts = TsData.of(start, pd_series.values)
        
        # Verify
        self.assertEqual(ts.length, 12)
        np.testing.assert_array_equal(ts.values, values)
    


if __name__ == '__main__':
    unittest.main()