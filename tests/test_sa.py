"""Tests for seasonal adjustment module."""

import unittest
import numpy as np
from datetime import date

from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.sa import (
    DecompositionMode, ComponentType,
    TramoSeatsSpecification, TramoSeatsProcessor
)
from jdemetra_py.sa.base import SeriesDecomposition


class TestDecomposition(unittest.TestCase):
    """Test decomposition structures."""
    
    def setUp(self):
        """Set up test data."""
        self.start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
        
        # Generate synthetic seasonal data
        n = 60  # 5 years
        t = np.arange(n)
        
        # Components
        trend = 100 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        irregular = np.random.normal(0, 2, n)
        
        self.original = TsData.of(self.start, trend + seasonal + irregular)
        self.trend = TsData.of(self.start, trend)
        self.seasonal = TsData.of(self.start, seasonal)
        self.irregular = TsData.of(self.start, irregular)
    
    def test_decomposition_creation(self):
        """Test decomposition creation."""
        decomp = SeriesDecomposition(mode=DecompositionMode.ADDITIVE)
        
        decomp.series = self.original
        decomp.trend = self.trend
        decomp.seasonal = self.seasonal
        decomp.irregular = self.irregular
        
        # Check components
        self.assertIsNotNone(decomp.series)
        self.assertIsNotNone(decomp.trend)
        self.assertIsNotNone(decomp.seasonal)
        self.assertIsNotNone(decomp.irregular)
        
        # Compute SA series
        sa = self.trend.values + self.irregular.values
        decomp.seasonally_adjusted = TsData.of(self.start, sa)
        
        self.assertIsNotNone(decomp.seasonally_adjusted)
    
    def test_component_info(self):
        """Test component information."""
        decomp = SeriesDecomposition(mode=DecompositionMode.ADDITIVE)
        
        decomp.series = self.original
        decomp.trend = self.trend
        decomp.seasonal = self.seasonal
        decomp.irregular = self.irregular
        
        # Get component
        trend_comp = decomp.get_component(ComponentType.TREND)
        self.assertIsNotNone(trend_comp)
        np.testing.assert_array_equal(trend_comp.values, self.trend.values)
        
        # List components
        components = decomp.get_components()
        self.assertIn(ComponentType.SERIES, components)
        self.assertIn(ComponentType.TREND, components)
        self.assertIn(ComponentType.SEASONAL, components)
        self.assertIn(ComponentType.IRREGULAR, components)


class TestTramoSeatsSpec(unittest.TestCase):
    """Test TRAMO-SEATS specification."""
    
    def test_default_spec(self):
        """Test default specification."""
        spec = TramoSeatsSpecification()
        
        # Check defaults
        self.assertIsNotNone(spec.tramo)
        self.assertIsNotNone(spec.seats)
        
        # Validate
        self.assertTrue(spec.validate())
    
    def test_predefined_specs(self):
        """Test predefined specifications."""
        # RSA1 - automatic
        spec1 = TramoSeatsSpecification.rsa1()
        self.assertTrue(spec1.tramo.arima.auto_model)
        
        # RSA2 - with calendar
        spec2 = TramoSeatsSpecification.rsa2()
        self.assertIsNotNone(spec2.tramo.trading_days)
        self.assertIsNotNone(spec2.tramo.easter)
        
        # RSA3 - with outliers
        spec3 = TramoSeatsSpecification.rsa3()
        self.assertTrue(spec3.tramo.outliers.enabled)
        
        # RSA5 - full automatic
        spec5 = TramoSeatsSpecification.rsa5()
        self.assertEqual(spec5.tramo.transform.function, "log")
    
    def test_spec_serialization(self):
        """Test specification serialization."""
        spec = TramoSeatsSpecification.rsa4()
        
        # To dict
        spec_dict = spec.to_dict()
        self.assertIn("tramo", spec_dict)
        self.assertIn("seats", spec_dict)
        
        # From dict
        spec2 = TramoSeatsSpecification().from_dict(spec_dict)
        self.assertEqual(spec2.tramo.transform.function, 
                        spec.tramo.transform.function)


class TestTramoSeatsProcessing(unittest.TestCase):
    """Test TRAMO-SEATS processing (mock)."""
    
    def setUp(self):
        """Set up test data."""
        self.start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
        
        # Generate more realistic seasonal data
        n = 84  # 7 years
        t = np.arange(n)
        
        # Components
        trend = 100 + 0.3 * t + 0.001 * t**2
        seasonal_pattern = np.array([
            -5, -3, 0, 2, 4, 6, 5, 3, 1, -2, -4, -6
        ])
        seasonal = np.tile(seasonal_pattern, n // 12 + 1)[:n]
        irregular = np.random.normal(0, 1.5, n)
        
        # Multiplicative series (with log)
        self.series = TsData.of(
            self.start, 
            np.exp(np.log(trend) + seasonal/100 + irregular/100)
        )
    
    def test_processor_creation(self):
        """Test processor creation."""
        spec = TramoSeatsSpecification.rsa1()
        processor = TramoSeatsProcessor(spec)
        
        self.assertIsNotNone(processor.spec)
        self.assertIsNotNone(processor.tramo_processor)
        self.assertIsNotNone(processor.seats_decomposer)
    
    def test_specification_validation(self):
        """Test specification validation."""
        spec = TramoSeatsSpecification()
        
        # Valid spec
        self.assertTrue(spec.validate())
        
        # Invalid ARIMA spec
        spec.tramo.arima.auto_model = False
        spec.tramo.arima.p = None
        self.assertFalse(spec.validate())
        
        # Invalid boundaries
        spec = TramoSeatsSpecification()
        spec.seats.trend_boundary = 1.5
        self.assertFalse(spec.validate())


class TestDiagnostics(unittest.TestCase):
    """Test diagnostic computations."""
    
    def setUp(self):
        """Set up test decomposition."""
        self.start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
        n = 60
        
        # Create simple decomposition
        self.decomp = SeriesDecomposition(mode=DecompositionMode.ADDITIVE)
        
        # Original
        t = np.arange(n)
        trend = 100 + 0.5 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 12)
        irregular = np.random.normal(0, 1, n)
        
        self.decomp.series = TsData.of(self.start, trend + seasonal + irregular)
        self.decomp.trend = TsData.of(self.start, trend)
        self.decomp.seasonal = TsData.of(self.start, seasonal)
        self.decomp.irregular = TsData.of(self.start, irregular)
        self.decomp.seasonally_adjusted = TsData.of(self.start, trend + irregular)
    
    def test_m_statistics(self):
        """Test M-statistics computation."""
        from jdemetra_py.sa.diagnostics.quality import MStatistics
        
        m_stats = MStatistics.compute(self.decomp)
        
        # Check some statistics exist
        self.assertIn("M1", m_stats)
        self.assertIn("M3", m_stats)
        self.assertIn("Q", m_stats)
        
        # Check reasonable ranges
        self.assertGreaterEqual(m_stats["Q"], 0)
        self.assertLessEqual(m_stats["Q"], 3)
    
    def test_seasonality_tests(self):
        """Test seasonality tests."""
        from jdemetra_py.sa.diagnostics.seasonality import SeasonalityTests
        
        # Test original series
        results = SeasonalityTests.test_all(self.decomp.series)
        
        self.assertIn("friedman", results)
        self.assertIn("kruskal_wallis", results)
        self.assertIn("stable_seasonality", results)
        
        # Original should show seasonality
        self.assertLess(results["friedman"].pvalue, 0.05)
        
        # Test SA series - should show less seasonality
        sa_results = SeasonalityTests.test_all(self.decomp.seasonally_adjusted)
        # P-value should be higher for SA series (less seasonal)
        self.assertGreater(sa_results["friedman"].pvalue, results["friedman"].pvalue)


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking methods."""
    
    def setUp(self):
        """Set up test data."""
        # Monthly series
        self.monthly_start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
        self.monthly_values = np.array([100, 102, 98, 105, 103, 107,
                                       110, 108, 112, 109, 111, 115])
        self.monthly = TsData.of(self.monthly_start, self.monthly_values)
        
        # Quarterly constraints
        self.quarterly_start = TsPeriod.of(TsFrequency.QUARTERLY, 2020, 0)
        self.quarterly_values = np.array([300, 320, 330, 335])
        self.quarterly = TsData.of(self.quarterly_start, self.quarterly_values)
    
    def test_denton_benchmarking(self):
        """Test Denton benchmarking."""
        from jdemetra_py.sa.benchmarking import DentonBenchmarking
        
        benchmarker = DentonBenchmarking(differencing=1)
        
        # Validate inputs
        try:
            benchmarker._validate_inputs(self.monthly, self.quarterly)
        except ValueError:
            self.fail("Valid inputs rejected")
        
        # Check aggregation matrix
        C = benchmarker._create_aggregation_matrix(self.monthly, self.quarterly)
        self.assertEqual(C.shape, (4, 12))
        
        # Check row sums
        np.testing.assert_array_equal(C.sum(axis=1), [3, 3, 3, 3])
    
    def test_cholette_benchmarking(self):
        """Test Cholette benchmarking."""
        from jdemetra_py.sa.benchmarking import CholetteBenchmarking
        
        benchmarker = CholetteBenchmarking(rho=1.0)
        
        # Validate inputs
        try:
            benchmarker._validate_inputs(self.monthly, self.quarterly)
        except ValueError:
            self.fail("Valid inputs rejected")
        
        # Test invalid frequency combination (yearly to quarterly is invalid)
        yearly = TsData.of(
            TsPeriod.of(TsFrequency.YEARLY, 2020, 0),
            [1200]
        )
        
        with self.assertRaises(ValueError):
            benchmarker._validate_inputs(yearly, self.quarterly)


if __name__ == '__main__':
    unittest.main()