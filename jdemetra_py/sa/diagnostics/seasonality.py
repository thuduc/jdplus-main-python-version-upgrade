"""Seasonality tests for time series."""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from scipy import stats

from ...toolkit.timeseries import TsData
from ...toolkit.stats.tests import TestResult


@dataclass
class SeasonalityTest:
    """Base class for seasonality tests."""
    
    # Test configuration
    max_lag: int = 2
    critical_value: float = 0.05
    
    def test(self, series: TsData, frequency: Optional[int] = None) -> TestResult:
        """Run seasonality test.
        
        Args:
            series: Time series data
            frequency: Seasonal frequency (if not provided, uses series frequency)
            
        Returns:
            Test result
        """
        if frequency is None:
            frequency = series.frequency.periods_per_year
        
        # Convert to array and handle missing values
        values = series.values
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        if len(clean_values) < 3 * frequency:
            return TestResult(
                statistic=np.nan,
                pvalue=1.0,
                critical_value=self.critical_value,
                test_name="Seasonality Test",
                description="Insufficient data"
            )
        
        # Compute QS statistic
        statistic = self._compute_qs_statistic(clean_values, frequency)
        
        # Compute p-value (chi-square distribution)
        df = frequency - 1
        pvalue = 1 - stats.chi2.cdf(statistic, df)
        
        return TestResult(
            statistic=statistic,
            pvalue=pvalue,
            critical_value=self.critical_value,
            test_name="QS Seasonality Test",
            description=f"No seasonality" if pvalue > self.critical_value else "Seasonality present"
        )
    
    def _compute_qs_statistic(self, values: np.ndarray, frequency: int) -> float:
        """Compute QS statistic for seasonality.
        
        Based on Ljung-Box type test on seasonal autocorrelations.
        """
        n = len(values)
        
        # Compute seasonal autocorrelations
        seasonal_acf = []
        for lag in range(frequency, min(n // 4, frequency * self.max_lag + 1), frequency):
            # Compute autocorrelation at seasonal lag
            c0 = np.var(values)
            ck = np.mean((values[:-lag] - np.mean(values)) * 
                        (values[lag:] - np.mean(values)))
            rk = ck / c0
            seasonal_acf.append((lag, rk))
        
        # Compute QS statistic
        qs = 0.0
        for lag, rk in seasonal_acf:
            qs += (rk ** 2) / (n - lag)
        
        qs *= n * (n + 2)
        
        return qs


@dataclass 
class FriedmanTest:
    """Friedman test for seasonality (non-parametric)."""
    
    critical_value: float = 0.05
    
    def test(self, series: TsData, frequency: Optional[int] = None) -> TestResult:
        """Run Friedman test.
        
        Args:
            series: Time series data
            frequency: Seasonal frequency
            
        Returns:
            Test result
        """
        if frequency is None:
            frequency = series.frequency.periods_per_year
        
        values = series.values
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        # Organize data by season
        n_years = len(clean_values) // frequency
        if n_years < 2:
            return TestResult(
                statistic=np.nan,
                pvalue=1.0,
                critical_value=self.critical_value,
                test_name="Friedman Test",
                description="Insufficient data"
            )
        
        # Create matrix: rows = years, columns = seasons
        seasonal_matrix = []
        for year in range(n_years):
            year_data = clean_values[year * frequency:(year + 1) * frequency]
            if len(year_data) == frequency:
                seasonal_matrix.append(year_data)
        
        seasonal_matrix = np.array(seasonal_matrix)
        
        # Apply Friedman test
        statistic, pvalue = stats.friedmanchisquare(*seasonal_matrix.T)
        
        return TestResult(
            statistic=statistic,
            pvalue=pvalue,
            critical_value=self.critical_value,
            test_name="Friedman Seasonality Test",
            description=f"No seasonality" if pvalue > self.critical_value else "Seasonality present"
        )


@dataclass
class KruskalWallisTest:
    """Kruskal-Wallis test for seasonality (non-parametric)."""
    
    critical_value: float = 0.05
    
    def test(self, series: TsData, frequency: Optional[int] = None) -> TestResult:
        """Run Kruskal-Wallis test.
        
        Args:
            series: Time series data
            frequency: Seasonal frequency
            
        Returns:
            Test result
        """
        if frequency is None:
            frequency = series.frequency.periods_per_year
        
        values = series.values
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        # Group by season
        seasonal_groups = [[] for _ in range(frequency)]
        for i, value in enumerate(clean_values):
            season = i % frequency
            seasonal_groups[season].append(value)
        
        # Remove empty groups
        seasonal_groups = [g for g in seasonal_groups if len(g) > 0]
        
        if len(seasonal_groups) < 2:
            return TestResult(
                statistic=np.nan,
                pvalue=1.0,
                critical_value=self.critical_value,
                test_name="Kruskal-Wallis Test",
                description="Insufficient data"
            )
        
        # Apply Kruskal-Wallis test
        statistic, pvalue = stats.kruskal(*seasonal_groups)
        
        return TestResult(
            statistic=statistic,
            pvalue=pvalue,
            critical_value=self.critical_value,
            test_name="Kruskal-Wallis Seasonality Test",
            description=f"No seasonality" if pvalue > self.critical_value else "Seasonality present"
        )


@dataclass
class CombinedSeasonalityTest:
    """Combined seasonality test using multiple methods."""
    
    critical_value: float = 0.05
    weights: Optional[dict] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "qs": 0.4,
                "friedman": 0.3,
                "kruskal": 0.3
            }
    
    def test(self, series: TsData, frequency: Optional[int] = None) -> TestResult:
        """Run combined seasonality test.
        
        Args:
            series: Time series data  
            frequency: Seasonal frequency
            
        Returns:
            Combined test result
        """
        # Run individual tests
        qs_test = SeasonalityTest()
        friedman_test = FriedmanTest()
        kruskal_test = KruskalWallisTest()
        
        results = {
            "qs": qs_test.test(series, frequency),
            "friedman": friedman_test.test(series, frequency),
            "kruskal": kruskal_test.test(series, frequency)
        }
        
        # Combine p-values using weighted harmonic mean
        valid_results = [(name, r) for name, r in results.items() 
                        if not np.isnan(r.pvalue)]
        
        if not valid_results:
            return TestResult(
                statistic=np.nan,
                pvalue=1.0,
                critical_value=self.critical_value,
                test_name="Combined Seasonality Test",
                description="All tests failed"
            )
        
        # Weighted combination
        combined_pvalue = 0.0
        total_weight = 0.0
        
        for name, result in valid_results:
            weight = self.weights.get(name, 1.0)
            combined_pvalue += weight / max(result.pvalue, 1e-10)
            total_weight += weight
        
        combined_pvalue = total_weight / combined_pvalue
        
        # Combined statistic (weighted average)
        combined_stat = sum(self.weights.get(name, 1.0) * result.statistic 
                          for name, result in valid_results) / total_weight
        
        # Return combined result
        return TestResult(
            statistic=combined_stat,
            pvalue=combined_pvalue,
            critical_value=self.critical_value,
            test_name="Combined Seasonality Test"
        )


@dataclass
class ResidualSeasonalityTest:
    """Test for residual seasonality after SA."""
    
    method: str = "combined"  # "friedman", "kruskal", or "combined"
    critical_value: float = 0.01  # Stricter for residuals
    
    def test(self, residuals: TsData, original: TsData) -> TestResult:
        """Test for residual seasonality.
        
        Args:
            residuals: Residual series (irregular component)
            original: Original series (for frequency info)
            
        Returns:
            Test result
        """
        # Apply appropriate test
        if self.method == "friedman":
            test = FriedmanTest()
        elif self.method == "kruskal":
            test = KruskalWallisTest()
        else:
            test = CombinedSeasonalityTest()
        
        result = test.test(residuals)
        
        # Modify description
        result.description = f"Residual seasonality: {result.description}"
        
        return result


class SeasonalityTests:
    """Collection of seasonality tests."""
    
    @staticmethod
    def qs_test(series: TsData, frequency: int) -> TestResult:
        """Run QS test for seasonality."""
        test = SeasonalityTest()
        return test.test(series, frequency)
    
    @staticmethod
    def friedman_test(series: TsData, frequency: int) -> TestResult:
        """Run Friedman test for seasonality."""
        test = FriedmanTest()
        return test.test(series, frequency)
    
    @staticmethod
    def kruskal_wallis_test(series: TsData, frequency: int) -> TestResult:
        """Run Kruskal-Wallis test for seasonality."""
        test = KruskalWallisTest()
        return test.test(series, frequency)
    
    @staticmethod
    def combined_test(series: TsData, frequency: int) -> TestResult:
        """Run combined seasonality test."""
        test = CombinedSeasonalityTest()
        return test.test(series, frequency)
    
    @staticmethod
    def test_all(series: TsData, frequency: Optional[int] = None) -> dict:
        """Run all seasonality tests.
        
        Args:
            series: Time series data
            frequency: Seasonal frequency (if not provided, uses series frequency)
            
        Returns:
            Dictionary of test results
        """
        if frequency is None:
            frequency = series.frequency.periods_per_year
            
        return {
            "friedman": SeasonalityTests.friedman_test(series, frequency),
            "kruskal_wallis": SeasonalityTests.kruskal_wallis_test(series, frequency),
            "qs": SeasonalityTests.qs_test(series, frequency),
            "stable_seasonality": SeasonalityTests.combined_test(series, frequency)
        }