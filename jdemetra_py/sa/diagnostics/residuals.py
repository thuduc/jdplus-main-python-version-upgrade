"""Residual diagnostics for seasonal adjustment."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy import stats

from ...toolkit.timeseries import TsData
from ...toolkit.stats.tests import (
    LjungBoxTest, BoxPierceTest, JarqueBeraTest,
    SkewnessTest, KurtosisTest, TestResult
)
from ...toolkit.stats.descriptive import autocorrelations


@dataclass
class ResidualsDiagnostics:
    """Container for residuals diagnostics."""
    
    # Descriptive statistics
    mean: float
    std: float
    skewness: float
    kurtosis: float
    
    # Normality tests
    normality_tests: Dict[str, TestResult]
    
    # Independence tests
    independence_tests: Dict[str, TestResult]
    
    # Stationarity tests
    stationarity_tests: Dict[str, TestResult]
    
    # Autocorrelations
    acf: Optional[np.ndarray] = None
    pacf: Optional[np.ndarray] = None
    
    def summary(self) -> str:
        """Get diagnostics summary."""
        lines = ["Residuals Diagnostics:"]
        
        # Basic stats
        lines.append(f"\nMean: {self.mean:.6f}")
        lines.append(f"Std Dev: {self.std:.6f}")
        lines.append(f"Skewness: {self.skewness:.6f}")
        lines.append(f"Kurtosis: {self.kurtosis:.6f}")
        
        # Test results
        lines.append("\nNormality Tests:")
        for name, result in self.normality_tests.items():
            sig = "*" if result.pvalue < 0.05 else ""
            lines.append(f"  {name}: p-value = {result.pvalue:.4f} {sig}")
        
        lines.append("\nIndependence Tests:")
        for name, result in self.independence_tests.items():
            sig = "*" if result.pvalue < 0.05 else ""
            lines.append(f"  {name}: p-value = {result.pvalue:.4f} {sig}")
        
        return "\n".join(lines)


class NormalityTests:
    """Normality tests for residuals."""
    
    @staticmethod
    def test_all(residuals: TsData) -> Dict[str, TestResult]:
        """Run all normality tests.
        
        Args:
            residuals: Residual series
            
        Returns:
            Dictionary of test results
        """
        values = residuals.values[~np.isnan(residuals.values)]
        
        results = {}
        
        # Jarque-Bera test
        results["jarque_bera"] = JarqueBeraTest.test(values)
        
        # Skewness test
        results["skewness"] = SkewnessTest.test(values)
        
        # Kurtosis test
        results["kurtosis"] = KurtosisTest.test(values)
        
        # Shapiro-Wilk test (if sample size allows)
        if len(values) <= 5000:
            stat, pvalue = stats.shapiro(values)
            results["shapiro_wilk"] = TestResult(
                statistic=stat,
                pvalue=pvalue,
                description="Shapiro-Wilk test for normality"
            )
        
        return results


class IndependenceTests:
    """Independence tests for residuals."""
    
    @staticmethod
    def test_all(residuals: TsData, max_lag: Optional[int] = None) -> Dict[str, TestResult]:
        """Run all independence tests.
        
        Args:
            residuals: Residual series
            max_lag: Maximum lag for tests
            
        Returns:
            Dictionary of test results
        """
        values = residuals.values[~np.isnan(residuals.values)]
        n = len(values)
        
        if max_lag is None:
            # Default: min(20, n/4)
            max_lag = min(20, n // 4)
        
        results = {}
        
        # Ljung-Box test
        results["ljung_box"] = LjungBoxTest.test(values, max_lag)
        
        # Box-Pierce test
        results["box_pierce"] = BoxPierceTest.test(values, max_lag)
        
        # Runs test
        results["runs"] = RunsTest.test(values)
        
        # Turning points test
        results["turning_points"] = TurningPointsTest.test(values)
        
        return results


class StationarityTests:
    """Stationarity tests for residuals."""
    
    @staticmethod
    def test_all(residuals: TsData) -> Dict[str, TestResult]:
        """Run stationarity tests.
        
        Args:
            residuals: Residual series
            
        Returns:
            Dictionary of test results
        """
        values = residuals.values[~np.isnan(residuals.values)]
        
        results = {}
        
        # ADF test
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(values, autolag='AIC')
        results["adf"] = TestResult(
            statistic=adf_result[0],
            pvalue=adf_result[1],
            description="Augmented Dickey-Fuller test"
        )
        
        # KPSS test
        from statsmodels.tsa.stattools import kpss
        kpss_result = kpss(values, regression='c', nlags='auto')
        results["kpss"] = TestResult(
            statistic=kpss_result[0],
            pvalue=kpss_result[1],
            description="KPSS test"
        )
        
        return results


class RunsTest:
    """Runs test for randomness."""
    
    @staticmethod
    def test(data: np.ndarray) -> TestResult:
        """Perform runs test.
        
        Args:
            data: Data array
            
        Returns:
            Test result
        """
        n = len(data)
        median = np.median(data)
        
        # Convert to binary (above/below median)
        binary = data > median
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if binary[i] != binary[i-1]:
                runs += 1
        
        # Count positive and negative
        n_pos = np.sum(binary)
        n_neg = n - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return TestResult(
                statistic=np.nan,
                pvalue=np.nan,
                description="Runs test (degenerate case)"
            )
        
        # Expected runs and variance
        expected_runs = 2 * n_pos * n_neg / n + 1
        var_runs = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n) / (n**2 * (n - 1))
        
        # Z-statistic
        if var_runs > 0:
            z_stat = (runs - expected_runs) / np.sqrt(var_runs)
            pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = np.nan
            pvalue = np.nan
        
        return TestResult(
            statistic=z_stat,
            pvalue=pvalue,
            description="Runs test for randomness"
        )


class TurningPointsTest:
    """Turning points test for randomness."""
    
    @staticmethod
    def test(data: np.ndarray) -> TestResult:
        """Perform turning points test.
        
        Args:
            data: Data array
            
        Returns:
            Test result
        """
        n = len(data)
        
        # Count turning points
        turning_points = 0
        for i in range(1, n - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1]) or \
               (data[i] < data[i-1] and data[i] < data[i+1]):
                turning_points += 1
        
        # Expected turning points and variance for random series
        expected_tp = 2 * (n - 2) / 3
        var_tp = (16 * n - 29) / 90
        
        # Z-statistic
        if var_tp > 0:
            z_stat = (turning_points - expected_tp) / np.sqrt(var_tp)
            pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = np.nan
            pvalue = np.nan
        
        return TestResult(
            statistic=z_stat,
            pvalue=pvalue,
            description="Turning points test"
        )


def compute_residuals_diagnostics(residuals: TsData,
                                 max_lag: Optional[int] = None) -> ResidualsDiagnostics:
    """Compute comprehensive residuals diagnostics.
    
    Args:
        residuals: Residual series
        max_lag: Maximum lag for autocorrelation tests
        
    Returns:
        ResidualsDiagnostics object
    """
    values = residuals.values[~np.isnan(residuals.values)]
    
    # Basic statistics
    mean = np.mean(values)
    std = np.std(values)
    skewness = stats.skew(values)
    kurtosis = stats.kurtosis(values)
    
    # Run tests
    normality_tests = NormalityTests.test_all(residuals)
    independence_tests = IndependenceTests.test_all(residuals, max_lag)
    stationarity_tests = StationarityTests.test_all(residuals)
    
    # Compute ACF
    if max_lag is None:
        max_lag = min(40, len(values) // 4)
    acf = autocorrelations(values, max_lag)
    
    return ResidualsDiagnostics(
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        normality_tests=normality_tests,
        independence_tests=independence_tests,
        stationarity_tests=stationarity_tests,
        acf=acf
    )