"""Unit tests for statistical functions."""

import pytest
import numpy as np

from jdemetra_py.toolkit.stats import (
    Normal, T, Chi2, F,
    LjungBoxTest, BoxPierceTest, JarqueBeraTest,
    SkewnessTest, KurtosisTest,
    DescriptiveStatistics
)


class TestDistributions:
    """Tests for probability distributions."""
    
    def test_normal(self):
        # Standard normal
        n = Normal.standard()
        assert n.mean == 0.0
        assert n.std == 1.0
        
        # PDF at 0
        assert n.pdf(0) == pytest.approx(0.3989422804)
        
        # CDF
        assert n.cdf(0) == pytest.approx(0.5)
        assert n.cdf(1.96) == pytest.approx(0.975, rel=1e-3)
        
        # Quantiles
        assert n.ppf(0.5) == pytest.approx(0.0)
        assert n.ppf(0.975) == pytest.approx(1.96, rel=1e-3)
        
        # Non-standard normal
        n2 = Normal(10, 2)
        assert n2.mean == 10
        assert n2.std == 2
        assert n2.pdf(10) == pytest.approx(n.pdf(0) / 2)
        
    def test_t_distribution(self):
        # T distribution with 10 df
        t = T(10)
        
        # Should be close to normal for large df
        assert t.pdf(0) == pytest.approx(Normal.standard().pdf(0), rel=0.1)
        
        # Heavier tails than normal
        assert t.pdf(3) > Normal.standard().pdf(3)
        
        # Properties
        assert t.mean == 0.0
        assert t.variance == pytest.approx(10.0 / 8.0)
        
        # T with 1 df (Cauchy) has undefined mean
        t1 = T(1)
        assert np.isnan(t1.mean)
        
    def test_chi2(self):
        # Chi-squared with 5 df
        chi = Chi2(5)
        
        assert chi.mean == 5.0
        assert chi.variance == 10.0
        
        # CDF properties
        assert chi.cdf(0) == 0.0
        assert 0 < chi.cdf(5) < 1
        
        # Critical values
        assert chi.ppf(0.95) == pytest.approx(11.07, rel=0.01)
        
    def test_f_distribution(self):
        # F distribution
        f = F(5, 10)
        
        # Check mean exists for df2 > 2
        assert f.mean == pytest.approx(10.0 / 8.0)
        
        # PDF properties
        assert f.pdf(0) == 0.0
        assert f.pdf(1) > 0
        
        # CDF properties
        assert f.cdf(0) == 0.0
        assert 0 < f.cdf(1) < 1


class TestStatisticalTests:
    """Tests for statistical hypothesis tests."""
    
    def test_ljung_box(self):
        # Generate white noise
        np.random.seed(42)
        white_noise = np.random.randn(100)
        
        # Should not reject for white noise
        result = LjungBoxTest.test(white_noise, lags=10)
        assert result.pvalue > 0.05
        assert result.df == 10
        
        # Generate autocorrelated series
        ar_series = np.zeros(100)
        ar_series[0] = np.random.randn()
        for i in range(1, 100):
            ar_series[i] = 0.8 * ar_series[i-1] + np.random.randn()
        
        # Should reject for autocorrelated series
        result = LjungBoxTest.test(ar_series, lags=10)
        assert result.pvalue < 0.05
        
    def test_box_pierce(self):
        # Similar to Ljung-Box but different statistic
        np.random.seed(42)
        white_noise = np.random.randn(100)
        
        result = BoxPierceTest.test(white_noise, lags=10)
        assert result.pvalue > 0.05
        
        # Should give similar but not identical results to Ljung-Box
        lb_result = LjungBoxTest.test(white_noise, lags=10)
        assert abs(result.pvalue - lb_result.pvalue) < 0.1
        
    def test_jarque_bera(self):
        # Normal data should not reject
        np.random.seed(42)
        normal_data = np.random.randn(100)
        
        result = JarqueBeraTest.test(normal_data)
        assert result.pvalue > 0.05
        assert result.df == 2
        
        # Skewed data should reject
        skewed_data = np.random.exponential(1, 100)
        result = JarqueBeraTest.test(skewed_data)
        assert result.pvalue < 0.05
        
    def test_skewness(self):
        # Symmetric data
        np.random.seed(42)
        symmetric = np.random.randn(100)
        
        result = SkewnessTest.test(symmetric)
        assert result.pvalue > 0.05
        
        # Skewed data
        skewed = np.random.exponential(1, 100)
        result = SkewnessTest.test(skewed)
        assert result.pvalue < 0.05
        assert result.statistic > 0  # Positive skew
        
    def test_kurtosis(self):
        # Normal data (excess kurtosis ≈ 0)
        np.random.seed(42)
        normal = np.random.randn(100)
        
        result = KurtosisTest.test(normal)
        assert result.pvalue > 0.05
        
        # Heavy-tailed data
        heavy_tailed = np.random.standard_t(3, 100)
        result = KurtosisTest.test(heavy_tailed)
        # May or may not reject depending on sample


class TestDescriptiveStatistics:
    """Tests for descriptive statistics."""
    
    def test_basic_stats(self):
        data = np.array([1, 2, 3, 4, 5])
        
        stats = DescriptiveStatistics.compute(data)
        
        assert stats.n == 5
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.std == pytest.approx(np.sqrt(2.5))
        
    def test_quartiles(self):
        data = np.arange(1, 101)  # 1 to 100
        
        stats = DescriptiveStatistics.compute(data)
        
        assert stats.q1 == 25.75  # numpy's linear interpolation
        assert stats.median == 50.5
        assert stats.q3 == 75.25
        assert stats.iqr == pytest.approx(49.5)
        
    def test_higher_moments(self):
        # Normal data
        np.random.seed(42)
        normal_data = np.random.randn(1000)
        
        stats = DescriptiveStatistics.compute(normal_data)
        
        # Should be close to 0 for normal
        assert abs(stats.skewness) < 0.2
        assert abs(stats.kurtosis) < 0.5
        
        # Skewed data
        skewed_data = np.random.exponential(1, 1000)
        stats = DescriptiveStatistics.compute(skewed_data)
        
        assert stats.skewness > 1.5  # Positive skew
        assert stats.kurtosis > 2.0  # Leptokurtic
        
    def test_nan_handling(self):
        data = np.array([1, 2, np.nan, 3, 4])
        
        # Omit NaN
        stats = DescriptiveStatistics.compute(data, nan_policy='omit')
        assert stats.n == 4
        assert stats.mean == 2.5
        
        # Raise on NaN
        with pytest.raises(ValueError):
            DescriptiveStatistics.compute(data, nan_policy='raise')
            
        # Propagate NaN
        stats = DescriptiveStatistics.compute(data, nan_policy='propagate')
        assert np.isnan(stats.mean)
        
    def test_derived_stats(self):
        data = np.array([10, 20, 30, 40, 50])
        
        stats = DescriptiveStatistics.compute(data)
        
        # Coefficient of variation
        assert stats.cv == pytest.approx(stats.std / stats.mean)
        
        # Standard error
        assert stats.stderr == pytest.approx(stats.std / np.sqrt(5))
        
        # Range
        assert stats.range == 40