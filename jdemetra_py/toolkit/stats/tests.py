"""Statistical tests matching JDemetra+ implementation."""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from scipy import stats

from .distributions import Normal, Chi2


@dataclass
class TestResult:
    """Result of a statistical test."""
    statistic: float
    pvalue: float
    df: Optional[int] = None
    description: str = ""
    critical_value: Optional[float] = None
    test_name: Optional[str] = None
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is significant at given level."""
        return self.pvalue < alpha


class LjungBoxTest:
    """Ljung-Box test for autocorrelation."""
    
    @staticmethod
    def test(residuals: np.ndarray, lags: int, df_adjust: int = 0) -> TestResult:
        """Perform Ljung-Box test.
        
        Args:
            residuals: Residual series
            lags: Number of lags to test
            df_adjust: Degrees of freedom adjustment (e.g., for ARMA models)
            
        Returns:
            Test result
        """
        n = len(residuals)
        
        # Compute autocorrelations
        acf = np.array([np.corrcoef(residuals[:-i], residuals[i:])[0, 1] 
                       for i in range(1, lags + 1)])
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
        
        # Degrees of freedom
        df = lags - df_adjust
        
        # P-value from chi-squared distribution
        pvalue = 1 - Chi2(df).cdf(lb_stat)
        
        return TestResult(
            statistic=lb_stat,
            pvalue=pvalue,
            df=df,
            description="Ljung-Box test for autocorrelation"
        )


class BoxPierceTest:
    """Box-Pierce test for autocorrelation."""
    
    @staticmethod
    def test(residuals: np.ndarray, lags: int, df_adjust: int = 0) -> TestResult:
        """Perform Box-Pierce test.
        
        Args:
            residuals: Residual series
            lags: Number of lags to test
            df_adjust: Degrees of freedom adjustment
            
        Returns:
            Test result
        """
        n = len(residuals)
        
        # Compute autocorrelations
        acf = np.array([np.corrcoef(residuals[:-i], residuals[i:])[0, 1] 
                       for i in range(1, lags + 1)])
        
        # Box-Pierce statistic
        bp_stat = n * np.sum(acf**2)
        
        # Degrees of freedom
        df = lags - df_adjust
        
        # P-value from chi-squared distribution
        pvalue = 1 - Chi2(df).cdf(bp_stat)
        
        return TestResult(
            statistic=bp_stat,
            pvalue=pvalue,
            df=df,
            description="Box-Pierce test for autocorrelation"
        )


class JarqueBeraTest:
    """Jarque-Bera test for normality."""
    
    @staticmethod
    def test(data: np.ndarray) -> TestResult:
        """Perform Jarque-Bera test.
        
        Args:
            data: Data to test
            
        Returns:
            Test result
        """
        n = len(data)
        
        # Compute skewness and kurtosis
        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=True)  # Excess kurtosis
        
        # Jarque-Bera statistic
        jb_stat = n / 6 * (skew**2 + kurt**2 / 4)
        
        # P-value from chi-squared distribution with 2 df
        pvalue = 1 - Chi2(2).cdf(jb_stat)
        
        return TestResult(
            statistic=jb_stat,
            pvalue=pvalue,
            df=2,
            description="Jarque-Bera test for normality"
        )


class DoornikHansenTest:
    """Doornik-Hansen test for multivariate normality."""
    
    @staticmethod
    def test(data: np.ndarray) -> TestResult:
        """Perform Doornik-Hansen test.
        
        Args:
            data: Data matrix (n x p)
            
        Returns:
            Test result
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n, p = data.shape
        
        # Standardize data
        data_std = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        
        # Compute skewness and kurtosis for each variable
        skew = stats.skew(data_std, axis=0)
        kurt = stats.kurtosis(data_std, axis=0, fisher=True)
        
        # Transform skewness
        beta = 3 * (n**2 + 27*n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
        w2 = -1 + np.sqrt(2 * (beta - 1))
        delta = 1 / np.sqrt(np.log(np.sqrt(w2)))
        y = skew * np.sqrt((w2 - 1) * (n + 1) * (n + 3) / (12 * (n - 2)))
        z1 = delta * np.log(y + np.sqrt(y**2 + 1))
        
        # Transform kurtosis  
        delta2 = (n - 3) * (n + 1) * (n**2 + 15*n - 4)
        a = ((n - 2) * (n + 5) * (n + 7) * (n**2 + 27*n - 70)) / (6 * delta2)
        c = ((n - 7) * (n + 5) * (n + 7) * (n**2 + 2*n - 5)) / (6 * delta2)
        k = ((n + 5) * (n + 7) * (n**3 + 37*n**2 + 11*n - 313)) / (12 * delta2)
        alpha = a + skew**2 * c
        chi = (kurt - 1 - skew**2) * 2 * k
        z2 = (((chi / (2 * alpha))**(1/3)) - 1 + 1/(9 * alpha)) * np.sqrt(9 * alpha)
        
        # Doornik-Hansen statistic
        dh_stat = np.sum(z1**2) + np.sum(z2**2)
        
        # P-value from chi-squared distribution
        pvalue = 1 - Chi2(2 * p).cdf(dh_stat)
        
        return TestResult(
            statistic=dh_stat,
            pvalue=pvalue,
            df=2 * p,
            description="Doornik-Hansen test for normality"
        )


class SkewnessTest:
    """Test for skewness."""
    
    @staticmethod
    def test(data: np.ndarray) -> TestResult:
        """Test if skewness is significantly different from zero.
        
        Args:
            data: Data to test
            
        Returns:
            Test result
        """
        n = len(data)
        
        # Compute skewness
        skew = stats.skew(data)
        
        # Standard error of skewness
        se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        
        # Test statistic (approximately normal)
        z_stat = skew / se_skew
        
        # Two-tailed p-value
        pvalue = 2 * (1 - Normal.standard().cdf(abs(z_stat)))
        
        return TestResult(
            statistic=z_stat,
            pvalue=pvalue,
            description="Test for skewness"
        )


class KurtosisTest:
    """Test for excess kurtosis."""
    
    @staticmethod
    def test(data: np.ndarray) -> TestResult:
        """Test if kurtosis is significantly different from zero.
        
        Args:
            data: Data to test
            
        Returns:
            Test result
        """
        n = len(data)
        
        # Compute excess kurtosis
        kurt = stats.kurtosis(data, fisher=True)
        
        # Standard error of kurtosis
        se_kurt = np.sqrt(24 * n * (n - 1)**2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5)))
        
        # Test statistic (approximately normal)
        z_stat = kurt / se_kurt
        
        # Two-tailed p-value
        pvalue = 2 * (1 - Normal.standard().cdf(abs(z_stat)))
        
        return TestResult(
            statistic=z_stat,
            pvalue=pvalue,
            description="Test for excess kurtosis"
        )