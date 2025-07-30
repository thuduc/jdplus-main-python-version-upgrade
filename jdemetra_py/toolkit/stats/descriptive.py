"""Descriptive statistics utilities."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class DescriptiveStatistics:
    """Container for descriptive statistics."""
    
    n: int
    mean: float
    std: float
    variance: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float
    q1: float
    q3: float
    
    @classmethod
    def compute(cls, data: np.ndarray, nan_policy: str = 'omit') -> 'DescriptiveStatistics':
        """Compute descriptive statistics for data.
        
        Args:
            data: Input data
            nan_policy: How to handle NaN values ('omit', 'raise', 'propagate')
            
        Returns:
            DescriptiveStatistics object
        """
        # Handle NaN values
        if nan_policy == 'omit':
            data = data[~np.isnan(data)]
        elif nan_policy == 'raise' and np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        
        # Compute statistics
        n = len(data)
        
        if n == 0:
            # Return NaN for empty data
            return cls(
                n=0,
                mean=np.nan,
                std=np.nan,
                variance=np.nan,
                min=np.nan,
                max=np.nan,
                median=np.nan,
                skewness=np.nan,
                kurtosis=np.nan,
                q1=np.nan,
                q3=np.nan
            )
        
        return cls(
            n=n,
            mean=np.mean(data),
            std=np.std(data, ddof=1),  # Sample standard deviation
            variance=np.var(data, ddof=1),  # Sample variance
            min=np.min(data),
            max=np.max(data),
            median=np.median(data),
            skewness=stats.skew(data),
            kurtosis=stats.kurtosis(data, fisher=True),  # Excess kurtosis
            q1=np.percentile(data, 25),
            q3=np.percentile(data, 75)
        )
    
    @property
    def iqr(self) -> float:
        """Interquartile range."""
        return self.q3 - self.q1
    
    @property
    def range(self) -> float:
        """Range (max - min)."""
        return self.max - self.min
    
    @property
    def cv(self) -> float:
        """Coefficient of variation."""
        return self.std / abs(self.mean) if self.mean != 0 else np.nan
    
    @property
    def stderr(self) -> float:
        """Standard error of the mean."""
        return self.std / np.sqrt(self.n) if self.n > 0 else np.nan
    
    def summary(self) -> str:
        """Get text summary of statistics."""
        lines = [
            f"Count: {self.n}",
            f"Mean: {self.mean:.6f}",
            f"Std: {self.std:.6f}",
            f"Min: {self.min:.6f}",
            f"25%: {self.q1:.6f}",
            f"50%: {self.median:.6f}",
            f"75%: {self.q3:.6f}",
            f"Max: {self.max:.6f}",
            f"Skewness: {self.skewness:.6f}",
            f"Kurtosis: {self.kurtosis:.6f}"
        ]
        return "\n".join(lines)


def autocorrelations(data: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Compute autocorrelation function.
    
    Args:
        data: Time series data
        nlags: Number of lags
        
    Returns:
        Array of autocorrelations
    """
    # Remove mean
    data = data - np.mean(data)
    c0 = np.dot(data, data) / len(data)
    
    acf = np.zeros(nlags + 1)
    acf[0] = 1.0
    
    for k in range(1, nlags + 1):
        c_k = np.dot(data[:-k], data[k:]) / len(data)
        acf[k] = c_k / c0
    
    return acf


def partial_autocorrelations(data: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Compute partial autocorrelation function.
    
    Args:
        data: Time series data  
        nlags: Number of lags
        
    Returns:
        Array of partial autocorrelations
    """
    # Use statsmodels for PACF calculation
    from statsmodels.tsa.stattools import pacf
    return pacf(data, nlags=nlags)


def cross_correlation(x: np.ndarray, y: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Compute cross-correlation function.
    
    Args:
        x: First time series
        y: Second time series
        nlags: Number of lags (both positive and negative)
        
    Returns:
        Array of cross-correlations
    """
    # Ensure same length
    n = min(len(x), len(y))
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    
    # Normalize
    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
    
    ccf = np.zeros(2 * nlags + 1)
    
    # Negative lags
    for k in range(1, nlags + 1):
        ccf[nlags - k] = np.sum(x[k:] * y[:-k]) / norm
    
    # Zero lag
    ccf[nlags] = np.sum(x * y) / norm
    
    # Positive lags
    for k in range(1, nlags + 1):
        ccf[nlags + k] = np.sum(x[:-k] * y[k:]) / norm
    
    return ccf