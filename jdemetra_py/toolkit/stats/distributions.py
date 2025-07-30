"""Statistical distributions matching JDemetra+ implementation."""

from typing import Union
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod


class Distribution(ABC):
    """Base class for probability distributions."""
    
    @abstractmethod
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        pass
    
    @abstractmethod
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        pass
    
    @abstractmethod
    def ppf(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        pass
    
    @abstractmethod
    def random(self, size: int = 1, seed: int = None) -> np.ndarray:
        """Generate random samples."""
        pass


class Normal(Distribution):
    """Normal (Gaussian) distribution."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """Initialize normal distribution.
        
        Args:
            mean: Mean of distribution
            std: Standard deviation
        """
        self.mean = mean
        self.std = std
        self._dist = stats.norm(loc=mean, scale=std)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        return self._dist.cdf(x)
    
    def ppf(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        return self._dist.ppf(p)
    
    def random(self, size: int = 1, seed: int = None) -> np.ndarray:
        """Generate random samples."""
        if seed is not None:
            np.random.seed(seed)
        return self._dist.rvs(size=size)
    
    @staticmethod
    def standard() -> 'Normal':
        """Standard normal distribution N(0,1)."""
        return Normal(0.0, 1.0)
    
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Log probability density function."""
        return self._dist.logpdf(x)


class T(Distribution):
    """Student's t-distribution."""
    
    def __init__(self, df: float):
        """Initialize t-distribution.
        
        Args:
            df: Degrees of freedom
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df = df
        self._dist = stats.t(df=df)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        return self._dist.cdf(x)
    
    def ppf(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        return self._dist.ppf(p)
    
    def random(self, size: int = 1, seed: int = None) -> np.ndarray:
        """Generate random samples."""
        if seed is not None:
            np.random.seed(seed)
        return self._dist.rvs(size=size)
    
    @property
    def mean(self) -> float:
        """Mean of distribution."""
        return 0.0 if self.df > 1 else np.nan
    
    @property
    def variance(self) -> float:
        """Variance of distribution."""
        if self.df > 2:
            return self.df / (self.df - 2)
        else:
            return np.inf if self.df > 1 else np.nan


class Chi2(Distribution):
    """Chi-squared distribution."""
    
    def __init__(self, df: float):
        """Initialize chi-squared distribution.
        
        Args:
            df: Degrees of freedom
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df = df
        self._dist = stats.chi2(df=df)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        return self._dist.cdf(x)
    
    def ppf(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        return self._dist.ppf(p)
    
    def random(self, size: int = 1, seed: int = None) -> np.ndarray:
        """Generate random samples."""
        if seed is not None:
            np.random.seed(seed)
        return self._dist.rvs(size=size)
    
    @property
    def mean(self) -> float:
        """Mean of distribution."""
        return self.df
    
    @property
    def variance(self) -> float:
        """Variance of distribution."""
        return 2 * self.df


class F(Distribution):
    """F-distribution."""
    
    def __init__(self, df1: float, df2: float):
        """Initialize F-distribution.
        
        Args:
            df1: Numerator degrees of freedom
            df2: Denominator degrees of freedom
        """
        if df1 <= 0 or df2 <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df1 = df1
        self.df2 = df2
        self._dist = stats.f(dfn=df1, dfd=df2)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        return self._dist.cdf(x)
    
    def ppf(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        return self._dist.ppf(p)
    
    def random(self, size: int = 1, seed: int = None) -> np.ndarray:
        """Generate random samples."""
        if seed is not None:
            np.random.seed(seed)
        return self._dist.rvs(size=size)
    
    @property
    def mean(self) -> float:
        """Mean of distribution."""
        if self.df2 > 2:
            return self.df2 / (self.df2 - 2)
        else:
            return np.nan
    
    @property
    def variance(self) -> float:
        """Variance of distribution."""
        if self.df2 > 4:
            num = 2 * self.df2**2 * (self.df1 + self.df2 - 2)
            den = self.df1 * (self.df2 - 2)**2 * (self.df2 - 4)
            return num / den
        else:
            return np.nan


class Beta(Distribution):
    """Beta distribution."""
    
    def __init__(self, a: float, b: float):
        """Initialize beta distribution.
        
        Args:
            a: Shape parameter alpha
            b: Shape parameter beta
        """
        if a <= 0 or b <= 0:
            raise ValueError("Shape parameters must be positive")
        self.a = a
        self.b = b
        self._dist = stats.beta(a=a, b=b)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        return self._dist.cdf(x)
    
    def ppf(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        return self._dist.ppf(p)
    
    def random(self, size: int = 1, seed: int = None) -> np.ndarray:
        """Generate random samples."""
        if seed is not None:
            np.random.seed(seed)
        return self._dist.rvs(size=size)
    
    @property
    def mean(self) -> float:
        """Mean of distribution."""
        return self.a / (self.a + self.b)
    
    @property
    def variance(self) -> float:
        """Variance of distribution."""
        return (self.a * self.b) / ((self.a + self.b)**2 * (self.a + self.b + 1))


class Gamma(Distribution):
    """Gamma distribution."""
    
    def __init__(self, shape: float, scale: float = 1.0):
        """Initialize gamma distribution.
        
        Args:
            shape: Shape parameter (alpha)
            scale: Scale parameter (beta)
        """
        if shape <= 0 or scale <= 0:
            raise ValueError("Shape and scale must be positive")
        self.shape = shape
        self.scale = scale
        self._dist = stats.gamma(a=shape, scale=scale)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        return self._dist.cdf(x)
    
    def ppf(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        return self._dist.ppf(p)
    
    def random(self, size: int = 1, seed: int = None) -> np.ndarray:
        """Generate random samples."""
        if seed is not None:
            np.random.seed(seed)
        return self._dist.rvs(size=size)
    
    @property
    def mean(self) -> float:
        """Mean of distribution."""
        return self.shape * self.scale
    
    @property
    def variance(self) -> float:
        """Variance of distribution."""
        return self.shape * self.scale**2