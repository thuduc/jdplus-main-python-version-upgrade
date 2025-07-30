"""Numba JIT compilation extensions for performance."""

import numpy as np
from typing import Tuple, Optional, Callable
import numba
from numba import jit, prange
import warnings

# Global flag to enable/disable Numba
_NUMBA_ENABLED = True

def enable_numba():
    """Enable Numba JIT compilation."""
    global _NUMBA_ENABLED
    _NUMBA_ENABLED = True

def disable_numba():
    """Disable Numba JIT compilation."""
    global _NUMBA_ENABLED
    _NUMBA_ENABLED = False

def jit_compile(*args, **kwargs):
    """Conditional JIT compilation based on global flag.
    
    Returns:
        Numba JIT decorator or identity function
    """
    if _NUMBA_ENABLED:
        # Set default options
        kwargs.setdefault('nopython', True)
        kwargs.setdefault('cache', True)
        kwargs.setdefault('fastmath', True)
        return numba.jit(*args, **kwargs)
    else:
        # Return identity decorator
        def identity(func):
            return func
        return identity


# Core mathematical operations

@jit_compile()
def fast_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Fast dot product using Numba.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Dot product
    """
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


@jit_compile()
def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast matrix multiplication using Numba.
    
    Args:
        A: First matrix
        B: Second matrix
        
    Returns:
        Product matrix
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    
    C = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            for l in range(k):
                C[i, j] += A[i, l] * B[l, j]
    
    return C


@jit_compile(parallel=True)
def fast_matrix_multiply_parallel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Parallel matrix multiplication using Numba.
    
    Args:
        A: First matrix
        B: Second matrix
        
    Returns:
        Product matrix
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    
    C = np.zeros((m, n))
    
    for i in prange(m):
        for j in range(n):
            for l in range(k):
                C[i, j] += A[i, l] * B[l, j]
    
    return C


# Time series specific operations

@jit_compile()
def fast_autocorrelation(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Fast autocorrelation calculation.
    
    Args:
        data: Time series data
        max_lag: Maximum lag
        
    Returns:
        Autocorrelation values
    """
    n = len(data)
    mean = np.mean(data)
    c0 = 0.0
    
    # Variance
    for i in range(n):
        c0 += (data[i] - mean) ** 2
    c0 /= n
    
    # Autocorrelations
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    
    for lag in range(1, max_lag + 1):
        c_lag = 0.0
        for i in range(n - lag):
            c_lag += (data[i] - mean) * (data[i + lag] - mean)
        c_lag /= n
        acf[lag] = c_lag / c0
    
    return acf


@jit_compile()
def fast_levinson_durbin(acf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fast Levinson-Durbin recursion for PACF calculation.
    
    Args:
        acf: Autocorrelation function values
        
    Returns:
        PACF values and AR coefficients
    """
    n = len(acf) - 1
    pacf = np.zeros(n + 1)
    phi = np.zeros((n + 1, n + 1))
    
    pacf[0] = 1.0
    sigma2 = acf[0]
    
    for k in range(1, n + 1):
        # Calculate k-th reflection coefficient
        sum_term = 0.0
        for j in range(1, k):
            sum_term += phi[k-1, j] * acf[k-j]
        
        pacf[k] = (acf[k] - sum_term) / sigma2
        phi[k, k] = pacf[k]
        
        # Update coefficients
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - pacf[k] * phi[k-1, k-j]
        
        # Update variance
        sigma2 *= (1 - pacf[k] ** 2)
    
    return pacf, phi


@jit_compile()
def fast_difference(data: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    """Fast differencing operation for ARIMA.
    
    Args:
        data: Input data
        d: Regular differencing order
        D: Seasonal differencing order
        s: Seasonal period
        
    Returns:
        Differenced data
    """
    result = data.copy()
    
    # Regular differencing
    for _ in range(d):
        diff = np.empty(len(result) - 1)
        for i in range(len(diff)):
            diff[i] = result[i + 1] - result[i]
        result = diff
    
    # Seasonal differencing
    for _ in range(D):
        if len(result) > s:
            diff = np.empty(len(result) - s)
            for i in range(len(diff)):
                diff[i] = result[i + s] - result[i]
            result = diff
    
    return result


# ARIMA model operations

@jit_compile()
def fast_arma_filter(data: np.ndarray, 
                    ar_params: np.ndarray,
                    ma_params: np.ndarray) -> np.ndarray:
    """Fast ARMA filtering.
    
    Args:
        data: Input data (innovations)
        ar_params: AR parameters
        ma_params: MA parameters
        
    Returns:
        Filtered series
    """
    n = len(data)
    p = len(ar_params)
    q = len(ma_params)
    
    y = np.zeros(n)
    
    for t in range(n):
        # MA part
        y[t] = data[t]
        for i in range(min(t, q)):
            y[t] += ma_params[i] * data[t - i - 1]
        
        # AR part
        for i in range(min(t, p)):
            y[t] += ar_params[i] * y[t - i - 1]
    
    return y


@jit_compile()
def fast_arima_loglikelihood(data: np.ndarray,
                            ar_params: np.ndarray,
                            ma_params: np.ndarray,
                            sigma2: float,
                            d: int, D: int, s: int) -> float:
    """Fast ARIMA log-likelihood calculation.
    
    Args:
        data: Time series data
        ar_params: AR parameters
        ma_params: MA parameters
        sigma2: Innovation variance
        d: Regular differencing
        D: Seasonal differencing
        s: Seasonal period
        
    Returns:
        Log-likelihood value
    """
    # Difference data
    diff_data = fast_difference(data, d, D, s)
    n = len(diff_data)
    
    # Calculate residuals
    residuals = np.zeros(n)
    
    for t in range(n):
        # Prediction
        pred = 0.0
        
        # AR part
        for i in range(min(t, len(ar_params))):
            pred += ar_params[i] * diff_data[t - i - 1]
        
        # MA part (would need past residuals)
        # Simplified for demonstration
        
        residuals[t] = diff_data[t] - pred
    
    # Log-likelihood
    ll = -0.5 * n * np.log(2 * np.pi * sigma2)
    ll -= 0.5 * np.sum(residuals ** 2) / sigma2
    
    return ll


# Kalman filter operations

@jit_compile()
def fast_kalman_filter_step(y: float, 
                           a: np.ndarray, 
                           P: np.ndarray,
                           Z: np.ndarray, 
                           T: np.ndarray,
                           R: np.ndarray, 
                           Q: np.ndarray,
                           H: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fast Kalman filter step.
    
    Args:
        y: Observation
        a: State vector
        P: State covariance
        Z: Observation matrix
        T: Transition matrix
        R: Selection matrix
        Q: State noise covariance
        H: Observation noise variance
        
    Returns:
        Updated state, covariance, innovation, and innovation variance
    """
    # Prediction step
    a_pred = T @ a
    P_pred = T @ P @ T.T + R @ Q @ R.T
    
    # Innovation
    v = y - Z @ a_pred
    F = Z @ P_pred @ Z.T + H
    
    # Update step
    if F > 0:
        K = P_pred @ Z.T / F
        a_filt = a_pred + K * v
        P_filt = P_pred - np.outer(K, K) * F
    else:
        a_filt = a_pred
        P_filt = P_pred
    
    return a_filt, P_filt, v, F


@jit_compile()
def fast_kalman_smoother(y: np.ndarray,
                        a0: np.ndarray,
                        P0: np.ndarray,
                        Z: np.ndarray,
                        T: np.ndarray,
                        R: np.ndarray,
                        Q: np.ndarray,
                        H: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fast Kalman smoother.
    
    Args:
        y: Observations
        a0: Initial state
        P0: Initial covariance
        Z: Observation matrix
        T: Transition matrix
        R: Selection matrix
        Q: State noise covariance
        H: Observation noise variance
        
    Returns:
        Smoothed states and covariances
    """
    n = len(y)
    m = len(a0)
    
    # Forward pass - filtering
    a_filt = np.zeros((n + 1, m))
    P_filt = np.zeros((n + 1, m, m))
    
    a_filt[0] = a0
    P_filt[0] = P0
    
    for t in range(n):
        a_filt[t + 1], P_filt[t + 1], _, _ = fast_kalman_filter_step(
            y[t], a_filt[t], P_filt[t], Z, T, R, Q, H
        )
    
    # Backward pass - smoothing
    a_smooth = np.zeros((n, m))
    P_smooth = np.zeros((n, m, m))
    
    a_smooth[-1] = a_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    for t in range(n - 2, -1, -1):
        # Prediction at t+1
        a_pred = T @ a_filt[t]
        P_pred = T @ P_filt[t] @ T.T + R @ Q @ R.T
        
        # Smoother gain
        if np.linalg.det(P_pred) > 1e-10:
            L = P_filt[t] @ T.T @ np.linalg.inv(P_pred)
        else:
            L = np.zeros((m, m))
        
        # Smoothed estimates
        a_smooth[t] = a_filt[t] + L @ (a_smooth[t + 1] - a_pred)
        P_smooth[t] = P_filt[t] + L @ (P_smooth[t + 1] - P_pred) @ L.T
    
    return a_smooth, P_smooth


# Seasonal adjustment operations

@jit_compile()
def fast_x11_iteration(data: np.ndarray, 
                      period: int,
                      n_iterations: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast X-11 style decomposition iteration.
    
    Args:
        data: Input data
        period: Seasonal period
        n_iterations: Number of iterations
        
    Returns:
        Trend, seasonal, and irregular components
    """
    n = len(data)
    
    # Initialize
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    irregular = np.zeros(n)
    
    for iteration in range(n_iterations):
        # Step 1: Estimate trend (centered MA)
        if period % 2 == 0:
            # Even period - 2x(period) MA
            ma_len = 2 * period
        else:
            # Odd period - period MA
            ma_len = period
        
        # Simple MA for trend
        for i in range(ma_len // 2, n - ma_len // 2):
            trend[i] = np.mean(data[i - ma_len // 2:i + ma_len // 2 + 1])
        
        # Extend trend to edges
        for i in range(ma_len // 2):
            trend[i] = trend[ma_len // 2]
        for i in range(n - ma_len // 2, n):
            trend[i] = trend[n - ma_len // 2 - 1]
        
        # Step 2: Detrend and estimate seasonal
        detrended = data - trend
        
        # Seasonal means
        seasonal_means = np.zeros(period)
        counts = np.zeros(period)
        
        for i in range(n):
            seasonal_means[i % period] += detrended[i]
            counts[i % period] += 1
        
        for i in range(period):
            if counts[i] > 0:
                seasonal_means[i] /= counts[i]
        
        # Center seasonal
        seasonal_means -= np.mean(seasonal_means)
        
        # Apply seasonal pattern
        for i in range(n):
            seasonal[i] = seasonal_means[i % period]
        
        # Step 3: Calculate irregular
        irregular = data - trend - seasonal
        
        # For next iteration, modify data
        if iteration < n_iterations - 1:
            # Remove extreme values from irregular
            irr_std = np.std(irregular)
            for i in range(n):
                if abs(irregular[i]) > 3 * irr_std:
                    data[i] = trend[i] + seasonal[i]
    
    return trend, seasonal, irregular


# Optimization utilities

def create_numba_cache():
    """Create cache directory for Numba compiled functions."""
    import os
    cache_dir = os.path.expanduser("~/.jdemetra_py/numba_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def benchmark_numba_performance():
    """Benchmark Numba vs pure Python performance."""
    import time
    
    # Test data
    n = 10000
    data = np.random.randn(n)
    
    # Pure Python autocorrelation
    def python_acf(data, max_lag):
        n = len(data)
        mean = np.mean(data)
        c0 = np.var(data)
        acf = [1.0]
        
        for lag in range(1, max_lag + 1):
            c_lag = sum((data[i] - mean) * (data[i + lag] - mean) 
                       for i in range(n - lag)) / n
            acf.append(c_lag / c0)
        
        return acf
    
    # Benchmark
    max_lag = 50
    
    # Python version
    start = time.time()
    for _ in range(10):
        python_result = python_acf(data, max_lag)
    python_time = time.time() - start
    
    # Numba version
    start = time.time()
    for _ in range(10):
        numba_result = fast_autocorrelation(data, max_lag)
    numba_time = time.time() - start
    
    print(f"Python time: {python_time:.3f}s")
    print(f"Numba time: {numba_time:.3f}s")
    print(f"Speedup: {python_time / numba_time:.1f}x")
    
    return python_time / numba_time