"""Vectorization utilities for performance optimization."""

import numpy as np
from typing import Callable, List, Any, Optional, Union, Tuple
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numba


def vectorize_operation(func: Callable, 
                       inputs: Union[List, np.ndarray],
                       output_shape: Optional[Tuple] = None,
                       dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Vectorize a scalar operation over array inputs.
    
    Args:
        func: Scalar function to vectorize
        inputs: Input array or list
        output_shape: Expected output shape (inferred if None)
        dtype: Output data type (inferred if None)
        
    Returns:
        Vectorized result
    """
    # Convert to array
    if not isinstance(inputs, np.ndarray):
        inputs = np.array(inputs)
    
    # Determine output shape and dtype
    if output_shape is None or dtype is None:
        # Test with first element
        test_result = func(inputs.flat[0])
        if dtype is None:
            dtype = np.array(test_result).dtype
        if output_shape is None:
            if np.isscalar(test_result):
                output_shape = inputs.shape
            else:
                output_shape = inputs.shape + np.array(test_result).shape
    
    # Create output array
    output = np.empty(output_shape, dtype=dtype)
    
    # Apply function
    if np.isscalar(func(inputs.flat[0])):
        # Scalar output
        output.flat[:] = [func(x) for x in inputs.flat]
    else:
        # Array output
        for i, x in enumerate(inputs.flat):
            output[i] = func(x)
    
    return output


def batch_process(func: Callable,
                 items: List[Any],
                 batch_size: int = 100,
                 progress_callback: Optional[Callable] = None) -> List[Any]:
    """Process items in batches for better memory efficiency.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        batch_size: Size of each batch
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of results
    """
    results = []
    n_items = len(items)
    n_batches = (n_items + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_items)
        batch = items[start:end]
        
        # Process batch
        batch_results = [func(item) for item in batch]
        results.extend(batch_results)
        
        # Progress update
        if progress_callback:
            progress = (i + 1) / n_batches
            progress_callback(progress)
    
    return results


def parallel_map(func: Callable,
                items: List[Any],
                n_workers: Optional[int] = None,
                use_threads: bool = False,
                chunksize: Optional[int] = None) -> List[Any]:
    """Parallel map operation.
    
    Args:
        func: Function to apply
        items: Items to process
        n_workers: Number of workers (defaults to CPU count)
        use_threads: Use threads instead of processes
        chunksize: Size of chunks for each worker
        
    Returns:
        List of results
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if chunksize is None:
        chunksize = max(1, len(items) // (n_workers * 4))
    
    # Choose executor
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with Executor(max_workers=n_workers) as executor:
        results = list(executor.map(func, items, chunksize=chunksize))
    
    return results


# Vectorized time series operations

def vectorized_lag(data: np.ndarray, lag: int) -> np.ndarray:
    """Vectorized lag operation.
    
    Args:
        data: Input array
        lag: Lag order
        
    Returns:
        Lagged array
    """
    if lag == 0:
        return data.copy()
    
    result = np.empty_like(data)
    
    if lag > 0:
        # Positive lag
        result[:lag] = np.nan
        result[lag:] = data[:-lag]
    else:
        # Negative lag (lead)
        result[lag:] = np.nan
        result[:lag] = data[-lag:]
    
    return result


def vectorized_diff(data: np.ndarray, 
                   order: int = 1,
                   seasonal: Optional[int] = None) -> np.ndarray:
    """Vectorized differencing operation.
    
    Args:
        data: Input array
        order: Differencing order
        seasonal: Seasonal differencing period
        
    Returns:
        Differenced array
    """
    result = data.copy()
    
    # Regular differencing
    for _ in range(order):
        result = np.diff(result, prepend=np.nan)
    
    # Seasonal differencing
    if seasonal is not None and seasonal > 1:
        seasonal_diff = np.empty_like(result)
        seasonal_diff[:seasonal] = np.nan
        seasonal_diff[seasonal:] = result[seasonal:] - result[:-seasonal]
        result = seasonal_diff
    
    return result


def vectorized_ma(data: np.ndarray, window: int) -> np.ndarray:
    """Vectorized moving average.
    
    Args:
        data: Input array
        window: Window size
        
    Returns:
        Moving average array
    """
    if window == 1:
        return data.copy()
    
    # Use convolution for efficiency
    kernel = np.ones(window) / window
    
    # Pad data
    pad_width = window // 2
    padded = np.pad(data, pad_width, mode='edge')
    
    # Convolve
    result = np.convolve(padded, kernel, mode='valid')
    
    # Handle edges
    if len(result) > len(data):
        # Center the result
        extra = len(result) - len(data)
        start = extra // 2
        result = result[start:start + len(data)]
    
    return result


# Vectorized statistical operations

def vectorized_acf(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Vectorized autocorrelation function.
    
    Args:
        data: Input array
        max_lag: Maximum lag
        
    Returns:
        ACF values
    """
    # Remove mean
    data_centered = data - np.mean(data)
    c0 = np.dot(data_centered, data_centered) / len(data)
    
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    
    for k in range(1, max_lag + 1):
        c_k = np.dot(data_centered[:-k], data_centered[k:]) / len(data)
        acf[k] = c_k / c0
    
    return acf


def vectorized_pacf(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Vectorized partial autocorrelation function.
    
    Args:
        data: Input array
        max_lag: Maximum lag
        
    Returns:
        PACF values
    """
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    
    # Use Yule-Walker equations
    acf = vectorized_acf(data, max_lag)
    
    for k in range(1, max_lag + 1):
        # Build Toeplitz matrix
        r = acf[:k]
        R = np.array([[acf[abs(i-j)] for j in range(k)] for i in range(k)])
        
        try:
            # Solve for PACF
            phi = np.linalg.solve(R, r[1:])
            pacf[k] = phi[-1]
        except np.linalg.LinAlgError:
            pacf[k] = 0.0
    
    return pacf


# Numba-accelerated functions

@numba.jit(nopython=True, cache=True)
def fast_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Fast moving average using Numba.
    
    Args:
        data: Input array
        window: Window size
        
    Returns:
        Moving average
    """
    n = len(data)
    result = np.empty(n)
    
    # Initial window
    window_sum = 0.0
    for i in range(window):
        window_sum += data[i]
    result[window-1] = window_sum / window
    
    # Sliding window
    for i in range(window, n):
        window_sum = window_sum - data[i-window] + data[i]
        result[i] = window_sum / window
    
    # Fill initial values
    for i in range(window-1):
        result[i] = np.nan
    
    return result


@numba.jit(nopython=True, cache=True)
def fast_seasonal_decomp(data: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fast seasonal decomposition using Numba.
    
    Args:
        data: Input array
        period: Seasonal period
        
    Returns:
        Trend and seasonal components
    """
    n = len(data)
    
    # Extract trend with centered MA
    if period % 2 == 0:
        # Even period - use 2x(period)+1 MA
        trend = fast_moving_average(data, period)
        trend = fast_moving_average(trend, 2)
    else:
        # Odd period - use period MA
        trend = fast_moving_average(data, period)
    
    # Detrend
    detrended = np.empty(n)
    for i in range(n):
        if not np.isnan(trend[i]):
            detrended[i] = data[i] - trend[i]
        else:
            detrended[i] = np.nan
    
    # Extract seasonal pattern
    seasonal = np.zeros(n)
    seasonal_means = np.zeros(period)
    counts = np.zeros(period)
    
    # Calculate seasonal means
    for i in range(n):
        if not np.isnan(detrended[i]):
            seasonal_means[i % period] += detrended[i]
            counts[i % period] += 1
    
    # Average
    for i in range(period):
        if counts[i] > 0:
            seasonal_means[i] /= counts[i]
    
    # Center seasonal means
    mean_seasonal = np.sum(seasonal_means) / period
    for i in range(period):
        seasonal_means[i] -= mean_seasonal
    
    # Apply seasonal pattern
    for i in range(n):
        seasonal[i] = seasonal_means[i % period]
    
    return trend, seasonal


# Optimized batch operations for time series

class VectorizedTsOperations:
    """Collection of vectorized time series operations."""
    
    @staticmethod
    def batch_transform(series_list: List[np.ndarray],
                       transform: Callable,
                       n_workers: Optional[int] = None) -> List[np.ndarray]:
        """Apply transformation to multiple series in parallel.
        
        Args:
            series_list: List of series arrays
            transform: Transformation function
            n_workers: Number of parallel workers
            
        Returns:
            List of transformed series
        """
        return parallel_map(transform, series_list, n_workers=n_workers)
    
    @staticmethod
    def batch_decompose(series_list: List[np.ndarray],
                       period: int,
                       n_workers: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Decompose multiple series in parallel.
        
        Args:
            series_list: List of series arrays
            period: Seasonal period
            n_workers: Number of parallel workers
            
        Returns:
            List of (trend, seasonal) tuples
        """
        decomp_func = partial(fast_seasonal_decomp, period=period)
        return parallel_map(decomp_func, series_list, n_workers=n_workers)
    
    @staticmethod
    def batch_forecast(models: List[Any],
                      series_list: List[np.ndarray],
                      n_ahead: int,
                      n_workers: Optional[int] = None) -> List[np.ndarray]:
        """Generate forecasts for multiple series in parallel.
        
        Args:
            models: List of fitted models
            series_list: List of series arrays
            n_ahead: Forecast horizon
            n_workers: Number of parallel workers
            
        Returns:
            List of forecast arrays
        """
        def forecast_one(model_series):
            model, series = model_series
            # Model-specific forecast logic
            return model.forecast(series, n_ahead)
        
        return parallel_map(forecast_one, zip(models, series_list), 
                          n_workers=n_workers)