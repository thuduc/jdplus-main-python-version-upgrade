"""Caching utilities for performance optimization."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple
import hashlib
import json
import weakref
from collections import OrderedDict
import numpy as np


class CacheManager:
    """Global cache manager."""
    
    _caches: Dict[str, weakref.WeakValueDictionary] = {}
    
    @classmethod
    def register_cache(cls, name: str, cache: dict):
        """Register a cache for management.
        
        Args:
            name: Cache name
            cache: Cache dictionary
        """
        cls._caches[name] = weakref.ref(cache)
    
    @classmethod
    def clear_cache(cls, name: str):
        """Clear specific cache.
        
        Args:
            name: Cache name
        """
        if name in cls._caches:
            cache_ref = cls._caches[name]
            cache = cache_ref()
            if cache is not None:
                cache.clear()
    
    @classmethod
    def clear_all_caches(cls):
        """Clear all registered caches."""
        for cache_ref in cls._caches.values():
            cache = cache_ref()
            if cache is not None:
                cache.clear()
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all caches.
        
        Returns:
            Cache information
        """
        info = {}
        for name, cache_ref in cls._caches.items():
            cache = cache_ref()
            if cache is not None:
                info[name] = {
                    'size': len(cache),
                    'type': type(cache).__name__
                }
        return info


def make_hashable(obj: Any) -> str:
    """Convert object to hashable string.
    
    Args:
        obj: Object to hash
        
    Returns:
        Hash string
    """
    if isinstance(obj, np.ndarray):
        # Hash array data
        return hashlib.md5(obj.tobytes()).hexdigest()
    elif isinstance(obj, (list, tuple)):
        # Recursively hash elements
        return hashlib.md5(
            json.dumps([make_hashable(x) for x in obj]).encode()
        ).hexdigest()
    elif isinstance(obj, dict):
        # Hash sorted items
        return hashlib.md5(
            json.dumps({k: make_hashable(v) for k, v in sorted(obj.items())}).encode()
        ).hexdigest()
    else:
        # Use string representation
        return hashlib.md5(str(obj).encode()).hexdigest()


def memoize(maxsize: Optional[int] = 128) -> Callable:
    """Memoization decorator with size limit.
    
    Args:
        maxsize: Maximum cache size (None for unlimited)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Use OrderedDict for LRU behavior
        cache = OrderedDict() if maxsize else {}
        cache_name = f"{func.__module__}.{func.__name__}"
        CacheManager.register_cache(cache_name, cache)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = make_hashable((args, sorted(kwargs.items())))
            
            # Check cache
            if key in cache:
                # Move to end (most recently used)
                if isinstance(cache, OrderedDict):
                    cache.move_to_end(key)
                return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Update cache
            cache[key] = result
            
            # Enforce size limit
            if maxsize and len(cache) > maxsize:
                # Remove oldest
                cache.popitem(last=False)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            'hits': 0,  # Would need to track
            'misses': 0,  # Would need to track
            'size': len(cache),
            'maxsize': maxsize
        }
        
        return wrapper
    
    return decorator


def lru_cache(maxsize: Optional[int] = 128) -> Callable:
    """LRU cache decorator (wrapper around functools.lru_cache).
    
    Args:
        maxsize: Maximum cache size
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Register with cache manager
        cache_name = f"{func.__module__}.{func.__name__}"
        
        # Use functools.lru_cache
        cached_func = functools.lru_cache(maxsize=maxsize)(func)
        
        # Register a proxy for management
        CacheManager.register_cache(cache_name, {})
        
        return cached_func
    
    return decorator


# Specialized caches for time series operations

class TsDataCache:
    """Cache for time series data operations."""
    
    def __init__(self, maxsize: int = 100):
        """Initialize cache.
        
        Args:
            maxsize: Maximum cache size
        """
        self._cache = OrderedDict()
        self._maxsize = maxsize
        CacheManager.register_cache("TsDataCache", self._cache)
    
    def get_or_compute(self, key: str, compute_func: Callable) -> Any:
        """Get from cache or compute.
        
        Args:
            key: Cache key
            compute_func: Function to compute value
            
        Returns:
            Cached or computed value
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        
        # Compute
        value = compute_func()
        
        # Store
        self._cache[key] = value
        
        # Enforce size limit
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        
        return value
    
    def invalidate(self, key: str):
        """Invalidate cache entry.
        
        Args:
            key: Cache key
        """
        self._cache.pop(key, None)
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()


# Global time series cache instance
_ts_cache = TsDataCache()


def cache_ts_operation(operation: str, series_id: str, *args) -> str:
    """Create cache key for time series operation.
    
    Args:
        operation: Operation name
        series_id: Series identifier
        *args: Additional arguments
        
    Returns:
        Cache key
    """
    key_parts = [operation, series_id] + [str(arg) for arg in args]
    return ":".join(key_parts)


# Cached implementations of expensive operations

@memoize(maxsize=256)
def cached_fft(data: np.ndarray) -> np.ndarray:
    """Cached FFT computation.
    
    Args:
        data: Input array
        
    Returns:
        FFT result
    """
    return np.fft.fft(data)


@memoize(maxsize=128)
def cached_matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """Cached matrix inverse.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Inverse matrix
    """
    return np.linalg.inv(matrix)


@memoize(maxsize=128)
def cached_polynomial_roots(coefficients: Tuple[float, ...]) -> np.ndarray:
    """Cached polynomial roots.
    
    Args:
        coefficients: Polynomial coefficients (as tuple for hashability)
        
    Returns:
        Roots array
    """
    return np.roots(coefficients)


# Cache decorators for specific use cases

def cache_seasonal_pattern(func: Callable) -> Callable:
    """Cache seasonal pattern extraction.
    
    Args:
        func: Function to cache
        
    Returns:
        Cached function
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(series, frequency: int):
        # Create key from series stats and frequency
        key = (
            series.length,
            frequency,
            float(np.mean(series.values)),
            float(np.std(series.values))
        )
        
        if key in cache:
            return cache[key]
        
        result = func(series, frequency)
        cache[key] = result
        
        # Limit cache size
        if len(cache) > 50:
            # Remove random entry
            cache.pop(next(iter(cache)))
        
        return result
    
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def cache_arima_estimation(func: Callable) -> Callable:
    """Cache ARIMA model estimation results.
    
    Args:
        func: Estimation function
        
    Returns:
        Cached function
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(series, order, **kwargs):
        # Create key from series stats and order
        series_key = (
            series.length,
            float(np.mean(series.values)),
            float(np.std(series.values)),
            float(np.min(series.values)),
            float(np.max(series.values))
        )
        
        key = (series_key, order, tuple(sorted(kwargs.items())))
        
        if key in cache:
            return cache[key]
        
        result = func(series, order, **kwargs)
        cache[key] = result
        
        # Limit cache size
        if len(cache) > 20:
            # Remove oldest
            cache.pop(next(iter(cache)))
        
        return result
    
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


# Utility functions

def clear_all_caches():
    """Clear all caches in the system."""
    CacheManager.clear_all_caches()
    
    # Clear module-level caches
    _ts_cache.clear()
    
    # Clear functools caches
    import gc
    gc.collect()


def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics.
    
    Returns:
        Dictionary of cache statistics
    """
    stats = {
        'caches': CacheManager.get_cache_info(),
        'ts_cache_size': len(_ts_cache._cache)
    }
    
    return stats