"""Performance optimization utilities."""

from .caching import (
    memoize,
    lru_cache,
    CacheManager,
    clear_all_caches
)
from .vectorization import (
    vectorize_operation,
    batch_process,
    parallel_map
)
from .numba_extensions import (
    jit_compile,
    enable_numba,
    disable_numba
)

__all__ = [
    # Caching
    "memoize",
    "lru_cache",
    "CacheManager",
    "clear_all_caches",
    # Vectorization
    "vectorize_operation",
    "batch_process",
    "parallel_map",
    # Numba
    "jit_compile",
    "enable_numba",
    "disable_numba",
]