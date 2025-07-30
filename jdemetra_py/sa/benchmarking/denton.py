"""Denton benchmarking method."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional

from ...toolkit.timeseries import TsData
from .base import BenchmarkingProcessor, BenchmarkingResults, BenchmarkingMethod


class DentonBenchmarking(BenchmarkingProcessor):
    """Denton benchmarking processor.
    
    The Denton method minimizes the sum of squared differences in the 
    adjustment factors while respecting the aggregation constraints.
    """
    
    def __init__(self, differencing: int = 1, modified: bool = True):
        """Initialize Denton benchmarking.
        
        Args:
            differencing: Order of differencing (1 or 2)
            modified: Use modified Denton method
        """
        if differencing not in [1, 2]:
            raise ValueError("Differencing must be 1 or 2")
        
        self.differencing = differencing
        self.modified = modified
    
    def benchmark(self, series: TsData, target: TsData, 
                 initial_values: Optional[np.ndarray] = None) -> BenchmarkingResults:
        """Benchmark series using Denton method.
        
        Args:
            series: High-frequency series to be benchmarked
            target: Low-frequency target constraints
            initial_values: Initial values for modified Denton
            
        Returns:
            Benchmarking results
        """
        # Validate inputs
        self._validate_inputs(series, target)
        
        # Get values
        x = series.values.copy()
        y = target.values
        
        # Handle missing values
        x_mask = ~np.isnan(x)
        y_mask = ~np.isnan(y)
        
        # Create aggregation matrix
        C = self._create_aggregation_matrix(series, target)
        
        # Apply Denton method
        if self.modified and initial_values is not None:
            x_benchmarked = self._modified_denton(x, y, C, initial_values, 
                                                 x_mask, y_mask)
        else:
            x_benchmarked = self._original_denton(x, y, C, x_mask, y_mask)
        
        # Create benchmarked series
        benchmarked = TsData.of(series.start, x_benchmarked)
        
        # Compute adjustment factors
        adj_factors = np.where(x != 0, x_benchmarked / x, 1.0)
        
        # Create results
        results = BenchmarkingResults(
            original=series,
            benchmarked=benchmarked,
            target=target,
            method=BenchmarkingMethod.DENTON,
            adjustment_factors=adj_factors
        )
        
        # Compute diagnostics
        results.diagnostics['movement_preservation'] = self._compute_movement_preservation(
            x, x_benchmarked
        )
        
        return results
    
    def _original_denton(self, x: np.ndarray, y: np.ndarray, C: np.ndarray,
                        x_mask: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
        """Apply original Denton method.
        
        Args:
            x: High-frequency series values
            y: Low-frequency target values
            C: Aggregation matrix
            x_mask: Mask for valid x values
            y_mask: Mask for valid y values
            
        Returns:
            Benchmarked series
        """
        n = len(x)
        m = len(y)
        
        # Build difference matrix D
        if self.differencing == 1:
            # First differences
            D = sparse.diags([1, -1], [0, 1], shape=(n-1, n))
        else:
            # Second differences
            D = sparse.diags([-1, 2, -1], [0, 1, 2], shape=(n-2, n))
        
        # Build system of equations
        # Minimize: (x_new - x)'(x_new - x) subject to C*x_new = y
        
        # Use Lagrangian approach
        # L = (x_new - x)'(x_new - x) + lambda'(C*x_new - y)
        
        # First order conditions lead to:
        # [2*I   C'] [x_new  ] = [2*x]
        # [C     0 ] [lambda ] = [y  ]
        
        # Only use valid constraints
        C_valid = C[y_mask, :][:, x_mask]
        y_valid = y[y_mask]
        
        if len(y_valid) == 0:
            # No valid constraints
            return x
        
        # Build augmented system
        n_valid = np.sum(x_mask)
        m_valid = len(y_valid)
        
        A = sparse.bmat([
            [2.0 * sparse.eye(n_valid), C_valid.T],
            [C_valid, None]
        ])
        
        b = np.concatenate([2.0 * x[x_mask], y_valid])
        
        # Solve system
        solution = spsolve(A.tocsr(), b)
        
        # Extract benchmarked values
        x_new = x.copy()
        x_new[x_mask] = solution[:n_valid]
        
        return x_new
    
    def _modified_denton(self, x: np.ndarray, y: np.ndarray, C: np.ndarray,
                        initial: np.ndarray, x_mask: np.ndarray, 
                        y_mask: np.ndarray) -> np.ndarray:
        """Apply modified Denton method.
        
        Args:
            x: High-frequency series values
            y: Low-frequency target values
            C: Aggregation matrix
            initial: Initial values
            x_mask: Mask for valid x values
            y_mask: Mask for valid y values
            
        Returns:
            Benchmarked series
        """
        n = len(x)
        
        # Modified Denton minimizes changes in ratios
        # Minimize: sum((x_new[i]/initial[i] - x_new[i-1]/initial[i-1])^2)
        
        # This can be reformulated as a quadratic program
        # For simplicity, use proportional adjustment initially
        
        # Compute initial aggregates
        y_initial = C @ initial
        
        # Compute adjustment factors
        factors = np.zeros(len(y))
        for i in range(len(y)):
            if y_mask[i] and y_initial[i] != 0:
                factors[i] = y[i] / y_initial[i]
            else:
                factors[i] = 1.0
        
        # Distribute factors to high frequency
        x_new = x.copy()
        
        ratio = len(x) // len(y)
        for i in range(len(y)):
            start_idx = i * ratio
            end_idx = min(start_idx + ratio, n)
            x_new[start_idx:end_idx] *= factors[i]
        
        return x_new
    
    def _compute_movement_preservation(self, original: np.ndarray, 
                                     benchmarked: np.ndarray) -> float:
        """Compute movement preservation statistic.
        
        Args:
            original: Original series
            benchmarked: Benchmarked series
            
        Returns:
            Movement preservation measure (0-1, higher is better)
        """
        # Compute period-to-period changes
        orig_changes = np.diff(original)
        bench_changes = np.diff(benchmarked)
        
        # Remove invalid values
        valid = ~(np.isnan(orig_changes) | np.isnan(bench_changes))
        
        if np.sum(valid) == 0:
            return np.nan
        
        # Compute correlation of changes
        correlation = np.corrcoef(orig_changes[valid], bench_changes[valid])[0, 1]
        
        # Return squared correlation as movement preservation
        return correlation ** 2 if not np.isnan(correlation) else 0.0