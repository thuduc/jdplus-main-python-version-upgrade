"""Cholette benchmarking method."""

import numpy as np
from typing import Optional

from ...toolkit.timeseries import TsData
from .base import BenchmarkingProcessor, BenchmarkingResults, BenchmarkingMethod


class CholetteBenchmarking(BenchmarkingProcessor):
    """Cholette benchmarking processor.
    
    The Cholette method adjusts a series to match benchmarks while
    preserving the original series' movement as much as possible.
    """
    
    def __init__(self, rho: float = 1.0):
        """Initialize Cholette benchmarking.
        
        Args:
            rho: Autoregressive parameter (0 < rho <= 1)
                 rho=1 gives the Cholette-Dagum method
        """
        if not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        
        self.rho = rho
    
    def benchmark(self, series: TsData, target: TsData) -> BenchmarkingResults:
        """Benchmark series using Cholette method.
        
        Args:
            series: High-frequency series to be benchmarked
            target: Low-frequency target constraints
            
        Returns:
            Benchmarking results
        """
        # Validate inputs
        self._validate_inputs(series, target)
        
        # Get values
        x = series.values.copy()
        y = target.values
        
        # Create aggregation matrix
        C = self._create_aggregation_matrix(series, target)
        
        # Apply Cholette method
        x_benchmarked = self._cholette_dagum(x, y, C)
        
        # Create benchmarked series
        benchmarked = TsData.of(series.start, x_benchmarked)
        
        # Compute adjustment factors
        adj_factors = np.where(x != 0, x_benchmarked / x, 1.0)
        
        # Create results
        results = BenchmarkingResults(
            original=series,
            benchmarked=benchmarked,
            target=target,
            method=BenchmarkingMethod.CHOLETTE,
            adjustment_factors=adj_factors
        )
        
        # Compute diagnostics
        results.diagnostics['movement_preservation'] = self._compute_movement_preservation(
            x, x_benchmarked
        )
        results.diagnostics['rho'] = self.rho
        
        return results
    
    def _cholette_dagum(self, x: np.ndarray, y: np.ndarray, 
                       C: np.ndarray) -> np.ndarray:
        """Apply Cholette-Dagum benchmarking.
        
        Args:
            x: High-frequency series values
            y: Low-frequency target values
            C: Aggregation matrix
            
        Returns:
            Benchmarked series
        """
        n = len(x)
        m = len(y)
        
        # Handle missing values
        x_valid = ~np.isnan(x)
        y_valid = ~np.isnan(y)
        
        if not np.any(y_valid):
            # No valid benchmarks
            return x
        
        # Compute preliminary series aggregates
        x_agg = C @ x
        
        # Compute discrepancies
        discrepancies = np.zeros(m)
        for i in range(m):
            if y_valid[i] and not np.isnan(x_agg[i]):
                discrepancies[i] = y[i] - x_agg[i]
        
        # Build adjustment series
        adjustments = np.zeros(n)
        
        # Distribute discrepancies
        if self.rho == 1.0:
            # Proportional distribution (Cholette-Dagum)
            for i in range(m):
                if discrepancies[i] != 0:
                    # Find high-frequency indices for this low-frequency period
                    ratio = n // m
                    start_idx = i * ratio
                    end_idx = min(start_idx + ratio, n)
                    
                    # Get sub-series
                    x_sub = x[start_idx:end_idx]
                    
                    # Compute proportional factors
                    x_sub_sum = np.nansum(x_sub)
                    if x_sub_sum != 0:
                        for j in range(start_idx, end_idx):
                            if x_valid[j]:
                                adjustments[j] = discrepancies[i] * x[j] / x_sub_sum
        else:
            # AR(1) distribution
            adjustments = self._ar1_distribution(x, discrepancies, C, self.rho)
        
        # Apply adjustments
        x_benchmarked = x + adjustments
        
        # Ensure non-negativity if original was non-negative
        if np.all(x[x_valid] >= 0):
            x_benchmarked = np.maximum(x_benchmarked, 0)
        
        return x_benchmarked
    
    def _ar1_distribution(self, x: np.ndarray, discrepancies: np.ndarray,
                         C: np.ndarray, rho: float) -> np.ndarray:
        """Distribute discrepancies using AR(1) model.
        
        Args:
            x: High-frequency series
            discrepancies: Low-frequency discrepancies
            C: Aggregation matrix
            rho: AR parameter
            
        Returns:
            High-frequency adjustments
        """
        n = len(x)
        m = len(discrepancies)
        ratio = n // m
        
        adjustments = np.zeros(n)
        
        # Build AR(1) covariance structure
        for i in range(m):
            if discrepancies[i] != 0:
                start_idx = i * ratio
                end_idx = min(start_idx + ratio, n)
                k = end_idx - start_idx
                
                # Build covariance matrix for this block
                V = np.zeros((k, k))
                for row in range(k):
                    for col in range(k):
                        V[row, col] = rho ** abs(row - col)
                
                # Compute weights
                ones = np.ones(k)
                V_inv = np.linalg.inv(V)
                weights = V_inv @ ones / (ones.T @ V_inv @ ones)
                
                # Distribute discrepancy
                adjustments[start_idx:end_idx] = discrepancies[i] * weights
        
        return adjustments
    
    def _compute_movement_preservation(self, original: np.ndarray,
                                     benchmarked: np.ndarray) -> float:
        """Compute movement preservation statistic.
        
        Args:
            original: Original series
            benchmarked: Benchmarked series
            
        Returns:
            Movement preservation measure
        """
        # Compute growth rates
        orig_growth = np.diff(original) / original[:-1]
        bench_growth = np.diff(benchmarked) / benchmarked[:-1]
        
        # Remove invalid values
        valid = ~(np.isnan(orig_growth) | np.isnan(bench_growth) | 
                 np.isinf(orig_growth) | np.isinf(bench_growth))
        
        if np.sum(valid) == 0:
            return np.nan
        
        # Compute correlation of growth rates
        correlation = np.corrcoef(orig_growth[valid], bench_growth[valid])[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0