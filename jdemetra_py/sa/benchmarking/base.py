"""Base classes for benchmarking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np

from ...toolkit.timeseries import TsData


class BenchmarkingMethod(Enum):
    """Benchmarking method enumeration."""
    
    DENTON = "denton"
    CHOLETTE = "cholette"
    PROPORTIONAL = "proportional"
    DIFFERENCE = "difference"


@dataclass
class BenchmarkingResults:
    """Results from benchmarking operation."""
    
    # Original and benchmarked series
    original: TsData
    benchmarked: TsData
    target: TsData
    
    # Method used
    method: BenchmarkingMethod
    
    # Adjustment factors
    adjustment_factors: Optional[np.ndarray] = None
    
    # Diagnostics
    diagnostics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.diagnostics is None:
            self.diagnostics = {}
    
    def summary(self) -> str:
        """Get summary of benchmarking results."""
        lines = ["Benchmarking Results"]
        lines.append("=" * 30)
        
        lines.append(f"\nMethod: {self.method.value}")
        lines.append(f"Original series length: {self.original.length}")
        lines.append(f"Target series length: {self.target.length}")
        
        # Compute discrepancy measures
        original_sum = np.nansum(self.original.values)
        target_sum = np.nansum(self.target.values)
        benchmarked_sum = np.nansum(self.benchmarked.values)
        
        lines.append(f"\nOriginal sum: {original_sum:.2f}")
        lines.append(f"Target sum: {target_sum:.2f}")
        lines.append(f"Benchmarked sum: {benchmarked_sum:.2f}")
        
        # Relative discrepancy
        if target_sum != 0:
            rel_disc = 100 * (benchmarked_sum - target_sum) / target_sum
            lines.append(f"Relative discrepancy: {rel_disc:.4f}%")
        
        # Movement preservation
        if 'movement_preservation' in self.diagnostics:
            mp = self.diagnostics['movement_preservation']
            lines.append(f"\nMovement preservation: {mp:.4f}")
        
        return "\n".join(lines)


class BenchmarkingProcessor(ABC):
    """Abstract base class for benchmarking processors."""
    
    @abstractmethod
    def benchmark(self, series: TsData, target: TsData, **kwargs) -> BenchmarkingResults:
        """Benchmark a series to match target constraints.
        
        Args:
            series: High-frequency series to be benchmarked
            target: Low-frequency target constraints
            **kwargs: Method-specific parameters
            
        Returns:
            Benchmarking results
        """
        pass
    
    def _validate_inputs(self, series: TsData, target: TsData):
        """Validate benchmarking inputs.
        
        Args:
            series: High-frequency series
            target: Low-frequency target
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check frequencies
        series_freq = series.domain.frequency.periods_per_year
        target_freq = target.domain.frequency.periods_per_year
        
        if series_freq <= target_freq:
            raise ValueError(
                f"Series frequency ({series_freq}) must be higher than "
                f"target frequency ({target_freq})"
            )
        
        # Check if frequencies are compatible
        if series_freq % target_freq != 0:
            raise ValueError(
                f"Series frequency ({series_freq}) must be a multiple of "
                f"target frequency ({target_freq})"
            )
    
    def _create_aggregation_matrix(self, series: TsData, target: TsData) -> np.ndarray:
        """Create aggregation matrix from high to low frequency.
        
        Args:
            series: High-frequency series
            target: Low-frequency target
            
        Returns:
            Aggregation matrix
        """
        n_high = series.length
        n_low = target.length
        
        # Periods per low-frequency observation
        ratio = series.domain.frequency.periods_per_year // target.domain.frequency.periods_per_year
        
        # Create aggregation matrix
        C = np.zeros((n_low, n_high))
        
        for i in range(n_low):
            start_idx = i * ratio
            end_idx = min(start_idx + ratio, n_high)
            C[i, start_idx:end_idx] = 1.0
        
        return C