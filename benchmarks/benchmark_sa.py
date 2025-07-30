"""Seasonal adjustment performance benchmarks."""

import time
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass

from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.sa import (
    TramoSeatsSpecification, TramoSeatsProcessor,
    X13Specification
)
from jdemetra_py.sa.diagnostics import SeasonalityTests, compute_comprehensive_quality
from .benchmark_core import BenchmarkRunner, BenchmarkResult


def generate_seasonal_data(n: int = 144, 
                          seasonal_amplitude: float = 10.0,
                          trend_slope: float = 0.5) -> TsData:
    """Generate seasonal time series for benchmarking.
    
    Args:
        n: Number of observations
        seasonal_amplitude: Amplitude of seasonal component
        trend_slope: Slope of trend component
        
    Returns:
        Time series with trend, seasonal, and irregular components
    """
    t = np.arange(n)
    
    # Components
    trend = 100 + trend_slope * t
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 12)
    irregular = np.random.normal(0, 2, n)
    
    # Multiplicative series
    series = trend * (1 + seasonal/100) * (1 + irregular/100)
    
    start = TsPeriod.of(TsFrequency.MONTHLY, 2015, 0)
    return TsData.of(start, series)


def benchmark_tramoseats_processing(series_lengths: List[int] = [60, 120, 240]) -> Dict[int, BenchmarkResult]:
    """Benchmark TRAMO-SEATS processing for different series lengths.
    
    Args:
        series_lengths: List of series lengths to test
        
    Returns:
        Dictionary of results by series length
    """
    results = {}
    
    for n in series_lengths:
        # Generate test data
        series = generate_seasonal_data(n)
        
        # Create processor
        spec = TramoSeatsSpecification.rsa1()
        processor = TramoSeatsProcessor(spec)
        
        # Benchmark
        runner = BenchmarkRunner(warmup_iterations=2, benchmark_iterations=10)
        
        def process():
            return processor.process(series)
        
        result = runner.run(process)
        result.name = f"TRAMO-SEATS (n={n})"
        results[n] = result
        
        print(f"TRAMO-SEATS (n={n}): {result}")
    
    return results


def benchmark_specifications() -> Dict[str, BenchmarkResult]:
    """Benchmark different TRAMO-SEATS specifications.
    
    Returns:
        Dictionary of results by specification
    """
    results = {}
    runner = BenchmarkRunner(warmup_iterations=2, benchmark_iterations=10)
    
    # Generate test data
    series = generate_seasonal_data(120)
    
    # Test different specifications
    specs = {
        'RSA1': TramoSeatsSpecification.rsa1(),  # Automatic
        'RSA2': TramoSeatsSpecification.rsa2(),  # With calendar
        'RSA3': TramoSeatsSpecification.rsa3(),  # With outliers
        'RSA4': TramoSeatsSpecification.rsa4(),  # Calendar + outliers
        'RSA5': TramoSeatsSpecification.rsa5(),  # Full automatic
    }
    
    for name, spec in specs.items():
        processor = TramoSeatsProcessor(spec)
        
        def process():
            return processor.process(series)
        
        result = runner.run(process)
        result.name = name
        results[name] = result
        
        print(f"{name}: {result}")
    
    return results


def benchmark_diagnostics(n_series: int = 100) -> Dict[str, BenchmarkResult]:
    """Benchmark diagnostic computations.
    
    Args:
        n_series: Number of series to process
        
    Returns:
        Dictionary of diagnostic benchmark results
    """
    results = {}
    runner = BenchmarkRunner(warmup_iterations=5, benchmark_iterations=20)
    
    # Generate test data and decomposition
    from jdemetra_py.sa.base import SeriesDecomposition, DecompositionMode
    
    series = generate_seasonal_data(120)
    
    # Create mock decomposition
    decomp = SeriesDecomposition(mode=DecompositionMode.MULTIPLICATIVE)
    decomp.series = series
    
    # Simple decomposition (for benchmarking)
    from scipy.ndimage import uniform_filter1d
    trend = uniform_filter1d(series.values, 13, mode='nearest')
    detrended = series.values / trend
    
    # Extract seasonal
    seasonal = np.zeros_like(series.values)
    for month in range(12):
        month_values = detrended[month::12]
        seasonal[month::12] = np.mean(month_values)
    
    irregular = detrended / seasonal
    
    decomp.trend = TsData.of(series.start, trend)
    decomp.seasonal = TsData.of(series.start, seasonal)
    decomp.irregular = TsData.of(series.start, irregular)
    decomp.seasonally_adjusted = TsData.of(series.start, series.values / seasonal)
    
    # Benchmark different diagnostics
    
    # Seasonality tests
    def seasonality_tests():
        return SeasonalityTests.test_all(series)
    
    results['seasonality_tests'] = runner.run(seasonality_tests)
    
    # Quality measures
    def quality_measures():
        return compute_comprehensive_quality(decomp)
    
    results['quality_measures'] = runner.run(quality_measures)
    
    # M-statistics
    from jdemetra_py.sa.diagnostics.quality import MStatistics
    
    def m_statistics():
        return MStatistics.compute(decomp)
    
    results['m_statistics'] = runner.run(m_statistics)
    
    # Residual diagnostics
    from jdemetra_py.sa.diagnostics.residuals import compute_residuals_diagnostics
    
    irregular_ts = decomp.irregular
    
    def residual_diag():
        return compute_residuals_diagnostics(irregular_ts)
    
    results['residual_diagnostics'] = runner.run(residual_diag)
    
    return results


def benchmark_batch_processing(n_series: int = 50, 
                             series_length: int = 120) -> BenchmarkResult:
    """Benchmark batch processing of multiple series.
    
    Args:
        n_series: Number of series to process
        series_length: Length of each series
        
    Returns:
        Benchmark result for batch processing
    """
    # Generate multiple series
    series_list = []
    for i in range(n_series):
        series = generate_seasonal_data(
            series_length, 
            seasonal_amplitude=np.random.uniform(5, 15),
            trend_slope=np.random.uniform(0.1, 1.0)
        )
        series_list.append(series)
    
    # Create processor
    spec = TramoSeatsSpecification.rsa1()
    processor = TramoSeatsProcessor(spec)
    
    # Benchmark batch processing
    runner = BenchmarkRunner(warmup_iterations=1, benchmark_iterations=5)
    
    def batch_process():
        results = []
        for series in series_list:
            result = processor.process(series)
            results.append(result)
        return results
    
    result = runner.run(batch_process)
    result.name = f"Batch processing ({n_series} series)"
    
    # Calculate per-series time
    time_per_series = result.time_seconds / (result.iterations * n_series)
    print(f"Batch processing: {time_per_series:.3f}s per series")
    
    return result


def benchmark_memory_usage() -> pd.DataFrame:
    """Benchmark memory usage for different operations.
    
    Returns:
        DataFrame with memory usage statistics
    """
    import tracemalloc
    
    results = []
    
    # Test different series lengths
    for n in [60, 120, 240, 480]:
        # Generate data
        series = generate_seasonal_data(n)
        
        # Measure TRAMO-SEATS memory
        tracemalloc.start()
        
        spec = TramoSeatsSpecification.rsa1()
        processor = TramoSeatsProcessor(spec)
        result = processor.process(series)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results.append({
            'Operation': 'TRAMO-SEATS',
            'Series Length': n,
            'Current Memory (MB)': current / 1024 / 1024,
            'Peak Memory (MB)': peak / 1024 / 1024
        })
    
    return pd.DataFrame(results)


def compare_methods() -> pd.DataFrame:
    """Compare performance of different SA methods.
    
    Returns:
        DataFrame with method comparison
    """
    results = []
    runner = BenchmarkRunner(warmup_iterations=2, benchmark_iterations=10)
    
    # Generate test data
    series = generate_seasonal_data(120)
    
    # TRAMO-SEATS
    ts_spec = TramoSeatsSpecification.rsa1()
    ts_processor = TramoSeatsProcessor(ts_spec)
    
    def tramoseats():
        return ts_processor.process(series)
    
    ts_result = runner.run(tramoseats)
    
    results.append({
        'Method': 'TRAMO-SEATS',
        'Time (s)': ts_result.time_seconds / ts_result.iterations,
        'Ops/sec': ts_result.ops_per_second
    })
    
    # Add more methods as they become available
    
    return pd.DataFrame(results)


# Performance optimization analysis

def analyze_bottlenecks():
    """Analyze performance bottlenecks in SA processing."""
    import cProfile
    import pstats
    from io import StringIO
    
    # Generate test data
    series = generate_seasonal_data(120)
    
    # Profile TRAMO-SEATS
    spec = TramoSeatsSpecification.rsa1()
    processor = TramoSeatsProcessor(spec)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run processing multiple times
    for _ in range(10):
        processor.process(series)
    
    profiler.disable()
    
    # Analyze results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("\nPerformance Profile:")
    print("=" * 50)
    print(s.getvalue())


# Main benchmark suite for SA

def run_sa_benchmarks():
    """Run all seasonal adjustment benchmarks."""
    print("Running Seasonal Adjustment Benchmarks...")
    print("=" * 50)
    
    # Processing benchmarks
    print("\n1. Processing Speed by Series Length:")
    print("-" * 40)
    length_results = benchmark_tramoseats_processing([60, 120, 240])
    
    # Specification benchmarks
    print("\n2. Different Specifications:")
    print("-" * 40)
    spec_results = benchmark_specifications()
    
    # Diagnostic benchmarks
    print("\n3. Diagnostic Computations:")
    print("-" * 40)
    diag_results = benchmark_diagnostics()
    for name, result in diag_results.items():
        print(f"  {name}: {result}")
    
    # Batch processing
    print("\n4. Batch Processing:")
    print("-" * 40)
    batch_result = benchmark_batch_processing(n_series=20)
    print(f"  {batch_result}")
    
    # Memory usage
    print("\n5. Memory Usage:")
    print("-" * 40)
    memory_df = benchmark_memory_usage()
    print(memory_df.to_string(index=False))
    
    # Method comparison
    print("\n6. Method Comparison:")
    print("-" * 40)
    comparison_df = compare_methods()
    print(comparison_df.to_string(index=False))
    
    # Bottleneck analysis
    print("\n7. Bottleneck Analysis:")
    print("-" * 40)
    analyze_bottlenecks()


if __name__ == "__main__":
    run_sa_benchmarks()