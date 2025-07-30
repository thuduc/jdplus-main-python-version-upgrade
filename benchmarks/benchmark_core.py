"""Core performance benchmarks."""

import time
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
import pandas as pd
from dataclasses import dataclass

from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.toolkit.math import FastMatrix, Polynomial
from jdemetra_py.toolkit.arima import ArimaModel, ArimaEstimator, ArimaOrder


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    
    name: str
    time_seconds: float
    memory_mb: float
    iterations: int
    ops_per_second: float
    
    def __str__(self):
        return (f"{self.name}: {self.time_seconds:.3f}s "
                f"({self.ops_per_second:.0f} ops/s)")


class BenchmarkRunner:
    """Runner for performance benchmarks."""
    
    def __init__(self, warmup_iterations: int = 10, 
                 benchmark_iterations: int = 100):
        """Initialize benchmark runner.
        
        Args:
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
    
    def run(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Run a benchmark.
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark result
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # Benchmark
        start_time = time.perf_counter()
        
        for _ in range(self.benchmark_iterations):
            func(*args, **kwargs)
        
        end_time = time.perf_counter()
        
        # Calculate results
        total_time = end_time - start_time
        time_per_op = total_time / self.benchmark_iterations
        ops_per_second = self.benchmark_iterations / total_time
        
        return BenchmarkResult(
            name=func.__name__,
            time_seconds=total_time,
            memory_mb=0,  # TODO: Add memory tracking
            iterations=self.benchmark_iterations,
            ops_per_second=ops_per_second
        )


# Time series benchmarks

def benchmark_ts_creation(n: int = 10000) -> BenchmarkResult:
    """Benchmark time series creation."""
    runner = BenchmarkRunner()
    
    start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
    values = np.random.randn(n)
    
    def create_ts():
        return TsData.of(start, values)
    
    return runner.run(create_ts)


def benchmark_ts_operations(n: int = 10000) -> Dict[str, BenchmarkResult]:
    """Benchmark time series operations."""
    runner = BenchmarkRunner()
    results = {}
    
    # Create test series
    start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
    ts = TsData.of(start, np.random.randn(n))
    
    # Lead/lag
    results['lag'] = runner.run(lambda: ts.lag(1))
    results['lead'] = runner.run(lambda: ts.lead(1))
    
    # Window operations
    results['drop'] = runner.run(lambda: ts.drop(10, 10))
    results['extend'] = runner.run(lambda: ts.extend(10, 10))
    
    # Function application
    results['log'] = runner.run(lambda: ts.fn(np.log))
    results['diff'] = runner.run(lambda: ts.delta(1))
    
    # Statistics
    results['average'] = runner.run(lambda: ts.average())
    results['sum'] = runner.run(lambda: ts.sum())
    
    return results


# Math benchmarks

def benchmark_matrix_operations(size: int = 100) -> Dict[str, BenchmarkResult]:
    """Benchmark matrix operations."""
    runner = BenchmarkRunner(warmup_iterations=5, benchmark_iterations=50)
    results = {}
    
    # Create test matrices
    data1 = np.random.randn(size, size)
    data2 = np.random.randn(size, size)
    
    m1 = FastMatrix(data1)
    m2 = FastMatrix(data2)
    
    # Operations
    results['multiply'] = runner.run(lambda: m1.multiply(m2))
    results['transpose'] = runner.run(lambda: m1.transpose())
    results['inverse'] = runner.run(lambda: m1.inverse())
    results['solve'] = runner.run(lambda: m1.solve(data2[:, 0]))
    
    # Decompositions
    results['lu'] = runner.run(lambda: m1.lu_decomposition())
    results['qr'] = runner.run(lambda: m1.qr_decomposition())
    results['svd'] = runner.run(lambda: m1.svd())
    
    return results


def benchmark_polynomial_operations() -> Dict[str, BenchmarkResult]:
    """Benchmark polynomial operations."""
    runner = BenchmarkRunner()
    results = {}
    
    # Create test polynomials
    coeffs1 = np.random.randn(10)
    coeffs2 = np.random.randn(8)
    
    p1 = Polynomial(coeffs1)
    p2 = Polynomial(coeffs2)
    
    # Operations
    results['multiply'] = runner.run(lambda: p1.multiply(p2))
    results['divide'] = runner.run(lambda: p1.divide(p2))
    results['evaluate'] = runner.run(lambda: p1.evaluate(0.5))
    results['roots'] = runner.run(lambda: p1.roots())
    
    return results


# ARIMA benchmarks

def benchmark_arima_estimation(n: int = 500) -> Dict[str, BenchmarkResult]:
    """Benchmark ARIMA estimation."""
    runner = BenchmarkRunner(warmup_iterations=2, benchmark_iterations=10)
    results = {}
    
    # Generate test data
    start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
    
    # AR(1) process
    ar_data = np.zeros(n)
    ar_data[0] = np.random.randn()
    for i in range(1, n):
        ar_data[i] = 0.7 * ar_data[i-1] + np.random.randn()
    
    ts = TsData.of(start, ar_data)
    
    # Estimation
    estimator = ArimaEstimator()
    
    # Fixed order
    order = ArimaOrder(1, 0, 0)
    results['ar1_estimation'] = runner.run(
        lambda: estimator.estimate(ts, order)
    )
    
    # ARMA(1,1)
    order = ArimaOrder(1, 0, 1)
    results['arma11_estimation'] = runner.run(
        lambda: estimator.estimate(ts, order)
    )
    
    # Auto selection (slower)
    runner_auto = BenchmarkRunner(warmup_iterations=1, benchmark_iterations=5)
    results['auto_arima'] = runner_auto.run(
        lambda: estimator.estimate_auto(ts, max_p=3, max_q=3)
    )
    
    return results


def benchmark_arima_forecast(n_obs: int = 200, 
                            n_ahead: int = 24) -> BenchmarkResult:
    """Benchmark ARIMA forecasting."""
    runner = BenchmarkRunner()
    
    # Create ARIMA model
    order = ArimaOrder(1, 1, 1)
    model = ArimaModel(order, ar_params=[0.7], ma_params=[0.3])
    
    # Create test data
    start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
    data = np.cumsum(np.random.randn(n_obs))
    ts = TsData.of(start, data)
    
    # Benchmark forecasting
    from jdemetra_py.toolkit.arima import ArimaForecaster
    forecaster = ArimaForecaster(model)
    
    return runner.run(lambda: forecaster.forecast(ts.values, n_ahead))


# Comparison with other libraries

def benchmark_comparison() -> pd.DataFrame:
    """Compare performance with other libraries."""
    results = []
    
    # Test data
    n = 1000
    data = np.random.randn(n)
    
    # JDemetra+ Python
    runner = BenchmarkRunner()
    
    start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
    
    def jdemetra_ops():
        ts = TsData.of(start, data)
        _ = ts.lag(1)
        _ = ts.fn(np.log)
        _ = ts.average()
    
    jd_result = runner.run(jdemetra_ops)
    results.append({
        'Library': 'JDemetra+ Python',
        'Operation': 'Basic TS ops',
        'Time (ms)': jd_result.time_seconds * 1000 / jd_result.iterations,
        'Ops/sec': jd_result.ops_per_second
    })
    
    # Pandas
    dates = pd.date_range('2020-01-01', periods=n, freq='MS')
    
    def pandas_ops():
        s = pd.Series(data, index=dates)
        _ = s.shift(1)
        _ = np.log(s)
        _ = s.mean()
    
    pd_result = runner.run(pandas_ops)
    results.append({
        'Library': 'Pandas',
        'Operation': 'Basic TS ops',
        'Time (ms)': pd_result.time_seconds * 1000 / pd_result.iterations,
        'Ops/sec': pd_result.ops_per_second
    })
    
    return pd.DataFrame(results)


# Main benchmark suite

def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks."""
    print("Running JDemetra+ Python Benchmarks...")
    print("=" * 50)
    
    results = {}
    
    # Time series benchmarks
    print("\nTime Series Operations:")
    print("-" * 30)
    
    ts_create = benchmark_ts_creation()
    print(f"  Creation: {ts_create}")
    results['ts_creation'] = ts_create
    
    ts_ops = benchmark_ts_operations(n=5000)
    for op, result in ts_ops.items():
        print(f"  {op}: {result}")
    results['ts_operations'] = ts_ops
    
    # Math benchmarks
    print("\nMath Operations:")
    print("-" * 30)
    
    matrix_ops = benchmark_matrix_operations(size=50)
    for op, result in matrix_ops.items():
        print(f"  Matrix {op}: {result}")
    results['matrix_operations'] = matrix_ops
    
    poly_ops = benchmark_polynomial_operations()
    for op, result in poly_ops.items():
        print(f"  Polynomial {op}: {result}")
    results['polynomial_operations'] = poly_ops
    
    # ARIMA benchmarks
    print("\nARIMA Operations:")
    print("-" * 30)
    
    arima_est = benchmark_arima_estimation(n=300)
    for model, result in arima_est.items():
        print(f"  {model}: {result}")
    results['arima_estimation'] = arima_est
    
    forecast = benchmark_arima_forecast()
    print(f"  Forecasting: {forecast}")
    results['arima_forecast'] = forecast
    
    # Comparison
    print("\nLibrary Comparison:")
    print("-" * 30)
    comparison = benchmark_comparison()
    print(comparison.to_string(index=False))
    results['comparison'] = comparison
    
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()