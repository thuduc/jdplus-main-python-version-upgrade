# JDemetra+ Python API Reference

## Core Modules

### `jdemetra_py.toolkit.timeseries`

Core time series data structures and operations.

#### Classes

##### `TsFrequency`
Enumeration of supported time series frequencies.

**Values:**
- `UNDEFINED`: Undefined frequency
- `YEARLY`: Annual data (1 observation per year)
- `HALF_YEARLY`: Semi-annual data (2 observations per year)
- `QUARTERLY`: Quarterly data (4 observations per year)
- `MONTHLY`: Monthly data (12 observations per year)
- `WEEKLY`: Weekly data (52 observations per year)
- `DAILY`: Daily data (365/366 observations per year)

##### `TsPeriod`
Represents a specific time period.

**Methods:**
- `of(frequency: TsFrequency, year: int, position: int) -> TsPeriod`: Create a period
- `plus(n: int) -> TsPeriod`: Add n periods
- `minus(n: int) -> TsPeriod`: Subtract n periods
- `until(end: TsPeriod) -> int`: Count periods until end
- `year() -> int`: Get year
- `position() -> int`: Get position within year
- `frequency() -> TsFrequency`: Get frequency

**Example:**
```python
# January 2020
period = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
# February 2020
next_period = period.plus(1)
```

##### `TsDomain`
Represents a time domain (range of periods).

**Methods:**
- `of(start: TsPeriod, length: int) -> TsDomain`: Create domain
- `range(start: TsPeriod, end: TsPeriod) -> TsDomain`: Create from range
- `get(index: int) -> TsPeriod`: Get period at index
- `contains(period: TsPeriod) -> bool`: Check if period is in domain
- `intersection(other: TsDomain) -> Optional[TsDomain]`: Find intersection
- `length() -> int`: Get number of periods

##### `TsData`
Time series data container.

**Methods:**
- `of(start: TsPeriod, values: np.ndarray) -> TsData`: Create from start and values
- `empty(length: int) -> TsData`: Create empty series
- `get(index: int) -> float`: Get value at index
- `set(index: int, value: float)`: Set value at index
- `values() -> np.ndarray`: Get all values
- `start() -> TsPeriod`: Get start period
- `end() -> TsPeriod`: Get end period
- `length() -> int`: Get length
- `frequency() -> TsFrequency`: Get frequency
- `domain() -> TsDomain`: Get time domain
- `lag(k: int) -> TsData`: Lag by k periods
- `lead(k: int) -> TsData`: Lead by k periods
- `delta(lag: int) -> TsData`: Difference operation
- `log() -> TsData`: Natural logarithm
- `exp() -> TsData`: Exponential
- `fn(func: Callable) -> TsData`: Apply function
- `window(start: TsPeriod, end: TsPeriod) -> TsData`: Extract window
- `extend(n_before: int, n_after: int) -> TsData`: Extend with NaN
- `clean_extremities() -> TsData`: Remove leading/trailing NaN
- `average() -> float`: Compute average
- `isAllFinite() -> bool`: Check if all values are finite

**Example:**
```python
# Create monthly series starting Jan 2020
start = TsPeriod.of(TsFrequency.MONTHLY, 2020, 0)
values = np.random.randn(24)
ts = TsData.of(start, values)

# Operations
ts_lag = ts.lag(1)
ts_diff = ts.delta(1)
ts_log = ts.log()
```

### `jdemetra_py.toolkit.arima`

ARIMA modeling and forecasting.

#### Classes

##### `SarimaOrder`
SARIMA model order specification.

**Attributes:**
- `p`: AR order
- `d`: Differencing order
- `q`: MA order
- `P`: Seasonal AR order
- `D`: Seasonal differencing order
- `Q`: Seasonal MA order
- `s`: Seasonal period

**Methods:**
- `__init__(p, d, q, P=0, D=0, Q=0, s=1)`: Initialize order
- `spec_str() -> str`: Get specification string

##### `ArimaModel`
ARIMA model representation.

**Attributes:**
- `order`: Model order
- `ar`: AR coefficients
- `ma`: MA coefficients
- `constant`: Constant term
- `innovation_variance`: Innovation variance

**Methods:**
- `__init__(order, ar, ma, constant=0, innovation_variance=1)`: Initialize
- `is_stationary() -> bool`: Check stationarity
- `is_invertible() -> bool`: Check invertibility
- `forecast(data: np.ndarray, n_ahead: int) -> np.ndarray`: Generate forecasts

##### `ArimaEstimator`
ARIMA parameter estimation.

**Methods:**
- `__init__(method='css-ml')`: Initialize with estimation method
- `estimate(data: TsData, order: SarimaOrder) -> ArimaModel`: Estimate model

**Example:**
```python
# Specify SARIMA(1,1,1)(1,1,1)12
order = SarimaOrder(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)

# Estimate model
estimator = ArimaEstimator()
model = estimator.estimate(ts, order)

# Forecast
forecasts = model.forecast(ts.values, n_ahead=12)
```

### `jdemetra_py.toolkit.ssf`

State space framework.

#### Classes

##### `StateSpaceModel`
General state space model representation.

**Attributes:**
- `transition`: Transition matrix T
- `measurement`: Measurement matrix Z
- `state_cov`: State covariance Q
- `measurement_cov`: Measurement covariance H

**Methods:**
- `__init__(transition, measurement, state_cov, measurement_cov)`: Initialize
- `dimension() -> int`: State dimension
- `is_time_invariant() -> bool`: Check time invariance

##### `KalmanFilter`
Kalman filtering algorithm.

**Methods:**
- `__init__(tol=1e-7)`: Initialize with tolerance
- `filter(y: np.ndarray, model: StateSpaceModel) -> FilterResults`: Run filter

##### `FilterResults`
Kalman filter output.

**Attributes:**
- `filtered_states`: Filtered state estimates
- `filtered_covariances`: Filtered state covariances
- `predictions`: One-step predictions
- `innovations`: Prediction errors
- `log_likelihood`: Log-likelihood value

### `jdemetra_py.sa`

Seasonal adjustment framework.

#### Base Classes

##### `SaSpecification`
Base class for SA method specifications.

**Methods:**
- `set_span(start: TsPeriod, end: TsPeriod)`: Set processing span
- `set_transform(transform: str)`: Set transformation ('log', 'none', 'auto')
- `set_outliers(outliers: List[str])`: Set outlier types to detect
- `is_valid() -> bool`: Validate specification

##### `SaProcessor`
Base class for SA processors.

**Methods:**
- `can_process(spec: SaSpecification) -> bool`: Check compatibility
- `process(series: TsData, spec: SaSpecification) -> SaResults`: Process series

##### `SaResults`
Container for SA results.

**Attributes:**
- `specification`: Used specification
- `preprocessing`: Pre-adjustment results
- `decomposition`: Series decomposition
- `diagnostics`: Quality diagnostics
- `processing_log`: Processing messages

**Methods:**
- `summary() -> str`: Get text summary
- `get_series(component: str) -> Optional[TsData]`: Get component series

##### `SeriesDecomposition`
Decomposition components.

**Attributes:**
- `original`: Original series
- `seasonally_adjusted`: SA series
- `trend`: Trend-cycle component
- `seasonal`: Seasonal component
- `irregular`: Irregular component
- `calendar`: Calendar effects (optional)
- `outliers`: Outlier effects (optional)

### `jdemetra_py.sa.tramoseats`

TRAMO/SEATS implementation.

#### Classes

##### `TramoSeatsSpecification`
TRAMO/SEATS specification.

**Static Methods:**
- `rsa0() -> TramoSeatsSpecification`: RSA0 - No log, no outliers
- `rsa1() -> TramoSeatsSpecification`: RSA1 - Log, outliers
- `rsa2() -> TramoSeatsSpecification`: RSA2 - Working days
- `rsa3() -> TramoSeatsSpecification`: RSA3 - Log, working days, outliers
- `rsa4() -> TramoSeatsSpecification`: RSA4 - Easter, working days
- `rsa5() -> TramoSeatsSpecification`: RSA5 - Log, Easter, working days, outliers

**Methods:**
- `set_seats_parameters(xl=None, epsphi=None, rmod=None)`: SEATS parameters
- `set_tramo_parameters(...)`: TRAMO parameters

##### `TramoSeatsProcessor`
TRAMO/SEATS processing engine.

**Methods:**
- `__init__(spec: TramoSeatsSpecification)`: Initialize with spec
- `process(series: TsData) -> SaResults`: Process series

**Example:**
```python
# Use RSA5 specification
spec = TramoSeatsSpecification.rsa5()

# Process series
processor = TramoSeatsProcessor(spec)
results = processor.process(ts)

# Access components
sa = results.decomposition.seasonally_adjusted
trend = results.decomposition.trend
seasonal = results.decomposition.seasonal
```

### `jdemetra_py.sa.x13`

X-13ARIMA-SEATS wrapper.

#### Classes

##### `X13Specification`
X-13ARIMA-SEATS specification.

**Static Methods:**
- `rsa0() -> X13Specification`: RSA0 equivalent
- `rsa1() -> X13Specification`: RSA1 equivalent
- `rsa2c() -> X13Specification`: RSA2c - Log, working days
- `rsa3() -> X13Specification`: RSA3 equivalent
- `rsa4c() -> X13Specification`: RSA4c - Easter, working days
- `rsa5c() -> X13Specification`: RSA5c - Full specification

##### `X13ArimaSeatsProcessor`
X-13 processing wrapper.

**Methods:**
- `__init__(x13_path: Optional[str] = None)`: Initialize with X-13 path
- `process(series: TsData, spec: X13Specification) -> SaResults`: Process

### `jdemetra_py.sa.diagnostics`

Quality diagnostics.

#### Classes

##### `SeasonalityTests`
Tests for presence of seasonality.

**Static Methods:**
- `qs_test(series: TsData, frequency: int) -> TestResult`: QS test
- `friedman_test(series: TsData, frequency: int) -> TestResult`: Friedman test
- `kruskal_wallis_test(series: TsData, frequency: int) -> TestResult`: KW test
- `combined_test(series: TsData, frequency: int) -> TestResult`: Combined test

##### `QualityMeasures`
M-statistics and quality measures.

**Methods:**
- `__init__(decomposition: SeriesDecomposition)`: Initialize
- `m1() -> float`: M1 - Relative contribution of irregular
- `m2() -> float`: M2 - Relative contribution of irregular (changes)
- `m3() -> float`: M3 - Irregular/Seasonal ratio
- `m4() -> float`: M4 - Autocorrelation of irregular
- `m5() -> float`: M5 - Number of periods for change
- `m6() -> float`: M6 - Year-on-year changes
- `m7() -> float`: M7 - Moving seasonality
- `q() -> float`: Overall Q statistic
- `q_m2() -> float`: Q-M2 statistic

##### `ResidualsDiagnostics`
Residual analysis tools.

**Methods:**
- `__init__(residuals: np.ndarray)`: Initialize
- `ljung_box_test(lags: int) -> TestResult`: Ljung-Box test
- `normality_test() -> TestResult`: Normality test
- `heteroscedasticity_test() -> TestResult`: Heteroscedasticity test
- `runs_test() -> TestResult`: Runs test

### `jdemetra_py.io`

Data input/output functionality.

#### Classes

##### `DataProvider`
Abstract base for data providers.

**Methods:**
- `can_read(source) -> bool`: Check if can read source
- `can_write(source) -> bool`: Check if can write to source
- `read(source, **kwargs) -> Union[TsData, TsCollection]`: Read data
- `write(data, target, **kwargs)`: Write data

##### `CsvDataProvider`
CSV file I/O provider.

**Methods:**
- `__init__(delimiter=',', encoding='utf-8')`: Initialize
- `read(source, date_column=0, value_columns=None, **kwargs)`: Read CSV
- `write(data, target, **kwargs)`: Write CSV

##### `ExcelDataProvider`
Excel file I/O provider.

**Methods:**
- `__init__(engine='openpyxl')`: Initialize
- `read(source, sheet_name=0, **kwargs)`: Read Excel
- `write(data, target, sheet_name='Data', **kwargs)`: Write Excel

##### `TsCollection`
Collection of time series.

**Methods:**
- `__init__(name='')`: Initialize collection
- `add(name: str, series: TsData)`: Add series
- `get(name: str) -> Optional[TsData]`: Get series
- `items() -> List[Tuple[str, TsData]]`: Get all items
- `names() -> List[str]`: Get series names
- `search(pattern: str) -> List[str]`: Search names

**Example:**
```python
# Read CSV
csv = CsvDataProvider()
collection = csv.read('data.csv', date_column='Date', 
                     value_columns=['Series1', 'Series2'])

# Access series
series1 = collection.get('Series1')

# Write to Excel
excel = ExcelDataProvider()
excel.write(collection, 'output.xlsx')
```

### `jdemetra_py.workspace`

Project management.

#### Classes

##### `SAItem`
Seasonal adjustment item.

**Attributes:**
- `name`: Item name
- `series`: Time series data
- `specification`: SA specification
- `results`: Processing results (optional)
- `enabled`: Processing enabled flag
- `metadata`: Additional metadata dict

##### `Workspace`
Container for SA projects.

**Methods:**
- `__init__(name='')`: Initialize workspace
- `add_sa_item(item: SAItem) -> str`: Add SA item
- `get_sa_item(item_id: str) -> Optional[SAItem]`: Get item
- `remove_sa_item(item_id: str) -> bool`: Remove item
- `list_sa_items() -> List[str]`: List item IDs
- `process_sa_item(item_id: str, processor: SaProcessor) -> bool`: Process item
- `process_all(processor: SaProcessor) -> Dict[str, bool]`: Process all items
- `save(path: str, format: str = 'json')`: Save workspace
- `load(path: str) -> 'Workspace'`: Load workspace

**Example:**
```python
# Create workspace
ws = Workspace('MyProject')

# Add items
item1 = SAItem('Sales', sales_ts, TramoSeatsSpecification.rsa5())
ws.add_sa_item(item1)

# Process all
processor = TramoSeatsProcessor()
results = ws.process_all(processor)

# Save
ws.save('project.json')
```

### `jdemetra_py.visualization`

Plotting utilities.

#### Functions

##### `plot_series`
Plot time series.

**Parameters:**
- `series: Union[TsData, List[TsData]]`: Series to plot
- `title: str = ''`: Plot title
- `figsize: Tuple[float, float] = (12, 6)`: Figure size
- `labels: Optional[List[str]] = None`: Series labels

**Returns:** `matplotlib.figure.Figure`

##### `plot_decomposition`
Plot seasonal adjustment decomposition.

**Parameters:**
- `decomposition: SeriesDecomposition`: Decomposition to plot
- `title: str = ''`: Plot title
- `figsize: Tuple[float, float] = (12, 10)`: Figure size

**Returns:** `matplotlib.figure.Figure`

##### `plot_diagnostics`
Plot SA diagnostics.

**Parameters:**
- `results: SaResults`: SA results
- `figsize: Tuple[float, float] = (15, 10)`: Figure size

**Returns:** `matplotlib.figure.Figure`

**Example:**
```python
# Plot original and SA series
fig = plot_series([ts, results.decomposition.seasonally_adjusted],
                  labels=['Original', 'SA'])

# Plot full decomposition
fig = plot_decomposition(results.decomposition)

# Plot diagnostics
fig = plot_diagnostics(results)
```

### `jdemetra_py.optimization`

Performance optimization utilities.

#### Decorators

##### `@memoize`
Memoization decorator.

**Parameters:**
- `maxsize: int = 128`: Maximum cache size

**Example:**
```python
@memoize(maxsize=256)
def expensive_computation(x, y):
    # Complex calculation
    return result
```

##### `@jit_compile`
Numba JIT compilation decorator.

**Parameters:**
- `nopython: bool = True`: Disable Python fallback
- `cache: bool = True`: Cache compiled functions
- `parallel: bool = False`: Enable parallel execution

**Example:**
```python
@jit_compile(parallel=True)
def fast_loop(data):
    result = np.zeros_like(data)
    for i in prange(len(data)):
        result[i] = data[i] ** 2
    return result
```

#### Functions

##### `enable_numba() / disable_numba()`
Control Numba JIT compilation globally.

##### `parallel_map`
Parallel processing utility.

**Parameters:**
- `func: Callable`: Function to apply
- `items: List`: Items to process
- `n_workers: Optional[int] = None`: Number of workers
- `use_threads: bool = False`: Use threads vs processes

**Returns:** List of results

## Data Formats

### Time Series Data Format

The package expects time series data in specific formats:

#### CSV Format
```csv
Date,Value
2020-01-01,100.5
2020-02-01,102.3
2020-03-01,98.7
```

#### DataFrame Format
```python
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=24, freq='M'),
    'value': np.random.randn(24) * 10 + 100
})
```

### Workspace Format

#### JSON Format
```json
{
  "name": "MyProject",
  "created": "2024-01-01T00:00:00",
  "items": [
    {
      "id": "series1",
      "name": "Sales Data",
      "series": {
        "start": {"year": 2020, "period": 0, "frequency": "MONTHLY"},
        "values": [100.5, 102.3, ...]
      },
      "specification": {
        "type": "tramoseats",
        "transform": "log",
        "outliers": ["AO", "LS"]
      }
    }
  ]
}
```

## Error Handling

The package uses custom exceptions for error handling:

- `JDemetraError`: Base exception
- `SpecificationError`: Invalid specification
- `ProcessingError`: Processing failure
- `DataError`: Data format/validation error

**Example:**
```python
from jdemetra_py.errors import ProcessingError

try:
    results = processor.process(ts)
except ProcessingError as e:
    print(f"Processing failed: {e}")
```

## Performance Tips

1. **Use Numba for intensive computations:**
   ```python
   from jdemetra_py.optimization import enable_numba
   enable_numba()
   ```

2. **Batch process multiple series:**
   ```python
   results = parallel_map(processor.process, series_list, n_workers=4)
   ```

3. **Cache expensive operations:**
   ```python
   @memoize(maxsize=100)
   def compute_seasonal_pattern(series):
       # Expensive computation
       return pattern
   ```

4. **Use vectorized operations:**
   ```python
   from jdemetra_py.optimization import vectorized_ma
   smoothed = vectorized_ma(data, window=12)
   ```

## Version History

- **0.1.0** (2024-01): Initial release
  - Core time series functionality
  - TRAMO/SEATS implementation
  - X-13ARIMA-SEATS wrapper
  - Basic visualization tools