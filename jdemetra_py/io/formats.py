"""Data formats and collections for time series."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator, Union
from enum import Enum
import pandas as pd

from ..toolkit.timeseries import TsData, TsPeriod, TsFrequency


class DataFormat(Enum):
    """Supported data formats."""
    
    CSV = "csv"
    EXCEL = "excel"
    XML = "xml"
    JSON = "json"
    SDMX = "sdmx"
    TSV = "tsv"
    
    @classmethod
    def from_extension(cls, filename: str) -> 'DataFormat':
        """Determine format from file extension.
        
        Args:
            filename: File name or path
            
        Returns:
            Data format
        """
        ext = filename.lower().split('.')[-1]
        
        mapping = {
            'csv': cls.CSV,
            'xlsx': cls.EXCEL,
            'xls': cls.EXCEL,
            'xml': cls.XML,
            'json': cls.JSON,
            'tsv': cls.TSV,
            'sdmx': cls.SDMX
        }
        
        return mapping.get(ext, cls.CSV)


class TsCollection:
    """Collection of time series."""
    
    def __init__(self):
        """Initialize empty collection."""
        self._series: Dict[str, TsData] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def add(self, name: str, series: TsData, metadata: Optional[Dict[str, Any]] = None):
        """Add series to collection.
        
        Args:
            name: Series name
            series: Time series data
            metadata: Optional metadata
        """
        self._series[name] = series
        if metadata:
            self._metadata[name] = metadata
    
    def get(self, name: str) -> Optional[TsData]:
        """Get series by name.
        
        Args:
            name: Series name
            
        Returns:
            Time series or None
        """
        return self._series.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove series from collection.
        
        Args:
            name: Series name
            
        Returns:
            True if removed
        """
        if name in self._series:
            del self._series[name]
            if name in self._metadata:
                del self._metadata[name]
            return True
        return False
    
    def names(self) -> List[str]:
        """Get all series names.
        
        Returns:
            List of names
        """
        return list(self._series.keys())
    
    def series(self) -> List[TsData]:
        """Get all series.
        
        Returns:
            List of series
        """
        return list(self._series.values())
    
    def items(self) -> Iterator[tuple[str, TsData]]:
        """Iterate over name-series pairs.
        
        Yields:
            (name, series) tuples
        """
        return iter(self._series.items())
    
    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for series.
        
        Args:
            name: Series name
            
        Returns:
            Metadata or None
        """
        return self._metadata.get(name)
    
    def set_metadata(self, name: str, metadata: Dict[str, Any]):
        """Set metadata for series.
        
        Args:
            name: Series name
            metadata: Metadata dictionary
        """
        if name in self._series:
            self._metadata[name] = metadata
    
    def __len__(self) -> int:
        """Get number of series."""
        return len(self._series)
    
    def __contains__(self, name: str) -> bool:
        """Check if series exists."""
        return name in self._series
    
    def __getitem__(self, name: str) -> TsData:
        """Get series by name."""
        return self._series[name]
    
    def __setitem__(self, name: str, series: TsData):
        """Set series by name."""
        self._series[name] = series
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert collection to DataFrame.
        
        Returns:
            DataFrame with series as columns
        """
        if not self._series:
            return pd.DataFrame()
        
        # Find common frequency
        frequencies = {ts.frequency for ts in self._series.values()}
        if len(frequencies) > 1:
            raise ValueError("Cannot convert mixed-frequency collection to DataFrame")
        
        # Find date range
        min_start = min(ts.start for ts in self._series.values())
        max_end = max(ts.domain.end for ts in self._series.values())
        
        # Create common domain
        freq = list(frequencies)[0]
        domain = TsDomain.of(min_start, max_end.minus(min_start) + 1)
        
        # Build DataFrame
        data = {}
        for name, ts in self._series.items():
            # Align series to common domain
            aligned = self._align_to_domain(ts, domain)
            data[name] = aligned
        
        # Generate date index
        dates = self._generate_date_index(domain)
        
        return pd.DataFrame(data, index=dates)
    
    def _align_to_domain(self, ts: TsData, domain: TsDomain) -> np.ndarray:
        """Align series to common domain.
        
        Args:
            ts: Time series
            domain: Target domain
            
        Returns:
            Aligned values array
        """
        import numpy as np
        
        aligned = np.full(domain.length, np.nan)
        
        # Find overlap
        ts_start_idx = ts.start.minus(domain.start)
        ts_end_idx = ts_start_idx + ts.length
        
        # Copy overlapping values
        if ts_start_idx < domain.length and ts_end_idx > 0:
            copy_start = max(0, ts_start_idx)
            copy_end = min(domain.length, ts_end_idx)
            source_start = max(0, -ts_start_idx)
            source_end = source_start + (copy_end - copy_start)
            
            aligned[copy_start:copy_end] = ts.values[source_start:source_end]
        
        return aligned
    
    def _generate_date_index(self, domain: TsDomain) -> pd.DatetimeIndex:
        """Generate date index for domain.
        
        Args:
            domain: Time domain
            
        Returns:
            Date index
        """
        freq_map = {
            TsFrequency.MONTHLY: 'MS',
            TsFrequency.QUARTERLY: 'QS',
            TsFrequency.YEARLY: 'YS',
            TsFrequency.DAILY: 'D'
        }
        
        if domain.frequency in freq_map:
            # Convert start to timestamp
            start = domain.start
            if domain.frequency == TsFrequency.MONTHLY:
                start_date = pd.Timestamp(year=start.year, month=start.position + 1, day=1)
            elif domain.frequency == TsFrequency.QUARTERLY:
                month = start.position * 3 + 1
                start_date = pd.Timestamp(year=start.year, month=month, day=1)
            elif domain.frequency == TsFrequency.YEARLY:
                start_date = pd.Timestamp(year=start.year, month=1, day=1)
            else:
                # Daily
                start_date = pd.Timestamp('1970-01-01') + pd.Timedelta(days=start.epoch_period)
            
            return pd.date_range(
                start=start_date,
                periods=domain.length,
                freq=freq_map[domain.frequency]
            )
        else:
            # Unknown frequency
            return pd.RangeIndex(domain.length)


class TsCollectionBuilder:
    """Builder for time series collections."""
    
    def __init__(self):
        """Initialize builder."""
        self._collection = TsCollection()
    
    def add_series(self, name: str, values: Union[List[float], np.ndarray],
                   start: Union[TsPeriod, str], frequency: Optional[TsFrequency] = None) -> 'TsCollectionBuilder':
        """Add series to collection.
        
        Args:
            name: Series name
            values: Data values
            start: Start period or date string
            frequency: Frequency (inferred if not provided)
            
        Returns:
            Self for chaining
        """
        # Parse start period
        if isinstance(start, str):
            start_period = self._parse_date_string(start, frequency)
        else:
            start_period = start
        
        # Create series
        ts = TsData.of(start_period, values)
        
        # Add to collection
        self._collection.add(name, ts)
        
        return self
    
    def add_dataframe(self, df: pd.DataFrame, date_column: Optional[str] = None,
                      frequency: Optional[TsFrequency] = None) -> 'TsCollectionBuilder':
        """Add series from DataFrame.
        
        Args:
            df: DataFrame with series as columns
            date_column: Name of date column (uses index if None)
            frequency: Frequency (inferred if not provided)
            
        Returns:
            Self for chaining
        """
        # Get dates
        if date_column:
            dates = pd.to_datetime(df[date_column])
            value_columns = [col for col in df.columns if col != date_column]
        else:
            dates = pd.to_datetime(df.index)
            value_columns = df.columns
        
        # Infer frequency if needed
        if frequency is None:
            frequency = self._infer_frequency(dates)
        
        # Determine start period
        start_period = self._date_to_period(dates.iloc[0], frequency)
        
        # Add each column as series
        for col in value_columns:
            values = df[col].values
            ts = TsData.of(start_period, values)
            self._collection.add(str(col), ts)
        
        return self
    
    def with_metadata(self, name: str, metadata: Dict[str, Any]) -> 'TsCollectionBuilder':
        """Add metadata to series.
        
        Args:
            name: Series name
            metadata: Metadata dictionary
            
        Returns:
            Self for chaining
        """
        self._collection.set_metadata(name, metadata)
        return self
    
    def build(self) -> TsCollection:
        """Build the collection.
        
        Returns:
            Time series collection
        """
        return self._collection
    
    def _parse_date_string(self, date_str: str, frequency: Optional[TsFrequency]) -> TsPeriod:
        """Parse date string to period.
        
        Args:
            date_str: Date string
            frequency: Expected frequency
            
        Returns:
            Time period
        """
        # Try pandas parsing
        date = pd.to_datetime(date_str)
        
        # Infer frequency if needed
        if frequency is None:
            # Default to monthly
            frequency = TsFrequency.MONTHLY
        
        return self._date_to_period(date, frequency)
    
    def _date_to_period(self, date: pd.Timestamp, frequency: TsFrequency) -> TsPeriod:
        """Convert date to period.
        
        Args:
            date: Timestamp
            frequency: Frequency
            
        Returns:
            Time period
        """
        year = date.year
        
        if frequency == TsFrequency.MONTHLY:
            return TsPeriod.of(frequency, year, date.month - 1)
        elif frequency == TsFrequency.QUARTERLY:
            return TsPeriod.of(frequency, year, (date.month - 1) // 3)
        elif frequency == TsFrequency.YEARLY:
            return TsPeriod.of(frequency, year, 0)
        else:
            # Daily or undefined
            epoch = pd.Timestamp('1970-01-01')
            days = (date - epoch).days
            return TsPeriod(frequency, days)
    
    def _infer_frequency(self, dates: pd.Series) -> TsFrequency:
        """Infer frequency from dates.
        
        Args:
            dates: Date series
            
        Returns:
            Frequency
        """
        if len(dates) < 2:
            return TsFrequency.UNDEFINED
        
        # Use pandas inference
        freq = pd.infer_freq(dates)
        
        if freq:
            if freq.startswith('M'):
                return TsFrequency.MONTHLY
            elif freq.startswith('Q'):
                return TsFrequency.QUARTERLY
            elif freq.startswith('A') or freq.startswith('Y'):
                return TsFrequency.YEARLY
            elif freq.startswith('D'):
                return TsFrequency.DAILY
        
        return TsFrequency.UNDEFINED