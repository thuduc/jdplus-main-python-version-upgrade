"""Data providers for reading time series from various sources."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from datetime import datetime

from ..toolkit.timeseries import TsData, TsPeriod, TsFrequency, TsDomain
from .formats import TsCollection, DataFormat


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def read(self, source: Any, **kwargs) -> Union[TsData, TsCollection]:
        """Read time series data from source.
        
        Args:
            source: Data source
            **kwargs: Provider-specific options
            
        Returns:
            Single series or collection
        """
        pass
    
    @abstractmethod
    def write(self, data: Union[TsData, TsCollection], target: Any, **kwargs):
        """Write time series data to target.
        
        Args:
            data: Time series data
            target: Target location
            **kwargs: Provider-specific options
        """
        pass
    
    @abstractmethod
    def can_handle(self, source: Any) -> bool:
        """Check if provider can handle the source.
        
        Args:
            source: Data source
            
        Returns:
            True if provider can handle source
        """
        pass


class FileDataProvider(DataProvider):
    """Base class for file-based data providers."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """Initialize file provider.
        
        Args:
            encoding: File encoding
        """
        self.encoding = encoding
    
    def can_handle(self, source: Any) -> bool:
        """Check if source is a valid file path."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            return path.exists() and path.is_file()
        return False


class CsvDataProvider(FileDataProvider):
    """CSV file data provider."""
    
    def __init__(self, encoding: str = 'utf-8', delimiter: str = ',',
                 date_format: Optional[str] = None):
        """Initialize CSV provider.
        
        Args:
            encoding: File encoding
            delimiter: Field delimiter
            date_format: Date format string
        """
        super().__init__(encoding)
        self.delimiter = delimiter
        self.date_format = date_format
    
    def read(self, source: Union[str, Path], **kwargs) -> Union[TsData, TsCollection]:
        """Read time series from CSV file.
        
        Args:
            source: CSV file path
            **kwargs: Additional pandas read_csv options
            
        Returns:
            Time series data
        """
        # Read CSV with pandas
        df = pd.read_csv(
            source,
            delimiter=self.delimiter,
            encoding=self.encoding,
            **kwargs
        )
        
        # Detect format
        if self._is_vertical_format(df):
            return self._read_vertical(df)
        else:
            return self._read_horizontal(df)
    
    def write(self, data: Union[TsData, TsCollection], target: Union[str, Path], **kwargs):
        """Write time series to CSV file.
        
        Args:
            data: Time series data
            target: Target file path
            **kwargs: Additional pandas to_csv options
        """
        # Convert to DataFrame
        if isinstance(data, TsData):
            df = self._ts_to_dataframe(data)
        else:
            df = self._collection_to_dataframe(data)
        
        # Write CSV
        df.to_csv(
            target,
            sep=self.delimiter,
            encoding=self.encoding,
            index=True,
            **kwargs
        )
    
    def _is_vertical_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in vertical format."""
        # Vertical format has columns like: date, series_name, value
        required_cols = {'date', 'value'}
        cols_lower = {col.lower() for col in df.columns}
        return required_cols.issubset(cols_lower)
    
    def _read_vertical(self, df: pd.DataFrame) -> TsCollection:
        """Read vertical format CSV."""
        collection = TsCollection()
        
        # Normalize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], format=self.date_format)
        
        # Group by series name if available
        if 'series' in df.columns or 'name' in df.columns:
            series_col = 'series' if 'series' in df.columns else 'name'
            for name, group in df.groupby(series_col):
                ts = self._create_ts_from_group(group)
                collection.add(str(name), ts)
        else:
            # Single series
            ts = self._create_ts_from_group(df)
            collection.add("series", ts)
        
        return collection
    
    def _read_horizontal(self, df: pd.DataFrame) -> TsCollection:
        """Read horizontal format CSV."""
        collection = TsCollection()
        
        # First column should be dates
        if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format=self.date_format)
        
        # Each additional column is a series
        date_col = df.columns[0]
        for col in df.columns[1:]:
            # Create TsData
            values = df[col].values
            dates = df[date_col]
            
            # Determine frequency
            freq = self._infer_frequency(dates)
            start_period = self._date_to_period(dates.iloc[0], freq)
            
            ts = TsData.of(start_period, values)
            collection.add(col, ts)
        
        return collection
    
    def _create_ts_from_group(self, group: pd.DataFrame) -> TsData:
        """Create TsData from grouped DataFrame."""
        # Sort by date
        group = group.sort_values('date')
        
        # Extract values
        values = group['value'].values
        dates = group['date']
        
        # Determine frequency
        freq = self._infer_frequency(dates)
        start_period = self._date_to_period(dates.iloc[0], freq)
        
        return TsData.of(start_period, values)
    
    def _infer_frequency(self, dates: pd.Series) -> TsFrequency:
        """Infer frequency from dates."""
        if len(dates) < 2:
            return TsFrequency.UNDEFINED
        
        # Use pandas frequency inference
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
        
        # Manual check
        diff = (dates.iloc[1] - dates.iloc[0]).days
        if 28 <= diff <= 31:
            return TsFrequency.MONTHLY
        elif 89 <= diff <= 92:
            return TsFrequency.QUARTERLY
        elif 364 <= diff <= 366:
            return TsFrequency.YEARLY
        else:
            return TsFrequency.UNDEFINED
    
    def _date_to_period(self, date: pd.Timestamp, freq: TsFrequency) -> TsPeriod:
        """Convert date to TsPeriod."""
        year = date.year
        
        if freq == TsFrequency.MONTHLY:
            return TsPeriod.of(freq, year, date.month - 1)
        elif freq == TsFrequency.QUARTERLY:
            return TsPeriod.of(freq, year, (date.month - 1) // 3)
        elif freq == TsFrequency.YEARLY:
            return TsPeriod.of(freq, year, 0)
        else:
            # For daily or undefined, use epoch
            epoch = pd.Timestamp('1970-01-01')
            days = (date - epoch).days
            return TsPeriod(freq, days)
    
    def _ts_to_dataframe(self, ts: TsData) -> pd.DataFrame:
        """Convert TsData to DataFrame."""
        # Generate date index
        dates = self._generate_dates(ts)
        
        # Create DataFrame
        df = pd.DataFrame({
            'value': ts.values
        }, index=dates)
        
        return df
    
    def _collection_to_dataframe(self, collection: TsCollection) -> pd.DataFrame:
        """Convert TsCollection to DataFrame."""
        frames = {}
        
        for name, ts in collection.items():
            dates = self._generate_dates(ts)
            frames[name] = pd.Series(ts.values, index=dates, name=name)
        
        return pd.DataFrame(frames)
    
    def _generate_dates(self, ts: TsData) -> pd.DatetimeIndex:
        """Generate dates for time series."""
        start = ts.start
        freq_map = {
            TsFrequency.MONTHLY: 'MS',
            TsFrequency.QUARTERLY: 'QS',
            TsFrequency.YEARLY: 'YS',
            TsFrequency.DAILY: 'D'
        }
        
        if ts.frequency in freq_map:
            # Convert start period to timestamp
            if ts.frequency == TsFrequency.MONTHLY:
                start_date = pd.Timestamp(year=start.year, month=start.position + 1, day=1)
            elif ts.frequency == TsFrequency.QUARTERLY:
                month = start.position * 3 + 1
                start_date = pd.Timestamp(year=start.year, month=month, day=1)
            elif ts.frequency == TsFrequency.YEARLY:
                start_date = pd.Timestamp(year=start.year, month=1, day=1)
            else:
                # Daily - use epoch
                start_date = pd.Timestamp('1970-01-01') + pd.Timedelta(days=start.epoch_period)
            
            # Generate date range
            return pd.date_range(
                start=start_date,
                periods=ts.length,
                freq=freq_map[ts.frequency]
            )
        else:
            # Unknown frequency - use integer index
            return pd.RangeIndex(start=0, stop=ts.length)


class ExcelDataProvider(FileDataProvider):
    """Excel file data provider."""
    
    def read(self, source: Union[str, Path], sheet_name: Union[str, int] = 0,
             **kwargs) -> Union[TsData, TsCollection]:
        """Read time series from Excel file.
        
        Args:
            source: Excel file path
            sheet_name: Sheet name or index
            **kwargs: Additional pandas read_excel options
            
        Returns:
            Time series data
        """
        # Read Excel with pandas
        df = pd.read_excel(
            source,
            sheet_name=sheet_name,
            **kwargs
        )
        
        # Use CSV provider logic for parsing
        csv_provider = CsvDataProvider()
        if csv_provider._is_vertical_format(df):
            return csv_provider._read_vertical(df)
        else:
            return csv_provider._read_horizontal(df)
    
    def write(self, data: Union[TsData, TsCollection], target: Union[str, Path],
              sheet_name: str = 'Sheet1', **kwargs):
        """Write time series to Excel file.
        
        Args:
            data: Time series data
            target: Target file path
            sheet_name: Sheet name
            **kwargs: Additional pandas to_excel options
        """
        # Convert to DataFrame
        csv_provider = CsvDataProvider()
        if isinstance(data, TsData):
            df = csv_provider._ts_to_dataframe(data)
        else:
            df = csv_provider._collection_to_dataframe(data)
        
        # Write Excel
        with pd.ExcelWriter(target, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=True, **kwargs)


class XmlDataProvider(FileDataProvider):
    """XML file data provider (TS-XML format)."""
    
    def read(self, source: Union[str, Path], **kwargs) -> TsCollection:
        """Read time series from XML file.
        
        Args:
            source: XML file path
            **kwargs: Additional options
            
        Returns:
            Time series collection
        """
        tree = ET.parse(source)
        root = tree.getroot()
        
        collection = TsCollection()
        
        # Handle different XML formats
        if root.tag == 'data':
            # TS-XML format
            for series_elem in root.findall('.//series'):
                ts = self._parse_series_element(series_elem)
                name = series_elem.get('name', 'series')
                collection.add(name, ts)
        
        return collection
    
    def write(self, data: Union[TsData, TsCollection], target: Union[str, Path], **kwargs):
        """Write time series to XML file.
        
        Args:
            data: Time series data
            target: Target file path
            **kwargs: Additional options
        """
        root = ET.Element('data')
        
        if isinstance(data, TsData):
            series_elem = self._create_series_element('series', data)
            root.append(series_elem)
        else:
            for name, ts in data.items():
                series_elem = self._create_series_element(name, ts)
                root.append(series_elem)
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(target, encoding=self.encoding, xml_declaration=True)
    
    def _parse_series_element(self, elem: ET.Element) -> TsData:
        """Parse series element from XML."""
        # Extract frequency
        freq_str = elem.get('frequency', 'undefined').upper()
        freq = TsFrequency[freq_str] if freq_str in TsFrequency.__members__ else TsFrequency.UNDEFINED
        
        # Extract start period
        start_year = int(elem.get('start_year', '2000'))
        start_period = int(elem.get('start_period', '0'))
        
        start = TsPeriod.of(freq, start_year, start_period)
        
        # Extract values
        values_text = elem.find('values').text
        values = [float(v) for v in values_text.split()]
        
        return TsData.of(start, values)
    
    def _create_series_element(self, name: str, ts: TsData) -> ET.Element:
        """Create series element for XML."""
        elem = ET.Element('series')
        elem.set('name', name)
        elem.set('frequency', ts.frequency.name)
        elem.set('start_year', str(ts.start.year))
        elem.set('start_period', str(ts.start.position))
        
        # Add values
        values_elem = ET.SubElement(elem, 'values')
        values_elem.text = ' '.join(str(v) for v in ts.values)
        
        return elem


class JsonDataProvider(FileDataProvider):
    """JSON file data provider."""
    
    def read(self, source: Union[str, Path], **kwargs) -> Union[TsData, TsCollection]:
        """Read time series from JSON file.
        
        Args:
            source: JSON file path
            **kwargs: Additional options
            
        Returns:
            Time series data
        """
        with open(source, 'r', encoding=self.encoding) as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'values' in data and 'start' in data:
                # Single series
                return self._parse_series_dict(data)
            else:
                # Collection
                collection = TsCollection()
                for name, series_data in data.items():
                    if isinstance(series_data, dict) and 'values' in series_data:
                        ts = self._parse_series_dict(series_data)
                        collection.add(name, ts)
                return collection
        else:
            raise ValueError("Invalid JSON format")
    
    def write(self, data: Union[TsData, TsCollection], target: Union[str, Path], **kwargs):
        """Write time series to JSON file.
        
        Args:
            data: Time series data
            target: Target file path
            **kwargs: Additional json.dump options
        """
        if isinstance(data, TsData):
            json_data = self._series_to_dict(data)
        else:
            json_data = {}
            for name, ts in data.items():
                json_data[name] = self._series_to_dict(ts)
        
        with open(target, 'w', encoding=self.encoding) as f:
            json.dump(json_data, f, indent=2, **kwargs)
    
    def _parse_series_dict(self, data: Dict) -> TsData:
        """Parse series from dictionary."""
        # Extract values
        values = data['values']
        
        # Extract start period
        start_info = data['start']
        if isinstance(start_info, dict):
            freq = TsFrequency[start_info['frequency'].upper()]
            year = start_info['year']
            period = start_info['period']
            start = TsPeriod.of(freq, year, period)
        else:
            # Assume monthly starting from given year
            start = TsPeriod.of(TsFrequency.MONTHLY, int(start_info), 0)
        
        return TsData.of(start, values)
    
    def _series_to_dict(self, ts: TsData) -> Dict:
        """Convert series to dictionary."""
        return {
            'start': {
                'frequency': ts.frequency.name,
                'year': ts.start.year,
                'period': ts.start.position
            },
            'values': ts.values.tolist(),
            'length': ts.length
        }