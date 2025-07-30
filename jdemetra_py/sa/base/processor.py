"""Base processor for seasonal adjustment."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

from ...toolkit.timeseries import TsData
from .specification import SaSpecification
from .results import SaResults


class SaProcessor(ABC):
    """Abstract base class for seasonal adjustment processors."""
    
    def __init__(self, specification: SaSpecification):
        """Initialize processor.
        
        Args:
            specification: SA specification
        """
        self.specification = specification
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, data: TsData, 
                metadata: Optional[Dict[str, Any]] = None) -> SaResults:
        """Process time series for seasonal adjustment.
        
        Args:
            data: Input time series
            metadata: Optional metadata
            
        Returns:
            Seasonal adjustment results
        """
        pass
    
    def validate_input(self, data: TsData) -> bool:
        """Validate input data.
        
        Args:
            data: Input time series
            
        Returns:
            True if valid
        """
        # Check for empty series
        if data.is_empty():
            self.logger.error("Empty time series")
            return False
        
        # Check minimum length (3 years)
        min_years = 3
        min_length = min_years * data.domain.frequency.periods_per_year
        if data.length < min_length:
            self.logger.error(f"Series too short: {data.length} < {min_length}")
            return False
        
        # Check for all missing values
        if np.all(np.isnan(data.values)):
            self.logger.error("All values are missing")
            return False
        
        # Check for constant series
        non_missing = data.values[~np.isnan(data.values)]
        if len(non_missing) > 0 and np.std(non_missing) == 0:
            self.logger.warning("Constant series detected")
        
        return True
    
    def prepare_data(self, data: TsData) -> TsData:
        """Prepare data for processing.
        
        Args:
            data: Input time series
            
        Returns:
            Prepared time series
        """
        # Clean extremities
        prepared = data.clean_extremities()
        
        # Check and apply transformation
        if self.specification.transformation.should_use_log(prepared):
            if np.any(prepared.values <= 0):
                # Add constant to make positive
                min_val = np.min(prepared.values)
                if min_val <= 0:
                    prepared = TsData.of(
                        prepared.start,
                        prepared.values - min_val + 1
                    )
            prepared = prepared.fn(np.log)
        
        return prepared
    
    @abstractmethod
    def get_name(self) -> str:
        """Get processor name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get processor version."""
        pass


class CompositeProcessor(SaProcessor):
    """Processor that combines multiple processors."""
    
    def __init__(self, processors: Dict[str, SaProcessor]):
        """Initialize composite processor.
        
        Args:
            processors: Dictionary of named processors
        """
        self.processors = processors
        # Use first processor's specification as default
        first_spec = next(iter(processors.values())).specification
        super().__init__(first_spec)
    
    def process(self, data: TsData,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, SaResults]:
        """Process with all processors.
        
        Args:
            data: Input time series
            metadata: Optional metadata
            
        Returns:
            Dictionary of results by processor name
        """
        results = {}
        
        for name, processor in self.processors.items():
            try:
                self.logger.info(f"Processing with {name}")
                result = processor.process(data, metadata)
                results[name] = result
            except Exception as e:
                self.logger.error(f"Error in {name}: {e}")
                results[name] = None
        
        return results
    
    def get_name(self) -> str:
        return "CompositeProcessor"
    
    def get_version(self) -> str:
        return "1.0.0"