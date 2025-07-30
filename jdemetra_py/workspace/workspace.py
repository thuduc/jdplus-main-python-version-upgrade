"""Workspace implementation for managing SA projects."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type, Union
from enum import Enum
from datetime import datetime
from pathlib import Path
import uuid

from ..toolkit.timeseries import TsData
from ..sa.base import SaSpecification, SaResults
from ..io.formats import TsCollection
from ..toolkit.calendars import CalendarDefinition


class WorkspaceItemType(Enum):
    """Types of workspace items."""
    
    SA_PROCESSING = "sa_processing"
    CALENDAR = "calendar"
    VARIABLE = "variable"
    SERIES = "series"
    DOCUMENT = "document"


@dataclass
class WorkspaceItem(ABC):
    """Base class for workspace items."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: WorkspaceItemType = WorkspaceItemType.DOCUMENT
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate item contents."""
        pass
    
    def update_modified(self):
        """Update modification timestamp."""
        self.modified = datetime.now()


@dataclass
class SAItem(WorkspaceItem):
    """Seasonal adjustment processing item."""
    
    type: WorkspaceItemType = field(default=WorkspaceItemType.SA_PROCESSING, init=False)
    
    # SA components
    series: Optional[TsData] = None
    specification: Optional[SaSpecification] = None
    results: Optional[SaResults] = None
    
    # Processing info
    method: str = ""  # "tramoseats", "x13", etc.
    status: str = "unprocessed"  # "unprocessed", "processing", "processed", "error"
    error_message: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate SA item."""
        return self.series is not None and self.specification is not None
    
    def is_processed(self) -> bool:
        """Check if item has been processed."""
        return self.status == "processed" and self.results is not None
    
    def clear_results(self):
        """Clear processing results."""
        self.results = None
        self.status = "unprocessed"
        self.error_message = None
        self.update_modified()


@dataclass
class CalendarItem(WorkspaceItem):
    """Calendar definition item."""
    
    type: WorkspaceItemType = field(default=WorkspaceItemType.CALENDAR, init=False)
    
    calendar: Optional[CalendarDefinition] = None
    
    def validate(self) -> bool:
        """Validate calendar item."""
        return self.calendar is not None


@dataclass
class VariableItem(WorkspaceItem):
    """Variable definition item."""
    
    type: WorkspaceItemType = field(default=WorkspaceItemType.VARIABLE, init=False)
    
    variable_type: str = ""  # "ts", "calendar", "regression", etc.
    data: Optional[Union[TsData, Any]] = None
    
    def validate(self) -> bool:
        """Validate variable item."""
        return self.variable_type and self.data is not None


class Workspace:
    """JDemetra+ workspace container."""
    
    def __init__(self, name: str = "Untitled"):
        """Initialize workspace.
        
        Args:
            name: Workspace name
        """
        self.name = name
        self.created = datetime.now()
        self.modified = datetime.now()
        self.items: Dict[str, WorkspaceItem] = {}
        self.collections: Dict[str, TsCollection] = {}
        self.metadata: Dict[str, Any] = {}
    
    # Item management
    
    def add_item(self, item: WorkspaceItem) -> str:
        """Add item to workspace.
        
        Args:
            item: Workspace item
            
        Returns:
            Item ID
        """
        if not item.name:
            item.name = f"{item.type.value}_{len(self.items) + 1}"
        
        self.items[item.id] = item
        self.modified = datetime.now()
        return item.id
    
    def get_item(self, item_id: str) -> Optional[WorkspaceItem]:
        """Get item by ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            Item or None
        """
        return self.items.get(item_id)
    
    def get_item_by_name(self, name: str, item_type: Optional[WorkspaceItemType] = None) -> Optional[WorkspaceItem]:
        """Get item by name.
        
        Args:
            name: Item name
            item_type: Optional type filter
            
        Returns:
            First matching item or None
        """
        for item in self.items.values():
            if item.name == name:
                if item_type is None or item.type == item_type:
                    return item
        return None
    
    def remove_item(self, item_id: str) -> bool:
        """Remove item from workspace.
        
        Args:
            item_id: Item ID
            
        Returns:
            True if removed
        """
        if item_id in self.items:
            del self.items[item_id]
            self.modified = datetime.now()
            return True
        return False
    
    def list_items(self, item_type: Optional[WorkspaceItemType] = None) -> List[WorkspaceItem]:
        """List workspace items.
        
        Args:
            item_type: Optional type filter
            
        Returns:
            List of items
        """
        items = list(self.items.values())
        
        if item_type:
            items = [item for item in items if item.type == item_type]
        
        return sorted(items, key=lambda x: x.name)
    
    # Collection management
    
    def add_collection(self, name: str, collection: TsCollection):
        """Add time series collection.
        
        Args:
            name: Collection name
            collection: Time series collection
        """
        self.collections[name] = collection
        self.modified = datetime.now()
    
    def get_collection(self, name: str) -> Optional[TsCollection]:
        """Get collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            Collection or None
        """
        return self.collections.get(name)
    
    def list_collections(self) -> List[str]:
        """List collection names.
        
        Returns:
            Collection names
        """
        return sorted(self.collections.keys())
    
    # SA processing helpers
    
    def create_sa_item(self, name: str, series: TsData, 
                      specification: SaSpecification, method: str) -> SAItem:
        """Create and add SA processing item.
        
        Args:
            name: Item name
            series: Time series
            specification: SA specification
            method: Method name
            
        Returns:
            Created item
        """
        item = SAItem(
            name=name,
            series=series,
            specification=specification,
            method=method
        )
        
        self.add_item(item)
        return item
    
    def process_sa_item(self, item_id: str, processor: 'SaProcessor') -> bool:
        """Process SA item.
        
        Args:
            item_id: Item ID
            processor: SA processor
            
        Returns:
            True if successful
        """
        item = self.get_item(item_id)
        if not isinstance(item, SAItem):
            return False
        
        if not item.validate():
            return False
        
        try:
            item.status = "processing"
            item.update_modified()
            
            # Process
            results = processor.process(item.series)
            
            item.results = results
            item.status = "processed"
            item.error_message = None
            item.update_modified()
            
            self.modified = datetime.now()
            return True
            
        except Exception as e:
            item.status = "error"
            item.error_message = str(e)
            item.update_modified()
            return False
    
    def batch_process(self, processor_map: Dict[str, 'SaProcessor'],
                     item_filter: Optional[callable] = None) -> Dict[str, bool]:
        """Batch process SA items.
        
        Args:
            processor_map: Map of method names to processors
            item_filter: Optional filter function
            
        Returns:
            Map of item IDs to success status
        """
        results = {}
        
        for item in self.list_items(WorkspaceItemType.SA_PROCESSING):
            if not isinstance(item, SAItem):
                continue
                
            if item_filter and not item_filter(item):
                continue
            
            processor = processor_map.get(item.method)
            if processor:
                results[item.id] = self.process_sa_item(item.id, processor)
            else:
                results[item.id] = False
        
        return results
    
    # Validation and info
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate workspace contents.
        
        Returns:
            Dictionary of errors by item ID
        """
        errors = {}
        
        for item_id, item in self.items.items():
            item_errors = []
            
            # Basic validation
            if not item.name:
                item_errors.append("Item has no name")
            
            if not item.validate():
                item_errors.append("Item validation failed")
            
            # Type-specific validation
            if isinstance(item, SAItem):
                if item.series is None:
                    item_errors.append("SA item has no series")
                if item.specification is None:
                    item_errors.append("SA item has no specification")
                if not item.method:
                    item_errors.append("SA item has no method")
            
            if item_errors:
                errors[item_id] = item_errors
        
        return errors
    
    def get_info(self) -> Dict[str, Any]:
        """Get workspace information.
        
        Returns:
            Info dictionary
        """
        # Count items by type
        type_counts = {}
        for item in self.items.values():
            type_counts[item.type.value] = type_counts.get(item.type.value, 0) + 1
        
        # Count SA items by status
        sa_status_counts = {}
        for item in self.items.values():
            if isinstance(item, SAItem):
                sa_status_counts[item.status] = sa_status_counts.get(item.status, 0) + 1
        
        return {
            "name": self.name,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "total_items": len(self.items),
            "items_by_type": type_counts,
            "collections": len(self.collections),
            "sa_items_by_status": sa_status_counts,
            "metadata": self.metadata
        }
    
    def clear(self):
        """Clear workspace contents."""
        self.items.clear()
        self.collections.clear()
        self.metadata.clear()
        self.modified = datetime.now()