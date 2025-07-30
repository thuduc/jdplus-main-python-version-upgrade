"""Workspace persistence implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import pickle

from .workspace import (
    Workspace, WorkspaceItem, WorkspaceItemType,
    SAItem, CalendarItem, VariableItem
)
from ..io.formats import TsCollection
from ..io.providers import JsonDataProvider, XmlDataProvider
from ..sa.base import SaSpecification
from ..sa.tramoseats import TramoSeatsSpecification
from ..sa.x13 import X13Specification


class WorkspacePersistence(ABC):
    """Abstract base class for workspace persistence."""
    
    @abstractmethod
    def save(self, workspace: Workspace, path: Path):
        """Save workspace to file.
        
        Args:
            workspace: Workspace to save
            path: Target file path
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> Workspace:
        """Load workspace from file.
        
        Args:
            path: Source file path
            
        Returns:
            Loaded workspace
        """
        pass


class JsonWorkspacePersistence(WorkspacePersistence):
    """JSON-based workspace persistence."""
    
    def save(self, workspace: Workspace, path: Path):
        """Save workspace to JSON file."""
        # Convert workspace to dictionary
        data = {
            "version": "1.0",
            "name": workspace.name,
            "created": workspace.created.isoformat(),
            "modified": workspace.modified.isoformat(),
            "metadata": workspace.metadata,
            "items": {},
            "collections": {}
        }
        
        # Serialize items
        for item_id, item in workspace.items.items():
            data["items"][item_id] = self._serialize_item(item)
        
        # Serialize collections
        provider = JsonDataProvider()
        for name, collection in workspace.collections.items():
            # Save collection to temp dict
            collection_data = {}
            for series_name, series in collection.items():
                collection_data[series_name] = provider._series_to_dict(series)
            data["collections"][name] = collection_data
        
        # Write JSON
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path) -> Workspace:
        """Load workspace from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Create workspace
        workspace = Workspace(data.get("name", "Untitled"))
        workspace.created = datetime.fromisoformat(data.get("created", datetime.now().isoformat()))
        workspace.modified = datetime.fromisoformat(data.get("modified", datetime.now().isoformat()))
        workspace.metadata = data.get("metadata", {})
        
        # Load items
        for item_id, item_data in data.get("items", {}).items():
            item = self._deserialize_item(item_data)
            if item:
                item.id = item_id
                workspace.items[item_id] = item
        
        # Load collections
        provider = JsonDataProvider()
        for name, collection_data in data.get("collections", {}).items():
            collection = TsCollection()
            for series_name, series_data in collection_data.items():
                series = provider._parse_series_dict(series_data)
                collection.add(series_name, series)
            workspace.collections[name] = collection
        
        return workspace
    
    def _serialize_item(self, item: WorkspaceItem) -> Dict[str, Any]:
        """Serialize workspace item."""
        base_data = {
            "type": item.type.value,
            "name": item.name,
            "created": item.created.isoformat(),
            "modified": item.modified.isoformat(),
            "metadata": item.metadata
        }
        
        if isinstance(item, SAItem):
            # Serialize SA item
            sa_data = {
                "method": item.method,
                "status": item.status,
                "error_message": item.error_message
            }
            
            # Serialize series
            if item.series:
                provider = JsonDataProvider()
                sa_data["series"] = provider._series_to_dict(item.series)
            
            # Serialize specification
            if item.specification:
                sa_data["specification"] = self._serialize_specification(item.specification)
            
            # Don't serialize results - they should be recomputed
            
            base_data.update(sa_data)
        
        elif isinstance(item, CalendarItem):
            # Serialize calendar
            if item.calendar:
                base_data["calendar"] = {
                    "name": item.calendar.name,
                    # Add calendar serialization
                }
        
        elif isinstance(item, VariableItem):
            # Serialize variable
            base_data["variable_type"] = item.variable_type
            # Add variable data serialization based on type
        
        return base_data
    
    def _deserialize_item(self, data: Dict[str, Any]) -> Optional[WorkspaceItem]:
        """Deserialize workspace item."""
        item_type = WorkspaceItemType(data.get("type", "document"))
        
        if item_type == WorkspaceItemType.SA_PROCESSING:
            item = SAItem()
            item.method = data.get("method", "")
            item.status = data.get("status", "unprocessed")
            item.error_message = data.get("error_message")
            
            # Deserialize series
            if "series" in data:
                provider = JsonDataProvider()
                item.series = provider._parse_series_dict(data["series"])
            
            # Deserialize specification
            if "specification" in data:
                item.specification = self._deserialize_specification(
                    data["specification"], item.method
                )
        
        elif item_type == WorkspaceItemType.CALENDAR:
            item = CalendarItem()
            # Deserialize calendar
        
        elif item_type == WorkspaceItemType.VARIABLE:
            item = VariableItem()
            item.variable_type = data.get("variable_type", "")
            # Deserialize variable data
        
        else:
            return None
        
        # Set common fields
        item.name = data.get("name", "")
        item.created = datetime.fromisoformat(data.get("created", datetime.now().isoformat()))
        item.modified = datetime.fromisoformat(data.get("modified", datetime.now().isoformat()))
        item.metadata = data.get("metadata", {})
        
        return item
    
    def _serialize_specification(self, spec: SaSpecification) -> Dict[str, Any]:
        """Serialize SA specification."""
        if hasattr(spec, 'to_dict'):
            return spec.to_dict()
        else:
            # Generic serialization
            return {
                "class": spec.__class__.__name__,
                # Add specific fields
            }
    
    def _deserialize_specification(self, data: Dict[str, Any], method: str) -> Optional[SaSpecification]:
        """Deserialize SA specification."""
        if method == "tramoseats":
            spec = TramoSeatsSpecification()
            if hasattr(spec, 'from_dict'):
                return spec.from_dict(data)
        elif method == "x13":
            spec = X13Specification()
            if hasattr(spec, 'from_dict'):
                return spec.from_dict(data)
        
        return None


class XmlWorkspacePersistence(WorkspacePersistence):
    """XML-based workspace persistence (JDemetra+ compatible)."""
    
    def save(self, workspace: Workspace, path: Path):
        """Save workspace to XML file."""
        # Create root element
        root = ET.Element("workspace")
        root.set("version", "2.2.0")
        root.set("name", workspace.name)
        
        # Add metadata
        metadata = ET.SubElement(root, "metadata")
        metadata.set("created", workspace.created.isoformat())
        metadata.set("modified", workspace.modified.isoformat())
        
        for key, value in workspace.metadata.items():
            meta_item = ET.SubElement(metadata, "item")
            meta_item.set("key", key)
            meta_item.text = str(value)
        
        # Add items
        items_elem = ET.SubElement(root, "items")
        for item_id, item in workspace.items.items():
            item_elem = self._item_to_xml(item)
            if item_elem is not None:
                item_elem.set("id", item_id)
                items_elem.append(item_elem)
        
        # Add collections
        collections_elem = ET.SubElement(root, "collections")
        provider = XmlDataProvider()
        for name, collection in workspace.collections.items():
            coll_elem = ET.SubElement(collections_elem, "collection")
            coll_elem.set("name", name)
            
            # Add series
            for series_name, series in collection.items():
                series_elem = provider._create_series_element(series_name, series)
                coll_elem.append(series_elem)
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(path, encoding='utf-8', xml_declaration=True)
    
    def load(self, path: Path) -> Workspace:
        """Load workspace from XML file."""
        tree = ET.parse(path)
        root = tree.getroot()
        
        # Create workspace
        workspace = Workspace(root.get("name", "Untitled"))
        
        # Load metadata
        metadata_elem = root.find("metadata")
        if metadata_elem is not None:
            created = metadata_elem.get("created")
            if created:
                workspace.created = datetime.fromisoformat(created)
            
            modified = metadata_elem.get("modified")
            if modified:
                workspace.modified = datetime.fromisoformat(modified)
            
            for meta_item in metadata_elem.findall("item"):
                key = meta_item.get("key")
                value = meta_item.text
                if key:
                    workspace.metadata[key] = value
        
        # Load items
        items_elem = root.find("items")
        if items_elem is not None:
            for item_elem in items_elem:
                item = self._xml_to_item(item_elem)
                if item:
                    item_id = item_elem.get("id", item.id)
                    item.id = item_id
                    workspace.items[item_id] = item
        
        # Load collections
        collections_elem = root.find("collections")
        if collections_elem is not None:
            provider = XmlDataProvider()
            for coll_elem in collections_elem.findall("collection"):
                name = coll_elem.get("name", "collection")
                collection = TsCollection()
                
                for series_elem in coll_elem.findall("series"):
                    series_name = series_elem.get("name", "series")
                    series = provider._parse_series_element(series_elem)
                    collection.add(series_name, series)
                
                workspace.collections[name] = collection
        
        return workspace
    
    def _item_to_xml(self, item: WorkspaceItem) -> Optional[ET.Element]:
        """Convert item to XML element."""
        elem = ET.Element("item")
        elem.set("type", item.type.value)
        elem.set("name", item.name)
        elem.set("created", item.created.isoformat())
        elem.set("modified", item.modified.isoformat())
        
        # Add metadata
        if item.metadata:
            meta_elem = ET.SubElement(elem, "metadata")
            for key, value in item.metadata.items():
                meta_item = ET.SubElement(meta_elem, "item")
                meta_item.set("key", key)
                meta_item.text = str(value)
        
        # Type-specific serialization
        if isinstance(item, SAItem):
            elem.tag = "saitem"
            elem.set("method", item.method)
            elem.set("status", item.status)
            
            if item.error_message:
                elem.set("error", item.error_message)
            
            # Add series
            if item.series:
                provider = XmlDataProvider()
                series_elem = provider._create_series_element("series", item.series)
                elem.append(series_elem)
            
            # Add specification
            if item.specification:
                spec_elem = ET.SubElement(elem, "specification")
                spec_elem.set("class", item.specification.__class__.__name__)
                # Add specification details
        
        return elem
    
    def _xml_to_item(self, elem: ET.Element) -> Optional[WorkspaceItem]:
        """Convert XML element to item."""
        item_type = elem.get("type")
        
        if elem.tag == "saitem" or item_type == WorkspaceItemType.SA_PROCESSING.value:
            item = SAItem()
            item.method = elem.get("method", "")
            item.status = elem.get("status", "unprocessed")
            item.error_message = elem.get("error")
            
            # Load series
            series_elem = elem.find("series")
            if series_elem is not None:
                provider = XmlDataProvider()
                item.series = provider._parse_series_element(series_elem)
            
            # Load specification
            spec_elem = elem.find("specification")
            if spec_elem is not None:
                # Deserialize specification based on class
                pass
        
        else:
            return None
        
        # Set common fields
        item.name = elem.get("name", "")
        created = elem.get("created")
        if created:
            item.created = datetime.fromisoformat(created)
        modified = elem.get("modified")
        if modified:
            item.modified = datetime.fromisoformat(modified)
        
        # Load metadata
        meta_elem = elem.find("metadata")
        if meta_elem is not None:
            for meta_item in meta_elem.findall("item"):
                key = meta_item.get("key")
                value = meta_item.text
                if key:
                    item.metadata[key] = value
        
        return item


class BinaryWorkspacePersistence(WorkspacePersistence):
    """Binary (pickle) workspace persistence for fast save/load."""
    
    def save(self, workspace: Workspace, path: Path):
        """Save workspace to binary file."""
        with open(path, 'wb') as f:
            pickle.dump(workspace, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, path: Path) -> Workspace:
        """Load workspace from binary file."""
        with open(path, 'rb') as f:
            return pickle.load(f)