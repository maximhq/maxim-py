import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FileAttachment:
    path: str
    id: Optional[str] = None
    name: Optional[str] = None
    size: Optional[int] = None
    mime_type: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[int] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "file",
            "name": self.name,
            "path": self.path,
            "mime_type": self.mime_type,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class FileDataAttachment:
    data: bytes
    id: Optional[str] = None
    name: Optional[str] = None
    size: Optional[int] = None
    mime_type: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[int] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "file_data",
            "name": self.name,
            "data": self.data,
            "mime_type": self.mime_type,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class UrlAttachment:
    url: str
    id: Optional[str] = None
    name: Optional[str] = None
    size: Optional[int] = None
    mime_type: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[int] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "url",
            "name": self.name,
            "url": self.url,
            "mime_type": self.mime_type,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
