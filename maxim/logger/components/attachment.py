import uuid
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class FileAttachment:
    path: str
    id: Optional[str] = None
    name: Optional[str] = None
    size: Optional[int] = None
    mime_type: Optional[str] = None
    tags: Optional[dict[str, str]] = None
    metadata: Optional[dict[str, Any]] = None
    timestamp: Optional[int] = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the file attachment to a dictionary.

        Returns:
            A dictionary containing the file attachment.
        """
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
    tags: Optional[dict[str, str]] = None
    metadata: Optional[dict[str, Any]] = None
    timestamp: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Initialize the file data attachment.
        """
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the file data attachment to a dictionary.

        Returns:
            A dictionary containing the file data attachment.
        """
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
    tags: Optional[dict[str, str]] = None
    metadata: Optional[dict[str, Any]] = None
    timestamp: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Initialize the url attachment.
        """
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the url attachment to a dictionary.

        Returns:
            A dictionary containing the url attachment.
        """
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


Attachment = Union[FileAttachment, FileDataAttachment, UrlAttachment]
