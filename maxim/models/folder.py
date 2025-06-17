import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class Folder:
    """
    This class represents a folder.

    Attributes:
        id: The id of the folder.
        name: The name of the folder.
        parent_folder_id: The id of the parent folder.
        tags: The tags of the folder.
    """

    id: str
    name: str
    parent_folder_id: str
    tags: Optional[Dict[str, Union[str, int, bool, None]]] = None

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "Folder":
        return Folder(
            id=obj["id"],
            name=obj["name"],
            parent_folder_id=obj["parentFolderId"],
            tags=obj["tags"],
        )


class FolderEncoder(json.JSONEncoder):
    """
    This class represents a JSON encoder for Folder.
    """

    def default(self, o):
        if isinstance(o, Folder):
            return asdict(o)
        return super().default(o)


@dataclass
class Error:
    message: str


@dataclass
class MaximFolderResponse:
    data: Folder
    error: Optional[Error] = None


@dataclass
class MaximFoldersResponse:
    data: List[Folder]
    error: Optional[Error] = None
