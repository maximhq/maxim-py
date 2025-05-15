import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class Folder:
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
