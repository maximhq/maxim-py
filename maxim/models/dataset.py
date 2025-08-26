from dataclasses import dataclass
import json
from typing import Any, Callable, Literal, Optional, TypeVar, TypedDict, Union
from ..logger.components.attachment import Attachment, FileAttachment, FileDataAttachment, UrlAttachment

class VariableType(str):
    """
    This class represents the type of a variable.
    """

    TEXT = "text"
    JSON = "json"
    FILE = "file"

class DatasetAttachmentUploadURL(TypedDict):
    """
    This class represents a signed upload target returned by the API.
    """
    url: str
    key: str

@dataclass
class Variable:
    """
    This class represents a variable.
    """
    type: Literal["text", "json", "file"]
    payload: Union[str, dict[str, Any], list[Attachment]]

    def to_json(self) -> dict[str, Any]:
        """Convert the Variable to a JSON-like dictionary."""

        if (
            isinstance(self.payload, list)
            and all(
                isinstance(item, (FileAttachment, FileDataAttachment, UrlAttachment))
                for item in self.payload
            )
        ):
            return {"type": self.type, "payload": [item.to_dict() for item in self.payload]}

        return {"type": self.type, "payload": self.payload}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Variable":
        """Create a Variable from a JSON-like dictionary.

        Args:
            data: Dictionary containing the variable data with 'type' and 'payload' keys

        Returns:
            Variable: The created variable instance

        Raises:
            ValueError/TypeError: If the data format is invalid or required fields are missing.
            Note: For type 'file', payload must be a list of typed Attachment objects
            (FileAttachment, FileDataAttachment, UrlAttachment).
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        if "type" not in data:
            raise ValueError("missing 'type'")

        if "payload" not in data:
            raise ValueError("missing 'payload'")

        var_type = data["type"]
        payload = data["payload"]

        # Validate type
        allowed = {VariableType.TEXT, VariableType.JSON, VariableType.FILE}
        if var_type not in allowed:
            raise ValueError(f"invalid type: {var_type}")

        validators = {
            # Accept either plain string or {"text": "..."} form
            "text": lambda p: isinstance(p, str)
            or (isinstance(p, dict) and isinstance(p.get("text"), str)),
            "json": lambda p: isinstance(p, dict),
            # Accept either List[Attachment] or {"files": [...]}
            "file": lambda p: (
                isinstance(p, list)
                and all(isinstance(item, (FileAttachment, FileDataAttachment, UrlAttachment)) for item in p)
            ) or (
                isinstance(p, dict)
                and isinstance(p.get("files"), list)
            ),
        }
        if not validators[var_type](payload):
            messages = {
                "text": "payload for 'text' must be str or dict with key 'text'",
                "json": "payload for 'json' must be dict",
                "file": "payload for 'file' must be List[Attachment] or dict with key 'files'",
            }
            raise TypeError(messages[var_type])

        return cls(type=var_type, payload=payload)

@dataclass
class VariableFileAttachment:
    """
    This class represents a variable file attachment.
    """
    id: str
    url: str
    hosted: bool
    props: dict[str, Any]
    prefix: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in {
            "id": self.id,
            "url": self.url,
            "hosted": self.hosted,
            "prefix": self.prefix,
            "props": self.props,
        }.items() if v is not None}

@dataclass
class FileVariablePayload:
    """
    This class represents a file variable payload.
    """
    text: Optional[str]
    files: list[VariableFileAttachment]
    entry_id: Optional[str] = None

@dataclass
class DatasetEntry:
    """
    This class represents a dataset entry.
    """
    entry: dict[str, Variable]
    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> "DatasetEntry":
        """
        Convert a single dictionary to a DatasetEntry object.

        Args:
            data_dict: Dictionary representing a dataset entry

        Returns:
            DatasetEntry: DatasetEntry object

        Raises:
            TypeError: If data_dict is not a dict
        """
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a dict")

        variables = {}
        for column_name, value in data_dict.items():
            if isinstance(value, str):
                var_type = VariableType.TEXT
                payload = value
            elif isinstance(value, dict):
                var_type = VariableType.JSON
                payload = value
            elif isinstance(value, list) and all(isinstance(item, (FileAttachment, FileDataAttachment, UrlAttachment)) for item in value):
                # Check for List[Attachment]
                var_type = VariableType.FILE
                payload = value
            else:
                # Default to text for unknown types
                var_type = VariableType.TEXT
                payload = str(value)

            variables[column_name] = Variable(type=var_type, payload=payload)

        return cls(entry=variables)


@dataclass
class DatasetEntryWithRowNo:
    """
    This class represents a dataset entry with a row number.
    """
    row_no: int
    column_name: str
    type: Literal["text", "json", "file"]
    payload: Union[str, dict[str, Any], list[Attachment]]
    column_id: Optional[str] = None

    @classmethod
    def from_dataset_entry(cls, dataset_entry: DatasetEntry, row_no: int) -> list["DatasetEntryWithRowNo"]:
        """
        Convert a DatasetEntry to a list of DatasetEntryWithRowNo objects.

        Args:
            dataset_entry: The DatasetEntry to convert
            row_no: The row number to assign

        Returns:
            list[DatasetEntryWithRowNo]: One object per column

        Raises:
            TypeError: If dataset_entry is not a DatasetEntry
        """
        if not isinstance(dataset_entry, DatasetEntry):
            raise TypeError("dataset_entry must be a DatasetEntry")
        result: list[DatasetEntryWithRowNo] = []
        for column_name, variable in dataset_entry.entry.items():
            result.append(cls(
                row_no=row_no,
                column_name=column_name,
                type=variable.type,
                payload=variable.payload,
            ))

        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the DatasetEntryWithRowNo to a dictionary.

        Returns:
            dict[str, Any]: Dictionary with rowNo, columnName, type, and value
        """
        value: Any
        if self.type == "file":
            value = []
        elif self.type == "json":
            # Stringify JSON payloads before sending to API
            value = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        else:
            value = self.payload

        result = {
            "rowNo": self.row_no,
            "columnName": self.column_name,
            "type": self.type,
            "value": value,
        }
        if self.column_id is not None:
            result["columnId"] = self.column_id

        return result

@dataclass
class DatasetRow:
    """
    This class represents a row of a dataset.
    """

    id: str
    data: dict[str, str]

    def __json__(self) -> dict[str, Any]:
        return {"id": self.id, "data": self.data}

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "data": self.data}

    @classmethod
    def dict_to_class(cls, data: dict[str, Any]) -> "DatasetRow":
        return cls(id=data["id"], data=data["data"])


InputColumn = Literal["INPUT"]
ExpectedOutputColumn = Literal["EXPECTED_OUTPUT"]
ContextToEvaluateColumn = Literal["CONTEXT_TO_EVALUATE"]
VariableColumn = Literal["VARIABLE"]
FileURLVariableColumn = Literal["FILE_URL_VARIABLE"]
NullableVariableColumn = Literal["NULLABLE_VARIABLE"]
OutputColumn = Literal["OUTPUT"]

DataStructure = dict[
    str,
    Union[
        InputColumn,
        ExpectedOutputColumn,
        ContextToEvaluateColumn,
        VariableColumn,
        FileURLVariableColumn,
        NullableVariableColumn,
    ],
]

T = TypeVar("T", bound=DataStructure)

DataValue = list[T]

LocalData = dict[str, Union[str, list[str], None]]
Data = Union[str, list[LocalData], LocalData, Callable[[int], Optional[LocalData]]]
