from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union
from ..logger.components.attachment import Attachment, FileAttachment, FileDataAttachment, UrlAttachment

class VariableType(str):
    """
    This class represents the type of a variable.
    """

    TEXT = "text"
    JSON = "json"
    FILE = "file"


@dataclass
class Variable:
    """
    This class represents a variable.
    """
    type: Literal["text", "json", "file"]
    payload: Union[str, Dict[str, Any], List[Attachment]]
    
    def to_json(self) -> Dict[str, Any]:
        """Convert the Variable to a JSON-serializable dictionary."""
        return {"type": self.type, "payload": self.payload}
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Variable":
        """Create a Variable from a JSON dictionary.
        
        Args:
            data: Dictionary containing the variable data with 'type' and 'payload' keys
            
        Returns:
            Variable: The created variable instance
            
        Raises:
            ValueError: If the data format is invalid or required fields are missing
        """
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if "type" not in data:
            raise ValueError("Required 'type' field is missing")
        
        if "payload" not in data:
            raise ValueError("Required 'payload' field is missing")
        
        var_type = data["type"]
        payload = data["payload"]
        
        # Validate type
        if var_type not in ["text", "json", "file"]:
            raise ValueError(f"Invalid variable type: {var_type}. Must be one of 'text', 'json', 'file'")
        
        # Validate payload based on type
        if var_type == "text":
            if not isinstance(payload, str):
                raise ValueError("Payload for 'text' type must be a string")
        elif var_type == "json":
            if not isinstance(payload, dict):
                raise ValueError("Payload for 'json' type must be a dictionary")
        elif var_type == "file":
            if not isinstance(payload, list):
                raise ValueError("Payload for 'file' type must be a list of attachments")
            # Note: We could add more validation for attachment objects here if needed
        
        return cls(type=var_type, payload=payload)

@dataclass
class VariableFileAttachment:
    """
    This class represents a variable file attachment.
    """
    id: str
    url: str
    hosted: bool
    prefix: Optional[str]
    props: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "url": self.url if not self.hosted else "",
            "hosted": self.hosted,
            "prefix": self.prefix,
            "props": self.props,
        }

@dataclass
class FileVariablePayload:
    """
    This class represents a file variable payload.
    """
    text: Optional[str]
    files: List[VariableFileAttachment]
    entryId: Optional[str] = None

@dataclass
class DatasetEntry:
    """
    This class represents a dataset entry.
    """
    entry: Dict[str, Variable]
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "DatasetEntry":
        """
        Convert a single dictionary to a DatasetEntry object.
        
        Args:
            data_dict: Dictionary representing a dataset entry
            
        Returns:
            DatasetEntry: DatasetEntry object
            
        Raises:
            ValueError: If the data format is invalid
        """
        if not isinstance(data_dict, dict):
            raise ValueError("Input must be a dictionary")
        
        # Convert the entry dictionary to Variables
        variables = {}
        for column_name, value in data_dict.items():
            # Determine the type based on the value
            if isinstance(value, str):
                var_type = "text"
                payload = value
            elif isinstance(value, dict):
                var_type = "json"
                payload = value
            elif isinstance(value, list) and all(isinstance(item, (FileAttachment, FileDataAttachment, UrlAttachment)) for item in value):
                # Check for List[Attachment]
                var_type = "file"
                payload = value
            else:
                # Default to text for unknown types
                var_type = "text"
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
    payload: Union[str, Dict[str, Any], List[Attachment]]

    @classmethod
    def from_dataset_entry(cls, dataset_entry: DatasetEntry, row_no: int) -> List["DatasetEntryWithRowNo"]:
        """
        Convert a DatasetEntry to a list of DatasetEntryWithRowNo objects.
        
        Args:
            dataset_entry: The DatasetEntry to convert
            row_no: The row number to assign
            
        Returns:
            List[DatasetEntryWithRowNo]: List of DatasetEntryWithRowNo objects, one for each column
            
        Raises:
            ValueError: If the dataset entry is invalid
        """
        if not isinstance(dataset_entry, DatasetEntry):
            raise ValueError("Input must be a DatasetEntry object")
        
        result = []
        for column_name, variable in dataset_entry.entry.items():
            result.append(cls(
                row_no=row_no,
                column_name=column_name,
                type=variable.type,
                payload=variable.payload
            ))
        
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DatasetEntryWithRowNo to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation with rowNo, columnName, type, and value
        """
        return {
            "rowNo": self.row_no,
            "columnName": self.column_name,
            "type": self.type,
            "value": [] if self.type == "file" else self.payload
        }

@dataclass
class DatasetRow:
    """
    This class represents a row of a dataset.
    """

    id: str
    data: Dict[str, str]

    def __json__(self):
        return {"id": self.id, "data": self.data}

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "data": self.data}

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "DatasetRow":
        return cls(id=data["id"], data=data["data"])


InputColumn = Literal["INPUT"]
ExpectedOutputColumn = Literal["EXPECTED_OUTPUT"]
ContextToEvaluateColumn = Literal["CONTEXT_TO_EVALUATE"]
VariableColumn = Literal["VARIABLE"]
FileURLVariableColumn = Literal["FILE_URL_VARIABLE"]
NullableVariableColumn = Literal["NULLABLE_VARIABLE"]
OutputColumn = Literal["OUTPUT"]

DataStructure = Dict[
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

LocalData = Dict[str, Union[str, List[str], None]]
Data = Union[str, List[LocalData], LocalData, Callable[[int], Optional[LocalData]]]
