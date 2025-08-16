from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union


class VariableType(str):
    """
    This class represents the type of a variable.
    """

    TEXT = "text"
    JSON = "json"


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


class Variable:
    """
    This class represents a variable.
    """

    def __init__(
        self, type_: str, payload: Dict[str, Union[str, int, bool, float, List[str]]]
    ):
        """
        This class represents a variable.

        Args:
            type_: The type of the variable.
            payload: The payload of the variable.
        """

        self.type = type_
        self.payload = payload

    def to_json(self):
        return {"type": self.type, "payload": self.payload}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Variable":
        return cls(type_=data["type"], payload=data["payload"])


class DatasetEntry:
    """
    This class represents a dataset entry.
    """

    def __init__(
        self,
        input: Variable,
        context: Optional[Variable] = None,
        expectedOutput: Optional[Variable] = None,
    ):
        """
        This class represents a dataset entry.

        Args:
            input: The input variable.
            context: The context variable.
            expectedOutput: The expected output variable.
        """
        self.input = input
        self.context = context
        self.expectedOutput = expectedOutput

    def to_json(self):
        return_dict = {}
        if self.input is not None:
            return_dict["input"] = {
                "type": self.input.type,
                "payload": self.input.payload,
            }
        if self.context is not None:
            return_dict["context"] = {
                "type": self.context.type,
                "payload": self.context.payload,
            }
        if self.expectedOutput is not None:
            return_dict["expectedOutput"] = {
                "type": self.expectedOutput.type,
                "payload": self.expectedOutput.payload,
            }
        return return_dict


# Column name constants used throughout the dataset utilities.
INPUT_COLUMN: str = "INPUT"
EXPECTED_OUTPUT_COLUMN: str = "EXPECTED_OUTPUT"
CONTEXT_TO_EVALUATE_COLUMN: str = "CONTEXT_TO_EVALUATE"
VARIABLE_COLUMN: str = "VARIABLE"
NULLABLE_VARIABLE_COLUMN: str = "NULLABLE_VARIABLE"
OUTPUT_COLUMN: str = "OUTPUT"

# Type aliases for column names.  These are used for type checking but also
# need to be concrete values at runtime for equality comparisons.
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
