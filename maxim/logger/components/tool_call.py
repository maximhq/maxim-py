from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, Union

from typing_extensions import deprecated

from ..writer import LogWriter
from .base import BaseContainer
from .types import Entity


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class ToolCallConfig:
    """Tool call config.

    This class represents a tool call config.
    """

    id: str
    name: str
    description: str
    args: str
    tags: Optional[Dict[str, str]] = None


class ToolCallConfigDict(TypedDict, total=False):
    """Tool call config dict.

    This class represents a tool call config dictionary.
    """

    id: str
    name: str
    description: str
    args: str
    tags: Optional[Dict[str, str]]
    start_timestamp: Optional[datetime]


def get_tool_call_config_dict(
    config: Union[ToolCallConfig, ToolCallConfigDict],
) -> ToolCallConfigDict:
    """Convert a tool call config to a tool call config dict else return the config.

    Args:
        config: The config to convert.

    Returns:
        ToolCallConfigDict: The tool call config dict.
    """
    return (
        ToolCallConfigDict(
            id=config.id,
            name=config.name,
            description=config.description,
            args=config.args,
            tags=config.tags,            
        )
        if isinstance(config, ToolCallConfig)
        else config
    )


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict instead."
)
@dataclass
class ToolCallError:
    """Tool call error.

    This class represents a tool call error.
    """

    message: str
    code: Optional[str] = None
    type: Optional[str] = None


class ToolCallErrorDict(TypedDict):
    message: str
    code: Optional[str]
    type: Optional[str]


def get_tool_call_error_dict(
    error: Union[ToolCallError, ToolCallErrorDict],
) -> dict[str, Any]:
    """Convert a tool call error to a tool call error dict else return the error.

    Args:
        error: The error to convert.

    Returns:
        dict[str, Any]: The tool call error dict.
    """
    return dict(
        ToolCallErrorDict(
            message=error.message,
            code=error.code,
            type=error.type,
        )
        if isinstance(error, ToolCallError)
        else dict(error)
    )


class ToolCall(BaseContainer):
    """Tool call.

    This class represents a tool call.
    """

    def __init__(
        self, config: Union[ToolCallConfig, ToolCallConfigDict], writer: LogWriter
    ):
        """
        Initialize a tool call.

        Args:
            config: The config to initialize the tool call with.
            writer: The writer to use.
        """
        final_config = get_tool_call_config_dict(config)
        if "id" not in final_config:
            raise ValueError("ID is required")
        super().__init__(Entity.TOOL_CALL, dict(final_config), writer)
        self._id = final_config.get("id")
        self._name = final_config.get("name", None)
        self.args = final_config.get("args", None)
        self.description = final_config.get("description", None)
        self.tags = final_config.get("tags", None)

    def add_metric(self, name: str, value: float) -> None:
        """
        Add a metric to this tool call.
        """
        self._commit("update", {"metrics": {"name": name, "value": value}})

    @staticmethod
    def add_metric_(writer: LogWriter, id: str, name: str, value: float):
        """
        Add a metric to the tool call.
        """
        BaseContainer._commit_(
            writer,
            Entity.TOOL_CALL,
            id,
            "update",
            {"metrics": {"name": name, "value": value}},
        )

    def update(self, data: Dict[str, Any]):
        """
        Update the tool call.

        Args:
            data: The data to update the tool call with.
        """
        self._commit("update", data)

    @staticmethod
    def update_(writer: LogWriter, id: str, data: Dict[str, Any]):
        """
        Update the tool call.

        Args:
            writer: The writer to use.
            id: The id of the tool call to update.
            data: The data to update the tool call with.
        """
        BaseContainer._commit_(writer, Entity.TOOL_CALL, id, "update", data)

    @staticmethod
    def result_(writer: LogWriter, id: str, result: str):
        """
        Update the tool call.

        Args:
            writer: The writer to use.
            id: The id of the tool call to update.
            result: The result to update the tool call with.
        """
        BaseContainer._commit_(
            writer, Entity.TOOL_CALL, id, "result", {"result": result}
        )
        BaseContainer._end_(
            writer,
            Entity.TOOL_CALL,
            id,
            {
                "endTimestamp": datetime.now(timezone.utc),
            },
        )

    def attach_evaluators(self, evaluators: List[str]):
        """
        Attach evaluators to the tool call.

        Args:
            evaluators: The evaluators to attach to the tool call.
        """
        raise NotImplementedError("attach_evaluators is not supported for ToolCall")

    def with_variables(self, for_evaluators: List[str], variables: Dict[str, str]):
        raise NotImplementedError("with_variables is not supported for ToolCall")

    def result(self, result: str):
        """
        Update the tool call.

        Args:
            result: The result to update the tool call with.
        """
        self._commit("result", {"result": result})
        self.end()

    def error(self, error: ToolCallError):
        """
        Add an error to the tool call.

        Args:
            error: The tool call error.
        """
        self._commit("error", {"error": error})
        self.end()

    @staticmethod
    def error_(
        writer: LogWriter, id: str, error: Union[ToolCallError, ToolCallErrorDict]
    ):
        """
        Add an error to the tool call.

        Args:
            writer: The writer to use.
            id: The id of the tool call to add the error to.
            error: The tool call error.
        """
        final_error = get_tool_call_error_dict(error)
        BaseContainer._commit_(
            writer, Entity.TOOL_CALL, id, "error", {"error": final_error}
        )
        BaseContainer._end_(
            writer,
            Entity.TOOL_CALL,
            id,
            {
                "endTimestamp": datetime.now(timezone.utc),
            },
        )

    def data(self) -> Dict[str, Any]:
        """
        Get the data for the tool call.

        Returns:
            Dict[str, Any]: The data for the tool call.
        """
        base_data = super().data()
        return {
            **base_data,
            "name": self._name,
            "description": self.description,
            "args": self.args,
        }
