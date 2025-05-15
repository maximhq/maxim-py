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
    id: str
    name: str
    description: str
    args: str
    tags: Optional[Dict[str, str]] = None


class ToolCallConfigDict(TypedDict, total=False):
    id: str
    name: str
    description: str
    args: str
    tags: Optional[Dict[str, str]]


def get_tool_call_config_dict(
    config: Union[ToolCallConfig, ToolCallConfigDict],
) -> ToolCallConfigDict:
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
    def __init__(
        self, config: Union[ToolCallConfig, ToolCallConfigDict], writer: LogWriter
    ):
        final_config = get_tool_call_config_dict(config)
        if "id" not in final_config:
            raise ValueError("ID is required")
        super().__init__(Entity.TOOL_CALL, dict(final_config), writer)
        self._id = final_config.get("id")
        self._name = final_config.get("name", None)
        self.args = final_config.get("args", None)
        self.description = final_config.get("description", None)
        self.tags = final_config.get("tags", None)

    def update(self, data: Dict[str, Any]):
        self._commit("update", data)

    @staticmethod
    def update_(writer: LogWriter, id: str, data: Dict[str, Any]):
        BaseContainer._commit_(writer, Entity.TOOL_CALL, id, "update", data)

    @staticmethod
    def result_(writer: LogWriter, id: str, result: str):
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
        raise NotImplementedError("attach_evaluators is not supported for ToolCall")

    def with_variables(self, for_evaluators: List[str], variables: Dict[str, str]):
        raise NotImplementedError("with_variables is not supported for ToolCall")

    def result(self, result: str):
        self._commit("result", {"result": result})
        self.end()

    def error(self, error: ToolCallError):
        self._commit("error", {"error": error})
        self.end()

    @staticmethod
    def error_(
        writer: LogWriter, id: str, error: Union[ToolCallError, ToolCallErrorDict]
    ):
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
        base_data = super().data()
        return {
            **base_data,
            "name": self._name,
            "description": self.description,
            "args": self.args,
        }
