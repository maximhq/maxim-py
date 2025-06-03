from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, Union

from typing_extensions import deprecated

from ..components.attachment import FileAttachment, FileDataAttachment, UrlAttachment
from ..writer import LogWriter
from .base import EventEmittingBaseContainer
from .error import Error, ErrorConfig
from .generation import (
    Generation,
    GenerationConfig,
    GenerationConfigDict,
    get_generation_config_dict,
)
from .retrieval import (
    Retrieval,
    RetrievalConfig,
    RetrievalConfigDict,
    get_retrieval_config_dict,
)
from .tool_call import (
    ToolCall,
    ToolCallConfig,
    ToolCallConfigDict,
    get_tool_call_config_dict,
)
from .trace import Trace
from .types import Entity


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class SpanConfig:
    id: str
    name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class SpanConfigDict(TypedDict, total=False):
    id: str
    name: Optional[str]
    tags: Optional[Dict[str, str]]


def get_span_config_dict(config: Union[SpanConfig, SpanConfigDict]) -> dict[str, Any]:
    return (
        dict(
            SpanConfigDict(
                id=config.id,
                name=config.name,
                tags=config.tags,
            )
        )
        if isinstance(config, SpanConfig)
        else dict(config)
    )


class Span(EventEmittingBaseContainer):
    ENTITY = Entity.SPAN

    def __init__(self, config: Union[SpanConfig, SpanConfigDict], writer: LogWriter):
        final_config = get_span_config_dict(config)
        super().__init__(self.ENTITY, final_config, writer)
        self.traces: List[Trace] = []

    def span(self, config: Union[SpanConfig, SpanConfigDict]):
        final_config = get_span_config_dict(config)
        span = Span(config, self.writer)
        span.span_id = self.id
        self._commit(
            "add-span",
            {
                "id": final_config["id"],
                **span.data(),
            },
        )
        return span

    def input(self, input: str):
        self._commit("update", {input: {"type": "text", "value": input}})

    @staticmethod
    def input_(writer: LogWriter, span_id: str, input: str):
        Span._commit_(
            writer,
            Entity.SPAN,
            span_id,
            "update",
            {"input": {"type": "text", "value": input}},
        )

    def add_error(self, config: ErrorConfig) -> Error:
        error = Error(config, self.writer)
        self._commit("add-error", error.data())
        return error

    @staticmethod
    def error_(writer: LogWriter, span_id: str, config: ErrorConfig) -> Error:
        error = Error(config, writer)
        Span._commit_(
            writer,
            Entity.SPAN,
            span_id,
            "add-error",
            error.data(),
        )
        return error

    @staticmethod
    def span_(
        writer: LogWriter, span_id: str, config: Union[SpanConfig, SpanConfigDict]
    ):
        final_config = get_span_config_dict(config)
        span = Span(config, writer)
        span.span_id = span_id
        Span._commit_(
            writer,
            Entity.SPAN,
            span_id,
            "add-span",
            {
                "id": final_config.get("id"),
                **span.data(),
            },
        )
        return span

    def generation(
        self, config: Union[GenerationConfig, GenerationConfigDict]
    ) -> Generation:
        final_config = get_generation_config_dict(config)
        generation = Generation(config, self.writer)
        payload = generation.data()
        payload["id"] = final_config.get("id")
        payload["spanId"] = self.id
        self._commit(
            "add-generation",
            {
                **payload,
            },
        )
        return generation

    def tool_call(self, config: Union[ToolCallConfig, ToolCallConfigDict]) -> ToolCall:
        final_config = get_tool_call_config_dict(config)
        tool_call = ToolCall(final_config, self.writer)
        self._commit(
            "add-tool-call",
            {
                **tool_call.data(),
                "id": tool_call.id,
            },
        )
        return tool_call

    @staticmethod
    def tool_call_(
        writer: LogWriter,
        span_id: str,
        config: Union[ToolCallConfig, ToolCallConfigDict],
    ) -> ToolCall:
        final_config = get_tool_call_config_dict(config)
        tool_call = ToolCall(final_config, writer)
        Span._commit_(
            writer,
            Entity.SPAN,
            span_id,
            "add-tool-call",
            {
                **tool_call.data(),
                "id": tool_call.id,
            },
        )
        return tool_call

    @staticmethod
    def generation_(
        writer: LogWriter,
        span_id: str,
        config: Union[GenerationConfig, GenerationConfigDict],
    ) -> Generation:
        final_config = get_generation_config_dict(config)
        generation = Generation(config, writer)
        Span._commit_(
            writer,
            Entity.SPAN,
            span_id,
            "add-generation",
            {
                **generation.data(),
                "id": final_config["id"],
            },
        )
        return generation

    def add_attachment(
        self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]
    ):
        """
        Add an attachment to this span.

        Args:
            attachment: The attachment to add.
        """
        self._commit("upload-attachment", attachment.to_dict())

    @staticmethod
    def add_attachment_(
        writer: LogWriter,
        span_id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Static method to add an attachment to a span.

        Args:
            writer: The LogWriter instance to use.
            span_id: The ID of the span to add the attachment to.
            attachment: The attachment to add.
        """
        Span._commit_(
            writer,
            Entity.SPAN,
            span_id,
            "upload-attachment",
            attachment.to_dict(),
        )

    def retrieval(self, config: Union[RetrievalConfig, RetrievalConfigDict]):
        final_config = get_retrieval_config_dict(config)
        retrieval = Retrieval(config, self.writer)
        self._commit(
            "add-retrieval",
            {
                "id": final_config["id"],
                **retrieval.data(),
            },
        )
        return retrieval

    @staticmethod
    def retrieval_(
        writer: LogWriter,
        span_id: str,
        config: Union[RetrievalConfig, RetrievalConfigDict],
    ):
        retrieval = Retrieval(config, writer)
        Span._commit_(
            writer,
            Entity.SPAN,
            span_id,
            "add-retrieval",
            {
                "id": retrieval.id,
                **retrieval.data(),
            },
        )
        return retrieval

    @staticmethod
    def end_(writer: LogWriter, span_id: str, data: Optional[Dict[str, str]] = None):
        if data is None:
            data = {}
        return EventEmittingBaseContainer._end_(
            writer,
            Entity.SPAN,
            span_id,
            {"endTimestamp": datetime.now(timezone.utc), **data},
        )

    @staticmethod
    def add_tag_(writer: LogWriter, span_id: str, key: str, value: str):
        return EventEmittingBaseContainer._add_tag_(
            writer, Entity.SPAN, span_id, key, value
        )

    @staticmethod
    def event_(
        writer: LogWriter,
        span_id: str,
        id: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        return EventEmittingBaseContainer._event_(
            writer, Entity.SPAN, span_id, id, name, tags, metadata
        )
