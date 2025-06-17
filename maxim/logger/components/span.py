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
    """Span config.

    This class represents a span config.
    """

    id: str
    name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class SpanConfigDict(TypedDict, total=False):
    """Span config dict.

    This class represents a span config dictionary.
    """

    id: str
    name: Optional[str]
    tags: Optional[Dict[str, str]]


def get_span_config_dict(config: Union[SpanConfig, SpanConfigDict]) -> dict[str, Any]:
    """Convert a span config to a span config dict else return the config.

    Args:
        config: The config to convert.

    Returns:
        dict[str, Any]: The span config dict.
    """
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
    """Span.

    This class represents a span.
    """

    ENTITY = Entity.SPAN

    def __init__(self, config: Union[SpanConfig, SpanConfigDict], writer: LogWriter):
        """
        Initialize a span.

        Args:
            config: The config to initialize the span with.
            writer: The writer to use.
        """
        final_config = get_span_config_dict(config)
        super().__init__(self.ENTITY, final_config, writer)
        self.traces: List[Trace] = []

    def span(self, config: Union[SpanConfig, SpanConfigDict]):
        """
        Add a span to this span.

        Args:
            config: The config to add the span to.
        """
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
        """
        Set the input for this span.

        Args:
            input: The input to set for this span.
        """
        self._commit("update", {"input": {"type": "text", "value": input}})

    @staticmethod
    def input_(writer: LogWriter, span_id: str, input: str):
        """
        Set the input for this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to set the input for.
            input: The input to set for this span.
        """
        Span._commit_(
            writer,
            Entity.SPAN,
            span_id,
            "update",
            {"input": {"type": "text", "value": input}},
        )

    def add_error(self, config: ErrorConfig) -> Error:
        """
        Add an error to this span.

        Args:
            config: The config to add the error to.

        Returns:
            Error: The error that was added.
        """
        error = Error(config, self.writer)
        self._commit("add-error", error.data())
        return error

    @staticmethod
    def error_(writer: LogWriter, span_id: str, config: ErrorConfig) -> Error:
        """
        Add an error to this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to add the error to.
            config: The config to add the error to.

        Returns:
            Error: The error that was added.
        """
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
        """
        Add a span to this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to add the span to.
            config: The config to add the span to.

        Returns:
            Span: The span that was added.
        """
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
        """
        Add a generation to this span.

        Args:
            config: The config to add the generation to.

        Returns:
            Generation: The generation that was added.
        """
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
        """
        Add a tool call to this span.

        Args:
            config: The config to add the tool call to.

        Returns:
            ToolCall: The tool call that was added.
        """
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
        """
        Add a tool call to this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to add the tool call to.
            config: The config to add the tool call to.

        Returns:
            ToolCall: The tool call that was added.
        """
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
        """
        Add a generation to this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to add the generation to.
            config: The config to add the generation to.

        Returns:
            Generation: The generation that was added.
        """
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
        """
        Add a retrieval to this span.

        Args:
            config: The config to add the retrieval to.

        Returns:
            Retrieval: The retrieval that was added.
        """
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
        """
        Add a retrieval to this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to add the retrieval to.
            config: The config to add the retrieval to.

        Returns:
            Retrieval: The retrieval that was added.
        """
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
        """
        End this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to end.
            data: The data to add to the span.
        """
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
        """
        Add a tag to this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to add the tag to.
            key: The key of the tag.
            value: The value of the tag.
        """
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
        """
        Add an event to this span.

        Args:
            writer: The writer to use.
            span_id: The id of the span to add the event to.
            id: The id of the event.
            name: The name of the event.
            tags: The tags of the event.
            metadata: The metadata of the event.
        """
        return EventEmittingBaseContainer._event_(
            writer, Entity.SPAN, span_id, id, name, tags, metadata
        )
