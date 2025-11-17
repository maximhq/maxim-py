from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict, Union

from typing_extensions import deprecated

from ...scribe import scribe
from ..parsers import validate_type
from ..writer import LogWriter
from .attachment import (
    FileAttachment,
    FileDataAttachment,
    UrlAttachment,
)
from .base import EventEmittingBaseContainer
from .error import Error, ErrorConfig
from .feedback import Feedback, FeedbackDict, get_feedback_dict
from .generation import (
    Generation,
    GenerationConfig,
    GenerationConfigDict,
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
from .types import Entity

if TYPE_CHECKING:
    from .span import Span, SpanConfig, SpanConfigDict  # Type checking only


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class TraceConfig:
    """Trace config.

    This class represents a trace config.
    """

    id: str
    name: Optional[str] = None
    session_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    input: Optional[str] = None
    start_timestamp: Optional[datetime] = None

class TraceConfigDict(TypedDict, total=False):
    """Trace config dict.

    This class represents a trace config dictionary.
    """

    id: str
    name: Optional[str]
    session_id: Optional[str]
    tags: Optional[Dict[str, str]]
    input: Optional[str]
    start_timestamp: Optional[datetime]


def get_trace_config_dict(
    config: Union[TraceConfig, TraceConfigDict],
) -> TraceConfigDict:
    """
    Convert a TraceConfig object to a TraceConfigDict.

    Args:
        config: Either a TraceConfig object or a TraceConfigDict dictionary.

    Returns:
        A TraceConfigDict dictionary representation of the config.
    """
    return (
        TraceConfigDict(
            id=config.id,
            name=config.name,
            session_id=config.session_id,
            tags=config.tags,
            input=config.input,
            start_timestamp=config.start_timestamp,
        )
        if isinstance(config, TraceConfig)
        else config
    )


class Trace(EventEmittingBaseContainer):
    """
    A class representing a trace in the logging system.

    A trace is a high-level container for tracking a complete operation or workflow.
    """

    def __init__(self, config: Union[TraceConfig, TraceConfigDict], writer: LogWriter):
        """
        Initialize a new Trace instance.

        Args:
            config: Configuration for the trace, either as a TraceConfig object or a TraceConfigDict.
            writer: The LogWriter instance to use for writing log entries.
        """
        self.output = None
        final_config = get_trace_config_dict(config)
        super().__init__(Entity.TRACE, dict(final_config), writer)
        payload_to_send = {
            **self.data(),
            "sessionId": final_config.get("session_id", None),
        }
        if input_to_send := final_config.get("input", None):
            payload_to_send["input"] = input_to_send
        self._commit("create", payload_to_send)

    def set_input(self, input: str):
        """
        Set the input for this trace.

        Args:
            input: The input string to set.
        """
        try:
            validate_type(input, str, "input")
        except ValueError:
            scribe().error("[MaximSDK] Input must be of type string")
            return
        self._commit("update", {"input": input})

    @staticmethod
    def set_input_(writer: LogWriter, trace_id: str, input: str):
        """
        Static method to set the input for a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to update.
            input: The input string to set.
        """
        try:
            validate_type(input, str, "input")
        except ValueError:
            scribe().error("[MaximSDK] Input must be of type string")
            return
        Trace._commit_(writer, Entity.TRACE, trace_id, "update", {"input": input})

    def add_metric(self, name: str, value: float) -> None:
        """
        Add a metric to this trace.
        """
        self._commit("update", {"metrics": {"name": name, "value": value}})

    @staticmethod
    def add_metric_(writer: LogWriter, trace_id: str, name: str, value: float):
        """
        Static method to add a metric to a trace.
        """
        Trace._commit_(
            writer,
            Entity.TRACE,
            trace_id,
            "update",
            {"metrics": {"name": name, "value": value}},
        )

    def set_output(self, output: str):
        """
        Set the output for this trace.

        Args:
            output: The output string to set.
        """
        try:
            validate_type(output, str, "output")
        except ValueError:
            scribe().error("[MaximSDK] Output must be of type string")
            return
        self.output = output
        self._commit("update", {"output": output})

    @staticmethod
    def set_output_(writer: LogWriter, trace_id: str, output: str):
        """
        Static method to set the output for a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to update.
            output: The output string to set.
        """
        try:
            validate_type(output, str, "output")
        except ValueError:
            scribe().error("[MaximSDK] Output must be of type string")
            return
        Trace._commit_(writer, Entity.TRACE, trace_id, "update", {"output": output})

    def generation(
        self, config: Union[GenerationConfig, GenerationConfigDict]
    ) -> Generation:
        """
        Add a generation to this trace.

        Args:
            config: Configuration for the generation.

        Returns:
            A new Generation instance.
        """
        generation = Generation(config, self.writer)
        self._commit(
            "add-generation",
            {
                **generation.data(),
                "id": generation.id,
            },
        )
        return generation

    def tool_call(self, config: Union[ToolCallConfig, ToolCallConfigDict]) -> ToolCall:
        """
        Add a tool call to this trace.

        Args:
            config: Configuration for the tool call.

        Returns:
            A new ToolCall instance.
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
        trace_id: str,
        config: Union[ToolCallConfig, ToolCallConfigDict],
    ) -> ToolCall:
        """
        Static method to add a tool call to a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to add the tool call to.
            config: Configuration for the tool call.

        Returns:
            A new ToolCall instance.
        """
        final_config = get_tool_call_config_dict(config)
        tool_call = ToolCall(final_config, writer)
        Trace._commit_(
            writer,
            Entity.TRACE,
            trace_id,
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
        trace_id: str,
        config: Union[GenerationConfig, GenerationConfigDict],
    ) -> Generation:
        """
        Static method to add a generation to a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to add the generation to.
            config: Configuration for the generation.

        Returns:
            A new Generation instance.
        """
        generation = Generation(config, writer)
        Trace._commit_(
            writer,
            Entity.TRACE,
            trace_id,
            "add-generation",
            {
                **generation.data(),
                "id": generation.id,
            },
        )
        return generation

    def add_error(self, config: ErrorConfig) -> Error:
        """
        Add an error to this trace.

        Args:
            config: Configuration for the error.

        Returns:
            A new Error instance.
        """
        error = Error(config, self.writer)
        self._commit("add-error", error.data())
        return error

    @staticmethod
    def error_(writer: LogWriter, trace_id: str, config: ErrorConfig) -> Error:
        """
        Static method to add an error to a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to add the error to.
            config: Configuration for the error.

        Returns:
            A new Error instance.
        """
        error = Error(config, writer)
        Trace._commit_(
            writer,
            Entity.TRACE,
            trace_id,
            "add-error",
            error.data(),
        )
        return error

    def retrieval(self, config: Union[RetrievalConfig, RetrievalConfigDict]):
        """
        Add a retrieval to this trace.

        Args:
            config: Configuration for the retrieval.

        Returns:
            A new Retrieval instance.
        """
        final_config = get_retrieval_config_dict(config)
        retrieval = Retrieval(config, self.writer)
        self._commit(
            "add-retrieval",
            {
                "id": final_config.get("id"),
                **retrieval.data(),
            },
        )
        return retrieval

    @staticmethod
    def retrieval_(
        writer: LogWriter,
        trace_id: str,
        config: Union[RetrievalConfig, RetrievalConfigDict],
    ):
        """
        Static method to add a retrieval to a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to add the retrieval to.
            config: Configuration for the retrieval.

        Returns:
            A new Retrieval instance.
        """
        final_config = get_retrieval_config_dict(config)
        retrieval = Retrieval(config, writer)
        Trace._commit_(
            writer,
            Entity.TRACE,
            trace_id,
            "add-retrieval",
            {
                "id": final_config.get("id"),
                **retrieval.data(),
            },
        )
        return retrieval

    def span(self, config: Union["SpanConfig", "SpanConfigDict"]) -> "Span":
        """
        Add a span to this trace.

        Args:
            config: Configuration for the span.

        Returns:
            A new Span instance.
        """
        from .span import Span, get_span_config_dict

        final_config = get_span_config_dict(config)
        span = Span(config, self.writer)
        self._commit(
            "add-span",
            {
                "id": final_config.get("id"),
                **span.data(),
            },
        )
        return span

    @staticmethod
    def span_(
        writer: LogWriter,
        trace_id: str,
        config: Union["SpanConfig", "SpanConfigDict"],
    ) -> "Span":
        """
        Static method to add a span to a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to add the span to.
            config: Configuration for the span.

        Returns:
            A new Span instance.
        """
        from .span import Span, get_span_config_dict

        final_config = get_span_config_dict(config)
        span = Span(config, writer)
        Trace._commit_(
            writer,
            Entity.TRACE,
            trace_id,
            "add-span",
            {
                "id": final_config.get("id"),
                **span.data(),
            },
        )

        return span

    def feedback(self, feedback: Union[Feedback, FeedbackDict]):
        """
        Add feedback to this trace.

        Args:
            feedback: The feedback to add.
        """
        self._commit("add-feedback", dict(get_feedback_dict(feedback)))

    @staticmethod
    def feedback_(
        writer: LogWriter, trace_id: str, feedback: Union[Feedback, FeedbackDict]
    ):
        """
        Static method to add feedback to a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to add the feedback to.
            feedback: The feedback to add.
        """
        Trace._commit_(
            writer,
            Entity.TRACE,
            trace_id,
            "add-feedback",
            dict(get_feedback_dict(feedback)),
        )

    @staticmethod
    def add_tag_(writer: LogWriter, id: str, key: str, value: str):
        """
        Static method to add a tag to a trace.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the trace to add the tag to.
            key: The tag key.
            value: The tag value.
        """
        EventEmittingBaseContainer._add_tag_(writer, Entity.TRACE, id, key, value)

    def add_attachment(
        self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]
    ):
        """
        Add an attachment to this trace.

        Args:
            attachment: The attachment to add.
        """
        self._commit("upload-attachment", attachment.to_dict())

    @staticmethod
    def add_attachment_(
        writer: LogWriter,
        trace_id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Static method to add an attachment to a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to add the attachment to.
            attachment: The attachment to add.
        """
        Trace._commit_(
            writer,
            Entity.TRACE,
            trace_id,
            "upload-attachment",
            attachment.to_dict(),
        )

    @staticmethod
    def end_(writer: LogWriter, trace_id: str, data: Optional[Dict[str, Any]] = None):
        """
        Static method to end a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to end.
            data: Additional data to include in the end event.

        Returns:
            The result of the end operation.
        """
        if data is None:
            data = {}
        # Only set endTimestamp if it's not already provided in data
        if "endTimestamp" not in data:
            data["endTimestamp"] = datetime.now(timezone.utc)
        return EventEmittingBaseContainer._end_(
            writer,
            Entity.TRACE,
            trace_id,
            data,
        )

    @staticmethod
    def event_(
        writer: LogWriter,
        trace_id: str,
        id: str,
        event: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Static method to add a custom event to a trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to add the event to.
            id: The ID of the event.
            event: The name of the event.
            tags: Optional tags to associate with the event.
            metadata: Optional metadata to include with the event.

        Returns:
            The result of the event operation.
        """
        return EventEmittingBaseContainer._event_(
            writer, Entity.TRACE, trace_id, id, event, tags, metadata
        )

    @staticmethod
    def set_start_timestamp_(writer: LogWriter, trace_id: str, timestamp: datetime):
        """
        Set the start timestamp for this trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to set the start timestamp for.
            timestamp: The start timestamp.
        """
        return EventEmittingBaseContainer._set_start_timestamp_(writer, Entity.TRACE, trace_id, timestamp)

    @staticmethod
    def set_end_timestamp_(writer: LogWriter, trace_id: str, timestamp: datetime):
        """
        Set the end timestamp for this trace.

        Args:
            writer: The LogWriter instance to use.
            trace_id: The ID of the trace to set the end timestamp for.
            timestamp: The end timestamp.
        """
        return EventEmittingBaseContainer._set_end_timestamp_(writer, Entity.TRACE, trace_id, timestamp)

    def data(self) -> Dict[str, Any]:
        """
        Get the data representation of this trace.

        Returns:
            A dictionary containing the trace data.
        """
        return {
            **super().data(),
            "output": self.output,
        }
