"""
Configuration class for Maxim Logger.

This class holds the configuration settings for the Logger, including
the logger ID, auto-flush setting, and flush interval.

Attributes:
    id (str): The unique identifier for the logger.
    auto_flush (bool): Whether to automatically flush logs. Defaults to True.
    flush_interval (int): The interval (in seconds) at which to flush logs when auto_flush is True. Defaults to 10 seconds.
"""

import json
import threading
from typing import Any, Dict, Optional, TypedDict, Union

from typing_extensions import deprecated

from ..logger.components.types import Entity
from ..scribe import scribe
from .components import (
    Error,
    ErrorConfig,
    Feedback,
    FeedbackDict,
    FileAttachment,
    FileDataAttachment,
    Generation,
    GenerationConfig,
    GenerationConfigDict,
    GenerationError,
    GenerationRequestMessage,
    Retrieval,
    RetrievalConfig,
    RetrievalConfigDict,
    Session,
    SessionConfig,
    SessionConfigDict,
    Span,
    SpanConfig,
    SpanConfigDict,
    ToolCall,
    ToolCallConfig,
    ToolCallConfigDict,
    ToolCallError,
    ToolCallErrorDict,
    Trace,
    TraceConfig,
    TraceConfigDict,
    UrlAttachment,
)
from .components.base import EvaluateContainer
from .writer import LogWriter, LogWriterConfig


@deprecated(
    "This class will be removed in a future version. Use LoggerConfigDict instead."
)
class LoggerConfig:
    """
    Configuration class for Maxim Logger.

    This class holds the configuration settings for the Logger, including
    the logger ID, auto-flush setting, and flush interval.

    Attributes:
        id (str): The unique identifier for the logger.
        auto_flush (bool): Whether to automatically flush logs.
        flush_interval (int): The interval (in seconds) at which to flush logs when auto_flush is True.
    """

    def __init__(self, id: str, auto_flush=True, flush_interval=10) -> None:
        self.id = id
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval


class LoggerConfigDict(TypedDict, total=False):
    id: str
    auto_flush: bool
    flush_interval: int


def get_logger_config_dict(
    config: Union[LoggerConfig, LoggerConfigDict],
) -> LoggerConfigDict:
    if isinstance(config, LoggerConfig):
        return LoggerConfigDict(
            id=config.id,
            auto_flush=config.auto_flush,
            flush_interval=config.flush_interval,
        )
    else:
        return config


class Logger:
    """
    A class representing a logger for the Maxim SDK.

    This logger provides methods for creating sessions, traces, and various logging components
    such as spans, generations, retrievals, and tool calls. It uses a LogWriter to handle the
    actual logging operations.

    Attributes:
        _id (str): The unique identifier for this logger instance.
        raise_exceptions (bool): Whether to raise exceptions during logging operations.
        is_debug (bool): Whether debug logging is enabled.
        writer (LogWriter): The LogWriter instance used for actual logging operations.
    """

    def __init__(
        self,
        config: LoggerConfigDict,
        api_key: str,
        base_url: str,
        is_debug=False,
        raise_exceptions=False,
    ) -> None:
        """
        Initializes the logger with the given configuration.

        Args:
            config (LoggerConfig): The configuration for the logger.
            api_key (str): The API key for the logger.
            base_url (str): The base URL for the logger.
            is_debug (bool, optional): Whether to enable debug logging. Defaults to False.
            raise_exceptions (bool, optional): Whether to raise exceptions. Defaults to False.
        """
        repo_id = config.get("id", None)
        if repo_id is None:
            raise ValueError("Logger must be initialized with id of the logger")
        self._id = repo_id
        self.raise_exceptions = raise_exceptions
        self.is_debug = is_debug
        self.config = config
        writer_config = LogWriterConfig(
            auto_flush="auto_flush" in config and config["auto_flush"] or True,
            flush_interval="flush_interval" in config
            and config["flush_interval"]
            or 10,
            base_url=base_url,
            api_key=api_key,
            is_debug=is_debug,
            repository_id=repo_id,
            raise_exceptions=raise_exceptions,
        )
        self.writer = LogWriter(writer_config)
        self.sinks: list[LogWriter] = []
        scribe().debug("[MaximSDK] Logger initialized")

    def add_sink(
        self,
        base_url: str,
        api_key: str,
        repo_id: str,
        is_debug: bool = False,
        raise_exceptions: bool = False,
    ) -> None:
        """
        Adds a sink to the logger. This will allow you to write logs to multiple repositories at the same time.

        Args:
            base_url (str): The base URL for the sink.
            api_key (str): The API key for the sink.
            repo_id (str): The repository ID for the sink.
            is_debug (bool): Whether to enable debug logging for the sink.
            raise_exceptions (bool): Whether to raise exceptions for the sink.
        """
        sink_config = LogWriterConfig(
            auto_flush=("auto_flush" in self.config and self.config["auto_flush"])
            or True,
            flush_interval=(
                "flush_interval" in self.config and self.config["flush_interval"]
            )
            or 10,
            base_url=base_url,
            api_key=api_key,
            is_debug=is_debug,
            repository_id=repo_id,
            raise_exceptions=raise_exceptions,
        )
        self.writer.add_sink(sink_config)

    def session(self, config: Union[SessionConfig, SessionConfigDict]) -> Session:
        """
        Creates a new session with the given configuration.

        Args:
            config (SessionConfig): The configuration for the new session.

        Returns:
            Session: The newly created session.
        """
        return Session(config, self.writer)

    def trace(self, config: Union[TraceConfig, TraceConfigDict]) -> Trace:
        """
        Creates a new trace with the given configuration.

        Args:
            config (TraceConfig): The configuration for the new trace.

        Returns:
            Trace: The newly created trace.
        """
        return Trace(config, self.writer)

    # Session methods
    def session_add_tag(self, session_id: str, key: str, value: str):
        """
        Adds a tag to the session.

        Args:
            session_id (str): The ID of the session.
            key (str): The key of the tag.
            value (str): The value of the tag.
        """
        Session.add_tag_(self.writer, session_id, key, value)

    def session_end(self, session_id: str):
        """
        Ends the session.

        Args:
            session_id (str): The ID of the session.
        """
        Session.end_(self.writer, session_id)

    def session_event(self, session_id: str, event_id: str, event: str, data: Any):
        """
        Adds an event to the session.

        Args:
            session_id (str): The ID of the session.
            event_id (str): The ID of the event.
            event (str): The name of the event.
            data (Any): The data associated with the event.
        """
        Session.event_(self.writer, session_id, event_id, event, data)

    @deprecated(
        "This method will be removed in a future version. Use session_add_feedback instead."
    )
    def session_feedback(
        self, session_id: str, feedback: Union[Feedback, FeedbackDict]
    ):
        """
        Adds a feedback to the session.

        Args:
            session_id (str): The ID of the session.
            feedback (Feedback): The feedback to add.
        """
        Session.feedback_(self.writer, session_id, feedback)

    def session_add_attachment(
        self,
        session_id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Adds an attachment to the session.
        """
        Session.add_attachment_(self.writer, session_id, attachment)

    def session_add_feedback(self, session_id: str, feedback: FeedbackDict):
        """
        Adds a feedback to the session.
        """
        Session.feedback_(self.writer, session_id, feedback)

    @deprecated(
        "This method will be removed in a future version. Use session_add_trace instead."
    )
    def session_trace(self, session_id: str, config: TraceConfig) -> Trace:
        """
        Adds a trace to the session.

        Args:
            session_id (str): The ID of the session.
            config (TraceConfig): The configuration for the trace.

        Returns:
            Trace: The newly created trace.
        """
        return Session.trace_(self.writer, session_id, config)

    def session_add_trace(
        self, session_id: str, config: Union[TraceConfig, TraceConfigDict]
    ) -> Trace:
        """
        Adds a trace to the session.

        Args:
            session_id (str): The ID of the session.
            config (TraceConfig): The configuration for the trace.

        Returns:
            Trace: The newly created trace.
        """
        return Session.trace_(self.writer, session_id, config)

    # Trace methods
    @deprecated(
        "This method will be removed in a future version. Use trace_add_generation instead."
    )
    def trace_generation(self, trace_id: str, config: GenerationConfig) -> Generation:
        """
        Adds a generation to the trace.

        Args:
            trace_id (str): The ID of the trace.
            config (GenerationConfig): The configuration for the generation.

        Returns:
            Generation: The newly created generation.
        """
        return Trace.generation_(self.writer, trace_id, config)

    def trace_add_generation(
        self, trace_id: str, config: Union[GenerationConfig, GenerationConfigDict]
    ) -> Generation:
        """
        Adds a generation to the trace.

        Args:
            trace_id (str): The ID of the trace.
            config (GenerationConfig): The configuration for the generation.

        Returns:
            Generation: The newly created generation.
        """
        return Trace.generation_(self.writer, trace_id, config)

    @deprecated(
        "This method will be removed in a future version. Use trace_add_retrieval instead."
    )
    def trace_retrieval(
        self, trace_id: str, config: Union[RetrievalConfig, RetrievalConfigDict]
    ) -> Retrieval:
        """
        Adds a retrieval to the trace.

        Args:
            trace_id (str): The ID of the trace.
            config (RetrievalConfig): The configuration for the retrieval.

        Returns:
            Retrieval: The newly created retrieval.
        """
        return Trace.retrieval_(self.writer, trace_id, config)

    def trace_add_retrieval(
        self, trace_id: str, config: Union[RetrievalConfig, RetrievalConfigDict]
    ) -> Retrieval:
        """
        Adds a retrieval to the trace.

        Args:
            trace_id (str): The ID of the trace.
            config (RetrievalConfig): The configuration for the retrieval.

        Returns:
            Retrieval: The newly created retrieval.
        """
        return Trace.retrieval_(self.writer, trace_id, config)

    @deprecated(
        "This method will be removed in a future version. Use trace_add_span instead."
    )
    def trace_span(
        self, trace_id: str, config: Union[SpanConfig, SpanConfigDict]
    ) -> Span:
        """
        Adds a span to the trace.

        Args:
            trace_id (str): The ID of the trace.
            config (SpanConfig): The configuration for the span.

        Returns:
            Span: The newly created span.
        """
        return Trace.span_(self.writer, trace_id, config)

    def trace_add_span(
        self, trace_id: str, config: Union[SpanConfig, SpanConfigDict]
    ) -> Span:
        """
        Adds a span to the trace.

        Args:
            trace_id (str): The ID of the trace.
            config (SpanConfig): The configuration for the span.

        Returns:
            Span: The newly created span.
        """
        return Trace.span_(self.writer, trace_id, config)

    def trace_add_error(self, trace_id: str, config: ErrorConfig) -> Error:
        """
        Adds an error to the trace.
        """
        return Trace.error_(self.writer, trace_id, config)

    def trace_add_tag(self, trace_id: str, key: str, value: str):
        """
        Adds a tag to the trace.

        Args:
            trace_id (str): The ID of the trace.
            key (str): The key of the tag.
            value (str): The value of the tag.
        """
        Trace.add_tag_(self.writer, trace_id, key, value)

    def trace_add_tool_call(
        self, trace_id: str, config: Union[ToolCallConfig, ToolCallConfigDict]
    ) -> ToolCall:
        """
        Adds a tool call to the trace.

        Args:
            trace_id (str): The ID of the trace.
            config (ToolCallConfig): The configuration for the tool call.

        Returns:
            ToolCall: The newly created tool call.
        """
        return Trace.tool_call_(self.writer, trace_id, config)

    def trace_evaluate(self, trace_id: str) -> EvaluateContainer:
        return Trace._evaluate_(self.writer, Entity.TRACE, trace_id)

    @deprecated(
        "This method will be removed in a future version. Use trace_add_event instead."
    )
    def trace_event(
        self,
        trace_id: str,
        event_id: str,
        event: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Adds an event to the trace.

        Args:
            trace_id (str): The ID of the trace.
            event_id (str): The ID of the event.
            event (str): The name of the event.
            tags (Optional[Dict[str, str]]): The tags associated with the event.
        """
        Trace.event_(self.writer, trace_id, event_id, event, tags, metadata)

    def trace_add_event(
        self,
        trace_id: str,
        event_id: str,
        event: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Adds an event to the trace.
        """
        Trace.event_(self.writer, trace_id, event_id, event, tags, metadata)

    def trace_set_input(self, trace_id: str, input: str):
        """
        Sets the input for the trace.

        Args:
            trace_id (str): The ID of the trace.
            input (str): The input for the trace.
        """
        Trace.set_input_(self.writer, trace_id, input)

    def trace_set_output(self, trace_id: str, output: str):
        """
        Sets the output for the trace.

        Args:
            trace_id (str): The ID of the trace.
            output (str): The output for the trace.
        """
        Trace.set_output_(self.writer, trace_id, output)

    @deprecated(
        "This method will be removed in a future version. Use trace_add_feedback instead."
    )
    def trace_feedback(self, trace_id: str, feedback: Feedback):
        """
        Adds a feedback to the trace.

        Args:
            trace_id (str): The ID of the trace.
            feedback (Feedback): The feedback to add.
        """
        Trace.feedback_(self.writer, trace_id, feedback)

    def trace_add_feedback(self, trace_id: str, feedback: FeedbackDict):
        """
        Adds a feedback to the trace.
        """
        Trace.feedback_(self.writer, trace_id, feedback)

    def trace_add_metadata(self, trace_id: str, metadata: Dict[str, Any]):
        """
        Adds metadata to the trace.

        Args:
            trace_id (str): The ID of the trace.
            metadata (Dict[str, Any]): The metadata to add.
        """
        Trace.add_metadata_(self.writer, Entity.TRACE, trace_id, metadata)

    def trace_add_attachment(
        self,
        trace_id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Adds an attachment to the trace.
        """
        Trace.add_attachment_(self.writer, trace_id, attachment)

    def trace_end(self, trace_id: str):
        """
        Ends the trace.

        Args:
            trace_id (str): The ID of the trace.
        """
        Trace.end_(self.writer, trace_id)

    # Generation methods
    def generation_set_model(self, generation_id: str, model: str):
        """
        Sets the model for the generation.

        Args:
            generation_id (str): The ID of the generation.
            model (str): The model for the generation.
        """
        Generation.set_model_(self.writer, generation_id, model)

    def generation_set_provider(self, generation_id: str, provider: str):
        """
        Sets the provider for the generation.
        """
        Generation.set_provider_(self.writer, generation_id, provider)

    def generation_add_message(
        self, generation_id: str, message: GenerationRequestMessage
    ):
        """
        Adds a message to the generation.

        Args:
            generation_id (str): The ID of the generation.
            message (Any): The OpenAI chat message to add.
        """
        Generation.add_message_(self.writer, generation_id, message)

    def generation_set_model_parameters(
        self, generation_id: str, model_parameters: Dict[str, Any]
    ):
        """
        Sets the model parameters for the generation.

        Args:
            generation_id (str): The ID of the generation.
            model_parameters (dict): The model parameters for the generation.
        """
        Generation.set_model_parameters_(self.writer, generation_id, model_parameters)

    def generation_result(self, generation_id: str, result: Any):
        """
        Sets the result for the generation.

        Args:
            generation_id (str): The ID of the generation.
            result (Any): The result for the generation.
        """
        Generation.result_(self.writer, generation_id, result)

    def generation_add_attachment(
        self,
        generation_id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Adds an attachment to the generation.
        """
        Generation.add_attachment_(self.writer, generation_id, attachment)

    def generation_end(self, generation_id: str):
        """
        Ends the generation.

        Args:
            generation_id (str): The ID of the generation.
        """
        Generation.end_(self.writer, generation_id)

    def generation_error(self, generation_id: str, error: GenerationError):
        """
        Sets the error for the generation.

        Args:
            generation_id (str): The ID of the generation.
            error (GenerationError): The error for the generation.
        """
        Generation.error_(self.writer, generation_id, error)

    def generation_add_metadata(self, generation_id: str, metadata: Dict[str, Any]):
        Generation.add_metadata_(
            self.writer, Entity.GENERATION, generation_id, metadata
        )

    def generation_evaluate(self, generation_id: str) -> EvaluateContainer:
        return Generation._evaluate_(self.writer, Entity.GENERATION, generation_id)

    # Span methods
    @deprecated(
        "This method will be removed in a future version. Use span_add_generation instead."
    )
    def span_generation(
        self, span_id: str, config: Union[GenerationConfig, GenerationConfigDict]
    ) -> Generation:
        """
        Adds a generation to the span.

        Args:
            span_id (str): The ID of the span.
            config (GenerationConfig): The configuration for the generation.

        Returns:
            Generation: The newly created generation.
        """
        return Span.generation_(self.writer, span_id, config)

    def span_add_generation(
        self, span_id: str, config: Union[GenerationConfig, GenerationConfigDict]
    ) -> Generation:
        """
        Adds a generation to the span.

        Args:
            span_id (str): The ID of the span.
            config (GenerationConfig): The configuration for the generation.

        Returns:
            Generation: The newly created generation.
        """
        return Span.generation_(self.writer, span_id, config)

    def span_add_error(self, span_id: str, config: ErrorConfig) -> Error:
        """
        Adds an error to the span.
        """
        return Span.error_(self.writer, span_id, config)

    @deprecated(
        "This method will be removed in a future version. Use span_add_retrieval instead."
    )
    def span_retrieval(
        self, span_id: str, config: Union[RetrievalConfig, RetrievalConfigDict]
    ) -> Retrieval:
        """
        Adds a retrieval to the span.

        Args:
            span_id (str): The ID of the span.
            config (RetrievalConfig): The configuration for the retrieval.

        Returns:
            Retrieval: The newly created retrieval.
        """
        return Span.retrieval_(self.writer, span_id, config)

    def span_add_retrieval(
        self, span_id: str, config: Union[RetrievalConfig, RetrievalConfigDict]
    ) -> Retrieval:
        """
        Adds a retrieval to the span.

        Args:
            span_id (str): The ID of the span.
            config (RetrievalConfig): The configuration for the retrieval.

        Returns:
            Retrieval: The newly created retrieval.
        """
        return Span.retrieval_(self.writer, span_id, config)

    def span_add_tool_call(
        self, span_id: str, config: Union[ToolCallConfig, ToolCallConfigDict]
    ) -> ToolCall:
        """
        Adds a tool call to the span.

        Args:
            span_id (str): The ID of the span.
            config (ToolCallConfig): The configuration for the tool call.

        Returns:
            ToolCall: The newly created tool call.
        """
        return Span.tool_call_(self.writer, span_id, config)

    def span_end(self, span_id: str):
        """
        Ends the span.

        Args:
            span_id (str): The ID of the span.
        """
        Span.end_(self.writer, span_id)

    def span_add_tag(self, span_id: str, key: str, value: str):
        """
        Adds a tag to the span.

        Args:
            span_id (str): The ID of the span.
            key (str): The key of the tag.
            value (str): The value of the tag.
        """
        Span.add_tag_(self.writer, span_id, key, value)

    def span_event(
        self,
        span_id: str,
        event_id: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Adds an event to the span.

        Args:
            span_id (str): The ID of the span.
            event_id (str): The ID of the event.
            name (str): The name of the event.
            tags (Optional[Dict[str, str]]): The tags associated with the event.
        """
        Span.event_(self.writer, span_id, event_id, name, tags, metadata)

    def span_add_metadata(self, span_id: str, metadata: Dict[str, Any]):
        """
        Adds metadata to the span.

        Args:
            span_id (str): The ID of the span.
            metadata (Dict[str, Any]): The metadata to add.
        """
        Span.add_metadata_(self.writer, Entity.SPAN, span_id, metadata)

    def span_add_attachment(
        self,
        span_id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Adds an attachment to the span.
        """
        Span.add_attachment_(self.writer, span_id, attachment)

    @deprecated(
        "This method will be removed in a future version. Use span_add_sub_span instead."
    )
    def span_span(
        self, span_id: str, config: Union[SpanConfig, SpanConfigDict]
    ) -> Span:
        """
        Adds a span to the span.

        Args:
            span_id (str): The ID of the span.
            config (SpanConfig): The configuration for the sub-span.

        Returns:
            Span: The newly created sub-span.
        """
        return Span.span_(self.writer, span_id, config)

    def span_add_sub_span(
        self, span_id: str, config: Union[SpanConfig, SpanConfigDict]
    ) -> Span:
        """
        Adds a sub-span to the span.

        Args:
            span_id (str): The ID of the span.
            config (SpanConfig): The configuration for the sub-span.

        Returns:
            Span: The newly created sub-span.
        """
        return Span.span_(self.writer, span_id, config)

    def span_evaluate(self, span_id: str) -> EvaluateContainer:
        return Span._evaluate_(self.writer, Entity.SPAN, span_id)

    # Retrieval methods
    def retrieval_end(self, retrieval_id: str):
        """
        Ends the retrieval.

        Args:
            retrieval_id (str): The ID of the retrieval.
        """
        Retrieval.end_(self.writer, retrieval_id)

    def retrieval_input(self, retrieval_id: str, query: Any):
        """
        Sets the input for the retrieval.

        Args:
            retrieval_id (str): The ID of the retrieval.
            query (Any): The input for the retrieval.
        """
        Retrieval.input_(self.writer, retrieval_id, query)

    def retrieval_output(self, retrieval_id: str, docs: Any):
        """
        Sets the output for the retrieval.

        Args:
            retrieval_id (str): The ID of the retrieval.
            docs (Any): The output for the retrieval.
        """
        Retrieval.output_(self.writer, retrieval_id, docs)

    def retrieval_add_tag(self, retrieval_id: str, key: str, value: str):
        """
        Adds a tag to the retrieval.

        Args:
            retrieval_id (str): The ID of the retrieval.
            key (str): The key of the tag.
            value (str): The value of the tag.
        """
        Retrieval.add_tag_(self.writer, retrieval_id, key, value)

    def retrieval_add_metadata(self, retrieval_id: str, metadata: Dict[str, Any]):
        Retrieval.add_metadata_(self.writer, Entity.RETRIEVAL, retrieval_id, metadata)

    def retrieval_evaluate(self, retrieval_id: str) -> EvaluateContainer:
        return Retrieval._evaluate_(self.writer, Entity.RETRIEVAL, retrieval_id)

    def retrieval_add_attachment(
        self,
        retrieval_id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Adds an attachment to the retrieval.
        """
        Retrieval.add_attachment_(self.writer, retrieval_id, attachment)

    # Tool call methods
    def tool_call_update(self, tool_call_id: str, data: Dict[str, Any]):
        """
        Updates the tool call.

        Args:
            tool_call_id (str): The ID of the tool call.
            data (Dict[str, Any]): The data to update the tool call with.
        """
        ToolCall.update_(self.writer, tool_call_id, data)

    def tool_call_result(self, tool_call_id: str, result: str):
        """
        Sets the result for the tool call.

        Args:
            tool_call_id (str): The ID of the tool call.
            result (Any): The result for the tool call.
        """
        if not isinstance(result, str):
            result = json.dumps(result)
        ToolCall.result_(self.writer, tool_call_id, result)

    def tool_call_error(
        self, tool_call_id: str, error: Union[ToolCallError, ToolCallErrorDict]
    ):
        """
        Sets the error for the tool call.

        Args:
            tool_call_id (str): The ID of the tool call.
            error (ToolCallError): The error for the tool call.
        """
        ToolCall.error_(self.writer, tool_call_id, error)

    def tool_call_add_metadata(self, tool_call_id: str, metadata: Dict[str, Any]):
        """
        Adds metadata to the tool call.

        Args:
            tool_call_id (str): The ID of the tool call.
            metadata (Dict[str, Any]): The metadata to add.
        """
        ToolCall.add_metadata_(self.writer, Entity.TOOL_CALL, tool_call_id, metadata)

    @property
    def id(self):
        """
        Returns the ID of the logger.
        """
        return self._id

    def flush(self):
        """
        Flushes the writer.
        """
        self.writer.flush()

    def cleanup(self, is_sync=False):
        """
        Cleans up the writer.
        """
        self.writer.cleanup(is_sync)
