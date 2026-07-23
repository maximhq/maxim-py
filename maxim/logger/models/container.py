"""Models for LangChain logging and tracing functionality.

This module contains data models used for tracking and logging LangChain operations,
including metadata storage and run information.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from typing_extensions import override

from typing import Union

from ...env_config import env_int
from ...scribe import scribe
from ..__init__ import (
    ErrorConfig,
    FileAttachment,
    FileDataAttachment,
    Generation,
    GenerationConfigDict,
    Logger,
    Retrieval,
    RetrievalConfigDict,
    Span,
    SpanConfigDict,
    ToolCall,
    ToolCallConfigDict,
    TraceConfigDict,
    UrlAttachment,
)


@dataclass
class Metadata:
    """
    RunMetadata class to holds the metadata info associated with a run
    """

    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    chain_name: Optional[str] = None
    span_name: Optional[str] = None
    trace_name: Optional[str] = None
    generation_name: Optional[str] = None
    retrieval_name: Optional[str] = None
    generation_tags: Optional[dict[str, str]] = None
    retrieval_tags: Optional[dict[str, str]] = None
    trace_tags: Optional[dict[str, str]] = None
    chain_tags: Optional[dict[str, str]] = None
    session_tags: Optional[dict[str, str]] = None

    def __init__(self, metadata: Optional[Dict[str, Any]]):
        """
        Initializes the RunMetadata object

        Args:
            metadata (Optional[Dict[str,Any]]): Metadata to initialize from
        """
        if metadata is None:
            return
        try:
            self.session_id = metadata.get("session_id", None)
            self.trace_id = metadata.get("trace_id", None)
            self.span_id = metadata.get("span_id", None)
            self.span_name = metadata.get("span_name", None)
            self.chain_name = metadata.get("chain_name", None)
            self.trace_name = metadata.get("trace_name", None)
            self.generation_name = metadata.get("generation_name", None)
            self.retrieval_name = metadata.get("retrieval_name", None)
            self.generation_tags = metadata.get("generation_tags", None)
            self.retrieval_tags = metadata.get("retrieval_tags", None)
            self.trace_tags = metadata.get("trace_tags", None)
            self.chain_tags = metadata.get("chain_tags", None)
            self.session_tags = metadata.get("session_tags", None)
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to parse metadata: %s\n%s",
                e,
                traceback.format_exc(),
            )


class Container(ABC):
    """
    Container class to hold the container id, type and name for logging
    """

    _logger: Logger
    _type: str
    _id: str
    _name: Optional[str] = None
    _parent: Optional[str] = None
    _created: bool = False

    def __init__(
        self,
        logger: Logger,
        container_id: str,
        container_type: str,
        name: Optional[str] = None,
        parent: Optional[str] = None,
        mark_created: bool = False,
    ):
        self._logger = logger
        self._type = container_type
        self._id = container_id
        self._name = name
        self._parent = parent
        self._created = mark_created

    def set_name(self, name: str) -> None:
        self._name = name

    def create(self, tags: Optional[dict[str, str]] = None) -> None:
        """
        Creates the container in the logger
        """

    def id(self) -> str:
        """
        Returns:
            str: id of the container
        """
        return self._id

    def type(self) -> str:
        """
        Returns:
            str: type of the container
        """
        return self._type

    def name(self) -> Optional[str]:
        """
        Returns:
            str: name of the container
        """
        return self._name

    def is_created(self) -> bool:
        """
        Checks if the container has been created
        Returns:
            bool: True if the container has been created, False otherwise
        """
        return self._created

    def parent(self) -> Optional[str]:
        """
        Returns:
            Container: parent container
        """
        return self._parent

    @abstractmethod
    def add_generation(self, config: GenerationConfigDict) -> Generation:
        """
        Adds a generation to the container
        Returns:
            Generation: Generation object
        """
        pass

    @abstractmethod
    def add_tool_call(self, config: ToolCallConfigDict) -> ToolCall:
        """
        Adds a tool call to the container
        Returns:
            ToolCall: ToolCall object
        """
        pass

    def add_event(
        self,
        event_id: str,
        name: str,
        tags: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Adds an event to the container.

        Args:
            event_id (str): Unique identifier for the event.
            name (str): Name of the event.
            tags (Dict[str, str]): Additional key-value pairs to associate with the event.

        Returns:
            None
        """

    @abstractmethod
    def add_span(self, config: SpanConfigDict) -> Span:
        """
        Adds a span to the container
        Returns:
            Span: Span object
        """
        pass

    @abstractmethod
    def add_retrieval(self, config: RetrievalConfigDict) -> Retrieval:
        """
        Adds a retrieval to the container
        Returns:
            Retrieval: Retrieval object
        """
        pass

    def add_tags(self, tags: dict[str, str]) -> None:
        """
        Adds tags to the container
        Args:
            tags (Optional[Dict[str,str]]): Tags to add
        """

    def add_error(self, error: ErrorConfig) -> None:
        """
        Adds an error to the container
        Args:
            error (GenerationError): Error to add
        """
        pass

    def set_input(self, input: str) -> None:
        """
        Sets the input to the container
        Args:
            input (str): Input to set
        """

    def set_output(self, output) -> None:
        """
        Sets the output to the container
        Args:
            output (str): Output to set
        """

    def add_metadata(self, metadata: dict[str, str]) -> None:
        """
        Adds metadata to the container
        Args:
            metadata (Optional[Dict[str,str]]): Metadata to add
        """
        pass

    def add_attachment(
        self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]
    ) -> None:
        """
        Adds an attachment to the container
        Args:
            attachment: The attachment to add
        """
        pass

    def end(self) -> None:
        """
        Ends the container
        """


class TraceContainer(Container):
    """
    A trace in the logger
    """

    def __init__(
        self,
        logger: Logger,
        trace_id: str,
        trace_name: Optional[str] = None,
        parent: Optional[str] = None,
        mark_created: bool = False,
    ):
        super().__init__(
            logger=logger,
            container_id=trace_id,
            container_type="trace",
            name=trace_name,
            parent=parent,
            mark_created=mark_created,
        )

    @override
    def create(self, tags: Optional[dict[str, str]] = None) -> None:
        config = TraceConfigDict({"id": self._id, "name": self._name, "tags": tags})
        if self._parent is not None:
            config["session_id"] = self._parent
        _ = self._logger.trace(config)
        self._created: bool = True

    @override
    def add_generation(self, config: GenerationConfigDict) -> Generation:
        """
        Adds a generation to the container
        Returns:
            Generation: Generation object
        """
        return self._logger.trace_add_generation(self._id, config)

    @override
    def add_retrieval(self, config: RetrievalConfigDict) -> Retrieval:
        return self._logger.trace_add_retrieval(self._id, config=config)

    @override
    def add_event(
        self,
        event_id: str,
        name: str,
        tags: dict[str, str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._logger.trace_add_event(self._id, event_id, name, tags, metadata)

    @override
    def add_span(self, config: SpanConfigDict) -> Span:
        return self._logger.trace_add_span(self._id, config)

    @override
    def add_error(self, error: ErrorConfig):
        _ = self._logger.trace_add_error(self._id, error)

    @override
    def set_input(self, input: str) -> None:
        return self._logger.trace_set_input(self._id, input)

    @override
    def set_output(self, output: str) -> None:
        return self._logger.trace_set_output(self._id, output)

    @override
    def add_tags(self, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            self._logger.trace_add_tag(self._id, key, value)

    @override
    def add_tool_call(self, config: ToolCallConfigDict) -> ToolCall:
        return self._logger.trace_add_tool_call(self._id, config)

    @override
    def add_metadata(self, metadata: dict[str, str]) -> None:
        return self._logger.trace_add_metadata(self._id, metadata)

    @override
    def add_attachment(
        self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]
    ) -> None:
        return self._logger.trace_add_attachment(self._id, attachment)

    @override
    def end(self) -> None:
        """
        Ends the container
        """
        self._logger.trace_end(self._id)


class SpanContainer(Container):
    """
    A span in the logger
    """

    def __init__(
        self,
        span_id: str,
        logger: Logger,
        span_name: Optional[str] = None,
        parent: Optional[str] = None,
        mark_created: bool = False,
    ):
        super().__init__(
            logger=logger,
            container_id=span_id,
            container_type="span",
            name=span_name,
            parent=parent,
            mark_created=mark_created,
        )

    @override
    def create(self, tags: Optional[dict[str, str]] = None) -> None:
        config = SpanConfigDict({"id": self._id, "name": self._name, "tags": tags})
        if self._parent is None:
            raise ValueError("[MaximSDK] Span without a parent is invalid")
        _ = self._logger.trace_add_span(self._parent, config)
        self._created: bool = True

    @override
    def add_generation(self, config: GenerationConfigDict) -> Generation:
        return self._logger.span_add_generation(self._id, config)

    @override
    def add_retrieval(self, config: RetrievalConfigDict) -> Retrieval:
        return self._logger.span_add_retrieval(self._id, config=config)

    @override
    def add_event(
        self,
        event_id: str,
        name: str,
        tags: dict[str, str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._logger.span_event(self._id, event_id, name, tags, metadata)

    @override
    def add_span(self, config: SpanConfigDict) -> Span:
        return self._logger.span_add_sub_span(self._id, config)

    @override
    def add_error(self, error: ErrorConfig):
        _ = self._logger.span_add_error(self._id, error)

    @override
    def add_tags(self, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            self._logger.span_add_tag(self._id, key, value)

    @override
    def set_input(self, input: str) -> None:
        return self._logger.span_add_metadata(self._id, {"input": input})

    @override
    def set_output(self, output: str) -> None:
        return self._logger.span_add_metadata(self._id, {"output": output})

    @override
    def add_tool_call(self, config: ToolCallConfigDict) -> ToolCall:
        return self._logger.span_add_tool_call(self._id, config)

    @override
    def add_metadata(self, metadata: dict[str, str]) -> None:
        return self._logger.span_add_metadata(self._id, metadata)

    @override
    def end(self) -> None:
        """
        Ends the container
        """
        self._logger.span_end(self._id)


class SessionContainer(Container):
    """
    A session container in the logger.
    Used for managing session-level operations in a unified container interface.
    """

    def __init__(
        self,
        logger: Logger,
        session_id: str,
        session_name: Optional[str] = None,
        mark_created: bool = False,
    ):
        super().__init__(
            logger=logger,
            container_id=session_id,
            container_type="session",
            name=session_name,
            parent=None,
            mark_created=mark_created,
        )

    @override
    def create(self, tags: Optional[dict[str, str]] = None) -> None:
        from ..components.session import SessionConfigDict

        config = SessionConfigDict(id=self._id, name=self._name, tags=tags)
        _ = self._logger.session(config)
        self._created = True

    def add_trace(self, config: TraceConfigDict) -> "TraceContainer":
        """
        Adds a trace to the session and returns a TraceContainer.

        Args:
            config: The trace configuration.

        Returns:
            TraceContainer: A container wrapping the created trace.
        """
        config["session_id"] = self._id
        _ = self._logger.trace(config)
        return TraceContainer(
            logger=self._logger,
            trace_id=config["id"],
            trace_name=config.get("name"),
            parent=self._id,
            mark_created=True,
        )

    @override
    def add_generation(self, config: GenerationConfigDict) -> Generation:
        raise NotImplementedError(
            "[MaximSDK] Cannot add generation directly to session. Add a trace first."
        )

    @override
    def add_retrieval(self, config: RetrievalConfigDict) -> Retrieval:
        raise NotImplementedError(
            "[MaximSDK] Cannot add retrieval directly to session. Add a trace first."
        )

    @override
    def add_span(self, config: SpanConfigDict) -> Span:
        raise NotImplementedError(
            "[MaximSDK] Cannot add span directly to session. Add a trace first."
        )

    @override
    def add_tool_call(self, config: ToolCallConfigDict) -> ToolCall:
        raise NotImplementedError(
            "[MaximSDK] Cannot add tool call directly to session. Add a trace first."
        )

    @override
    def add_tags(self, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            self._logger.session_add_tag(self._id, key, value)

    @override
    def add_attachment(self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]) -> None:
        return self._logger.session_add_attachment(self._id, attachment)

    @override
    def end(self) -> None:
        """
        Ends the session.
        """
        self._logger.session_end(self._id)


# Backstop TTL (seconds) after which a run_id mapping is considered abandoned
# and swept. This is only a safety net for runs whose terminal callback
# (on_*_end / on_*_error) never fires - e.g. a cancelled stream, a client
# disconnect, or a process killed mid-run.
#
# The timestamp is refreshed on every access, and every child callback touches
# its parent, so the window only has to outlast the gap between two consecutive
# callbacks of the same run - never the run's total duration. A multi-step
# agent that keeps emitting callbacks is never evicted no matter how long it
# runs. The one thing this window must exceed is a single leaf call (one LLM
# generation or tool execution) that emits no callbacks while in flight, so an
# active-but-slow trace is not swept out from under its own terminal callback.
#
# One hour by default: comfortably longer than any realistic single LLM or tool
# call. Now that terminal callbacks release their mappings, the only entries
# that actually reach this TTL are genuinely abandoned runs, whose volume is
# tiny, so a generous window costs almost nothing in memory. Deployments with
# legitimately long-idle (e.g. human-in-the-loop) traces can raise it via
# MAXIM_CONTAINER_MAPPING_TTL_SECONDS.
CONTAINER_MAPPING_TTL_ENV = "MAXIM_CONTAINER_MAPPING_TTL_SECONDS"
DEFAULT_CONTAINER_MAPPING_TTL = 60 * 60


def _default_container_ttl() -> int:
    return env_int(CONTAINER_MAPPING_TTL_ENV, DEFAULT_CONTAINER_MAPPING_TTL, minimum=1)


class ContainerManager:
    """
    Manages mapping between LangChain run IDs and Maxim containers (trace/span).

    This mirrors the behavior of the JS ContainerManager used by the Maxim LangChain tracer,
    ensuring we never overwrite a parent trace container mapping with a child span container
    and allowing proper lifecycle management.

    Each mapping is timestamped so that runs which never reach a terminal callback
    (cancelled streams, client disconnects, exceptions raised inside LangChain before
    the end event) cannot accumulate indefinitely. Timestamps are refreshed on access
    and expired entries are swept on writes.
    """

    def __init__(self, ttl_seconds: Optional[int] = None) -> None:
        # Map a run_id (as string) to (container, last_touched_epoch)
        self._run_id_to_container: dict[str, tuple[Container, float]] = {}
        # Track top-level root trace containers keyed by run_id -> (trace, last_touched_epoch)
        self._root_run_id_to_trace: dict[str, tuple[TraceContainer, float]] = {}
        # Read the env default lazily (at construction, not import) so it can be
        # set by an embedding app or a test before the manager is created. An
        # explicit ttl_seconds always wins.
        self._ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else _default_container_ttl()
        )

    def _sweep_expired(self) -> None:
        """Remove mappings that have not been touched within the TTL window."""
        cutoff = time.time() - self._ttl_seconds
        expired_runs = [
            run_id
            for run_id, (_, touched) in self._run_id_to_container.items()
            if touched < cutoff
        ]
        for run_id in expired_runs:
            del self._run_id_to_container[run_id]
        expired_roots = [
            run_id
            for run_id, (_, touched) in self._root_run_id_to_trace.items()
            if touched < cutoff
        ]
        for run_id in expired_roots:
            del self._root_run_id_to_trace[run_id]

    def get_container(self, run_id: str) -> Optional[Container]:
        entry = self._run_id_to_container.get(run_id)
        if entry is None:
            return None
        container, _ = entry
        # Refresh the timestamp so active runs are never swept.
        self._run_id_to_container[run_id] = (container, time.time())
        return container

    def set_container(self, run_id: str, container: Container) -> None:
        self._run_id_to_container[run_id] = (container, time.time())
        self._sweep_expired()

    def remove_run_id_mapping(self, run_id: str) -> None:
        self._run_id_to_container.pop(run_id, None)

    def set_root_trace(self, run_id: str, trace_container: TraceContainer) -> None:
        self._root_run_id_to_trace[run_id] = (trace_container, time.time())
        self._sweep_expired()

    def get_root_trace(self, run_id: str) -> Optional[TraceContainer]:
        entry = self._root_run_id_to_trace.get(run_id)
        if entry is None:
            return None
        trace_container, _ = entry
        self._root_run_id_to_trace[run_id] = (trace_container, time.time())
        return trace_container

    def pop_root_trace(self, run_id: str) -> Optional[TraceContainer]:
        entry = self._root_run_id_to_trace.pop(run_id, None)
        if entry is None:
            return None
        trace_container, _ = entry
        return trace_container
