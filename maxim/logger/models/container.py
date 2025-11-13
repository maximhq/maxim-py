"""Models for LangChain logging and tracing functionality.

This module contains data models used for tracking and logging LangChain operations,
including metadata storage and run information.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...scribe import scribe
from ..logger import (
    ErrorConfig,
    Generation,
    GenerationConfig,
    Logger,
    Retrieval,
    RetrievalConfig,
    Span,
    SpanConfig,
    ToolCall,
    ToolCallConfig,
    TraceConfig,
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
    generation_tags: Optional[Dict[str, str]] = None
    retrieval_tags: Optional[Dict[str, str]] = None
    trace_tags: Optional[Dict[str, str]] = None
    chain_tags: Optional[Dict[str, str]] = None

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
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to parse metadata: %s\n%s",
                e,
                traceback.format_exc(),
            )


class Container:
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
        mark_created=False,
    ):
        self._logger = logger
        self._type = container_type
        self._id = container_id
        self._name = name
        self._parent = parent
        self._created = mark_created

    def set_name(self, name: str) -> None:
        self._name = name

    def create(self, tags: Optional[Dict[str, str]] = None) -> None:
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
    def add_generation(self, config: GenerationConfig) -> Generation:
        """
        Adds a generation to the container
        Returns:
            Generation: Generation object
        """
        pass

    @abstractmethod
    def add_tool_call(self, config: ToolCallConfig) -> ToolCall:
        """
        Adds a tool call to the container
        Returns:
            ToolCall: ToolCall object
        """
        pass

    def add_event(self, event_id: str, name: str, tags: Dict[str, str], metadata:Optional[Dict[str,Any]]=None) -> None:
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
    def add_span(self, config: SpanConfig) -> Span:
        """
        Adds a span to the container
        Returns:
            Span: Span object
        """
        pass

    @abstractmethod
    def add_retrieval(self, config: RetrievalConfig) -> Retrieval:
        """
        Adds a retrieval to the container
        Returns:
            Retrieval: Retrieval object
        """
        pass

    def add_tags(self, tags: Dict[str, str]) -> None:
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

    def add_metadata(self, metadata: Dict[str, str]) -> None:
        """
        Adds metadata to the container
        Args:
            metadata (Optional[Dict[str,str]]): Metadata to add
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
        mark_created=False,
    ):
        super().__init__(
            logger=logger,
            container_id=trace_id,
            container_type="trace",
            name=trace_name,
            parent=parent,
            mark_created=mark_created,
        )

    def create(self, tags: Optional[Dict[str, str]] = None) -> None:
        config = TraceConfig(id=self._id, name=self._name, tags=tags)
        if self._parent is not None:
            config.session_id = self._parent
        self._logger.trace(config)
        self._created = True

    def add_generation(self, config: GenerationConfig) -> Generation:
        """
        Adds a generation to the container
        Returns:
            Generation: Generation object
        """
        return self._logger.trace_add_generation(self._id, config)

    def add_retrieval(self, config: RetrievalConfig) -> Retrieval:
        return self._logger.trace_add_retrieval(self._id, config=config)

    def add_event(self, event_id: str, name: str, tags: Dict[str, str],metadata:Optional[Dict[str,Any]]=None) -> None:
        self._logger.trace_event(self._id, event_id, name, tags,metadata)

    def add_span(self, config: SpanConfig) -> Span:
        return self._logger.trace_add_span(self._id, config)

    def add_error(self, error: ErrorConfig):
        self._logger.trace_add_error(self._id, error)

    def set_input(self, input: str) -> None:
        return self._logger.trace_set_input(self._id, input)

    def set_output(self, output: str) -> None:
        return self._logger.trace_set_output(self._id, output)

    def add_tags(self, tags: Dict[str, str]) -> None:
        for key, value in tags.items():
            self._logger.trace_add_tag(self._id, key, value)

    def add_tool_call(self, config: ToolCallConfig) -> ToolCall:
        return self._logger.trace_add_tool_call(self._id, config)

    def add_metadata(self, metadata: Dict[str, str]) -> None:
        return self._logger.trace_add_metadata(self._id, metadata)

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
        mark_created=False,
    ):
        super().__init__(
            logger=logger,
            container_id=span_id,
            container_type="span",
            name=span_name,
            parent=parent,
            mark_created=mark_created,
        )

    def create(self, tags: Optional[Dict[str, str]] = None) -> None:
        config = SpanConfig(id=self._id, name=self._name, tags=tags)
        if self._parent is None:
            raise ValueError("[MaximSDK] Span without a parent is invalid")
        self._logger.trace_add_span(self._parent, config)
        self._created = True

    def add_generation(self, config: GenerationConfig) -> Generation:
        return self._logger.span_add_generation(self._id, config)

    def add_retrieval(self, config: RetrievalConfig) -> Retrieval:
        return self._logger.span_add_retrieval(self._id, config=config)

    def add_event(self, event_id: str, name: str, tags: Dict[str, str],metadata:Optional[Dict[str,Any]]=None) -> None:
        self._logger.span_event(self._id, event_id, name, tags,metadata)

    def add_span(self, config: SpanConfig) -> Span:
        return self._logger.span_add_sub_span(self._id, config)

    def add_error(self, error: ErrorConfig):
        self._logger.span_add_error(self._id, error)

    def add_tags(self, tags: Dict[str, str]) -> None:
        for key, value in tags.items():
            self._logger.span_add_tag(self._id, key, value)

    def set_input(self, input: str) -> None:
        return self._logger.span_add_metadata(self._id, {"input": input})

    def set_output(self, output) -> None:
        return self._logger.span_add_metadata(self._id, {"output": output})

    def add_tool_call(self, config: ToolCallConfig) -> ToolCall:
        return self._logger.span_add_tool_call(self._id, config)

    def add_metadata(self, metadata: Dict[str, str]) -> None:
        return self._logger.span_add_metadata(self._id, metadata)

    def end(self) -> None:
        """
        Ends the container
        """
        self._logger.span_end(self._id)


class ContainerManager:
    """
    Manages mapping between LangChain run IDs and Maxim containers (trace/span).

    This mirrors the behavior of the JS ContainerManager used by the Maxim LangChain tracer,
    ensuring we never overwrite a parent trace container mapping with a child span container
    and allowing proper lifecycle management.
    """

    def __init__(self) -> None:
        # Map a run_id (as string) to its current container (TraceContainer or SpanContainer)
        self._run_id_to_container: Dict[str, Container] = {}
        # Track top-level root trace containers keyed by the originating run_id (no parent)
        self._root_run_id_to_trace: Dict[str, TraceContainer] = {}

    def get_container(self, run_id: str) -> Optional[Container]:
        return self._run_id_to_container.get(run_id)

    def set_container(self, run_id: str, container: Container) -> None:
        self._run_id_to_container[run_id] = container

    def remove_run_id_mapping(self, run_id: str) -> None:
        if run_id in self._run_id_to_container:
            self._run_id_to_container.pop(run_id)

    def set_root_trace(self, run_id: str, trace_container: TraceContainer) -> None:
        self._root_run_id_to_trace[run_id] = trace_container

    def get_root_trace(self, run_id: str) -> Optional[TraceContainer]:
        return self._root_run_id_to_trace.get(run_id)

    def pop_root_trace(self, run_id: str) -> Optional[TraceContainer]:
        return self._root_run_id_to_trace.pop(run_id, None)
