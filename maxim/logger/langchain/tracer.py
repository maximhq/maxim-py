import json
import uuid
from dataclasses import fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Literal
from typing_extensions import Protocol, TypedDict
from uuid import UUID

# Use LangChain features
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult

from ...expiring_key_value_store import ExpiringKeyValueStore
from ...logger import (
    GenerationConfig,
    Logger,
    RetrievalConfig,
    SpanConfig,
    ToolCallConfig,
    ToolCallError,
)
from ...scribe import scribe
from ..logger import Generation
from ..models import Container, Metadata, SpanContainer, TraceContainer
from ..models.container import ContainerManager
from .utils import (
    parse_langchain_llm_error,
    parse_langchain_llm_result,
    parse_langchain_messages,
    parse_langchain_model_and_provider,
    parse_langchain_model_parameters,
    parse_langchain_provider,
)

# 20 minutes
DEFAULT_TIMEOUT = 60 * 20

tracer_callback_type = Callable[[str, Any], None]

class MaximLangchainTracer(BaseCallbackHandler):
    """
    A callback handler that logs langchain outputs to Maxim logger

    Args:
        logger: Logger: Maxim Logger instance to log outputs
    """

    def __init__(
        self,
        logger: Logger,
        metadata: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, List[str]]] = None,
        callback: Optional[tracer_callback_type] = None,
    ) -> None:
        """Initializes the Langchain Tracer
        Args:
            logger: Logger: Maxim Logger instance to log outputs
            metadata: Optional[Dict[str, Any]]: Additional metadata for tracing
            eval_config: Optional[Dict[str, List[str]]]: Evaluation configuration
            callback: Optional[tracer_callback_type]: Callback function that receives
                events with signature: callback(event_type: str, event_data: Dict[str, Any])
        """
        super().__init__()
        self.run_inline = True
        self.logger = logger
        self.container_manager: ContainerManager = ContainerManager()
        self.metadata_store = ExpiringKeyValueStore()
        self.to_be_evaluated_container_store = ExpiringKeyValueStore()
        self.metadata = None
        self.eval_config = eval_config
        self.callback = callback
        if metadata is not None:
            self.__validate_maxim_metadata(metadata)
            self.metadata = metadata

    def _parse_metadata(self, metadata: Optional[Dict[str, Any]]):
        """
        Parses the metadata
        """
        try:
            if (
                metadata is None
                or not isinstance(metadata, dict)
                or "maxim" not in metadata
            ):
                return
            maxim_metadata = metadata.get("maxim", None)
            if maxim_metadata is None:
                return
            self.__validate_maxim_metadata(maxim_metadata)
            self.metadata = maxim_metadata
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to parse metadata: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def __validate_maxim_metadata(self, metadata: Optional[Dict[str, Any]]):
        """
        Validates the metadata
        """
        if metadata is None:
            return
        id_keys = ["session_id", "trace_id", "span_id"]
        present_keys = [key for key in id_keys if key in metadata]
        if len(present_keys) > 1:
            raise ValueError(
                f"Multiple keys found in metadata: {present_keys}. You can pass only one of these."
            )
        valid_keys = [field.name for field in fields(Metadata)]
        invalid_keys = [key for key in metadata if key not in valid_keys]
        if len(invalid_keys) > 0:
            raise ValueError(
                f"Invalid keys found in metadata: {invalid_keys}. Valid keys are {valid_keys}"
            )

    def __get_container_from_metadata(self, run_id: UUID) -> Container:
        """
        Gets the container from the metadata
        """
        maxim_metadata = self.__get_metadata()
        container: Optional[Container] = None
        if maxim_metadata is not None:
            span_id: Optional[str] = maxim_metadata.span_id
            if span_id is not None:
                span_name: Optional[str] = maxim_metadata.span_name
                container = SpanContainer(
                    span_id=span_id,
                    logger=self.logger,
                    span_name=span_name,
                    mark_created=True,
                )
                # we intentionally add trace tags to the span container
                if maxim_metadata.trace_tags is not None:
                    scribe().info(
                        f"[MaximSDK] Adding trace tags to span container: {maxim_metadata.trace_tags}"
                    )
                    container.add_tags(maxim_metadata.trace_tags)
            trace_id = maxim_metadata.trace_id
            if trace_id is not None:
                trace_name = maxim_metadata.trace_name
                container = TraceContainer(
                    trace_id=trace_id,
                    logger=self.logger,
                    trace_name=trace_name,
                    mark_created=True,
                )
                if maxim_metadata.trace_tags is not None:
                    scribe().info(
                        f"[MaximSDK] Adding trace tags to trace container: {maxim_metadata.trace_tags}"
                    )
                    container.add_tags(maxim_metadata.trace_tags)

        if container is None:
            # We will check if the container is created with run_id via container manager
            container = self.container_manager.get_container(str(run_id))

        if container is None:
            session_id: Optional[str] = (
                maxim_metadata.session_id if maxim_metadata else None
            )
            container = TraceContainer(
                trace_id=str(uuid.uuid4()),
                logger=self.logger,
                trace_name="Trace",
                parent=session_id,
            )
            if self.callback is not None:
                self.callback(
                    "trace.started",
                    {
                        "trace_id": container.id(),
                        "trace_name": container.name(),
                    },
                )
            container.create()
            if maxim_metadata is not None and maxim_metadata.trace_tags is not None:
                container.add_tags(maxim_metadata.trace_tags)
            # Register this trace against the current run id and as root trace
            self.container_manager.set_container(str(run_id), container)
            if isinstance(container, TraceContainer):
                self.container_manager.set_root_trace(str(run_id), container)

        return container

    def __get_container(
        self, run_id: UUID, parent_run_id: Optional[UUID] = None
    ) -> Optional[Container]:
        """
        Gets the container for the given run_id
        """
        container: Optional[Container] = None
        if parent_run_id is None:
            # This is the first activity in this run - either get existing or create new trace
            container = self.container_manager.get_container(str(run_id))
            if container is None:
                container = self.__get_container_from_metadata(run_id)
        else:
            container = self.container_manager.get_container(str(parent_run_id))
            if container is None:
                # Create a trace container and register it
                scribe().warning(
                    f"[MaximSDK] Couldn't find a container for parent run id {parent_run_id}. Creating a new trace."
                )
                trace_container = TraceContainer(
                    trace_id=str(parent_run_id),
                    logger=self.logger,
                    trace_name="Trace",
                    mark_created=True,
                )
                trace_container.create()
                if self.callback is not None:
                    self.callback(
                        "trace.started",
                        {
                            "trace_id": trace_container.id(),
                            "trace_name": trace_container.name(),
                        },
                    )
                self.container_manager.set_container(
                    str(parent_run_id), trace_container
                )
                if isinstance(trace_container, TraceContainer):
                    self.container_manager.set_root_trace(
                        str(parent_run_id), trace_container
                    )
                container = trace_container
        return container

    def __get_metadata(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Metadata]:
        """
        Gets the metadata
        """
        if (
            metadata is not None
            and "maxim" in metadata
            and metadata.get("maxim", None) is not None
        ):
            return Metadata(metadata.get("maxim", None))
        if self.metadata is not None:
            return Metadata(self.metadata)
        return None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Runs when LLM starts"""
        try:
            self._parse_metadata(metadata)
            scribe().debug("[MaximSDK: Langchain] on_llm_start called")
            model, model_parameters = parse_langchain_model_parameters(**kwargs)
            provider = parse_langchain_provider(serialized)
            model, provider = parse_langchain_model_and_provider(model, provider)
            maxim_messages = parse_langchain_messages(prompts)
            last_input_message = ""
            if maxim_messages is not None:
                for message in maxim_messages:
                    if "role" in message and message["role"] == "user":
                        last_input_message = message["content"]
            generation_id = str(run_id)
            maxim_metadata = self.__get_metadata(metadata)
            # Prepare generation config
            generation_config = GenerationConfig(
                id=generation_id,
                name=maxim_metadata.generation_name if maxim_metadata else None,
                provider=provider,
                model=model,
                messages=maxim_messages,
                model_parameters=model_parameters,
                tags=maxim_metadata.generation_tags if maxim_metadata else None,
            )
            # Adding it back to the container
            container = self.__get_container(run_id, parent_run_id)
            if container is None:
                scribe().error(
                    "[MaximSDK][on_llm_start] Couldn't find a container for generation"
                )
                return
            if not container.is_created():
                container.create()
            if container.parent() is None:
                self.container_manager.set_container(str(run_id), container)
                if isinstance(container, TraceContainer):
                    self.container_manager.set_root_trace(str(run_id), container)
            # Set trace input from last user message like JS tracer
            try:
                if isinstance(container, TraceContainer) and last_input_message:
                    container.set_input(last_input_message)
            except Exception:
                pass
            generation_container = container.add_generation(generation_config)
            # checking if we need to attach evaluator
            if (
                metadata is not None
                and "langgraph_node" in metadata
                and self.eval_config is not None
                and metadata["langgraph_node"] in self.eval_config
            ):
                eval_data = {
                    "container_id": generation_container.id,
                    "generation_container": generation_container,
                    "evaluators": (
                        self.eval_config.get(metadata["langgraph_node"], [])
                        if self.eval_config
                        else []
                    ),
                    "node_name": metadata["langgraph_node"],
                    "input": last_input_message,
                }
                self.to_be_evaluated_container_store.set(
                    str(generation_container.id), eval_data, DEFAULT_TIMEOUT
                )
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to process llm-start: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Runs when a chat model call starts"""
        try:
            scribe().debug("[MaximSDK: Langchain] on_chat_model_start called")
            self._parse_metadata(metadata)
            run_id = kwargs.get("run_id", None)
            model, model_parameters = parse_langchain_model_parameters(**kwargs)
            provider = parse_langchain_provider(serialized)
            model, provider = parse_langchain_model_and_provider(model, provider)
            maxim_messages = parse_langchain_messages(messages)
            # get user message
            last_input_message = ""
            if maxim_messages is not None:
                for message in maxim_messages:
                    if "role" in message and message["role"] == "user":
                        last_input_message = message["content"]
            generation_id = str(run_id)
            maxim_metadata = self.__get_metadata(metadata)
            generation_name = None
            if kwargs.get("parent_run_id", None) is None:
                generation_name = (
                    maxim_metadata.generation_name if maxim_metadata else None
                )
            # Checking if generation id already present for this run_id
            generation_config = GenerationConfig(
                id=generation_id,
                name=generation_name,
                provider=provider,
                model=model,
                messages=maxim_messages,
                model_parameters=model_parameters,
                tags=maxim_metadata.generation_tags if maxim_metadata else None,
            )
            container = self.__get_container(run_id, kwargs.get("parent_run_id", None))
            if container is None:
                scribe().error(
                    "[MaximSDK][on_chat_model_start] Couldn't find a container for generation]"
                )
                return
            if not container.is_created():
                container.create()
            if container.parent() is None:
                self.container_manager.set_container(str(run_id), container)
                if isinstance(container, TraceContainer):
                    self.container_manager.set_root_trace(str(run_id), container)
            generation_container = container.add_generation(generation_config)
            # checking if we need to attach evaluator
            if (
                metadata is not None
                and "langgraph_node" in metadata
                and self.eval_config is not None
                and metadata["langgraph_node"] in self.eval_config
            ):
                eval_data = {
                    "container_id": generation_container.id,
                    "generation_container": generation_container,
                    "evaluators": (
                        self.eval_config.get(metadata["langgraph_node"], [])
                        if self.eval_config
                        else []
                    ),
                    "node_name": metadata["langgraph_node"],
                    "input": last_input_message,
                }
                self.to_be_evaluated_container_store.set(
                    str(generation_container.id), eval_data, DEFAULT_TIMEOUT
                )
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to process chat-model-start: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        try:
            scribe().debug("[MaximSDK][Langchain] on_llm_end called")
            run_id = kwargs.get("run_id", None)
            parent_run_id = kwargs.get("parent_run_id", None)
            result = parse_langchain_llm_result(response)
            container = self.__get_container(run_id, parent_run_id)
            if container is None:
                scribe().error(
                    "[MaximSDK][on_llm_end] Couldn't find a container for generation"
                )
                return
            # here generation_id is the run_id
            self.logger.generation_result(str(run_id), result)
            # Remove mapping for this run_id when top-level (do not end here)
            if container.parent() is None:
                self.container_manager.remove_run_id_mapping(str(run_id))
                try:
                    root = self.container_manager.pop_root_trace(str(run_id))
                    if root is not None:
                        root.end()
                        if self.callback is not None:
                            self.callback(
                                "trace.ended",
                                {
                                    "trace_id": root.id(),
                                    "trace_name": root.name(),
                                },
                            )
                except Exception:
                    pass
            # check if need to attach evaluator
            if self.to_be_evaluated_container_store.get(str(run_id)):
                obj = self.to_be_evaluated_container_store.get(str(run_id))
                if obj:
                    generation_container: Generation = obj["generation_container"]
                    output = None
                    tool_call_name = None
                    tool_call_args = None
                    # extract output
                    if "choices" in result:
                        if isinstance(result["choices"], list):
                            last_choice = result["choices"][-1]
                            message = last_choice["message"]
                            # check if message content is empty : that means its tool call
                            if len(message["content"]) > 0:
                                output = message["content"]
                            elif len(message["tool_calls"]) > 0:
                                tool_message = message["tool_calls"][-1]
                                if tool_message["type"] == "function":
                                    tool_call_name = tool_message["function"].get(
                                        "name", ""
                                    )
                                    tool_call_args = tool_message["function"].get(
                                        "arguments", ""
                                    )

                    # only keeping non None type variables
                    variable_dict = {
                        k: v
                        for k, v in {
                            "input": obj["input"],
                            "output": output,
                            "tool_call_name": tool_call_name,
                            "tool_call_args": tool_call_args,
                        }.items()
                        if v is not None
                    }

                    generation_container.evaluate().with_evaluators(
                        *obj["evaluators"]
                    ).with_variables(variable_dict)
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to process llm-end: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_llm_error(
        self, error: Union[Exception, BaseException, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        try:
            scribe().debug("[MaximSDK] on_llm_error called")
            run_id = kwargs.get("run_id", None)
            parent_run_id = kwargs.get("parent_run_id", None)
            container = self.__get_container(run_id, parent_run_id)
            if container is None:
                scribe().error(
                    "[MaximSDK][on_llm_error] Couldn't find a container for generation"
                )
                return
            generation_error = parse_langchain_llm_error(error)
            self.logger.generation_error(str(run_id), generation_error)
            # Remove mapping for this run_id when top-level (do not end here)
            if container.parent() is None:
                self.container_manager.remove_run_id_mapping(str(run_id))
                try:
                    root = self.container_manager.pop_root_trace(str(run_id))
                    if root is not None:
                        root.end()
                        if self.callback is not None:
                            self.callback(
                                "trace.ended",
                                {
                                    "trace_id": root.id(),
                                    "trace_name": root.name(),
                                },
                            )
                except Exception:
                    pass
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to process llm-error: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run when Retriever starts running.
        """
        try:
            scribe().debug("[MaximSDK] on_retriever_start called")
            retrieval_id = str(run_id)
            retrieval_config = RetrievalConfig(id=retrieval_id)
            container = self.__get_container(run_id, parent_run_id)
            if container is None:
                scribe().error("[MaximSDK] Couldn't find a container for retrieval")
                return
            if not container.is_created():
                container.create()
            retrieval = container.add_retrieval(retrieval_config)
            retrieval.input(query)
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to process retriever-start: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            scribe().debug("[MaximSDK] on_retriever_end called")
            container = self.__get_container(run_id, parent_run_id)
            if container is None:
                scribe().error("[MaximSDK] Couldn't find a container for retrieval")
                return
            documents_list: List[str] = [doc.page_content for doc in documents]
            self.logger.retrieval_output(str(run_id), documents_list)
            if container.parent() is None:
                self.container_manager.remove_run_id_mapping(str(run_id))
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to process retriever-end: %s\n%s",
                e,
                traceback.format_exc(),
            )

    # Chain callbacks

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        try:
            scribe().debug("[MaximSDK: Langchain] on_chain_start called")
            if "metadata" in kwargs and kwargs.get("metadata", None) is not None:
                self._parse_metadata(kwargs.get("metadata", None))
            run_id = kwargs.get("run_id", None)
            # Creating new span for the chain node
            name = kwargs.get("name", None)
            tags: Dict[str, str] = {}
            if (
                self.metadata is not None
                and self.metadata.get("chain_tags") is not None
            ):
                # its a dict
                for key, value in self.metadata["chain_tags"].items():
                    tags[key] = str(value)
            tags["run_id"] = run_id
            if kwargs.get("parent_run_id", None) is not None:
                tags["parent_run_id"] = str(kwargs.get("parent_run_id", None))
            if kwargs.get("tags", None) is not None:
                for tag in kwargs["tags"]:
                    key, value = tag.split(":", 1)
                    tags[key.strip()] = value.strip()
            if kwargs.get("metadata", None) is not None:
                for key, value in kwargs["metadata"].items():
                    tags[key.strip()] = str(value)

            container = self.__get_container(run_id, kwargs.get("parent_run_id", None))
            if container is None:
                scribe().error("[MaximSDK] Couldn't find a container for chain")
                return

            if not container.is_created():
                container.create()

            span_config = SpanConfig(id=str(run_id), name=name, tags=tags)
            span = container.add_span(span_config)

            # Add inputs metadata to the span like JS tracer
            try:
                span.add_metadata({"inputs": inputs})
            except Exception:
                pass

            # Create a SpanContainer for this run_id to handle child operations
            # This mirrors the JS tracer behavior
            parent_trace_id = (
                container.id()
                if isinstance(container, TraceContainer)
                else container.parent()
            )
            span_container = SpanContainer(
                span_id=str(run_id),
                logger=self.logger,
                parent=parent_trace_id,
                mark_created=True,
            )
            self.container_manager.set_container(str(run_id), span_container)
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Error while processing chain_start %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_chain_end(self, outputs: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
        try:
            run_id = kwargs.get("run_id", None)
            parent_run_id = kwargs.get("parent_run_id", None)
            # We hide the hidden chains
            tags = {
                tag.split(":")[0]: tag.split(":")[1]
                for tag in kwargs.get("tags", [])
                if ":" in tag
            }
            # Get the container for this run_id directly (should be the SpanContainer we created)
            container = self.container_manager.get_container(str(run_id))
            if container is None:
                scribe().error("[MaximSDK] Couldn't find a container for chain")
                return
            container.add_tags(tags)
            # Attach outputs metadata similar to JS tracer
            try:
                container.add_metadata({"outputs": outputs})
            except Exception:
                pass
            container.end()
            self.container_manager.remove_run_id_mapping(str(run_id))

            # If this is a top-level chain (no parent), also end the root trace
            if parent_run_id is None:
                try:
                    parent_container = self.container_manager.pop_root_trace(
                        str(run_id)
                    )
                    if parent_container is not None:
                        parent_container.end()
                        if self.callback is not None:
                            self.callback(
                                "trace.ended",
                                {
                                    "trace_id": parent_container.id(),
                                    "trace_name": parent_container.name(),
                                },
                            )
                except Exception:
                    pass

        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to parse chain-end: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_chain_error(
        self, error: Union[Exception, BaseException, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        scribe().info("[MaximSDK] on_chain_error called")
        try:
            run_id = kwargs.get("run_id", None)
            parent_run_id = kwargs.get("parent_run_id", None)
            container = self.__get_container(run_id, parent_run_id)
            if container is None:
                scribe().error("[MaximSDK] Couldn't find a container for chain")
                return
            chain_error = parse_langchain_llm_error(error)
            container.add_error(
                {
                    "id": str(run_id),
                    "message": chain_error.message,
                    "type": chain_error.type,
                    "code": chain_error.code,
                }
            )
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to process chain-error: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            container = self.__get_container(run_id, kwargs.get("parent_run_id", None))
            if container is None:
                scribe().error("[MaximSDK] Couldn't find a container for event")
                return
            final_tags = {
                tag.split(":")[0]: tag.split(":")[1]
                for tag in (tags or [])
                if ":" in tag
            }
            container.add_event(str(run_id), name, final_tags)
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to process custom-event: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        scribe().warning("[MaximSDK] On agent action is not supported")

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        scribe().warning("[MaximSDK] On agent action is not supported")

    # Tool callbacks

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        try:
            run_id = kwargs.get("run_id", None)
            parent_run_id = kwargs.get("parent_run_id", None)
            name = serialized.get("name", None)
            description = serialized.get("description", None)
            container = self.__get_container(run_id, parent_run_id)
            if container is None:
                scribe().error(
                    "[MaximSDK] Couldn't find a container for this tool call"
                )
                return
            if not container.is_created():
                container.create()
            container.add_tool_call(
                ToolCallConfig(
                    id=str(run_id), name=name, description=description, args=input_str
                )
            )
            if container.parent() is None:
                self.container_manager.set_container(str(run_id), container)
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to parse tool-start: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        try:
            run_id = kwargs.get("run_id", None)
            parent_run_id = kwargs.get("parent_run_id", None)
            container = self.__get_container(run_id, parent_run_id)
            if container is None:
                scribe().error("[MaximSDK] Couldn't find a container for tool call")
                return
            if isinstance(output, ToolMessage):
                result = ""
                if output.content is not None:
                    if isinstance(output.content, str):
                        result = output.content
                    else:
                        result = json.dumps(output.content)
                self.logger.tool_call_result(str(run_id), result)
            elif isinstance(output, dict):
                if output.get("status", None) == "success":
                    self.logger.tool_call_result(
                        str(run_id), output.get("content", None)
                    )
                elif output.get("status", None) == "error":
                    if isinstance(output.get("content", None), str):
                        self.logger.tool_call_error(
                            str(run_id),
                            ToolCallError(message=output.get("content", None)),
                        )
                    else:
                        self.logger.tool_call_error(
                            str(run_id),
                            ToolCallError(message=str(output.get("content", None))),
                        )
                elif output.get("content", None) is not None:
                    self.logger.tool_call_result(
                        str(run_id), output.get("content", None)
                    )

            if container.parent() is None:
                self.container_manager.remove_run_id_mapping(str(run_id))
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to parse tool-end: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_tool_error(
        self, error: Union[Exception, BaseException, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        try:
            run_id = kwargs.get("run_id", None)
            container = self.__get_container(run_id, kwargs.get("parent_run_id", None))
            if container is None:
                scribe().error("[MaximSDK] Couldn't find a container for tool_call")
                return
            if error is not None and hasattr(error, "__str__"):
                self.logger.tool_call_error(
                    str(run_id), ToolCallError(message=str(error))
                )
            if container.parent() is None:
                self.container_manager.remove_run_id_mapping(str(run_id))
        except Exception as e:
            import traceback

            scribe().error(
                "[MaximSDK] Failed to parse tool-end: %s\n%s",
                e,
                traceback.format_exc(),
            )

    def on_text(self, text: str, **kwargs: Any) -> Any:
        scribe().warning("[MaximSDK] Text models are not supported by Maxim callback")
