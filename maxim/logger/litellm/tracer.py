from typing import Any, Dict, Optional
from uuid import uuid4

from litellm.integrations.custom_logger import CustomLogger

from maxim.logger.components.generation import GenerationRequestMessage

from ...scribe import scribe
from ..logger import GenerationConfig, GenerationError, Logger
from ..models import Container, SpanContainer, TraceContainer


class MaximLiteLLMTracer(CustomLogger):
    """
    Custom logger for Litellm.
    """

    def __init__(self, logger: Logger):
        super().__init__()
        scribe().warning("[MaximSDK] Litellm support is in beta")
        self.logger = logger
        self.containers: Dict[str, Container] = {}

    def __get_container_from_metadata(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Container:
        if metadata is not None and metadata["trace_id"] is not None:
            trace_id = metadata["trace_id"] if "trace_id" in metadata else None
            span_name = metadata["span_name"] if "span_name" in metadata else None
            tags = metadata["span_tags"] if "span_tags" in metadata else None
            if trace_id is not None:
                # Here we will create a new span and send back that as container
                container = SpanContainer(
                    span_id=str(str(uuid4())),
                    logger=self.logger,
                    span_name=span_name,
                    parent=trace_id,
                )
                container.create()
                if tags is not None:
                    container.add_tags(tags)
                return container
            # We will be creating trace from scratch
            tags = metadata["trace_tags"] if "trace_tags" in metadata else None
            trace_name = metadata["trace_name"] if "trace_name" in metadata else None
            session_id = metadata["session_id"] if "session_id" in metadata else None
            container = TraceContainer(
                trace_id=str(str(uuid4())),
                logger=self.logger,
                trace_name=trace_name,
                parent=session_id,
            )
            container.create()
            if tags is not None:
                container.add_tags(tags)
            return container

        return TraceContainer(
            trace_id=str(uuid4()), logger=self.logger, trace_name="LiteLLM"
        )

    def log_pre_api_call(self, model, messages, kwargs):
        """
        Runs when a LLM call starts.
        """
        try:
            if kwargs.get("call_type", None) == "embedding":
                return
            metadata: Optional[dict[str, Any]] = None
            generation_name = None
            tags = {}
            litellm_metadata = kwargs["litellm_params"]["metadata"] or {}
            if litellm_metadata:
                metadata = litellm_metadata.get("maxim", None)
                if metadata is not None:
                    generation_name = (
                        metadata["generation_name"]
                        if "generation_name" in metadata
                        else None
                    )
                    tags = (
                        metadata["generation_tags"]
                        if "generation_tags" in metadata
                        else None
                    )

            # checking if trace_id present in metadata
            container = self.__get_container_from_metadata(metadata)
            if not container.is_created():
                container.create()
            call_id = kwargs["litellm_call_id"]
            self.containers[call_id] = container
            # starting trace
            provider = kwargs["litellm_params"]["custom_llm_provider"]
            params: Dict[str, Any] = (
                kwargs["optional_params"] if "optional_params" in kwargs else {}
            )
            request_messages: list[GenerationRequestMessage] = []
            for message in messages:
                request_messages.append(
                    GenerationRequestMessage(
                        role=message.get("role", "user"),
                        content=message.get("content", ""),
                    )
                )
            _ = container.add_generation(
                GenerationConfig(
                    id=call_id,
                    messages=request_messages,
                    model=model,
                    provider=provider,
                    name=generation_name,
                    tags=tags,
                    model_parameters=params,
                )
            )
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling pre_api_call for litellm: {str(e)}"
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call succeeds.
        """
        try:
            if kwargs.get("call_type", None) == "embedding":
                return
            call_id = kwargs["litellm_call_id"]
            container = self.containers[call_id] if call_id in self.containers else None
            if container is None:
                scribe().warning(
                    "[MaximSDK] Couldn't find container for logging Litellm post call."
                )
                return
            self.logger.generation_result(call_id, result=response_obj)
            container.end()
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling log_success_event for litellm: {str(e)}"
            )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call fails.
        """
        try:
            if kwargs.get("call_type", None) == "embedding":
                return
            call_id = kwargs["litellm_call_id"]
            container = self.containers[call_id] if call_id in self.containers else None
            if container is None:
                # This means that this was an litellm level error
                container = self.__get_container_from_metadata(None)
                if not container.is_created():
                    container.create()
                model = kwargs["model"] if "model" in kwargs else None
                messages = kwargs["messages"] if "messages" in kwargs else None
                provider = (
                    kwargs["custom_llm_provider"]
                    if "custom_llm_providervider" in kwargs
                    else None
                )
                container.add_generation(
                    GenerationConfig(
                        id=call_id,
                        messages=messages,
                        model=model or "Unknown",
                        provider=provider or "Unknown",
                    )
                )
            exception = kwargs["exception"] or None
            if exception is not None:
                self.logger.generation_error(
                    generation_id=call_id,
                    error=GenerationError(
                        message=exception.message,
                        code=str(exception.status_code),
                    ),
                )
            container.end()
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling log_failure_event for litellm {str(e)}"
            )

    async def async_log_pre_api_call(self, model, messages, kwargs):
        """
        Runs when a LLM call starts.
        """
        try:
            if kwargs.get("call_type", None) == "embedding":
                return
            metadata: Optional[dict[str, Any]] = None
            generation_name = None
            tags = {}
            litellm_metadata = kwargs["litellm_params"]["metadata"] or {}
            if litellm_metadata:
                metadata = litellm_metadata.get("maxim", None)
                if metadata is not None:
                    generation_name = metadata["generation_name"]
                    tags = metadata["generation_tags"]

            # checking if trace_id present in metadata
            container = self.__get_container_from_metadata(metadata)
            if not container.is_created():
                container.create()
            call_id = kwargs["litellm_call_id"]
            self.containers[call_id] = container
            # starting trace
            request_messages: list[GenerationRequestMessage] = []
            for message in messages:
                request_messages.append(
                    GenerationRequestMessage(
                        role=message.get("role", "user"),
                        content=message.get("content", ""),
                    )
                )
            provider = kwargs["litellm_params"]["custom_llm_provider"]
            params: Dict[str, Any] = (
                kwargs["optional_params"] if "optional_params" in kwargs else {}
            )
            _ = container.add_generation(
                GenerationConfig(
                    id=call_id,
                    messages=request_messages,
                    model=model,
                    provider=provider,
                    model_parameters=params,
                    name=generation_name,
                    tags=tags,
                )
            )
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling async_log_pre_api_call for litellm: {str(e)}"
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call succeeds.
        """
        try:
            if kwargs.get("call_type", None) == "embedding":
                return
            call_id = kwargs["litellm_call_id"]
            container = self.containers[call_id] if call_id in self.containers else None
            if container is None:
                scribe().warning(
                    "[MaximSDK] Couldn't find container for logging Litellm post call."
                )
                return
            self.logger.generation_result(call_id, result=response_obj)
            container.end()
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling async_log_success_event for litellm: {str(e)}"
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call fails.
        """
        try:
            if kwargs.get("call_type", None) == "embedding":
                return
            call_id = kwargs["litellm_call_id"]
            container = self.containers[call_id] if call_id in self.containers else None
            if container is None:
                # This means that this was an litellm level error
                container = self.__get_container_from_metadata(None)
                if not container.is_created():
                    container.create()
                model = kwargs["model"] if "model" in kwargs else None
                messages = kwargs["messages"] if "messages" in kwargs else None
                provider = (
                    kwargs["custom_llm_provider"]
                    if "custom_llm_provider" in kwargs
                    else None
                )
                container.add_generation(
                    GenerationConfig(
                        id=call_id,
                        messages=messages,
                        model=model or "Unknown",
                        provider=provider or "Unknown",
                    )
                )
            exception = kwargs["exception"] or None
            if exception is not None:
                self.logger.generation_error(
                    generation_id=call_id,
                    error=GenerationError(
                        message=exception.message,
                        code=str(exception.status_code),
                    ),
                )
            container.end()
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling async_log_failure_event for litellm: {str(e)}"
            )
