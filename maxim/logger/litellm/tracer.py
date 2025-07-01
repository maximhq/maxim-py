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
        """
        This class represents a MaximLiteLLMTracer.

        Args:
            logger: The logger to use.
        """
        super().__init__()
        scribe().warning("[MaximSDK] Litellm support is in beta")
        self.logger = logger
        self.containers: Dict[str, Container] = {}

    def __get_container_from_metadata(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Container:
        """
        Get the container from the metadata.

        Args:
            metadata: The metadata to get the container from.

        Returns:
            The container.
        """
        if metadata is not None and metadata.get("trace_id") is not None:
            trace_id = metadata.get("trace_id")
            span_name = metadata.get("span_name")
            tags = metadata.get("span_tags")
            if trace_id is not None:
                # Here we will create a new span and send back that as container
                container = SpanContainer(
                    span_id=str(uuid4()),
                    logger=self.logger,
                    span_name=span_name,
                    parent=trace_id,
                )
                container.create()
                if tags is not None:
                    container.add_tags(tags)
                return container
            # We will be creating trace from scratch
            tags = metadata.get("trace_tags")
            trace_name = metadata.get("trace_name")
            session_id = metadata.get("session_id")
            container = TraceContainer(
                trace_id=str(uuid4()),
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

    def _extract_input_from_messages(self, messages: Any) -> Optional[str]:
        """
        Extract text input from messages for logging purposes.
        Note: Only processes messages with role 'user' for input extraction.

        Args:
            messages: The messages to extract input from.

        Returns:
            The input text.
        """
        if messages is None:
            return None
        for message in messages:
            if message.get("role", "user") != "user":
                continue
            content = message.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
        return None

    def log_pre_api_call(self, model, messages, kwargs):
        """
        Runs when a LLM call starts.
        """
        try:
            if kwargs.get("call_type") == "embedding":
                return
            metadata: Optional[Dict[str, Any]] = None
            generation_name = None
            tags = {}

            litellm_metadata = kwargs.get("litellm_params", {}).get("metadata") or {}
            if litellm_metadata:
                metadata = litellm_metadata.get("maxim")
                if metadata is not None:
                    generation_name = metadata.get("generation_name")
                    tags = metadata.get("generation_tags") or {}

            # checking if trace_id present in metadata
            container = self.__get_container_from_metadata(metadata)
            if not container.is_created():
                container.create()
            call_id = kwargs["litellm_call_id"]
            self.containers[call_id] = container

            # Extract provider and parameters
            litellm_params = kwargs.get("litellm_params", {})
            provider = litellm_params.get("custom_llm_provider")
            params: Dict[str, Any] = kwargs.get("optional_params", {})

            # Handle chat/completion calls
            request_messages: list[GenerationRequestMessage] = []
            input_text = self._extract_input_from_messages(messages)
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
            if input_text is not None:
                container.set_input(input_text)
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling pre_api_call for litellm: {str(e)}"
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call succeeds.
        """
        try:
            if kwargs.get("call_type") == "embedding":
                return
            call_id = kwargs["litellm_call_id"]
            container = self.containers.get(call_id)
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
            if kwargs.get("call_type") == "embedding":
                return
            call_id = kwargs["litellm_call_id"]
            container = self.containers.get(call_id)
            if container is None:
                # This means that this was an litellm level error
                container = self.__get_container_from_metadata(None)
                if not container.is_created():
                    container.create()
                model = kwargs.get("model")
                messages = kwargs.get("messages")
                provider = kwargs.get("custom_llm_provider")
                container.add_generation(
                    GenerationConfig(
                        id=call_id,
                        messages=messages,
                        model=model or "Unknown",
                        provider=provider or "Unknown",
                    )
                )
            exception = kwargs.get("exception")
            if exception is not None:
                self.logger.generation_error(
                    generation_id=call_id,
                    error=GenerationError(
                        message=getattr(exception, "message", str(exception)),
                        code=str(getattr(exception, "status_code", "unknown")),
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
            if kwargs.get("call_type") == "embedding":
                return
            metadata: Optional[Dict[str, Any]] = None
            generation_name = None
            tags = {}

            litellm_metadata = kwargs.get("litellm_params", {}).get("metadata") or {}
            if litellm_metadata:
                metadata = litellm_metadata.get("maxim")
                if metadata is not None:
                    generation_name = metadata.get("generation_name")
                    tags = metadata.get("generation_tags") or {}

            # checking if trace_id present in metadata
            container = self.__get_container_from_metadata(metadata)
            if not container.is_created():
                container.create()
            call_id = kwargs["litellm_call_id"]
            self.containers[call_id] = container

            # Extract provider and parameters
            litellm_params = kwargs.get("litellm_params", {})
            provider = litellm_params.get("custom_llm_provider")
            params: Dict[str, Any] = kwargs.get("optional_params", {})

            # Handle chat/completion calls
            request_messages: list[GenerationRequestMessage] = []
            input_text = self._extract_input_from_messages(messages)
            for message in messages:
                request_messages.append(
                    GenerationRequestMessage(
                        role=message.get("role", "user"),
                        content=message.get("content", ""),
                    ),
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
            if input_text is not None:
                container.set_input(input_text)
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling async_log_pre_api_call for litellm: {str(e)}"
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call succeeds.
        """
        try:
            if kwargs.get("call_type") == "embedding":
                return
            call_id = kwargs["litellm_call_id"]
            container = self.containers.get(call_id)
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
            if kwargs.get("call_type") == "embedding":
                return
            call_id = kwargs["litellm_call_id"]
            container = self.containers.get(call_id)
            if container is None:
                # This means that this was an litellm level error
                container = self.__get_container_from_metadata(None)
                if not container.is_created():
                    container.create()
                model = kwargs.get("model")
                messages = kwargs.get("messages")
                provider = kwargs.get("custom_llm_provider")
                container.add_generation(
                    GenerationConfig(
                        id=call_id,
                        messages=messages,
                        model=model or "Unknown",
                        provider=provider or "Unknown",
                    )
                )
            exception = kwargs.get("exception")
            if exception is not None:
                self.logger.generation_error(
                    generation_id=call_id,
                    error=GenerationError(
                        message=getattr(exception, "message", str(exception)),
                        code=str(getattr(exception, "status_code", "unknown")),
                    ),
                )
            container.end()
        except Exception as e:
            scribe().error(
                f"[MaximSDK] Error while handling async_log_failure_event for litellm: {str(e)}"
            )
