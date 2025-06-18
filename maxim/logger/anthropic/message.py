from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union
from uuid import uuid4

import httpx
from anthropic import Anthropic, MessageStopEvent, MessageStreamEvent
from anthropic._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from anthropic.resources.messages import Messages
from anthropic.types.message_param import MessageParam
from anthropic.types.metadata_param import MetadataParam
from anthropic.types.model_param import ModelParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.tool_choice_param import ToolChoiceParam
from anthropic.types.tool_param import ToolParam

from ...scribe import scribe
from ..logger import (
    Generation,
    GenerationConfig,
    Logger,
    Trace,
    TraceConfig,
)
from .stream_manager import StreamWrapper
from .utils import AnthropicUtils


class MaximAnthropicMessages(Messages):
    """Maxim-enhanced Anthropic Messages client.

    This class extends the Anthropic Messages resource to integrate with Maxim's
    logging and monitoring capabilities. It automatically tracks message creation,
    both streaming and non-streaming, and logs them through the Maxim platform.

    The class handles trace management, generation logging, and error handling
    while maintaining compatibility with the original Anthropic Messages API.

    Attributes:
        _logger (Logger): The Maxim logger instance for tracking interactions.
    """

    def __init__(self, client: Anthropic, logger: Logger):
        """Initialize the Maxim Anthropic Messages client.

        Args:
            client (Anthropic): The Anthropic client instance.
            logger (Logger): The Maxim logger instance for tracking and
                logging message interactions.
        """
        super().__init__(client)
        self._logger = logger

    def create_non_stream(self, *args, **kwargs) -> Any:
        """Create a non-streaming message with Maxim logging.

        This method handles non-streaming message creation while automatically
        logging the interaction through Maxim. It manages trace creation,
        generation tracking, and error handling.

        Args:
            *args: Variable length argument list passed to the parent create method.
            **kwargs: Arbitrary keyword arguments passed to the parent create method.
                Special headers:
                - x-maxim-trace-id: Optional trace ID for associating with existing trace.
                - x-maxim-generation-name: Optional name for the generation.

        Returns:
            Any: The response from the Anthropic API create method.

        Note:
            If logging fails, the method will still return the API response
            but will log a warning message.
        """
        extra_headers = kwargs.get("extra_headers", None)
        trace_id = None
        generation_name = None
        if extra_headers is not None:
            trace_id = extra_headers.get("x-maxim-trace-id", None)
            generation_name = extra_headers.get("x-maxim-generation-name", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None

        try:
            openai_style_messages = None
            system = kwargs.get("system", None)
            messages = kwargs.get("messages", None)
            model = kwargs.get("model", None)
            if system is not None:
                openai_style_messages = [{"role": "system", "content": system}] + list(
                    messages
                )
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="anthropic",
                name=generation_name,
                model_parameters=AnthropicUtils.get_model_params(
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["messages", "system"]
                    }
                ),
                messages=AnthropicUtils.parse_message_param(
                    openai_style_messages
                    if openai_style_messages is not None
                    else messages  # type:ignore
                ),
            )
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][AnthropicClient] Error in generating content: {str(e)}"
            )

        response = super().create(*args, **kwargs)

        try:
            if generation is not None:
                generation.result(response)
            if is_local_trace and trace is not None:
                if response is not None:
                    trace.set_output(str(response.content)) # type: ignore
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][AnthropicClient] Error in logging generation: {str(e)}"
            )

        return response

    def create_stream(self, *args, **kwargs) -> Any:
        """Create a streaming message with Maxim logging.

        This method handles streaming message creation while automatically
        logging the interaction through Maxim. It manages trace creation,
        generation tracking, and processes streaming events.

        Args:
            *args: Variable length argument list passed to the parent stream method.
            **kwargs: Arbitrary keyword arguments passed to the parent stream method.
                Special headers:
                - x-maxim-trace-id: Optional trace ID for associating with existing trace.
                - x-maxim-generation-name: Optional name for the generation.

        Returns:
            StreamWrapper: A wrapped stream manager that processes chunks and
                handles logging of streaming events.

        Note:
            The method returns a StreamWrapper that automatically processes
            stream chunks and logs the final result when the stream ends.
        """
        extra_headers = kwargs.get("extra_headers", None)
        trace_id = None
        generation_name = None
        if extra_headers is not None:
            trace_id = extra_headers.get("x-maxim-trace-id", None)
            generation_name = extra_headers.get("x-maxim-generation-name", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None

        try:
            openai_style_messages = None
            system = kwargs.get("system", None)
            messages = kwargs.get("messages", None)
            model = kwargs.get("model", None)
            if system is not None:
                openai_style_messages = [{"role": "system", "content": system}] + list(
                    messages
                )
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="anthropic",
                name=generation_name,
                model_parameters=AnthropicUtils.get_model_params(
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["messages", "system"]
                    }
                ),
                messages=AnthropicUtils.parse_message_param(
                    openai_style_messages
                    if openai_style_messages is not None
                    else messages  # type:ignore
                ),
            )
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][AnthropicClient] Error in generating content: {str(e)}"
            )

        response = super().stream(*args, **kwargs)

        def process_chunk(chunk: MessageStreamEvent):
            """Process individual stream chunks for logging.

            Args:
                chunk (MessageStreamEvent): Individual event from the message stream.
            """
            try:
                if isinstance(chunk, MessageStopEvent):
                    if chunk.type != "message_stop":
                        return
                    message = chunk.message
                    scribe().info(f"final_chunk: {message}")
                    if generation is not None and message is not None:
                        scribe().info(f"final_chunk: {message}")
                        generation.result(message)
                        if is_local_trace and trace is not None:
                            trace.end()
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][AnthropicClient] Error in background stream listener: {str(e)}"
                )

        return StreamWrapper(response, process_chunk)

    def create(
        self,
        *args,
        max_tokens: int,
        messages: Iterable[MessageParam],
        model: str,
        metadata: MetadataParam | NotGiven = NOT_GIVEN,
        stop_sequences: List[str] | NotGiven = NOT_GIVEN,
        system: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        tool_choice: dict | NotGiven = NOT_GIVEN,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> Any:
        """Create a message with automatic streaming detection and Maxim logging.

        This is the main entry point for message creation. It automatically
        detects whether streaming is requested and routes to the appropriate
        handler while ensuring all interactions are logged through Maxim.

        Args:
            max_tokens (int): The maximum number of tokens to generate.
            messages (Iterable[MessageParam]): The conversation messages.
            model (str): The model to use for generation.
            metadata (MetadataParam | NotGiven): Additional metadata for the request.
            stop_sequences (List[str] | NotGiven): Sequences that will stop generation.
            system (Union[str, Iterable[TextBlockParam]] | NotGiven): System message.
            temperature (float | NotGiven): Sampling temperature (0-1).
            tool_choice (dict | NotGiven): How the model should use tools.
            tools (Iterable[ToolParam] | NotGiven): Available tools for the model.
            top_k (int | NotGiven): Top-k sampling parameter.
            top_p (float | NotGiven): Top-p (nucleus) sampling parameter.
            extra_headers (Headers | None): Additional HTTP headers.
            extra_query (Query | None): Additional query parameters.
            extra_body (Body | None): Additional request body data.
            timeout (float | httpx.Timeout | None | NotGiven): Request timeout.
            **kwargs: Additional arguments, including 'stream' for streaming mode.

        Returns:
            Any: Either a direct message response or a StreamWrapper for streaming.

        Note:
            The method automatically detects streaming mode via the 'stream' parameter
            in kwargs and routes accordingly.
        """
        stream = kwargs.get("stream", False)
        # Add all parameters back to kwargs
        kwargs["max_tokens"] = max_tokens
        kwargs["messages"] = messages
        kwargs["model"] = model

        if metadata is not NOT_GIVEN:
            kwargs["metadata"] = metadata
        if stop_sequences is not NOT_GIVEN:
            kwargs["stop_sequences"] = stop_sequences
        if system is not NOT_GIVEN:
            kwargs["system"] = system
        if temperature is not NOT_GIVEN:
            kwargs["temperature"] = temperature
        if tool_choice is not NOT_GIVEN:
            kwargs["tool_choice"] = tool_choice
        if tools is not NOT_GIVEN:
            kwargs["tools"] = tools
        if top_k is not NOT_GIVEN:
            kwargs["top_k"] = top_k
        if top_p is not NOT_GIVEN:
            kwargs["top_p"] = top_p

        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers
        if extra_query is not None:
            kwargs["extra_query"] = extra_query
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        if timeout is not NOT_GIVEN:
            kwargs["timeout"] = timeout

        if stream:
            return self.create_stream(*args, **kwargs)
        else:
            return self.create_non_stream(*args, **kwargs)

    def stream(self, *args, **kwargs) -> Any:
        """Create a streaming message with Maxim logging.

        This method is a direct alias for create_stream, providing compatibility
        with the standard Anthropic Messages API while adding Maxim logging.

        Args:
            *args: Variable length argument list passed to create_stream.
            **kwargs: Arbitrary keyword arguments passed to create_stream.

        Returns:
            StreamWrapper: A wrapped stream manager with logging capabilities.
        """
        return self.create_stream(*args, **kwargs)
