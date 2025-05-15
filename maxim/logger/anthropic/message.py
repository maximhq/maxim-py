from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union
from uuid import uuid4

import httpx
from anthropic import Anthropic, MessageStreamEvent
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
    def __init__(self, client: Anthropic, logger: Logger):
        super().__init__(client)
        self._logger = logger

    def create_non_stream(self, *args, **kwargs) -> Any:
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
                    trace.set_output(str(response.content))
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][AnthropicClient] Error in logging generation: {str(e)}"
            )

        return response

    def create_stream(self, *args, **kwargs) -> Any:
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
        stream_wrapper = StreamWrapper(response, lambda chunk: process_chunk(chunk))

        def process_chunk(chunk: MessageStreamEvent):
            try:
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
            if is_local_trace and trace is not None:
                trace.end()

        return stream_wrapper

    def create(
        self,
        *args,
        max_tokens: int,
        messages: Iterable[MessageParam],
        model: ModelParam,
        metadata: MetadataParam | NotGiven = NOT_GIVEN,
        stop_sequences: List[str] | NotGiven = NOT_GIVEN,
        system: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoiceParam | NotGiven = NOT_GIVEN,
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
        return self.create_stream(*args, **kwargs)
