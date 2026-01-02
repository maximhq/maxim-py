"""Groq API instrumentation for Maxim logging.

This module provides instrumentation for the Groq SDK to integrate with Maxim's
logging and monitoring capabilities. It patches the Groq client methods to
automatically track API calls, model parameters, and responses.

The instrumentation supports both synchronous and asynchronous chat completions,
streaming responses, and various model parameters specific to Groq.
"""

import functools
import json
from typing import Any, Optional
from uuid import uuid4
from groq.resources.chat import Completions, AsyncCompletions
from .utils import GroqUtils
from .helpers import GroqHelpers
from ..logger import Generation, Logger, Trace, GenerationConfigDict
from ...scribe import scribe

_INSTRUMENTED = False
_GROQ_INFLIGHT_TOOL_CALLS: dict[str, Any] = {}


def instrument_groq(logger: Logger) -> None:
    """Patch Groq's chat completion methods for Maxim logging.

    This function instruments the Groq SDK by patching the chat completion
    methods to automatically log API calls, model parameters, and responses to
    Maxim. It supports both synchronous and asynchronous operations, streaming
    responses, and various Groq specific features.

    The instrumentation is designed to be non-intrusive and maintains the original
    API behavior while adding comprehensive logging capabilities.

    Args:
        logger (Logger): The Maxim logger instance to use for tracking and
            logging API interactions. This logger will be used to create
            traces and generations for each API call.
    """

    global _INSTRUMENTED
    if _INSTRUMENTED:
        scribe().debug("[MaximSDK] Groq already instrumented")
        return

    def _complete_groq_tool_results_from_messages(messages: Any) -> None:
        """
        Complete Maxim ToolCall entities using OpenAI-style `role: "tool"` messages.

        This mirrors the Anthropic pattern where we only set tool_call.result once
        the caller has actually sent a concrete tool result back to the model.
        """
        if messages is None:
            return
        if not isinstance(messages, list):
            return

        for msg in messages:
            role = None
            tool_call_id = None
            content = None
            if isinstance(msg, dict):
                role = msg.get("role")
                tool_call_id = msg.get("tool_call_id")
                content = msg.get("content")
            else:
                role = getattr(msg, "role", None)
                tool_call_id = getattr(msg, "tool_call_id", None)
                content = getattr(msg, "content", None)

            if role != "tool":
                continue
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue

            if content is None:
                result_str = ""
            elif isinstance(content, str):
                result_str = content
            else:
                try:
                    result_str = json.dumps(content)
                except Exception:
                    result_str = str(content)

            tool_call = _GROQ_INFLIGHT_TOOL_CALLS.get(tool_call_id)
            if tool_call is None:
                continue

            try:
                tool_call.result(result_str)
                _GROQ_INFLIGHT_TOOL_CALLS.pop(tool_call_id, None)
            except Exception as e:  # pragma: no cover - defensive
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Failed to set tool_call result for id={tool_call_id}: {e}"
                )

    def wrap_sync_create(create_func):
        """Wrapper for synchronous chat completion create method.

        This wrapper function intercepts synchronous chat completion requests
        to Groq and adds comprehensive logging capabilities while
        preserving the original API behavior.

        Args:
            create_func: The original Groq chat completion create method
                to be wrapped with logging capabilities.

        Returns:
            Wrapped function that provides the same interface as the original
            but with added Maxim logging and monitoring.
        """

        @functools.wraps(create_func)
        def wrapper(self: Completions, *args: Any, **kwargs: Any):
            # Extract Maxim-specific headers for trace and generation configuration
            extra_headers = kwargs.get("extra_headers", None)
            trace_id = None
            generation_name = None
            generation_tags = None
            trace_tags = None

            if extra_headers is not None:
                trace_id = extra_headers.get("x-maxim-trace-id", None)
                generation_name = extra_headers.get("x-maxim-generation-name", None)
                generation_tags = extra_headers.get("x-maxim-generation-tags", None)
                trace_tags = extra_headers.get("x-maxim-trace-tags", None)

            # Determine if we need to create a new trace or use existing one
            is_local_trace = trace_id is None
            model = kwargs.get("model", None)
            final_trace_id = trace_id or str(uuid4())
            generation: Optional[Generation] = None
            trace: Optional[Trace] = None
            messages = kwargs.get("messages", None)
            is_streaming = kwargs.get("stream", False)

            _complete_groq_tool_results_from_messages(messages or [])

            # Initialize trace and generation for logging
            try:
                trace = logger.trace({"id": final_trace_id})
                gen_config = GenerationConfigDict(
                    id=str(uuid4()),
                    model=model or "",
                    provider="groq",
                    name=generation_name,
                    model_parameters=GroqUtils.get_model_params(**kwargs),
                    messages=GroqUtils.parse_message_param(messages or []),
                )
                generation = trace.generation(gen_config)

                # Check for image URLs in messages and add as attachments
                GroqUtils.add_image_attachments_from_messages(
                    generation, messages or []
                )

            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error in generating content: {e}",
                )

            try:
                # Call the original Groq API method
                response = create_func(self, *args, **kwargs)
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                # We will raise the error back to the caller and not handle it here
                raise

            # Process response and log results
            try:
                if generation is not None:
                    if is_streaming:
                        response = GroqHelpers.sync_stream_helper(
                            response, generation, trace, is_local_trace
                        )
                    else:
                        # Non-streaming: log generation result and any tool calls present.
                        parsed = GroqUtils.parse_completion(response)
                        generation.result(parsed)

                        # Create ToolCall entities on the trace when tools are used.
                        try:
                            if trace is not None:
                                tool_calls = (
                                    GroqUtils.extract_tool_calls_from_completion(
                                        response
                                    )
                                )
                                for tc in tool_calls:
                                    # tc can be ChatCompletionMessageToolCall or dict
                                    fn = (
                                        getattr(tc, "function", None)
                                        if hasattr(tc, "function")
                                        else (
                                            tc.get("function")
                                            if isinstance(tc, dict)
                                            else None
                                        )
                                    )
                                    tool_call_id = (
                                        getattr(tc, "id", None)
                                        if hasattr(tc, "id")
                                        else (
                                            tc.get("id")
                                            if isinstance(tc, dict)
                                            else None
                                        )
                                    )
                                    if (
                                        not isinstance(tool_call_id, str)
                                        or not tool_call_id
                                    ):
                                        tool_call_id = str(uuid4())
                                    tool_name = (
                                        getattr(fn, "name", "unknown")
                                        if fn is not None and hasattr(fn, "name")
                                        else (
                                            fn.get("name", "unknown")
                                            if isinstance(fn, dict)
                                            else "unknown"
                                        )
                                    )
                                    tool_args = (
                                        getattr(fn, "arguments", "")
                                        if fn is not None and hasattr(fn, "arguments")
                                        else (
                                            fn.get("arguments", "")
                                            if isinstance(fn, dict)
                                            else ""
                                        )
                                    )

                                    tc_entity = trace.tool_call(
                                        {
                                            "id": tool_call_id,
                                            "name": tool_name,
                                            "description": "Groq tool call",
                                            "args": tool_args,
                                        }
                                    )
                                    _GROQ_INFLIGHT_TOOL_CALLS[tool_call_id] = tc_entity
                        except Exception as e:
                            scribe().warning(
                                f"[MaximSDK][GroqInstrumentation] Error creating tool_call entities: {e}"
                            )

                        if is_local_trace and trace is not None:
                            if response.choices and len(response.choices) > 0:
                                trace.set_output(
                                    response.choices[0].message.content or ""
                                )
                            else:
                                trace.set_output("")
                            trace.end()
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error in logging generation: {e}"
                )

            # Apply tags if provided
            if generation_tags is not None and generation is not None:
                for key, value in generation_tags.items():
                    generation.add_tag(key, value)
            if trace_tags is not None and trace is not None:
                for key, value in trace_tags.items():
                    trace.add_tag(key, value)

            return response

        return wrapper

    def wrap_async_create(create_func):
        """Wrapper for asynchronous chat completion create method.

        This wrapper function intercepts asynchronous chat completion requests
        to Groq and adds comprehensive logging capabilities while
        preserving the original API behavior and async semantics.

        Args:
            create_func: The original Groq async chat completion create method
                to be wrapped with logging capabilities.

        Returns:
            Wrapped async function that provides the same interface as the original
            but with added Maxim logging and monitoring.
        """

        @functools.wraps(create_func)
        async def wrapper(self: AsyncCompletions, *args: Any, **kwargs: Any):
            # Extract Maxim-specific headers for trace and generation configuration
            extra_headers = kwargs.get("extra_headers", None)
            trace_id = None
            generation_name = None
            generation_tags = None
            trace_tags = None

            if extra_headers is not None:
                trace_id = extra_headers.get("x-maxim-trace-id", None)
                generation_name = extra_headers.get("x-maxim-generation-name", None)
                generation_tags = extra_headers.get("x-maxim-generation-tags", None)
                trace_tags = extra_headers.get("x-maxim-trace-tags", None)

            # Determine if we need to create a new trace or use existing one
            is_local_trace = trace_id is None
            model = kwargs.get("model", None)
            final_trace_id = trace_id or str(uuid4())
            generation: Optional[Generation] = None
            trace: Optional[Trace] = None
            messages = kwargs.get("messages", None)
            is_streaming = kwargs.get("stream", False)

            # Before creating a new generation, complete any prior tool calls whose
            # results are being sent in this request.
            _complete_groq_tool_results_from_messages(messages or [])

            # Initialize trace and generation for logging
            try:
                trace = logger.trace({"id": final_trace_id})
                gen_config = GenerationConfigDict(
                    id=str(uuid4()),
                    model=model or "",
                    provider="groq",
                    name=generation_name,
                    model_parameters=GroqUtils.get_model_params(**kwargs),
                    messages=GroqUtils.parse_message_param(messages or []),
                )
                generation = trace.generation(gen_config)

                # Check for image URLs in messages and add as attachments
                GroqUtils.add_image_attachments_from_messages(
                    generation, messages or []
                )

            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error in generating content: {e}",
                )

            try:
                # Call the original Groq API method
                response = await create_func(self, *args, **kwargs)
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                # We will raise the error back to the caller and not handle it here
                raise

            # Process response and log results
            try:
                if generation is not None:
                    if is_streaming:
                        response = GroqHelpers.async_stream_helper(
                            response, generation, trace, is_local_trace
                        )
                    else:
                        # Non-streaming: log generation result and any tool calls present.
                        parsed = GroqUtils.parse_completion(response)
                        generation.result(parsed)

                        # Create ToolCall entities on the trace when tools are used.
                        try:
                            if trace is not None:
                                tool_calls = (
                                    GroqUtils.extract_tool_calls_from_completion(
                                        response
                                    )
                                )
                                for tc in tool_calls:
                                    fn = (
                                        getattr(tc, "function", None)
                                        if hasattr(tc, "function")
                                        else (
                                            tc.get("function")
                                            if isinstance(tc, dict)
                                            else None
                                        )
                                    )
                                    tool_call_id = (
                                        getattr(tc, "id", None)
                                        if hasattr(tc, "id")
                                        else (
                                            tc.get("id")
                                            if isinstance(tc, dict)
                                            else None
                                        )
                                    )
                                    if (
                                        not isinstance(tool_call_id, str)
                                        or not tool_call_id
                                    ):
                                        tool_call_id = str(uuid4())
                                    tool_name = (
                                        getattr(fn, "name", "unknown")
                                        if fn is not None and hasattr(fn, "name")
                                        else (
                                            fn.get("name", "unknown")
                                            if isinstance(fn, dict)
                                            else "unknown"
                                        )
                                    )
                                    tool_args = (
                                        getattr(fn, "arguments", "")
                                        if fn is not None and hasattr(fn, "arguments")
                                        else (
                                            fn.get("arguments", "")
                                            if isinstance(fn, dict)
                                            else ""
                                        )
                                    )

                                    tc_entity = trace.tool_call(
                                        {
                                            "id": tool_call_id,
                                            "name": tool_name,
                                            "description": "Groq tool call",
                                            "args": tool_args,
                                        }
                                    )
                                    _GROQ_INFLIGHT_TOOL_CALLS[tool_call_id] = tc_entity
                        except Exception as e:
                            scribe().warning(
                                f"[MaximSDK][GroqInstrumentation] Error creating tool_call entities: {e}"
                            )

                        if is_local_trace and trace is not None:
                            if response.choices and len(response.choices) > 0:
                                trace.set_output(
                                    response.choices[0].message.content or ""
                                )
                            else:
                                trace.set_output("")
                            trace.end()
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error in logging generation: {e}",
                )

            # Apply tags if provided
            if generation_tags is not None and generation is not None:
                for key, value in generation_tags.items():
                    generation.add_tag(key, value)
            if trace_tags is not None and trace is not None:
                for key, value in trace_tags.items():
                    trace.add_tag(key, value)

            return response

        return wrapper

    # Apply the patches to both sync and async chat completion methods
    setattr(Completions, "create", wrap_sync_create(Completions.create))
    setattr(AsyncCompletions, "create", wrap_async_create(AsyncCompletions.create))
    _INSTRUMENTED = True
