"""Fireworks API instrumentation for Maxim logging.

This module provides instrumentation for the Fireworks Build SDK to integrate with Maxim's
logging and monitoring capabilities. It patches the Fireworks Build SDK client methods to
automatically track API calls, model parameters, and responses.

The instrumentation supports both synchronous and asynchronous chat completions,
streaming responses, and various model parameters specific to Fireworks AI.
"""

import functools
import json
from uuid import uuid4
from typing import Any, Optional

try:
    from fireworks.llm.llm import ChatCompletion  # For older versions
except ImportError:
    from fireworks.llm.LLM import ChatCompletion

from maxim.logger.components.generation import GenerationConfigDict
from ..logger import Logger, Generation, Trace
from ...scribe import scribe
from .utils import FireworksUtils
from .helpers import FireworksHelpers

_INSTRUMENTED = False
_INFLIGHT_TOOL_CALLS: dict[str, Any] = {}


def instrument_fireworks(logger: Logger) -> None:
    """Patch Fireworks's chat completion methods for Maxim logging.

    This function instruments the Fireworks SDK by patching the chat completion
    methods to automatically log API calls, model parameters, and responses to
    Maxim. It supports both synchronous and asynchronous operations, streaming
    responses, and various Fireworks specific features.
    """

    global _INSTRUMENTED
    if _INSTRUMENTED:
        scribe().info("[MaximSDK] Fireworks already instrumented")
        return

    def _complete_fireworks_tool_results_from_messages(messages: Any) -> None:
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
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "tool":
                continue

            tool_call_id = msg.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue

            raw_content = msg.get("content")
            if raw_content is None:
                result_str = ""
            elif isinstance(raw_content, str):
                result_str = raw_content
            else:
                try:
                    result_str = json.dumps(raw_content)
                except Exception:
                    result_str = str(raw_content)

            tool_call = _INFLIGHT_TOOL_CALLS.get(tool_call_id)
            if tool_call is None:
                continue

            try:
                tool_call.result(result_str)
                _INFLIGHT_TOOL_CALLS.pop(tool_call_id, None)
            except Exception as e:  # pragma: no cover - defensive
                scribe().warning(
                    "[MaximSDK][FireworksInstrumentation] "
                    f"Failed to set tool_call result for id={tool_call_id}: {e}",
                )

    def _log_fireworks_tool_calls(
        trace: Optional[Trace],
        response: Any,
        tools: Any,
    ) -> None:
        """
        Extract tool calls from a Fireworks response and log them to the current trace.

        This mirrors the behavior implemented for Anthropic / Gemini / LiteLLM:
        - Look for OpenAI-style tool_calls on the assistant message
        - Use the provided tools schema (if any) to populate descriptions
        - Create Maxim ToolCall entities via trace.tool_call(...)
        """
        if trace is None or response is None:
            return

        try:
            # Fireworks ChatCompletionResponse has choices[0].message.tool_calls
            if not hasattr(response, "choices") or not response.choices:
                return

            first_choice = response.choices[0]
            message = getattr(first_choice, "message", None)
            if message is None:
                return

            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return

            # Build a mapping from tool name -> description from the tools param
            tool_descriptions: dict[str, str] = {}
            if tools:
                try:
                    for tool in tools:
                        name: Optional[str] = None
                        description: str = ""

                        if isinstance(tool, dict):
                            fn = tool.get("function") or {}
                            name = fn.get("name") or tool.get("name")
                            description = fn.get("description") or tool.get(
                                "description", ""
                            )
                        else:
                            fn = getattr(tool, "function", None)
                            if fn is not None:
                                name = getattr(fn, "name", None)
                                description = getattr(fn, "description", "") or ""
                            else:
                                name = getattr(tool, "name", None)
                                description = getattr(tool, "description", "") or ""

                        if name:
                            tool_descriptions[name] = description
                except Exception as e:  # pragma: no cover - defensive
                    scribe().warning(
                        "[MaximSDK][FireworksInstrumentation] "
                        f"Error building tool description map: {e}",
                    )

            # Create ToolCall entities on the trace for each tool_call
            for tool_call in tool_calls:
                fn = getattr(tool_call, "function", None)
                name = getattr(fn, "name", None) if fn is not None else None
                arguments = getattr(fn, "arguments", None) if fn is not None else None
                tool_call_id = getattr(tool_call, "id", None) or str(uuid4())

                description = ""
                if name and name in tool_descriptions:
                    description = tool_descriptions[name]

                try:
                    tc_entity = trace.tool_call(
                        {
                            "id": tool_call_id,
                            "name": name or "unknown",
                            "description": description,
                            "args": arguments,
                        }
                    )
                    _INFLIGHT_TOOL_CALLS[tool_call_id] = tc_entity
                except Exception as e:  # pragma: no cover - defensive
                    scribe().warning(
                        "[MaximSDK][FireworksInstrumentation] "
                        f"Error logging tool call to Maxim: {e}",
                    )

        except Exception as e:  # pragma: no cover - defensive
            scribe().warning(
                "[MaximSDK][FireworksInstrumentation] "
                f"Unexpected error while extracting tool calls: {e}",
            )

    def wrap_create(create_func):
        """Wrapper for synchronous chat completion create method.

        This wrapper function intercepts synchronous chat completion requests
        to Fireworks and adds comprehensive logging capabilities while
        preserving the original API behavior.
        """

        @functools.wraps(create_func)
        def wrapper(self: ChatCompletion, *args: Any, **kwargs: Any):
            extra_headers = kwargs.get("extra_headers", None)
            trace_id = None
            generation_name = None
            generation_tags = None
            trace_tags = None
            session_id = None

            if extra_headers is not None:
                trace_id = extra_headers.get("x-maxim-trace-id", None)
                generation_name = extra_headers.get("x-maxim-generation-name", None)
                generation_tags = extra_headers.get("x-maxim-generation-tags", None)
                trace_tags = extra_headers.get("x-maxim-trace-tags", None)
                session_id = extra_headers.get("x-maxim-session-id", None)

            # Determine if we need to create a new trace or use existing one
            is_local_trace = trace_id is None
            final_trace_id = trace_id or str(uuid4())
            generation: Optional[Generation] = None
            trace: Optional[Trace] = None
            messages = kwargs.get("messages", None)
            tools = kwargs.get("tools", None)
            is_streaming = kwargs.get("stream", False)

            # Try to get model from self (ChatCompletion instance)
            model = "unknown"
            try:
                model_result = self._create_setup()
                if model_result is not None:
                    model = model_result.split("/")[-1]
            except Exception:
                # If private method fails, fall back to "unknown"
                model = "unknown"

            try:
                trace_config: dict[str, Any] = {"id": final_trace_id}
                if session_id is not None:
                    trace_config["session_id"] = session_id
                trace = logger.trace(trace_config)
                gen_config = GenerationConfigDict(
                    id=str(uuid4()),
                    model=FireworksUtils.map_fireworks_model_name(model),
                    provider="fireworks",
                    name=generation_name,
                    model_parameters=FireworksUtils.get_model_params(**kwargs),
                    messages=FireworksUtils.parse_message_param(messages or []),
                )
                generation = trace.generation(gen_config)

                # Check for image URLs in messages and add as attachments
                FireworksUtils.add_image_attachments_from_messages(
                    generation, messages or []
                )
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in generating content: {e}",
                )

            # If this request includes tool role messages, complete prior tool calls
            _complete_fireworks_tool_results_from_messages(messages or [])

            # Call the original Fireworks API method
            # Not cleaning out the model name here in case the user sends one as it is a wrong method call
            # and should not be handled by the SDK (Fireworks does not have a model name property in the call
            # signature, directly while creating the LLM instance)
            clean_kwargs = {k: v for k, v in kwargs.items() if k != "extra_headers"}
            try:
                response = create_func(self, *args, **clean_kwargs)
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in generating content: {e}",
                )
                raise

            try:
                if generation is not None and trace is not None:
                    if is_streaming:
                        response = FireworksHelpers.sync_stream_helper(
                            response, generation, trace, is_local_trace
                        )
                    else:
                        _log_fireworks_tool_calls(trace, response, tools)
                        parsed = FireworksUtils.parse_completion(response)
                        generation.result(parsed)
                        if is_local_trace and trace is not None:
                            if response.choices and len(response.choices) > 0:
                                trace.set_output(
                                    response.choices[0].message.content or ""
                                )
                            else:
                                trace.set_output("")
                            trace.end()
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in logging generation: {e}",
                )
                if generation is not None:
                    generation.error({"message": str(e)})

            # Apply tags if provided
            FireworksUtils.apply_tags(generation, trace, generation_tags, trace_tags)

            return response

        return wrapper

    def wrap_acreate(acreate_func):
        """Wrapper for asynchronous chat completion create method.

        This wrapper function intercepts asynchronous chat completion requests
        to Fireworks and adds comprehensive logging capabilities while
        preserving the original API behavior.
        """

        @functools.wraps(acreate_func)
        async def wrapper(self: ChatCompletion, *args: Any, **kwargs: Any):
            # Extract Maxim-specific headers for trace and generation configuration
            extra_headers = kwargs.get("extra_headers", None)
            trace_id = None
            generation_name = None
            generation_tags = None
            trace_tags = None
            session_id = None

            if extra_headers is not None:
                trace_id = extra_headers.get("x-maxim-trace-id", None)
                generation_name = extra_headers.get("x-maxim-generation-name", None)
                generation_tags = extra_headers.get("x-maxim-generation-tags", None)
                trace_tags = extra_headers.get("x-maxim-trace-tags", None)
                session_id = extra_headers.get("x-maxim-session-id", None)

            # Determine if we need to create a new trace or use existing one
            is_local_trace = trace_id is None
            final_trace_id = trace_id or str(uuid4())
            generation: Optional[Generation] = None
            trace: Optional[Trace] = None
            messages = kwargs.get("messages", None)
            tools = kwargs.get("tools", None)
            is_streaming = kwargs.get("stream", False)

            # Try to get model from self (ChatCompletion instance)
            model = "unknown"
            try:
                model_result = self._create_setup()
                if model_result is not None:
                    model = model_result.split("/")[-1]
            except Exception:
                # If private method fails, fall back to "unknown"
                model = "unknown"

            # Initialize trace and generation for logging
            try:
                trace_config: dict[str, Any] = {"id": final_trace_id}
                if session_id is not None:
                    trace_config["session_id"] = session_id
                trace = logger.trace(trace_config)
                gen_config = GenerationConfigDict(
                    id=str(uuid4()),
                    model=FireworksUtils.map_fireworks_model_name(model),
                    provider="fireworks",
                    name=generation_name,
                    model_parameters=FireworksUtils.get_model_params(**kwargs),
                    messages=FireworksUtils.parse_message_param(messages or []),
                )
                generation = trace.generation(gen_config)

                # Check for image URLs in messages and add as attachments
                FireworksUtils.add_image_attachments_from_messages(
                    generation, messages or []
                )

            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in generating content: {e}",
                )
                if generation is not None:
                    generation.error({"message": str(e)})

            # If this request includes tool role messages, complete prior tool calls
            _complete_fireworks_tool_results_from_messages(messages or [])

            # Call the actual async Fireworks completion
            clean_kwargs = {k: v for k, v in kwargs.items() if k != "extra_headers"}
            try:
                response = await acreate_func(self, *args, **clean_kwargs)
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in generating content: {e}",
                )
                raise

            # Process response and log results
            try:
                if generation is not None and trace is not None:
                    if is_streaming:
                        response = FireworksHelpers.async_stream_helper(
                            response, generation, trace, is_local_trace
                        )
                    else:
                        _log_fireworks_tool_calls(trace, response, tools)
                        parsed = FireworksUtils.parse_completion(response)
                        generation.result(parsed)
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
                    f"[MaximSDK][FireworksInstrumentation] Error in logging generation: {e}",
                )

            # Apply tags if provided
            FireworksUtils.apply_tags(generation, trace, generation_tags, trace_tags)

            return response

        return wrapper

    # Patch the create and acreate methods
    setattr(ChatCompletion, "create", wrap_create(ChatCompletion.create))
    setattr(ChatCompletion, "acreate", wrap_acreate(ChatCompletion.acreate))

    _INSTRUMENTED = True
