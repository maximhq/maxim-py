"""Maxim integration for Pydantic AI agent framework."""

import contextvars
import functools
import inspect
import logging
import traceback
import uuid
from time import time
from typing import Any, Optional, Union

try:
    from pydantic_ai import Agent
    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.tools import Tool
    from pydantic_ai.models import Model
    from pydantic_ai.run import AgentRun
    from pydantic_ai._agent_graph import ModelRequestNode, CallToolsNode

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None
    AbstractAgent = None
    Tool = None
    Model = None
    AgentRun = None
    ModelRequestNode = None
    CallToolsNode = None

from ...logger import (
    GenerationConfigDict,
    Logger,
    Retrieval,
    Session,
    SessionConfigDict,
    ToolCall,
    Trace,
)
from ..models import Metadata
from ...scribe import scribe
from .utils import (
    pydantic_ai_postprocess_inputs,
    dictify,
    extract_tool_details,
    get_agent_display_name,
    get_tool_display_name,
)

_last_llm_usages = {}
_session: Union[Session, None] = None  # Global Maxim session
_pydantic_metadata: Optional[Metadata] = None  # Optional user-provided Maxim metadata

_global_maxim_trace: contextvars.ContextVar[Union[Trace, None]] = (
    contextvars.ContextVar("maxim_trace_context_var", default=None)
)


def get_log_level(debug: bool) -> int:
    """Set logging level based on debug flag."""
    return logging.DEBUG if debug else logging.WARNING


class MaximEvalConfig:
    """Maxim eval config."""

    evaluators: list[str]
    additional_variables: list[dict[str, str]]

    def __init__(self):
        self.evaluators = []
        self.additional_variables = []


def extract_usage_from_response(response) -> dict:
    """Extract usage information from various response types."""
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "input_tokens"):
                usage_info["prompt_tokens"] = getattr(usage, "input_tokens", 0)
            if hasattr(usage, "output_tokens"):
                usage_info["completion_tokens"] = getattr(usage, "output_tokens", 0)
            if hasattr(usage, "total_tokens"):
                usage_info["total_tokens"] = getattr(usage, "total_tokens", 0)
            elif usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                usage_info["total_tokens"] = (
                    usage_info["prompt_tokens"] + usage_info["completion_tokens"]
                )
        elif hasattr(response, "prompt_tokens"):
            # Direct attributes on response
            usage_info["prompt_tokens"] = getattr(response, "prompt_tokens", 0)
            usage_info["completion_tokens"] = getattr(response, "completion_tokens", 0)
            usage_info["total_tokens"] = getattr(response, "total_tokens", 0)
    except Exception as e:
        scribe().debug(f"[MaximSDK] Error extracting usage: {e}")

    return usage_info


def extract_model_info(model_obj) -> dict:
    """Extract model information from model object."""
    model_info = {"model": "unknown", "provider": "unknown"}

    try:
        if hasattr(model_obj, "model_name"):
            model_info["model"] = model_obj.model_name or "unknown"
        elif hasattr(model_obj, "model"):
            model_info["model"] = str(model_obj.model)

        if hasattr(model_obj, "system"):
            model_info["provider"] = model_obj.system or "unknown"
        elif "openai" in str(type(model_obj)).lower():
            model_info["provider"] = "openai"
        elif "anthropic" in str(type(model_obj)).lower():
            model_info["provider"] = "anthropic"
        elif hasattr(model_obj, "provider_name"):
            model_info["provider"] = model_obj.provider_name or "unknown"
    except Exception as e:
        scribe().debug(f"[MaximSDK] Error extracting model info: {e}")

    return model_info


def convert_messages_to_maxim_format(pydantic_messages) -> list:
    """Convert Pydantic AI messages to Maxim format."""
    messages = []
    try:
        for msg in pydantic_messages:
            if hasattr(msg, "parts"):
                # Convert ModelRequest to Maxim format
                maxim_msg = {"role": "user", "content": []}  # Default role

                # Determine role and extract content
                for part in msg.parts:
                    if hasattr(part, "content"):
                        maxim_msg["content"].append(
                            {"type": "text", "text": str(part.content)}
                        )
                    elif hasattr(part, "text"):
                        maxim_msg["content"].append(
                            {"type": "text", "text": str(part.text)}
                        )

                # If no content found, try string representation
                if not maxim_msg["content"]:
                    maxim_msg["content"] = str(msg)

                messages.append(maxim_msg)
            else:
                # Fallback for other message types
                messages.append({"role": "user", "content": str(msg)})
    except Exception as e:
        scribe().debug(f"[MaximSDK] Error converting messages: {e}")
        # Fallback to simple string conversion
        messages = [str(msg) for msg in pydantic_messages]

    return messages


def extract_content_from_response(response) -> str:
    """Extract meaningful content from Pydantic AI response."""
    try:
        if hasattr(response, "parts"):
            content_parts = []
            for part in response.parts:
                if hasattr(part, "content"):
                    content_parts.append(part.content)
                elif hasattr(part, "tool_name"):
                    # For tool calls, show the tool name and args
                    tool_info = f"Tool: {part.tool_name}"
                    if hasattr(part, "args"):
                        tool_info += f"({part.args})"
                    content_parts.append(tool_info)
            return " | ".join(content_parts) if content_parts else str(response)
        else:
            return str(response)
    except Exception as e:
        scribe().debug(f"[MaximSDK] Error extracting content: {e}")
        return str(response)


def extract_tool_calls_from_response(response) -> list:
    """Extract tool calls from Pydantic AI response."""
    tool_calls = []

    try:
        # Check if response has tool calls in parts
        if hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "tool_name") and hasattr(part, "args"):
                    tool_calls.append(
                        {
                            "name": part.tool_name,
                            "args": part.args,
                            "tool_call_id": getattr(
                                part, "tool_call_id", str(uuid.uuid4())
                            ),
                        }
                    )

        # Check if response is a ModelResponse with tool calls
        if hasattr(response, "tool_calls"):
            for tool_call in response.tool_calls:
                tool_calls.append(
                    {
                        "name": getattr(tool_call, "tool_name", "unknown"),
                        "args": getattr(tool_call, "args", {}),
                        "tool_call_id": getattr(
                            tool_call, "tool_call_id", str(uuid.uuid4())
                        ),
                    }
                )

        # Alternative check for tool calls in response structure
        if isinstance(response, dict) and "tool_calls" in response:
            for tool_call in response["tool_calls"]:
                tool_calls.append(
                    {
                        "name": tool_call.get("function", {}).get("name", "unknown"),
                        "args": tool_call.get("function", {}).get("arguments", {}),
                        "tool_call_id": tool_call.get("id", str(uuid.uuid4())),
                    }
                )

    except Exception as e:
        scribe().debug(f"[MaximSDK] Error extracting tool calls: {e}")

    return tool_calls


def start_session(
    maxim_logger: Logger,
    name: str = "Pydantic AI Session",
    metadata: Optional[Metadata] = None,
):
    """Start a global Maxim session for multiple agent runs."""
    global _session, _pydantic_metadata

    # Resolve effective metadata for this session: per-call overrides global
    effective_metadata: Optional[Metadata] = metadata or _pydantic_metadata

    if _session is None:
        # Prefer user-provided session_id from metadata if available
        session_id: Optional[str] = None
        if effective_metadata is not None:
            try:
                session_id = effective_metadata.session_id
            except Exception:
                # Best-effort; fall back to autogenerated ID
                session_id = None

        if not session_id:
            session_id = str(uuid.uuid4())

        # Base session tags
        session_tags: dict[str, str] = {
            "integration": "pydantic_ai",
            "type": "session",
        }

        # Merge user-provided session tags from metadata, if any
        if effective_metadata is not None and effective_metadata.session_tags:
            try:
                for key, value in effective_metadata.session_tags.items():
                    session_tags[str(key)] = str(value)
            except Exception as e:
                scribe().debug(
                    f"[MaximSDK] Failed to merge session tags from metadata: {e}"
                )

        _session = maxim_logger.session(
            SessionConfigDict({"id": session_id, "name": name, "tags": session_tags})
        )
        scribe().debug(f"[MaximSDK] Started session: {session_id}")

    return _session


def end_session(maxim_logger: Logger):
    """End the global Maxim session."""
    global _session

    if _session:
        _session.end()
        maxim_logger.flush()
        _session = None
        _global_maxim_trace.set(None)
        scribe().debug("[MaximSDK] Ended session")


def instrument_pydantic_ai(
    maxim_logger: Logger,
    debug: bool = False,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Patches Pydantic AI's core components with proper async handling and usage tracking.
    """
    global _pydantic_metadata

    # Parse optional Maxim metadata to extract session_id, etc.
    if metadata is not None:
        try:
            # Support both {"maxim": {...}} and flat {"session_id": ...} styles
            maxim_meta_dict = metadata.get("maxim", metadata)
            _pydantic_metadata = Metadata(maxim_meta_dict)
        except Exception as e:
            _pydantic_metadata = None
            scribe().error(f"[MaximSDK] Failed to parse Pydantic AI metadata: {e}")

    if not PYDANTIC_AI_AVAILABLE:
        scribe().warning(
            "[MaximSDK] Pydantic AI not available. Skipping instrumentation."
        )
        return

    def make_maxim_wrapper(
        original_method,
        base_op_name: str,
        input_processor=None,
        output_processor=None,
        display_name_fn=None,
    ):
        @functools.wraps(original_method)
        def maxim_wrapper(self, *args, **kwargs):
            scribe().debug(f"――― Start: {base_op_name} ―――")

            global _global_maxim_trace
            global _last_llm_usages
            global _session
            global _pydantic_metadata

            # Process inputs
            bound_args = {}
            processed_inputs = {}
            final_op_name = base_op_name
            call_metadata: Optional[Metadata] = None

            try:
                sig = inspect.signature(original_method)
                bound_values = sig.bind(self, *args, **kwargs)
                bound_values.apply_defaults()
                bound_args = bound_values.arguments

                processed_inputs = bound_args
                if input_processor:
                    processed_inputs = input_processor(bound_args)

                if display_name_fn:
                    final_op_name = display_name_fn(processed_inputs)

                # Extract Maxim metadata from model_settings.extra_body.maxim_metadata
                # and strip it before calling the original method.
                raw_model_settings = bound_args.get("model_settings")
                if isinstance(raw_model_settings, dict):
                    extra_body = raw_model_settings.get("extra_body")
                    if isinstance(extra_body, dict) and "maxim_metadata" in extra_body:
                        maxim_meta_dict = extra_body.pop("maxim_metadata", None)
                        if maxim_meta_dict is not None:
                            try:
                                # maxim_meta_dict is expected to have keys like session_id,
                                # session_tags, trace_tags, etc. compatible with Metadata.
                                call_metadata = Metadata(maxim_meta_dict)
                            except Exception as e:
                                scribe().error(
                                    "[MaximSDK] Failed to parse maxim_metadata from "
                                    f"model_settings: {e}"
                                )

            except Exception as e:
                scribe().debug(
                    f"[MaximSDK] Failed to process inputs for {base_op_name}: {e}"
                )
                processed_inputs = {"self": self, "args": args, "kwargs": kwargs}

            # Resolve effective metadata for this invocation without mutating global state
            effective_metadata: Optional[Metadata] = call_metadata or _pydantic_metadata

            trace = None
            span = None
            generation = None
            tool_call = None
            trace_token = None

            # Initialize tracing based on object type
            if isinstance(self, Agent):
                # Use or create a Maxim session
                session = start_session(
                    maxim_logger,
                    "Pydantic AI Agent Session",
                    metadata=effective_metadata,
                )

                # Create a new trace for this user input (per Agent.run* invocation)
                trace_id = str(uuid.uuid4())
                trace_tags: dict[str, str] = {
                    "integration": "pydantic_ai",
                    "operation": base_op_name,
                }
                # Merge user-provided trace tags from metadata, if any
                if (
                    effective_metadata is not None
                    and effective_metadata.trace_tags is not None
                ):
                    try:
                        for key, value in effective_metadata.trace_tags.items():
                            trace_tags[str(key)] = str(value)
                    except Exception as e:
                        scribe().debug(
                            f"[MaximSDK] Failed to merge trace tags from metadata: {e}"
                        )

                trace = session.trace(
                    {
                        "id": trace_id,
                        "name": final_op_name or "Pydantic AI Agent Run",
                        "tags": trace_tags,
                    }
                )

                # Set this trace as the current one for the duration of the call
                _global_maxim_trace.set(trace)

                # Store the session and trace context for later use
                setattr(self, "_trace_context", session)
                setattr(self, "_session", session)
                setattr(self, "_maxim_trace", trace)
                scribe().debug(
                    f"[MaximSDK] Agent run started, session + trace created: {trace_id}"
                )

                # Propagate session + trace context to the model and tools
                if hasattr(self, "model"):
                    setattr(self.model, "_trace_context", session)
                    setattr(self.model, "_session", session)
                    setattr(self.model, "_maxim_trace", trace)
                if hasattr(self, "tools"):
                    for tool in self.tools:
                        setattr(tool, "_trace_context", session)
                        setattr(tool, "_session", session)
                        setattr(tool, "_maxim_trace", trace)

            elif isinstance(self, Model):
                generation_id = str(uuid.uuid4())
                setattr(self, "_maxim_generation_id", generation_id)

                model_info = extract_model_info(self)

                # Extract and convert messages
                messages = []
                if args and len(args) > 0:
                    pydantic_messages = args[0] if isinstance(args[0], list) else []
                    messages = convert_messages_to_maxim_format(pydantic_messages)

                model_generation_config = GenerationConfigDict(
                    {
                        "id": generation_id,
                        "name": "LLM Call",
                        "provider": model_info["provider"],
                        "model": model_info["model"],
                        "messages": messages,
                    }
                )

                # Find or create a Maxim trace for this model call
                trace = getattr(self, "_maxim_trace", None)

                if trace is None:
                    # Prefer the session propagated from the Agent
                    session = getattr(self, "_trace_context", None)
                    if session is None:
                        # Fallback to any existing session or create a new one
                        session = getattr(self, "_session", None)
                        if session is None:
                            session = start_session(
                                maxim_logger,
                                "Pydantic AI Session",
                                metadata=effective_metadata,
                            )

                    trace_id = str(uuid.uuid4())
                    model_trace_tags: dict[str, str] = {
                        "agent_name": "pydantic_ai.Agent",
                        "model": model_info["model"],
                    }
                    # Merge user-provided trace tags from metadata, if any
                    if (
                        effective_metadata is not None
                        and effective_metadata.trace_tags is not None
                    ):
                        try:
                            for key, value in effective_metadata.trace_tags.items():
                                model_trace_tags[str(key)] = str(value)
                        except Exception as e:
                            scribe().debug(
                                "[MaximSDK] Failed to merge trace tags from metadata (model): "
                                f"{e}"
                            )

                    trace = session.trace(
                        {
                            "id": trace_id,
                            "name": "Pydantic AI Model Call",
                            "tags": model_trace_tags,
                        }
                    )
                    setattr(self, "_maxim_trace", trace)
                    scribe().debug(
                        f"[MaximSDK] Created trace for model call: {trace_id}"
                    )

                # Set this trace as the current one in the context
                _global_maxim_trace.set(trace)

                # Create generation within the trace
                generation = trace.generation(model_generation_config)
                scribe().debug(
                    f"[MaximSDK] Created generation in trace: {generation_id}"
                )

                setattr(self, "_input", messages)
                setattr(self, "_model_info", model_info)

            elif isinstance(self, Tool):
                # Try to find the current trace for this tool call
                current_trace = getattr(self, "_maxim_trace", None)
                if current_trace is None:
                    current_trace = _global_maxim_trace.get()

                if not current_trace:
                    scribe().warning("[MaximSDK] No trace context found for tool")
                    return original_method(self, *args, **kwargs)

                tool_id = str(uuid.uuid4())
                tool_details = extract_tool_details(self)
                tool_args = str(processed_inputs.get("args", processed_inputs))
                tool_call = current_trace.tool_call(
                    {
                        "id": tool_id,
                        "name": tool_details["name"]
                        or getattr(self, "name", "unknown"),
                        "description": tool_details["description"]
                        or getattr(self, "description", "unknown"),
                        "args": tool_args,
                        "tags": {"tool_id": str(id(self))},
                    }
                )
                scribe().debug(f"[MaximSDK] Created tool call in trace: {tool_id}")

                # Store tool call for later processing
                setattr(self, "_tool_call", tool_call)
                tool_call = getattr(
                    self, "_tool_call", None
                )  # Ensure tool_call is available in scope

            scribe().debug(f"[MaximSDK] --- Calling: {final_op_name} ---")

            try:
                # Call the original method
                output = original_method(self, *args, **kwargs)

                # Handle async responses with proper context preservation
                if hasattr(output, "__await__"):
                    scribe().debug(f"[MaximSDK] Handling coroutine for {final_op_name}")

                    async def async_wrapper():
                        try:
                            # Preserve the context token across the async boundary
                            current_context_trace = _global_maxim_trace.get()

                            result = await output

                            # Restore context if needed
                            if (
                                current_context_trace
                                and _global_maxim_trace.get() != current_context_trace
                            ):
                                _global_maxim_trace.set(current_context_trace)

                            # Process the result and extract tool calls
                            if isinstance(self, Model) and generation:
                                await process_model_result(self, generation, result)

                            # End trace at the end of an async Agent run turn
                            if isinstance(self, Agent) and not base_op_name.endswith(
                                ".run_stream"
                            ):
                                try:
                                    agent_trace = getattr(self, "_maxim_trace", None)
                                    if agent_trace is not None:
                                        agent_trace.end()
                                except Exception as e:
                                    scribe().debug(
                                        f"[MaximSDK] Failed to end agent trace (async): {e}"
                                    )

                            return result
                        except Exception as e:
                            scribe().error(f"[MaximSDK] Error in async wrapper: {e}")
                            if generation:
                                generation.error({"message": str(e)})
                            raise

                    return async_wrapper()

                # Special handling for Agent.run_stream: wrap the async context manager
                if isinstance(self, Agent) and base_op_name.endswith(".run_stream"):
                    # We expect output to be an async context manager; wrap to end trace on __aexit__
                    class MaximAgentStreamWrapper:
                        def __init__(self, inner_cm, agent_obj):
                            self._inner_cm = inner_cm
                            self._agent_obj = agent_obj

                        async def __aenter__(self):
                            return await self._inner_cm.__aenter__()

                        async def __aexit__(self, exc_type, exc, tb):
                            try:
                                agent_trace = getattr(
                                    self._agent_obj, "_maxim_trace", None
                                )
                                if agent_trace is not None:
                                    agent_trace.end()
                            except (
                                Exception
                            ) as e:  # pragma: no cover - best-effort cleanup
                                scribe().debug(
                                    f"[MaximSDK] Failed to end agent trace (stream): {e}"
                                )
                            return await self._inner_cm.__aexit__(exc_type, exc, tb)

                    return MaximAgentStreamWrapper(output, self)

                # Handle synchronous responses (non-stream)
                processed_output = output
                if output_processor:
                    try:
                        processed_output = output_processor(output)
                    except Exception as e:
                        scribe().debug(f"[MaximSDK] Failed to process output: {e}")

                # Complete tool calls
                if tool_call is not None:
                    if isinstance(tool_call, Retrieval):
                        tool_call.output(processed_output)
                    else:
                        tool_call.result(processed_output)
                    scribe().debug("[MaximSDK] TOOL: Completed tool call")

                # Complete generations for sync calls
                if generation and not hasattr(output, "__await__"):
                    process_model_result_sync(self, generation, processed_output)

                # End the trace at the end of a synchronous Agent turn
                if isinstance(self, Agent) and not base_op_name.endswith(".run_stream"):
                    try:
                        agent_trace = getattr(self, "_maxim_trace", None)
                        if agent_trace is not None:
                            agent_trace.end()
                    except Exception as e:
                        scribe().debug(
                            f"[MaximSDK] Failed to end agent trace (sync): {e}"
                        )

                return output

            except Exception as e:
                traceback.print_exc()
                scribe().error(f"[MaximSDK] {type(e).__name__} in {final_op_name}: {e}")

                # Error handling for all components
                if tool_call:
                    if isinstance(tool_call, Retrieval):
                        tool_call.output(f"Error occurred while calling tool: {e}")
                    else:
                        tool_call.result(f"Error occurred while calling tool: {e}")

                if generation:
                    generation.error({"message": str(e)})

                if span:
                    span.add_error({"message": str(e)})
                    span.end()

                raise

        return maxim_wrapper

    async def process_model_result(model_self, generation, result):
        """Process model result and handle tool calls for async calls."""
        usage_info = extract_usage_from_response(result)
        model_info = getattr(model_self, "_model_info", {})

        # Extract tool calls from the response
        tool_calls = extract_tool_calls_from_response(result)

        # Extract meaningful content from the response
        content = extract_content_from_response(result)

        # Create generation result
        gen_result = {
            "id": f"gen_{generation.id}",
            "object": "chat.completion",
            "created": int(time()),
            "model": model_info.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_info,
        }

        # Add tool calls to the generation result if found
        if tool_calls:
            # Convert tool calls to the format Maxim expects
            maxim_tool_calls = []
            for tool_call in tool_calls:
                maxim_tool_call = {
                    "id": tool_call.get("tool_call_id", str(uuid.uuid4())),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", "unknown"),
                        "arguments": str(tool_call.get("args", {})),
                    },
                }
                maxim_tool_calls.append(maxim_tool_call)

            gen_result["choices"][0]["message"]["tool_calls"] = maxim_tool_calls
            scribe().debug(
                f"[MaximSDK] Found {len(tool_calls)} tool calls in model response"
            )

            # Create tool call spans for each tool call
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("tool_call_id", str(uuid.uuid4()))
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})

                # Get the parent trace (trace associated with this model call)
                parent_trace = getattr(model_self, "_maxim_trace", None)
                if not parent_trace:
                    parent_trace = _global_maxim_trace.get()
                    if not parent_trace or not hasattr(parent_trace, "id"):
                        scribe().warning(
                            "[MaximSDK] No parent trace found for tool call span"
                        )
                        continue

                # Create tool call entry on the trace
                tool_span = parent_trace.tool_call(
                    {
                        "id": tool_call_id,
                        "name": tool_name,
                        "description": f"Tool call to {tool_name}",
                        "args": str(tool_args),
                        "tags": {"tool_call_id": tool_call_id},
                    }
                )
                scribe().debug(
                    f"[MaximSDK] Created tool call span: {tool_call_id} for {tool_name}"
                )

        generation.result(gen_result)
        scribe().debug("[MaximSDK] GEN: Completed async generation")

    def process_model_result_sync(model_self, generation, result):
        """Process model result and handle tool calls for sync calls."""
        usage_info = extract_usage_from_response(result)
        model_info = getattr(model_self, "_model_info", {})

        # Extract tool calls from the response
        tool_calls = extract_tool_calls_from_response(result)

        # Extract meaningful content from the response
        content = extract_content_from_response(result)

        # Create generation result
        gen_result = {
            "id": f"gen_{generation.id}",
            "object": "chat.completion",
            "created": int(time()),
            "model": model_info.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_info,
        }

        # Add tool calls to the generation result if found
        if tool_calls:
            # Convert tool calls to the format Maxim expects
            maxim_tool_calls = []
            for tool_call in tool_calls:
                maxim_tool_call = {
                    "id": tool_call.get("tool_call_id", str(uuid.uuid4())),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", "unknown"),
                        "arguments": str(tool_call.get("args", {})),
                    },
                }
                maxim_tool_calls.append(maxim_tool_call)

            gen_result["choices"][0]["message"]["tool_calls"] = maxim_tool_calls
            scribe().debug(
                f"[MaximSDK] Found {len(tool_calls)} tool calls in model response"
            )

            # Create tool call spans for each tool call
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("tool_call_id", str(uuid.uuid4()))
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})

                # Get the parent trace (trace associated with this model call)
                parent_trace = getattr(model_self, "_maxim_trace", None)
                if not parent_trace:
                    parent_trace = _global_maxim_trace.get()
                    if not parent_trace or not hasattr(parent_trace, "id"):
                        scribe().warning(
                            "[MaximSDK] No parent trace found for tool call span"
                        )
                        continue

                # Create tool call entry on the trace
                tool_span = parent_trace.tool_call(
                    {
                        "id": tool_call_id,
                        "name": tool_name,
                        "description": f"Tool call to {tool_name}",
                        "args": str(tool_args),
                        "tags": {"tool_call_id": tool_call_id},
                    }
                )
                scribe().debug(
                    f"[MaximSDK] Created tool call span: {tool_call_id} for {tool_name}"
                )

        generation.result(gen_result)
        scribe().debug("[MaximSDK] GEN: Completed sync generation")

    # Patch Agent methods
    if Agent is not None:
        agent_methods = ["run", "run_sync", "run_stream"]
        for method_name in agent_methods:
            if hasattr(Agent, method_name):
                original_method = getattr(Agent, method_name)
                wrapper = make_maxim_wrapper(
                    original_method,
                    f"pydantic_ai.Agent.{method_name}",
                    input_processor=pydantic_ai_postprocess_inputs,
                    display_name_fn=get_agent_display_name,
                )
                setattr(Agent, method_name, wrapper)
                scribe().info(f"[MaximSDK] Patched pydantic_ai.Agent.{method_name}")

    # Patch AbstractAgent methods
    if AbstractAgent is not None:
        abstract_agent_methods = ["run", "run_sync", "run_stream"]
        for method_name in abstract_agent_methods:
            if hasattr(AbstractAgent, method_name):
                original_method = getattr(AbstractAgent, method_name)
                wrapper = make_maxim_wrapper(
                    original_method,
                    f"pydantic_ai.AbstractAgent.{method_name}",
                    input_processor=pydantic_ai_postprocess_inputs,
                    display_name_fn=get_agent_display_name,
                )
                setattr(AbstractAgent, method_name, wrapper)
                scribe().info(
                    f"[MaximSDK] Patched pydantic_ai.AbstractAgent.{method_name}"
                )

    # Patch Tool methods
    if Tool is not None:
        tool_methods = ["__call__", "run"]
        for method_name in tool_methods:
            if hasattr(Tool, method_name):
                original_method = getattr(Tool, method_name)
                wrapper = make_maxim_wrapper(
                    original_method,
                    f"pydantic_ai.Tool.{method_name}",
                    input_processor=lambda inputs: dictify(inputs),
                    output_processor=lambda output: dictify(output),
                    display_name_fn=get_tool_display_name,
                )
                setattr(Tool, method_name, wrapper)
                scribe().info(f"[MaximSDK] Patched pydantic_ai.Tool.{method_name}")

    # Patch specific model classes
    try:
        from pydantic_ai.models.openai import OpenAIChatModel

        if OpenAIChatModel is not None:
            openai_methods = ["request", "request_stream"]
            for method_name in openai_methods:
                if hasattr(OpenAIChatModel, method_name):
                    original_method = getattr(OpenAIChatModel, method_name)
                    wrapper = make_maxim_wrapper(
                        original_method,
                        f"pydantic_ai.OpenAIChatModel.{method_name}",
                        input_processor=lambda inputs: dictify(inputs),
                        output_processor=lambda output: str(output) if output else None,
                    )
                    setattr(OpenAIChatModel, method_name, wrapper)
                    scribe().info(
                        f"[MaximSDK] Patched pydantic_ai.OpenAIChatModel.{method_name}"
                    )
    except ImportError:
        scribe().warning(
            "[MaximSDK] OpenAIChatModel not found. Skipping OpenAI model patching."
        )

    # Expose session management functions
    instrument_pydantic_ai.start_session = (
        lambda name="Pydantic AI Session": start_session(maxim_logger, name)
    )
    instrument_pydantic_ai.end_session = lambda: end_session(maxim_logger)

    scribe().info("[MaximSDK] Finished applying patches to Pydantic AI.")
