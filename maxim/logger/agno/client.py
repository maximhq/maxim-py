"""Agno integration for the Maxim SDK.

This module instruments the :mod:`agno` library so ``Agent.run`` and
``Agent.arun`` calls are automatically traced via :class:`~maxim.logger.Logger`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from uuid import uuid4
from agno.agent import RunEvent
import inspect
from ..logger import GenerationConfig, Logger, TraceConfig, RetrievalConfig
from ...scribe import scribe


def _extract_agno_messages_and_params(agent, args, kwargs):
    messages = []
    model_parameters = {}
    # System prompt/instructions
    instructions = getattr(agent, "instructions", None)
    if instructions:
        messages.append({"role": "system", "content": instructions})
    # User message (common arg: message or input)
    user_message = None
    if "message" in kwargs:
        user_message = kwargs["message"]
    elif "input" in kwargs:
        user_message = kwargs["input"]
    elif args:
        # If first arg is a string, treat as message
        if isinstance(args[0], str):
            user_message = args[0]
    if user_message:
        messages.append({"role": "user", "content": user_message})
    # Model parameters - try various possible locations
    if "model_settings" in kwargs and isinstance(kwargs["model_settings"], dict):
        model_parameters = dict(kwargs["model_settings"])
    elif hasattr(agent, "model_settings") and isinstance(agent.model_settings, dict):
        model_parameters = dict(agent.model_settings)
    elif hasattr(agent, "model") and hasattr(agent.model, "model_settings"):
        if isinstance(agent.model.model_settings, dict):
            model_parameters = dict(agent.model.model_settings)
    elif hasattr(agent, "model") and hasattr(agent.model, "kwargs"):
        if isinstance(agent.model.kwargs, dict):
            model_parameters = dict(agent.model.kwargs)
    return messages, model_parameters


def _start_trace(
    logger: Logger,
    agent: Any,
    trace_id: Optional[str],
    generation_name: Optional[str],
    *args,
    **kwargs,
) -> tuple[Any, Any, str, bool]:
    """Create a trace and generation for the run, extracting real messages and parameters."""
    is_local_trace = trace_id is None
    final_trace_id = trace_id or str(uuid4())
    trace = logger.trace(TraceConfig(id=final_trace_id))
    messages, model_parameters = _extract_agno_messages_and_params(agent, args, kwargs)

    # Extract model name and provider directly from the model object
    model_name = "unknown"
    provider = "agno"
    if hasattr(agent, "model"):
        model = agent.model
        model_name = getattr(model, "id", model_name)
        provider = getattr(model, "provider", provider)

    gen_config = GenerationConfig(
        id=str(uuid4()),
        model=model_name,
        provider=provider,
        name=generation_name,
        messages=messages,
        model_parameters=model_parameters,
    )
    generation = trace.generation(gen_config)

    # Attach the trace span to tools and knowledge for instrumentation
    _attach_span_to_tools_and_knowledge(agent, trace)

    return trace, generation, final_trace_id, is_local_trace


def to_serializable_dict(obj):
    """Recursively convert an object to a JSON-serializable dict."""
    import enum
    import datetime
    import collections.abc

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value if hasattr(obj, 'value') else str(obj)
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {to_serializable_dict(k): to_serializable_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable_dict(item) for item in obj]
    if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
        # Pydantic/BaseModel
        return to_serializable_dict(obj.dict())
    if hasattr(obj, '__dict__'):
        return to_serializable_dict(vars(obj))
    if hasattr(obj, '_asdict') and callable(getattr(obj, '_asdict')):
        # namedtuple
        return to_serializable_dict(obj._asdict())
    # Fallback: string representation
    return str(obj)


def _extract_tool_execution_fields(tool_exec: Any) -> dict:
    """Extract fields from an Agno ToolExecution into a flat dict.

    Works identically for v1 (1.7.x) and v2 (2.4.x) — both have the same
    core fields: tool_call_id, tool_name, tool_args, result, tool_call_error.
    """
    import json

    tool_call_id = getattr(tool_exec, "tool_call_id", None) or str(uuid4())
    tool_name = getattr(tool_exec, "tool_name", None) or "unknown"
    tool_args = getattr(tool_exec, "tool_args", None)
    tool_result = getattr(tool_exec, "result", None)
    tool_error = getattr(tool_exec, "tool_call_error", None)

    args_str = ""
    if tool_args is not None:
        try:
            args_str = json.dumps(tool_args)
        except Exception:
            args_str = str(tool_args)

    return {
        "id": tool_call_id,
        "name": tool_name,
        "args": args_str,
        "result": tool_result,
        "error": tool_error,
    }


def _log_single_tool_call(trace: Any, tool_exec: Any) -> None:
    """Create a tool_call entity on the trace from a single ToolExecution."""
    try:
        fields = _extract_tool_execution_fields(tool_exec)
        tc = trace.tool_call({
            "id": fields["id"],
            "name": fields["name"],
            "description": f"Tool call: {fields['name']}",
            "args": fields["args"],
        })
        if fields["error"]:
            tc.error({"message": str(fields["result"] or "Tool call error")})
        else:
            tc.result(str(fields["result"]) if fields["result"] is not None else "")
    except Exception as e:
        scribe().warning(
            f"[MaximSDK][Agno] Failed to log tool call "
            f"{getattr(tool_exec, 'tool_name', '?')}: {e}"
        )


def _log_tool_calls_from_response(trace: Any, agno_response: Any) -> None:
    """Extract tool executions from a non-streaming Agno response and log each."""
    tools = getattr(agno_response, "tools", None)
    if not tools:
        return
    for tool_exec in tools:
        _log_single_tool_call(trace, tool_exec)


def _maybe_log_tool_call_from_chunk(trace: Any, chunk: Any) -> None:
    """If *chunk* is a ToolCallCompleted event, log the tool call immediately."""
    event = getattr(chunk, "event", None)
    if event != RunEvent.tool_call_completed.value:
        return
    tool_exec = getattr(chunk, "tool", None)
    if tool_exec is not None:
        _log_single_tool_call(trace, tool_exec)


def _build_tool_calls_for_generation(tool_executions: list) -> list:
    """Build an OpenAI-style ``tool_calls`` list from Agno ToolExecution objects."""
    result = []
    for tool_exec in tool_executions:
        fields = _extract_tool_execution_fields(tool_exec)
        result.append({
            "id": fields["id"],
            "type": "function",
            "function": {
                "name": fields["name"],
                "arguments": fields["args"],
            },
        })
    return result


def _collect_tool_executions_from_chunks(chunks: list) -> list:
    """Gather ToolExecution objects from ToolCallCompleted stream events."""
    tool_execs = []
    for chunk in chunks:
        event = getattr(chunk, "event", None)
        if event != RunEvent.tool_call_completed.value:
            continue
        tool_exec = getattr(chunk, "tool", None)
        if tool_exec is not None:
            tool_execs.append(tool_exec)
    return tool_execs


def _log_event(
    trace: Any,
    generation: Any,
    agno_response: Any,
    tool_executions: list | None = None,
) -> None:
    """Log an Agno response to the trace and record results.

    Args:
        tool_executions: Optional list of Agno ToolExecution objects to include
            as ``tool_calls`` in the generation message.
    """
    import time
    import collections.abc

    # Robustly convert agno_response to a dict if possible
    response_dict = None
    if isinstance(agno_response, dict):
        response_dict = agno_response
    elif hasattr(agno_response, 'dict') and callable(getattr(agno_response, 'dict')):
        # Pydantic/BaseModel
        response_dict = agno_response.dict()
    elif hasattr(agno_response, '__dict__'):
        response_dict = vars(agno_response)
    elif isinstance(agno_response, collections.abc.Mapping):
        response_dict = dict(agno_response)
    else:
        # Fallback: treat as string content
        response_dict = {"content": str(agno_response)}

    # Extract fields and ensure serializability
    model = response_dict.get("model", "unknown")
    model_provider = response_dict.get("model_provider", "unknown")
    metrics = to_serializable_dict(response_dict.get("metrics", {}))
    messages = to_serializable_dict(response_dict.get("messages", []))
    agent_team = to_serializable_dict(response_dict.get("agent_team", {}))
    tools = to_serializable_dict(response_dict.get("tools", []))
    run_id = response_dict.get("run_id", None)
    session_id = response_dict.get("session_id", None)
    agent_id = response_dict.get("agent_id", None)
    created_at = response_dict.get("created_at", int(time.time()))
    context_rules = to_serializable_dict(response_dict.get("context_rules", {}))

    # Helper to get first value if list, else int, else 0
    def _first(val):
        if isinstance(val, list) and val:
            return val[0]
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        return 0

    prompt_tokens = _first(metrics.get("prompt_tokens", 0))
    completion_tokens = _first(metrics.get("completion_tokens", 0))
    total_tokens = _first(metrics.get("total_tokens", 0))

    gen_tool_calls = None
    if tool_executions:
        gen_tool_calls = _build_tool_calls_for_generation(tool_executions)

    result_data = {
        "id": f"agno_{uuid4()}",
        "object": "chat.completion",
        "created": created_at,
        "model": model,
        "model_provider": model_provider,
        "run_id": run_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "agent_team": agent_team,
        "context_rules": context_rules,
        "messages": messages,
        "tools": tools,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_dict.get("content", ""),
                    "tool_calls": gen_tool_calls,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    generation.result(to_serializable_dict(result_data))


def _log_event_from_chunks(trace: Any, generation: Any, chunks: list) -> None:
    """Combine accumulated stream chunks into a single response and log once.

    Tool call entities are already logged inline during iteration via
    ``_maybe_log_tool_call_from_chunk``; here we only collect the
    ToolExecution objects so they appear in the generation message.
    """
    if not chunks:
        return

    # Collect tool executions for the generation message
    tool_execs = _collect_tool_executions_from_chunks(chunks)

    last_chunk = chunks[-1]

    # Concatenate content across all chunks
    combined_content_parts = []
    for chunk in chunks:
        content = getattr(chunk, "content", None)
        if content is None and isinstance(chunk, dict):
            content = chunk.get("content")
        if isinstance(content, str):
            combined_content_parts.append(content)
        elif content is not None:
            combined_content_parts.append(str(content))
    combined_content = "".join(combined_content_parts)

    import collections.abc
    response_dict = None
    if isinstance(last_chunk, dict):
        response_dict = dict(last_chunk)
    elif hasattr(last_chunk, 'dict') and callable(getattr(last_chunk, 'dict')):
        response_dict = last_chunk.dict()
    elif hasattr(last_chunk, '__dict__'):
        response_dict = dict(vars(last_chunk))
    elif isinstance(last_chunk, collections.abc.Mapping):
        response_dict = dict(last_chunk)
    else:
        response_dict = {}

    response_dict["content"] = combined_content
    _log_event(trace, generation, response_dict, tool_executions=tool_execs or None)



def _instrument_knowledge_retrieval(logger: Logger):
    """Instrument Agno knowledge base searches for retrieval tracing."""
    try:
        from agno.knowledge import AgentKnowledge
    except ImportError:
        scribe().warning("[MaximSDK][Agno] AgentKnowledge not available for instrumentation")
        return

    # Patch AgentKnowledge.search method
    if hasattr(AgentKnowledge, 'search') and not getattr(AgentKnowledge, '_maxim_knowledge_patched', False):
        original_search = AgentKnowledge.search

        def instrumented_search(self, query, num_documents=None, filters=None):
            # Try to get the current span from the agent
            span = getattr(self, '_maxim_span', None)
            if not span:
                # No span available, call original
                return original_search(self, query, num_documents, filters)

            # Create retrieval
            retrieval_id = str(uuid4())
            retrieval_config = RetrievalConfig(
                id=retrieval_id,
                name="Agno Knowledge Search",
            )

            retrieval = span.retrieval(retrieval_config)
            retrieval.input(query)

            try:
                results = original_search(self, query, num_documents, filters)
                # Format results for output
                if results and hasattr(results, '__iter__'):
                    output_text = "\n".join([str(result) for result in results])
                else:
                    output_text = str(results) if results else "No results found"
                retrieval.output(output_text)
                return results
            except Exception as e:
                retrieval.output(f"Error: {str(e)}")
                raise

        AgentKnowledge.search = instrumented_search
        setattr(AgentKnowledge, "_maxim_knowledge_patched", True)
        scribe().info("[MaximSDK][Agno] AgentKnowledge.search instrumented for retrieval tracing")


def _attach_span_to_tools_and_knowledge(agent: Any, trace: Any):
    """Attach the trace to agent and its knowledge for instrumentation."""
    # Attach trace to agent for tool execution
    setattr(agent, '_maxim_trace', trace)
    
    # Attach span to knowledge for retrieval tracing
    if hasattr(agent, 'knowledge') and agent.knowledge:
        setattr(agent.knowledge, '_maxim_span', trace)


def _wrap_sync(logger: Logger, fn: Callable) -> Callable:
    """Wrap a synchronous ``Agent.run`` implementation."""

    def wrapper(
        self,
        *args: Any,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        trace, generation, _, end_trace = _start_trace(
            logger, self, trace_id, generation_name, *args, **kwargs
        )
        result = None
        try:
            result = fn(self, *args, **kwargs)
            if inspect.isgenerator(result):
                def _iterate() -> Any:
                    chunks = []
                    try:
                        for chunk in result:
                            _maybe_log_tool_call_from_chunk(trace, chunk)
                            chunks.append(chunk)
                            yield chunk
                    finally:
                        _log_event_from_chunks(trace, generation, chunks)
                        if end_trace:
                            trace.end()

                return _iterate()

            _log_tool_calls_from_response(trace, result)
            tool_execs = getattr(result, "tools", None) or []
            _log_event(
                trace,
                generation,
                result,
                tool_executions=tool_execs or None,
            )
            return result
        except Exception as exc:  # pragma: no cover - passthrough
            generation.error({"message": str(exc), "code": "AGNO_EXECUTION_ERROR"})
            raise
        finally:
            if end_trace and not (result is not None and inspect.isgenerator(result)):
                trace.end()

    return wrapper


def _wrap_async(logger: Logger, fn: Callable) -> Callable:
    """Wrap an asynchronous ``Agent.arun`` implementation."""

    # Agno v2: arun is an async generator function (yields events directly).
    # We must wrap it as an async generator so callers can
    # ``async for chunk in agent.arun(...)`` without awaiting first.
    if inspect.isasyncgenfunction(fn):
        async def wrapper(
            self,
            *args: Any,
            trace_id: Optional[str] = None,
            generation_name: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            trace, generation, _, end_trace = _start_trace(
                logger, self, trace_id, generation_name, *args, **kwargs
            )
            chunks = []
            try:
                async for chunk in fn(self, *args, **kwargs):
                    _maybe_log_tool_call_from_chunk(trace, chunk)
                    chunks.append(chunk)
                    yield chunk
            except Exception as exc:  # pragma: no cover - passthrough
                generation.error({"message": str(exc), "code": "AGNO_EXECUTION_ERROR"})
                raise
            finally:
                _log_event_from_chunks(trace, generation, chunks)
                if end_trace:
                    trace.end()
        return wrapper

    # Agno v1: arun is a regular async function that returns a coroutine.
    async def wrapper(
        self,
        *args: Any,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        trace, generation, _, end_trace = _start_trace(
            logger, self, trace_id, generation_name, *args, **kwargs
        )
        result = None
        try:
            result = await fn(self, *args, **kwargs)
            if inspect.isasyncgen(result):
                async def _iterate() -> Any:
                    chunks = []
                    try:
                        async for chunk in result:
                            _maybe_log_tool_call_from_chunk(trace, chunk)
                            chunks.append(chunk)
                            yield chunk
                    finally:
                        _log_event_from_chunks(trace, generation, chunks)
                        if end_trace:
                            trace.end()

                return _iterate()

            _log_tool_calls_from_response(trace, result)
            tool_execs = getattr(result, "tools", None) or []
            _log_event(
                trace,
                generation,
                result,
                tool_executions=tool_execs or None,
            )
            return result
        except Exception as exc:  # pragma: no cover - passthrough
            generation.error({"message": str(exc), "code": "AGNO_EXECUTION_ERROR"})
            raise
        finally:
            if end_trace and not (result is not None and inspect.isasyncgen(result)):
                trace.end()

    return wrapper


def instrument_agno(logger: Logger) -> None:
    """Patch Agno's Agent.run and Agent.arun methods to log via Maxim."""
    try:
        from agno.agent.agent import Agent
    except Exception as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "Agno library is required for instrumentation",
        ) from exc

    if getattr(Agent, "_maxim_patched", False):
        return

    # Instrument basic agent methods
    original_run = Agent.run
    original_arun = getattr(Agent, "arun", None)

    Agent.run = _wrap_sync(logger, original_run)
    if original_arun is not None:
        Agent.arun = _wrap_async(logger, original_arun)
    setattr(Agent, "_maxim_patched", True)

    _instrument_knowledge_retrieval(logger)

    scribe().info("[MaximSDK] Agno instrumentation enabled with tool and knowledge tracing")
