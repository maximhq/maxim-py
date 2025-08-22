"""Smolagents integration for the Maxim SDK.

This module instruments the :mod:`smolagents` library so ``CodeAgent.run``, 
``ToolCallingAgent.run`` calls and tool executions are automatically traced 
via :class:`~maxim.logger.Logger`.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import traceback
import uuid
from time import time
from typing import Any, Callable, Optional, Union

from ...logger import (
    Generation,
    GenerationConfigDict,
    Logger,
    Retrieval,
    Span,
    SpanConfigDict,
    ToolCall,
    Trace,
)
from ...scribe import scribe

# Global variables for tracking spans and state
_last_llm_usages = {}
_agent_span_ids = {}

_global_maxim_trace: contextvars.ContextVar[Union[Trace, None]] = (
    contextvars.ContextVar("maxim_trace_context_var", default=None)
)


def get_log_level(debug: bool) -> int:
    """Set logging level based on debug flag."""
    return logging.DEBUG if debug else logging.WARNING


class MaximEvalConfig:
    """Maxim eval configuration for Smolagents."""
    
    def __init__(self):
        self.evaluators = []
        self.additional_variables = []


class MaximUsageCallback:
    """Maxim usage callback for Smolagents.

    This class represents a usage callback that captures token usage
    from any LLM provider response.
    """

    def __init__(self, generation_id: str):
        """Initialize a usage callback."""
        self.generation_id = generation_id

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log a success event.

        Args:
            kwargs: The kwargs.
            response_obj: The response object.
            start_time: The start time.
            end_time: The end time.
        """
        global _last_llm_usages
        usage_info = response_obj.get("usage")
        if usage_info:
            if isinstance(usage_info, dict):
                _last_llm_usages[self.generation_id] = usage_info
            elif hasattr(usage_info, "prompt_tokens"):
                _last_llm_usages[self.generation_id] = {
                    "prompt_tokens": getattr(usage_info, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage_info, "completion_tokens", 0),
                    "total_tokens": getattr(usage_info, "total_tokens", 0),
                }
            else:
                _last_llm_usages[self.generation_id] = None  # Couldn't parse
            scribe().debug(
                f"[MaximSDK] Smolagents callback captured usage: {_last_llm_usages[self.generation_id] is not None}"
            )
        else:
            _last_llm_usages[self.generation_id] = None
            scribe().debug(
                "[MaximSDK] Smolagents callback did not find usage info in response_obj"
            )


def extract_smolagents_messages_and_params(agent, task, *args, **kwargs):
    """Extract messages and model parameters from Smolagents agent execution."""
    messages = []
    model_parameters = {}
    
    # Get system prompt/instructions from agent
    if hasattr(agent, 'system_prompt') and agent.system_prompt:
        messages.append({"role": "system", "content": agent.system_prompt})
    
    # Add the user task/query
    if isinstance(task, str):
        messages.append({"role": "user", "content": task})
    else:
        messages.append({"role": "user", "content": str(task)})
    
    # Extract model parameters from the agent's model
    if hasattr(agent, 'model'):
        model = agent.model
        # Common model parameter attributes
        for attr in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if hasattr(model, attr):
                model_parameters[attr] = getattr(model, attr)
        
        # Check for kwargs or additional parameters
        if hasattr(model, 'kwargs') and isinstance(model.kwargs, dict):
            model_parameters.update(model.kwargs)
    
    return messages, model_parameters


def to_serializable_dict(obj):
    """Convert object to JSON-serializable dict."""
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
        return to_serializable_dict(obj.dict())
    if hasattr(obj, '__dict__'):
        return to_serializable_dict(vars(obj))
    if hasattr(obj, '_asdict') and callable(getattr(obj, '_asdict')):
        return to_serializable_dict(obj._asdict())
    return str(obj)


def log_agent_execution(trace: Any, generation: Any, agent: Any, result: Any) -> None:
    """Log Smolagents execution results to trace."""
    # Get model info directly from agent's model (same as crewai approach)
    model_name = "unknown"
    provider = "smolagents"
    
    if hasattr(agent, 'model') and agent.model:
        model = agent.model
        # Get model name directly
        model_name = str(getattr(model, "model_id", getattr(model, "model_name", getattr(model, "name", "unknown"))))
        
        # Get provider directly (same simple logic as crewai)
        if hasattr(model, "is_anthropic") and model.is_anthropic:
            provider = "anthropic"
        elif hasattr(model, "is_openai") and model.is_openai:
            provider = "openai"
        elif hasattr(model, "is_google") and model.is_google:
            provider = "google"
        elif hasattr(model, "is_huggingface") and model.is_huggingface:
            provider = "huggingface"
        else:
            # Fallback: try to infer from model class name
            model_class_name = model.__class__.__name__.lower()
            if 'openai' in model_class_name:
                provider = "openai"
            elif 'anthropic' in model_class_name:
                provider = "anthropic"
            elif 'huggingface' in model_class_name or 'inference' in model_class_name:
                provider = "huggingface"
            elif 'google' in model_class_name or 'gemini' in model_class_name:
                provider = "google"
    
    # Handle different result types
    if hasattr(result, '__dict__'):
        result_dict = vars(result)
    elif isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {"content": str(result)}
    
    # Extract usage information from captured callback data
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_data = _last_llm_usages.get(generation.id)
    if usage_data and isinstance(usage_data, dict):
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", 0)
        scribe().debug(
            f"[MaximSDK] Smolagents: Using captured token usage: P={prompt_tokens}, C={completion_tokens}, T={total_tokens}"
        )
    else:
        scribe().debug(
            f"[MaximSDK] Smolagents: Using default token usage (0). Captured data: {usage_data}"
        )
    
    # Create result data compatible with Maxim format
    result_data = {
        "id": f"smolagents_{uuid.uuid4()}",
        "object": "agent.completion",
        "created": int(time()),
        "model": model_name,
        "provider": provider,
        "agent_type": agent.__class__.__name__,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": str(result),
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
    
    # Add additional metadata if available
    if hasattr(agent, 'tools') and agent.tools:
        result_data["tools_used"] = [tool.name if hasattr(tool, 'name') else str(tool) for tool in agent.tools]
    
    generation.result(to_serializable_dict(result_data))


def instrument_smolagents_tools():
    """Instrument Smolagents tool execution for tracing."""
    try:
        from smolagents.tools import Tool
    except ImportError:
        scribe().warning("[MaximSDK][Smolagents] Tool class not available for instrumentation")
        return
    
    # Patch Tool.__call__ method
    if hasattr(Tool, '__call__') and not getattr(Tool, '_maxim_tool_patched', False):
        original_call = Tool.__call__
        
        def instrumented_call(self, *args, **kwargs):
            # Try to get the current trace from tool or global context
            trace = getattr(self, '_maxim_trace', None) or _global_maxim_trace.get()
            if not trace:
                # No trace available, call original
                return original_call(self, *args, **kwargs)
            
            # Create tool call
            tool_call_id = str(uuid.uuid4())
            tool_name = getattr(self, 'name', self.__class__.__name__)
            tool_description = getattr(self, 'description', '')
            
            # Format arguments
            tool_args = {}
            if args:
                tool_args['args'] = args
            if kwargs:
                tool_args['kwargs'] = kwargs
            
            tool_call = trace.tool_call({
                "id": tool_call_id,
                "name": tool_name,
                "description": tool_description,
                "args": str(tool_args),
                "tags": {"tool_id": tool_call_id},
            })
            
            try:
                result = original_call(self, *args, **kwargs)
                tool_call.result(str(result))
                return result
            except Exception as e:
                tool_call.result(f"Error: {str(e)}")
                raise
        
        Tool.__call__ = instrumented_call
        setattr(Tool, "_maxim_tool_patched", True)
        scribe().info("[MaximSDK][Smolagents] Tool.__call__ instrumented for tool tracing")


def attach_trace_to_tools(agent: Any, trace: Any):
    """Attach trace to agent's tools for instrumentation."""
    if hasattr(agent, 'tools') and agent.tools:
        for tool in agent.tools:
            # Skip non-object tools (e.g., strings) or tools that disallow new attributes
            try:
                setattr(tool, '_maxim_trace', trace)
            except Exception:
                # Best-effort: Tool.__call__ will fallback to _global_maxim_trace
                continue


def make_smolagents_wrapper(
    original_method,
    base_op_name: str,
    agent_type: str = "unknown"
):
    """Create wrapper for Smolagents agent methods."""
    
    @functools.wraps(original_method)
    def smolagents_wrapper(self, task, *args, **kwargs):
        scribe().debug(f"――― Start: {base_op_name} ―――")
        
        global _global_maxim_trace
        
        # Check if we already have a trace context
        trace = _global_maxim_trace.get()
        is_local_trace = trace is None
        trace_token = None
        
        if is_local_trace:
            # Create new trace
            trace_id = str(uuid.uuid4())
            trace = maxim_logger.trace({
                "id": trace_id,
                "name": f"Smolagents {agent_type} Execution",
                "tags": {
                    "agent_type": agent_type,
                    "agent_id": str(getattr(self, "id", "")),
                },
                "input": str(task),
            })
            trace_token = _global_maxim_trace.set(trace)
        
        # Create generation for LLM tracking
        generation_id = str(uuid.uuid4())
        messages, model_parameters = extract_smolagents_messages_and_params(self, task, *args, **kwargs)
        
        # Get model info directly (same as crewai approach)
        model_name = "unknown"
        provider = "smolagents"
        
        if hasattr(self, 'model') and self.model:
            model = self.model
            # Get model name directly
            model_name = str(getattr(model, "model_id", getattr(model, "model_name", getattr(model, "name", "unknown"))))
            
            # Get provider directly (same simple logic as crewai)
            if hasattr(model, "is_anthropic") and model.is_anthropic:
                provider = "anthropic"
            elif hasattr(model, "is_openai") and model.is_openai:
                provider = "openai"
            elif hasattr(model, "is_google") and model.is_google:
                provider = "google"
            elif hasattr(model, "is_huggingface") and model.is_huggingface:
                provider = "huggingface"
            else:
                # Fallback: try to infer from model class name
                model_class_name = model.__class__.__name__.lower()
                if 'openai' in model_class_name:
                    provider = "openai"
                elif 'anthropic' in model_class_name:
                    provider = "anthropic"
                elif 'huggingface' in model_class_name or 'inference' in model_class_name:
                    provider = "huggingface"
                elif 'google' in model_class_name or 'gemini' in model_class_name:
                    provider = "google"
        
        generation_config = GenerationConfigDict({
            "id": generation_id,
            "name": "Smolagents Agent Run",
            "provider": provider,
            "model": model_name,
            "messages": messages,
            "model_parameters": model_parameters,
        })
        
        generation = trace.generation(generation_config)

        # Set generation ID on the agent's model for callback attachment
        if hasattr(self, 'model') and self.model:
            setattr(self.model, "_maxim_generation_id", generation_id)
        
        # Attach trace to tools for instrumentation
        attach_trace_to_tools(self, trace)
        
        try:
            # Call original method
            result = original_method(self, task, *args, **kwargs)
            
            # Handle streaming/generator results
            if inspect.isgenerator(result):
                def _iterate():
                    final_result = None
                    for chunk in result:
                        final_result = chunk
                        yield chunk
                    
                    # Log final result
                    if final_result is not None:
                        log_agent_execution(trace, generation, self, final_result)
                    
                    if is_local_trace:
                        trace.set_output(str(final_result) if final_result else "")
                        trace.end()
                        if trace_token:
                            _global_maxim_trace.reset(trace_token)
                        maxim_logger.flush()
                
                return _iterate()
            
            # Handle async generator results  
            elif inspect.isasyncgen(result):
                async def _async_iterate():
                    final_result = None
                    async for chunk in result:
                        final_result = chunk
                        yield chunk
                    
                    # Log final result
                    if final_result is not None:
                        log_agent_execution(trace, generation, self, final_result)
                    
                    if is_local_trace:
                        trace.set_output(str(final_result) if final_result else "")
                        trace.end()
                        if trace_token:
                            _global_maxim_trace.reset(trace_token)
                        maxim_logger.flush()
                
                return _async_iterate()
            
            # Handle regular results
            else:
                log_agent_execution(trace, generation, self, result)
                
                if is_local_trace:
                    trace.set_output(str(result))
                    trace.end()
                    if trace_token:
                        _global_maxim_trace.reset(trace_token)
                    maxim_logger.flush()
                
                return result
                
        except Exception as e:
            traceback.print_exc()
            scribe().error(f"[MaximSDK] {type(e).__name__} in {base_op_name}")
            
            generation.error({"message": str(e)})
            
            if is_local_trace:
                trace.add_error({"message": str(e)})
                trace.end()
                if trace_token:
                    _global_maxim_trace.reset(trace_token)
                maxim_logger.flush()
            
            raise e
        
        finally:
            # Clean up usage data
            try:
                _last_llm_usages.pop(generation.id, None)
            except Exception:
                pass
            scribe().debug(f"――― End: {base_op_name} ―――\n")
    
    return smolagents_wrapper


def instrument_smolagents_llm_callback():
    """Instrument Smolagents LLM models to attach usage callbacks."""
    try:
        # Patch the base model classes to attach callbacks
        from smolagents.models import InferenceClientModel, HfApiModel, OpenAIServerModel
        
        # Common callback attachment function
        def attach_callback_to_model(model_class):
            if hasattr(model_class, '__call__') and not getattr(model_class, '_maxim_callback_patched', False):
                original_call = model_class.__call__
                
                def wrapped_call(self, messages, *args, **kwargs):
                    # Get generation ID if set
                    generation_id = getattr(self, '_maxim_generation_id', None)
                    
                    # Create callback if generation ID is available
                    callbacks = kwargs.get('callbacks', [])
                    if generation_id and isinstance(callbacks, list):
                        callback = MaximUsageCallback(generation_id)
                        callbacks.append(callback)
                        kwargs['callbacks'] = callbacks
                    
                    return original_call(self, messages, *args, **kwargs)
                
                model_class.__call__ = wrapped_call
                setattr(model_class, '_maxim_callback_patched', True)
                scribe().info(f"[MaximSDK][Smolagents] {model_class.__name__} patched for callback attachment")
        
        # Patch common model classes
        for model_class in [InferenceClientModel, HfApiModel, OpenAIServerModel]:
            try:
                attach_callback_to_model(model_class)
            except Exception:
                pass
                
    except Exception as e:
        scribe().warning(f"[MaximSDK][Smolagents] Failed to patch LLM callbacks: {e}")


def instrument_smolagents(maxim_logger_instance: Logger, debug: bool = False):
    """
    Instrument Smolagents agents for comprehensive logging and tracing.
    
    This patches:
    - CodeAgent.run
    - ToolCallingAgent.run  
    - Tool execution methods
    - LLM models to attach usage callbacks (provider-agnostic)
    
    Args:
        maxim_logger_instance (Logger): Maxim Logger instance for tracing
        debug (bool): Enable debug logging
    """
    global maxim_logger
    maxim_logger = maxim_logger_instance
    
    scribe().set_level(get_log_level(debug))
    
    try:
        # Import Smolagents classes
        from smolagents import MultiStepAgent, CodeAgent, ToolCallingAgent
        # Try to import ManagedAgent if available
        try:
            from smolagents import ManagedAgent
        except ImportError:
            ManagedAgent = None
    except ImportError as exc:
        raise ImportError("Smolagents library is required for instrumentation") from exc
    
    # Check if already patched
    if getattr(MultiStepAgent, "_maxim_patched", False):
        scribe().info("[MaximSDK] Smolagents already instrumented")
        return
    
    # Patch MultiStepAgent.run (base class)
    if hasattr(MultiStepAgent, 'run'):
        original_multi_run = getattr(MultiStepAgent, 'run')
        wrapper_multi = make_smolagents_wrapper(
            original_multi_run,
            "smolagents.MultiStepAgent.run",
            "MultiStepAgent"
        )
        setattr(MultiStepAgent, 'run', wrapper_multi)
        scribe().info("[MaximSDK] Patched smolagents.MultiStepAgent.run")
    
    # Patch CodeAgent.run (if it has its own implementation)
    if hasattr(CodeAgent, 'run') and CodeAgent.run != MultiStepAgent.run:
        original_code_run = getattr(CodeAgent, 'run')
        wrapper_code = make_smolagents_wrapper(
            original_code_run,
            "smolagents.CodeAgent.run",
            "CodeAgent"
        )
        setattr(CodeAgent, 'run', wrapper_code)
        scribe().info("[MaximSDK] Patched smolagents.CodeAgent.run")
    
    # Patch ToolCallingAgent.run (if it has its own implementation)
    if hasattr(ToolCallingAgent, 'run') and ToolCallingAgent.run != MultiStepAgent.run:
        original_tool_run = getattr(ToolCallingAgent, 'run')
        wrapper_tool = make_smolagents_wrapper(
            original_tool_run,
            "smolagents.ToolCallingAgent.run", 
            "ToolCallingAgent"
        )
        setattr(ToolCallingAgent, 'run', wrapper_tool)
        scribe().info("[MaximSDK] Patched smolagents.ToolCallingAgent.run")
    
    # Patch ManagedAgent if available
    if ManagedAgent and hasattr(ManagedAgent, 'run'):
        original_managed_run = getattr(ManagedAgent, 'run')
        wrapper_managed = make_smolagents_wrapper(
            original_managed_run,
            "smolagents.ManagedAgent.run",
            "ManagedAgent"
        )
        setattr(ManagedAgent, 'run', wrapper_managed)
        scribe().info("[MaximSDK] Patched smolagents.ManagedAgent.run")
    
    # Instrument tool execution
    instrument_smolagents_tools()

    # Instrument LLM models to attach usage callbacks
    instrument_smolagents_llm_callback()
    
    # Mark as patched
    setattr(MultiStepAgent, "_maxim_patched", True)
    setattr(CodeAgent, "_maxim_patched", True)
    setattr(ToolCallingAgent, "_maxim_patched", True)
    if ManagedAgent:
        setattr(ManagedAgent, "_maxim_patched", True)
    
    scribe().info("[MaximSDK] Smolagents instrumentation completed")