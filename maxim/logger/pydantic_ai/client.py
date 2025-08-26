"""Maxim integration for Pydantic AI agent framework."""

import contextvars
import functools
import inspect
import logging
import traceback
import uuid
from time import time
from typing import Union

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
    Span,
    SpanConfigDict,
    ToolCall,
    Trace,
)
from ...scribe import scribe
from .utils import (
    pydantic_ai_postprocess_inputs,
    dictify,
    extract_tool_details,
    get_agent_display_name,
    get_tool_display_name,
)

_last_llm_usages = {}
_agent_span_ids = {}
_session_trace = None  # Global session trace

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
        if hasattr(response, 'usage'):
            usage = response.usage
            if hasattr(usage, 'input_tokens'):
                usage_info["prompt_tokens"] = getattr(usage, 'input_tokens', 0)
            if hasattr(usage, 'output_tokens'):
                usage_info["completion_tokens"] = getattr(usage, 'output_tokens', 0)
            if hasattr(usage, 'total_tokens'):
                usage_info["total_tokens"] = getattr(usage, 'total_tokens', 0)
            elif usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"]
        elif hasattr(response, 'prompt_tokens'):
            # Direct attributes on response
            usage_info["prompt_tokens"] = getattr(response, 'prompt_tokens', 0)
            usage_info["completion_tokens"] = getattr(response, 'completion_tokens', 0)
            usage_info["total_tokens"] = getattr(response, 'total_tokens', 0)
    except Exception as e:
        scribe().debug(f"[MaximSDK] Error extracting usage: {e}")
    
    return usage_info


def extract_model_info(model_obj) -> dict:
    """Extract model information from model object."""
    model_info = {"model": "unknown", "provider": "unknown"}
    
    try:
        if hasattr(model_obj, 'model_name'):
            model_info["model"] = model_obj.model_name or "unknown"
        elif hasattr(model_obj, 'model'):
            model_info["model"] = str(model_obj.model)
        
        if hasattr(model_obj, 'system'):
            model_info["provider"] = model_obj.system or "unknown"
        elif "openai" in str(type(model_obj)).lower():
            model_info["provider"] = "openai"
        elif "anthropic" in str(type(model_obj)).lower():
            model_info["provider"] = "anthropic"
        elif hasattr(model_obj, 'provider_name'):
            model_info["provider"] = model_obj.provider_name or "unknown"
    except Exception as e:
        scribe().debug(f"[MaximSDK] Error extracting model info: {e}")
    
    return model_info


def convert_messages_to_maxim_format(pydantic_messages) -> list:
    """Convert Pydantic AI messages to Maxim format."""
    messages = []
    try:
        for msg in pydantic_messages:
            if hasattr(msg, 'parts'):
                # Convert ModelRequest to Maxim format
                maxim_msg = {
                    "role": "user",  # Default role
                    "content": []
                }
                
                # Determine role and extract content
                for part in msg.parts:
                    if hasattr(part, 'content'):
                        maxim_msg["content"].append({
                            "type": "text",
                            "text": str(part.content)
                        })
                    elif hasattr(part, 'text'):
                        maxim_msg["content"].append({
                            "type": "text", 
                            "text": str(part.text)
                        })
                
                # If no content found, try string representation
                if not maxim_msg["content"]:
                    maxim_msg["content"] = str(msg)
                
                messages.append(maxim_msg)
            else:
                # Fallback for other message types
                messages.append({
                    "role": "user",
                    "content": str(msg)
                })
    except Exception as e:
        scribe().debug(f"[MaximSDK] Error converting messages: {e}")
        # Fallback to simple string conversion
        messages = [str(msg) for msg in pydantic_messages]
    
    return messages


def extract_content_from_response(response) -> str:
    """Extract meaningful content from Pydantic AI response."""
    try:
        if hasattr(response, 'parts'):
            content_parts = []
            for part in response.parts:
                if hasattr(part, 'content'):
                    content_parts.append(part.content)
                elif hasattr(part, 'tool_name'):
                    # For tool calls, show the tool name and args
                    tool_info = f"Tool: {part.tool_name}"
                    if hasattr(part, 'args'):
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
        if hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'tool_name') and hasattr(part, 'args'):
                    tool_calls.append({
                        "name": part.tool_name,
                        "args": part.args,
                        "tool_call_id": getattr(part, 'tool_call_id', str(uuid.uuid4()))
                    })
        
        # Check if response is a ModelResponse with tool calls
        if hasattr(response, 'tool_calls'):
            for tool_call in response.tool_calls:
                tool_calls.append({
                    "name": getattr(tool_call, 'tool_name', 'unknown'),
                    "args": getattr(tool_call, 'args', {}),
                    "tool_call_id": getattr(tool_call, 'tool_call_id', str(uuid.uuid4()))
                })
                
        # Alternative check for tool calls in response structure
        if isinstance(response, dict) and 'tool_calls' in response:
            for tool_call in response['tool_calls']:
                tool_calls.append({
                    "name": tool_call.get('function', {}).get('name', 'unknown'),
                    "args": tool_call.get('function', {}).get('arguments', {}),
                    "tool_call_id": tool_call.get('id', str(uuid.uuid4()))
                })
                
    except Exception as e:
        scribe().debug(f"[MaximSDK] Error extracting tool calls: {e}")
    
    return tool_calls


def start_session_trace(maxim_logger: Logger, name: str = "Pydantic AI Session"):
    """Start a global session trace for multiple agent runs."""
    global _session_trace
    
    if _session_trace is None:
        trace_id = str(uuid.uuid4())
        _session_trace = maxim_logger.trace({
            "id": trace_id,
            "name": name,
            "tags": {"session": "true"},
            "input": "Session started",
        })
        _global_maxim_trace.set(_session_trace)
        scribe().debug(f"[MaximSDK] Started session trace: {trace_id}")
    
    return _session_trace


def end_session_trace(maxim_logger: Logger):
    """End the global session trace."""
    global _session_trace
    
    if _session_trace:
        _session_trace.end()
        maxim_logger.flush()
        _session_trace = None
        _global_maxim_trace.set(None)
        scribe().debug("[MaximSDK] Ended session trace")


def instrument_pydantic_ai(maxim_logger: Logger, debug: bool = False):
    """
    Patches Pydantic AI's core components with proper async handling and usage tracking.
    """
    if not PYDANTIC_AI_AVAILABLE:
        scribe().warning("[MaximSDK] Pydantic AI not available. Skipping instrumentation.")
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
            global _agent_span_ids
            global _last_llm_usages
            global _session_trace

            # Process inputs
            bound_args = {}
            processed_inputs = {}
            final_op_name = base_op_name

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

            except Exception as e:
                scribe().debug(f"[MaximSDK] Failed to process inputs for {base_op_name}: {e}")
                processed_inputs = {"self": self, "args": args, "kwargs": kwargs}

            trace = None
            span = None
            generation = None
            tool_call = None
            trace_token = None

                        # Initialize tracing based on object type
            if isinstance(self, Agent):
                # Use or create session trace
                if _session_trace is None:
                    trace = start_session_trace(maxim_logger, "Pydantic AI Agent Session")
                else:
                    trace = _session_trace
                
                # Don't create agent span immediately - wait for actual operations
                # Store the trace context for later use
                setattr(self, "_trace_context", trace)
                scribe().debug(f"[MaximSDK] Agent run started, trace context stored")
                
                # Set the trace context for the model and tools
                if hasattr(self, "model"):
                    setattr(self.model, "_trace_context", trace)
                if hasattr(self, "tools"):
                    for tool in self.tools:
                        setattr(tool, "_trace_context", trace)

            elif isinstance(self, Model):
                generation_id = str(uuid.uuid4())
                setattr(self, "_maxim_generation_id", generation_id)
                
                model_info = extract_model_info(self)
                
                # Extract and convert messages
                messages = []
                if args and len(args) > 0:
                    pydantic_messages = args[0] if isinstance(args[0], list) else []
                    messages = convert_messages_to_maxim_format(pydantic_messages)

                model_generation_config = GenerationConfigDict({
                    "id": generation_id,
                    "name": "LLM Call",
                    "provider": model_info["provider"],
                    "model": model_info["model"],
                    "messages": messages,
                })

                # Try to find or create agent span for this model call
                agent_span = getattr(self, "_agent_span", None)
                generation = None
                
                if not agent_span:
                    # Look for trace context from agent
                    trace_context = getattr(self, "_trace_context", None)
                    if trace_context:
                        # Create agent span for this model call
                        span_id = str(uuid.uuid4())
                        agent_span = trace_context.span({
                            "id": span_id,
                            "name": "Agent Run: LLM Call",
                            "tags": {
                                "agent_name": "pydantic_ai.Agent",
                                "model": model_info["model"],
                            },
                        })
                        _agent_span_ids[id(self)] = span_id
                        scribe().debug(f"[MaximSDK] Created agent span for model call: {span_id}")
                        
                        # Set the agent span as context for this model
                        setattr(self, "_agent_span", agent_span)
                    else:
                        # Fallback to session trace
                        current_trace = _global_maxim_trace.get()
                        if current_trace and hasattr(current_trace, 'id'):
                            generation = current_trace.generation(model_generation_config)
                            scribe().debug(f"[MaximSDK] Created generation in session trace: {generation_id}")
                        else:
                            scribe().warning("[MaximSDK] No trace context found for model call")
                            trace = start_session_trace(maxim_logger, "LLM Call Session")
                            generation = trace.generation(model_generation_config)
                
                # Create generation within the agent span if we have one
                if agent_span and not generation:
                    generation = agent_span.generation(model_generation_config)
                    scribe().debug(f"[MaximSDK] Created generation in agent span: {generation_id}")
                
                setattr(self, "_input", messages)
                setattr(self, "_model_info", model_info)

            elif isinstance(self, Tool):
                # Try to find the agent span for this tool call
                agent_span = getattr(self, "_agent_span", None)
                if not agent_span:
                    # Look for agent span in the current context
                    current_trace = _global_maxim_trace.get()
                    if not current_trace:
                        scribe().warning("[MaximSDK] No trace context found for tool")
                        return original_method(self, *args, **kwargs)
                    
                    # Create tool call directly in the session trace
                    tool_id = str(uuid.uuid4())
                    tool_details = extract_tool_details(self)
                    tool_args = str(processed_inputs.get("args", processed_inputs))
                    tool_call = current_trace.tool_call({
                        "id": tool_id,
                        "name": tool_details['name'] or getattr(self, 'name', 'unknown'),
                        "description": tool_details["description"] or getattr(self, "description", "unknown"),
                        "args": tool_args,
                        "tags": {"tool_id": str(id(self))},
                    })
                    scribe().debug(f"[MaximSDK] Created tool call in session trace: {tool_id}")
                else:
                    # Create tool call within the agent span
                    tool_id = str(uuid.uuid4())
                    tool_details = extract_tool_details(self)
                    tool_args = str(processed_inputs.get("args", processed_inputs))
                    tool_call = agent_span.tool_call({
                        "id": tool_id,
                        "name": tool_details['name'] or getattr(self, 'name', 'unknown'),
                        "description": tool_details["description"] or getattr(self, "description", "unknown"),
                        "args": tool_args,
                        "tags": {"tool_id": str(id(self)), "span_id": agent_span.id},
                    })
                    scribe().debug(f"[MaximSDK] Created tool call in agent span: {tool_id}")
                
                # Store tool call for later processing
                setattr(self, "_tool_call", tool_call)
                tool_call = getattr(self, "_tool_call", None)  # Ensure tool_call is available in scope

            scribe().debug(f"[MaximSDK] --- Calling: {final_op_name} ---")

            try:
                # Call the original method
                output = original_method(self, *args, **kwargs)
                
                # Handle async responses with proper context preservation
                if hasattr(output, '__await__'):
                    scribe().debug(f"[MaximSDK] Handling coroutine for {final_op_name}")
                    
                    async def async_wrapper():
                        try:
                            # Preserve the context token across the async boundary
                            current_context_trace = _global_maxim_trace.get()
                            
                            result = await output
                            
                            # Restore context if needed
                            if current_context_trace and _global_maxim_trace.get() != current_context_trace:
                                _global_maxim_trace.set(current_context_trace)
                            
                            # Process the result and extract tool calls
                            if isinstance(self, Model) and generation:
                                await process_model_result(self, generation, result)
                            
                            return result
                        except Exception as e:
                            scribe().error(f"[MaximSDK] Error in async wrapper: {e}")
                            if generation:
                                generation.error({"message": str(e)})
                            raise
                    
                    return async_wrapper()
                
                # Handle synchronous responses
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
                if generation and not hasattr(output, '__await__'):
                    process_model_result_sync(self, generation, processed_output)

                # Complete spans
                if span and not isinstance(self, Agent):  # Agent spans are managed differently
                    span.end()
                    scribe().debug("[MaximSDK] SPAN: Completed span")

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
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }],
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
                        "arguments": str(tool_call.get("args", {}))
                    }
                }
                maxim_tool_calls.append(maxim_tool_call)
            
            gen_result["choices"][0]["message"]["tool_calls"] = maxim_tool_calls
            scribe().debug(f"[MaximSDK] Found {len(tool_calls)} tool calls in model response")
            
            # Create tool call spans for each tool call
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("tool_call_id", str(uuid.uuid4()))
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                # Get the parent span (agent span or session trace)
                parent_span = getattr(model_self, "_agent_span", None)
                if not parent_span:
                    current_trace = _global_maxim_trace.get()
                    if current_trace and hasattr(current_trace, 'id'):
                        parent_span = current_trace
                    else:
                        scribe().warning("[MaximSDK] No parent context found for tool call span")
                        continue
                
                # Create tool call span
                tool_span = parent_span.tool_call({
                    "id": tool_call_id,
                    "name": tool_name,
                    "description": f"Tool call to {tool_name}",
                    "args": str(tool_args),
                    "tags": {"tool_call_id": tool_call_id},
                })
                scribe().debug(f"[MaximSDK] Created tool call span: {tool_call_id} for {tool_name}")
        
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
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }],
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
                        "arguments": str(tool_call.get("args", {}))
                    }
                }
                maxim_tool_calls.append(maxim_tool_call)
            
            gen_result["choices"][0]["message"]["tool_calls"] = maxim_tool_calls
            scribe().debug(f"[MaximSDK] Found {len(tool_calls)} tool calls in model response")
            
            # Create tool call spans for each tool call
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("tool_call_id", str(uuid.uuid4()))
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                # Get the parent span (agent span or session trace)
                parent_span = getattr(model_self, "_agent_span", None)
                if not parent_span:
                    current_trace = _global_maxim_trace.get()
                    if current_trace and hasattr(current_trace, 'id'):
                        parent_span = current_trace
                    else:
                        scribe().warning("[MaximSDK] No parent context found for tool call span")
                        continue
                
                # Create tool call span
                tool_span = parent_span.tool_call({
                    "id": tool_call_id,
                    "name": tool_name,
                    "description": f"Tool call to {tool_name}",
                    "args": str(tool_args),
                    "tags": {"tool_call_id": tool_call_id},
                })
                scribe().debug(f"[MaximSDK] Created tool call span: {tool_call_id} for {tool_name}")
        
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
                scribe().info(f"[MaximSDK] Patched pydantic_ai.AbstractAgent.{method_name}")

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
                    scribe().info(f"[MaximSDK] Patched pydantic_ai.OpenAIChatModel.{method_name}")
    except ImportError:
        scribe().warning("[MaximSDK] OpenAIChatModel not found. Skipping OpenAI model patching.")

    # Expose session management functions
    instrument_pydantic_ai.start_session = lambda name="Pydantic AI Session": start_session_trace(maxim_logger, name)
    instrument_pydantic_ai.end_session = lambda: end_session_trace(maxim_logger)

    scribe().info("[MaximSDK] Finished applying patches to Pydantic AI.")