"""Maxim integration for Google Agent Development Kit (ADK)."""

import contextvars
import functools
import inspect
import logging
import traceback
import uuid
from time import time
from typing import Union, Optional, Any

try:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.runners import Runner, InMemoryRunner
    from google.adk.tools.base_tool import BaseTool
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.google_llm import Gemini
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins.base_plugin import BasePlugin
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.tools.tool_context import ToolContext
    from google.adk.agents.callback_context import CallbackContext
    from google.genai import types
    GOOGLE_ADK_AVAILABLE = True
except ImportError:
    GOOGLE_ADK_AVAILABLE = False
    BaseAgent = None
    LlmAgent = None
    Runner = None
    InMemoryRunner = None
    BaseTool = None
    BaseLlm = None
    Gemini = None
    LlmRequest = None
    LlmResponse = None
    BasePlugin = object  # Use object as base class when google-adk is not available
    InvocationContext = None
    ToolContext = None
    CallbackContext = None
    types = None

from ...logger import (
    GenerationConfigDict,
    Logger,
    Retrieval,
    Trace,
)
from ...scribe import scribe
from .utils import (
    google_adk_postprocess_inputs,
    dictify,
    extract_tool_details,
    get_agent_display_name,
    get_tool_display_name,
    extract_usage_from_response,
    extract_model_info,
    convert_messages_to_maxim_format,
    extract_content_from_response,
    extract_tool_calls_from_response,
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


class MaximInstrumentationPlugin(BasePlugin):
    """Maxim instrumentation plugin for Google ADK."""

    def __init__(self, maxim_logger: Logger, debug: bool = False):
        if GOOGLE_ADK_AVAILABLE:
            super().__init__(name="maxim_instrumentation")
        else:
            super().__init__()
        self.maxim_logger = maxim_logger
        self.debug = debug
        self._trace = None
        self._spans = {}
        self._generations = {}
        self._tool_calls = {}
        self._request_to_generation = {}  # Map request ID to generation ID

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        """Start a trace for the agent run."""
        global _session_trace
        
        scribe().info(f"[MaximSDK] before_run_callback called for agent: {invocation_context.agent.name}")
        
        if _session_trace is None:
            trace_id = str(uuid.uuid4())
            _session_trace = self.maxim_logger.trace({
                "id": trace_id,
                "name": "Google ADK Agent Run",
                "tags": {
                    "agent_name": invocation_context.agent.name,
                    "invocation_id": invocation_context.invocation_id,
                    "app_name": getattr(invocation_context, 'app_name', 'unknown'),
                },
                "input": "Agent run started",
            })
            _global_maxim_trace.set(_session_trace)
            scribe().info(f"[MaximSDK] Started session trace: {trace_id}")
        
        self._trace = _session_trace
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """End the trace after agent run completes."""
        if self._trace:
            self._trace.end()
            self.maxim_logger.flush()
            scribe().debug("[MaximSDK] Ended session trace")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """Instrument LLM request before sending to model."""
        print(f"[MaximSDK] before_model_callback called")
        scribe().info(f"[MaximSDK] before_model_callback called")
        
        if not self._trace:
            print("[MaximSDK] WARNING: No trace available for LLM call")
            scribe().warning("[MaximSDK] No trace available for LLM call")
            return None

        generation_id = str(uuid.uuid4())
        request_id = id(llm_request)  # Use request object ID as key
        
        # Extract model information
        model_info = extract_model_info(llm_request)
        print(f"[MaximSDK] Model info: {model_info}")
        scribe().info(f"[MaximSDK] Model info: {model_info}")
        
        # Convert messages to Maxim format
        messages = convert_messages_to_maxim_format(llm_request.contents)
        print(f"[MaximSDK] Messages: {len(messages)} messages")
        scribe().info(f"[MaximSDK] Messages: {len(messages)} messages")
        
        # Create generation config
        generation_config = GenerationConfigDict({
            "id": generation_id,
            "name": "LLM Call",
            "provider": model_info.get("provider", "google"),
            "model": model_info.get("model", "unknown"),
            "messages": messages,
        })

        # Create generation within the trace
        generation = self._trace.generation(generation_config)
        self._generations[generation_id] = generation
        
        # Store mapping from request to generation
        self._request_to_generation[request_id] = generation_id
        
        print(f"[MaximSDK] Created generation: {generation_id} for request: {request_id}")
        scribe().info(f"[MaximSDK] Created generation: {generation_id} for request: {request_id}")
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """Instrument LLM response after receiving from model."""
        print(f"[MaximSDK] after_model_callback called")
        scribe().info(f"[MaximSDK] after_model_callback called")
        
        if not self._trace:
            print("[MaximSDK] WARNING: No trace available for LLM response")
            scribe().warning("[MaximSDK] No trace available for LLM response")
            return None

        # Get the generation ID from the callback_context's llm_request
        request_id = id(callback_context.llm_request) if hasattr(callback_context, 'llm_request') else None
        generation_id = self._request_to_generation.get(request_id) if request_id else None
        
        if not generation_id:
            print(f"[MaximSDK] WARNING: No generation ID found for request: {request_id}")
            scribe().warning(f"[MaximSDK] No generation ID found for request: {request_id}")
            return None

        generation = self._generations.get(generation_id)
        if not generation:
            print(f"[MaximSDK] WARNING: No generation found for ID: {generation_id}")
            scribe().warning(f"[MaximSDK] No generation found for ID: {generation_id}")
            return None

        # Extract usage information
        usage_info = extract_usage_from_response(llm_response)
        
        # Extract content from response
        content = extract_content_from_response(llm_response)
        
        # Extract tool calls from response
        tool_calls = extract_tool_calls_from_response(llm_response)
        
        print(f"[MaximSDK] Usage info: {usage_info}")
        print(f"[MaximSDK] Content length: {len(content) if content else 0}")
        print(f"[MaximSDK] Tool calls: {len(tool_calls) if tool_calls else 0}")
        
        # Create generation result
        gen_result = {
            "id": f"gen_{generation_id}",
            "object": "chat.completion",
            "created": int(time()),
            "model": getattr(llm_response, "model", "unknown"),
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
            maxim_tool_calls = []
            for tool_call in tool_calls:
                # Ensure tool_call_id is never None
                tool_call_id = tool_call.get("tool_call_id") or str(uuid.uuid4())
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                maxim_tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": str(tool_args)
                    }
                }
                maxim_tool_calls.append(maxim_tool_call)
                print(f"[MaximSDK] Tool call: {tool_name} (ID: {tool_call_id}, Args: {tool_args})")
                scribe().info(f"[MaximSDK] Tool call: {tool_name} (ID: {tool_call_id})")
            
            gen_result["choices"][0]["message"]["tool_calls"] = maxim_tool_calls
            print(f"[MaximSDK] Added {len(tool_calls)} tool calls to generation result")
            scribe().info(f"[MaximSDK] Added {len(tool_calls)} tool calls to generation result")
            
            # Create tool call spans for each tool call
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("tool_call_id", str(uuid.uuid4()))
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                # Create tool call span
                self._trace.tool_call({
                    "id": tool_call_id,
                    "name": tool_name,
                    "description": f"Tool call to {tool_name}",
                    "args": str(tool_args),
                    "tags": {"tool_call_id": tool_call_id},
                })
                print(f"[MaximSDK] Created tool call span: {tool_call_id} for {tool_name}")
                scribe().debug(f"[MaximSDK] Created tool call span: {tool_call_id} for {tool_name}")
        
        generation.result(gen_result)
        print(f"[MaximSDK] Completed generation: {generation_id}")
        scribe().debug(f"[MaximSDK] GEN: Completed generation: {generation_id}")
        
        # Clean up
        del self._generations[generation_id]
        if request_id in self._request_to_generation:
            del self._request_to_generation[request_id]
        
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        """Instrument tool execution before calling tool."""
        if not self._trace:
            return None

        tool_id = str(uuid.uuid4())
        tool_details = extract_tool_details(tool)
        tool_args_str = str(tool_args)
        
        # Get trace ID safely (handle both object and dict)
        trace_id = self._trace.id if hasattr(self._trace, 'id') else self._trace.get('id', 'unknown')
        
        # Create tool call span
        tool_call = self._trace.tool_call({
            "id": tool_id,
            "name": tool_details.get('name', tool.name),
            "description": tool_details.get("description", tool.description),
            "args": tool_args_str,
            "tags": {"tool_id": str(id(tool)), "span_id": trace_id},
        })
        
        self._tool_calls[tool_id] = tool_call
        setattr(tool, "_maxim_tool_call", tool_call)
        
        scribe().debug(f"[MaximSDK] Created tool call: {tool_id} for {tool.name}")
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        """Instrument tool execution after calling tool."""
        if not self._trace:
            return None

        tool_call = getattr(tool, "_maxim_tool_call", None)
        if not tool_call:
            scribe().warning(f"[MaximSDK] No tool call found for tool: {tool.name}")
            return None

        # Complete the tool call
        tool_call.result(result)
        scribe().debug(f"[MaximSDK] TOOL: Completed tool call for {tool.name}")
        
        # Clean up
        delattr(tool, "_maxim_tool_call")
        
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        """Handle tool execution errors."""
        if not self._trace:
            return None

        tool_call = getattr(tool, "_maxim_tool_call", None)
        if tool_call:
            tool_call.result(f"Error occurred while calling tool: {error}")
            scribe().debug(f"[MaximSDK] TOOL: Completed tool call with error for {tool.name}")
            delattr(tool, "_maxim_tool_call")
        
        return None


def instrument_google_adk(maxim_logger: Logger, debug: bool = False):
    """
    Patches Google ADK's core components to add comprehensive logging and tracing.

    This wrapper enhances Google ADK with:
    - Detailed operation tracing for agent runs
    - Token usage tracking for LLM calls
    - Tool execution monitoring
    - Span-based operation tracking
    - Error handling and reporting

    Args:
        maxim_logger (Logger): A Maxim Logger instance for handling the tracing and logging operations.
        debug (bool): If True, show INFO and DEBUG logs. If False, show only WARNING and ERROR logs.
    """
    print(f"[MaximSDK] instrument_google_adk called! GOOGLE_ADK_AVAILABLE={GOOGLE_ADK_AVAILABLE}, Gemini={Gemini}")
    scribe().info(f"[MaximSDK] instrument_google_adk called! GOOGLE_ADK_AVAILABLE={GOOGLE_ADK_AVAILABLE}, Gemini={Gemini}")
    
    if not GOOGLE_ADK_AVAILABLE:
        scribe().warning("[MaximSDK] Google ADK not available. Skipping instrumentation.")
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
            if isinstance(self, Runner):
                if _global_maxim_trace.get() is None:
                    trace_id = str(uuid.uuid4())
                    scribe().debug(f"[MaximSDK] Creating trace for runner [{trace_id}]")

                    trace = maxim_logger.trace({
                        "id": trace_id,
                        "name": "Google ADK Runner",
                        "tags": {
                            "app_name": getattr(self, "app_name", "unknown"),
                            "agent_name": getattr(self.agent, "name", "unknown") if hasattr(self, "agent") else "unknown",
                        },
                        "input": str(processed_inputs.get("new_message", "-")),
                    })

                    trace_token = _global_maxim_trace.set(trace)

            elif isinstance(self, BaseAgent):
                current_trace = _global_maxim_trace.get()
                if current_trace:
                    span_id = str(uuid.uuid4())
                    scribe().debug(f"[MaximSDK] Agent span [{span_id}] for '{self.name}'")

                    span = current_trace.span({
                        "id": span_id,
                        "name": f"Agent: {self.name}",
                        "tags": {
                            "agent_id": str(getattr(self, "id", "")),
                            "agent_type": type(self).__name__,
                        },
                    })

                    _agent_span_ids[id(self)] = span_id

            elif isinstance(self, BaseLlm):
                current_trace = _global_maxim_trace.get()
                print(f"[MaximSDK] LLM method called! Current trace: {current_trace is not None}")
                scribe().info(f"[MaximSDK] LLM method called! Current trace: {current_trace is not None}")
                if current_trace:
                    generation_id = str(uuid.uuid4())
                    setattr(self, "_maxim_generation_id", generation_id)
                    print(f"[MaximSDK] LLM generation [{generation_id}]")
                    scribe().info(f"[MaximSDK] LLM generation [{generation_id}]")

                    model_info = extract_model_info(self)
                    
                    # Extract and convert messages
                    messages = []
                    if args and len(args) > 0:
                        llm_request = args[0] if isinstance(args[0], LlmRequest) else None
                        if llm_request and hasattr(llm_request, 'contents'):
                            messages = convert_messages_to_maxim_format(llm_request.contents)

                    print(f"[MaximSDK] Model: {model_info.get('model', 'unknown')}, Messages: {len(messages)}")
                    scribe().info(f"[MaximSDK] Model: {model_info.get('model', 'unknown')}, Messages: {len(messages)}")

                    llm_generation_config = GenerationConfigDict({
                        "id": generation_id,
                        "name": "LLM Call",
                        "provider": model_info.get("provider", "google"),
                        "model": model_info.get("model", "unknown"),
                        "messages": messages,
                    })

                    generation = current_trace.generation(llm_generation_config)
                    setattr(self, "_input", messages)
                    print(f"[MaximSDK] Created generation for LLM call")
                    scribe().info(f"[MaximSDK] Created generation for LLM call")
                else:
                    generation = None

            elif isinstance(self, BaseTool):
                current_trace = _global_maxim_trace.get()
                if current_trace:
                    tool_id = str(uuid.uuid4())
                    tool_details = extract_tool_details(self)
                    tool_args = str(processed_inputs.get("args", processed_inputs))

                    if tool_details.get('name') == "RagTool":
                        scribe().debug(f"[MaximSDK] RAG: Retrieval tool [{tool_id}]")
                        trace_id = current_trace.id if hasattr(current_trace, 'id') else current_trace.get('id', 'unknown')
                        tool_call = current_trace.retrieval({
                            "id": tool_id,
                            "name": f"RAG: {self.name}",
                            "tags": {"span_id": trace_id},
                        })
                        setattr(tool_call, "_input", processed_inputs.get("query", ""))
                        tool_call.input(processed_inputs.get("query", ""))
                    else:
                        scribe().debug(f"[MaximSDK] TOOL: {self.name} [{tool_id}]")
                        trace_id = current_trace.id if hasattr(current_trace, 'id') else current_trace.get('id', 'unknown')
                        tool_call = current_trace.tool_call({
                            "id": tool_id,
                            "name": f"{tool_details['name'] or self.name}",
                            "description": tool_details["description"] or self.description,
                            "args": tool_args,
                            "tags": {"tool_id": tool_id, "span_id": trace_id},
                        })
                        setattr(tool_call, "_input", tool_args)

            scribe().debug(f"[MaximSDK] --- Calling: {final_op_name} ---")

            try:
                # Call the original method
                output = original_method(self, *args, **kwargs)
                
                # Handle async generators (for streaming responses)
                if hasattr(output, '__aiter__'):
                    print(f"[MaximSDK] Handling async generator for {final_op_name}")
                    scribe().debug(f"[MaximSDK] Handling async generator for {final_op_name}")
                    
                    async def async_generator_wrapper():
                        try:
                            final_result = None
                            chunk_count = 0
                            async for chunk in output:
                                chunk_count += 1
                                final_result = chunk
                                yield chunk
                            
                            print(f"[MaximSDK] Generator exhausted after {chunk_count} chunks. final_result={final_result is not None}")
                            scribe().info(f"[MaximSDK] Generator exhausted after {chunk_count} chunks")
                            
                            # Process the final result for generations
                            if isinstance(self, BaseLlm):
                                print(f"[MaximSDK] Is BaseLlm instance: True, generation={generation is not None}, final_result={final_result is not None}")
                                if generation and final_result:
                                    print(f"[MaximSDK] Processing LLM result after generator completion")
                                    scribe().info(f"[MaximSDK] Processing LLM result after generator completion")
                                    await process_llm_result(self, generation, final_result)
                                elif not generation:
                                    print(f"[MaximSDK] No generation object to process!")
                                elif not final_result:
                                    print(f"[MaximSDK] No final_result to process!")
                        except Exception as e:
                            print(f"[MaximSDK] Error in async generator wrapper: {e}")
                            traceback.print_exc()
                            scribe().error(f"[MaximSDK] Error in async generator wrapper: {e}")
                            if generation:
                                generation.error({"message": str(e)})
                            raise
                    
                    return async_generator_wrapper()
                
                # Handle async responses (coroutines)
                elif hasattr(output, '__await__'):
                    print(f"[MaximSDK] Handling coroutine for {final_op_name}")
                    scribe().debug(f"[MaximSDK] Handling coroutine for {final_op_name}")
                    
                    async def async_wrapper():
                        try:
                            result = await output
                            
                            # Process the result for generations
                            if isinstance(self, BaseLlm) and generation:
                                print(f"[MaximSDK] Processing LLM result after coroutine completion")
                                await process_llm_result(self, generation, result)
                            
                            return result
                        except Exception as e:
                            print(f"[MaximSDK] Error in async wrapper: {e}")
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
                if tool_call:
                    if isinstance(tool_call, Retrieval):
                        tool_call.output(processed_output)
                        scribe().debug("[MaximSDK] RAG: Completed retrieval")
                    else:
                        tool_call.result(processed_output)
                        scribe().debug("[MaximSDK] TOOL: Completed tool call")

                # Complete generations for sync calls
                if generation and not hasattr(output, '__await__'):
                    process_llm_result_sync(self, generation, processed_output)

                # Complete spans
                if span and not isinstance(self, Runner):
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

                if trace:
                    trace.add_error({"message": str(e)})
                    trace.end()
                    if trace_token is not None:
                        _global_maxim_trace.reset(trace_token)
                    else:
                        _global_maxim_trace.set(None)
                    maxim_logger.flush()

                raise

        return maxim_wrapper

    async def process_llm_result(llm_self, generation, result):
        """Process LLM result and handle tool calls for async calls."""
        print(f"[MaximSDK] Processing LLM result - type: {type(result)}")
        usage_info = extract_usage_from_response(result)
        model_info = getattr(llm_self, "_model_info", extract_model_info(llm_self))
        
        # Extract tool calls from the response
        tool_calls = extract_tool_calls_from_response(result)
        
        # Extract meaningful content from the response
        content = extract_content_from_response(result)
        
        print(f"[MaximSDK] Usage: {usage_info}")
        print(f"[MaximSDK] Model: {model_info.get('model', 'unknown')}")
        print(f"[MaximSDK] Content length: {len(content) if content else 0}")
        print(f"[MaximSDK] Tool calls: {len(tool_calls) if tool_calls else 0}")
        
        # Get generation ID safely (handle both object and dict)
        gen_id = generation.id if hasattr(generation, 'id') else generation.get('id', str(uuid.uuid4()))
        
        # Create generation result
        gen_result = {
            "id": f"gen_{gen_id}",
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
            maxim_tool_calls = []
            for tool_call in tool_calls:
                # Ensure tool_call_id is never None
                tool_call_id = tool_call.get("tool_call_id") or str(uuid.uuid4())
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                maxim_tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": str(tool_args)
                    }
                }
                maxim_tool_calls.append(maxim_tool_call)
                print(f"[MaximSDK] Tool call: {tool_name} (ID: {tool_call_id}, Args: {tool_args})")
                scribe().info(f"[MaximSDK] Tool call: {tool_name} (ID: {tool_call_id})")
            
            gen_result["choices"][0]["message"]["tool_calls"] = maxim_tool_calls
            print(f"[MaximSDK] Added {len(tool_calls)} tool calls to generation result")
            scribe().info(f"[MaximSDK] Added {len(tool_calls)} tool calls to generation result")
        
        generation.result(gen_result)
        print(f"[MaximSDK] Generation result recorded!")
        scribe().debug("[MaximSDK] GEN: Completed async generation")

    def process_llm_result_sync(llm_self, generation, result):
        """Process LLM result and handle tool calls for sync calls."""
        usage_info = extract_usage_from_response(result)
        model_info = getattr(llm_self, "_model_info", {})
        
        # Extract tool calls from the response
        tool_calls = extract_tool_calls_from_response(result)
        
        # Extract meaningful content from the response
        content = extract_content_from_response(result)
        
        # Get generation ID safely (handle both object and dict)
        gen_id = generation.id if hasattr(generation, 'id') else generation.get('id', str(uuid.uuid4()))
        
        # Create generation result
        gen_result = {
            "id": f"gen_{gen_id}",
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
            maxim_tool_calls = []
            for tool_call in tool_calls:
                # Ensure tool_call_id is never None
                tool_call_id = tool_call.get("tool_call_id") or str(uuid.uuid4())
                maxim_tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", "unknown"),
                        "arguments": str(tool_call.get("args", {}))
                    }
                }
                maxim_tool_calls.append(maxim_tool_call)
            
            gen_result["choices"][0]["message"]["tool_calls"] = maxim_tool_calls
            scribe().debug(f"[MaximSDK] Found {len(tool_calls)} tool calls in model response")
        
        generation.result(gen_result)
        scribe().debug("[MaximSDK] GEN: Completed sync generation")

    # Patch Runner methods
    if Runner is not None:
        runner_methods = ["run_async", "run"]
        for method_name in runner_methods:
            if hasattr(Runner, method_name):
                original_method = getattr(Runner, method_name)
                # Skip if already patched (idempotency guard)
                if getattr(original_method, "__maxim_patched__", False):
                    scribe().debug(f"[MaximSDK] Skipping already patched Runner.{method_name}")
                    continue
                wrapper = make_maxim_wrapper(
                    original_method,
                    f"google.adk.Runner.{method_name}",
                    input_processor=google_adk_postprocess_inputs,
                    display_name_fn=get_agent_display_name,
                )
                setattr(Runner, method_name, wrapper)
                # Mark as patched to prevent double-wrapping
                setattr(getattr(Runner, method_name), "__maxim_patched__", True)
                scribe().info(f"[MaximSDK] Patched google.adk.Runner.{method_name}")

    # Patch InMemoryRunner methods (only if they override Runner methods)
    if InMemoryRunner is not None:
        inmemory_runner_methods = ["run_async", "run"]
        for method_name in inmemory_runner_methods:
            # Only patch if method is defined in InMemoryRunner's own __dict__ (not inherited from Runner)
            if hasattr(InMemoryRunner, method_name) and method_name in getattr(InMemoryRunner, "__dict__", {}):
                original_method = getattr(InMemoryRunner, method_name)
                # Skip if already patched (idempotency guard)
                if getattr(original_method, "__maxim_patched__", False):
                    scribe().debug(f"[MaximSDK] Skipping already patched InMemoryRunner.{method_name}")
                    continue
                wrapper = make_maxim_wrapper(
                    original_method,
                    f"google.adk.InMemoryRunner.{method_name}",
                    input_processor=google_adk_postprocess_inputs,
                    display_name_fn=get_agent_display_name,
                )
                setattr(InMemoryRunner, method_name, wrapper)
                # Mark as patched to prevent double-wrapping
                setattr(getattr(InMemoryRunner, method_name), "__maxim_patched__", True)
                scribe().info(f"[MaximSDK] Patched google.adk.InMemoryRunner.{method_name}")
            else:
                scribe().debug(f"[MaximSDK] Skipping InMemoryRunner.{method_name} (inherited from Runner)")

    # Patch BaseLlm methods
    try:
        if BaseLlm is not None:
            llm_methods = ["generate_content_async"]
            for method_name in llm_methods:
                if hasattr(BaseLlm, method_name):
                    original_method = getattr(BaseLlm, method_name)
                    # Skip if already patched (idempotency guard)
                    if getattr(original_method, "__maxim_patched__", False):
                        scribe().debug(f"[MaximSDK] Skipping already patched BaseLlm.{method_name}")
                        continue
                    wrapper = make_maxim_wrapper(
                        original_method,
                        f"google.adk.BaseLlm.{method_name}",
                        input_processor=lambda inputs: dictify(inputs),
                        output_processor=lambda output: dictify(output),
                    )
                    setattr(BaseLlm, method_name, wrapper)
                    # Mark as patched to prevent double-wrapping
                    setattr(getattr(BaseLlm, method_name), "__maxim_patched__", True)
                    print(f"[MaximSDK] Patched google.adk.BaseLlm.{method_name}")
                    scribe().info(f"[MaximSDK] Patched google.adk.BaseLlm.{method_name}")
    except Exception as e:
        print(f"[MaximSDK] ERROR patching BaseLlm: {e}")
        scribe().error(f"[MaximSDK] ERROR patching BaseLlm: {e}")
    
    # Also patch Gemini class specifically (it's the concrete implementation)
    try:
        print(f"[MaximSDK] About to check Gemini class... Gemini={Gemini}")
        scribe().info(f"[MaximSDK] About to check Gemini class... Gemini={Gemini}")
        if Gemini is not None:
            print(f"[MaximSDK] Gemini class found, attempting to patch...")
            scribe().info(f"[MaximSDK] Gemini class found, attempting to patch...")
            llm_methods = ["generate_content_async"]
            for method_name in llm_methods:
                print(f"[MaximSDK] Checking if Gemini has {method_name}: {hasattr(Gemini, method_name)}")
                if hasattr(Gemini, method_name):
                    original_method = getattr(Gemini, method_name)
                    # Skip if already patched (idempotency guard)
                    if getattr(original_method, "__maxim_patched__", False):
                        scribe().debug(f"[MaximSDK] Skipping already patched Gemini.{method_name}")
                        continue
                    print(f"[MaximSDK] Original method type: {type(original_method)}")
                    wrapper = make_maxim_wrapper(
                        original_method,
                        f"google.adk.Gemini.{method_name}",
                        input_processor=lambda inputs: dictify(inputs),
                        output_processor=lambda output: dictify(output),
                    )
                    setattr(Gemini, method_name, wrapper)
                    # Mark as patched to prevent double-wrapping
                    setattr(getattr(Gemini, method_name), "__maxim_patched__", True)
                    print(f"[MaximSDK] Patched google.adk.Gemini.{method_name}")
                    scribe().info(f"[MaximSDK] Patched google.adk.Gemini.{method_name}")
                else:
                    print(f"[MaximSDK] Gemini does not have {method_name}")
                    scribe().info(f"[MaximSDK] Gemini does not have {method_name}")
        else:
            print(f"[MaximSDK] Gemini class is None!")
            scribe().info(f"[MaximSDK] Gemini class is None!")
    except Exception as e:
        print(f"[MaximSDK] ERROR patching Gemini: {e}")
        scribe().error(f"[MaximSDK] ERROR patching Gemini: {e}")

    # Patch BaseTool methods
    if BaseTool is not None:
        tool_methods = ["run_async"]
        for method_name in tool_methods:
            if hasattr(BaseTool, method_name):
                original_method = getattr(BaseTool, method_name)
                # Skip if already patched (idempotency guard)
                if getattr(original_method, "__maxim_patched__", False):
                    scribe().debug(f"[MaximSDK] Skipping already patched BaseTool.{method_name}")
                    continue
                wrapper = make_maxim_wrapper(
                    original_method,
                    f"google.adk.BaseTool.{method_name}",
                    input_processor=lambda inputs: dictify(inputs),
                    output_processor=lambda output: dictify(output),
                    display_name_fn=get_tool_display_name,
                )
                setattr(BaseTool, method_name, wrapper)
                # Mark as patched to prevent double-wrapping
                setattr(getattr(BaseTool, method_name), "__maxim_patched__", True)
                scribe().info(f"[MaximSDK] Patched google.adk.BaseTool.{method_name}")

    scribe().info("[MaximSDK] Finished applying patches to Google ADK.")


def create_maxim_plugin(maxim_logger: Logger, debug: bool = False) -> MaximInstrumentationPlugin:
    """Create a Maxim instrumentation plugin for Google ADK."""
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError(
            "google-adk is required. Install via `pip install google-adk` or "
            "an optional extra (e.g., maxim-py[google-adk])."
        )
    return MaximInstrumentationPlugin(maxim_logger, debug)
