"""Maxim integration for Google Agent Development Kit (ADK)."""

import contextvars
import logging
import uuid
from time import time
from typing import Union, Optional, Any

try:
    from google.adk.tools.base_tool import BaseTool
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
    Trace,
)
from ...scribe import scribe
from .utils import (
    extract_tool_details,
    extract_usage_from_response,
    extract_model_info,
    convert_messages_to_maxim_format,
    extract_content_from_response,
    extract_tool_calls_from_response,
)

_last_llm_usages = {}
_agent_span_ids = {}
_session_trace = None  # Global session trace
_current_tool_call_span = None  # Current tool call span (for nesting agents under tool calls)
_open_tool_calls = {}  # Dict mapping tool_name -> list of tool_call_info (for nesting agents)
_current_maxim_session = None  # Current Maxim session (not Google ADK session)
_current_maxim_session_id = None  # Current Maxim session ID
_global_maxim_logger = None  # Global Maxim logger for end_maxim_session
_global_callbacks = {}  # Global callbacks from instrument_google_adk

_global_maxim_trace: contextvars.ContextVar[Union[Trace, None]] = (
    contextvars.ContextVar("maxim_trace_context_var", default=None)
)
_current_agent_span: contextvars.ContextVar[Union[Any, None]] = (
    contextvars.ContextVar("maxim_agent_span_context_var", default=None)
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

    def __init__(
        self, 
        maxim_logger: Logger, 
        debug: bool = False, 
        parent_trace=None, 
        parent_agent_span=None,
        # User-provided callbacks
        before_generation_callback=None,
        after_generation_callback=None,
        before_trace_callback=None,
        after_trace_callback=None,
        before_span_callback=None,
        after_span_callback=None,
    ):
        if GOOGLE_ADK_AVAILABLE:
            super().__init__(name="maxim_instrumentation")
        else:
            super().__init__(name="non_maxim_instrumentation")
        self.maxim_logger = maxim_logger
        self.debug = debug
        self._parent_trace = parent_trace  # Parent trace for nested agents
        self._parent_agent_span = parent_agent_span  # Parent agent span for nested agents
        self._trace = None
        self._agent_span = None  # Main agent span
        self._spans = {}
        self._generations = {}
        self._tool_calls = {}
        self._request_to_generation = {}  # Map request ID to generation ID
        self._pending_tool_calls = {}  # Store tool calls by ID for matching with agent invocations
        self._last_llm_response = None  # Store last LLM response for trace output
        self._trace_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # Aggregate usage

        # Store user callbacks
        self._before_generation_callback = before_generation_callback
        self._after_generation_callback = after_generation_callback
        self._before_trace_callback = before_trace_callback
        self._after_trace_callback = after_trace_callback
        self._before_span_callback = before_span_callback
        self._after_span_callback = after_span_callback

    def _extract_user_input_from_messages(self, messages) -> str:
        """Extract user input from the last user message in the conversation."""
        try:
            if not messages:
                return "Agent run started"
            
            # Find the last user message by iterating in reverse
            for message in reversed(messages):
                if isinstance(message, dict):
                    role = message.get('role', '')
                    content = message.get('content', '')
                    if role == 'user' and content:
                        # Handle both string and list content
                        if isinstance(content, str):
                            return content
                        elif isinstance(content, list):
                            # Extract text from content parts
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get('type') == 'text':
                                    text_parts.append(part.get('text', ''))
                                elif isinstance(part, str):
                                    text_parts.append(part)
                            if text_parts:
                                return " ".join(text_parts)
                elif hasattr(message, 'role') and hasattr(message, 'content'):
                    if message.role == 'user' and message.content:
                        return str(message.content)
            
            return "Agent run started"
            
        except Exception as e:
            scribe().warning(f"[MaximSDK] Failed to extract user input from messages: {e}")
            return "Agent run started"

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        """Start a trace for the agent run."""
        global _session_trace, _current_maxim_session, _current_maxim_session_id
        
        print(f"🔵 [MaximSDK] before_run_callback called for agent: {invocation_context.agent.name}")
        scribe().info(f"[MaximSDK] before_run_callback called for agent: {invocation_context.agent.name}")
        
        # Check if we're a nested agent (called via AgentTool)
        is_nested = self._parent_trace is not None
        
        if is_nested:
            # Use parent trace for nested agents
            self._trace = self._parent_trace
            scribe().info(f"[MaximSDK] Using parent trace for nested agent: {invocation_context.agent.name}")
        else:
            # For Google ADK, user input is not available at session/trace level
            # It will be extracted from the first LLM generation
            user_input = "Agent run started"  # Placeholder until we get the actual input from LLM generation
            
            # Create Maxim session if it doesn't exist (persists across multiple traces)
            if _current_maxim_session is None:
                _current_maxim_session_id = str(uuid.uuid4())
                _current_maxim_session = self.maxim_logger.session({
                    "id": _current_maxim_session_id,
                    "name": "Google ADK Session",
                    "tags": {
                        "app_name": getattr(invocation_context, 'app_name', 'unknown'),
                        "agent_name": invocation_context.agent.name,
                    },
                })
                print(f"📦 [MaximSDK] Created Maxim session: {_current_maxim_session_id}")
                scribe().info(f"[MaximSDK] Created Maxim session: {_current_maxim_session_id}")
            
            # Create trace within the session (new trace for each agent run)
            # Always create a new trace for each root agent run to ensure unique inputs
            # Call before_trace_callback if provided
            if self._before_trace_callback:
                try:
                    await self._before_trace_callback(invocation_context=invocation_context, user_input=user_input)
                except Exception as e:
                    scribe().error(f"[MaximSDK] Error in before_trace_callback: {e}")
            
            trace_id = str(uuid.uuid4())
            _session_trace = _current_maxim_session.trace({
                "id": trace_id,
                "name": f"Trace-{trace_id}",
                "tags": {
                    "agent_name": invocation_context.agent.name,
                    "invocation_id": invocation_context.invocation_id,
                },
                "input": user_input,
            })
            _global_maxim_trace.set(_session_trace)
            print(f"🔗 [MaximSDK] Created new trace in session: {trace_id}")
            scribe().info(f"[MaximSDK] Created new trace in session: {trace_id}")
            
            self._trace = _session_trace
        
        # Determine parent context for agent span
        # Note: We always use a Span as parent context (not ToolCall) because only Spans have .span() method
        if is_nested and self._parent_agent_span:
            # For nested agents, use the parent agent span as parent context
            agent_name = invocation_context.agent.name
            parent_context = self._parent_agent_span
            
            # Check if there's a matching tool call to mark it as used
            if agent_name in self._pending_tool_calls:
                del self._pending_tool_calls[agent_name]  # Remove from pending
                scribe().info(f"[MaximSDK] Nested agent '{agent_name}' matched with tool call, using parent agent span as parent")
            else:
                scribe().info(f"[MaximSDK] Nested agent '{agent_name}' using parent agent span")
        else:
            # Root agent uses trace as parent
            parent_context = self._trace
        
        # Call before_span_callback if provided
        if self._before_span_callback:
            try:
                await self._before_span_callback(invocation_context=invocation_context, parent_context=parent_context)
            except Exception as e:
                scribe().error(f"[MaximSDK] Error in before_span_callback: {e}")
        
        # Create an agent span to contain all agent operations
        agent_span_id = str(uuid.uuid4())
        self._agent_span = parent_context.span({
            "id": agent_span_id,
            "name": f"Agent: {invocation_context.agent.name}",
            "tags": {
                "agent_type": invocation_context.agent.__class__.__name__,
                "invocation_id": invocation_context.invocation_id,
                "is_nested": str(is_nested),
            },
        })
        scribe().info(f"[MaximSDK] Created agent span: {agent_span_id} for {invocation_context.agent.name}")
        
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """End the trace after agent run completes."""
        # Use the last LLM response we captured
        agent_output = self._last_llm_response
        
        # End the agent span first
        if self._agent_span:
            # Call after_span_callback if provided
            if self._after_span_callback:
                try:
                    await self._after_span_callback(
                        invocation_context=invocation_context, 
                        agent_span=self._agent_span,
                        agent_output=agent_output
                    )
                except Exception as e:
                    scribe().error(f"[MaximSDK] Error in after_span_callback: {e}")
            
            self._agent_span.end()
            scribe().info(f"[MaximSDK] Ended agent span for {invocation_context.agent.name}")
            
            # If this was a nested agent, also complete its tool call span
            agent_name = invocation_context.agent.name
            if agent_name in self._tool_calls:
                tool_call_span = self._tool_calls[agent_name]
                # Use captured output or fallback
                output_text = agent_output or 'Agent completed'
                tool_call_span.result(str(output_text))
                del self._tool_calls[agent_name]
                scribe().info(f"[MaximSDK] Completed tool call span for nested agent: {agent_name}")
            
            self._agent_span = None
        
        # For root agent (not nested), set output and end the trace
        if not self._parent_trace and self._trace:
            global _session_trace
            
            # Set the trace output if we have it
            if agent_output:
                self._trace.set_output(agent_output)
                print(f"📝 [MaximSDK] Set trace output ({len(agent_output)} chars)")
                scribe().info("[MaximSDK] Set trace output")
            
            # Set trace-level metadata (tokens, cost, etc.)
            if self._trace_usage["total_tokens"] > 0:
                # Add usage as tags/metadata on the trace
                self._trace.add_tag("prompt_tokens", str(self._trace_usage["prompt_tokens"]))
                self._trace.add_tag("completion_tokens", str(self._trace_usage["completion_tokens"]))
                self._trace.add_tag("total_tokens", str(self._trace_usage["total_tokens"]))
                print(f"💰 [MaximSDK] Set trace usage: {self._trace_usage}")
                scribe().info("[MaximSDK] Set trace usage metadata")
            
            # Call after_trace_callback if provided
            if self._after_trace_callback:
                try:
                    await self._after_trace_callback(
                        invocation_context=invocation_context,
                        trace=self._trace,
                        agent_output=agent_output,
                        trace_usage=self._trace_usage
                    )
                except Exception as e:
                    scribe().error(f"[MaximSDK] Error in after_trace_callback: {e}")
            
            self._trace.end()
            self.maxim_logger.flush()
            print("✅ [MaximSDK] Trace completed and flushed to Maxim")
            scribe().info("[MaximSDK] Trace completed and flushed to Maxim")
            
            # Reset for next interaction - always reset _session_trace so each agent run gets a new trace
            _session_trace = None
            _global_maxim_trace.set(None)
            self._trace_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        scribe().debug("[MaximSDK] Completed after_run_callback")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """Instrument LLM request before sending to model."""
        print(f"🟡 [MaximSDK] before_model_callback called, trace={self._trace}, agent_span={self._agent_span}")
        scribe().info("[MaximSDK] before_model_callback called")
        
        # Use agent span if available, otherwise fall back to trace
        parent_context = self._agent_span if self._agent_span else self._trace
        
        if not parent_context:
            print("[MaximSDK] WARNING: No trace or agent span available for LLM call")
            scribe().warning("[MaximSDK] No trace or agent span available for LLM call")
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
        
        # Extract user input from the first user message in the conversation
        user_input = self._extract_user_input_from_messages(messages)
        if user_input and user_input != "Agent run started":
            # Update the trace with the actual user input
            if self._trace:
                self._trace.set_input(user_input)
                print(f"[MaximSDK] Updated trace input: {user_input[:100]}")
                scribe().info(f"[MaximSDK] Updated trace input: {user_input[:100]}")
            
            # Also update the session input if available
            global _current_maxim_session
            if _current_maxim_session and hasattr(_current_maxim_session, 'set_input'):
                _current_maxim_session.set_input(user_input)
                print(f"[MaximSDK] Updated session input: {user_input[:100]}")
                scribe().info(f"[MaximSDK] Updated session input: {user_input[:100]}")
        
        # Determine agent name for better context
        agent_name = "Unknown Agent"
        try:
            if hasattr(callback_context, 'agent') and callback_context.agent:
                agent_name = getattr(callback_context.agent, 'name', 'Unknown Agent')
            elif self._agent_span and hasattr(self._agent_span, 'name'):
                # Extract agent name from agent span name (format: "Agent: agent_name")
                span_name = self._agent_span.name
                if span_name.startswith("Agent: "):
                    agent_name = span_name.replace("Agent: ", "")
        except Exception as e:
            scribe().debug(f"[MaximSDK] Could not extract agent name: {e}")
        
        # Call before_generation_callback if provided
        if self._before_generation_callback:
            try:
                await self._before_generation_callback(
                    callback_context=callback_context,
                    llm_request=llm_request,
                    model_info=model_info,
                    messages=messages
                )
            except Exception as e:
                scribe().error(f"[MaximSDK] Error in before_generation_callback: {e}")
        
        # Create generation config with agent context
        generation_name = f"{agent_name} - LLM Generation" if agent_name != "Unknown Agent" else "LLM Generation"
        generation_config = GenerationConfigDict({
            "id": generation_id,
            "name": generation_name,
            "provider": model_info.get("provider", "google"),
            "model": model_info.get("model", "unknown"),
            "messages": messages,
        })
        
        # Add agent name as metadata
        if agent_name != "Unknown Agent":
            generation_config["tags"] = {"agent_name": agent_name}

        # Create generation within the agent span (or trace if no agent span)
        generation = parent_context.generation(generation_config)
        self._generations[generation_id] = generation
        
        # Store mapping from request to generation
        self._request_to_generation[request_id] = generation_id
        
        context_type = "agent span" if self._agent_span else "trace"
        print(f"[MaximSDK] Created generation: {generation_id} in {context_type}")
        scribe().info(f"[MaximSDK] Created generation: {generation_id} in {context_type}")
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """Instrument LLM response after receiving from model."""
        print("[MaximSDK] after_model_callback called")
        scribe().info("[MaximSDK] after_model_callback called")
        
        if not self._trace:
            print("[MaximSDK] WARNING: No trace available for LLM response")
            scribe().warning("[MaximSDK] No trace available for LLM response")
            return None

        # Try multiple ways to find the matching generation
        generation = None
        generation_id = None
        
        # Method 1: Try to match by request ID
        request_id = id(callback_context.llm_request) if hasattr(callback_context, 'llm_request') and callback_context.llm_request else None
        if request_id:
            generation_id = self._request_to_generation.get(request_id)
            if generation_id:
                generation = self._generations.get(generation_id)
                print(f"[MaximSDK] Found generation by request ID: {generation_id}")
        
        # Method 2: If only one generation is pending, use that
        if not generation and len(self._generations) == 1:
            generation_id = list(self._generations.keys())[0]
            generation = self._generations[generation_id]
            print(f"[MaximSDK] Using single pending generation: {generation_id}")
        
        # Method 3: Try to find the most recent generation for this agent
        if not generation and self._agent_span:
            # Get the most recently created generation
            if self._generations:
                generation_id = list(self._generations.keys())[-1]
                generation = self._generations[generation_id]
                print(f"[MaximSDK] Using most recent generation: {generation_id}")
        
        if not generation:
            print(f"[MaximSDK] WARNING: No generation found (request_id={request_id}, pending={len(self._generations)})")
            scribe().warning("[MaximSDK] No generation found for request")
            return None

        # Extract usage information
        usage_info = extract_usage_from_response(llm_response)
        
        # Extract content from response
        content = extract_content_from_response(llm_response)
        
        # Extract tool calls from response
        tool_calls = extract_tool_calls_from_response(llm_response)
        
        print("\n========== PLUGIN after_model_callback ==========")
        print(f"[MaximSDK] Usage info: {usage_info}")
        print(f"[MaximSDK] Content length: {len(content) if content else 0}")
        print(f"[MaximSDK] Tool calls detected: {len(tool_calls) if tool_calls else 0}")
        if tool_calls:
            print("[MaximSDK] TOOL CALLS FOUND IN THIS LLM RESPONSE:")
            for tc in tool_calls:
                print(f"  - {tc.get('name')} (ID: {tc.get('tool_call_id')})")
        else:
            print("[MaximSDK] NO TOOL CALLS in this LLM response")
        print("========== END PLUGIN after_model_callback ==========\n")
        
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
            parent_context = self._agent_span if self._agent_span else self._trace
            
            # Get agent name for display
            agent_name = "Unknown Agent"
            try:
                if self._agent_span and hasattr(self._agent_span, 'name'):
                    span_name = self._agent_span.name
                    if span_name.startswith("Agent: "):
                        agent_name = span_name.replace("Agent: ", "")
            except Exception as e:
                scribe().debug(f"[MaximSDK] Could not extract agent name: {e}")
            
            if parent_context:
                for tool_call in tool_calls:
                    tool_call_id = tool_call.get("tool_call_id", str(uuid.uuid4()))
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    
                    # Create tool call span
                    try:
                        tool_display_name = f"{tool_name}"  # Don't prefix with agent name - cleaner
                        
                        tool_span = parent_context.tool_call({
                            "id": tool_call_id,
                            "name": tool_display_name,
                            "description": f"Tool/Agent call: {tool_name}",
                            "args": str(tool_args),
                            "tags": {
                                "tool_call_id": tool_call_id,
                                "from_llm_response": "true",
                                "calling_agent": agent_name,
                                "tool_name": tool_name,
                            },
                        })
                        
                        # Store tool call span - will be used by nested agents
                        self._tool_calls[tool_call_id] = tool_span
                        self._pending_tool_calls[tool_name] = tool_span  # Store by name for matching
                        
                        print(f"[MaximSDK] Created tool call span for '{tool_name}' (ID: {tool_call_id})")
                        scribe().info(f"[MaximSDK] Created tool call span: {tool_call_id} for {tool_name}")
                    except Exception as e:
                        scribe().error(f"[MaximSDK] Failed to create tool call span: {e}")
            else:
                scribe().warning("[MaximSDK] No parent context available for tool call spans")
        
        generation.result(gen_result)
        print(f"[MaximSDK] Completed generation: {generation_id}")
        scribe().debug(f"[MaximSDK] GEN: Completed generation: {generation_id}")
        
        # Call after_generation_callback if provided
        if self._after_generation_callback:
            try:
                await self._after_generation_callback(
                    callback_context=callback_context,
                    llm_response=llm_response,
                    generation=generation,
                    generation_result=gen_result,
                    usage_info=usage_info,
                    content=content,
                    tool_calls=tool_calls
                )
            except Exception as e:
                scribe().error(f"[MaximSDK] Error in after_generation_callback: {e}")
        
        # Store the content for trace output
        self._last_llm_response = content
        
        # Aggregate token usage for the trace
        if usage_info:
            self._trace_usage["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
            self._trace_usage["completion_tokens"] += usage_info.get("completion_tokens", 0)
            self._trace_usage["total_tokens"] += usage_info.get("total_tokens", 0)
            print(f"📊 [MaximSDK] Aggregated usage: {self._trace_usage}")
        
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
        # Skip AgentTool - already handled in after_model_callback
        from google.adk.tools.agent_tool import AgentTool
        if isinstance(tool, AgentTool):
            scribe().debug(f"[MaximSDK] Skipping before_tool_callback for AgentTool: {tool.name} (already created in after_model_callback)")
            return None
        
        # Use agent span if available, otherwise fall back to trace
        parent_context = self._agent_span if self._agent_span else self._trace
        
        if not parent_context:
            scribe().warning("[MaximSDK] No trace or agent span available for tool call")
            return None

        tool_id = str(uuid.uuid4())
        tool_details = extract_tool_details(tool)
        tool_args_str = str(tool_args)
        
        # Get parent context ID safely
        parent_id = parent_context.id if hasattr(parent_context, 'id') else parent_context.get('id', 'unknown')
        
        # Extract agent name for context
        agent_name = None
        try:
            if hasattr(tool_context, 'agent') and tool_context.agent:
                agent_name = getattr(tool_context.agent, 'name', None)
        except Exception:
            pass
        
        # Create tool name with agent context
        tool_name = tool_details.get('name', tool.name)
        if agent_name:
            tool_display_name = f"{agent_name} - {tool_name}"
        else:
            tool_display_name = tool_name
        
        # Create tool call span within the agent span (or trace if no agent span)
        tool_call = parent_context.tool_call({
            "id": tool_id,
            "name": tool_display_name,
            "description": tool_details.get("description", tool.description),
            "args": tool_args_str,
            "tags": {
                "tool_id": str(id(tool)),
                "parent_id": parent_id,
                "agent_name": agent_name if agent_name else "unknown"
            },
        })
        
        self._tool_calls[tool_id] = tool_call
        setattr(tool, "_maxim_tool_call", tool_call)
        
        context_type = "agent span" if self._agent_span else "trace"
        scribe().info(f"[MaximSDK] Created tool call: {tool_id} for {tool.name} in {context_type}")
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

        # For AgentTool, the tool call span was already created in after_model_callback
        # and will be completed when the nested agent completes
        # So we only handle non-agent tools here
        from google.adk.tools.agent_tool import AgentTool
        if isinstance(tool, AgentTool):
            scribe().debug(f"[MaximSDK] Skipping after_tool_callback for AgentTool: {tool.name} (handled by nested agent)")
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


def instrument_google_adk(
    maxim_logger: Logger, 
    debug: bool = False,
    before_generation_callback=None,
    after_generation_callback=None,
    before_trace_callback=None,
    after_trace_callback=None,
    before_span_callback=None,
    after_span_callback=None,
):
    """
    Single-line instrumentation for Google ADK with automatic plugin injection.

    This instrumentation enhances Google ADK with:
    - Detailed operation tracing for agent runs
    - Token usage tracking for LLM calls
    - Tool execution monitoring
    - Span-based operation tracking
    - Error handling and reporting
    - Proper nesting of sub-agents under tool calls
    - Automatic plugin injection into all Runner instances
    - User-provided callbacks for custom logic

    Args:
        maxim_logger (Logger): A Maxim Logger instance for handling the tracing and logging operations.
        debug (bool): If True, show INFO and DEBUG logs. If False, show only WARNING and ERROR logs.
        before_generation_callback: Optional async callback called before LLM generation.
            Signature: async def(callback_context, llm_request, model_info, messages)
        after_generation_callback: Optional async callback called after LLM generation.
            Signature: async def(callback_context, llm_response, generation, generation_result, usage_info, content, tool_calls)
        before_trace_callback: Optional async callback called before trace creation.
            Signature: async def(invocation_context, user_input)
        after_trace_callback: Optional async callback called after trace completion.
            Signature: async def(invocation_context, trace, agent_output, trace_usage)
        before_span_callback: Optional async callback called before span creation.
            Signature: async def(invocation_context, parent_context)
        after_span_callback: Optional async callback called after span completion.
            Signature: async def(invocation_context, agent_span, agent_output)
    
    Usage:
        from maxim import Maxim
        from maxim.logger.google_adk import instrument_google_adk
        
        # Basic usage
        maxim = Maxim()
        instrument_google_adk(maxim.logger())
        
        # With callbacks
        async def my_before_generation(callback_context, llm_request, model_info, messages):
            print(f"About to call {model_info['model']}")
        
        instrument_google_adk(
            maxim.logger(),
            before_generation_callback=my_before_generation
        )
        
        # Now all runners automatically have Maxim tracing
        runner = InMemoryRunner(agent=my_agent)
    """
    global _global_maxim_logger, _global_callbacks
    
    # Store callbacks globally
    _global_callbacks = {
        "before_generation": before_generation_callback,
        "after_generation": after_generation_callback,
        "before_trace": before_trace_callback,
        "after_trace": after_trace_callback,
        "before_span": before_span_callback,
        "after_span": after_span_callback,
    }
    
    print(f"[MaximSDK] instrument_google_adk called! GOOGLE_ADK_AVAILABLE={GOOGLE_ADK_AVAILABLE}")
    scribe().info(f"[MaximSDK] instrument_google_adk called! GOOGLE_ADK_AVAILABLE={GOOGLE_ADK_AVAILABLE}")
    
    if not GOOGLE_ADK_AVAILABLE:
        scribe().warning("[MaximSDK] Google ADK not available. Skipping instrumentation.")
        return
    
    # Store logger globally for end_maxim_session
    _global_maxim_logger = maxim_logger
    
    # Patch Runner.__init__ to automatically inject Maxim plugin
    try:
        from google.adk.runners import Runner
        
        _original_runner_init = Runner.__init__
        
        def _patched_runner_init(self, **kwargs):
            """Patched Runner.__init__ that automatically adds Maxim plugin."""
            plugins = kwargs.get('plugins') or []  # Handle None case
            
            # Check if Maxim plugin is already in the list
            has_maxim_plugin = any(isinstance(p, MaximInstrumentationPlugin) for p in plugins)
            
            if not has_maxim_plugin:
                # Create and add Maxim plugin with callbacks
                maxim_plugin = MaximInstrumentationPlugin(
                    maxim_logger, 
                    debug,
                    before_generation_callback=_global_callbacks.get("before_generation"),
                    after_generation_callback=_global_callbacks.get("after_generation"),
                    before_trace_callback=_global_callbacks.get("before_trace"),
                    after_trace_callback=_global_callbacks.get("after_trace"),
                    before_span_callback=_global_callbacks.get("before_span"),
                    after_span_callback=_global_callbacks.get("after_span"),
                )
                plugins = list(plugins) + [maxim_plugin]
                kwargs['plugins'] = plugins
                scribe().info("[MaximSDK] Auto-injected Maxim plugin into Runner")
            
            # Call original __init__
            _original_runner_init(self, **kwargs)
        
        Runner.__init__ = _patched_runner_init
        scribe().info("[MaximSDK] Patched Runner.__init__ for automatic plugin injection")
        print("[MaximSDK] ✅ Patched Runner for automatic Maxim plugin injection")
        
    except Exception as e:
        scribe().error(f"[MaximSDK] Failed to patch Runner: {e}")
        print(f"[MaximSDK] ⚠️  Failed to patch Runner: {e}")
    
    # Patch AgentTool to properly pass plugin context to nested agents
    try:
        from google.adk.tools.agent_tool import AgentTool
        _original_agent_tool_run_async = AgentTool.run_async
        
        async def _patched_agent_tool_run_async(self, *, args, tool_context):
            """Patched run_async that passes plugin context to nested agents."""
            from google.adk.runners import Runner
            from google.adk.sessions.in_memory_session_service import InMemorySessionService
            from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
            from google.adk.tools._forwarding_artifact_service import ForwardingArtifactService
            from google.genai import types
            from google.adk.agents.llm_agent import LlmAgent
            from google.adk.utils.context_utils import Aclosing
            
            if self.skip_summarization:
                tool_context.actions.skip_summarization = True

            if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
                input_value = self.agent.input_schema.model_validate(args)
                content = types.Content(
                    role='user',
                    parts=[
                        types.Part.from_text(
                            text=input_value.model_dump_json(exclude_none=True)
                        )
                    ],
                )
            else:
                content = types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=args['request'])],
                )
            
            # Get parent plugins and create nested plugin with trace context
            parent_plugins = list(tool_context._invocation_context.plugin_manager.plugins)
            nested_plugins = []
            
            for plugin in parent_plugins:
                if isinstance(plugin, MaximInstrumentationPlugin):
                    # Create a new plugin instance that inherits trace context and callbacks
                    nested_plugin = MaximInstrumentationPlugin(
                        maxim_logger=plugin.maxim_logger,
                        debug=plugin.debug,
                        parent_trace=plugin._trace,  # Pass parent trace
                        parent_agent_span=plugin._agent_span,  # Pass parent agent span
                        # Pass callbacks from parent plugin
                        before_generation_callback=plugin._before_generation_callback,
                        after_generation_callback=plugin._after_generation_callback,
                        before_trace_callback=plugin._before_trace_callback,
                        after_trace_callback=plugin._after_trace_callback,
                        before_span_callback=plugin._before_span_callback,
                        after_span_callback=plugin._after_span_callback,
                    )
                    # Transfer pending tool calls
                    nested_plugin._pending_tool_calls = plugin._pending_tool_calls
                    nested_plugin._tool_calls = plugin._tool_calls
                    nested_plugins.append(nested_plugin)
                    scribe().info(f"[MaximSDK] Created nested plugin for AgentTool: {self.agent.name}")
                else:
                    nested_plugins.append(plugin)
            
            runner = Runner(
                app_name=self.agent.name,
                agent=self.agent,
                artifact_service=ForwardingArtifactService(tool_context),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
                credential_service=tool_context._invocation_context.credential_service,
                plugins=nested_plugins,  # Use nested plugins with trace context
            )

            state_dict = {
                k: v
                for k, v in tool_context.state.to_dict().items()
                if not k.startswith('_adk')  # Filter out adk internal states
            }
            session = await runner.session_service.create_session(
                app_name=self.agent.name,
                user_id=tool_context._invocation_context.user_id,
                state=state_dict,
            )

            last_content = None
            async with Aclosing(
                runner.run_async(
                    user_id=session.user_id, session_id=session.id, new_message=content
                )
            ) as agen:
                async for event in agen:
                    # Forward state delta to parent session.
                    if event.actions.state_delta:
                        tool_context.state.update(event.actions.state_delta)
                    if event.content:
                        last_content = event.content

            if not last_content:
                return ''
            merged_text = '\n'.join(p.text for p in last_content.parts if p.text)
            if isinstance(self.agent, LlmAgent) and self.agent.output_schema:
                tool_result = self.agent.output_schema.model_validate_json(
                    merged_text
                ).model_dump(exclude_none=True)
            else:
                tool_result = merged_text
            return tool_result
        
        # Apply the patch
        AgentTool.run_async = _patched_agent_tool_run_async
        scribe().info("[MaximSDK] Patched AgentTool.run_async to propagate plugin context")
        print("[MaximSDK] ✅ Patched AgentTool for nested agent tracing")
        
    except Exception as e:
        scribe().error(f"[MaximSDK] Failed to patch AgentTool: {e}")
        print(f"[MaximSDK] ⚠️  Failed to patch AgentTool: {e}")
    
    print("\n✅ [MaximSDK] Single-line instrumentation complete!")
    print("📌 All Runner instances will now automatically include Maxim tracing")
    print("📌 Simply create runners as usual: runner = InMemoryRunner(agent=my_agent)")
    scribe().info("[MaximSDK] Google ADK instrumentation complete with automatic plugin injection")


def create_maxim_plugin(
    maxim_logger: Logger, 
    debug: bool = False, 
    parent_trace=None, 
    parent_agent_span=None,
    before_generation_callback=None,
    after_generation_callback=None,
    before_trace_callback=None,
    after_trace_callback=None,
    before_span_callback=None,
    after_span_callback=None,
) -> MaximInstrumentationPlugin:
    """
    Create a Maxim instrumentation plugin for Google ADK.
    
    Note: When using instrument_google_adk(), you don't need to call this function.
    This is only for advanced use cases where you want to manually create plugins.
    
    Args:
        maxim_logger: The Maxim logger instance
        debug: Enable debug logging
        parent_trace: Optional parent trace for nested agents
        parent_agent_span: Optional parent agent span for nested agents
        before_generation_callback: Optional async callback before LLM generation
        after_generation_callback: Optional async callback after LLM generation
        before_trace_callback: Optional async callback before trace creation
        after_trace_callback: Optional async callback after trace completion
        before_span_callback: Optional async callback before span creation
        after_span_callback: Optional async callback after span completion
    
    Returns:
        MaximInstrumentationPlugin instance
    """
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError(
            "google-adk is required. Install via `pip install google-adk` or "
            "an optional extra (e.g., maxim-py[google-adk])."
        )
    return MaximInstrumentationPlugin(
        maxim_logger, 
        debug, 
        parent_trace, 
        parent_agent_span,
        before_generation_callback=before_generation_callback,
        after_generation_callback=after_generation_callback,
        before_trace_callback=before_trace_callback,
        after_trace_callback=after_trace_callback,
        before_span_callback=before_span_callback,
        after_span_callback=after_span_callback,
    )


# Legacy wrapper code removed - we use plugin-only approach now
# This eliminates conflicts and duplicate spans

_old_make_maxim_wrapper_code_removed = """
Old wrapper-based instrumentation has been removed to avoid conflicts with the plugin approach.
If you need the old behavior, please use an earlier version of the SDK.
The plugin-based approach is cleaner and avoids duplicate spans.
"""


def end_maxim_session(maxim_logger=None):
    """
    Explicitly end the current Maxim session and flush all data.
    Call this at the end of your application/session to ensure all traces are sent to Maxim.
    
    Args:
        maxim_logger: Optional Logger instance. If not provided, will use the global logger from instrument_google_adk().
    """
    global _current_maxim_session, _current_maxim_session_id, _session_trace, _global_maxim_logger
    
    if _current_maxim_session_id and _current_maxim_session:
        from maxim.scribe import scribe
        
        # End any open trace first
        if _session_trace:
            try:
                _session_trace.end()
                print("✅ [MaximSDK] Ended open trace")
                scribe().info("[MaximSDK] Ended open trace")
            except Exception as e:
                scribe().warning(f"[MaximSDK] Failed to end trace: {e}")
            _session_trace = None
        
        # End the session
        try:
            _current_maxim_session.end()
            print(f"📦 [MaximSDK] Ended Maxim session: {_current_maxim_session_id}")
            scribe().info(f"[MaximSDK] Ended Maxim session: {_current_maxim_session_id}")
        except Exception as e:
            scribe().warning(f"[MaximSDK] Failed to end session: {e}")
        
        # Flush the logger to ensure session is sent
        logger_to_flush = maxim_logger or _global_maxim_logger
        if logger_to_flush:
            logger_to_flush.flush()
            print("💾 [MaximSDK] Flushed logger after ending session")
            scribe().info("[MaximSDK] Flushed logger after ending session")
        
        # Reset globals
        _current_maxim_session = None
        _current_maxim_session_id = None
        
        return True
    else:
        print("⚠️  [MaximSDK] No active Maxim session to end")
        return False
