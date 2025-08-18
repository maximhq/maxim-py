import functools
from uuid import uuid4
import contextvars
from typing import Union 
import json
import ast
import time

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    AgentOutput,
    FunctionAgent,
    ReActAgent,
    ToolCall,
    ToolCallResult,
    AgentInput,
    AgentStream,
)
from llama_index.core.settings import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager

from maxim.logger.components.generation import GenerationToolCall, GenerationToolCallFunction

from .. import Logger
from ...scribe import scribe
from ..components import Trace, Span, Generation, GenerationConfigDict 
from .utils import LlamaIndexUtils

_INSTRUMENTED = False
_global_maxim_trace: contextvars.ContextVar[Union[Trace, None]] = (
    contextvars.ContextVar("maxim_trace_context_var", default=None)
)
_agent_spans: contextvars.ContextVar[Union[dict[str, Span], None]] = (
    contextvars.ContextVar("maxim_agent_spans_context_var", default=None)
)
_current_generation: contextvars.ContextVar[Union[dict[str, Generation], None]] = (
    contextvars.ContextVar("maxim_current_generation_context_var", default=None)
)

# Create a single global token handler
_token_handler = TokenCountingHandler()
Settings.callback_manager = CallbackManager([_token_handler])

def instrument_llamaindex(logger: Logger, *, debug: bool = False):
    """
    Patches LlamaIndex's core workflow components to add comprehensive logging and tracing.
    
    This wrapper enhances LlamaIndex with:
    - Detailed operation tracing for Agent Workflows
    - Tool execution monitoring
    - Agent state transitions
    - Input/Output tracking
    - Error handling and reporting
    
    Args:
        logger (Logger): A Maxim Logger instance for handling the tracing and logging operations.
        debug (bool): If True, show INFO and DEBUG logs. If False, show only WARNING and ERROR logs.
    """

    # Set logging level based on debug flag
    if debug:
        scribe().set_level("DEBUG")
    else:
        scribe().set_level("WARNING")

    def _check_instrumented():
        global _INSTRUMENTED
        if _INSTRUMENTED:
            scribe().debug("[MaximSDK] LlamaIndex already instrumented")
            return True
        _INSTRUMENTED = True
        return False

    if _check_instrumented():
        return

    def make_maxim_wrapper(original_method, base_op_name):
        scribe().debug(f"[MaximSDK] Creating wrapper for method: {base_op_name}")
        scribe().debug(
            "[MaximSDK] Original method details: "
            f"{original_method.__name__} from {original_method.__module__}"
        )

        @functools.wraps(original_method)
        async def maxim_wrapper(self, *args, **kwargs):
            # Using contextvars, no need for global statements

            current_agent = None
            trace_tags = {}

            # Create trace if not exists (for root workflow)
            if (
                isinstance(self, (AgentWorkflow, FunctionAgent, ReActAgent))
                and _global_maxim_trace.get() is None
            ):
                trace_id = str(uuid4())
                scribe().debug(f"[MaximSDK] Creating new trace with ID: {trace_id}")
                if isinstance(self, AgentWorkflow):
                    trace_tags["workflow_type"] = "agent_workflow"
                    trace_tags["root_agent"] = getattr(self, "root_agent", "unknown")
                elif isinstance(self, FunctionAgent):
                    trace_tags["agent_type"] = "function_agent"
                elif isinstance(self, ReActAgent):
                    trace_tags["agent_type"] = "react_agent"

                trace = logger.trace({
                    "id": trace_id,
                    "name": (
                        "LlamaIndex Workflow"
                        if isinstance(self, AgentWorkflow)
                        else "LlamaIndex Agent"
                    ),
                    "tags": trace_tags,
                })
                _global_maxim_trace.set(trace)

            try:
                # Call the original method
                scribe().debug(f"[MaximSDK] Calling original method: {original_method.__name__}")
                handler = original_method(self, *args, **kwargs)

                # Set up event handling for workflow / agent
                if isinstance(self, (AgentWorkflow, FunctionAgent, ReActAgent)):
                    trace = _global_maxim_trace.get()
                    if trace is None:
                        scribe().warning("[MaximSDK] No trace found for workflow")
                        return handler

                    async for event in handler.stream_events():
                        # Handle agent transitions
                        if hasattr(event, "current_agent_name"):
                            agent_name = event.current_agent_name
                            if(current_agent is None or current_agent != agent_name):
                                current_agent = agent_name

                            agent_spans = _agent_spans.get() or {}
                            if agent_name not in agent_spans:
                                span_id = str(uuid4())
                                agent_spans = agent_spans.copy()
                                agent_spans[agent_name] = trace.span({
                                    "id": span_id,
                                    "name": f"Agent: {agent_name}",
                                    "tags": {
                                        "agent_type": (
                                            trace_tags.get("agent_type", "unknown")
                                        )
                                    },
                                })
                                _agent_spans.set(agent_spans)

                        # Handle agent inputs
                        if isinstance(event, AgentInput):
                            agent_spans = _agent_spans.get() or {}
                            current_span = agent_spans.get(event.current_agent_name)
                            if isinstance(self, AgentWorkflow):
                                input_agent = self.agents.get(event.current_agent_name)
                            else:
                                input_agent = self
                            model_used = "unknown"
                            provider = "unknown"
                            model_parameters = {}
                            if input_agent is not None:
                                model_used = input_agent.llm.metadata.model_name
                                provider = input_agent.llm.__class__.__name__
                                model_parameters = LlamaIndexUtils.parse_model_parameters(
                                    input_agent.llm
                                )

                                if provider is not None:
                                    provider = provider.lower()

                            if current_span:
                                gen_id = str(uuid4())
                                agent_input_messages = LlamaIndexUtils.parse_messages_to_generation_request(
                                    event.input
                                )

                                try:
                                    gen_config: GenerationConfigDict = {
                                        "id": gen_id,
                                        "name": "Agent Input",
                                        "provider": provider,
                                        "model": model_used,
                                        "messages": agent_input_messages,
                                        "model_parameters": model_parameters
                                    }
                                    current_generations = _current_generation.get() or {}
                                    current_generations = current_generations.copy()
                                    current_generations[event.current_agent_name] = current_span.generation(gen_config)
                                    _current_generation.set(current_generations)
                                except Exception as e:
                                    scribe().error(
                                        f"[MaximSDK] Error creating generation config: {e}"
                                    )

                        elif isinstance(event, AgentStream):
                            pass

                        # Handle agent outputs
                        elif isinstance(event, AgentOutput):
                            current_generations = _current_generation.get() or {}
                            current_gen = current_generations.get(event.current_agent_name)
                            if current_gen:
                                if event.response.content:
                                    token_usage = {
                                        "prompt_tokens": _token_handler.prompt_llm_token_count,
                                        "completion_tokens": _token_handler.completion_llm_token_count,
                                        "total_tokens": _token_handler.total_llm_token_count,
                                    }

                                    raw_response = event.raw or {}
                                    current_gen.result({
                                        "id": raw_response.get("id", str(uuid4())),
                                        "usage": token_usage,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": event.response.content,
                                                },
                                                "finish_reason": raw_response.get(
                                                    "finish_reason", "stop"
                                                ),
                                            }
                                        ],
                                        "created": raw_response.get("created", int(time.time())),
                                    })
                                elif event.tool_calls:
                                    tool_calls = []
                                    for tool_call in event.tool_calls:
                                        tool_calls.append(GenerationToolCall(
                                            id=tool_call.tool_id,
                                            type="function",
                                            function=GenerationToolCallFunction(
                                                name=tool_call.tool_name,
                                                arguments=json.dumps(tool_call.tool_kwargs)
                                            )
                                        ))

                                    token_usage = {
                                        "prompt_tokens": _token_handler.prompt_llm_token_count,
                                        "completion_tokens": _token_handler.completion_llm_token_count,
                                        "total_tokens": _token_handler.total_llm_token_count,
                                    }

                                    current_gen.result({
                                        "id": str(uuid4()),
                                        "usage": token_usage,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "message": {
                                                    "role": "assistant",
                                                    "tool_calls": tool_calls
                                                },
                                                "finish_reason": "tool_calls",
                                            }
                                        ],
                                        "created": int(time.time()),
                                    })

                                current_gen.end()
                                current_generations = _current_generation.get() or {}
                                current_generations = current_generations.copy()
                                current_generations.pop(event.current_agent_name, None)
                                _current_generation.set(current_generations)
                                # Reset token handler for next agent
                                _token_handler.reset_counts()
                                scribe().debug("[MaximSDK] Generation completed and cleaned up")

                        # Handle tool calls
                        elif isinstance(event, ToolCall):
                            # This should be an assistant message with tool call
                            if current_agent is not None:
                                current_generations = _current_generation.get() or {}
                                current_gen = current_generations.get(current_agent)

                                if current_gen:
                                    if event.response.content and event.raw:
                                        token_usage = {
                                            "prompt_tokens": _token_handler.prompt_llm_token_count,
                                            "completion_tokens": _token_handler.completion_llm_token_count,
                                            "total_tokens": _token_handler.total_llm_token_count,
                                        }

                                        current_gen.result({
                                            "id": event.raw.get("id", str(uuid4())),
                                            "usage": token_usage,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "message": {
                                                        "role": "assistant",
                                                        "tool_calls": [
                                                            {
                                                                "id": event.tool_id,
                                                                "type": "function",
                                                                "function": {
                                                                    "name": event.tool_name,
                                                                    "arguments": json.dumps(event.tool_kwargs)
                                                                },
                                                            }
                                                        ]
                                                    },
                                                    "finish_reason": event.raw.get("finish_reason", "stop"),
                                                }
                                            ],
                                            "created": event.raw.get("created", int(time.time())),
                                        })
                                    current_gen.end()
                                    if current_agent is not None:
                                        current_generations = _current_generation.get() or {}
                                        current_generations = current_generations.copy()
                                        current_generations.pop(current_agent, None)
                                        _current_generation.set(current_generations)
                                    # Reset token handler for next agent
                                    _token_handler.reset_counts()
                                    scribe().debug("[MaximSDK] Generation completed and cleaned up")

                        # Handle tool results
                        elif isinstance(event, ToolCallResult):
                            if current_agent is not None:
                                agent_spans = _agent_spans.get() or {}
                                current_span = agent_spans.get(current_agent)
                                if current_span:
                                    tool_id = event.tool_id or str(uuid4())
                                    tool_call = current_span.tool_call({
                                        "id": tool_id,
                                        "name": event.tool_name,
                                        "args": json.dumps(event.tool_kwargs)
                                    })
                                    # For simple string outputs, wrap them in a result object
                                    if isinstance(event.tool_output.content, str):
                                        tool_call.result(event.tool_output.content)
                                    else:
                                        try:
                                            tool_call_result_dict = ast.literal_eval(
                                                str(event.tool_output.content)
                                            )
                                            tool_call.result(
                                                json.dumps(
                                                    tool_call_result_dict,
                                                    indent=2,
                                                )
                                            )
                                        except Exception as e:
                                            scribe().error(f"Error parsing string: {e}")
                                            scribe().error(
                                                "First 100 characters: "
                                                f"{repr(event.tool_output.content)}"
                                            )
                                            # If parsing fails, wrap in result object
                                            tool_call.result(
                                                json.dumps(
                                                    {"result": str(event.tool_output.content)},
                                                    indent=2,
                                                )
                                            )

                scribe().debug("[MaximSDK] Event stream processing completed")
                return handler

            except Exception as e:
                scribe().error(f"[MaximSDK] {type(e).__name__} in {base_op_name}")
                scribe().error(f"[MaximSDK] Exception details: {str(e)}")

                # Handle errors in current spans/generations
                current_generations = _current_generation.get() or {}
                if current_generations:
                    scribe().error("[MaximSDK] Cleaning up generations due to error")
                    for gen in current_generations.values():
                        gen.error({"message": f"{e!s}"})
                        gen.end()
                    _current_generation.set({})

                agent_spans = _agent_spans.get() or {}
                if agent_spans:
                    scribe().error("[MaximSDK] Cleaning up spans due to error")
                    for span in agent_spans.values():
                        span.add_error({"message": f"{e!s}"})
                        span.end()
                    _agent_spans.set({})

                trace = _global_maxim_trace.get()
                if trace is not None:
                    scribe().error("[MaximSDK] Cleaning up trace due to error")
                    trace.add_error({"message": f"{e!s}"})
                    trace.end()
                    _global_maxim_trace.set(None)

                raise

            finally:
                scribe().debug(f"――― End: {base_op_name} ―――")

        return maxim_wrapper

    # Patch AgentWorkflow.run
    if hasattr(AgentWorkflow, "run"):
        original_run = AgentWorkflow.run
        scribe().debug("[MaximSDK] Patching AgentWorkflow.run")
        wrapper = make_maxim_wrapper(original_run, "llama_index.AgentWorkflow.run")
        setattr(AgentWorkflow, "run", wrapper)
        scribe().debug("[MaximSDK] Successfully patched llama_index.AgentWorkflow.run")

    # Patch FunctionAgent.run
    if hasattr(FunctionAgent, "run"):
        original_run = FunctionAgent.run
        scribe().debug("[MaximSDK] Patching FunctionAgent.run")
        wrapper = make_maxim_wrapper(original_run, "llama_index.FunctionAgent.run")
        setattr(FunctionAgent, "run", wrapper)
        scribe().debug("[MaximSDK] Successfully patched llama_index.FunctionAgent.run")

    # Patch ReActAgent.run
    if hasattr(ReActAgent, "run"):
        original_run = ReActAgent.run
        scribe().debug("[MaximSDK] Patching ReActAgent.run")
        wrapper = make_maxim_wrapper(original_run, "llama_index.ReActAgent.run")
        setattr(ReActAgent, "run", wrapper)
        scribe().debug("[MaximSDK] Successfully patched llama_index.ReActAgent.run")
