import functools
from uuid import uuid4
import contextvars
from typing import Union, Dict
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
    AgentSetup
)
from llama_index.core.settings import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager

from .. import Logger
from ...scribe import scribe
from ..components import Trace, Span, Generation, GenerationConfigDict 
from .utils import LlamaIndexUtils

_INSTRUMENTED = False
_global_maxim_trace: contextvars.ContextVar[Union[Trace, None]] = (
    contextvars.ContextVar("maxim_trace_context_var", default=None)
)
_agent_spans: Dict[str, Span] = {}
_current_generation: Dict[str, Generation] = {}

# Create a single global token handler
_token_handler = TokenCountingHandler()
Settings.callback_manager = CallbackManager([_token_handler])

def instrument_llamaindex(logger: Logger, debug: bool = False):
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
    global _INSTRUMENTED
    if _INSTRUMENTED:
        scribe().debug("[MaximSDK] LlamaIndex already instrumented")
        return

    def make_maxim_wrapper(original_method, base_op_name):
        print(f"[MaximSDK] Creating wrapper for method: {base_op_name}")
        print(f"[MaximSDK] Original method details: {original_method.__name__} from {original_method.__module__}")

        @functools.wraps(original_method)
        async def maxim_wrapper(self, *args, **kwargs):
            global _global_maxim_trace
            global _agent_spans
            global _current_generation
            global _token_handler

            current_agent = None

            # Create trace if not exists (for root workflow)
            # TODO: Check other types of agents
            if isinstance(self, (AgentWorkflow, FunctionAgent, ReActAgent)) and _global_maxim_trace.get() is None:
                trace_id = str(uuid4())
                print(f"[MaximSDK] Creating new trace with ID: {trace_id}")
                trace_tags = {}
                if isinstance(self, AgentWorkflow):
                    trace_tags["workflow_type"] = "agent_workflow"
                    trace_tags["root_agent"] = getattr(self, "root_agent", "unknown")
                elif isinstance(self, FunctionAgent):
                    trace_tags["agent_type"] = "function_agent"
                elif isinstance(self, ReActAgent):
                    trace_tags["agent_type"] = "react_agent"

                trace = logger.trace({
                    "id": trace_id,
                    "name": "LlamaIndex Workflow" if isinstance(self, AgentWorkflow) else "LlamaIndex Agent",
                    "tags": trace_tags
                })
                _global_maxim_trace.set(trace)

            try:
                # Call the original method
                print(f"[MaximSDK] Calling original method: {original_method.__name__}")
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

                            if agent_name not in _agent_spans:
                                span_id = str(uuid4())
                                _agent_spans[agent_name] = trace.span({
                                    "id": span_id,
                                    "name": f"Agent: {agent_name}",
                                    "tags": {"agent_type": trace_tags["agent_type"] if "agent_type" in trace_tags else "unknown"}
                                })
                                
                        # Handle agent inputs
                        if isinstance(event, AgentInput):
                            current_span = _agent_spans.get(event.current_agent_name)
                            input_agent = self.agents.get(event.current_agent_name) if isinstance(self, AgentWorkflow) else self
                            model_used = "unknown"
                            provider = "unknown"
                            model_parameters = {}
                            if input_agent is not None:
                                model_used = input_agent.llm.metadata.model_name
                                provider = input_agent.llm.__class__.__name__
                                model_parameters = LlamaIndexUtils.parse_model_parameters(input_agent.llm)

                                # ? Could make the below a bit more robust using a map of provider names
                                if provider is not None:
                                    provider = provider.lower()
                                    
                            if current_span:
                                gen_id = str(uuid4())
                                agent_input_messages = LlamaIndexUtils.parse_messages_to_generation_request(event.input)

                                try:
                                    gen_config: GenerationConfigDict = {
                                        "id": gen_id,
                                        "name": "Agent Input",
                                        "provider": provider,
                                        "model": model_used,
                                        "messages": agent_input_messages,
                                        "model_parameters": model_parameters
                                    }
                                    _current_generation[event.current_agent_name] = current_span.generation(gen_config)
                                except Exception as e:
                                    scribe().error(f"[MaximSDK] Error creating generation config: {e}")

                        elif isinstance(event, AgentStream):
                            pass

                        # Handle agent outputs
                        elif isinstance(event, AgentOutput):
                            current_gen = _current_generation.get(event.current_agent_name)
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
                                                "finish_reason": raw_response.get("finish_reason", "stop"),
                                            }
                                        ],
                                        "created": raw_response.get("created", int(time.time())),
                                    })
                                current_gen.end()
                                del _current_generation[event.current_agent_name]
                                # Reset token handler for next agent
                                _token_handler.reset_counts()
                                print("[MaximSDK] Generation completed and cleaned up")

                        # Handle tool calls
                        elif isinstance(event, ToolCall):
                            # This should be an assistant message with tool call
                            if current_agent is not None:
                                current_gen = _current_generation.get(current_agent)

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
                                                            "function": event.tool_kwargs,
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
                                    del _current_generation[current_agent]
                                # Reset token handler for next agent
                                _token_handler.reset_counts()
                                print("[MaximSDK] Generation completed and cleaned up")

                        # Handle tool results
                        elif isinstance(event, ToolCallResult):
                            if current_agent is not None:
                                current_span = _agent_spans.get(current_agent)
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
                                            tool_call_result_dict = ast.literal_eval(str(event.tool_output.content))
                                            tool_call.result(json.dumps(tool_call_result_dict, indent=2))
                                        except (SyntaxError, ValueError) as e:
                                            print(f"Error parsing string: {e}")
                                            print("First 100 characters:", repr(event.tool_output.content))
                                            # If parsing fails, wrap in result object
                                            tool_call.result(json.dumps({"result": str(event.tool_output.content)}, indent=2))

                print("[MaximSDK] Event stream processing completed")
                return handler

            except Exception as e:
                scribe().error(f"[MaximSDK] {type(e).__name__} in {base_op_name}")
                print(f"[MaximSDK] Exception details: {str(e)}")
                
                # Handle errors in current spans/generations
                if _current_generation:
                    print("[MaximSDK] Cleaning up generations due to error")
                    for gen in _current_generation.values():
                        gen.error({"message": str(e)})
                        gen.end()
                    _current_generation.clear()

                if _agent_spans:
                    print("[MaximSDK] Cleaning up spans due to error")
                    for span in _agent_spans.values():
                        span.add_error({"message": str(e)})
                        span.end()
                    _agent_spans.clear()

                trace = _global_maxim_trace.get()
                if trace is not None:
                    print("[MaximSDK] Cleaning up trace due to error")
                    trace.add_error({"message": str(e)})
                    trace.end()
                    _global_maxim_trace.set(None)

                raise e

            finally:
                print(f"――― End: {base_op_name} ―――")

        return maxim_wrapper

    # Patch AgentWorkflow.run
    if hasattr(AgentWorkflow, "run"):
        original_run = AgentWorkflow.run
        print("[MaximSDK] Patching AgentWorkflow.run")
        wrapper = make_maxim_wrapper(original_run, "llama_index.AgentWorkflow.run")
        setattr(AgentWorkflow, "run", wrapper)
        print("[MaximSDK] Successfully patched llama_index.AgentWorkflow.run")

    # Patch FunctionAgent.run
    if hasattr(FunctionAgent, "run"):
        original_run = FunctionAgent.run
        print("[MaximSDK] Patching FunctionAgent.run")
        wrapper = make_maxim_wrapper(original_run, "llama_index.FunctionAgent.run")
        setattr(FunctionAgent, "run", wrapper)
        print("[MaximSDK] Successfully patched llama_index.FunctionAgent.run")
        
    if hasattr(ReActAgent, "run"):
        original_run = ReActAgent.run
        print("[MaximSDK] Patching ReActAgent.run")
        wrapper = make_maxim_wrapper(original_run, "llama_index.ReActAgent.run")
        setattr(ReActAgent, "run", wrapper)
        print("[MaximSDK] Successfully patched llama_index.ReActAgent.run")

    _INSTRUMENTED = True
