import contextvars
import functools
import importlib
import inspect
import logging
import traceback
import uuid
from time import time
from typing import Union

from crewai import LLM, Agent, Crew, Flow, Task
from crewai.agents.agent_builder.base_agent import BaseAgent

try:
    from crewai.tools import BaseTool
except ImportError:
    # Backward compatibility for older versions
    from crewai.tools.agent_tools.agent_tools import BaseTool

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
from .utils import (
    crew_kickoff_postprocess_inputs,
    crewai_postprocess_inputs,
    dictify,
    extract_tool_details,
    extract_tool_name,
    get_agent_display_name,
    get_task_display_name,
)

_last_llm_usages = {}
_task_span_ids = {}

_global_maxim_trace: contextvars.ContextVar[Union[Trace, None]] = (
    contextvars.ContextVar("maxim_trace_context_var", default=None)
)


def get_log_level(debug: bool) -> int:
    """
    Set logging level based on debug flag.
    debug=False: Only WARNING and ERROR logs
    debug=True: INFO and DEBUG logs
    """
    return logging.DEBUG if debug else logging.WARNING

class MaximEvalConfig:
    """Maxim eval config.

    This class represents a maxim eval config.
    """

    evaluators: list[str]
    additional_variables: list[dict[str, str]]

    def __init__(self):
        self.evaluators = []
        self.additional_variables = []

class MaximUsageCallback:
    """Maxim usage callback.

    This class represents a usage callback.
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
                f"[MaximSDK] Callback captured usage: {_last_llm_usages[self.generation_id] is not None}"
            )
        else:
            _last_llm_usages[self.generation_id] = None
            scribe().debug(
                "[MaximSDK] Callback did not find usage info in response_obj"
            )


# --- Wrapper Factory for _handle_non_streaming_response ---
def make_handle_non_streaming_wrapper(original_method):
    """Make a handle non streaming wrapper.

    This function wraps the original method to capture usage.
    """

    @functools.wraps(original_method)
    def handle_non_streaming_wrapper(self, *args, **kwargs):
        _maxim_generation_id = getattr(self, "_maxim_generation_id", None)
        if not isinstance(_maxim_generation_id, str):
            scribe().warning(
                "[MaximSDK] No generation ID found for LLM call. Skipping usage capture."
            )
            return original_method(self, *args, **kwargs)

        custom_callback = MaximUsageCallback(_maxim_generation_id)

        # Try to find callbacks in args or kwargs
        callbacks = None
        if len(args) > 1:
            callbacks = args[1]
        elif 'callbacks' in kwargs:
            callbacks = kwargs['callbacks']

        # Ensure callbacks is a list and add our custom one
        current_callbacks = callbacks if callbacks is not None else []
        if not isinstance(current_callbacks, list):  # Safety check
            scribe().warning(
                "[MaximSDK] Original callbacks is not a list, creating new list."
            )
            current_callbacks = []
        current_callbacks.append(custom_callback)

        # Update callbacks in the arguments
        if len(args) > 1:
            # Update in positional arguments
            args_list = list(args)
            args_list[1] = current_callbacks
            args = tuple(args_list)
        elif 'callbacks' in kwargs:
            # Update in keyword arguments
            kwargs['callbacks'] = current_callbacks
        else:
            # Add as keyword argument if not present
            kwargs['callbacks'] = current_callbacks

        # Debug logging to understand the call signature
        scribe().debug(f"[MaximSDK] _handle_non_streaming_response called with {len(args)} args and {len(kwargs)} kwargs")
        scribe().debug(f"[MaximSDK] Args: {args}")
        scribe().debug(f"[MaximSDK] Kwargs keys: {list(kwargs.keys())}")

        # Call the original method with the augmented callbacks
        result = original_method(self, *args, **kwargs)
        return result

    return handle_non_streaming_wrapper


def instrument_crewai(maxim_logger: Logger, debug: bool = False):
    """
    Patches CrewAI's core components (Crew, Agent, Task, Flow, LLM) to add comprehensive logging and tracing.

    This wrapper enhances CrewAI with:
    - Detailed operation tracing for Crew, Flow, and Task executions
    - Token usage tracking for LLM calls
    - Tool execution monitoring
    - Span-based operation tracking
    - Error handling and reporting

    The patching is done by wrapping key methods like:
    - Crew.kickoff
    - Agent.execute_task
    - Task.execute_sync
    - LLM.call and _handle_non_streaming_response
    - Tool._run methods

    Args:
        maxim_logger (Logger): A Maxim Logger instance for handling the tracing and logging operations.
        debug (bool): If True, show INFO and DEBUG logs. If False, show only WARNING and ERROR logs.
    """
    global logger

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
            global _task_span_ids
            global _last_llm_usages

            # Combine args and kwargs into a dictionary for processing
            bound_args = {}
            processed_inputs = {}
            final_op_name = base_op_name

            try:
                sig = inspect.signature(original_method)
                bound_values = sig.bind(self, *args, **kwargs)
                bound_values.apply_defaults()
                bound_args = bound_values.arguments

                # Process inputs if processor is provided
                processed_inputs = bound_args
                if input_processor:
                    try:
                        processed_inputs = input_processor(bound_args)
                    except Exception as e:
                        scribe().debug(
                            f"[MaximSDK] Failed to process inputs for {base_op_name}: {e}"
                        )

                if display_name_fn:
                    try:
                        final_op_name = display_name_fn(processed_inputs)
                    except Exception as e:
                        scribe().debug(
                            f"[MaximSDK] Failed to generate display name for {base_op_name}: {e}"
                        )

            except Exception as e:
                scribe().debug(
                    f"[MaximSDK] Failed to bind/process inputs for {base_op_name}: {e}"
                )
                # Fallback for inputs display
                processed_inputs = {"self": self, "args": args, "kwargs": kwargs}

            trace: Union[Trace, None] = None
            span: Union[Span, None] = None
            generation: Union[Generation, None] = None
            tool_call: Union[ToolCall, Retrieval, None] = None
            planner_span: Union[Span, None] = None
            trace_token: Union[contextvars.Token[Union[Trace, None]], None] = None

            if isinstance(self, Flow):
                if _global_maxim_trace.get() is None:
                    trace_id = str(uuid.uuid4())
                    scribe().debug(f"Creating trace for flow [{trace_id}]")

                    trace = maxim_logger.trace(
                        {
                            "id": trace_id,
                            "name": "Flow Kickoff",
                            "tags": {
                                "flow_id": str(getattr(self, "id", "")),
                                "flow_name": final_op_name,
                            },
                            "input": str(processed_inputs["inputs"] or "-"),
                        }
                    )

                    # Store the token for later restoration
                    trace_token = _global_maxim_trace.set(trace)

            elif isinstance(self, Crew):
                if _global_maxim_trace.get() is None:
                    trace_id = str(uuid.uuid4())

                    if original_method.__name__ == "kickoff_for_each":
                        scribe().debug(
                            "[MaximSDK] Creating trace for crew kickoff_for_each"
                        )

                        trace = maxim_logger.trace(
                            {
                                "id": trace_id,
                                "name": "Crew Kickoff For Each",
                                "tags": {
                                    "crew_id": str(getattr(self, "id", "")),
                                },
                                "input": str(processed_inputs["inputs"] or "-"),
                            }
                        )

                    else:
                        scribe().debug(
                            f"[MaximSDK] Creating trace for crew [{trace_id}]"
                        )

                        trace = maxim_logger.trace(
                            {
                                "id": trace_id,
                                "name": "Crew Kickoff",
                                "tags": {
                                    "crew_id": str(getattr(self, "id", "")),
                                    "crew_name": final_op_name,
                                },
                                "input": str(processed_inputs["inputs"] or "-"),
                            }
                        )

                        # Attach trace to all tasks in the crew
                        if hasattr(self, "tasks") and self.tasks:
                            scribe().debug(
                                f"[MaximSDK] Attaching trace to {len(self.tasks)} tasks"
                            )
                            for task in self.tasks:
                                setattr(task, "_trace", trace)
                                scribe().debug(
                                    f"[MaximSDK] Task: {task.description[:40]}{'...' if len(task.description) > 40 else ''}"
                                )

                    trace_token = _global_maxim_trace.set(trace)
                else:
                    span_id = str(uuid.uuid4())

                    if original_method.__name__ == "kickoff_for_each":
                        scribe().debug(
                            f"[MaximSDK] Attaching span to crew kickoff_for_each [{span_id}]"
                        )

                        span = _global_maxim_trace.get().span(
                            {
                                "id": span_id,
                                "name": "Crew Kickoff For Each",
                                "tags": {
                                    "crew_id": str(getattr(self, "id", "")),
                                },
                            }
                        )
                    else:
                        scribe().debug(f"[MaximSDK] Attaching span to crew [{span_id}]")

                        span = _global_maxim_trace.get().span(
                            {
                                "id": span_id,
                                "name": "Crew Kickoff",
                                "tags": {
                                    "crew_id": str(getattr(self, "id", "")),
                                    "crew_name": final_op_name,
                                },
                            }
                        )

                        # Attach trace to all tasks in the crew
                        if hasattr(self, "tasks") and self.tasks:
                            scribe().debug(
                                f"[MaximSDK] Attaching trace to {len(self.tasks)} tasks"
                            )
                            for task in self.tasks:
                                setattr(task, "_span", span)
                                scribe().debug(
                                    f"[MaximSDK] Task: {task.description[:40]}{'...' if len(task.description) > 40 else ''}"
                                )

            elif isinstance(self, Task):
                # Get the trace from the crew
                trace = getattr(self, "_trace", None)
                if not isinstance(trace, Trace):
                    trace = None

                span = getattr(self, "_span", None)
                if not isinstance(span, Span):
                    span = None

                span_id = str(uuid.uuid4())

                scribe().debug(
                    f"[MaximSDK] Task span [{span_id}] for '{self.name or self.description[:40]}'"
                )

                task_span_config = SpanConfigDict(
                    {
                        "id": span_id,
                        "name": f"Task: {self.name or 'None'}",
                        "tags": {"task_id": str(getattr(self, "id", ""))},
                    }
                )

                if not trace and not span:  # Will happen for Planner tasks
                    # TODO get crew info for flow workflows crew placement
                    scribe().debug(
                        "[MaximSDK] Parent trace/span not found, creating new task span on global trace"
                    )
                    if _global_maxim_trace.get() is None:
                        scribe().warning(
                            "[MaximSDK] No global trace found, skipping logging"
                        )
                        return
                    span = _global_maxim_trace.get().span(task_span_config)
                else:
                    if trace:
                        scribe().debug(f"[MaximSDK] Found parent trace: {trace.id}")
                        span = trace.span(task_span_config)
                    else:
                        if not span:
                            scribe().warning(
                                "[MaximSDK] No parent span found, skipping logging"
                            )
                            return
                        scribe().debug(f"[MaximSDK] Found parent span: {span.id}")
                        span = span.span(task_span_config)

                metadata = {
                    "name": self.name or "None",
                    "description": self.description or "None",
                    "expected_output": self.expected_output or "None",
                }

                if self.output_file:
                    metadata["output_file"] = self.output_file

                if self.output_json:
                    metadata["output_json"] = self.output_json

                if self.max_retries:
                    metadata["max_retries"] = self.max_retries

                if self.tools:
                    metadata["tools"] = [tool.name for tool in self.tools]

                span.add_metadata(metadata)

                _task_span_ids[self.id] = span_id

                if self.agent is not None:
                    scribe().debug(
                        f"[MaximSDK] Attaching span to agent [{self.agent.id}]"
                    )
                    setattr(self.agent, "_span", span)
                else:
                    # Check if agent is provided in args/kwargs
                    agent_from_kwargs = kwargs.get("agent", None) if kwargs else None
                    agent_from_args = None
                    args_list = list(args)  # Convert to list to allow modification

                    if not agent_from_kwargs and args:
                        # If first arg is agent, use it (common pattern in execute_sync/execute_core)
                        if len(args) > 0 and isinstance(args[0], BaseAgent):
                            agent_from_args = args[0]

                    # Use whichever agent we found
                    agent_to_use = agent_from_kwargs or agent_from_args

                    if agent_to_use:
                        scribe().debug(
                            f"[MaximSDK] Found agent in args, attaching span to agent [{agent_to_use.role}]"
                        )
                        setattr(agent_to_use, "_span", span)

                        # Update the agent in its original location
                        if agent_from_kwargs:
                            kwargs["agent"] = agent_to_use
                        elif agent_from_args:
                            args_list[0] = agent_to_use
                            args = tuple(args_list)  # Convert back to tuple
                    else:
                        scribe().warning("[MaximSDK] Task has no agent assigned")

                trace = None

            elif isinstance(self, Agent):
                span = getattr(self, "_span", None)
                if not isinstance(span, Span):
                    span = None

                span_id = str(uuid.uuid4())

                eval_config = MaximEvalConfig()
                if (
                    hasattr(self, "config")
                    and self.config is not None
                    and isinstance(self.config, dict)
                ):
                    maxim_config = self.config.get("maxim-eval", {})
                    if maxim_config is not None and isinstance(maxim_config, dict):
                        eval_config.evaluators = maxim_config.get("evaluators", [])
                        eval_config.additional_variables = maxim_config.get("additional_variables", [])

                    # avoid passing maxim-eval config to the original method
                    self.config.pop("maxim-eval", None)  # Using pop to safely remove the key

                agent_span_config = SpanConfigDict(
                    {
                        "id": span_id,
                        "name": f"Agent: {self.role}",
                        "tags": {
                            "agent_id": str(getattr(self, "id", "")),
                            "evaluators_attached": "true" if eval_config.evaluators else "false"
                        },
                    }
                )

                skip_logging = False

                if span:
                    scribe().debug(
                        f"[MaximSDK] Agent span [{span_id}] for '{self.role}'"
                    )
                    span = span.span(agent_span_config)
                else:
                    scribe().debug("[MaximSDK] Agent has no span, checking task")

                    # First check args/kwargs for task
                    task: Union[Task, None] = None
                    if len(args) > 0:
                        for arg in args:
                            if isinstance(arg, Task):
                                task = arg
                                break

                    if not task and kwargs:  # Check kwargs if task not found in args
                        for kwarg_value in kwargs.values():
                            if isinstance(kwarg_value, Task):
                                task = kwarg_value
                                break

                    # Fallback to agent_executor.task if no task found in args or kwargs
                    if not task and hasattr(self, "agent_executor"):
                        task = self.agent_executor.task

                    if task:
                        span_id = _task_span_ids.get(task.id)
                        if span_id:
                            span = maxim_logger.span_add_sub_span(
                                span_id, agent_span_config
                            )
                        else:
                            scribe().debug(
                                f"[MaximSDK] No span found for task {task.id}, creating new task span"
                            )
                            # Create a new task span since none exists
                            task_span_id = str(uuid.uuid4())
                            task_config = SpanConfigDict(
                                {
                                    "id": task_span_id,
                                    "name": f"Task: {task.name or task.description}",
                                    "tags": {
                                        "task_id": str(task.id),
                                    },
                                }
                            )
                            if (
                                _global_maxim_trace.get() is not None
                            ):  # TODO check for check in flows
                                span = _global_maxim_trace.get().span(task_config)
                                _task_span_ids[task.id] = task_span_id
                                # Now create the agent span as a child of the task span
                                span = span.span(agent_span_config)
                            else:
                                scribe().debug(
                                    "[MaximSDK] No global trace found, skipping logging"
                                )
                                skip_logging = True
                    else:
                        scribe().warning(
                            f"[MaximSDK] Agent {self.role} has no task or span, skipping logging"
                        )
                        skip_logging = True

                if not skip_logging:
                    if hasattr(self, "llm") and self.llm:
                        scribe().debug(
                            f"[MaximSDK] LLM: {getattr(self.llm, 'model', 'unknown')}"
                        )
                    if isinstance(self.llm, LLM):
                        setattr(self.llm, "_span", span)
                        setattr(self.llm, "_eval_config", eval_config)

                    # Check for tools in both agent's tools attribute and in execute_task arguments
                    tools_to_set_context = []

                    # Add tools from agent's tools attribute
                    if hasattr(self, "tools") and self.tools:
                        # Filter for BaseTool instances from agent's tools
                        for tool in self.tools:
                            if (
                                isinstance(tool, BaseTool)
                                and tool not in tools_to_set_context
                            ):
                                tools_to_set_context.append(tool)

                    # Check for tools in execute_task arguments
                    if original_method.__name__ == "execute_task":
                        # Check kwargs for tools
                        tools_from_kwargs = (
                            kwargs.get("tools", None) if kwargs else None
                        )
                        if tools_from_kwargs:
                            if tools_from_kwargs is not None:
                                if isinstance(tools_from_kwargs, list):
                                    # Filter for BaseTool instances from kwargs list
                                    for tool in tools_from_kwargs:
                                        if (
                                            isinstance(tool, BaseTool)
                                            and tool not in tools_to_set_context
                                        ):
                                            tools_to_set_context.append(tool)
                                elif (
                                    isinstance(tools_from_kwargs, BaseTool)
                                    and tools_from_kwargs not in tools_to_set_context
                                ):
                                    tools_to_set_context.append(tools_from_kwargs)

                        # Check args for tools (usually after task argument)
                        if len(args) > 1 and isinstance(args[1], (list, BaseTool)):
                            tools_from_args = args[1]
                            if isinstance(tools_from_args, list):
                                # Filter for BaseTool instances from args list
                                for tool in tools_from_args:
                                    if (
                                        isinstance(tool, BaseTool)
                                        and tool not in tools_to_set_context
                                    ):
                                        tools_to_set_context.append(tool)
                            elif (
                                isinstance(tools_from_args, BaseTool)
                                and tools_from_args not in tools_to_set_context
                            ):
                                tools_to_set_context.append(tools_from_args)

                    # Set context for all collected tools
                    if tools_to_set_context:
                        scribe().debug(
                            f"[MaximSDK] Attaching span to {len(tools_to_set_context)} unique valid tools"
                        )
                        for tool in tools_to_set_context:
                            setattr(tool, "_span", span)
                            setattr(tool, "_eval_config", eval_config)

            elif isinstance(self, LLM):
                span = getattr(self, "_span", None)
                if not isinstance(span, Span):
                    span = None

                generation_id = str(uuid.uuid4())
                setattr(self, "_maxim_generation_id", generation_id)
                scribe().debug(f"[MaximSDK] LLM generation [{generation_id}]")

                llm_generation_config = GenerationConfigDict(
                    {
                        "id": generation_id,
                        "name": "LLM Call",
                        "provider": (
                            "anthropic" if self.is_anthropic else "openai"
                        ),  # TODO: Add more providers
                        "model": str(getattr(self, "model", "unknown")),
                        "messages": args[0],
                    }
                )

                if span:
                    llm_generation_config["span_id"] = span.id
                    generation = span.generation(llm_generation_config)

                    span = None
                else:
                    scribe().warning(
                        "[MaximSDK] No parent span found for LLM call, creating new planner span"
                    )
                    if _global_maxim_trace.get() is None:
                        scribe().warning(
                            "[MaximSDK] No global trace found, skipping logging"
                        )
                        return
                    planner_span = _global_maxim_trace.get().span(
                        {
                            "id": str(uuid.uuid4()),
                            "name": "Planner",
                        }
                    )
                    llm_generation_config["span_id"] = planner_span.id

                    generation = planner_span.generation(llm_generation_config)

                setattr(self, "_input", args[0])
            elif isinstance(self, BaseTool):
                span = getattr(self, "_span", None)
                if not isinstance(span, Span):
                    span = None

                if span:
                    tool_id = str(uuid.uuid4())
                    tool_name = extract_tool_name(final_op_name)

                    if tool_name == "RagTool":
                        scribe().debug(f"[MaximSDK] RAG: Retrieval tool [{tool_id}]")
                        tool_call = span.retrieval(
                            {
                                "id": tool_id,
                                "name": f"RAG: {self.name}",
                                "tags": {"span_id": span.id},
                            }
                        )
                        setattr(tool_call, "_input", processed_inputs.get("query", ""))
                        tool_call.input(processed_inputs.get("query", ""))
                    else:
                        scribe().debug(f"[MaximSDK] TOOL: {self.name} [{tool_id}]")

                        tool_details = extract_tool_details(self.description)
                        tool_args = str(tool_details["args"]) if tool_details["args"] is not None else str(processed_inputs)
                        tool_call = span.tool_call(
                            {
                                "id": tool_id,
                                "name": f"{tool_details['name'] or self.name}",
                                "description": tool_details["description"]
                                or self.description,
                                "args": tool_args,
                                "tags": {"tool_id": tool_id, "span_id": span.id},
                            }
                        )
                        setattr(tool_call, "_input", tool_args)

                    span = None
                else:
                    scribe().warning("[MaximSDK] No parent span found for tool call")

            scribe().debug(f"\n[MaximSDK] --- Calling: {final_op_name} ---")

            try:
                # Call the original method (bound to self)
                output = original_method.__get__(self, self.__class__)(*args, **kwargs)
            except TypeError as e:
                # Handle signature mismatch errors
                if "takes from" in str(e) and "positional arguments but" in str(e):
                    scribe().warning(f"[MaximSDK] Method signature mismatch in {final_op_name}: {e}")
                    scribe().warning("[MaximSDK] Attempting to call original method directly...")
                    try:
                        # Try calling the original method directly without binding
                        output = original_method(*args, **kwargs)
                    except Exception as e2:
                        scribe().error(f"[MaximSDK] Direct call also failed: {e2}")
                        raise e  # Re-raise the original exception
                else:
                    raise e
            except Exception as e:
                traceback.print_exc()
                scribe().error(f"[MaximSDK] {type(e).__name__} in {final_op_name}")

                if tool_call:
                    if isinstance(tool_call, Retrieval):
                        tool_call.output(f"Error occurred while calling tool: {e}")
                        scribe().debug("[MaximSDK] RAG: Completed retrieval with error")
                    else:
                        tool_call.result(f"Error occurred while calling tool: {e}")
                        scribe().debug(
                            "[MaximSDK] TOOL: Completed tool call with error"
                        )

                if generation:
                    generation.error({"message": str(e)})
                    scribe().debug("[MaximSDK] GEN: Completed generation with error")
                    _last_llm_usages.pop(generation.id, None)

                if span:
                    span.add_error({"message": str(e)})
                    span.end()
                    scribe().debug("[MaximSDK] SPAN: Completed span with error")

                if trace and "crewai.Crew" in final_op_name:
                    trace.add_error({"message": str(e)})
                    trace.end()
                    scribe().debug("[MaximSDK] TRACE: Completed trace with error")

                    # Get the stored token and reset the context
                    if trace_token is not None:
                        _global_maxim_trace.reset(trace_token)
                    else:
                        _global_maxim_trace.set(None)

                    maxim_logger.flush()

                raise e  # Re-raise the original exception

            processed_output = output
            if output_processor:
                try:
                    processed_output = output_processor(output)
                except Exception as e:
                    scribe().debug(f"[MaximSDK] Failed to process output: {e}")

            if tool_call:
                if hasattr(self, "_eval_config") and \
                    isinstance(self._eval_config, MaximEvalConfig):
                    # Create a new dictionary for evaluation variables
                    eval_vars = {}
                    # Add any existing variables from config
                    for var_dict in self._eval_config.additional_variables:
                        eval_vars.update(var_dict)
                    # Add input and output
                    if hasattr(tool_call, "_input"):
                        eval_vars["input"] = str(tool_call._input)
                    eval_vars["output"] = processed_output

                    # Evaluate with the variables
                    current_eval_config = getattr(self, "_eval_config", None)
                    if current_eval_config:
                        current_evaluators = current_eval_config.evaluators
                    else:
                        current_evaluators = self._eval_config.evaluators
                    
                    if len(current_evaluators) > 0:
                        tool_call.evaluate().with_evaluators(*current_evaluators).with_variables(eval_vars)

                if isinstance(tool_call, Retrieval):
                    tool_call.output(processed_output)
                    scribe().debug("[MaximSDK] RAG: Completed retrieval")
                else:
                    tool_call.result(processed_output)
                    scribe().debug("[MaximSDK] TOOL: Completed tool call")


            if generation:
                # Create a structured result compatible with GenerationResult
                # Retrieve usage data captured by the callback

                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

                usage_data = _last_llm_usages.get(generation.id)

                if usage_data and isinstance(usage_data, dict):
                    prompt_tokens = usage_data.get("prompt_tokens", 0)
                    completion_tokens = usage_data.get("completion_tokens", 0)
                    total_tokens = usage_data.get("total_tokens", 0)
                    scribe().debug(
                        f"[MaximSDK] GEN: Using captured token usage: P={prompt_tokens}, C={completion_tokens}, T={total_tokens}"
                    )
                else:
                    scribe().debug(
                        f"[MaximSDK] GEN: Using default token usage (0). Captured data: {usage_data}"
                    )

                # Important to set and retrieve internal variables
                if hasattr(self, "_eval_config") and \
                    isinstance(self._eval_config, MaximEvalConfig):
                    # Create a new dictionary for evaluation variables
                    eval_vars = {}
                    # Add any existing variables from config
                    for var_dict in self._eval_config.additional_variables:
                        eval_vars.update(var_dict)
                    # Add input and output
                    if hasattr(generation, "_input"):
                        eval_vars["input"] = str(generation._input)
                    eval_vars["output"] = str(processed_output)

                    # Evaluate with the variables
                    current_eval_config = getattr(self, "_eval_config", None)
                    if current_eval_config:
                        current_evaluators = current_eval_config.evaluators
                    else:
                        current_evaluators = self._eval_config.evaluators
                    
                    if len(current_evaluators) > 0:
                        generation.evaluate().with_evaluators(*current_evaluators).with_variables(eval_vars)
                    else:
                        scribe().warning("[MaximSDK] No evaluators found for generation")

                result = {
                    "id": f"gen_{generation.id}",
                    "object": "chat.completion",
                    "created": int(time()),
                    "model": str(getattr(self, "model", "unknown")),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": str(processed_output),
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

                generation.result(result)
                scribe().debug("[MaximSDK] GEN: Completed generation")

                if planner_span:
                    planner_span.end()
                    scribe().debug("[MaximSDK] PLANNER: Completed planner")

                del _last_llm_usages[generation.id]

            if span:
                scribe().debug("[MaximSDK] SPAN: Completing span")
                span.end()

            if trace:
                scribe().debug(f"[MaximSDK] TRACE: Completing trace [{trace.id}]")
                if processed_output is not None:
                    trace.set_output(str(processed_output))
                trace.end()

                # Get the stored token and reset the context
                if trace_token is not None:
                    _global_maxim_trace.reset(trace_token)
                else:
                    _global_maxim_trace.set(None)

                maxim_logger.flush()

            scribe().debug(f"――― End: {final_op_name} ―――\n")

            return output

        return maxim_wrapper

    # --- 0. Patch LLM._handle_non_streaming_response to capture usage ---
    if LLM is not None and hasattr(LLM, "_handle_non_streaming_response"):
        try:
            original_handle_method = getattr(LLM, "_handle_non_streaming_response")
            wrapper_handle = make_handle_non_streaming_wrapper(original_handle_method)
            setattr(LLM, "_handle_non_streaming_response", wrapper_handle)
            scribe().info(
                "[MaximSDK] Patched crewai.LLM._handle_non_streaming_response to capture usage."
            )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK] Failed to patch _handle_non_streaming_response: {e}. Usage tracking may not work."
            )

    # --- 1. Patch Crew Methods ---
    crew_methods_to_patch = ["kickoff"]
    for method_name in crew_methods_to_patch:
        if hasattr(Crew, method_name):
            original_method = getattr(Crew, method_name)
            op_name = f"crewai.Crew.{method_name}"
            wrapper = make_maxim_wrapper(
                original_method,
                op_name,
                input_processor=crew_kickoff_postprocess_inputs,
            )
            setattr(Crew, method_name, wrapper)
            scribe().info(f"[MaximSDK] Patched crewai.Crew.{method_name} for printing.")

    # --- 2. Patch Agent.execute_task ---
    agent_methods_to_patch = ["execute_task"]
    for method_name in agent_methods_to_patch:
        if hasattr(Agent, method_name):
            original_method = getattr(Agent, method_name)
            op_name = f"crewai.Agent.{method_name}"
            wrapper = make_maxim_wrapper(
                original_method,
                op_name,
                input_processor=crewai_postprocess_inputs,
                display_name_fn=get_agent_display_name,
            )
            setattr(Agent, method_name, wrapper)
            scribe().info(
                f"[MaximSDK] Patched crewai.Agent.{method_name} for printing."
            )

    # --- 3. Patch Task.execute_sync ---
    task_methods_to_patch = ["execute_sync", "execute_async"]
    for method_name in task_methods_to_patch:
        if hasattr(Task, method_name):
            original_method = getattr(Task, method_name)
            op_name = f"crewai.Task.{method_name}"
            wrapper = make_maxim_wrapper(
                original_method,
                op_name,
                input_processor=crewai_postprocess_inputs,
                display_name_fn=get_task_display_name,
            )
            setattr(Task, method_name, wrapper)
            scribe().info(f"[MaximSDK] Patched crewai.Task.{method_name} for printing.")

    # --- 4. Patch LLM.call ---
    if LLM is not None and hasattr(LLM, "call"):
        original_method = getattr(LLM, "call")
        op_name = "crewai.LLM.call"

        wrapper = make_maxim_wrapper(
            original_method,
            op_name,
            input_processor=lambda inputs: dictify(inputs),
            output_processor=lambda output: dictify(output),
        )
        setattr(LLM, "call", wrapper)
        scribe().info("[MaximSDK] Patched crewai.LLM.call for printing.")

    # --- 5. Patch CrewAI Tools ---
    try:
        crewai_tools_module = importlib.import_module("crewai_tools")
        tool_names = [
            t
            for t in dir(crewai_tools_module)
            if "Tool" in t and not t.startswith("Base")
        ]

        for tool_name in tool_names:
            try:
                tool_class = getattr(crewai_tools_module, tool_name)
                if (
                    isinstance(tool_class, type)
                    and issubclass(tool_class, BaseTool)
                    and hasattr(tool_class, "_run")
                ):
                    original_tool_run = getattr(tool_class, "_run")
                    op_name = f"crewai_tools.{tool_name}._run"
                    wrapper = make_maxim_wrapper(
                        original_tool_run,
                        op_name,
                        input_processor=lambda inputs: dictify(inputs),
                        output_processor=lambda output: dictify(output),
                    )
                    setattr(tool_class, "_run", wrapper)
                    scribe().info(f"[MaximSDK] Patched {op_name} for printing.")
            except (AttributeError, TypeError, ImportError) as e:
                scribe().warning(
                    f"[MaximSDK] Skipping patching for tool {tool_name}: {e}"
                )
                continue
    except ImportError:
        scribe().warning(
            "[MaximSDK] crewai_tools or BaseTool not found. Skipping tool patching."
        )
    except Exception as e:
        scribe().error(f"[MaximSDK] ERROR during tool patching: {e}")

    # --- 6. Patch Flow Methods ---
    if Flow is not None:
        # Patch Flow kickoff methods
        flow_kickoff_methods = ["kickoff", "kickoff_async"]
        for method_name in flow_kickoff_methods:
            if hasattr(Flow, method_name):
                original_method = getattr(Flow, method_name)
                op_name = f"crewai.Flow.{method_name}"
                wrapper = make_maxim_wrapper(
                    original_method,
                    op_name,
                    input_processor=lambda inputs: dictify(inputs),
                    output_processor=lambda output: dictify(output),
                )
                setattr(Flow, method_name, wrapper)
                scribe().info(
                    f"[MaximSDK] Patched crewai.Flow.{method_name} for printing."
                )

    scribe().info("[MaximSDK] Finished applying patches to CrewAI.")
