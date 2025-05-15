import importlib
import functools
import inspect
import logging
from .utils import (
    crew_kickoff_postprocess_inputs,
    crewai_postprocess_inputs,
    get_task_display_name,
    get_agent_display_name,
    dictify,
    extract_tool_name,
    extract_tool_details,
)
import uuid
from maxim.logger.components.trace import Trace
from maxim.logger.components.span import Span
from maxim.logger.components.generation import Generation
from maxim.logger.components.tool_call import ToolCall
from maxim.logger.components.retrieval import Retrieval
from maxim import Logger
from crewai import Agent, Task, Crew, Flow, LLM
from crewai.tools.base_tool import BaseTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from time import time

_crewai_patched_print_only = False
_global_maxim_trace: Trace | None = None
_last_llm_usages = {}
_task_span_ids = {}

# Configure basic logging
logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_log_level(debug: bool) -> int:
    """
    Set logging level based on debug flag.
    debug=False: Only WARNING and ERROR logs
    debug=True: INFO and DEBUG logs
    """
    return logging.DEBUG if debug else logging.WARNING


class MaximUsageCallback:
    def __init__(self, generation_id: str):
        self.generation_id = generation_id

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
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
            logger.debug(
                f"Callback captured usage: {_last_llm_usages[self.generation_id] is not None}"
            )
        else:
            _last_llm_usages[self.generation_id] = None
            logger.debug(f"Callback did not find usage info in response_obj")


# --- Wrapper Factory for _handle_non_streaming_response ---
def make_handle_non_streaming_wrapper(original_method):
    @functools.wraps(original_method)
    def handle_non_streaming_wrapper(
        self,
        params: dict,
        callbacks: list | None = None,
        available_functions: dict | None = None,
    ):
        _maxim_generation_id = getattr(self, "_maxim_generation_id", None)
        if not isinstance(_maxim_generation_id, str):
            logger.warning(
                "No generation ID found for LLM call. Skipping usage capture."
            )
            return original_method(self, params, callbacks, available_functions)

        custom_callback = MaximUsageCallback(_maxim_generation_id)

        # Ensure callbacks is a list and add our custom one
        current_callbacks = callbacks if callbacks is not None else []
        if not isinstance(current_callbacks, list):  # Safety check
            logger.warning("Original callbacks is not a list, creating new list.")
            current_callbacks = []
        current_callbacks.append(custom_callback)

        # Call the original method with the augmented callbacks
        result = original_method(self, params, current_callbacks, available_functions)
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
    - Crew.kickoff and kickoff_async
    - Agent.execute_task
    - Task.execute_sync
    - LLM.call and _handle_non_streaming_response
    - Tool._run methods

    Args:
        maxim_logger (Logger): A Maxim Logger instance for handling the tracing and logging operations.
        debug (bool): If True, show INFO and DEBUG logs. If False, show only WARNING and ERROR logs.
    """
    global _crewai_patched_print_only, logger
    if _crewai_patched_print_only:
        logger.info("CrewAI already patched for printing.")
        return

    # Set the logging level based on debug parameter
    logger.setLevel(get_log_level(debug))

    def make_maxim_wrapper(
        original_method,
        base_op_name: str,
        input_processor=None,
        output_processor=None,
        display_name_fn=None,
    ):
        @functools.wraps(original_method)
        def maxim_wrapper(self, *args, **kwargs):

            logger.debug(f"――― Start: {base_op_name} ―――")

            global _global_maxim_trace
            global _task_span_ids

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
                        logger.error(
                            f"Failed to process inputs for {base_op_name}: {e}"
                        )

                if display_name_fn:
                    try:
                        final_op_name = display_name_fn(processed_inputs)
                    except Exception as e:
                        logger.error(
                            f"Failed to generate display name for {base_op_name}: {e}"
                        )

            except Exception as e:
                logger.error(f"Failed to bind/process inputs for {base_op_name}: {e}")
                # Fallback for inputs display
                processed_inputs = {"self": self, "args": args, "kwargs": kwargs}

            trace: Trace | None = None
            span: Span | None = None
            generation: Generation | None = None
            tool_call: ToolCall | Retrieval | None = None
            planner_span: Span | None = None

            if isinstance(self, Flow):
                if _global_maxim_trace is None:
                    trace_id = str(uuid.uuid4())
                    logger.debug(f"Creating trace for flow [{trace_id}]")

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

                    _global_maxim_trace = trace

            elif isinstance(self, Crew):
                if _global_maxim_trace is None:
                    trace_id = str(uuid.uuid4())

                    if original_method.__name__ == "kickoff_for_each":
                        logger.debug(f"Creating trace for crew kickoff_for_each")

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
                        logger.debug(f"Creating trace for crew [{trace_id}]")

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
                            logger.debug(f"Attaching trace to {len(self.tasks)} tasks")
                            for task in self.tasks:
                                setattr(task, "_trace", trace)
                                logger.debug(
                                    f"Task: {task.description[:40]}{'...' if len(task.description) > 40 else ''}"
                                )

                    _global_maxim_trace = trace
                else:
                    span_id = str(uuid.uuid4())

                    if original_method.__name__ == "kickoff_for_each":
                        logger.debug(
                            f"Attaching span to crew kickoff_for_each [{span_id}]"
                        )

                        span = _global_maxim_trace.span(
                            {
                                "id": span_id,
                                "name": "Crew Kickoff For Each",
                                "tags": {
                                    "crew_id": str(getattr(self, "id", "")),
                                },
                            }
                        )
                    else:
                        logger.debug(f"Attaching span to crew [{span_id}]")

                        span = _global_maxim_trace.span(
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
                            logger.debug(f"Attaching trace to {len(self.tasks)} tasks")
                            for task in self.tasks:
                                setattr(task, "_span", span)
                                logger.debug(
                                    f"Task: {task.description[:40]}{'...' if len(task.description) > 40 else ''}"
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

                logger.debug(
                    f"Task span [{span_id}] for '{self.name or self.description[:40]}'"
                )

                config = {
                    "id": span_id,
                    "name": f"Task: {self.name or 'None'}",
                    "tags": {"task_id": str(getattr(self, "id", ""))},
                }

                if not trace and not span:  # Will happen for Planner tasks
                    # TODO get crew info for flow workflows crew placement
                    logger.debug(
                        f"Parent trace/span not found, creating new task span on global trace"
                    )
                    span = _global_maxim_trace.span(config)
                else:
                    if trace:
                        logger.debug(f"Found parent trace: {trace.id}")
                        span = trace.span(config)
                    else:
                        logger.debug(f"Found parent span: {span.id}")
                        span = span.span(config)

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
                    logger.debug(f"Attaching span to agent [{self.agent.id}]")
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
                        logger.debug(
                            f"Found agent in args, attaching span to agent [{agent_to_use.role}]"
                        )
                        setattr(agent_to_use, "_span", span)

                        # Update the agent in its original location
                        if agent_from_kwargs:
                            kwargs["agent"] = agent_to_use
                        elif agent_from_args:
                            args_list[0] = agent_to_use
                            args = tuple(args_list)  # Convert back to tuple
                    else:
                        logger.warning(f"Task has no agent assigned")

                trace = None

            elif isinstance(self, Agent):
                span = getattr(self, "_span", None)
                if not isinstance(span, Span):
                    span = None

                span_id = str(uuid.uuid4())

                config = {
                    "id": span_id,
                    "name": f"Agent: {self.role}",
                    "tags": {"agent_id": str(getattr(self, "id", ""))},
                }

                skip_logging = False

                if span:
                    logger.debug(f"Agent span [{span_id}] for '{self.role}'")
                    span = span.span(config)
                else:
                    logger.debug(f"Agent has no span, checking task")

                    if len(args) > 0:
                        logger.debug(args[0])

                    # First check args/kwargs for task
                    task: Task = None
                    if len(args) > 0:
                        task = args[0]

                    if task:
                        logger.debug(
                            f"Found task in args: {task.id} and task description: {task.description}"
                        )

                    # Fallback to agent_executor.task if no task found in args
                    if not task and hasattr(self, "agent_executor"):
                        task = self.agent_executor.task

                    if task:
                        span_id = _task_span_ids.get(task.id)
                        if span_id:
                            span = logger.span_add_sub_span(span_id, config)
                        else:
                            logger.debug(
                                f"No span found for task {task.id}, creating new task span"
                            )
                            # Create a new task span since none exists
                            task_span_id = str(uuid.uuid4())
                            task_config = {
                                "id": task_span_id,
                                "name": f"Task: {task.name or task.description}",
                                "tags": {
                                    "task_id": str(task.id),
                                },
                            }
                            if _global_maxim_trace:  # TODO check for check in flows
                                span = _global_maxim_trace.span(task_config)
                                _task_span_ids[task.id] = task_span_id
                                # Now create the agent span as a child of the task span
                                span = span.span(config)
                            else:
                                logger.debug("No global trace found, skipping logging")
                                skip_logging = True
                    else:
                        logger.warning(
                            f"Agent {self.role} has no task or span, skipping logging"
                        )
                        skip_logging = True

                if not skip_logging:
                    if hasattr(self, "llm") and self.llm:
                        logger.debug(f"LLM: {getattr(self.llm, 'model', 'unknown')}")
                    if isinstance(self.llm, LLM):
                        setattr(self.llm, "_span", span)

                    if hasattr(self, "tools") and self.tools:
                        logger.debug(f"Attaching span to {len(self.tools)} tools")
                        for tool in self.tools:
                            setattr(tool, "_span", span)

            elif isinstance(self, LLM):
                span = getattr(self, "_span", None)
                if not isinstance(span, Span):
                    span = None

                generation_id = str(uuid.uuid4())
                setattr(self, "_maxim_generation_id", generation_id)
                logger.debug(f"LLM generation [{generation_id}]")

                config = {
                    "id": generation_id,
                    "name": "LLM Call",
                    "provider": (
                        "anthropic" if self.is_anthropic else "openai"
                    ),  # TODO: Add more providers
                    "model": str(getattr(self, "model", "unknown")),
                    "messages": args[0],
                }

                if span:
                    config["span_id"] = span.id
                    generation = span.generation(config)

                    span = None
                else:
                    logger.warning(
                        f"No parent span found for LLM call, creating new planner span"
                    )
                    planner_span = _global_maxim_trace.span(
                        {
                            "id": str(uuid.uuid4()),
                            "name": "Planner",
                        }
                    )
                    config["span_id"] = planner_span.id

                    generation = planner_span.generation(config)

            elif isinstance(self, BaseTool):
                span = getattr(self, "_span", None)
                if not isinstance(span, Span):
                    span = None

                if span:
                    tool_id = str(uuid.uuid4())
                    tool_name = extract_tool_name(final_op_name)

                    if tool_name == "RagTool":
                        logger.debug(f"RAG: Retrieval tool [{tool_id}]")
                        tool_call = span.retrieval(
                            {
                                "id": tool_id,
                                "name": f"RAG: {self.name}",
                                "tags": {"span_id": span.id},
                            }
                        )
                        tool_call.input(processed_inputs["query"])
                    else:
                        logger.debug(f"TOOL: {self.name} [{tool_id}]")

                        tool_details = extract_tool_details(self.description)
                        tool_call = span.tool_call(
                            {
                                "id": tool_id,
                                "name": f"{tool_details['name'] or self.name}",
                                "description": tool_details["description"]
                                or self.description,
                                "args": (
                                    str(tool_details['args'])
                                    if tool_details['args'] is not None
                                    else str(processed_inputs)
                                ),
                                "tags": {"tool_id": tool_id, "span_id": span.id},
                            }
                        )

                    span = None
                else:
                    logger.warning(f"No parent span found for tool call")

            logger.debug(f"\n--- Calling: {final_op_name} ---")

            try:
                # Call the original method (bound to self)
                output = original_method.__get__(self, self.__class__)(*args, **kwargs)
            except Exception as e:
                logger.error(f"{type(e).__name__} in {final_op_name}")

                if tool_call:
                    if isinstance(tool_call, Retrieval):
                        tool_call.output("Error occurred while calling tool")
                        logger.debug("RAG: Completed retrieval with error")
                    else:
                        tool_call.result("Error occurred while calling tool")
                        logger.debug("TOOL: Completed tool call with error")

                if generation:
                    generation.error({"message": str(e)})
                    logger.debug("GEN: Completed generation with error")

                if span:
                    span.add_error({"message": str(e)})
                    span.end()
                    logger.debug("SPAN: Completed span with error")

                if trace and "crewai.Crew" in final_op_name:
                    trace.add_error({"message": str(e)})
                    trace.end()
                    logger.debug("TRACE: Completed trace with error")
                    _global_maxim_trace = None

                raise e  # Re-raise the original exception

            processed_output = output
            if output_processor:
                try:
                    processed_output = output_processor(output)
                except Exception as e:
                    logger.error(f"Failed to process output: {e}")

            if tool_call:
                if isinstance(tool_call, Retrieval):
                    tool_call.output(processed_output)
                    logger.debug("RAG: Completed retrieval")
                else:
                    tool_call.result(processed_output)
                    logger.debug("TOOL: Completed tool call")

            if generation:
                # Create a structured result compatible with GenerationResult
                # Retrieve usage data captured by the callback
                global _last_llm_usages  # Ensure access to the global

                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                usage_data = _last_llm_usages[generation_id]

                if usage_data and isinstance(usage_data, dict):
                    prompt_tokens = usage_data.get("prompt_tokens", 0)
                    completion_tokens = usage_data.get("completion_tokens", 0)
                    total_tokens = usage_data.get("total_tokens", 0)
                    logger.debug(
                        f"GEN: Using captured token usage: P={prompt_tokens}, C={completion_tokens}, T={total_tokens}"
                    )
                else:
                    logger.debug(
                        f"GEN: Using default token usage (0). Captured data: {usage_data}"
                    )

                result = {
                    "id": f"gen_{generation_id}",
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
                logger.debug("GEN: Completed generation")

                if planner_span:
                    planner_span.end()
                    logger.debug("PLANNER: Completed planner")

                del _last_llm_usages[generation_id]

            if span:
                span.output(str(processed_output))
                logger.debug("SPAN: Completing span")
                span.end()

            if trace:
                logger.debug(f"TRACE: Completing trace [{trace.id}]")
                trace.set_output(str(processed_output))
                # if isinstance(output, CrewOutput):
                #     logger.debug(
                #         f"In crewai.Crew.kickoff, output is of type: {type(output)}"
                #     )
                #     token_usage = output.token_usage.model_dump()

                #     logger.info(f"Token usage: {token_usage}")

                #     trace.add_metadata(token_usage)

                trace.end()
                _global_maxim_trace = None

            logger.debug(f"――― End: {final_op_name} ―――\n")

            return output

        return maxim_wrapper

    # --- 0. Patch LLM._handle_non_streaming_response to capture usage ---
    if LLM is not None and hasattr(LLM, "_handle_non_streaming_response"):
        original_handle_method = getattr(LLM, "_handle_non_streaming_response")
        wrapper_handle = make_handle_non_streaming_wrapper(original_handle_method)
        setattr(LLM, "_handle_non_streaming_response", wrapper_handle)
        logger.info(
            "Patched crewai.LLM._handle_non_streaming_response to capture usage."
        )

    # --- 1. Patch Crew Methods ---
    crew_methods_to_patch = ["kickoff", "kickoff_async", "kickoff_for_each"]
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
            logger.info(f"Patched crewai.Crew.{method_name} for printing.")

    # --- 2. Patch Agent.execute_task ---
    agent_methods_to_patch = ["execute_task", "execute_task_async"]
    for method_name in agent_methods_to_patch:
        if hasattr(Agent, method_name):
            original_method = getattr(Agent, method_name)
            op_name = f"crewai.Crew.{method_name}"
            wrapper = make_maxim_wrapper(
                original_method,
                op_name,
                input_processor=crewai_postprocess_inputs,
                display_name_fn=get_agent_display_name,
            )
            setattr(Agent, method_name, wrapper)
            logger.info(f"Patched crewai.Agent.{method_name} for printing.")

    # --- 3. Patch Task.execute_sync ---
    task_methods_to_patch = ["execute_sync", "execute_sync_async"]
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
        logger.info(f"Patched crewai.Task.{method_name} for printing.")

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
        logger.info("Patched crewai.LLM.call for printing.")

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
                    logger.info(f"Patched {op_name} for printing.")
            except (AttributeError, TypeError, ImportError) as e:
                logger.warning(f"Skipping patching for tool {tool_name}: {e}")
                continue
    except ImportError:
        logger.warning("crewai_tools or BaseTool not found. Skipping tool patching.")
    except Exception as e:
        logger.error(f"ERROR during tool patching: {e}")

    # --- 6. Patch Flow Methods/Decorators (If Flow exists) ---
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
                logger.info(f"Patched crewai.Flow.{method_name} for printing.")

    _crewai_patched_print_only = True
    logger.info("Finished applying patches to CrewAI.")
