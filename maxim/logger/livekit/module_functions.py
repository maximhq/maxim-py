import functools
import inspect
from datetime import datetime, timezone

from livekit.agents import AgentSession
from livekit.agents.voice.generation import ToolContext, _ToolOutput

from ...scribe import scribe
from .store import get_session_store


def intercept_execute_tools_task_result(
    result, start_timestamp: datetime, *args, **kwargs
):
    """
    Interceptor for the module-level execute_tools_task function completion
    """
    print(f"[MAXIM] execute_tools_task completed with result: {result}")
    try:
        agent_session: AgentSession = kwargs.get("agent_session")
        tool_ctx: ToolContext = kwargs.get("tool_ctx")
        tool_output: _ToolOutput = kwargs.get("tool_output")
        print(f"[MAXIM] tool_output: {tool_output} {tool_ctx} {agent_session}")
        trace = get_session_store().get_current_trace_for_agent_session(
            id(agent_session)
        )
        if trace is None:
            return

        for output in tool_output.output:
            # All of these were called
            fn_call = output.fnc_call
            tool_call = tool_ctx.function_tools.get(fn_call.name)
            tool_call = trace.tool_call(
                {
                    "id": fn_call.id,
                    "name": fn_call.name,
                    "description": tool_call.__livekit_tool_info.description
                    if tool_call.__livekit_tool_info.description is not None
                    else "",
                    "args": fn_call.args,
                    "start_timestamp": start_timestamp,
                }
            )
            tool_call.result(output.output)

    except Exception as e:
        scribe().warning(f"[MAXIM] Failed to intercept execute_tools_task: {e}")


def instrument_execute_tools_task():
    """
    Instrument the execute_tools_task function in livekit.agents.voice.generation
    """
    try:
        import importlib

        module = importlib.import_module("livekit.agents.voice.generation")

        # Correct function name check
        if hasattr(module, "_execute_tools_task"):
            original_func = getattr(module, "_execute_tools_task")

            if inspect.iscoroutinefunction(original_func):

                @functools.wraps(original_func)
                async def async_wrapper(*args, **kwargs):
                    result = None
                    start_timestamp = datetime.now(timezone.utc)
                    try:
                        result = await original_func(*args, **kwargs)
                        return result
                    finally:
                        intercept_execute_tools_task_result(
                            result, start_timestamp, *args, **kwargs
                        )

                setattr(module, "_execute_tools_task", async_wrapper)

                @functools.wraps(original_func)
                def sync_wrapper(*args, **kwargs):
                    result = None
                    start_timestamp = datetime.now(timezone.utc)
                    try:
                        result = original_func(*args, **kwargs)
                        return result
                    finally:
                        intercept_execute_tools_task_result(
                            result, start_timestamp, *args, **kwargs
                        )

                setattr(module, "execute_tools_task", sync_wrapper)
            return True

        else:
            return False

    except ImportError:
        return False
    except Exception:
        return False


# Keep the old function name for backward compatibility but make it call the new one
def instrument_module_functions():
    """
    Backward compatibility wrapper - now instruments perform_tool_executions
    """
    instrument_execute_tools_task()
