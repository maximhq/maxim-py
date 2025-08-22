import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Optional
from uuid import uuid4
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.openai.llm import _LLMOptions

from livekit.agents.llm import LLM

from .store import (
    LLMUsage,
    SessionStoreEntry,
    Turn,
    get_livekit_callback,
    get_maxim_logger,
    get_session_store,
)
from ...scribe import scribe


class SameThreadExecutor(Executor):
    """
    A mock executor that runs submitted callables on the same thread, synchronously.
    Mimics the interface of concurrent.futures.Executor.
    """

    def __init__(self):
        # Don't call super().__init__() to avoid creating actual thread pool
        pass

    def submit(self, fn, *args, **kwargs):
        class _ImmediateResult:
            def __init__(self, value=None, exception=None):
                self._value = value
                self._exception = exception

            def result(self, timeout=None):
                if self._exception:
                    raise self._exception
                return self._value

            def done(self):
                return True

            def add_done_callback(self, fn):
                fn(self)

        try:
            value = fn(*args, **kwargs)
            return _ImmediateResult(value=value)
        except Exception as e:
            return _ImmediateResult(exception=e)


# Create a global thread pool for processing
_thread_pool_executor = ThreadPoolExecutor(max_workers=1)
_thread_pool_lock = threading.Lock()

def get_thread_pool_executor() -> ThreadPoolExecutor:
    """Get the global thread pool executor for processing."""
    if _thread_pool_executor is None:
        raise ValueError("Thread pool executor is not initialized")
    return _thread_pool_executor

def shutdown_thread_pool_executor(wait=True):
    """Shutdown the global thread pool executor."""
    global _thread_pool_executor
    with _thread_pool_lock:
        if _thread_pool_executor is not None:
            _thread_pool_executor.shutdown(wait=wait)
            _thread_pool_executor = None


def start_new_turn(session_info: SessionStoreEntry):
    """
    This function will start a new turn and return the current turn.
    If the current turn is interrupted or empty, it will return None.
    If the current turn is not interrupted and not empty, it will return the current turn.
    If the current turn is interrupted and not empty, it will return None.
    If the current turn is empty, it will return None.
    If the current turn is not interrupted and empty, it will return None.
    If the current turn is interrupted and empty, it will return None.

    Args:
        session_info: The session information.

    Returns:
        The new turn or None if the current turn is interrupted or empty.
    """
    turn = session_info.current_turn
    trace = get_session_store().get_current_trace_for_agent_session(
        session_info.agent_session_id,
    )
    # Here if the turn was interrupted, we need to push pending changes to the llm call as well
    if trace is not None and turn is not None:
        trace.end()
        callback = get_livekit_callback()
        if callback is not None:
            try:
                callback("maxim.trace.ended", {"trace_id": trace.id, "trace": trace})
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK] An error was captured during LiveKit callback execution: {e!s}",
                    exc_info=True,
                )
    next_turn_sequence = 1
    if turn is not None and turn.turn_sequence is not None:
        next_turn_sequence = turn.turn_sequence + 1
    # Creating a new turn and new trace
    session_id = session_info.mx_session_id
    trace_id = str(uuid4())
    tags = {}
    if session_info.room_id is not None:
        tags["room_id"] = session_info.room_id
    if session_info.agent_id is not None:
        tags["agent_id"] = session_info.agent_id
    if session_info.room_name is not None:
        tags["room_name"] = session_info.room_name
    if session_info.agent_session_id is not None:
        tags["agent_session_id"] = session_info.agent_session_id
    current_turn = Turn(
        turn_id=str(uuid4()),
        turn_sequence=next_turn_sequence,
        turn_timestamp=datetime.now(timezone.utc),
        turn_input_audio_buffer=BytesIO(),
        is_interrupted=False,
        turn_input_transcription="",
        turn_output_transcription="",
        turn_output_audio_buffer=BytesIO(),
    )
    trace = get_maxim_logger().trace(
        {
            "id": trace_id,
            "name": f"Turn {next_turn_sequence}",
            "session_id": session_id,
            "tags": tags,
        }
    )
    session_info.user_speaking = True
    session_info.current_turn = current_turn
    session_info.mx_current_trace_id = trace_id
    get_session_store().set_session(session_info)
    callback = get_livekit_callback()
    if callback is not None:
        try:
            callback("maxim.trace.started", {"trace_id": trace_id, "trace": trace})
        except Exception as e:
            scribe().warning(
                f"[MaximSDK] An error was captured during LiveKit callback execution: {e!s}",
                exc_info=True,
            )
    return current_turn

def extract_llm_usage(session_id: str, llm: Optional[LLM]) -> Optional[LLMUsage]:
    if llm is None:
        return None

    if hasattr(llm, "_events") and llm._events is not None:
        events = llm._events or getattr(llm, "_events", None)
        if isinstance(events, dict) and "metrics_collected" in events:
            session_info = get_session_store().get_session_by_agent_session_id(
                session_id,
            )
            if session_info is not None:
                turn = session_info.current_turn
                if turn is not None:
                    return turn.usage
    return None

def extract_llm_model_parameters(llm: Optional[_LLMOptions]) -> Optional[dict[str, Any]]:
    """
    Extract the model parameters from the LLM object.
    """
    if llm is None:
        return None
    model_parameters = {}
    if hasattr(llm, "temperature") and llm.temperature is not None and llm.temperature != NOT_GIVEN:
        model_parameters["temperature"] = llm.temperature
    if hasattr(llm, "max_completion_tokens") and llm.max_completion_tokens is not None and llm.max_completion_tokens != NOT_GIVEN:
        model_parameters["max_completion_tokens"] = llm.max_completion_tokens
    if hasattr(llm, "tool_choice") and llm.tool_choice is not None and llm.tool_choice != NOT_GIVEN:
        model_parameters["tool_choice"] = llm.tool_choice
    return model_parameters
